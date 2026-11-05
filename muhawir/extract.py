"""Text from an uploaded file or photo, for the «تحقق من نص» box and the question box.

Everything happens in memory or in a temporary folder that is removed at once: the upload is never kept, and the text
read from it is returned to the visitor's page only. No language model is called, so a file costs nothing but a little
processor time on the server, with one exception: a photo that Tesseract on the server cannot read (handwriting) is
then sent to the vision model to be transcribed, when one is configured (the owner turns it off with
MUHAWIR_VISION_OCR=0). The upload button says so. Text is read from:

  photos and scans   Tesseract (Arabic and English) on the server, after a little cleaning of the picture
  PDF                the text layer (pypdf); a page with no usable text layer, or whose layer holds visually ordered
                     Arabic (presentation forms, which read backwards), is read from a picture of the page instead
  DOCX, ODT          the paragraphs, taken out of the zip with no XML parser (nothing to expand, nothing to attack)
  HTML, TXT, MD, CSV the text itself, in UTF-8 or Windows-1256

The kind of file is found from its first bytes, never from its name or declared type.
"""
from __future__ import annotations

import html
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import zipfile
from html.parser import HTMLParser

MAX_UPLOAD = 8 * 1024 * 1024       # bytes
MAX_TEXT = 20000                   # characters returned
MAX_PDF_PAGES = 40                 # pages whose text layer is read
OCR_PDF_PAGES = 3                  # pages read from a picture
MAX_PIXELS = 40_000_000            # a picture larger than this is refused (decompression bombs)
MAX_UNZIPPED = 30 * 1024 * 1024    # what a DOCX or ODT may expand to
OCR_TIMEOUT = 25                   # seconds for one picture
OCR_BUDGET = 45                    # seconds for one file
MIN_LAYER_CHARS = 40               # a PDF page with fewer letters than this has no usable text layer
PRESENTATION_FORMS = re.compile("[ﭐ-﷿ﹰ-﻿]")
ARABIC_LETTER = re.compile("[؀-ۿﭐ-﷿ﹰ-﻿]")


class ExtractError(Exception):
    """A file that cannot be read; `code` is what the page translates, never a path or a library message."""

    def __init__(self, code: str, note: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.note = note  # a short technical hint for the owner (an HTTP status), never a path or a key


_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


def clean(text: str) -> str:
    """Printable text with tidy spaces and lines; presentation-form letters turned back into ordinary letters."""
    text = unicodedata.normalize("NFKC", text) if PRESENTATION_FORMS.search(text) else text
    text = _CONTROL.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


IMAGE_FORMATS = ("PNG", "JPEG", "GIF", "TIFF", "BMP", "WEBP")  # the picture kinds sniff() accepts


def sniff(data: bytes) -> str:
    """The kind of a file from its first bytes: pdf, zip, image, or text."""
    head = data[:16]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if (head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8\xff") or head[:4] in (b"GIF8", b"II*\x00", b"MM\x00*")
            or head.startswith(b"BM") or (head.startswith(b"RIFF") and data[8:12] == b"WEBP")):
        return "image"
    return "text"


# ---- text kinds -------------------------------------------------------------------------------------------------

def decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1256"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class _Visible(HTMLParser):
    BLOCKS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.skip += 1
        elif tag in self.BLOCKS:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCKS:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def from_html(text: str) -> str:
    parser = _Visible()
    parser.feed(text)
    return "".join(parser.out)


def _zip_member(data: bytes, name: str) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos = z.infolist()
            if len(infos) > 2000 or sum(i.file_size for i in infos) > MAX_UNZIPPED:
                raise ExtractError("unreadable")
            if name not in z.namelist():
                raise ExtractError("unsupported")
            return z.read(name).decode("utf-8", errors="replace")
    except zipfile.BadZipFile:
        raise ExtractError("unreadable") from None


def _zip_names(data: bytes) -> list[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return z.namelist()
    except zipfile.BadZipFile:
        raise ExtractError("unreadable") from None


def from_docx(data: bytes) -> str:
    xml = _zip_member(data, "word/document.xml")
    paragraphs = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        p = re.sub(r"<w:(?:tab|br)\b[^>]*/>", "\x00", p)
        runs = re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>|(\x00)", p, flags=re.S)
        paragraphs.append("".join(html.unescape(t) if t else " " for t, sep in runs))
    return "\n".join(paragraphs)


def from_odt(data: bytes) -> str:
    xml = _zip_member(data, "content.xml")
    xml = re.sub(r"<text:(?:tab|line-break)\b[^>]*/>", " ", xml)
    paragraphs = re.findall(r"<text:(?:p|h)\b[^>]*>(.*?)</text:(?:p|h)>", xml, flags=re.S)
    return "\n".join(html.unescape(re.sub(r"<[^>]+>", "", p)) for p in paragraphs)


def from_zip(data: bytes) -> str:
    names = _zip_names(data)
    if "word/document.xml" in names:
        return from_docx(data)
    if "content.xml" in names and "mimetype" in names:
        return from_odt(data)
    raise ExtractError("unsupported")


# ---- pictures -----------------------------------------------------------------------------------------------------

_languages: str | None = None


def ocr_languages() -> set[str]:
    """The Tesseract languages installed; ExtractError when Tesseract or the Arabic data is not there."""
    global _languages
    if _languages is None:
        if shutil.which("tesseract") is None:
            raise ExtractError("ocr_unavailable")
        try:
            listed = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, timeout=10).stdout.split()
        except (OSError, subprocess.SubprocessError):
            raise ExtractError("ocr_unavailable") from None
        if "ara" not in listed:
            raise ExtractError("ocr_unavailable")
        _languages = set(listed)
    return _languages


def _prepare(data: bytes):
    """The picture upright, grey, with enough pixels to read and its contrast stretched; a PIL image."""
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        # only the kinds sniff() lets through as an image: no other decoder is reached by an upload (audit HIGH-002)
        image = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
        if image.width * image.height > MAX_PIXELS:
            raise ExtractError("too_large")
        image = ImageOps.exif_transpose(image)
        image.load()
    except ExtractError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError):
        raise ExtractError("unreadable") from None
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, "white")
        background.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[-1])
        image = background
    image = ImageOps.autocontrast(image.convert("L"))
    shortest = min(image.size)
    if shortest < 900:  # small text reads badly: up to twice as large
        factor = min(2.0, 900 / shortest)
        image = image.resize((int(image.width * factor), int(image.height * factor)), Image.LANCZOS)
    return image


def _arabic_share(text: str) -> float:
    letters = re.findall(r"[^\W\d_]", text)
    return len(ARABIC_LETTER.findall(text)) / len(letters) if letters else 0.0


def ocr_image(image, deadline: float) -> str:
    """Tesseract on one prepared picture. Arabic alone is read first: with English added, the marks of a vowelled text
    (the fatha, the shadda) are read as Latin letters and the Arabic gets worse (measured: 0.97 against 0.84 similarity
    on a vowelled verse). A picture that comes out mostly non-Arabic is read again as English."""
    languages = ocr_languages()
    first = _run_tesseract(image, "ara", deadline)
    if _arabic_share(first) < 0.3 and "eng" in languages:
        second = _run_tesseract(image, "eng", deadline)
        if len(re.findall(r"[A-Za-z]", second)) > len(re.findall(r"[A-Za-z]", first)):
            return second
    return first


def _run_tesseract(image, language: str, deadline: float) -> str:
    """Tesseract on one prepared picture, in a folder that is removed when it finishes."""
    remaining = deadline - time.monotonic()
    if remaining < 3:
        raise ExtractError("too_slow")
    with tempfile.TemporaryDirectory(prefix="muhawir-ocr-") as folder:
        path = os.path.join(folder, "page.png")
        image.save(path)
        try:
            done = subprocess.run(["tesseract", path, "stdout", "-l", language, "--psm", "3"], capture_output=True, text=True,
                                  timeout=min(OCR_TIMEOUT, remaining), env={**os.environ, "OMP_THREAD_LIMIT": "1"})
        except subprocess.TimeoutExpired:
            raise ExtractError("too_slow") from None
        except OSError:
            raise ExtractError("ocr_unavailable") from None
    return done.stdout if done.returncode == 0 else ""


# Photo reading by a vision model (handwriting). The text is received as it is written (streaming), so the wait
# limit applies only while the model is silent, not to the whole page: a full handwritten page took longer than one
# 25-second wait and failed with ReadTimeout (live, November 2026). A model that has not answered within a few seconds
# does not hold up the next one: the next is started beside it, and the first full transcription is used.
VISION_STALL = 40.0    # seconds a model may stay silent (before its first words or between two pieces of text)
VISION_HEDGE = 8.0     # seconds without an answer before the next model is started beside it
VISION_TIMEOUT = 80.0  # seconds for the whole reading, on its own clock (the Tesseract attempt may have used OCR_BUDGET)
VISION_FALLBACK = "gemini-flash-latest"  # only when Google's model list cannot be read (GEMINI_VISION_FALLBACK)
VISION_LISTED = 2      # newest listed flash models tried after the configured one
STRONG_OCR_LETTERS = 60  # Tesseract with at least this many Arabic letters has read a printed page: no need to wait
MIN_OCR_LETTERS = 20   # fewer Arabic letters than this from Tesseract: the picture is probably handwriting
VISION_PROMPT = ("Transcribe all the text in this picture exactly as written, in its own language and line order, "
                 "including handwriting. Reply with the text only, no comments. If there is no text, reply with nothing.")


def vision_enabled() -> bool:
    return (os.environ.get("MUHAWIR_VISION_OCR") != "0" and bool(os.environ.get("GEMINI_API_KEY"))
            and bool(os.environ.get("GEMINI_MODEL")))


def _client():
    """One HTTP client per reading attempt (replaced in tests)."""
    import httpx
    return httpx.Client(timeout=httpx.Timeout(connect=10.0, read=VISION_STALL, write=20.0, pool=10.0))


class _Stopped(Exception):
    """Another model has already read the picture, or the time for the reading is over."""


def _sse(response, stop, deadline):
    """The JSON objects of a server-sent event stream, one by one, until it ends, `stop` is set or time is up."""
    import json
    for line in response.iter_lines():
        if stop.is_set() or time.monotonic() > deadline:
            raise _Stopped()
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue


_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"
_THINKING: dict[str, dict] = {}       # model -> the generationConfig it accepted (remembered: no refused try each time)
_MISSING: dict[str, float] = {}       # model -> when it answered «not found» (a retired name): skipped for a while
_LISTED: dict[str, object] = {"at": 0.0, "models": []}
MISSING_SECONDS = 6 * 3600


def _thinking_settings(model: str) -> list[dict]:
    """The generationConfig to try, least thinking first. A transcription needs none, and a model that thinks first
    sends nothing for a long time (live ReadTimeout, 5 November 2026). Gemini 3 and later take thinkingLevel (lowest:
    «minimal»), 2.5 takes thinkingBudget 0; sending the one a model does not know is refused (400), so the next is
    tried, and the one that worked is remembered for that model."""
    base = {"temperature": 0}
    known = _THINKING.get(model)
    if known is not None:
        return [known]
    match = re.search(r"gemini-(\d+)(?:\.(\d+))?", model)
    newer = not match or int(match.group(1)) >= 3  # «gemini-flash-latest» and other aliases: assume the newest kind
    level = {**base, "thinkingConfig": {"thinkingLevel": "minimal"}}
    budget = {**base, "thinkingConfig": {"thinkingBudget": 0}}
    return [level, budget, base] if newer else [budget, level, base]


def _read_gemini(model: str, data_b64: str, stop, deadline) -> str:
    """The transcription from one Gemini model, streamed, with the least thinking it accepts."""
    body = {"contents": [{"role": "user", "parts": [{"text": VISION_PROMPT},
                                                    {"inlineData": {"mimeType": "image/jpeg", "data": data_b64}}]}]}
    url = f"{_GEMINI}/{model}:streamGenerateContent?alt=sse"
    headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"], "content-type": "application/json"}
    settings = _thinking_settings(model)
    for i, config in enumerate(settings):
        with _client() as client, client.stream("POST", url, headers=headers,
                                                json={**body, "generationConfig": config}) as response:
            if response.status_code == 400 and i + 1 < len(settings):
                continue
            if response.status_code == 404:  # a retired or unknown name: not asked again for a while
                _MISSING[model] = time.monotonic()
            response.raise_for_status()
            _THINKING[model] = config
            text = []
            for event in _sse(response, stop, deadline):
                for candidate in event.get("candidates") or []:
                    text += [p.get("text", "") for p in (candidate.get("content") or {}).get("parts") or []]
            return "".join(text).strip()
    return ""


def _listed_flash_models() -> list[str]:
    """The Gemini «flash» models the key can use now, newest first, from Google's own list (read once every few hours):
    fallbacks are never names written in the code, which retire (gemini-2.5-flash answered 404, November 2026)."""
    now = time.monotonic()
    if _LISTED["models"] and now - float(_LISTED["at"]) < MISSING_SECONDS:
        return list(_LISTED["models"])
    found: list[tuple[tuple, str]] = []
    try:
        with _client() as client:
            r = client.get(_GEMINI, params={"pageSize": 200}, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
            r.raise_for_status()
            for m in r.json().get("models") or []:
                name = str(m.get("name", "")).removeprefix("models/")
                methods = m.get("supportedGenerationMethods") or []
                if "generateContent" not in methods or "flash" not in name:
                    continue
                if re.search(r"tts|image|audio|live|embedding|thinking|exp", name):
                    continue
                version = tuple(int(x) for x in re.findall(r"\d+", name.split("-flash")[0])) or (0,)
                found.append(((("lite" not in name), ("preview" not in name), version), name))
    except Exception as exc:  # noqa: BLE001 - no list: the configured model and the fixed fallbacks still serve
        logging.getLogger("muhawir.extract").warning("vision model list unavailable (%s)", type(exc).__name__)
        return []
    models = [name for _, name in sorted(found, key=lambda x: x[0], reverse=True)][:VISION_LISTED]
    _LISTED.update(at=now, models=models)
    return models


def _read_compat(data_b64: str, stop, deadline) -> str:
    """The transcription through the OpenAI-compatible endpoint (OPENAI_COMPAT_BASE_URL), streamed;
    OPENAI_COMPAT_VISION_MODEL names the model (on OpenRouter, google/gemini-2.5-flash by default)."""
    from urllib.parse import urlsplit
    base, key = os.environ.get("OPENAI_COMPAT_BASE_URL", "").rstrip("/"), os.environ.get("OPENAI_COMPAT_API_KEY", "")
    model = os.environ.get("OPENAI_COMPAT_VISION_MODEL") or ("google/gemini-2.5-flash" if urlsplit(base).hostname == "openrouter.ai" else "")
    with _client() as client, client.stream("POST", f"{base}/chat/completions",
            headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
            json={"model": model, "temperature": 0, "stream": True, "messages": [{"role": "user", "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data_b64}}]}]}) as response:
        response.raise_for_status()
        text = []
        for event in _sse(response, stop, deadline):
            for choice in event.get("choices") or []:
                text.append((choice.get("delta") or choice.get("message") or {}).get("content") or "")
        return "".join(text).strip()


def _vision_readers() -> list[tuple[str, object]]:
    """(name, read) for each model that can read a picture, in order: the configured model, then the newest flash
    models Google lists for the key (or GEMINI_VISION_FALLBACK when the owner names them), then the OpenAI-compatible
    endpoint when one is configured. A name that answered «not found» lately is left out."""
    chosen = os.environ.get("GEMINI_VISION_MODEL") or os.environ["GEMINI_MODEL"]
    named = [m.strip() for m in os.environ.get("GEMINI_VISION_FALLBACK", "").split(",") if m.strip()]
    now = time.monotonic()
    models = [m for m in dict.fromkeys([chosen, *(named or _listed_flash_models() or VISION_FALLBACK.split(","))])
              if now - _MISSING.get(m, -MISSING_SECONDS) >= MISSING_SECONDS][:VISION_LISTED + 1]
    readers = [(m, (lambda m: lambda b64, stop, deadline: _read_gemini(m, b64, stop, deadline))(m)) for m in models]
    base = os.environ.get("OPENAI_COMPAT_BASE_URL", "")
    if base and os.environ.get("OPENAI_COMPAT_API_KEY") and (os.environ.get("OPENAI_COMPAT_VISION_MODEL") or "openrouter.ai" in base):
        readers.append(("compat", _read_compat))
    return readers


def _problem(name: str, error: Exception) -> str:
    """What went wrong with one model, for the owner (a status code or an error type, never a key or a text)."""
    import httpx
    if isinstance(error, httpx.HTTPStatusError):
        return f"{name} HTTP {error.response.status_code}"
    return f"{name} {type(error).__name__}"


def vision_text(image) -> tuple[str, str]:
    """(text, error): the picture's text (handwriting included) read by the vision models, and what went wrong with
    each ("" when one succeeded). Models run one after another, but a model silent for VISION_HEDGE seconds no longer
    holds up the next: it is started beside it and the first full transcription wins."""
    import base64
    import threading
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    shown = image.copy()
    shown.thumbnail((1280, 1280))
    buffer = io.BytesIO()
    shown.convert("RGB").save(buffer, "JPEG", quality=82)
    data_b64 = base64.b64encode(buffer.getvalue()).decode()
    log = logging.getLogger("muhawir.extract")
    readers = _vision_readers()
    stop, deadline = threading.Event(), time.monotonic() + VISION_TIMEOUT
    pool = ThreadPoolExecutor(max_workers=len(readers))
    running: dict = {}
    failed: list[str] = []
    waiting = list(readers)

    def start() -> None:
        name, read = waiting.pop(0)
        running[pool.submit(read, data_b64, stop, deadline)] = name

    try:
        start()
        while running:
            done, _ = wait(running, timeout=max(0.0, min(VISION_HEDGE, deadline - time.monotonic())),
                           return_when=FIRST_COMPLETED)
            lost = False
            for future in done:
                name = running.pop(future)
                try:
                    text = future.result()
                except _Stopped:
                    continue
                except Exception as error:  # noqa: BLE001 - every failure is reported by name, then the next model
                    failed.append(_problem(name, error))
                    log.warning("vision transcription failed: %s", failed[-1])
                    lost = True
                    continue
                if text:
                    return text, ""
                failed.append(f"{name} empty")
                lost = True
            if time.monotonic() >= deadline:
                failed += [f"{name} timeout" for name in running.values()]
                break
            if waiting and (not done or lost):  # silent too long, or one failed: the next model starts now
                start()
        return "", ", ".join(failed)
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)


def from_image(data: bytes, deadline: float) -> str:
    """Tesseract on the server first. Only a picture it cannot read well (handwriting) is sent to the vision model,
    and only when the owner has a vision model configured (MUHAWIR_VISION_OCR=0 turns it off): a printed page never
    leaves the server (audit HIGH-004; it used to be sent at the same time, in case it was needed)."""
    image = _prepare(data)
    try:
        text = ocr_image(image, deadline)
    except ExtractError:
        if not vision_enabled():
            raise
        text = ""
    if _read_well(text) or not vision_enabled():
        return text
    seen, problem = vision_text(image)
    count = lambda t: len(ARABIC_LETTER.findall(t)) + len(re.findall(r"[A-Za-z]", t))  # noqa: E731
    if count(seen) > count(text):
        return seen
    if problem and count(text) < MIN_OCR_LETTERS:
        raise ExtractError("vision_failed", problem)
    return text


def _read_well(text: str) -> bool:
    """Tesseract has read a printed page: enough Arabic letters and mostly Arabic, or enough English letters."""
    arabic = len(ARABIC_LETTER.findall(text))
    return (arabic >= STRONG_OCR_LETTERS and _arabic_share(text) >= 0.6) or len(re.findall(r"[A-Za-z]", text)) >= 80


# ---- PDF ---------------------------------------------------------------------------------------------------------

def _page_pictures(data: bytes, first: int, last: int) -> list[bytes]:
    """PNG pictures of pages first..last, by poppler's pdftoppm when it is installed."""
    if shutil.which("pdftoppm") is None:
        return []
    with tempfile.TemporaryDirectory(prefix="muhawir-pdf-") as folder:
        source = os.path.join(folder, "in.pdf")
        with open(source, "wb") as f:
            f.write(data)
        try:
            subprocess.run(["pdftoppm", "-png", "-r", "170", "-f", str(first), "-l", str(last), source, os.path.join(folder, "p")],
                           capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return []
        return [open(os.path.join(folder, n), "rb").read() for n in sorted(os.listdir(folder)) if n.startswith("p") and n.endswith(".png")]


def from_pdf(data: bytes, deadline: float) -> tuple[str, str]:
    """(text, method): the text layer when it is usable, else the first pages read from pictures."""
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractError("encrypted")
        pages = []
        for i in range(min(len(reader.pages), MAX_PDF_PAGES)):  # the time limit is checked between pages (HIGH-005)
            if time.monotonic() > deadline:
                raise ExtractError("too_slow")
            pages.append(reader.pages[i].extract_text() or "")
    except ExtractError:
        raise
    except (PyPdfError, ValueError, KeyError, RecursionError, OSError):
        raise ExtractError("unreadable") from None
    text = "\n\n".join(pages)
    letters = ARABIC_LETTER.findall(text)
    visual = bool(letters) and len(PRESENTATION_FORMS.findall(text)) / len(letters) > 0.3  # visually ordered Arabic reads backwards
    if len(letters) + len(re.findall("[A-Za-z]", text)) >= MIN_LAYER_CHARS * max(1, min(len(pages), 3)) and not visual:
        return text, "pdf-text"
    texts = []
    for picture in _page_pictures(data, 1, OCR_PDF_PAGES):
        texts.append(ocr_image(_prepare(picture), deadline))
    if not texts:
        if text.strip():
            return text, "pdf-text"
        raise ExtractError("ocr_unavailable")
    return "\n\n".join(texts), "pdf-ocr"


# ---- entry points ------------------------------------------------------------------------------------------------

def extract(data: bytes) -> dict:
    """{"kind", "method", "text", "truncated"} for the bytes of an upload; ExtractError(code) when unreadable."""
    if not data:
        raise ExtractError("unreadable")
    if len(data) > MAX_UPLOAD:
        raise ExtractError("too_large")
    deadline = time.monotonic() + OCR_BUDGET
    kind = sniff(data)
    if kind == "pdf":
        text, method = from_pdf(data, deadline)
    elif kind == "zip":
        text, method = from_zip(data), "office"
    elif kind == "image":
        text, method = from_image(data, deadline), "ocr"
    else:
        raw = decode(data)
        if "\x00" in raw[:2000]:
            raise ExtractError("unsupported")
        text, method = (from_html(raw) if re.search(r"<(?:html|body|p|div)\b", raw[:4000], re.I) else raw), "text"
    text = clean(text)
    if not ARABIC_LETTER.search(text) and not re.search(r"[A-Za-z]{3}", text):
        raise ExtractError("no_text")
    return {"kind": kind, "method": method, "text": text[:MAX_TEXT], "truncated": len(text) > MAX_TEXT}


ISOLATED_SECONDS = OCR_BUDGET + 85  # the whole reading in its own process: OCR, then the vision model if needed


def extract_isolated(data: bytes) -> dict:
    """extract(data) in a separate process with time, memory and processor limits (extract_worker.py): a file made to
    exhaust its parser ends that process, not the server (audit HIGH-005). MUHAWIR_EXTRACT_ISOLATED=0 reads in this
    process (tests, or a system where a child process cannot be started)."""
    if len(data) > MAX_UPLOAD:
        raise ExtractError("too_large")
    if os.environ.get("MUHAWIR_EXTRACT_ISOLATED") == "0":
        return extract(data)
    import json
    import sys
    try:
        done = subprocess.run([sys.executable, "-m", "muhawir.extract_worker"], input=data, capture_output=True,
                              timeout=ISOLATED_SECONDS, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except subprocess.TimeoutExpired:
        raise ExtractError("too_slow") from None  # subprocess.run kills the child on timeout
    except OSError:
        raise ExtractError("unreadable", "worker") from None
    try:
        reply = json.loads(done.stdout.decode("utf-8") or "{}")
    except ValueError:
        reply = {}
    if "result" in reply:
        return reply["result"]
    if reply.get("code"):
        raise ExtractError(reply["code"], reply.get("note", ""))
    # killed by its memory or processor limit, or crashed in a parser: the file is not read, the server goes on
    logging.getLogger("muhawir.extract").warning("file reader ended without a reply (exit %s)", done.returncode)
    raise ExtractError("unreadable", f"worker exit {done.returncode}")


_SENTENCE_END = re.compile(r"(?<=[.!?\u061f\u061b\u06d4])\s+")


def pieces(text: str, size: int = 300, limit: int = 40) -> list[str]:
    """The text as passages to pick from, to check one at a time: each line is its own passage (a verse on its own
    line must not be joined to the next), and a long line is cut at sentence ends, then at a word, to about `size`
    characters. Passages of fewer than three words are left out."""
    out: list[str] = []
    for line in text.split("\n"):
        current = ""
        for sentence in (s.strip() for s in _SENTENCE_END.split(line)):
            if not sentence:
                continue
            while len(sentence) > size:
                cut = sentence.rfind(" ", 0, size)
                cut = cut if cut > size // 2 else size
                head, sentence = sentence[:cut].strip(), sentence[cut:].strip()
                if current:
                    out.append(current)
                    current = ""
                out.append(head)
            if current and len(current) + 1 + len(sentence) > size:
                out.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            out.append(current)
    return [p for p in out if len(p.split()) >= 3][:limit]
