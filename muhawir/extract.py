"""Text from an uploaded file or photo, for the «تحقق من نص» box and the question box.

Everything happens in memory or in a temporary folder that is removed at once: the upload is never kept, and the text
read from it is returned to the visitor's page only. No language model is called, so a file costs nothing but a little
processor time on the server. Text is read from:

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

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


def clean(text: str) -> str:
    """Printable text with tidy spaces and lines; presentation-form letters turned back into ordinary letters."""
    text = unicodedata.normalize("NFKC", text) if PRESENTATION_FORMS.search(text) else text
    text = _CONTROL.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


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
        image = Image.open(io.BytesIO(data))
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


def from_image(data: bytes, deadline: float) -> str:
    return ocr_image(_prepare(data), deadline)


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
        pages = [(reader.pages[i].extract_text() or "") for i in range(min(len(reader.pages), MAX_PDF_PAGES))]
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
