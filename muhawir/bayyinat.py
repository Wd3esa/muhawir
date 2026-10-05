"""Import «بينات: أسئلة منتقاة حول الإسلام» (Osoul Center, Riyadh, 1st ed., 1445 AH; 1228 pages, 257
questions), the source the challenge's package names for objections and frequent questions
(dawa.center/file/7937). Each question in the book has the same parts: the question, similar phrasings,
the gist of the question, a short answer, a detailed answer, a closing note and keywords («كلمات دلالية»).

The book is a PDF made for print, so its text is rebuilt from the glyphs on each page (PyMuPDF's text
trace: every glyph with its position and glyph id), with these rules, all of them checked against the
printed pages:

  - a line is the glyphs on one baseline, read right to left by position; numbers and Latin are read
    left to right;
  - a ligature (لا، الله، في…) is one glyph followed by the letters it stands for: they are kept together;
  - a few lam-alef glyphs of the bold and poem fonts carry only the lam: their alef is put back (LAM_ALEF);
  - diacritics (tashkeel) and tatweel are left out: their positions in the file are not reliable enough to
    put each mark back on its letter, and a wrong mark would change the text;
  - Quran verses are set in a special Quran font whose glyphs carry no letters: a verse is shown as
    «(الآية)» before its reference (e.g. «(الآية) [يس: 40]»), never retyped; Muhawir quotes verses from the
    Quran text itself;
  - the honorific signs (e.g. «رضي الله عنه»، «عليه السلام») are set as single glyphs of a symbols font;
    they are written out in words (HONORIFIC); «ﷺ» stays as it is;
  - page headers, icons and decorative chapter titles are left out.

No word is added or changed otherwise. Passages are the parts of each question, joined paragraphs up to
MAX_CHUNK characters within one part, each with the question number, its title, the part and the printed
page, so it can be checked in the book.

The book is a contemporary work: its rights remain with the Osoul Center and its publisher; it is listed in
the challenge's source package, shown attributed with its page, and not stored in the repository.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

SOURCE_ID = "bayyinat"
PREFIX = "b"
FILE = "bayyinat.pdf"
SHA256 = "619b7201833419b8fbf86c463208462b9a2a7f02ad2306a2667490f3b410ad4e"
DOWNLOAD: tuple[str, ...] = ()  # the package's file page is dawa.center/file/7937; set when a direct link is confirmed
MAX_CHUNK = 1300
MIN_QUESTIONS = 250  # the book has 257: a damaged file is refused
VERSE = "(الآية)"

SOURCE = {
    "id": SOURCE_ID,
    "name": "بينات: أسئلة منتقاة حول الإسلام",
    "about": "«بينات: أسئلة وأجوبة عن الإسلام» (أسئلة منتقاة حول الإسلام)، مركز أصول، جمعية الدعوة والإرشاد وتوعية "
             "الجاليات بالربوة، الرياض، الطبعة الأولى 1445هـ، 1228 صفحة (ردمك 978-603-92157-0-7). وهو المصدر الذي "
             "تسميه الحزمة العلمية للتحدي للشبهات والأسئلة المتكررة (dawa.center/file/7937). عمل معاصر، حقوقه لأصحابها، "
             "يُعرض منسوبًا إليهم مع رقم الصفحة. نُقل النص من ملف PDF بلا تشكيل، وتظهر الآيات فيه «(الآية)» مع موضعها.",
    "url": "https://dawa.center/file/7937",
}

# lam-alef glyphs that carry only the lam in this file (font, glyph id) -> the alef they leave out
LAM_ALEF = {("adwaassalaf-Bold", 239): "ا", ("adwaassalaf-Bold-SC700", 239): "ا",
            ("adwaassalaf-Bold-SC700", 241): "أ", ("adwaassalaf-Bold", 139): "آ", ("(AH)-Manal-High", 527): "ا"}

# runs read left to right: a number (with its inner separators) or a Latin phrase
_LTR = re.compile(r"\d+(?:[.,:]\d+)*|[A-Za-z](?:[A-Za-z0-9'.\-]| (?=[A-Za-z0-9]))*")
# the honorific glyphs of the KFGQPC symbols font, written out in words (each checked on the printed page)
HONORIFIC = {"h": "رضي الله عنه", "i": "رضي الله عنها", "j": "رضي الله عنهم", "k": "رضي الله عنهما",
             "l": "رضي الله عنهن", "n": "عليه السلام", "o": "عليها السلام", "p": "عليهم السلام",
             "q": "عليهما السلام", "s": "رحمهم الله", "\uf072": "رحمه الله"}
_VERSE_RUN = re.compile(r"﴿[\uE000\s]*﴾|\uE000(?:\s*\uE000)*")
_SPACES = re.compile(r"[ \t ]+")
_NUMBER = re.compile(r"ل(\d+)س")  # «المسألة (12)» as the title font gives it
_KEYWORDS = re.compile(r"^كلمات دلالية\s*:\s*")


class ImportError_(ValueError):
    pass


def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch) == "Mn"


class _Glyph:
    __slots__ = ("x0", "x1", "y", "text", "font", "size", "quran")


def _page_lines(page) -> list[tuple[float, str, float, str]]:
    """(baseline, main font, size, text) for each line of a page, in reading order."""
    glyphs: list[_Glyph] = []
    for span in page.get_texttrace():
        font, size = span["font"], span["size"]
        last = None
        for code, gid, origin, bbox in span["chars"]:
            ch = chr(code) if code else ""
            if not ch or _is_mark(ch) or ch in "ـ�":
                continue
            if gid == -1 and last is not None:  # the rest of a ligature
                last.text += ch
                continue
            g = _Glyph()
            g.x0, g.x1, g.y, g.text = bbox[0], bbox[2], origin[1], ch
            g.font, g.size, g.quran = font, size, font.startswith("QCF")
            if "Symbols" in font and ch in HONORIFIC:
                g.text = f" {HONORIFIC[ch]} "
            if ch == "ل" and (font, gid) in LAM_ALEF:
                g.text += LAM_ALEF[(font, gid)]
            glyphs.append(g)
            last = g
    rows: dict[int, list[_Glyph]] = defaultdict(list)
    for g in glyphs:
        rows[round(g.y / 3)].append(g)
    lines: list[list[_Glyph]] = []
    previous = None
    for key in sorted(rows):
        if previous is not None and key - previous <= 1:
            lines[-1].extend(rows[key])
        else:
            lines.append(list(rows[key]))
        previous = key
    out = []
    for line in lines:
        line.sort(key=lambda g: -(g.x0 + g.x1))
        parts, prev = [], None
        for g in line:
            if prev is not None and prev.x0 - g.x1 > 4 and g.text != " " and prev.text != " ":
                parts.append(" ")  # two halves of a verse of poetry
            parts.append("\uE000" if g.quran else g.text)
            prev = g
        text = _LTR.sub(lambda m: m.group(0)[::-1], "".join(parts))
        fonts: dict[tuple[str, int], int] = defaultdict(int)
        for g in line:
            if not g.quran and g.text.strip():
                fonts[(g.font, round(g.size))] += 1
        font, size = max(fonts, key=fonts.get) if fonts else ("QCF", 13)
        out.append((min(g.y for g in line), font, size, text))
    return out


def read_pdf(path: str | Path) -> list[dict]:
    """Every line of the book: {"page" (printed), "y", "font", "size", "text"}."""
    import pymupdf  # only needed to build the data

    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ImportError_("this is not the expected PDF of «بينات» (checksum differs)")
    doc = pymupdf.open(stream=data, filetype="pdf")
    lines = []
    for index in range(doc.page_count):
        for y, font, size, text in _page_lines(doc[index]):
            text = _SPACES.sub(" ", text).strip()
            if not text or font == "icomoon" or (font == "(AH)-Manal-High" and size == 12 and y < 50):
                continue  # empty lines, icons, page headers
            lines.append({"page": index, "y": y, "font": font, "size": size, "text": text})
    return lines


def _label(line: dict) -> str | None:
    """The part of a question a heading line starts, or None for a body line."""
    font, size, text = line["font"], line["size"], line["text"]
    if font.startswith("DINNext"):
        return "question" if "سؤال" in text else ("answer" if "جواب" in text else None)
    if font == "(AH)-Manal-High" and size == 14:
        for key, label in (("مشا", "similar"), ("مضمو", "gist"), ("مختصر", "short"), ("تفصيل", "detail"),
                           ("خاتمة", "closing"), ("الخيصة", "closing"), ("مراجع", "end")):
            if key in text:
                return label
        return "skip"
    return None


PART_AR = {"question": "السؤال", "similar": "عبارات مشابهة للسؤال", "gist": "مضمون السؤال",
           "short": "مختصر الإجابة", "detail": "الجواب التفصيلي", "closing": "خاتمة الجواب"}


def parse(lines: list[dict]) -> list[dict]:
    """Questions in order: {"number", "title", "keywords", "parts": [(part, [(page, paragraph), ...])]}."""
    questions: list[dict] = []
    current = None
    part = None
    in_title = False
    in_keywords = False
    last_y = last_page = None
    for line in lines:
        font, size, text = line["font"], line["size"], line["text"]
        if font == "AbdoLine":  # a question title, maybe on two lines
            if in_title and current is not None:
                current["title"] += " " + text
            else:
                number = _NUMBER.search(text)
                title = re.split(r"س\s*[-:]\s*", text, maxsplit=1)[-1].strip()
                current = {"number": int(number.group(1)) if number else len(questions) + 1, "title": title,
                           "keywords": "", "parts": []}
                questions.append(current)
                part = None
            in_title, in_keywords = True, False
            continue
        in_title = False
        if current is None:
            continue
        label = _label(line)
        if label == "end":
            break
        if label == "skip":
            continue
        if label == "answer":
            part = None
            continue
        if label:
            part = label
            current["parts"].append((part, []))
            in_keywords = False
            last_y = None
            continue
        if font.startswith("(AH)") or font.startswith("AGA") or font.startswith("fotograami"):
            continue  # decorative chapter titles
        keywords = _KEYWORDS.match(text)
        if keywords or in_keywords:
            current["keywords"] += (" " if current["keywords"] else "") + (text[keywords.end():] if keywords else text)
            in_keywords = True
            continue
        if part is None:
            continue
        paragraphs = current["parts"][-1][1]
        new = (not paragraphs or last_y is None
               or (line["page"] == last_page and line["y"] - last_y > 26)
               or (line["page"] != last_page and re.search(r"[.:؟?!»]$", paragraphs[-1][1])))
        if new:
            paragraphs.append((line["page"], text))
        else:
            page, previous = paragraphs[-1]
            paragraphs[-1] = (page, previous + " " + text)
        last_y, last_page = line["y"], line["page"]
    return questions


def _clean(text: str) -> str:
    text = _VERSE_RUN.sub(VERSE, text)
    text = re.sub(r"\(الآية\)(?:\s*\(الآية\))+", VERSE, text)
    text = _SPACES.sub(" ", text.replace("\uE000", ""))
    return re.sub(r" +([،؛])", r"\1", text).strip()  # a written-out honorific before a comma


def _where(question: dict, part: str, pages: list[int]) -> str:
    where = f"المسألة ({question['number']}): {question['title']} — {PART_AR.get(part, part)}"
    first, last = min(pages), max(pages)
    return where + (f" (ص{first})" if first == last else f" (ص{first}–{last})")


def build(lines: list[dict], min_questions: int = MIN_QUESTIONS) -> tuple[dict, list[dict]]:
    """Source record and passages: the question and its short answer, then the detailed answer in chunks."""
    questions = parse(lines)
    if len(questions) < min_questions:
        raise ImportError_(f"only {len(questions)} questions from «بينات»: the file looks incomplete")
    passages: list[dict] = []
    for q in questions:
        q["title"] = _clean(q["title"])
        similar = [_clean(t) for part, ps in q["parts"] if part == "similar" for _, t in ps]
        keywords = "؛ ".join(x for x in [q["title"], *similar, _clean(q["keywords"])] if x)
        for part, paragraphs in q["parts"]:
            if part == "similar" or not paragraphs:
                continue
            group: list[tuple[int, str]] = []

            def flush() -> None:
                if not group:
                    return
                pages = [p for p, _ in group]
                text = "\n".join(t for _, t in group)
                if part in ("question", "gist", "short"):
                    text = f"{PART_AR[part]}: {text}"
                passages.append({"id": f"{PREFIX}:{len(passages) + 1}", "source_id": SOURCE_ID, "kind": "qa",
                                 "location": _where(q, part, pages), "text": text, "keywords": keywords})
                group.clear()

            for page, paragraph in paragraphs:
                paragraph = _clean(paragraph)
                if not paragraph:
                    continue
                if group and sum(len(t) + 1 for _, t in group) + len(paragraph) > MAX_CHUNK:
                    flush()
                group.append((page, paragraph))
            flush()
    return dict(SOURCE), passages


def add_to_corpus(corpus: dict, pdf_path: str | Path) -> dict:
    """Add «بينات» to a corpus built by quranpedia.build_corpus."""
    source, passages = build(read_pdf(pdf_path))
    corpus["sources"].append(source)
    corpus["passages"].extend(passages)
    corpus.setdefault("_provenance", {})["bayyinat_sha256"] = SHA256
    return corpus
