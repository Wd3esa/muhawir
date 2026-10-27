"""Import «بداية المجتهد ونهاية المقتصد» by Ibn Rushd (d. 595 AH), a classical book of
comparative fiqh: for each issue it names the views of the jurists (Malik, Abu Hanifa,
al-Shafi'i, Ahmad and others), their evidence and the cause of their disagreement.

Source: the OpenITI corpus (github.com/OpenITI/0600AH), version
0595IbnRushdHafid.BidayatMujtahid.Shamela0021739-ara1 (Dar al-Hadith, Cairo, 1425/2004),
pinned to one commit so every build gives the same passages. Licence: CC BY-NC-SA 4.0
(non-commercial use with attribution); cite OpenITI, DOI 10.5281/zenodo.17767721.

The file is in OpenITI mARkdown: "### |" lines are headings, "# " starts a paragraph and
"~~" continues it, "PageV01P132" marks the END of page 132 of volume 1, and "ms0127" is a
milestone. Only these markers are removed; the wording is unchanged. Passages are whole
paragraphs joined up to about MAX_CHUNK characters within one heading, and a new issue
(«المسألة…») always starts a new passage, so one issue's views stay together, and each passage
carries its book, chapter, volume and page so it can be checked in the printed edition.
"""
from __future__ import annotations

import re

COMMIT = "ea4bdc6517a49d07106f223aa0869aa7c21b9589"
PATH = ("data/0595IbnRushdHafid/0595IbnRushdHafid.BidayatMujtahid/"
        "0595IbnRushdHafid.BidayatMujtahid.Shamela0021739-ara1")
FILE = "0595IbnRushdHafid.BidayatMujtahid.Shamela0021739-ara1"
DOWNLOAD = (f"https://raw.githubusercontent.com/OpenITI/0600AH/{COMMIT}/{PATH}",
            f"https://cdn.jsdelivr.net/gh/OpenITI/0600AH@{COMMIT}/{PATH}")
SOURCE_ID = "bidayat-al-mujtahid"
PREFIX = "f"
MAX_CHUNK = 1200
MIN_PASSAGES = 1500  # a truncated download is refused

SOURCE = {
    "id": SOURCE_ID,
    "name": "بداية المجتهد",
    "about": "«بداية المجتهد ونهاية المقتصد» لابن رشد الحفيد (ت 595هـ)، طبعة دار الحديث، القاهرة 1425هـ/2004م. "
             "كتاب في الفقه المقارن يذكر أقوال الفقهاء وأدلتهم وسبب اختلافهم، وما يرجّحه فيه ابن رشد فهو رأيه. "
             "النص من مدونة OpenITI المفتوحة (github.com/OpenITI، رخصة CC BY-NC-SA 4.0، "
             "DOI 10.5281/zenodo.17767721).",
    "url": "https://shamela.ws/book/21739",
}

_PAGE = re.compile(r"PageV(\d+)P(\d+)")
# the start of a new issue in a section: «المسألة الرابعة…», «وأما المسألة الثانية…», «فأما المسألة…»
_ISSUE = re.compile(r"^(?:[وف]?أما\s+)?المسألة\s")
_MILESTONE = re.compile(r"\bms\d+\b")
_SPACES = re.compile(r"[ \t ]+")
_TAG = re.compile(r"</?span\b[^>]*>")  # HTML left in some OpenITI files (e.g. <span class="matn">)
# A printed page that ends inside a sentence starts a new paragraph in the digital text, and a passage then began
# «، ومن رأى أن…» or «صلى الله عليه وسلم أنها خمسة…» (21 October 2026). A paragraph goes on in the next one when
# it does not end a sentence (a footnote number after the end is allowed), or when the next one opens with a comma
# or with words that cannot start a sentence.
_SENTENCE_DONE = re.compile(r"[.؟?!»\"”)\]﴾}]\s*(?:[(\[][\d٠-٩]+[)\]])?\s*$")
_GOES_ON = re.compile(r"^(?:[،؛,]|(?:صلى الله عليه وسلم|عليه السلام|رضي الله عنهما|رضي الله عنه|رحمه الله|تعالى)(?:\s|$|[:،.]))")


def _joined(pieces) -> str:
    """Pieces of one paragraph joined by a space, with no space left before a comma that goes on a sentence."""
    return re.sub(r" (?=[،؛,])", "", " ".join(pieces))


def continues(before: str, after: str) -> bool:
    """True when paragraph `after` is the rest of the sentence that paragraph `before` left unfinished."""
    if not before or _ISSUE.match(after):
        return False
    return not _SENTENCE_DONE.search(before) or bool(_GOES_ON.match(after))


class ImportError_(ValueError):
    """Raised when the source file does not have the structure this importer expects."""
    pass


def _clean(text: str) -> str:
    return _SPACES.sub(" ", _MILESTONE.sub("", _TAG.sub("", text))).strip()


def _heading(line: str) -> str:
    text = _clean(_PAGE.sub("", line.lstrip("#").strip().lstrip("|").strip()))
    return text.strip("[] ").rstrip(":.").strip()


def parse(text: str) -> list[dict]:
    """Paragraphs in order: {"text", "heading", "book", "pages": [(volume, page), ...]}."""
    if "#META#Header#End#" not in text or "بداية المجتهد" not in text.split("#META#Header#End#")[0]:
        raise ImportError_("this is not the OpenITI file of «بداية المجتهد»")
    body = text.split("#META#Header#End#", 1)[1]

    # Each text piece is tagged with the number of page ends seen before it; a page marker
    # ends the page, so a piece lies on the next marker's page.
    paragraphs: list[dict] = []
    ends: list[tuple[int, int]] = []
    book = heading = ""
    current: dict | None = None

    def add_piece(piece: str) -> None:
        """Add a cleaned piece of text to the section being read; text outside any section is ignored."""
        piece = _clean(piece)
        if piece and current is not None:
            current["pieces"].append((piece, len(ends)))

    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("###"):
            heading = _heading(line)
            if heading.startswith("كتاب"):
                book = heading
            current = None
            continue
        if line.startswith("#") and not line.startswith("# "):
            continue  # other OpenITI tags
        if line.startswith("# ") and not _PAGE.fullmatch(line[2:].strip()):
            last = current["pieces"][-1][0] if current is not None and current["pieces"] else ""
            if not continues(last, _clean(_PAGE.sub("", line[2:]))):
                current = {"heading": heading, "book": book, "pieces": []}
                paragraphs.append(current)
            line = line[2:]
        elif line.startswith("~~"):
            line = line[2:]
        elif line.startswith("# "):
            line = line[2:]  # a page marker on its own line
        # split the line at page markers: text before a marker belongs to the page it ends
        pos = 0
        for m in _PAGE.finditer(line):
            add_piece(line[pos:m.start()])
            ends.append((int(m.group(1)), int(m.group(2))))
            pos = m.end()
        add_piece(line[pos:])

    out = []
    for p in paragraphs:
        if not p["pieces"]:
            continue
        pages = [ends[min(i, len(ends) - 1)] for _, i in p["pieces"]] if ends else []
        out.append({"text": _joined(t for t, _ in p["pieces"]), "heading": p["heading"],
                    "book": p["book"], "pages": pages})
    return out


def _where(book: str, heading: str, pages: list[tuple[int, int]]) -> str:
    parts = [x for x in dict.fromkeys((book, heading)) if x]
    where = "، ".join(parts) or "مقدمة المؤلف"
    if pages:
        (v1, p1), (v2, p2) = pages[0], pages[-1]
        span = f"ص{p1}" if (v1, p1) == (v2, p2) else (f"ص{p1}–{p2}" if v1 == v2 else f"ص{p1}–ج{v2} ص{p2}")
        where += f" (ج{v1}، {span})"
    return where


def build(text: str, min_passages: int = MIN_PASSAGES) -> tuple[dict, list[dict]]:
    """Source record and passages: whole paragraphs, joined up to MAX_CHUNK characters under one heading."""
    passages: list[dict] = []
    group: list[dict] = []

    def flush() -> None:
        """Write the paragraphs gathered so far as one passage, then start a new group."""
        if not group:
            return
        pages = [pg for p in group for pg in p["pages"]]
        first = group[0]
        passages.append({"id": f"{PREFIX}:{len(passages) + 1}", "source_id": SOURCE_ID, "kind": "fiqh",
                         "location": _where(first["book"], first["heading"], pages),
                         "text": "\n".join(p["text"] for p in group),
                         "keywords": "؛ ".join(x for x in dict.fromkeys((first["book"], first["heading"])) if x)})
        group.clear()

    for para in parse(text):
        size = sum(len(p["text"]) + 1 for p in group)
        new_issue = bool(_ISSUE.match(para["text"]))  # «المسألة الرابعة…» starts its own passage
        if group and (new_issue or para["heading"] != group[0]["heading"] or size + len(para["text"]) > MAX_CHUNK):
            flush()
        group.append(para)
    flush()
    if len(passages) < min_passages:
        raise ImportError_(f"only {len(passages)} passages from «بداية المجتهد»: the file looks incomplete")
    return dict(SOURCE), passages


def add_to_corpus(corpus: dict, text: str) -> dict:
    """Add «بداية المجتهد» to a corpus built by quranpedia.build_corpus."""
    source, passages = build(text)
    corpus["sources"].append(source)
    corpus["passages"].extend(passages)
    corpus.setdefault("_provenance", {})["bidaya_commit"] = COMMIT
    return corpus
