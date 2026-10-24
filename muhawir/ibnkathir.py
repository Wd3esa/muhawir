"""Import «تفسير القرآن العظيم» by Ibn Kathir (d. 774 AH): a second tafsir beside al-Tabari, shorter and plainer,
that explains verses by verses and by the hadith, and often says how sound a report is.

Source: the OpenITI corpus (github.com/OpenITI/0775AH), version 0774IbnKathir.TafsirQuran.Shamela0008473-ara1
(ed. Sami b. Muhammad Salama, Dar Tayba, 2nd ed., 1420/1999), the corpus's main version of the book, pinned to one
commit so every build gives the same passages. Licence: CC BY-NC-SA 4.0 (non-commercial use with attribution); cite
OpenITI, DOI 10.5281/zenodo.17767721.

The file is in OpenITI mARkdown: "### | تفسير سورة الكهف" opens a surah, "### || {الحمد لله الذي أنزل … (1) قيما … (5)}"
opens the commentary on the verses it quotes (with their numbers), other "### ||" headings («فصل») are sections of
it; "# " starts a paragraph and "~~" continues it. Removed: "PageV05P134" (the END of a page), "ms2914", and the
editor's footnote numbers «(3)» in the text (the footnotes themselves are not in the file). The wording is unchanged.
Passages are whole paragraphs joined up to about MAX_CHUNK characters under one heading; each names its surah and
verses («تفسير سورة الكهف، الآيات 1–5»). The editor's preface is left out; Ibn Kathir's own introduction and his
«كتاب فضائل القرآن» are kept.
"""
from __future__ import annotations

import re

from .bidaya import _PAGE, _clean
from .sahihayn import split_long

COMMIT = "61b0c7891699554aeff8dffdb8b1a2a59d2d22ab"
PATH = ("data/0774IbnKathir/0774IbnKathir.TafsirQuran/"
        "0774IbnKathir.TafsirQuran.Shamela0008473-ara1.mARkdown")
FILE = "0774IbnKathir.TafsirQuran.Shamela0008473-ara1.mARkdown"
DOWNLOAD = (f"https://raw.githubusercontent.com/OpenITI/0775AH/{COMMIT}/{PATH}",
            f"https://cdn.jsdelivr.net/gh/OpenITI/0775AH@{COMMIT}/{PATH}")
SOURCE_ID = "tafsir-ibn-kathir"
PREFIX = "ik"
MAX_CHUNK = 1200
MIN_PASSAGES = 4000  # a truncated download is refused

SOURCE = {
    "id": SOURCE_ID,
    "name": "تفسير ابن كثير",
    "about": "«تفسير القرآن العظيم» لأبي الفداء إسماعيل بن عمر بن كثير الدمشقي (ت 774هـ)، تحقيق سامي بن محمد سلامة، "
             "دار طيبة، الطبعة الثانية 1420هـ/1999م. يفسر القرآن بالقرآن وبالحديث وأقوال الصحابة والتابعين، ويبيّن كثيرًا "
             "حال ما يرويه من الأخبار. النص من مدونة OpenITI المفتوحة (github.com/OpenITI، رخصة CC BY-NC-SA 4.0، "
             "DOI 10.5281/zenodo.17767721).",
    "url": "https://shamela.ws/book/8473",
}

_FOOTNOTE = re.compile(r"\s*\(\s*[\d٠-٩]+\s*\)")
_NUMBER = re.compile(r"\(\s*([\d٠-٩]+)\s*\)")
_SURAH = re.compile(r"^(?:تفسير\s+)?سور[ةه]\s+(.+)$")
_SKIP = re.compile(r"^مقدمة\s+(?:المحقق|التحقيق|الناشر)")


class ImportError_(ValueError):
    pass


def _title(line: str) -> str:
    return _clean(_PAGE.sub("", line.lstrip("#").strip().lstrip("|").strip()))


def _verses(heading: str) -> str:
    """«الآيات 1–5» from the numbers in a heading that quotes verses, «الآية 6» for one; "" when it quotes none."""
    if "{" not in heading:
        return ""
    numbers = [int(n.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))) for n in _NUMBER.findall(heading)]
    if not numbers:
        return ""
    first, last = min(numbers), max(numbers)
    return f"الآية {first}" if first == last else f"الآيات {first}–{last}"


def parse(text: str) -> list[dict]:
    """Paragraphs in order: {"text", "chapter", "verses", "heading", "pages"}."""
    head, sep, body = text.partition("#META#Header#End#")
    if not sep or "تفسير القرآن العظيم" not in head:
        raise ImportError_("this is not the OpenITI file of «تفسير القرآن العظيم» by Ibn Kathir")
    paragraphs: list[dict] = []
    ends: list[tuple[int, int]] = []
    chapter = verses = heading = ""
    current: dict | None = None
    skipping = False

    def add_piece(piece: str) -> None:
        piece = _clean(_FOOTNOTE.sub("", piece))
        if piece and current is not None:
            current["pieces"].append((piece, len(ends)))

    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("###"):
            title = _title(line)
            if line.startswith("### ||"):
                found = _verses(title)
                verses, heading = (found, "") if found else (verses, _FOOTNOTE.sub("", title).strip(" :"))
            else:
                m = _SURAH.match(title)
                chapter = f"تفسير سورة {m.group(1).strip()}" if m else _FOOTNOTE.sub("", title).strip(" :")
                if chapter == "فاتحة الكتاب":
                    chapter = "تفسير سورة الفاتحة"
                verses = heading = ""
                skipping = bool(_SKIP.match(title))
            current = None
            ends += [(int(a), int(b)) for a, b in _PAGE.findall(line)]
            continue
        if line.startswith("#") and not line.startswith("# "):
            continue  # other OpenITI tags
        if line.startswith("# ") and not _PAGE.fullmatch(line[2:].strip()):
            current = None if skipping else {"chapter": chapter, "verses": verses, "heading": heading, "pieces": []}
            if current is not None:
                paragraphs.append(current)
            line = line[2:]
        elif line.startswith(("~~", "# ")):
            line = line[2:]
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
        out.append({"text": " ".join(t for t, _ in p["pieces"]), "chapter": p["chapter"], "verses": p["verses"],
                    "heading": p["heading"], "pages": pages})
    return out


def _where(chapter: str, verses: str, heading: str, pages: list[tuple[int, int]]) -> str:
    where = "، ".join(x for x in (chapter, verses, heading) if x) or "المقدمة"
    if pages:
        (v1, p1), (v2, p2) = pages[0], pages[-1]
        span = f"ص{p1}" if (v1, p1) == (v2, p2) else (f"ص{p1}–{p2}" if v1 == v2 else f"ص{p1}–ج{v2} ص{p2}")
        where += f" (ج{v1}، {span})"
    return where


def build(text: str, min_passages: int = MIN_PASSAGES) -> tuple[dict, list[dict]]:
    """Source record and passages: whole paragraphs joined up to MAX_CHUNK characters under one heading."""
    passages: list[dict] = []
    group: list[dict] = []

    def flush() -> None:
        if not group:
            return
        first = group[0]
        pages = [pg for p in group for pg in p["pages"]]
        surah = first["chapter"].removeprefix("تفسير ")
        passages.append({"id": f"{PREFIX}:{len(passages) + 1}", "source_id": SOURCE_ID, "kind": "tafsir",
                         "location": _where(first["chapter"], first["verses"], first["heading"], pages),
                         "text": "\n".join(p["text"] for p in group),
                         "keywords": "؛ ".join(x for x in (surah, first["verses"]) if x)})
        group.clear()

    paragraphs = [dict(p, text=part) for p in parse(text) for part in split_long(p["text"], MAX_CHUNK)]
    for para in paragraphs:  # a paragraph longer than a passage is cut at sentence ends first
        size = sum(len(p["text"]) + 1 for p in group)
        key = (para["chapter"], para["verses"], para["heading"])
        if group and (key != (group[0]["chapter"], group[0]["verses"], group[0]["heading"])
                      or size + len(para["text"]) > MAX_CHUNK):
            flush()
        group.append(para)
    flush()
    if len(passages) < min_passages:
        raise ImportError_(f"only {len(passages)} passages from Ibn Kathir: the file looks incomplete")
    return dict(SOURCE), passages


def add_to_corpus(corpus: dict, text: str) -> dict:
    """Add Ibn Kathir's Tafsir to a corpus built by quranpedia.build_corpus."""
    source, passages = build(text)
    corpus["sources"].append(source)
    corpus["passages"].extend(passages)
    corpus.setdefault("_provenance", {})["ibnkathir_commit"] = COMMIT
    return corpus
