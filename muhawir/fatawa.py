"""Import published fatwa collections of two Saudi scholars, so Muhawir can quote a fatwa with its
author instead of judging a matter itself:

  «مجموع فتاوى العلامة عبد العزيز بن باز» (Ibn Baz, d. 1420 AH), 30 volumes
  «مجموع فتاوى ورسائل فضيلة الشيخ محمد بن صالح العثيمين» (d. 1421 AH), Dar al-Watan / Dar al-Thurayya, 1413 AH
  «فتاوى أركان الإسلام» (Ibn Uthaymeen), Dar al-Thurayya, Riyadh, 1st ed., 1424 AH

Source: the OpenITI corpus (github.com/OpenITI/1425AH), the Shamela versions 21537, 12293 and 9924, pinned
to one commit so every build gives the same passages. OpenITI publishes its data under CC BY-NC-SA 4.0
(DOI 10.5281/zenodo.17767721). These are contemporary works: the rights in the text itself remain with
their holders, so every passage is shown attributed to its author with volume and page, and SOURCES.md
and LICENSES.md say so.

The files are in OpenITI mARkdown (see bidaya.py): "### |" is the title of a fatwa or treatise, "# "
starts a paragraph, "~~" continues it, "PageV01P132" marks the END of a page. A heading that is only a
number («1 - ») is an item of a list inside the same fatwa, not a new fatwa, so it stays with it. Only
the markers are removed; the wording is unchanged. Passages are whole paragraphs joined up to about
MAX_CHUNK characters within one fatwa, and each carries the collection, the fatwa's title, the volume
and the page.
"""
from __future__ import annotations

import re

from .bidaya import _PAGE, _clean

COMMIT = "6d71d2af1be5a49f1db3b52cdda15ea11da569de"
REPO = "1425AH"
MAX_CHUNK = 1500
LICENCE = ("النص من مدونة OpenITI المفتوحة (github.com/OpenITI، رخصة CC BY-NC-SA 4.0، DOI 10.5281/zenodo.17767721). "
           "وهو عمل معاصر، وحقوق نصه لأصحابها: يُعرض منسوبًا إلى صاحبه مع الجزء والصفحة، ويُطلب إذن أصحاب الحقوق لاستعماله.")

BOOKS = {
    "fatawa-ibn-baz": {
        "prefix": "z",
        "path": "data/1420IbnBaz/1420IbnBaz.MajmucFatawa/1420IbnBaz.MajmucFatawa.Shamela0021537-ara1",
        "title": "مجموع فتاوى العلامة عبد العزيز بن باز",
        "name": "مجموع فتاوى ابن باز",
        "about": "«مجموع فتاوى العلامة عبد العزيز بن باز رحمه الله» لسماحة الشيخ عبد العزيز بن عبد الله بن باز "
                 "(ت 1420هـ)، المفتي العام للمملكة العربية السعودية ورئيس هيئة كبار العلماء، في 30 جزءًا، "
                 "بنسخة المكتبة الشاملة (الكتاب 21537). فتاوى الشيخ أجوبةٌ عن أسئلة سُئل عنها، تُنقل منسوبة إليه. ",
        "url": "https://shamela.ws/book/21537",
        "min_passages": 8000,
    },
    "fatawa-ibn-uthaymeen": {
        "prefix": "u",
        "path": ("data/1421MuhammadCuthaymin/1421MuhammadCuthaymin.MajmucFatawa/"
                 "1421MuhammadCuthaymin.MajmucFatawa.Shamela0012293-ara1"),
        "title": "مجموع فتاوى ورسائل فضيلة الشيخ محمد بن صالح العثيمين",
        "name": "مجموع فتاوى ابن عثيمين",
        "about": "«مجموع فتاوى ورسائل فضيلة الشيخ محمد بن صالح العثيمين» للشيخ محمد بن صالح العثيمين (ت 1421هـ)، "
                 "عضو هيئة كبار العلماء في المملكة العربية السعودية، دار الوطن - دار الثريا، الطبعة الأخيرة 1413هـ، "
                 "بنسخة المكتبة الشاملة (الكتاب 12293). فتاوى الشيخ أجوبةٌ عن أسئلة سُئل عنها، تُنقل منسوبة إليه. ",
        "url": "https://shamela.ws/book/12293",
        "min_passages": 8000,
    },
    "fatawa-arkan-islam": {
        "prefix": "k",
        "path": ("data/1421MuhammadCuthaymin/1421MuhammadCuthaymin.FatawaArkanIslam/"
                 "1421MuhammadCuthaymin.FatawaArkanIslam.Shamela0009924-ara1"),
        "title": "فتاوى أركان الإسلام",
        "name": "فتاوى أركان الإسلام لابن عثيمين",
        "about": "«فتاوى أركان الإسلام» للشيخ محمد بن صالح العثيمين (ت 1421هـ)، دار الثريا للنشر والتوزيع، الرياض، "
                 "الطبعة الأولى 1424هـ، بنسخة المكتبة الشاملة (الكتاب 9924). أجوبة الشيخ عن أسئلة في العقيدة "
                 "وأركان الإسلام، تُنقل منسوبة إليه. ",
        "url": "https://shamela.ws/book/9924",
        "min_passages": 300,
    },
}

for _key, _book in BOOKS.items():
    _book["file"] = _book["path"].rsplit("/", 1)[1]
    _book["download"] = (f"https://raw.githubusercontent.com/OpenITI/{REPO}/{COMMIT}/{_book['path']}",
                         f"https://cdn.jsdelivr.net/gh/OpenITI/{REPO}@{COMMIT}/{_book['path']}")

FILES = [book["file"] for book in BOOKS.values()]
_LIST_ITEM = re.compile(r"^[\d٠-٩]+\s*[-–.)]?\s*$")  # «1 - »: an item of a list inside one fatwa


class ImportError_(ValueError):
    pass


def _heading(line: str) -> str:
    text = _clean(_PAGE.sub("", line.lstrip("#").strip().lstrip("|").strip()))
    return text.strip("[]$ ").rstrip(":.").strip()  # "$" is an OpenITI tag before some numbered titles


def parse(text: str, title: str) -> list[dict]:
    """Paragraphs in order: {"text", "heading", "pages": [(volume, page), ...]}."""
    head, sep, body = text.partition("#META#Header#End#")
    if not sep or title not in head:
        raise ImportError_(f"this is not the OpenITI file of «{title}»")

    paragraphs: list[dict] = []
    ends: list[tuple[int, int]] = []
    heading = ""
    current: dict | None = None

    def add_piece(piece: str) -> None:
        piece = _clean(piece)
        if piece and current is not None:
            current["pieces"].append((piece, len(ends)))

    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("###"):
            title_text = _heading(line)
            for m in _PAGE.finditer(line):
                ends.append((int(m.group(1)), int(m.group(2))))
            if _LIST_ITEM.match(title_text):  # a numbered item: part of the same fatwa
                current = {"heading": heading, "pieces": []}
                paragraphs.append(current)
                add_piece(title_text)
                continue
            heading = title_text
            current = None
            continue
        if line.startswith("#") and not line.startswith("# "):
            continue  # other OpenITI tags
        if line.startswith("# ") and not _PAGE.fullmatch(line[2:].strip()):
            current = {"heading": heading, "pieces": []}
            paragraphs.append(current)
            line = line[2:]
        elif line.startswith("~~"):
            line = line[2:]
        elif line.startswith("# "):
            line = line[2:]  # a page marker on its own line
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
        out.append({"text": " ".join(t for t, _ in p["pieces"]), "heading": p["heading"], "pages": pages})
    return out


def _where(heading: str, pages: list[tuple[int, int]]) -> str:
    where = heading or "مقدمة"
    if pages:
        (v1, p1), (v2, p2) = pages[0], pages[-1]
        span = f"ص{p1}" if (v1, p1) == (v2, p2) else (f"ص{p1}–{p2}" if v1 == v2 else f"ص{p1}–ج{v2} ص{p2}")
        where += f" (ج{v1}، {span})"
    return where


def build(source_id: str, text: str, min_passages: int | None = None) -> tuple[dict, list[dict]]:
    """Source record and passages of one collection: paragraphs joined up to MAX_CHUNK characters per fatwa."""
    book = BOOKS[source_id]
    passages: list[dict] = []
    group: list[dict] = []

    def flush() -> None:
        if not group:
            return
        pages = [pg for p in group for pg in p["pages"]]
        heading = group[0]["heading"]
        passages.append({"id": f"{book['prefix']}:{len(passages) + 1}", "source_id": source_id, "kind": "fatwa",
                         "location": _where(heading, pages),
                         "text": "\n".join(p["text"] for p in group),
                         "keywords": heading})
        group.clear()

    for para in parse(text, book["title"]):
        size = sum(len(p["text"]) + 1 for p in group)
        if group and (para["heading"] != group[0]["heading"] or size + len(para["text"]) > MAX_CHUNK):
            flush()
        group.append(para)
    flush()
    needed = book["min_passages"] if min_passages is None else min_passages
    if len(passages) < needed:
        raise ImportError_(f"only {len(passages)} passages from «{book['title']}»: the file looks incomplete")
    source = {"id": source_id, "name": book["name"], "about": book["about"] + LICENCE, "url": book["url"]}
    return source, passages


def add_to_corpus(corpus: dict, texts: dict[str, str]) -> dict:
    """Add the fatwa collections whose texts are given ({source_id: file text}) to a corpus."""
    for source_id, text in texts.items():
        source, passages = build(source_id, text)
        corpus["sources"].append(source)
        corpus["passages"].extend(passages)
    corpus.setdefault("_provenance", {})["fatawa_commit"] = COMMIT
    return corpus
