"""Import «السيرة النبوية» by Ibn Hisham (d. 213 AH), his recension of the Sira of Ibn Ishaq
(d. 151 AH): the Prophet's life ﷺ from his lineage to his death, told in reports with their chains.

Source: the OpenITI corpus (github.com/OpenITI/0225AH), version
0213IbnHisham.SiraNabawiyya.Shamela0023833-ara1.completed (ed. Mustafa al-Saqqa, Ibrahim al-Abyari and
Abd al-Hafiz Shalabi, Mustafa al-Babi al-Halabi, Cairo, 2nd ed., 1375/1955), marked "completed" (checked)
by OpenITI and pinned to one commit so every build gives the same passages. Licence: CC BY-NC-SA 4.0
(non-commercial use with attribution); cite OpenITI, DOI 10.5281/zenodo.17767721.

The file is in OpenITI mARkdown, like «بداية المجتهد» (see bidaya.py): "### |" is a chapter heading,
"### ||" is a sub-heading the editors added in brackets, "# " starts a paragraph and "~~" continues it,
"PageV01P132" marks the END of page 132 of volume 1. Only these markers are removed; the wording is
unchanged. Passages are whole paragraphs joined up to about MAX_CHUNK characters within one sub-heading,
and each carries its chapter, sub-heading, volume and page so it can be checked in the printed edition.

Sections the editors titled as poems («شعر…», «قصيدة…», «رثاء…», «ما قيل من الشعر…») are left out: they
are verse, not reports of events, and would only crowd the search. Poems quoted inside a narrative are kept.
"""
from __future__ import annotations

import re

from .bidaya import _PAGE, _clean
from .normalize import normalize

COMMIT = "2cfdc8cf1b89abfd15463d1c429fbb822aa5d9a5"
PATH = ("data/0213IbnHisham/0213IbnHisham.SiraNabawiyya/"
        "0213IbnHisham.SiraNabawiyya.Shamela0023833-ara1.completed")
FILE = "0213IbnHisham.SiraNabawiyya.Shamela0023833-ara1.completed"
DOWNLOAD = (f"https://raw.githubusercontent.com/OpenITI/0225AH/{COMMIT}/{PATH}",
            f"https://cdn.jsdelivr.net/gh/OpenITI/0225AH@{COMMIT}/{PATH}")
SOURCE_ID = "sira-ibn-hisham"
PREFIX = "s"
MAX_CHUNK = 1200
MIN_PASSAGES = 1500  # a truncated download is refused

SOURCE = {
    "id": SOURCE_ID,
    "name": "السيرة النبوية لابن هشام",
    "about": "«السيرة النبوية» لعبد الملك بن هشام (ت 213هـ)، وهي تهذيبه لسيرة محمد بن إسحاق (ت 151هـ)، "
             "تحقيق مصطفى السقا وإبراهيم الأبياري وعبد الحفيظ الشلبي، مطبعة مصطفى البابي الحلبي بمصر، الطبعة الثانية "
             "1375هـ/1955م. أخبارها روايات تاريخية بأسانيدها، متفاوتة في الصحة، فيُنسب ما فيها إلى راويه. "
             "النص من مدونة OpenITI المفتوحة (github.com/OpenITI، رخصة CC BY-NC-SA 4.0، DOI 10.5281/zenodo.17767721).",
    "url": "https://shamela.ws/book/23833",
}

# a section the editors titled as a poem (normalized spelling, without the brackets)
_POEM = re.compile(r"^(?:شعر|قصيده|رثاء|مرثي|ارتجاز|رجز|ما قيل من الشعر|ما قيل فيه من الشعر)")


class ImportError_(ValueError):
    pass


def _heading(line: str) -> str:
    text = _clean(_PAGE.sub("", line.lstrip("#").strip().lstrip("|").strip()))
    text = text.strip().rstrip(":").strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text.rstrip(":.").strip()


def is_poem(heading: str) -> bool:
    return bool(_POEM.match(normalize(heading)))


def parse(text: str) -> list[dict]:
    """Paragraphs in order: {"text", "chapter", "heading", "pages": [(volume, page), ...]}; poem sections left out."""
    head, sep, body = text.partition("#META#Header#End#")
    if not sep or "السيرة النبوية لابن هشام" not in head:
        raise ImportError_("this is not the OpenITI file of «السيرة النبوية» by Ibn Hisham")

    paragraphs: list[dict] = []
    ends: list[tuple[int, int]] = []
    chapter = heading = ""
    current: dict | None = None
    skipping = False

    def add_piece(piece: str) -> None:
        piece = _clean(piece)
        if piece and current is not None:
            current["pieces"].append((piece, len(ends)))

    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("###"):
            level2 = line.startswith("### ||")
            title = _heading(line)
            if level2:
                heading = title
            else:
                chapter, heading = title, ""
            skipping = is_poem(title) or (level2 and is_poem(chapter))
            current = None
            # a heading line may also close a page: keep the page count right
            for m in _PAGE.finditer(line):
                ends.append((int(m.group(1)), int(m.group(2))))
            continue
        if line.startswith("#") and not line.startswith("# "):
            continue  # other OpenITI tags
        if line.startswith("# ") and not _PAGE.fullmatch(line[2:].strip()):
            current = None if skipping else {"chapter": chapter, "heading": heading, "pieces": []}
            if current is not None:
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
        out.append({"text": " ".join(t for t, _ in p["pieces"]), "chapter": p["chapter"],
                    "heading": p["heading"], "pages": pages})
    return out


def _where(chapter: str, heading: str, pages: list[tuple[int, int]]) -> str:
    where = "، ".join(x for x in dict.fromkeys((chapter, heading)) if x) or "مقدمة"
    if pages:
        (v1, p1), (v2, p2) = pages[0], pages[-1]
        span = f"ص{p1}" if (v1, p1) == (v2, p2) else (f"ص{p1}–{p2}" if v1 == v2 else f"ص{p1}–ج{v2} ص{p2}")
        where += f" (ج{v1}، {span})"
    return where


def build(text: str, min_passages: int = MIN_PASSAGES) -> tuple[dict, list[dict]]:
    """Source record and passages: whole paragraphs, joined up to MAX_CHUNK characters under one sub-heading."""
    passages: list[dict] = []
    group: list[dict] = []

    def flush() -> None:
        if not group:
            return
        pages = [pg for p in group for pg in p["pages"]]
        first = group[0]
        passages.append({"id": f"{PREFIX}:{len(passages) + 1}", "source_id": SOURCE_ID, "kind": "seerah",
                         "location": _where(first["chapter"], first["heading"], pages),
                         "text": "\n".join(p["text"] for p in group),
                         "keywords": "؛ ".join(x for x in dict.fromkeys((first["chapter"], first["heading"])) if x)})
        group.clear()

    for para in parse(text):
        size = sum(len(p["text"]) + 1 for p in group)
        same = group and (para["chapter"], para["heading"]) == (group[0]["chapter"], group[0]["heading"])
        if group and (not same or size + len(para["text"]) > MAX_CHUNK):
            flush()
        group.append(para)
    flush()
    if len(passages) < min_passages:
        raise ImportError_(f"only {len(passages)} passages from the Sira: the file looks incomplete")
    return dict(SOURCE), passages


def add_to_corpus(corpus: dict, text: str) -> dict:
    """Add Ibn Hisham's Sira to a corpus built by quranpedia.build_corpus."""
    source, passages = build(text)
    corpus["sources"].append(source)
    corpus["passages"].extend(passages)
    corpus.setdefault("_provenance", {})["sira_commit"] = COMMIT
    return corpus
