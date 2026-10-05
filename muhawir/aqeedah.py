"""Import four creed (aqeedah) books of the early scholars, the domain the challenge's source package
assigns to «sources of the first three centuries»:

  «أصول السنة» by Imam Ahmad ibn Hanbal (d. 241 AH), Dar al-Manar, al-Kharj, 1st ed., 1411 AH
  «شرح السنة» by al-Muzani (d. 264 AH), ed. Jamal Azzun, Maktabat al-Ghuraba al-Athariyya, 1st ed., 1415/1995
  «كتاب التوحيد وإثبات صفات الرب عز وجل» by Ibn Khuzayma (d. 311 AH), ed. Abd al-Aziz al-Shahwan,
      Maktabat al-Rushd, Riyadh, 5th ed., 1414/1994
  «العقيدة الطحاوية» by al-Tahawi (d. 321 AH), ed. al-Albani, al-Maktab al-Islami, Beirut, 1st ed., 1398/1978

Ibn Khuzayma and al-Tahawi died in the first quarter of the fourth century; SOURCES.md says so.

Source: the OpenITI corpus (github.com/OpenITI), pinned to one commit per repository so every build gives
the same passages; CC BY-NC-SA 4.0 (DOI 10.5281/zenodo.17767721). The files are OpenITI mARkdown, read as
in fatawa.py; besides the markers, only these are removed: the «[ص: 16]» page headings of the Ibn Khuzayma
file, the «$» tag and stray HTML tags; the verse marks «@QB@…@QE@» of the al-Tahawi file become «﴿…﴾». The wording is unchanged. Passages are whole paragraphs joined up to MAX_CHUNK
characters under one heading, each with the book, its heading and the page.
"""
from __future__ import annotations

import re

from .fatawa import parse

MAX_CHUNK = 1200
LICENCE = "النص من مدونة OpenITI المفتوحة (github.com/OpenITI، رخصة CC BY-NC-SA 4.0، DOI 10.5281/zenodo.17767721)."

BOOKS = {
    "usul-al-sunna": {
        "prefix": "ua", "repo": "0250AH", "commit": "0cc00c0646c81988c75a3815fe6d9a5fa65ce592",
        "path": "data/0241IbnHanbal/0241IbnHanbal.UsulSunna/0241IbnHanbal.UsulSunna.Shamela0006418-ara1",
        "title": "أصول السنة", "name": "أصول السنة للإمام أحمد",
        "about": "«أصول السنة» للإمام أحمد بن حنبل (ت 241هـ)، دار المنار، الخرج، الطبعة الأولى 1411هـ "
                 "(المكتبة الشاملة، الكتاب 6418). ",
        "url": "https://shamela.ws/book/6418", "min_passages": 3,
    },
    "sharh-al-sunna-muzani": {
        "prefix": "um", "repo": "0275AH", "commit": "44e1c36738a2bf5c14dafa232a6ae1891e6171cd",
        "path": ("data/0264IbnYahyaMuzani/0264IbnYahyaMuzani.SharhSunna/"
                 "0264IbnYahyaMuzani.SharhSunna.Shamela0006484-ara1"),
        "title": "شرح السنة", "name": "شرح السنة للمزني",
        "about": "«شرح السنة» لأبي إبراهيم إسماعيل بن يحيى المزني (ت 264هـ)، صاحب الإمام الشافعي، تحقيق جمال عزون، "
                 "مكتبة الغرباء الأثرية، الطبعة الأولى 1415هـ/1995م (المكتبة الشاملة، الكتاب 6484). ",
        "url": "https://shamela.ws/book/6484", "min_passages": 5,
    },
    "tawhid-ibn-khuzayma": {
        "prefix": "uk", "repo": "0325AH", "commit": "089e665b4958e0f145a46941987fb81cf3dda1b8",
        "path": ("data/0311IbnKhuzaymaNaysaburi/0311IbnKhuzaymaNaysaburi.Tawhid/"
                 "0311IbnKhuzaymaNaysaburi.Tawhid.Shamela0013011-ara1"),
        "title": "كتاب التوحيد", "name": "كتاب التوحيد لابن خزيمة",
        "about": "«كتاب التوحيد وإثبات صفات الرب عز وجل» لأبي بكر محمد بن إسحاق بن خزيمة (ت 311هـ)، تحقيق عبد العزيز "
                 "الشهوان، مكتبة الرشد، الرياض، الطبعة الخامسة 1414هـ/1994م (المكتبة الشاملة، الكتاب 13011). "
                 "أحاديثه مروية بأسانيدها، ولا يُذكر في المقطع حكم عليها. ",
        "url": "https://shamela.ws/book/13011", "min_passages": 300,
    },
    "aqeeda-tahawiyya": {
        "prefix": "ut", "repo": "0325AH", "commit": "089e665b4958e0f145a46941987fb81cf3dda1b8",
        "path": "data/0321Tahawi/0321Tahawi.MatnCaqida/0321Tahawi.MatnCaqida.JK000126-ara1",
        "title": "العقيدة الطحاوية", "name": "العقيدة الطحاوية",
        "about": "«العقيدة الطحاوية» لأبي جعفر الطحاوي (ت 321هـ)، بشرح وتعليق محمد ناصر الدين الألباني، المكتب "
                 "الإسلامي، بيروت، الطبعة الأولى 1398هـ/1978م؛ والمنقول هنا متن الطحاوي وحده. ",
        "url": "", "min_passages": 10,
    },
}

for _book in BOOKS.values():
    _book["file"] = _book["path"].rsplit("/", 1)[1]
    _book["download"] = (
        f"https://raw.githubusercontent.com/OpenITI/{_book['repo']}/{_book['commit']}/{_book['path']}",
        f"https://cdn.jsdelivr.net/gh/OpenITI/{_book['repo']}@{_book['commit']}/{_book['path']}")

FILES = [book["file"] for book in BOOKS.values()]
_PAGE_HEADING = re.compile(r"^###\s*\|+\s*\[ص:\s*\d+\]\s*$", re.MULTILINE)  # «[ص: 16]» in the Ibn Khuzayma file
_DOLLAR = re.compile(r"(?<=^# )\$\s*", re.MULTILINE)


class ImportError_(ValueError):
    pass


def _where(name: str, heading: str, pages: list[tuple[int, int]]) -> str:
    where = heading if heading and heading != name else ""
    if pages:
        (v1, p1), (v2, p2) = pages[0], pages[-1]
        span = f"ص{p1}" if (v1, p1) == (v2, p2) else (f"ص{p1}–{p2}" if v1 == v2 else f"ج{v1} ص{p1}–ج{v2} ص{p2}")
        where = f"{where} ({span})" if where else span
    return where or name


def build(source_id: str, text: str, min_passages: int | None = None) -> tuple[dict, list[dict]]:
    """Source record and passages of one book: paragraphs joined up to MAX_CHUNK characters per heading."""
    book = BOOKS[source_id]
    text = _DOLLAR.sub("", _PAGE_HEADING.sub("", text))
    text = text.replace("@QB@", "﴿").replace("@QE@", "﴾")  # verses marked in the al-Tahawi file
    try:
        paragraphs = parse(text, book["title"])
    except ValueError as exc:
        raise ImportError_(str(exc)) from exc
    passages: list[dict] = []
    group: list[dict] = []

    def flush() -> None:
        if not group:
            return
        pages = [pg for p in group for pg in p["pages"]]
        heading = group[0]["heading"]
        passages.append({"id": f"{book['prefix']}:{len(passages) + 1}", "source_id": source_id, "kind": "aqeedah",
                         "location": _where(book["title"], heading, pages),
                         "text": "\n".join(p["text"] for p in group),
                         "keywords": heading if heading != book["title"] else ""})
        group.clear()

    for para in paragraphs:
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
    """Add the creed books whose texts are given ({source_id: file text}) to a corpus."""
    for source_id, text in texts.items():
        source, passages = build(source_id, text)
        corpus["sources"].append(source)
        corpus["passages"].extend(passages)
    corpus.setdefault("_provenance", {})["aqeedah_books"] = sorted(texts)
    return corpus
