"""Import chosen classical books copied out of a Shamela 4 library with tools/shamela/ShamelaExport.java.

For each book the export gives `<n>.page.jsonl` (one line per page: «id» «<n>-<page id>», «body» the book's text,
«foot» the editor's footnotes), `<n>.title.jsonl` (the headings) and `<n>.db` (each page's volume and printed page
number, and the heading tree). Only the author's text is taken:

- the editor's footnotes («foot») are left out, and their markers in the text («(¬١)», ««١»») removed;
- the editor's or publisher's introduction and the indexes are left out (each book's first and last page of the
  author's text are set below, read from its headings);
- headings inside the page text are dropped (each passage is placed under its heading); nothing else in the
  wording is changed.

The books are classical (their authors died centuries ago); the editions are named in each source record. Passages
are paragraphs joined up to MAX_CHUNK characters under one heading, each placed by heading, volume and page, so a
reader can open the printed book at the same place. The copied files stay off GitHub (see .gitignore).
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

MAX_CHUNK = 1200
MAX_ENTRY = 2400  # one numbered entry of a grading book may run longer: the saying, its routes and the verdict
FOLDER = "shamela"
NOTE = "النص من نسخة المكتبة الشاملة الحاسوبية، دون حواشي المحقق ومقدمته."

# Shamela book number -> what the book is, its edition, and the pages that hold the author's own text
BOOKS = {
    1263: {"kind": "hadith", "name": "المقاصد الحسنة للسخاوي", "first": 23,
           "about": "«المقاصد الحسنة في بيان كثير من الأحاديث المشتهرة على الألسنة» لشمس الدين السخاوي (ت 902هـ)، "
                    "تحقيق عبد الله الصديق الغماري، تقديم عبد الوهاب عبد اللطيف. في الحكم على الأحاديث المشتهرة. "},
    856: {"kind": "hadith", "name": "كشف الخفاء للعجلوني", "first": 11,
          "about": "«كشف الخفاء ومزيل الإلباس عما اشتهر من الأحاديث على ألسنة الناس» لإسماعيل العجلوني (ت 1162هـ)، "
                   "مكتبة القدسي. في الحكم على الأحاديث المشتهرة. "},
    6056: {"kind": "hadith", "name": "الأسرار المرفوعة لعلي القاري", "first": 1,
           "about": "«الأسرار المرفوعة في الأخبار الموضوعة» (الموضوعات الكبرى) للملا علي القاري (ت 1014هـ). "
                    "في بيان الأحاديث الموضوعة. "},
    8566: {"kind": "hadith", "name": "المنار المنيف لابن القيم", "first": 1,
           "about": "«المنار المنيف في الصحيح والضعيف» لابن قيم الجوزية (ت 751هـ)، تحقيق عبد الفتاح أبو غدة. "},
    23587: {"kind": "tafsir", "name": "إعراب القرآن للنحاس", "first": 9,
            "about": "«إعراب القرآن» لأبي جعفر النحاس (ت 338هـ). في إعراب آيات القرآن ووجوه قراءتها. "},
    22928: {"kind": "tafsir", "name": "التبيان في إعراب القرآن للعكبري", "first": 2,
            "about": "«التبيان في إعراب القرآن» لأبي البقاء العكبري (ت 616هـ). "},
    23636: {"kind": "tafsir", "name": "المفردات في غريب القرآن للراغب", "first": 39, "last": 878,
            "about": "«المفردات في غريب القرآن» للراغب الأصفهاني (ت نحو 502هـ). في معاني ألفاظ القرآن. "},
    8473: {"kind": "tafsir", "name": "تفسير ابن كثير", "first": 56,
           "about": "«تفسير القرآن العظيم» لابن كثير (ت 774هـ)، تحقيق سامي بن محمد السلامة، دار طيبة. "},
    20855: {"kind": "tafsir", "name": "تفسير القرطبي", "first": 7,
            "about": "«الجامع لأحكام القرآن» لأبي عبد الله القرطبي (ت 671هـ)، تحقيق أحمد البردوني وإبراهيم أطفيش، "
                     "دار الكتب المصرية، الطبعة الثانية. "},
    41: {"kind": "tafsir", "name": "تفسير البغوي", "first": 28,
         "about": "«معالم التنزيل» (تفسير البغوي) لأبي محمد البغوي (ت 516هـ)، دار طيبة. "},
    2348: {"kind": "hadith", "name": "رياض الصالحين للنووي", "first": 7,
           "about": "«رياض الصالحين» ليحيى بن شرف النووي (ت 676هـ)، تحقيق ماهر الفحل. "},
    1956: {"kind": "hadith", "name": "الأذكار للنووي", "first": 1,
           "about": "«الأذكار» ليحيى بن شرف النووي (ت 676هـ)، تحقيق عبد القادر الأرنؤوط. "},
    197: {"kind": "seerah", "name": "زاد المعاد لابن القيم", "first": 154,
          "about": "«زاد المعاد في هدي خير العباد» لابن قيم الجوزية (ت 751هـ)، طبعة عطاءات العلم. "},
}

_TAG = re.compile(r"<[^>]{0,200}>")
# a heading inside the page text: it is the passage's location already, so it is not repeated as text
_HEADING = re.compile(r"<span data-type=['\"]title['\"][^>]*>.*?</span>", re.DOTALL)
_NOTE_MARK = re.compile(r"\s?\(¬[0-9٠-٩]+\)|\s?«[0-9٠-٩]+»")  # the editor's footnote numbers in the text
_SPACES = re.compile(r"[ \t\u00a0]+")
_ENTRY = re.compile(r"^[0-9٠-٩]{1,5}\s?[-–]\s?\S")  # «٣٩ - حديث: …»


class ImportError_(ValueError):
    pass


def source_id(book: int) -> str:
    return f"shamela-{book}"


def clean(body: str, strip_headings: bool = True) -> list[str]:
    """The author's text of one page as paragraphs: heading tags made plain lines, footnote markers removed."""
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    text = _HEADING.sub("", text) if strip_headings else text
    text = _TAG.sub("", text)
    text = _NOTE_MARK.sub("", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return [line for line in lines if line and line != "﷽"]


def _pieces(para: str) -> list[str]:
    """A paragraph longer than a passage, cut after a full stop, a question mark or a colon (a comma at worst)."""
    if len(para) <= MAX_ENTRY:
        return [para]
    out, rest = [], para
    while len(rest) > MAX_CHUNK:
        window = rest[:MAX_CHUNK]
        cut = max(window.rfind(m) for m in (". ", "؟ ", ": ", ".\u200f "))
        if cut < MAX_CHUNK // 3:
            cut = window.rfind("، ")
        if cut < MAX_CHUNK // 3:
            cut = window.rfind(" ")
        if cut <= 0:  # one unbroken run of letters: cut it where it reaches the length
            cut = MAX_CHUNK - 1
        out.append(rest[:cut + 1].strip())
        rest = rest[cut + 1:].strip()
    return out + ([rest] if rest else [])


def _pages(folder: Path, book: int) -> list[dict]:
    """The book's pages in order, each with its volume, printed page and heading."""
    con = sqlite3.connect(f"file:{folder / f'{book}.db'}?mode=ro", uri=True)
    try:
        meta = {r[0]: (r[1], r[2]) for r in con.execute("SELECT id, part, page FROM page")}
        title_rows = con.execute("SELECT id, page, parent FROM title ORDER BY page, id").fetchall()
    finally:
        con.close()
    titles = {}
    with open(folder / f"{book}.title.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            titles[int(row["id"].split("-", 1)[1])] = " ".join(clean(row.get("body", ""), strip_headings=False))
    parent = {tid: up for tid, _, up in title_rows}

    def path(tid: int) -> str:
        """«تفسير سورة الفلق، 113»: the top heading and the deepest one, when they differ."""
        top, seen = tid, set()
        while parent.get(top) and top not in seen:
            seen.add(top)
            top = parent[top]
        deep, high = titles.get(tid, ""), titles.get(top, "")
        name = deep if not high or high == deep else (f"{high}، {deep}" if deep else high)
        return name[:120]
    starts = sorted((page, path(tid)) for tid, page, _ in title_rows)  # the deepest heading starting on a page wins
    pages = []
    with open(folder / f"{book}.page.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = int(row["id"].split("-", 1)[1])
            pages.append({"id": pid, "body": row.get("body", ""), "part": meta.get(pid, (None, None))[0],
                          "page": meta.get(pid, (None, None))[1]})
    pages.sort(key=lambda p: p["id"])
    i, heading = 0, ""
    for p in pages:
        while i < len(starts) and starts[i][0] <= p["id"]:
            heading = starts[i][1] or heading
            i += 1
        p["heading"] = heading
    return pages


def _where(heading: str, spans: list[tuple]) -> str:
    def one(part, page) -> str:
        return (f"ج{part} ص{page}" if part and str(part).isdigit() else f"ص{page}") if page else ""
    first, last = one(*spans[0]), one(*spans[-1])
    span = first if first == last else f"{first}–{last}"
    return f"{heading} ({span})" if heading and span else (heading or span)


def build(folder: Path, book: int, min_passages: int = 10) -> tuple[dict, list[dict]]:
    """Source record and passages of one exported book."""
    info = BOOKS[book]
    for suffix in (".db", ".page.jsonl", ".title.jsonl"):
        if not (folder / f"{book}{suffix}").exists():
            raise ImportError_(f"{book}{suffix} is missing from {folder}")
    passages: list[dict] = []
    group: list[tuple[str, tuple]] = []
    group_heading = ""

    def flush() -> None:
        if not group:
            return
        passages.append({"id": f"h{book}:{len(passages) + 1}", "source_id": source_id(book), "kind": info["kind"],
                         "location": _where(group_heading, [g[1] for g in group]),
                         "text": "\n".join(g[0] for g in group), "keywords": group_heading})
        group.clear()

    for p in _pages(folder, book):
        if p["id"] < info["first"] or ("last" in info and p["id"] > info["last"]) or not p["body"]:
            continue
        if p["heading"] != group_heading:
            flush()
            group_heading = p["heading"]
        for para in (piece for line in clean(p["body"]) for piece in _pieces(line)):
            size = sum(len(g[0]) + 1 for g in group)
            # a numbered entry («٣٩ - حديث: …») starts its own passage, so one saying and its grading stay together
            entry = _ENTRY.match(para) is not None
            if group and ((entry and size > 0) or size + len(para) > (MAX_ENTRY if _ENTRY.match(group[0][0]) else MAX_CHUNK)):
                flush()
            group.append((para, (p["part"], p["page"])))
    flush()
    if len(passages) < min_passages:
        raise ImportError_(f"only {len(passages)} passages from «{info['name']}»: the export looks incomplete")
    source = {"id": source_id(book), "name": info["name"], "about": info["about"] + NOTE,
              "url": f"https://shamela.ws/book/{book}"}
    return source, passages


def available(folder: Path) -> list[int]:
    """The chosen books whose three exported files are all in the folder."""
    return [b for b in BOOKS if all((folder / f"{b}{s}").exists() for s in (".db", ".page.jsonl", ".title.jsonl"))]


def add_to_corpus(corpus: dict, folder: Path) -> dict:
    """Add every chosen book found in `folder` to a corpus."""
    books = available(folder)
    for book in books:
        source, passages = build(folder, book)
        corpus["sources"].append(source)
        corpus["passages"].extend(passages)
    corpus.setdefault("_provenance", {})["shamela_books"] = books
    return corpus
