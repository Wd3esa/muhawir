"""Turn books dumped from the Shamela database into the append-only manifest that Muhawir reads
(approved-extra-sources.json), beside the other private source files.

Input: the folder of dumps, one set per book id: <id>.db (its pages and headings), <id>.page.jsonl (the text of every
page) and <id>.title.jsonl (the text of every heading), and Shamela's master.db for the names. Nothing is downloaded.

Passages: paragraphs in order under the same heading are joined up to MAX_CHUNK characters; a longer paragraph is cut
at sentence ends. The location is the heading, then the volume and page of the passage («ج 3، ص 45»). Only markup is
removed (tags, footnote marks); the wording is not changed. A book listed with footnotes keeps each page's notes as a
paragraph beginning «الحاشية:» right after the page (the hadith references of a book of hadith are in them).

The rights fields are written as given on the command line: this script cannot tell whose text a book is, and the
person who adds it answers for that.

    python shamela_to_manifest.py DUMPS master.db OUT.json --holder "..." --document "..."
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

MAX_CHUNK = 1200

# book id -> (source id, kind, keep the page footnotes)
BOOKS = {
    20855: ("extra-qurtubi", "tafsir", False),
    41: ("extra-baghawi", "tafsir", False),
    2348: ("extra-riyad-salihin", "other", True),
    1956: ("extra-adhkar-nawawi", "other", True),
    856: ("extra-kashf-khafa", "takhrij", True),
    6056: ("extra-asrar-marfua", "takhrij", True),
    8566: ("extra-manar-munif", "takhrij", True),
    197: ("extra-zad-maad", "other", False),
    23587: ("extra-irab-nahhas", "other", False),
    22928: ("extra-tibyan-irab", "other", False),
}

_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t ]+")
_MARK = re.compile(r"\s*\(¬\s*[0-9٠-٩]+\)")
_FOOT_LEAD = re.compile(r"^\s*\(¬\s*[0-9٠-٩]+\)\s*", re.M)
_SENTENCE_END = re.compile(r"(?<=[.!?؟؛])\s+")
_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def clean(text: str) -> str:
    text = _TAG.sub("", text or "")
    return text.replace("\r", "\n").replace("\u200f", "").replace("\u200e", "")


def paragraphs(text: str) -> list[str]:
    out = []
    for line in clean(text).split("\n"):
        line = _SPACES.sub(" ", line).strip()
        if line:
            out.append(line)
    return out


def split_long(text: str, limit: int = MAX_CHUNK) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > limit:  # a sentence with no stop in it: cut at a space
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > limit:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        parts.append(current)
    return [p for p in parts if p]


def _headings(db: Path, titles_file: Path, book: int) -> tuple[dict[int, list[int]], dict[int, tuple[str, int]]]:
    """Heading ids that start on each page, and each heading's text and parent."""
    text = {}
    for line in titles_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        text[int(row["id"].rsplit("-", 1)[1])] = _SPACES.sub(" ", clean(row.get("body", ""))).strip()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    by_page: dict[int, list[int]] = {}
    info: dict[int, tuple[str, int]] = {}
    for tid, page, parent in con.execute("SELECT id, page, parent FROM title ORDER BY id"):
        by_page.setdefault(page, []).append(tid)
        info[tid] = (text.get(tid, ""), parent)
    con.close()
    return by_page, info


def _trail(tid: int, info: dict[int, tuple[str, int]], depth: int = 3) -> str:
    names = []
    while tid and tid in info and len(names) < depth:
        names.append(info[tid][0])
        tid = info[tid][1]
    return " / ".join(n for n in reversed(names) if n)


def _where(first: tuple, last: tuple) -> str:
    """«ج 3، ص 45» for one page, «ج 3، ص 45–46» for a passage that runs over, and the second volume too when it changes."""
    def num(v):
        return str(v).translate(_DIGITS)

    (p1, a), (p2, b) = first, last
    if a is None:
        return ""
    volume = f"ج {num(p1)}، " if p1 not in (None, "", "1") else ""
    if b is None or (p1 == p2 and a == b):
        return f"{volume}ص {num(a)}"
    if p1 == p2:
        return f"{volume}ص {num(a)}–{num(b)}"
    return f"{volume}ص {num(a)} – ج {num(p2)}، ص {num(b)}"


def build_book(dumps: Path, master: sqlite3.Connection, book: int, sid: str, kind: str, feet: bool, args) -> dict:
    page_file = dumps / f"{book}.page.jsonl"
    by_page, info = _headings(dumps / f"{book}.db", dumps / f"{book}.title.jsonl", book)
    pages = sqlite3.connect(f"file:{dumps / f'{book}.db'}?mode=ro", uri=True)
    where = {pid: (part, page) for pid, part, page in pages.execute("SELECT id, part, page FROM page")}
    pages.close()
    meta = master.execute("SELECT b.book_name, a.author_name, a.death_number FROM book b "
                          "LEFT JOIN author a ON a.author_id = b.main_author WHERE b.book_id = ?", (book,)).fetchone()
    name, author, died = meta
    trail, passages = "", []
    group: list[str] = []
    group_first = group_last = (None, None)

    def flush():
        nonlocal group, group_first, group_last
        if group:
            place = " — ".join(x for x in (trail, _where(group_first, group_last)) if x)
            for part in group:
                passages.append({"location": place or name, "text": part, "kind": kind})
        group = []

    size = 0
    for line in page_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        pid = int(row["id"].rsplit("-", 1)[1])
        for tid in by_page.get(pid, ()):
            flush(); size = 0
            trail = _trail(tid, info)
        paras = [_MARK.sub("", p) for p in paragraphs(row.get("body", ""))]
        if feet and row.get("foot"):
            notes = [_FOOT_LEAD.sub("", p) for p in paragraphs(row["foot"])]
            if notes:
                paras.append("الحاشية: " + " ".join(notes))
        position = where.get(pid, (None, None))
        for para in paras:
            for piece in split_long(para):
                if group and size + len(piece) + 1 > MAX_CHUNK:
                    flush(); size = 0
                if not group:
                    group_first = position
                group_last = position
                if group and size:
                    group[-1] = group[-1] + "\n" + piece
                else:
                    group.append(piece)
                size += len(piece) + 1
    flush()
    return {"id": sid, "name": name, "about": f"{author} (ت {died} هـ). النص من قاعدة الشاملة ١٤٤٨.",
            "url": f"https://shamela.ws/book/{book}", "edition": name, "rights_holder": args.holder,
            "rights_basis": args.basis, "rights_document": args.document, "passages": passages}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dumps", type=Path)
    ap.add_argument("master", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--books", type=int, nargs="*", help="book ids (default: all known in BOOKS)")
    ap.add_argument("--holder", required=True)
    ap.add_argument("--basis", default="permission", choices=["permission", "license", "public-domain"])
    ap.add_argument("--document", required=True)
    args = ap.parse_args()
    master = sqlite3.connect(f"file:{args.master}?mode=ro", uri=True)
    ids = args.books or list(BOOKS)
    sources = []
    for book in ids:
        if book not in BOOKS:
            sys.exit(f"book {book} is not in BOOKS")
        sid, kind, feet = BOOKS[book]
        if not (args.dumps / f"{book}.page.jsonl").exists():
            print(f"skipped {book}: no dump", file=sys.stderr)
            continue
        record = build_book(args.dumps, master, book, sid, kind, feet, args)
        sources.append(record)
        print(f"{book} {record['name']}: {len(record['passages'])} passages")
    args.out.write_text(json.dumps({"schema_version": 1, "sources": sources}, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
