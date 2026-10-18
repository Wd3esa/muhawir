"""Books copied out of a Shamela 4 library: only the author's text, placed by heading, volume and page.
Synthetic files only: the «book» below is made-up test text."""
import json
import sqlite3

import pytest

from muhawir import shamela

BOOK = 99999


def export(folder):
    con = sqlite3.connect(folder / f"{BOOK}.db")
    con.execute("CREATE TABLE page (id INTEGER PRIMARY KEY, part TEXT, page INTEGER, number INTEGER, services TEXT)")
    con.execute("CREATE TABLE title (id INTEGER PRIMARY KEY, page INTEGER, parent INTEGER)")
    con.executemany("INSERT INTO page VALUES (?, ?, ?, NULL, NULL)",
                    [(1, "المقدمة", 1), (2, "1", 5), (3, "1", 6), (4, "2", 1)])
    con.executemany("INSERT INTO title VALUES (?, ?, ?)", [(1, 1, 0), (2, 2, 0), (3, 4, 2)])
    con.commit(); con.close()
    pages = [
        {"id": f"{BOOK}-1", "foot": "مقدمة المحقق: كلام المحقق لا يُنقل."},
        {"id": f"{BOOK}-2", "body": "<span data-type='title' id=toc-2>حرف الهمزة</span>\r١ - حديث: النخلة تشرب الماء (¬١)، لا أصل له.\r٢ - حديث: القمر بدر، ضعيف «١».",
         "foot": "(¬١) حاشية المحقق."},
        {"id": f"{BOOK}-3", "body": "تتمة الكلام على الحديث الثاني."},
        {"id": f"{BOOK}-4", "body": "٣ - حديث: الجمل صبور، صحيح المعنى."},
    ]
    (folder / f"{BOOK}.page.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pages), encoding="utf-8")
    titles = [{"id": f"{BOOK}-1", "body": "مقدمة التحقيق"}, {"id": f"{BOOK}-2", "body": "حرف الهمزة"},
              {"id": f"{BOOK}-3", "body": "الهمزة مع الجيم"}]
    (folder / f"{BOOK}.title.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in titles), encoding="utf-8")


@pytest.fixture
def book(monkeypatch, tmp_path):
    monkeypatch.setitem(shamela.BOOKS, BOOK, {"kind": "hadith", "name": "كتاب تجريبي", "first": 2, "about": "مصطنع للاختبار. "})
    monkeypatch.setattr(shamela, "BOOKS", {BOOK: shamela.BOOKS[BOOK]})
    export(tmp_path)
    return tmp_path


def test_only_the_authors_text_each_entry_on_its_own_and_placed_by_volume_and_page(book):
    source, passages = shamela.build(book, BOOK, min_passages=1)
    texts = [p["text"] for p in passages]
    assert texts[0].startswith("١ - حديث: النخلة تشرب الماء، لا أصل له")  # footnote marker gone
    assert "المحقق" not in " ".join(texts)  # neither the editor's introduction nor his footnotes
    assert texts[1] == "٢ - حديث: القمر بدر، ضعيف.\nتتمة الكلام على الحديث الثاني."
    assert passages[1]["location"] == "حرف الهمزة (ج1 ص5–ج1 ص6)"
    assert passages[2]["location"] == "حرف الهمزة، الهمزة مع الجيم (ج2 ص1)"
    assert source["url"] == f"https://shamela.ws/book/{BOOK}" and passages[0]["id"] == f"h{BOOK}:1"
    assert all(p["kind"] == "hadith" for p in passages)


def test_too_few_passages_means_a_broken_export(book):
    with pytest.raises(shamela.ImportError_, match="incomplete"):
        shamela.build(book, BOOK)


def test_a_missing_file_is_named(book):
    (book / f"{BOOK}.title.jsonl").unlink()
    with pytest.raises(shamela.ImportError_, match="title.jsonl"):
        shamela.build(book, BOOK)
    assert shamela.available(book) == []


def test_a_very_long_paragraph_is_cut_at_sentence_ends():
    para = "جملة طويلة. " * 400
    pieces = shamela._pieces(para.strip())
    assert all(len(p) <= shamela.MAX_CHUNK for p in pieces) and all(p.endswith(".") for p in pieces[:-1])
    assert shamela._pieces("ب" * 5000)  # no spaces at all: still cut, never stuck
