"""The Shamela importer: a tiny dump built here, checked through the manifest validator Muhawir uses."""
import importlib.util
import json
import sqlite3
from pathlib import Path

from muhawir import extra_sources

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "shamela_to_manifest.py"
spec = importlib.util.spec_from_file_location("shamela_to_manifest", SCRIPT)
imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imp)


def make_dump(folder: Path, book: int = 2348):
    con = sqlite3.connect(folder / f"{book}.db")
    con.executescript("CREATE TABLE page (id INTEGER PRIMARY KEY, part TEXT, page INTEGER, number INTEGER, services TEXT);"
                      "CREATE TABLE title (id INTEGER PRIMARY KEY, page INTEGER, parent INTEGER);")
    con.executemany("INSERT INTO page VALUES (?,?,?,?,NULL)", [(1, None, None, None), (2, "1", 10, 1), (3, "1", 11, 2)])
    con.executemany("INSERT INTO title VALUES (?,?,?)", [(1, 2, 0), (2, 3, 1)])
    con.commit(); con.close()
    pages = [{"id": f"{book}-1", "body": "مقدمة الناشر"},
             {"id": f"{book}-2", "body": "١ - عن عمر ﵁ قال: «إنما الأعمال بالنيات». (¬١)", "foot": "(¬١) أخرجه: البخاري (١)."},
             {"id": f"{book}-3", "body": "<span data-type=\"title\">باب</span>\rنص ثان طويل. " + "كلمة " * 400}]
    (folder / f"{book}.page.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pages), encoding="utf-8")
    titles = [{"id": f"{book}-1", "body": "كتاب الإخلاص"}, {"id": f"{book}-2", "body": "باب النية"}]
    (folder / f"{book}.title.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in titles), encoding="utf-8")
    master = sqlite3.connect(folder / "master.db")
    master.executescript("CREATE TABLE book (book_id INTEGER, book_name TEXT, main_author INTEGER);"
                         "CREATE TABLE author (author_id INTEGER, author_name TEXT, death_number INTEGER);"
                         "INSERT INTO book VALUES (2348, 'رياض الصالحين', 7); INSERT INTO author VALUES (7, 'النووي', 676);")
    master.commit(); master.close()


def test_a_dump_becomes_a_manifest_the_loader_accepts(tmp_path, monkeypatch):
    make_dump(tmp_path)
    out = tmp_path / "approved-extra-sources.json"
    monkeypatch.setattr("sys.argv", ["x", str(tmp_path), str(tmp_path / "master.db"), str(out), "--books", "2348",
                                     "--holder", "h", "--document", "d"])
    assert imp.main() == 0
    corpus = {"sources": [], "passages": [], "_provenance": {}}
    count = extra_sources.append_reviewed(corpus, tmp_path)
    assert count == len(corpus["passages"]) > 2
    first = corpus["passages"][0]
    assert "مقدمة الناشر" in first["text"]
    note = next(p for p in corpus["passages"] if "الحاشية: أخرجه: البخاري" in p["text"])
    assert "(¬" not in note["text"] and "ج " not in note["location"] and "كتاب الإخلاص" in note["location"] and "ص ١٠" in note["location"]
    assert all(len(p["text"]) <= 1300 for p in corpus["passages"])
    assert "<span" not in json.dumps(corpus["passages"], ensure_ascii=False)
    assert corpus["sources"][0]["name"] == "رياض الصالحين" and "النووي" in corpus["sources"][0]["about"]


def test_where_names_the_volume_and_a_run_of_pages():
    assert imp._where(("1", 10), ("1", 10)) == "ص ١٠"
    assert imp._where(("3", 45), ("3", 46)) == "ج ٣، ص ٤٥–٤٦"
    assert imp._where((None, None), (None, None)) == ""
