import importlib.util
import json
import sqlite3
from pathlib import Path

from muhawir.store import SCHEMA

_spec = importlib.util.spec_from_file_location("merge_books", Path(__file__).resolve().parent.parent / "scripts" / "merge_books.py")
merge_books = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_books)


def _db(path, sources, provenance=None):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.execute("INSERT INTO meta VALUES ('provenance', ?)", (json.dumps(provenance or {}),))
    for sid, count in sources:
        con.execute("INSERT INTO sources VALUES (?, ?, 'a', 'https://x')", (sid, sid))
        for n in range(1, count + 1):
            con.execute("INSERT INTO passages (id, source_id, location, kind, grade, text, keywords, tokens) "
                        "VALUES (?, ?, 'l', 'other', '', ?, '', ?)", (f"{sid}:{n}", sid, f"نص {sid} {n}", f"نص {sid} {n}"))
    con.commit()
    con.close()


def test_candidate_only_books_are_appended_and_live_order_is_kept(tmp_path):
    _db(tmp_path / "live.db", [("a", 3), ("b", 2)])
    prov = {"extra_sources": {"sha256": "h", "sources": [{"id": "c", "edition": "e", "rights_holder": "r",
                                                          "rights_basis": "permission", "rights_document": "d"}]}}
    _db(tmp_path / "cand.db", [("a", 3), ("c", 4)], prov)
    kept, added, sources = merge_books.merge(tmp_path / "live.db", tmp_path / "cand.db", tmp_path / "out.db")
    assert (kept, added, sources) == (5, 4, ["c"])
    con = sqlite3.connect(tmp_path / "out.db")
    ids = [r[0] for r in con.execute("SELECT id FROM passages ORDER BY rid")]
    assert ids[:5] == ["a:1", "a:2", "a:3", "b:1", "b:2"] and ids[5:] == ["c:1", "c:2", "c:3", "c:4"]
    assert con.execute("SELECT count(*) FROM passages_fts WHERE passages_fts MATCH 'c'").fetchone()[0] == 4
    assert json.loads(con.execute("SELECT value FROM meta WHERE key='provenance'").fetchone()[0])["extra_sources_added"]["sources"][0]["id"] == "c"


def test_inputs_are_not_changed_and_existing_output_is_refused(tmp_path):
    _db(tmp_path / "live.db", [("a", 1)])
    _db(tmp_path / "cand.db", [("a", 1), ("c", 1)])
    before = (tmp_path / "live.db").read_bytes()
    merge_books.merge(tmp_path / "live.db", tmp_path / "cand.db", tmp_path / "out.db")
    assert (tmp_path / "live.db").read_bytes() == before
    try:
        merge_books.merge(tmp_path / "live.db", tmp_path / "cand.db", tmp_path / "out.db")
    except SystemExit:
        return
    raise AssertionError("an existing output must be refused")
