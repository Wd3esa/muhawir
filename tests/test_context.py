"""«اقرأ ما قبله وما بعده» (22 October 2026): the passages around a cited one, from the same book only, in the
book's order, from the JSON corpus and from the database alike. No model is called."""
from muhawir.corpus import parse_corpus
from muhawir.store import SqliteCorpus, build_db

DATA = {"synthetic": True, "sources": [{"id": "a", "name": "أ", "about": "أ"}, {"id": "b", "name": "ب", "about": "ب"}],
        "passages": [{"id": "a:1", "source_id": "a", "location": "1", "kind": "fiqh", "text": "الأول من الكتاب"},
                     {"id": "a:2", "source_id": "a", "location": "2", "kind": "fiqh", "text": "الثاني من الكتاب"},
                     {"id": "a:3", "source_id": "a", "location": "3", "kind": "fiqh", "text": "الثالث من الكتاب"},
                     {"id": "b:1", "source_id": "b", "location": "1", "kind": "fiqh", "text": "أول كتاب آخر"}]}


def _ids(pair):
    return tuple(p.id if p else None for p in pair)


def test_the_passages_around_one_come_from_its_own_book_in_order(tmp_path):
    build_db(DATA, tmp_path / "m.db")
    for corpus in (parse_corpus(DATA), SqliteCorpus(tmp_path / "m.db")):
        assert _ids(corpus.neighbours("a:2")) == ("a:1", "a:3")
        assert _ids(corpus.neighbours("a:3")) == ("a:2", None)  # the next passage is another book: not shown
        assert _ids(corpus.neighbours("b:1")) == (None, None)
        assert _ids(corpus.neighbours("x:9")) == (None, None)


def test_the_context_endpoint_reads_the_engines_corpus_and_refuses_unknown_ids(monkeypatch):
    from fastapi.testclient import TestClient
    from muhawir import server
    monkeypatch.setattr(server.engine, "corpus", parse_corpus(DATA))
    client = TestClient(server.app)
    r = client.get("/api/context/a:2").json()
    assert r["before"]["passage_id"] == "a:1" and r["after"]["text"] == "الثالث من الكتاب"
    assert client.get("/api/context/x:9").status_code == 404
