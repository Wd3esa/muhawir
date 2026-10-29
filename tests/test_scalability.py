"""Bounded admission and optional index integrity; no model or network calls."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from muhawir.pipeline import AnswerStore
from muhawir import vectors


def test_answer_cache_uses_wal(tmp_path):
    store = AnswerStore(tmp_path / "answers.db", "test")
    assert store._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._db.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    store._db.close()


def test_bounded_queue_waits_without_charging_busy_requests(monkeypatch):
    from muhawir import server

    slot = threading.BoundedSemaphore(1)
    slot.acquire()
    monkeypatch.setattr(server, "_ask_slots", slot)
    monkeypatch.setattr(server, "_ask_waiters", threading.BoundedSemaphore(1))
    monkeypatch.setattr(server, "ASK_QUEUE_TIMEOUT", .02)
    monkeypatch.setattr(server, "_allow_question", lambda address: True)
    monkeypatch.setattr(server, "_ask_calls", __import__("collections").deque())
    with pytest.raises(HTTPException) as error:
        server.ask(server.Ask(question="اختبار"))
    assert error.value.status_code == 429
    assert not server._ask_calls
    assert server._ask_waiters.acquire(blocking=False)
    server._ask_waiters.release()
    slot.release()


def test_queue_serves_when_slot_opens(monkeypatch):
    from muhawir import server

    slot = threading.BoundedSemaphore(1)
    slot.acquire()
    monkeypatch.setattr(server, "_ask_slots", slot)
    monkeypatch.setattr(server, "_ask_waiters", threading.BoundedSemaphore(1))
    monkeypatch.setattr(server, "ASK_QUEUE_TIMEOUT", 2)
    monkeypatch.setattr(server, "_allow_question", lambda address: True)
    monkeypatch.setattr(server, "_run_ask", lambda body: {"ok": body.question})
    timer = threading.Timer(.02, slot.release)
    timer.start()
    try:
        assert server.ask(server.Ask(question="مرحبا")) == {"ok": "مرحبا"}
    finally:
        timer.join()


def test_ann_manifest_invalidates_changed_vectors(tmp_path, monkeypatch):
    folder = tmp_path
    np.save(folder / "vectors.npy", np.eye(2, dtype=np.float16))
    (folder / "vectors_ids.json").write_text(json.dumps(["a", "b"]))
    (folder / "vectors_model.txt").write_text("local:fake/model@abc")
    monkeypatch.setenv("MUHAWIR_VECTOR_INDEX", "usearch")
    # Missing index must retain exact vector search rather than dropping to BM25.
    index = vectors.load_vectors(folder)
    assert index is not None and index.ann is None
    (folder / "vectors.usearch.json").write_text(json.dumps({"sha256": "bad", "count": 2, "dimensions": 2}))
    assert vectors.load_vectors(folder).ann is None


def test_ann_query_failure_falls_back_to_exact(monkeypatch):
    index = vectors.VectorIndex(["a", "b"], np.eye(2, dtype=np.float16), "local:fake/model@abc",
                                lambda question: [1.0, 0.0], ann=SimpleNamespace(search=lambda *args, **kwargs: []))
    hybrid = vectors.HybridRetriever(SimpleNamespace(search=lambda *args: []), index)
    assert hybrid._vector_ids("a", 1) == ["a"]
    assert index.ann is None


def test_ann_build_and_reopen_existing_vectors(tmp_path, monkeypatch):
    pytest.importorskip("usearch")
    np.save(tmp_path / "vectors.npy", np.eye(4, dtype=np.float16))
    (tmp_path / "vectors_ids.json").write_text(json.dumps(["a", "b", "c", "d"]))
    (tmp_path / "vectors_model.txt").write_text("local:fake/model@abc")
    monkeypatch.delenv("MUHAWIR_VECTOR_INDEX", raising=False)
    assert vectors.build_ann(tmp_path) == 4
    monkeypatch.setenv("MUHAWIR_VECTOR_INDEX", "usearch")
    loaded = vectors.load_vectors(tmp_path)
    assert loaded is not None and loaded.ann is not None
    assert loaded.ann.expansion_search == 1024
    assert loaded.ann.connectivity == 32
    for key in range(4):
        assert np.array_equal(np.asarray(loaded.ann.get(key)).reshape(-1), np.eye(4)[key])
    assert np.array_equal(np.load(tmp_path / "vectors.npy"), np.eye(4))
    assert json.loads((tmp_path / "vectors_ids.json").read_text()) == ["a", "b", "c", "d"]
    loaded.query_embedder = lambda question: [1., 0., 0., 0.]
    hybrid = vectors.HybridRetriever(SimpleNamespace(search=lambda *args: []), loaded)
    assert hybrid._vector_ids("a", 2)[0] == "a"
    # The same-size source matrix must still invalidate the index if any value changes.
    np.save(tmp_path / "vectors.npy", np.ones((4, 4), dtype=np.float16))
    assert vectors.load_vectors(tmp_path).ann is None


def test_ann_restore_sets_search_expansion_instead_of_library_default(tmp_path, monkeypatch):
    import sys

    index = vectors.VectorIndex(["a", "b"], np.eye(2), "local:fake/model@abc")
    restored = SimpleNamespace(ndim=2, expansion_search=64)
    class FakeIndex:
        def __len__(self):
            return 2
        ndim = restored.ndim
        expansion_search = restored.expansion_search
        @staticmethod
        def restore(path, view):
            assert view is True
            return reopened
    reopened = FakeIndex()
    monkeypatch.setitem(sys.modules, "usearch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "usearch.index", SimpleNamespace(Index=FakeIndex))
    monkeypatch.setattr(vectors, "_index_fingerprint", lambda *args: "matching")
    (tmp_path / "vectors.usearch.json").write_text(json.dumps({
        "sha256": "matching", "count": 2, "dimensions": 2,
    }))
    assert vectors._load_ann(tmp_path, index) is reopened
    assert reopened.expansion_search == 1024
