"""Query batching uses only synthetic data and local fake embeddings."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import numpy as np
import pytest

from muhawir import pipeline, vectors
from muhawir.generate import ExtractiveGenerator
from muhawir.store import SqliteCorpus, SqliteRetriever, build_db
from muhawir.vectors import HybridRetriever, VectorIndex, _LazyQueryEmbedder


class Client:
    def __init__(self):
        self.batches = []
        self.singles = []
        self.failure = None

    def embed(self, questions):
        self.batches.append(list(questions))
        if isinstance(self.failure, Exception):
            raise self.failure
        if self.failure is not None:
            return self.failure
        return np.array([self.row(q) for q in questions], dtype=np.float32)

    def embed_query(self, model, question):
        self.singles.append(question)
        return self.row(question)

    @staticmethod
    def row(q):
        return [len(q) + 1, 2, 3]


def make_retriever(sqlite=None):
    client = Client()
    embedder = _LazyQueryEmbedder("local:fake/model@revision")
    embedder.client = client
    index = VectorIndex(["a", "b", "c"], np.eye(3, dtype=np.float16),
                        embedder.model_name, embedder)
    return HybridRetriever(sqlite, index), client


def test_batch_skips_duplicates_and_cached_queries_and_preserves_vectors():
    retriever, client = make_retriever()
    cached = retriever._query_vector("cached").copy()
    retriever.prepare_queries(["cached", "first", "second", "first"])
    assert client.batches == [["first", "second"]]
    assert client.singles == ["cached"]
    assert np.array_equal(retriever._query_vector("cached"), cached)
    for q in ["first", "second"]:
        expected = np.array(client.row(q), dtype=np.float32)
        expected /= np.linalg.norm(expected)
        assert np.array_equal(retriever._query_vector(q), expected)
    retriever.prepare_queries(["first", "second"])
    assert len(client.batches) == 1
    assert client.singles == ["cached"]


@pytest.mark.parametrize("bad", [
    RuntimeError("private query text must not be logged"),
    np.ones((1, 3)), np.ones((2, 4)),
    [[1, 2, 3], [float("nan"), 2, 3]],
    [[1, 2, 3], [0, 0, 0]],
])
def test_bad_batch_does_not_publish_partial_vectors_or_disable_individual_path(bad, caplog):
    retriever, client = make_retriever()
    retriever._query_vector("cached")
    client.failure = bad
    retriever.prepare_queries(["private first", "private second"])
    assert list(retriever._query_cache) == ["cached"]
    assert not retriever._disabled
    assert "private" not in caplog.text
    assert np.isfinite(retriever._query_vector("private first")).all()
    assert client.singles == ["cached", "private first"]


def test_overlapping_preparations_share_cache_and_do_not_deadlock():
    retriever, client = make_retriever()
    meeting = threading.Barrier(4)
    def prepare(_):
        meeting.wait(timeout=3)
        retriever.prepare_queries(["first", "second"])
        return retriever._query_vector("first")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(prepare, range(4)))
    assert client.batches == [["first", "second"]]
    assert client.singles == []
    assert all(np.array_equal(rows[0], row) for row in rows)


def test_cache_remains_bounded_and_evicted_query_can_be_recomputed(monkeypatch):
    monkeypatch.setattr(vectors, "QUERY_CACHE_SIZE", 2)
    retriever, client = make_retriever()
    retriever._query_vector("old")
    retriever.prepare_queries(["new", "other"])
    assert list(retriever._query_cache) == ["new", "other"]
    retriever._query_vector("old")
    assert client.singles == ["old", "old"]
    assert len(retriever._query_cache) == 2
    retriever.prepare_queries(["one", "two", "three"])
    assert client.batches == [["new", "other"]]


def test_prefetch_never_initializes_remote_client(monkeypatch):
    retriever, _ = make_retriever()
    retriever.vectors.model_name = "gemini-embedding-2"
    retriever.vectors.query_embedder = _LazyQueryEmbedder("gemini-embedding-2")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr(vectors, "GeminiEmbeddingClient", lambda: pytest.fail("remote initialization"))
    retriever.prepare_queries(["first", "second"])
    assert not retriever._query_cache


def test_single_disabled_and_callable_only_paths_remain_lazy():
    retriever, client = make_retriever()
    retriever.prepare_queries([])
    retriever.prepare_queries(["single", "single"])
    retriever._disabled = True
    retriever.prepare_queries(["one", "two"])
    retriever._disabled = False
    retriever.vectors.query_embedder = lambda _: [1, 2, 3]
    retriever.prepare_queries(["one", "two"])
    assert client.batches == [] and client.singles == []
    assert not retriever._query_cache


def test_lazy_batch_and_single_use_one_local_model(monkeypatch):
    client = Client()
    loads = []
    def load(name):
        loads.append(name)
        return client
    monkeypatch.setattr(vectors.LocalEmbedder, "from_name", load)
    embedder = _LazyQueryEmbedder("local:fake/model@revision")
    embedder.embed_many(["one", "two"])
    embedder("three")
    assert loads == [embedder.model_name]
    assert client.batches == [["one", "two"]] and client.singles == ["three"]


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("ann", [False, True])
def test_gather_context_and_query_order_match_individual_path(tmp_path, monkeypatch, threads, ann):
    monkeypatch.setattr(pipeline, "SEARCH_THREADS", threads)
    text = "نص تجريبي طويل للاختبار فقط ولا يمثل مصدرا دينيا ولا يعتمد عليه في أي جواب"
    path = tmp_path / "test.db"
    build_db({"synthetic": True, "sources": [{"id": "s", "name": "اختبار", "about": "اختبار"}],
              "passages": [{"id": pid, "source_id": "s", "location": pid,
                            "kind": "other", "text": text + " " + pid}
                           for pid in ["a", "b", "c"]]}, path)
    corpus = SqliteCorpus(path)
    retriever, client = make_retriever(SqliteRetriever(corpus))
    if ann:
        class Ann:
            def search(self, query, count):
                from types import SimpleNamespace
                keys = np.argsort(-query, kind="stable")[:count]
                return [SimpleNamespace(key=int(key)) for key in keys]
        retriever.vectors.ann = Ann()
    # This baseline exposes the same embeddings but cannot prefetch them.
    baseline = HybridRetriever(SqliteRetriever(corpus), replace(
        retriever.vectors, query_embedder=lambda q: client.row(q)))
    queries = ["نص تجريبي", "للاختبار فقط", "نص تجريبي"]
    engine = pipeline.Muhawir(corpus, ExtractiveGenerator(), retriever)
    before = pipeline.Muhawir(corpus, ExtractiveGenerator(), baseline)._gather(
        "نص طويل للاختبار", "", queries)
    after = engine._gather("نص طويل للاختبار", "", queries)
    assert before[0] and before == after
    assert after[1] == queries + ["نص طويل للاختبار"]
    assert client.batches == [["نص تجريبي", "للاختبار فقط", "نص طويل للاختبار"]]
    assert client.singles == []
    assert not retriever._disabled
