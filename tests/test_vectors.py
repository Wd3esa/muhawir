"""Optional Gemini vector retrieval tests, using only local fakes."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from muhawir import server
from muhawir.retrieve import Hit
from muhawir.store import build_db
from muhawir.vectors import (
    DIMENSIONS,
    GeminiEmbeddingClient,
    HybridRetriever,
    VectorIndex,
    build_vectors,
    load_vectors,
    select_embedding_model,
)

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = ROOT / "data" / "synthetic_corpus.json"


class FakeSqliteRetriever:
    def __init__(self, hits, all_ids=(), passages=None):
        self.hits = hits
        available = set(all_ids) | {hit.passage.id for hit in hits}
        self.passages = passages or {}
        self.corpus = SimpleNamespace(
            passage=lambda pid: self.passages.get(pid) or (
                SimpleNamespace(id=pid, text="", keywords="") if pid in available else None
            )
        )

    def search(self, question, k=3):
        return self.hits[:k]


def hit(pid, score=1.2, coverage=0.8):
    return Hit(SimpleNamespace(id=pid), score, coverage)


def vector_index(ids, vectors, embedder=None):
    matrix = np.zeros((len(ids), DIMENSIONS), dtype=np.float16)
    for index, values in enumerate(vectors):
        matrix[index, :len(values)] = values
    if embedder is not None:
        original_embedder = embedder

        def embedder(question):
            values = np.asarray(original_embedder(question), dtype=np.float32).reshape(-1)
            if values.size < DIMENSIONS:
                padded = np.zeros(DIMENSIONS, dtype=np.float32)
                padded[:values.size] = values
                return padded
            return values

    return VectorIndex(ids, matrix, "gemini-embedding-2", embedder)


def test_select_embedding_model_prefers_current_stable_model():
    models = [
        {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/gemini-embedding-2", "supportedGenerationMethods": ["embedContent"]},
    ]
    assert select_embedding_model(models) == "gemini-embedding-2"


def test_gemini_client_waits_and_retries_after_rate_limit(monkeypatch):
    class Response:
        def __init__(self, status, payload):
            self.status_code = status
            self.headers = {"Retry-After": "0.25"}
            self.payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("unexpected HTTP status")

        def json(self):
            return self.payload

    class HttpClient:
        def __init__(self):
            self.responses = [Response(429, {}), Response(200, {"models": []})]

        def request(self, *_args, **_kwargs):
            return self.responses.pop(0)

    client = GeminiEmbeddingClient.__new__(GeminiEmbeddingClient)
    client._client = HttpClient()
    client.api_calls = 0
    client.interval = 0.0
    client._last_request = None
    waits = []
    monkeypatch.setattr("muhawir.vectors.time.sleep", waits.append)

    assert client._request("GET", "https://example.invalid") == {"models": []}
    assert client.api_calls == 2
    assert waits == [0.25]


def test_rrf_fusion_promotes_a_passage_present_in_both_rankings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    base = FakeSqliteRetriever([hit("a"), hit("b")], ["a", "b", "c"])
    embed_calls = []

    def embed(_question):
        embed_calls.append(_question)
        return [1.0, 0.0]

    vectors = vector_index(["b", "c", "a"], [[1, 0], [0.9, 0.1], [0.7, 0.7]], embed)
    hybrid = HybridRetriever(base, vectors)

    first = hybrid.search("test question", 3)
    second = hybrid.search("test question", 3)

    assert [item.passage.id for item in first] == ["b", "a", "c"]
    assert [item.passage.id for item in second] == ["b", "a", "c"]
    assert first[0].score == base.hits[1].score
    assert embed_calls == ["test question"]


def test_vector_only_hit_keeps_the_existing_lexical_coverage_gate(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    passage = SimpleNamespace(id="semantic", text="zakat types", keywords="")
    base = FakeSqliteRetriever([hit("bm25")], ["semantic"], {"semantic": passage})
    vectors = vector_index(["semantic"], [[1, 0]], lambda _question: [1, 0])

    results = HybridRetriever(base, vectors).search("zakat types", 2)

    semantic_hit = next(item for item in results if item.passage.id == "semantic")
    assert semantic_hit.score == 0.0
    assert semantic_hit.coverage == 1.0


def test_missing_vectors_and_missing_key_return_unchanged_bm25(caplog, monkeypatch):
    base = FakeSqliteRetriever([hit("bm25-a"), hit("bm25-b")])
    expected = base.search("question", 2)

    missing_index = HybridRetriever(base, None)
    assert missing_index.search("question", 2) == expected
    assert missing_index.search("question", 2) == expected
    assert sum("Vector search unavailable" in record.message for record in caplog.records) == 1

    caplog.clear()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    never_called = lambda _question: pytest.fail("query embedding must not run without a key")
    missing_key = HybridRetriever(base, vector_index(["bm25-a"], [[1, 0]], never_called))
    assert missing_key.search("question", 2) == expected
    assert sum("Vector search unavailable" in record.message for record in caplog.records) == 1


def test_query_embedding_failure_warns_once_then_uses_bm25(caplog, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    base = FakeSqliteRetriever([hit("bm25-a")])
    broken = vector_index(["bm25-a"], [[1, 0]], lambda _question: (_ for _ in ()).throw(OSError("offline")))
    hybrid = HybridRetriever(base, broken)

    assert hybrid.search("question") == base.search("question")
    assert hybrid.search("question") == base.search("question")
    assert sum("Vector search unavailable" in record.message for record in caplog.records) == 1


class FakeEmbeddingAPI:
    def __init__(self, fail_on_embedding_call=None):
        self.api_calls = 0
        self.embedding_calls = 0
        self.fail_on_embedding_call = fail_on_embedding_call

    def list_embedding_models(self):
        self.api_calls += 1
        return [{"name": "models/gemini-embedding-2", "supportedGenerationMethods": ["embedContent"]}]

    def embed_documents(self, _model, texts):
        self.api_calls += 1
        self.embedding_calls += 1
        if self.embedding_calls == self.fail_on_embedding_call:
            raise OSError("simulated interruption")
        rows = []
        for offset, _text in enumerate(texts):
            row = np.zeros(DIMENSIONS, dtype=np.float32)
            row[(self.embedding_calls + offset) % DIMENSIONS] = 1.0
            rows.append(row)
        return rows


def test_build_vectors_resumes_completed_batches(tmp_path, capsys):
    corpus_data = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
    db_path = tmp_path / "muhawir.db"
    build_db(corpus_data, db_path)
    data_dir = tmp_path / "vector-data"

    interrupted = FakeEmbeddingAPI(fail_on_embedding_call=2)
    with pytest.raises(OSError, match="simulated interruption"):
        build_vectors(interrupted, db_path=db_path, data_dir=data_dir, batch_size=2)
    saved_ids = json.loads((data_dir / "vectors.partial_ids.json").read_text(encoding="utf-8"))
    assert len(saved_ids) == 2

    resumed_api = FakeEmbeddingAPI()
    result = build_vectors(resumed_api, db_path=db_path, data_dir=data_dir, batch_size=2)
    index = load_vectors(data_dir)

    assert result["model"] == "gemini-embedding-2"
    assert result["passages"] == 5
    assert result["resumed"] == 2
    assert result["api_calls"] == 3  # one model listing and two remaining batches
    assert index is not None
    assert index.matrix.shape == (5, DIMENSIONS)
    assert index.matrix.dtype == np.float16
    assert len(index.ids) == 5
    assert "Using embedding model: gemini-embedding-2" in capsys.readouterr().out
    assert not (data_dir / "vectors.partial.f16").exists()


def test_server_wraps_sqlite_retriever_only_when_vector_flag_is_enabled(tmp_path, monkeypatch):
    db_path = tmp_path / "muhawir.db"
    build_db(json.loads(SYNTHETIC.read_text(encoding="utf-8")), db_path)
    monkeypatch.setenv("MUHAWIR_DB", str(db_path))
    monkeypatch.setenv("MUHAWIR_VECTORS", "1")
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    engine = server.build()

    assert isinstance(engine.retriever, HybridRetriever)
    assert engine.retriever._disabled is True


class FakeLocalEmbedder:
    """Stands in for the local open model: 8 dimensions, a vector from the text's characters."""
    name = "local:fake/model@abc123"
    dimensions = 8

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def embed(self, texts):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise OSError("simulated interruption")
        rows = []
        for text in texts:
            row = np.zeros(self.dimensions, dtype=np.float32)
            for i, ch in enumerate(text):
                row[(ord(ch) + i) % self.dimensions] += 1.0
            rows.append(row)
        return np.asarray(rows)

    def embed_query(self, _model, question):
        return self.embed([question])[0]


def test_local_build_resumes_and_loads_with_its_own_vector_size(tmp_path, capsys):
    from muhawir.vectors import build_local_vectors

    db_path = tmp_path / "muhawir.db"
    build_db(json.loads(SYNTHETIC.read_text(encoding="utf-8")), db_path)
    data_dir = tmp_path / "vector-data"
    with pytest.raises(OSError, match="simulated interruption"):
        build_local_vectors(FakeLocalEmbedder(fail_after=1), db_path=db_path, data_dir=data_dir, batch_size=2)
    result = build_local_vectors(FakeLocalEmbedder(), db_path=db_path, data_dir=data_dir, batch_size=2)
    index = load_vectors(data_dir)

    assert result["resumed"] == 2 and result["passages"] == 5 and result["api_calls"] == 0
    assert index is not None and index.matrix.shape == (5, 8) and index.model_name == "local:fake/model@abc123"
    assert "Using local embedding model: local:fake/model@abc123" in capsys.readouterr().out


def test_a_local_index_needs_no_gemini_key_and_queries_the_same_model(monkeypatch):
    from muhawir import vectors

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    made = []

    def from_name(name):
        made.append(name)
        return FakeLocalEmbedder()

    monkeypatch.setattr(vectors.LocalEmbedder, "from_name", staticmethod(from_name))
    fake = FakeLocalEmbedder()
    ids = ["a", "b"]
    matrix = np.asarray(fake.embed(["ماذا تحتاج النخلة", "حيوان صبور"]), dtype=np.float16)
    index = VectorIndex(ids, matrix, "local:fake/model@abc123", None)
    retriever = HybridRetriever(FakeSqliteRetriever([hit("b")], all_ids=ids), index)

    assert retriever._disabled is False
    found = [h.passage.id for h in retriever.search("ماذا تحتاج النخلة", 2)]
    assert "a" in found and made == ["local:fake/model@abc123"]  # the question used the index's own model


def test_warm_up_loads_the_query_model_or_falls_back_to_keyword_search(monkeypatch):
    from muhawir import vectors

    def broken(_name):
        raise RuntimeError("no model")

    monkeypatch.setattr(vectors.LocalEmbedder, "from_name", staticmethod(broken))
    index = VectorIndex(["a"], np.ones((1, 8), dtype=np.float16), "local:fake/model@abc123", None)
    retriever = HybridRetriever(FakeSqliteRetriever([hit("a")], all_ids=["a"]), index)
    retriever.warm_up()
    assert retriever._disabled is True
    assert [h.passage.id for h in retriever.search("سؤال", 1)] == ["a"]  # keyword search still answers
