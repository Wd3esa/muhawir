"""Optional vector search, fused with the existing SQLite BM25 retriever.

Two embedding backends: a local open model (BAAI/bge-m3, the default, run on the server itself
with sentence-transformers, so no quota and no data leaves the server) or the Gemini API.
The index is deliberately local and disposable. Building checkpoints each completed batch so an
interrupted build can continue without re-embedding the completed passages.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import quote

import httpx
import numpy as np

from .normalize import tokenize
from .retrieve import Hit

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_DB = DATA_DIR / "muhawir.db"
DIMENSIONS = 768  # Gemini embeddings
LOCAL_PREFIX = "local:"  # index model names of the local backend: "local:<model>@<revision>"
LOCAL_MODEL = os.environ.get("MUHAWIR_LOCAL_EMBED_MODEL", "BAAI/bge-m3")
LOCAL_MAX_TOKENS = int(os.environ.get("MUHAWIR_EMBED_MAX_TOKENS", "512"))  # longer passages are cut
RRF_K = 60
QUERY_CACHE_SIZE = 256
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
PREFERRED_MODELS = (
    "gemini-embedding-2",
    "gemini-embedding-2-preview",
    "gemini-embedding-001",
)


class GeminiEmbeddingClient:
    """Small REST client that uses only ``GEMINI_API_KEY`` from the environment."""

    def __init__(self, *, rpm: int | None = None, timeout: float = 90.0) -> None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings")
        self._client = httpx.Client(
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            timeout=timeout,
        )
        self.api_calls = 0
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._last_request: float | None = None

    def close(self) -> None:
        self._client.close()

    def _pace(self) -> None:
        if not self.interval or self._last_request is None:
            return
        wait = self.interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def _request(self, method: str, url: str, **kwargs) -> dict:
        for attempt in range(8):
            self._pace()
            self.api_calls += 1
            response = self._client.request(method, url, **kwargs)
            self._last_request = time.monotonic()
            if response.status_code == 429:
                if attempt == 7:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = max(0.0, float(retry_after))
                except ValueError:
                    try:
                        delay = max(0.0, parsedate_to_datetime(retry_after).timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError):
                        delay = min(60.0, 2.0 ** attempt)
                # A 429 always waits before retrying, even if the service omitted Retry-After.
                time.sleep(max(delay, min(self.interval, 1.0)))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("Gemini request retries exhausted")

    def list_embedding_models(self) -> list[dict]:
        """List all API models that advertise the embedContent method."""
        models: list[dict] = []
        page_token: str | None = None
        while True:
            params = {"pageSize": 100}
            if page_token:
                params["pageToken"] = page_token
            result = self._request("GET", f"{API_ROOT}/models", params=params)
            for model in result.get("models", []):
                actions = model.get("supportedGenerationMethods") or model.get("supported_generation_methods") or []
                if "embedContent" in actions:
                    models.append(model)
            page_token = result.get("nextPageToken")
            if not page_token:
                return models

    def _embed(self, model: str, texts: Sequence[str], *, query: bool) -> list[list[float]]:
        model_id = model.removeprefix("models/")
        requests = []
        for text in texts:
            if model_id == "gemini-embedding-2" or model_id.startswith("gemini-embedding-2-"):
                prepared = f"task: search result | query: {text}" if query else text
                config = {"outputDimensionality": DIMENSIONS}
            else:
                prepared = text
                config = {
                    "outputDimensionality": DIMENSIONS,
                    "taskType": "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
                }
            requests.append({
                "model": f"models/{model_id}",
                "content": {"parts": [{"text": prepared}]},
                "embedContentConfig": config,
            })
        result = self._request(
            "POST",
            f"{API_ROOT}/models/{quote(model_id, safe='-')}:batchEmbedContents",
            json={"requests": requests},
        )
        embeddings = result.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise RuntimeError("Gemini returned an unexpected number of embeddings")
        return [item["values"] for item in embeddings]

    def embed_documents(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(model, texts, query=False)

    def embed_query(self, model: str, question: str) -> list[float]:
        return self._embed(model, [question], query=True)[0]


def _model_name(model: dict | str) -> str:
    name = model if isinstance(model, str) else model.get("name", "")
    return str(name).removeprefix("models/")


def select_embedding_model(models: Sequence[dict | str]) -> str:
    names = [_model_name(model) for model in models]
    for preferred in PREFERRED_MODELS:
        if preferred in names:
            return preferred
    for name in names:
        if "embedding" in name.casefold():
            return name
    if names:
        return names[0]
    raise RuntimeError("The Gemini API did not list an embedding model")


def is_local(model: str) -> bool:
    return model.startswith(LOCAL_PREFIX)


def dimensions_of(model: str) -> int | None:
    """Vector size of an index model: fixed for Gemini, read from the index for a local model."""
    return None if is_local(model) else DIMENSIONS


class LocalEmbedder:
    """An open embedding model run on this machine (sentence-transformers). Passages and questions
    go through the same model and revision, so their vectors can be compared. It runs on the CPU unless
    MUHAWIR_EMBED_DEVICE names another device (e.g. "cuda" for a one-time build on a free GPU notebook)."""

    def __init__(self, model: str = LOCAL_MODEL, revision: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # optional: requirements-vectors.txt
            raise RuntimeError("the local embedder needs: pip install -r requirements-vectors.txt") from exc
        self.model_id = model
        self.revision = revision or os.environ.get("MUHAWIR_LOCAL_EMBED_REVISION") or self._latest_revision(model)
        self.device = os.environ.get("MUHAWIR_EMBED_DEVICE", "cpu")
        self.model = SentenceTransformer(model, revision=self.revision, device=self.device)
        self.model.max_seq_length = LOCAL_MAX_TOKENS
        self.dimensions = int(self.embed(["بسم الله"]).shape[1])  # works across library versions
        self.name = f"{LOCAL_PREFIX}{model}@{self.revision}"

    @staticmethod
    def _latest_revision(model: str) -> str:
        """The exact commit of the model on Hugging Face, recorded with the index."""
        try:
            from huggingface_hub import HfApi

            return HfApi().model_info(model).sha or "main"
        except Exception:  # offline: the cached copy is used, recorded as "main"
            return "main"

    @classmethod
    def from_name(cls, name: str) -> "LocalEmbedder":
        model, _, revision = name[len(LOCAL_PREFIX):].partition("@")
        return cls(model, revision or None)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), batch_size=16 if self.device == "cpu" else 64, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, _model: str, question: str) -> np.ndarray:
        return self.embed([question])[0]


@dataclass
class VectorIndex:
    ids: list[str]
    matrix: np.ndarray
    model_name: str
    query_embedder: Callable[[str], Sequence[float]] | None = None


class _LazyQueryEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.client: GeminiEmbeddingClient | LocalEmbedder | None = None

    def __call__(self, question: str) -> Sequence[float]:
        if self.client is None:
            self.client = LocalEmbedder.from_name(self.model_name) if is_local(self.model_name) else GeminiEmbeddingClient()
        return self.client.embed_query(self.model_name, question)


def load_vectors(data_dir: str | Path = DATA_DIR) -> VectorIndex | None:
    """Load a complete local index, returning ``None`` for an absent or invalid index."""
    folder = Path(data_dir)
    vector_path = folder / "vectors.npy"
    ids_path = folder / "vectors_ids.json"
    model_path = folder / "vectors_model.txt"
    if not (vector_path.is_file() and ids_path.is_file() and model_path.is_file()):
        return None
    try:
        matrix = np.load(vector_path, mmap_mode="r", allow_pickle=False)
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        model_name = model_path.read_text(encoding="utf-8").strip()
        expected = dimensions_of(model_name) or (matrix.shape[1] if matrix.ndim == 2 else 0)
        if matrix.ndim != 2 or matrix.shape != (len(ids), expected) or expected < 1 or not model_name:
            return None
        if not all(isinstance(item, str) for item in ids):
            return None
        return VectorIndex(ids, matrix, model_name, _LazyQueryEmbedder(model_name))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


class HybridRetriever:
    """RRF fusion over SQLite BM25 and cosine-ranked local Gemini embeddings.

    On any vector setup or query error, this instance disables vector search and returns the
    SQLite retriever's original result list without changing its ordering or Hit values.
    """

    def __init__(self, sqlite_retriever, vectors: VectorIndex | None) -> None:
        self.sqlite_retriever = sqlite_retriever
        self.vectors = vectors
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._warned = False
        self._disabled = False
        self._warn_reason: str | None = None
        if vectors is None:
            self._disable("vector index is missing")
        elif not is_local(vectors.model_name) and not os.environ.get("GEMINI_API_KEY"):
            self._disable("GEMINI_API_KEY is missing")
        elif vectors.matrix.ndim != 2 or vectors.matrix.shape[0] != len(vectors.ids) or vectors.matrix.shape[1] != (
                dimensions_of(vectors.model_name) or vectors.matrix.shape[1]):
            self._disable("vector index shape is invalid")

    def warm_up(self) -> None:
        """Load the query model now (a local model takes seconds to load), so the first question is not slow."""
        if self._disabled:
            return
        try:
            self._query_vector("ما أركان الإسلام")
        except Exception as exc:
            self._disable(f"query model could not be loaded ({type(exc).__name__})")

    def _disable(self, reason: str) -> None:
        self._disabled = True
        self._warn_reason = reason
        self._warn_once(reason)

    def _warn_once(self, reason: str) -> None:
        if not self._warned:
            log.warning("Vector search unavailable (%s); using BM25 only.", reason)
            self._warned = True

    def _query_vector(self, question: str) -> np.ndarray:
        assert self.vectors is not None
        if question in self._query_cache:
            self._query_cache.move_to_end(question)
            return self._query_cache[question]
        embedder = self.vectors.query_embedder
        if embedder is None:
            embedder = _LazyQueryEmbedder(self.vectors.model_name)
            self.vectors.query_embedder = embedder
        vector = np.asarray(embedder(question), dtype=np.float32).reshape(-1)
        if vector.shape != (self.vectors.matrix.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("query embedding has an invalid shape or value")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("query embedding is zero")
        vector /= norm
        self._query_cache[question] = vector
        if len(self._query_cache) > QUERY_CACHE_SIZE:
            self._query_cache.popitem(last=False)
        return vector

    def _vector_ids(self, question: str, limit: int) -> list[str]:
        assert self.vectors is not None
        query = self._query_vector(question)
        matrix = self.vectors.matrix
        similarities = np.empty(len(self.vectors.ids), dtype=np.float32)
        # Convert in small chunks to avoid keeping a second full float32 copy of the index.
        for start in range(0, len(self.vectors.ids), 4096):
            end = min(start + 4096, len(self.vectors.ids))
            block = np.asarray(matrix[start:end], dtype=np.float32)
            norms = np.linalg.norm(block, axis=1)
            similarities[start:end] = np.divide(
                block @ query, norms, out=np.full(end - start, -1.0, dtype=np.float32), where=norms > 0
            )
        order = np.argsort(-similarities, kind="stable")[:limit]
        return [self.vectors.ids[int(i)] for i in order]

    def search(self, question: str, k: int = 3) -> list[Hit]:
        bm25_hits = self.sqlite_retriever.search(question, k)
        if self._disabled or k <= 0:
            return bm25_hits
        try:
            vector_ids = self._vector_ids(question, max(k * 4, 20))
            weights: dict[str, float] = {}
            bm25_ranks: dict[str, int] = {}
            vector_ranks: dict[str, int] = {}
            hits_by_id: dict[str, Hit] = {}
            for rank, hit in enumerate(bm25_hits, 1):
                pid = hit.passage.id
                bm25_ranks[pid] = rank
                hits_by_id[pid] = hit
                weights[pid] = weights.get(pid, 0.0) + 1.0 / (RRF_K + rank)
            for rank, pid in enumerate(vector_ids, 1):
                vector_ranks[pid] = rank
                weights[pid] = weights.get(pid, 0.0) + 1.0 / (RRF_K + rank)
                if pid not in hits_by_id:
                    passage = self.sqlite_retriever.corpus.passage(pid)
                    if passage is not None:
                        # Keep the existing lexical coverage gate for downstream retrieval. Strict
                        # mode also rejects this zero BM25 score, so vector similarity alone does
                        # not establish that a passage supports an answer.
                        terms = list(dict.fromkeys(tokenize(question)))
                        passage_terms = set(tokenize(f"{passage.text} {passage.keywords}"))
                        coverage = sum(term in passage_terms for term in terms) / len(terms) if terms else 0.0
                        hits_by_id[pid] = Hit(passage, 0.0, coverage)
            order = sorted(
                (pid for pid in weights if pid in hits_by_id),
                key=lambda pid: (
                    -weights[pid],
                    bm25_ranks.get(pid, float("inf")),
                    vector_ranks.get(pid, float("inf")),
                    pid,
                ),
            )
            return [hits_by_id[pid] for pid in order[:k]]
        except Exception as exc:  # vector search is optional; keep the BM25 path available
            self._disabled = True
            self._warn_once(f"query embedding/search failed ({type(exc).__name__})")
            return bm25_hits


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_npy_atomic(path: Path, matrix: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, matrix, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _passages(db_path: Path) -> tuple[list[str], list[str]]:
    import sqlite3

    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT p.id, p.text, p.location, s.name FROM passages p "
            "JOIN sources s ON s.id = p.source_id ORDER BY p.rid"
        ).fetchall()
    ids = [row[0] for row in rows]
    documents = [f"title: {row[3]} — {row[2]} | text: {row[1]}" for row in rows]
    return ids, documents


def _normalize_rows(values: Sequence[Sequence[float]], dims: int = DIMENSIONS) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dims or not np.isfinite(matrix).all():
        raise ValueError("embedding batch has an invalid shape or value")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding batch contains a zero vector")
    return (matrix / norms).astype(np.float16)


def _read_json_ids(path: Path) -> list[str] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, list) and all(isinstance(x, str) for x in value) else None


def _start_checkpoint(folder: Path, ids: list[str], model: str,
                      dims: int = DIMENSIONS) -> tuple[Path, Path, Path, int]:
    raw_path = folder / "vectors.partial.f16"
    checkpoint_ids = folder / "vectors.partial_ids.json"
    checkpoint_model = folder / "vectors.partial_model.txt"
    saved_ids = _read_json_ids(checkpoint_ids) if checkpoint_model.exists() and checkpoint_model.read_text(encoding="utf-8").strip() == model else None
    expected_bytes_per_row = dims * np.dtype(np.float16).itemsize

    if saved_ids is None or ids[:len(saved_ids)] != saved_ids or len(saved_ids) > len(ids):
        # A completed final index is also a valid prefix if the source database has grown.
        try:
            final_ids = _read_json_ids(folder / "vectors_ids.json")
            final_model = (folder / "vectors_model.txt").read_text(encoding="utf-8").strip()
            final_matrix = np.load(folder / "vectors.npy", mmap_mode="r", allow_pickle=False)
            if (final_model != model or final_ids is None or ids[:len(final_ids)] != final_ids
                    or final_matrix.shape != (len(final_ids), dims)):
                raise ValueError("stale index")
            with raw_path.open("wb") as stream:
                np.asarray(final_matrix, dtype=np.float16).tofile(stream)
            saved_ids = final_ids
            _atomic_write_text(checkpoint_ids, json.dumps(saved_ids, ensure_ascii=False))
            _atomic_write_text(checkpoint_model, model + "\n")
        except (OSError, ValueError, TypeError):
            saved_ids = []
            raw_path.write_bytes(b"")
            _atomic_write_text(checkpoint_ids, "[]")
            _atomic_write_text(checkpoint_model, model + "\n")

    if not raw_path.exists():
        raw_path.write_bytes(b"")
    expected_size = len(saved_ids) * expected_bytes_per_row
    actual_size = raw_path.stat().st_size if raw_path.exists() else 0
    if actual_size < expected_size:
        saved_ids = []
        raw_path.write_bytes(b"")
        _atomic_write_text(checkpoint_ids, "[]")
        _atomic_write_text(checkpoint_model, model + "\n")
    else:
        # Remove bytes written by a process that stopped before atomically updating its ID list.
        with raw_path.open("r+b") as stream:
            stream.truncate(expected_size)
    return raw_path, checkpoint_ids, checkpoint_model, len(saved_ids)


def _embed_corpus(folder: Path, corpus_ids: list[str], documents: list[str], model: str, dims: int,
                  batch_size: int, embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
                  progress: Callable[[int, int], None] | None = None) -> int:
    """Embed every passage in batches with checkpoint/resume, then atomically publish the float16
    index (vectors.npy, vectors_ids.json, vectors_model.txt). Returns where the build resumed."""
    raw_path, ids_path, model_path, resume_at = _start_checkpoint(folder, corpus_ids, model, dims)
    for start in range(resume_at, len(corpus_ids), batch_size):
        end = min(start + batch_size, len(corpus_ids))
        vectors = _normalize_rows(embed(documents[start:end]), dims)
        if len(vectors) != end - start:
            raise ValueError("embedding batch length does not match passage batch")
        with raw_path.open("r+b") as stream:
            stream.seek(start * dims * np.dtype(np.float16).itemsize)
            vectors.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_write_text(ids_path, json.dumps(corpus_ids[:end], ensure_ascii=False))
        _atomic_write_text(model_path, model + "\n")
        if progress:
            progress(end, len(corpus_ids))

    if corpus_ids:
        raw = np.fromfile(raw_path, dtype=np.float16)
        if raw.size != len(corpus_ids) * dims:
            raise ValueError("checkpoint size does not match the passage count")
        matrix = raw.reshape((len(corpus_ids), dims))
    else:
        matrix = np.empty((0, dims), dtype=np.float16)
    _write_npy_atomic(folder / "vectors.npy", matrix)
    _atomic_write_text(folder / "vectors_ids.json", json.dumps(corpus_ids, ensure_ascii=False))
    _atomic_write_text(folder / "vectors_model.txt", model + "\n")
    raw_path.unlink(missing_ok=True)
    ids_path.unlink(missing_ok=True)
    model_path.unlink(missing_ok=True)
    return resume_at


def build_local_vectors(embedder: LocalEmbedder | None = None, *, db_path: str | Path | None = None,
                        data_dir: str | Path = DATA_DIR, batch_size: int = 32) -> dict:
    """Embed the corpus with the local open model (no API, no quota), with checkpoint/resume."""
    started = time.monotonic()
    embedder = embedder or LocalEmbedder()
    print(f"Using local embedding model: {embedder.name} ({embedder.dimensions} dimensions, "
          f"up to {LOCAL_MAX_TOKENS} tokens per passage)")
    folder = Path(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    corpus_ids, documents = _passages(Path(db_path or os.environ.get("MUHAWIR_DB") or DEFAULT_DB))

    def progress(done: int, total: int) -> None:
        if done == total or done % (batch_size * 20) < batch_size:
            elapsed = time.monotonic() - started
            print(f"{done}/{total} passages, {elapsed / 60:.1f} min", flush=True)

    resume_at = _embed_corpus(folder, corpus_ids, documents, embedder.name, embedder.dimensions,
                              batch_size, embedder.embed, progress)
    return {
        "model": embedder.name,
        "passages": len(corpus_ids),
        "resumed": resume_at,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "api_calls": 0,
        "vector_bytes": (folder / "vectors.npy").stat().st_size,
    }


def build_vectors(
    api: GeminiEmbeddingClient | None = None,
    *,
    db_path: str | Path | None = None,
    data_dir: str | Path = DATA_DIR,
    batch_size: int = 64,
    rate_limit_rpm: int | None = None,
) -> dict:
    """Embed the SQLite corpus with checkpoint/resume and atomically publish a float16 index."""
    if batch_size < 1 or batch_size > 100:
        raise ValueError("batch_size must be between 1 and 100")
    own_api = api is None
    if api is None:
        rpm = rate_limit_rpm
        if rpm is None:
            rpm = int(os.environ.get("MUHAWIR_EMBED_RPM", "60"))
        api = GeminiEmbeddingClient(rpm=rpm)
    started = time.monotonic()
    folder = Path(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        models = api.list_embedding_models()
        model_names = [_model_name(item) for item in models]
        if not model_names:
            raise RuntimeError("Gemini returned no models supporting embedContent")
        print("Available embedding models: " + ", ".join(model_names))
        model = select_embedding_model(models)
        print(f"Using embedding model: {model}")

        corpus_ids, documents = _passages(Path(db_path or os.environ.get("MUHAWIR_DB") or DEFAULT_DB))
        resume_at = _embed_corpus(folder, corpus_ids, documents, model, DIMENSIONS, batch_size,
                                  lambda texts: api.embed_documents(model, texts))
        return {
            "model": model,
            "passages": len(corpus_ids),
            "resumed": resume_at,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "api_calls": getattr(api, "api_calls", None),
            "vector_bytes": (folder / "vectors.npy").stat().st_size,
        }
    finally:
        if own_api:
            api.close()


COMPARE_QUESTIONS = (
    "ما حكم قراءة البسملة في الصلاة",
    "أنواع الزكاة",
    "هل يشترط الحول في زكاة المال",
    "من خلق الله",
    "ما هي أركان الإسلام",
    "لماذا نصوم",
)


def _print_results(label: str, hits: Sequence[Hit]) -> None:
    print(label)
    for position, hit in enumerate(hits, 1):
        p = hit.passage
        where = getattr(p, "location", "") or ""
        text = " ".join((getattr(p, "text", "") or "").split()[:12])
        print(f"  {position}. {p.id} | {where} | {text}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build and inspect optional Muhawir vector search")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="embed passages and save a resumable local index")
    build_parser.add_argument("--backend", choices=("local", "gemini"),
                              default=os.environ.get("MUHAWIR_EMBEDDER", "local"),
                              help="local: open model on this machine (default); gemini: Gemini API")
    test_parser = commands.add_parser("test", help="compare BM25 and hybrid results (no language model used)")
    test_parser.add_argument("question", nargs="?", help="one question; default: a fixed set of problem questions")
    args = parser.parse_args(argv)
    if args.command == "build":
        try:
            batch = int(os.environ.get("MUHAWIR_EMBED_BATCH", "32"))  # more passages per step on a GPU
            report = build_local_vectors(batch_size=batch) if args.backend == "local" else build_vectors()
        except Exception as exc:
            print(f"Build failed: {exc}", file=sys.stderr)
            return 2
        print(
            f"Built {report['passages']} passage vectors in {report['elapsed_seconds']}s; "
            f"API calls: {report['api_calls']}; file: {report['vector_bytes']} bytes"
        )
        return 0

    from .store import SqliteCorpus, SqliteRetriever

    db = Path(os.environ.get("MUHAWIR_DB") or DEFAULT_DB)
    retriever = SqliteRetriever(SqliteCorpus(db))
    hybrid = HybridRetriever(retriever, load_vectors())
    for question in ([args.question] if args.question else COMPARE_QUESTIONS):
        print(f"\n=== {question}")
        _print_results("BM25 (keywords only)", retriever.search(question, 5))
        _print_results("Hybrid (keywords + vectors)", hybrid.search(question, 5))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    raise SystemExit(main())
