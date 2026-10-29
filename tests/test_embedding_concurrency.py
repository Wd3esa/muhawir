"""Exercise the real embedder wrapper with a fake model, without downloads or Torch."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sys
import threading
import time

import numpy as np
import pytest

from muhawir.vectors import LocalEmbedder


def make_embedder(monkeypatch, device):
    class Model:
        def __init__(self, *args, **kwargs):
            self.active = self.peak = 0
            self.guard = threading.Lock()
            self.batches = []
            self.meeting = None
            self.fail = False

        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            assert kwargs["convert_to_numpy"] is True
            with self.guard:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.batches.append(kwargs["batch_size"])
            try:
                if self.fail:
                    raise ValueError("fake encoding failure")
                if self.meeting:
                    self.meeting.wait(timeout=2)
                else:
                    time.sleep(.02)  # yield so overlapping callers would enter an unguarded model
                return np.array([[len(text), 1.] for text in texts], dtype=np.float32)
            finally:
                with self.guard:
                    self.active -= 1

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    monkeypatch.setenv("MUHAWIR_EMBED_DEVICE", device)
    return LocalEmbedder("fake/model", "test-revision")


def test_cpu_callers_share_one_encoding_slot_and_keep_their_own_vectors(monkeypatch):
    embedder = make_embedder(monkeypatch, "cpu")
    start = threading.Barrier(4)
    def encode(text):
        start.wait(timeout=2)
        return embedder.embed_query(embedder.name, text)
    texts = ["a", "bb", "ccc", "dddd"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(encode, texts))
    assert embedder.model.peak == 1
    assert np.array_equal(rows, [[1., 1.], [2., 1.], [3., 1.], [4., 1.]])
    assert set(embedder.model.batches) == {16}


def test_gpu_encoding_keeps_existing_parallel_path(monkeypatch):
    embedder = make_embedder(monkeypatch, "cuda")
    embedder.model.meeting = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda text: embedder.embed([text]), ["a", "bb"]))
    assert embedder.model.peak == 2
    assert set(embedder.model.batches) == {64}
    assert np.array_equal(rows, [[[1., 1.]], [[2., 1.]]])


def test_encoding_failure_releases_cpu_slot(monkeypatch):
    embedder = make_embedder(monkeypatch, "cpu")
    embedder.model.fail = True
    with pytest.raises(ValueError, match="fake encoding failure"):
        embedder.embed(["bad"])
    assert embedder._encode_lock.acquire(blocking=False)
    embedder._encode_lock.release()
    embedder.model.fail = False
    assert np.array_equal(embedder.embed(["ok"]), [[2., 1.]])
