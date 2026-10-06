"""Optional reranking of search results with a cross-encoder, which reads the question and each passage together.

Off unless MUHAWIR_RERANK names a model. Not used in answers until scripts/eval_retrieval.py shows, on the server,
that it finds the right passage more often and how many seconds it adds per question (6 October 2026).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("muhawir.rerank")

MAX_PASSAGE_CHARS = 1200  # a cross-encoder reads about 512 tokens: the start of a passage holds its matter


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # optional dependency (requirements-vectors.txt)
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def rerank(self, question: str, hits: list, top: int | None = None) -> list:
        """The same hits, best first by the cross-encoder's score."""
        if not hits:
            return hits
        scores = self._load().predict([(question, h.passage.text[:MAX_PASSAGE_CHARS]) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        ranked = [hits[i] for i in order]
        return ranked[:top] if top else ranked


def from_env() -> Reranker | None:
    name = os.environ.get("MUHAWIR_RERANK", "").strip()
    return Reranker(name) if name else None
