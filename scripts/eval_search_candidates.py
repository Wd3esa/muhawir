"""Compare retrieval candidates on the existing corpus, offline and without answer-model calls.

Run on the server with its local vectors. This is a shadow evaluation: it never changes
the live service or its environment. Do not use this script with a remote embedding index.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muhawir.store import SqliteCorpus, SqliteRetriever  # noqa: E402
from muhawir.vectors import HybridRetriever, is_local, load_vectors  # noqa: E402


def first_gold(hits, gold: set[str]) -> int | None:
    return next((rank for rank, hit in enumerate(hits, 1) if hit.passage.id in gold), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "muhawir.db")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--gold", type=Path, default=ROOT / "scripts" / "retrieval_gold.json")
    parser.add_argument("--depth", type=int, default=30)
    args = parser.parse_args(argv)
    if args.depth < 10:
        parser.error("depth must be at least 10")

    # The evaluation is explicitly local; an uncached model should fail rather than download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MUHAWIR_VECTOR_INDEX"] = "exact"
    index = load_vectors(args.data_dir)
    if index is None or not is_local(index.model_name):
        parser.error("a complete local embedding index is required; no remote model is called")
    os.environ["MUHAWIR_VECTOR_INDEX"] = "usearch"
    ann_index = load_vectors(args.data_dir)
    if ann_index is None or ann_index.ann is None:
        parser.error("a valid USearch index is required; run scripts/prepare_optional_assets.py")
    ann = ann_index.ann

    corpus = SqliteCorpus(args.db)
    bm25 = SqliteRetriever(corpus)
    questions = json.loads(args.gold.read_text(encoding="utf-8"))["questions"]
    if not questions:
        parser.error("the gold set is empty")
    missing = sorted({pid for item in questions for pid in item["gold"] if corpus.passage(pid) is None})
    unusable = [item["q"] for item in questions if not any(corpus.passage(pid) for pid in item["gold"])]
    print(f"Gold: {len(questions)} questions, {len(missing)} missing passage IDs, "
          f"{len(unusable)} questions without any present gold passage")
    for pid in missing:
        print(f"  missing: {pid}")

    # All variants use the same lazy local embedder and question-vector cache.
    embed = index.query_embedder
    query_vectors: dict[str, object] = {}

    def cached_embed(question: str):
        if question not in query_vectors:
            query_vectors[question] = embed(question)
        return query_vectors[question]

    index.query_embedder = cached_embed
    modes = (
        ("BM25", None, 1, 1),
        ("exact", None, 1, 1),
        ("exact + wider BM25", None, 4, 1),
        ("USearch", ann, 1, 1),
        ("USearch + exact rerank", ann, 1, 4),
        ("USearch + rerank + wider BM25", ann, 4, 4),
    )
    outcomes: dict[str, tuple[list[int | None], float]] = {}
    for name, use_ann, bm25_factor, ann_factor in modes:
        index.ann = use_ann
        retriever = (bm25 if name == "BM25" else HybridRetriever(
            bm25, index, bm25_pool_factor=bm25_factor, ann_rerank_factor=ann_factor))
        if isinstance(retriever, HybridRetriever):
            retriever.warm_up()
            if retriever._disabled:
                raise RuntimeError(f"{name}: {retriever._warn_reason}")
        ranks, elapsed = [], []
        for item in questions:
            start = time.perf_counter()
            hits = retriever.search(item["q"], args.depth)
            elapsed.append(time.perf_counter() - start)
            ranks.append(first_gold(hits, set(item["gold"])))
        if use_ann is not None and index.ann is None:
            raise RuntimeError(f"{name}: USearch failed and fell back to exact search")
        outcomes[name] = ranks, statistics.median(elapsed) * 1000

    print("| search | first | top 5 | top 10 | MRR@10 | median ms |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, (ranks, median) in outcomes.items():
        n = len(ranks)
        mrr = sum(1 / rank for rank in ranks if rank and rank <= 10) / n
        print(f"| {name} | {sum(rank == 1 for rank in ranks)}/{n} | "
              f"{sum(rank is not None and rank <= 5 for rank in ranks)}/{n} | "
              f"{sum(rank is not None and rank <= 10 for rank in ranks)}/{n} | "
              f"{mrr:.2f} | {median:.0f} |")
    print("\nPer question: exact / wide exact / USearch / reranked USearch / wide reranked")
    for i, item in enumerate(questions):
        names = ("exact", "exact + wider BM25", "USearch", "USearch + exact rerank",
                 "USearch + rerank + wider BM25")
        ranks = [outcomes[name][0][i] for name in names]
        print(f"{item['q']} | " + " / ".join(str(rank or "-") for rank in ranks))
    if unusable:
        print("\nQuestions without present gold: " + " | ".join(unusable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
