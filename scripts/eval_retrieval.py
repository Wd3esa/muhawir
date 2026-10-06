"""Measure whether search finds the passage that answers a question: keywords, keywords with vectors, and reranked.

For each question in scripts/retrieval_gold.json: is one of its passages first, in the first 5, in the first 10
(and the mean reciprocal rank), plus the time per question. No model calls: it can run on the server at no cost.
The questions are searched as the user wrote them; in answers the search also uses phrases from the understanding
step, so these numbers are a floor, not the exact figure for the site.

Usage (on the server, in ~/muhawir):
  MUHAWIR_VECTORS=1 .venv/bin/python scripts/eval_retrieval.py \
      --rerankers BAAI/bge-reranker-v2-m3,cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
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

DEPTH = 30  # results read from each search, and reranked


def score(ranked_ids: list[str], gold: set[str]) -> int | None:
    """1-based rank of the first passage that answers, or None."""
    return next((i for i, pid in enumerate(ranked_ids, 1) if pid in gold), None)


def summary(name: str, ranks: list[int | None], times: list[float]) -> str:
    n = len(ranks)
    at = lambda k: sum(1 for r in ranks if r and r <= k)  # noqa: E731
    mrr = sum(1 / r for r in ranks if r and r <= 10) / n
    return (f"| {name} | {at(1)}/{n} | {at(5)}/{n} | {at(10)}/{n} | {mrr:.2f} | "
            f"{statistics.median(times) * 1000:.0f} ms |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=os.environ.get("MUHAWIR_DB") or str(ROOT / "data" / "muhawir.db"))
    parser.add_argument("--rerankers", default="", help="comma-separated cross-encoder model names")
    args = parser.parse_args()
    gold = json.loads((ROOT / "scripts" / "retrieval_gold.json").read_text(encoding="utf-8"))["questions"]
    corpus = SqliteCorpus(args.db)
    bm25 = SqliteRetriever(corpus)
    systems = {"keywords (BM25)": bm25}
    if os.environ.get("MUHAWIR_VECTORS") == "1":
        from muhawir.vectors import HybridRetriever, load_vectors
        hybrid = HybridRetriever(bm25, load_vectors())
        hybrid.warm_up()
        if not hybrid._disabled:
            systems["keywords + vectors (RRF)"] = hybrid
        else:
            print(f"vectors unavailable: {hybrid._warn_reason}")
    missing = sorted({pid for g in gold for pid in g["gold"] if corpus.passage(pid) is None})
    if missing:
        print(f"note: {len(missing)} gold passage id(s) not in this database: {', '.join(missing)}")

    results: dict[str, tuple[list, list]] = {}
    hits_of: dict[str, list] = {}
    for name, retriever in systems.items():
        ranks, times = [], []
        for g in gold:
            started = time.time()
            hits = retriever.search(g["q"], k=DEPTH)
            times.append(time.time() - started)
            hits_of[(name, g["q"])] = hits
            ranks.append(score([h.passage.id for h in hits], set(g["gold"])))
        results[name] = (ranks, times)
    base = list(systems)[-1]  # rerank the best list we have
    from muhawir.rerank import Reranker
    for model in [m.strip() for m in args.rerankers.split(",") if m.strip()]:
        reranker = Reranker(model)
        try:
            reranker.rerank("تجربة", hits_of[(base, gold[0]["q"])][:2])  # load once, outside the timing
        except Exception as exc:  # noqa: BLE001
            print(f"reranker {model} could not load: {type(exc).__name__}: {exc}")
            continue
        ranks, times = [], []
        for g in gold:
            started = time.time()
            ranked = reranker.rerank(g["q"], hits_of[(base, g["q"])])
            times.append(time.time() - started)
            ranks.append(score([h.passage.id for h in ranked], set(g["gold"])))
        results[f"{base} → reranked by {model}"] = (ranks, times)

    print(f"\n{len(gold)} questions; each search reads {DEPTH} results\n")
    print("| search | first | in top 5 | in top 10 | MRR@10 | time per question |")
    print("|---|---|---|---|---|---|")
    for name, (ranks, times) in results.items():
        print(summary(name, ranks, times))
    print("\nnot in the top 10:")
    for name, (ranks, _) in results.items():
        lost = [g["q"] for g, r in zip(gold, ranks) if not r or r > 10]
        print(f"- {name}: {len(lost)}" + (f" ({' | '.join(lost)})" if lost else ""))


if __name__ == "__main__":
    main()
