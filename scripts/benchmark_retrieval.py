"""Paired, warmed exact/USearch comparison using local embeddings only.

Separately times query embeddings, warms every retrieval mode, then alternates their
order across repeated trials. Never imports the answer pipeline or changes the service.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muhawir.store import SqliteCorpus, SqliteRetriever  # noqa: E402
from muhawir.vectors import HybridRetriever, is_local, load_vectors  # noqa: E402


def prepare_queries(embed, questions, clock=time.perf_counter):
    """Embed each distinct query once, outside every retrieval timing."""
    vectors, seconds = {}, []
    for item in questions:
        q = item["q"]
        if q not in vectors:
            start = clock()
            vectors[q] = embed(q)
            seconds.append(clock() - start)
    return vectors, seconds


def percentile95(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1]


def paired_search(questions, modes, depth, repeats, clock=time.perf_counter):
    """Time warmed retrieval with balanced ordering; retain actual evidence and every gold rank."""
    if not questions or not modes or repeats < 1 or depth < 10:
        raise ValueError("questions, modes, positive repeats and depth >= 10 are required")

    def search(name, q):
        retriever = modes[name]
        hits = retriever.search(q, depth)
        if isinstance(retriever, HybridRetriever):
            if retriever._disabled:
                raise RuntimeError(f"{name}: {retriever._warn_reason}")
            if name != "exact" and retriever.vectors.ann is None:
                raise RuntimeError(f"{name}: ANN failed; refusing to report exact fallback as ANN timing")
        return hits

    names = list(modes)
    for item in questions:
        for name in names:
            search(name, item["q"])  # warm every mode and its query cache, without recording timing
    result = {name: {"seconds": [], "ranks": [[] for _ in questions], "hits": {}} for name in names}
    for trial in range(repeats):
        for i, item in enumerate(questions):
            offset = (trial + i) % len(names)
            order = names[offset:] + names[:offset]
            if trial % 2:
                order.reverse()
            for name in order:
                start = clock()
                hits = search(name, item["q"])
                result[name]["seconds"].append(clock() - start)
                rank = next((r for r, h in enumerate(hits, 1) if h.passage.id in item["gold"]), None)
                result[name]["ranks"][i].append(rank)
                result[name]["hits"].setdefault(i, hits)
    return result


def card(corpus, passage):
    return {"id": passage.id, "source": corpus.source_of(passage).name,
            "location": passage.location, "text": passage.text}


def review_report(corpus, questions, outcomes):
    """Keep missing reference IDs, valid reference texts and competing hits for human review."""
    rows = []
    for i, item in enumerate(questions):
        gold = [corpus.passage(pid) for pid in item["gold"]]
        rows.append({"question": item["q"], "gold": item["gold"],
                     "missing_gold": [pid for pid, p in zip(item["gold"], gold) if p is None],
                     "present_gold": [card(corpus, p) for p in gold if p is not None],
                     "modes": {name: {"gold_ranks": data["ranks"][i],
                                      "hits": [card(corpus, h.passage) for h in data["hits"][i]]}
                               for name, data in outcomes.items()}})
    return {"timing_scope": "warmed retrieval with precomputed query embeddings; not full answer latency",
            "results": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=ROOT / "data/muhawir.db")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--gold", type=Path, default=ROOT / "scripts/retrieval_gold.json")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.depth < 10 or args.repeats < 1:
        parser.error("depth must be >= 10 and repeats must be positive")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MUHAWIR_VECTOR_INDEX="exact")
    index = load_vectors(args.data_dir)
    if index is None or not is_local(index.model_name):
        parser.error("a complete local embedding index is required; no remote embedding is permitted")
    os.environ["MUHAWIR_VECTOR_INDEX"] = "usearch"
    ann_index = load_vectors(args.data_dir)
    if ann_index is None or ann_index.ann is None:
        parser.error("a matching USearch index is required; build it with vectors build-ann first")
    questions = json.loads(args.gold.read_text(encoding="utf-8"))["questions"]
    if not questions:
        parser.error("the gold set is empty")
    corpus = SqliteCorpus(args.db)
    if index.ids != [row[0] for row in corpus.con.execute("SELECT id FROM passages ORDER BY rid")]:
        parser.error("vector IDs must match the corpus in order; refusing a mismatched benchmark")
    query_vectors, embedding_seconds = prepare_queries(index.query_embedder, questions)
    modes = {}
    for name, ann, rerank in (("exact", None, 1), ("USearch", ann_index.ann, 1),
                              ("USearch + rerank", ann_index.ann, 4)):
        # A separate VectorIndex per mode prevents an ANN fallback or cache mutation crossing modes.
        candidate = replace(index, ann=ann, query_embedder=lambda q: query_vectors[q].copy())
        modes[name] = HybridRetriever(SqliteRetriever(corpus), candidate, ann_rerank_factor=rerank)
    outcomes = paired_search(questions, modes, args.depth, args.repeats)
    report = review_report(corpus, questions, outcomes)
    report.update(repeats=args.repeats, depth=args.depth, model=index.model_name,
                  embedding_seconds=embedding_seconds)
    missing = sorted({pid for row in report["results"] for pid in row["missing_gold"]})
    print(f"Gold: {len(questions)} questions; {len(missing)} missing IDs; "
          f"{sum(not row['present_gold'] for row in report['results'])} questions without present gold")
    print(f"Embeddings separately: total {sum(embedding_seconds):.2f}s; "
          f"first {embedding_seconds[0]:.2f}s (includes lazy model load); "
          f"median {statistics.median(embedding_seconds):.2f}s")
    print(f"Retrieval only: all modes warmed; {args.repeats} alternating trials; depth {args.depth}")
    print("Gold columns use the first trial; rank variation is reported separately.")
    print("| mode | gold top 1 | gold top 5 | gold top 10 | median ms | p95 ms | unstable questions |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    report["summary"] = {}
    for name, data in outcomes.items():
        ranks = [runs[0] for runs in data["ranks"]]
        metrics = {f"gold_top_{k}": sum(r is not None and r <= k for r in ranks) for k in (1, 5, 10)}
        metrics.update(median_ms=statistics.median(data["seconds"]) * 1000,
                       p95_ms=percentile95(data["seconds"]) * 1000,
                       unstable_questions=sum(len(set(runs)) > 1 for runs in data["ranks"]))
        report["summary"][name] = metrics
        print(f"| {name} | {metrics['gold_top_1']}/{len(ranks)} | {metrics['gold_top_5']}/{len(ranks)} | "
              f"{metrics['gold_top_10']}/{len(ranks)} | {metrics['median_ms']:.0f} | "
              f"{metrics['p95_ms']:.0f} | {metrics['unstable_questions']} |")
    print("Gold hits measure this reference list only, not every valid source or answer quality.")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Review evidence saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
