"""Export old/new search results for human source review, offline and without answer calls.

Run on the machine with the real SQLite corpus and its complete local vectors.
The default questions are committed test questions. Review any custom question file
for private information before sharing the report.
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

# Additional questions from the project's dialogue review. The labeled questions
# in retrieval_gold.json are included first, without inventing gold labels for these.
EXTRA_QUESTIONS = (
    "اشرح الزكاة وأنواعها",
    "ما هو الحول في الزكاة؟",
    "ما معنى التوحيد؟",
    "هل تارك الصلاة كافر؟",
    "ما هي الشفاعة؟",
    "إذا كان لكل شيء خالق، فمن خلق الله؟",
    "لماذا خلق الله الشر والمرض والزلازل إذا كان رحيمًا؟",
    "ما ذنب من لم يسمع بالإسلام ولم يصله القرآن؟",
    "لماذا لا يُرينا الله معجزة اليوم حتى نؤمن؟",
    "هل الربا في البنوك حلال؟",
    "Why would a merciful God allow suffering?",
    "المضاربة",
)


def questions_from_files(gold_path: Path, question_path: Path | None) -> list[dict]:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))["questions"]
    questions = {item["q"]: {"question": item["q"], "gold": item["gold"]} for item in gold}
    if question_path is None:
        extras = EXTRA_QUESTIONS
    else:
        extras = question_path.read_text(encoding="utf-8").splitlines()
    for line in extras:
        question = line.strip()
        if question and not question.startswith("#"):
            questions.setdefault(question, {"question": question, "gold": []})
    if not questions:
        raise ValueError("no questions to review")
    return list(questions.values())


def export(corpus, old, new, questions: list[dict], *, depth: int = 10) -> dict:
    rows = []
    elapsed = {"old": [], "new": []}
    for item in questions:
        question = item["question"]
        present = [pid for pid in item["gold"] if corpus.passage(pid) is not None]
        missing = [pid for pid in item["gold"] if pid not in present]
        row = {"question": question, "present_gold": present, "missing_gold": missing}
        for name, retriever in (("old", old), ("new", new)):
            start = time.perf_counter()
            hits = retriever.search(question, depth)
            elapsed[name].append((time.perf_counter() - start) * 1000)
            row[name] = [{
                "rank": i, "id": hit.passage.id,
                "source": corpus.source_of(hit.passage).name,
                "location": hit.passage.location,
                "text": hit.passage.text[:600],
            } for i, hit in enumerate(hits, 1)]
            row[f"{name}_gold_rank"] = next(
                (h["rank"] for h in row[name] if h["id"] in present), None)
        rows.append(row)
    return {
        "method": "raw question to local BM25 + exact vectors; no answer model",
        "depth": depth,
        "questions": len(rows),
        "missing_gold_ids": sorted({pid for row in rows for pid in row["missing_gold"]}),
        "median_ms": {name: round(statistics.median(times), 1) for name, times in elapsed.items()},
        "results": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "muhawir.db")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--gold", type=Path, default=ROOT / "scripts" / "retrieval_gold.json")
    parser.add_argument("--questions", type=Path, help="optional extra real questions, one per line")
    parser.add_argument("--output", type=Path, default=Path("/tmp/muhawir-search-review.json"))
    args = parser.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MUHAWIR_VECTOR_INDEX"] = "exact"
    vectors = load_vectors(args.data_dir)
    if vectors is None or not is_local(vectors.model_name):
        parser.error("complete local vectors required; no remote embedding API will be used")
    corpus = SqliteCorpus(args.db)
    base = SqliteRetriever(corpus)
    # Share query embeddings between comparisons; the model is loaded once.
    embed = vectors.query_embedder
    cache = {}

    def cached_embed(question):
        if question not in cache:
            cache[question] = embed(question)
        return cache[question]

    vectors.query_embedder = cached_embed
    old = HybridRetriever(base, vectors, bm25_pool_factor=1)
    new = HybridRetriever(base, vectors, bm25_pool_factor=4)
    old.warm_up()
    if old._disabled:
        parser.error(f"local query model unavailable: {old._warn_reason}")
    items = questions_from_files(args.gold, args.questions)
    report = export(corpus, old, new, items)
    report["embedding_model"] = vectors.model_name
    if old._disabled or new._disabled:
        parser.error("vector search failed during the comparison; report not saved")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(items)} questions to {args.output}; no answer-model calls")
    print(f"Missing labeled IDs: {len(report['missing_gold_ids'])}")
    for name in ("old", "new"):
        found = sum(row[f"{name}_gold_rank"] is not None for row in report["results"] if row["present_gold"])
        labeled = sum(bool(row["present_gold"]) for row in report["results"])
        print(f"{name}: labeled source in top 10: {found}/{labeled}; median search {report['median_ms'][name]} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
