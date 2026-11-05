"""Search benchmark against a public yardstick: Qur'an QA 2023, Task A (passage retrieval), November 2026.

Answers the question «is bge-m3 + BM25 the right choice?» with numbers instead of a choice made once. No answer model is
called: only search. Runs on the server (the embedding models download from Hugging Face on first use, then work
offline). Nothing from the dataset is written into this repository (its licence, CC BY-NC-ND 4.0, allows no derived
copies); it is downloaded beside it.

Two parts:
  1. options   every search option on the official Thematic Qur'anic Passage Collection (1,266 passages): BM25,
               each embedding model alone, each with BM25 (RRF, as Muhawir merges), and with a reranker on the top 50.
               Small enough for every model on the server's processor, and comparable with the published results.
  2. muhawir   Muhawir's own search as deployed, over its whole corpus: the verses and tafsir passages it finds are
               mapped to the official passages that hold them, and scored on the same questions.

Scores on the questions that have an answer (the official set also has questions with none; see --help for the note):
  MAP@10, MRR@10 (as the shared task reports them), Recall@10, and milliseconds per question on this machine.

    pip install sentence-transformers            # already there when vectors are on
    .venv/bin/python scripts/benchmark_search.py                       # default options, test questions
    .venv/bin/python scripts/benchmark_search.py --split dev --models BAAI/bge-m3 intfloat/multilingual-e5-large
    .venv/bin/python scripts/benchmark_search.py --skip-muhawir --no-rerank   # quicker

Published reference points on the test set (shared-task overview, ArabicNLP 2023): the organisers' BM25 baseline
MAP@10 0.090, MRR@10 0.226 (all questions, official zero-answer scoring); the best systems about MAP@10 0.31-0.43.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OFFICIAL = "https://gitlab.com/bigirqu/quran-qa-2023"
MIRROR = "https://github.com/YahyaHammoudeh0/islamic-rag-quran"  # carries the same Task A data under islamic-rag/test_data
DATA_HOME = Path.home() / "benchmarks" / "quran-qa-2023"
DEFAULT_MODELS = ("BAAI/bge-m3", "intfloat/multilingual-e5-large",
                  "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2")
RERANKER = "BAAI/bge-reranker-v2-m3"
DEPTH, POOL, RRF_K = 10, 50, 60
# models that expect a prefix on questions and on passages
PREFIXES = {"intfloat/": ("query: ", "passage: ")}


# ---- data ---------------------------------------------------------------------------------------------------------

def fetch_data(home: Path = DATA_HOME) -> Path:
    """The Task A data folder, downloaded once (the official GitLab, else the GitHub mirror), outside the repository."""
    for candidate in (home / "Task-A" / "data", home / "islamic-rag" / "test_data" / "quran-qa-2023" / "Task-A" / "data"):
        if (candidate / "Thematic_QPC").is_dir():
            return candidate
    home.parent.mkdir(parents=True, exist_ok=True)
    for url, folder in ((OFFICIAL, "Task-A/data"), (MIRROR, "islamic-rag/test_data/quran-qa-2023/Task-A/data")):
        if home.exists():
            subprocess.run(["rm", "-rf", str(home)], check=False)
        clone = subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", url, str(home)],
                               capture_output=True, text=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
        if clone.returncode == 0:
            subprocess.run(["git", "-C", str(home), "sparse-checkout", "set", folder], capture_output=True)
            if (home / folder / "Thematic_QPC").is_dir():
                return home / folder
    raise SystemExit("could not download the Qur'an QA 2023 data (GitLab or GitHub); pass --data DIR")


def load_passages(data: Path) -> list[tuple[str, str]]:
    """(passage id «sura:first-last», text) for the Thematic Qur'anic Passage Collection."""
    out = []
    for line in next((data / "Thematic_QPC").glob("*.tsv")).read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            pid, text = line.split("\t", 1)
            out.append((pid.strip(), text.strip()))
    return out


def load_questions(data: Path, split: str) -> dict[str, str]:
    path = next(data.glob(f"QQA23_TaskA_*{split}.tsv"))
    return dict(line.split("\t", 1) for line in path.read_text(encoding="utf-8").splitlines() if "\t" in line)


def load_qrels(data: Path, split: str) -> dict[str, set[str]]:
    """question id -> the relevant passage ids (a question with only «-1» has no answer in the Qur'an)."""
    qrels: dict[str, set[str]] = {}
    for line in next((data / "qrels").glob(f"QQA23_TaskA_*qrels_{split}.gold")).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] != "-1" and int(parts[3]) > 0:
            qrels.setdefault(parts[0], set()).add(parts[2])
        elif parts:
            qrels.setdefault(parts[0], set())
    return qrels


# ---- metrics ------------------------------------------------------------------------------------------------------

def scores(ranked: dict[str, list[str]], qrels: dict[str, set[str]], depth: int = DEPTH) -> dict[str, float]:
    """MAP@depth (precision at each relevant rank, over all relevant passages, as trec_eval's map_cut), MRR@depth and
    Recall@depth, averaged over the questions that have an answer."""
    answered = [q for q, rel in qrels.items() if rel]
    ap = rr = recall = 0.0
    for q in answered:
        rel, hits, total = qrels[q], 0, 0.0
        first = 0.0
        for rank, pid in enumerate(ranked.get(q, [])[:depth], 1):
            if pid in rel:
                hits += 1
                total += hits / rank
                first = first or 1 / rank
        ap += total / len(rel)
        rr += first
        recall += hits / len(rel)
    n = max(1, len(answered))
    return {"MAP@10": ap / n, "MRR@10": rr / n, "Recall@10": recall / n, "questions": len(answered)}


# ---- search options -----------------------------------------------------------------------------------------------

def bm25_ranker(passages: list[tuple[str, str]]):
    """Muhawir's own BM25 (its Arabic normalisation and tokenizer) over the collection."""
    from muhawir.corpus import Corpus, Passage, Source
    from muhawir.retrieve import Retriever
    corpus = Corpus({"qpc": Source("qpc", "QPC", "")},
                    [Passage(id=pid, source_id="qpc", location=pid, text=text, kind="quran") for pid, text in passages])
    retriever = Retriever(corpus)
    return lambda question, k: [h.passage.id for h in retriever.search(question, k=k)]


def dense_ranker(model_name: str, passages: list[tuple[str, str]], encoder=None):
    """One embedding model: passages encoded once, each question encoded and compared by cosine similarity."""
    import numpy as np
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name, device="cpu")
    q_prefix, p_prefix = next((v for k, v in PREFIXES.items() if model_name.startswith(k)), ("", ""))
    matrix = np.asarray(encoder.encode([p_prefix + t for _, t in passages], batch_size=16, normalize_embeddings=True))
    ids = [pid for pid, _ in passages]

    def rank(question, k):
        v = np.asarray(encoder.encode([q_prefix + question], normalize_embeddings=True))[0]
        order = np.argsort(-(matrix @ v))[:k]
        return [ids[i] for i in order]
    return rank


def rrf(*rankers):
    """Reciprocal-rank fusion, as Muhawir merges keyword and meaning search (k = 60)."""
    def rank(question, k):
        weights: dict[str, float] = {}
        for ranker in rankers:
            for position, pid in enumerate(ranker(question, POOL), 1):
                weights[pid] = weights.get(pid, 0.0) + 1 / (RRF_K + position)
        return sorted(weights, key=lambda p: -weights[p])[:k]
    return rank


def reranked(ranker, passages: list[tuple[str, str]], model_name: str = RERANKER, cross=None):
    """The top POOL of a ranker reordered by a cross-encoder that reads question and passage together."""
    if cross is None:
        from sentence_transformers import CrossEncoder
        cross = CrossEncoder(model_name, device="cpu", max_length=512)
    text = dict(passages)

    def rank(question, k):
        pool = ranker(question, POOL)
        if not pool:
            return []
        order = cross.predict([(question, text[pid]) for pid in pool], batch_size=16)
        return [pid for _, pid in sorted(zip(order, pool), key=lambda x: -x[0])][:k]
    return rank


def muhawir_ranker(passages: list[tuple[str, str]], engine=None):
    """Muhawir's search as deployed, over its own corpus: each verse it finds («q:S:A»), and each tafsir passage on a
    verse («سورة …، الآية N»), mapped to the official passage holding that verse; first occurrence kept."""
    if engine is None:
        from muhawir.server import engine as live
        engine = live
    holding: dict[tuple[int, int], str] = {}
    for pid, _ in passages:
        m = re.fullmatch(r"(\d+):(\d+)-(\d+)", pid)
        if m:
            for ayah in range(int(m.group(2)), int(m.group(3)) + 1):
                holding[(int(m.group(1)), ayah)] = pid
    tafsir_place = re.compile(r"(\d+):(\d+)")

    def place(p):
        if p.kind == "quran":
            parts = p.id.split(":")
            return (int(parts[1]), int(parts[2])) if len(parts) == 3 and parts[1].isdigit() else None
        if p.kind == "tafsir":
            m = tafsir_place.search(p.id.split(":", 1)[-1]) if p.id.count(":") >= 2 else None
            if m:
                return int(m.group(1)), int(m.group(2))
            verses = engine._referenced_verses(p.location)
            if verses:
                parts = verses[0].id.split(":")
                return int(parts[1]), int(parts[2])
        return None

    def rank(question, k):
        out: list[str] = []
        for hit in engine.retriever.search(question, k=POOL):
            where = place(hit.passage)
            pid = holding.get(where) if where else None
            if pid and pid not in out:
                out.append(pid)
            if len(out) >= k:
                break
        return out
    return rank


# ---- run ----------------------------------------------------------------------------------------------------------

def evaluate(name: str, ranker, questions: dict[str, str], qrels: dict[str, set[str]]) -> dict:
    started = time.perf_counter()
    ranked = {q: ranker(text, DEPTH) for q, text in questions.items() if q in qrels}
    per_question = (time.perf_counter() - started) * 1000 / max(1, len(ranked))
    result = {"option": name, **scores(ranked, qrels), "ms_per_question": round(per_question)}
    print(f"{name:<62} MAP@10 {result['MAP@10']:.3f}  MRR@10 {result['MRR@10']:.3f}  "
          f"R@10 {result['Recall@10']:.3f}  {result['ms_per_question']:>6} ms", flush=True)
    return result


def _service_settings() -> None:
    """The settings the service runs with (~/muhawir.env), when this shell has not loaded them; nothing is printed."""
    path = Path(os.environ.get("MUHAWIR_ENV_FILE") or Path.home() / "muhawir.env")
    if os.environ.get("LLM_PROVIDER") or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    _service_settings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog="Questions with no answer in the Qur'an are left out of the averages here; the "
                                        "shared task scores them too, so compare with its numbers by order, not digit.")
    ap.add_argument("--data", type=Path, help="the Task-A/data folder (downloaded when not given)")
    ap.add_argument("--split", default="test", choices=("test", "dev", "train"))
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--skip-muhawir", action="store_true", help="do not load Muhawir's corpus and search")
    ap.add_argument("--out", type=Path, help="write the results as JSON here")
    args = ap.parse_args()

    data = args.data or fetch_data()
    passages = load_passages(data)
    questions = load_questions(data, args.split)
    qrels = load_qrels(data, args.split)
    print(f"Qur'an QA 2023 Task A, {args.split}: {len(questions)} questions "
          f"({sum(1 for r in qrels.values() if r)} with an answer), {len(passages)} passages\n")

    results = []
    bm25 = bm25_ranker(passages)
    results.append(evaluate("BM25 (Muhawir's tokenizer)", bm25, questions, qrels))
    cross = None
    for model in args.models:
        try:
            dense = dense_ranker(model, passages)
        except Exception as exc:  # noqa: BLE001 - a model that cannot load is reported and skipped
            print(f"{model}: skipped ({type(exc).__name__}: {str(exc)[:120]})")
            continue
        results.append(evaluate(f"{model}", dense, questions, qrels))
        hybrid = rrf(bm25, dense)
        results.append(evaluate(f"BM25 + {model} (RRF, as Muhawir)", hybrid, questions, qrels))
        if not args.no_rerank:
            try:
                if cross is None:
                    from sentence_transformers import CrossEncoder
                    cross = CrossEncoder(RERANKER, device="cpu", max_length=512)
                results.append(evaluate(f"BM25 + {model} + rerank", reranked(hybrid, passages, cross=cross),
                                        questions, qrels))
            except Exception as exc:  # noqa: BLE001
                print(f"reranker skipped ({type(exc).__name__})")
                args.no_rerank = True
    if not args.skip_muhawir:
        try:
            results.append(evaluate("Muhawir as deployed (its corpus, mapped)", muhawir_ranker(passages), questions, qrels))
        except Exception as exc:  # noqa: BLE001
            print(f"Muhawir's own search skipped ({type(exc).__name__}: {str(exc)[:120]})")
    if args.out:
        args.out.write_text(json.dumps({"split": args.split, "results": results}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    best = max(results, key=lambda r: r["MAP@10"])
    print(f"\nbest MAP@10: {best['option']} ({best['MAP@10']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
