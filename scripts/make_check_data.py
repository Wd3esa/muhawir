"""Training data for a dedicated support checker: (sentence, passage) pairs labelled supported (1) or not (0).

Off-the-shelf small models failed the stress test (scripts/checker_stress.py, November 2026): Qwen3-8B rejected every
true sentence and took minutes per check on the server's CPU; Gemma 4 E4B let changed rulings and schools through. A
small model trained for this one question, «does this sentence say what this passage says?», is the planned fix.

This script makes its data from Muhawir's own books, with no model and no cost:
  supported (1)   a sentence of the passage, word for word ("faithful")
  not (0)         that sentence changed by code: negated, consensus added, number changed, ruling flipped, school
                  swapped, condition dropped (scripts/checker_stress.py); cited against a passage of another book
                  ("other_passage"); or merged with a sentence of another book ("merged")
A model trained only on word-for-word positives would learn that every reworded sentence is wrong, so the script
also writes the true sentences to reword (rewrite_todo.jsonl): a larger open model rewords them on Kaggle, those
rewordings become positives too, and their distortions negatives (step 2).

Books are split, not passages: the test books are never seen in training, so the score shows what the checker does on
books it has not read.

    .venv/bin/python scripts/make_check_data.py --out ~/check-data --passages 3000
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from checker_stress import DISTORTIONS, first_sentence  # noqa: E402

from muhawir.corpus import Passage  # noqa: E402
from muhawir.generate import CHECK_PROMPT_TUNED, check_request  # noqa: E402
from muhawir.verify import Claim  # noqa: E402

# the verdict a trained checker must give for each kind of pair, in the service's own words (CHECK_PROMPT)
PROBLEM = {"faithful": "none", "negated": "distortion", "number": "distortion", "ruling_flip": "distortion",
           "school_swap": "distortion", "added_consensus": "addition", "merged": "addition",
           "other_passage": "addition", "condition_dropped": "dropped_condition"}

KINDS = ("fiqh", "hadith", "tafsir", "aqeedah", "seerah", "fatwa", "qa")


def sentences(text: str, most: int = 2) -> list[str]:
    """Up to ``most`` sentences of 8 to 40 words from a passage, word for word, in order."""
    out, rest = [], text
    while rest and len(out) < most:
        s = first_sentence(rest)
        if not s:
            break
        out.append(s)
        rest = rest[rest.index(s) + len(s):]
    return out


def load(db: Path, n: int, seed: int) -> list[dict]:
    """``n`` mid-length passages of the answer kinds, the same ones for the same seed."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, source_id, kind, text, location, grade FROM passages "
        f"WHERE kind IN ({','.join('?' * len(KINDS))}) AND length(text) BETWEEN 120 AND 2000", KINDS).fetchall()
    random.Random(seed).shuffle(rows)
    return [{"id": r[0], "source": r[1], "kind": r[2], "text": r[3], "location": r[4] or "", "grade": r[5] or ""}
            for r in rows[:n]]


def split_sources(passages: list[dict], seed: int, test_share: float = 0.15, dev_share: float = 0.1) -> dict[str, str]:
    """Each book to train, dev or test; whole books, so a test book is never seen in training."""
    books = sorted({p["source"] for p in passages})
    random.Random(seed).shuffle(books)
    n_test, n_dev = max(1, round(len(books) * test_share)), max(1, round(len(books) * dev_share))
    return {b: "test" if i < n_test else "dev" if i < n_test + n_dev else "train" for i, b in enumerate(books)}


def pairs_for(p: dict, s: str, others: list[dict], rng: random.Random, negatives: int) -> list[dict]:
    """The true pair for sentence ``s`` of passage ``p`` and up to ``negatives`` wrong ones of different kinds."""
    base = {"passage_id": p["id"], "source": p["source"], "kind": p["kind"], "location": p.get("location", ""),
            "grade": p.get("grade", "")}
    out = [{**base, "sentence": s, "passage": p["text"], "label": 1, "type": "faithful"}]
    wrong = []
    for name, make in DISTORTIONS.items():
        changed = make(s)
        if changed and changed != s:
            wrong.append({**base, "sentence": changed, "passage": p["text"], "label": 0, "type": name})
    other = next((o for o in rng.sample(others, min(len(others), 20)) if o["source"] != p["source"]), None)
    if other:
        wrong.append({**base, "sentence": s, "passage": other["text"], "passage_id": other["id"],
                      "passage_source": other["source"], "kind": other["kind"], "location": other.get("location", ""),
                      "grade": other.get("grade", ""), "label": 0, "type": "other_passage"})
        other_s = sentences(other["text"], 1)
        if other_s:  # two narrations merged into one sentence, cited against the first only
            wrong.append({**base, "sentence": s.rstrip(".،, ") + "، و" + other_s[0], "passage": p["text"],
                          "label": 0, "type": "merged"})
    rng.shuffle(wrong)
    return out + wrong[:negatives]


def chat_example(r: dict) -> dict:
    """A pair as a training conversation: the service's own check message (check_request) under the short instruction,
    and the reply the trained checker must give. A true sentence's witness is its first words (found in the passage
    word for word, as the service requires); a wrong one has no witness and names its problem."""
    p = Passage(id=r["passage_id"], source_id=r.get("passage_source", r["source"]), location=r.get("location", ""), text=r["passage"],
                kind=r["kind"], grade=r.get("grade", ""))
    ok = r["label"] == 1
    reply = {"on_topic": [True], "missing": [[]], "evidence": [" ".join(r["sentence"].split()[:15]) if ok else ""],
             "problem": [PROBLEM[r["type"]]]}
    return {"messages": [{"role": "system", "content": CHECK_PROMPT_TUNED},
                         {"role": "user", "content": check_request([Claim(r["sentence"], (p.id,))], {p.id: p})},
                         {"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)}],
            "label": r["label"], "type": r["type"]}


def build(passages: list[dict], seed: int, per_passage: int = 2, negatives: int = 3):
    """(pairs by split, sentences to reword by split)."""
    rng = random.Random(seed)
    splits = split_sources(passages, seed)
    data = {"train": [], "dev": [], "test": []}
    todo = {"train": [], "dev": [], "test": []}
    by_split = {part: [p for p in passages if splits[p["source"]] == part] for part in data}
    for p in passages:
        part = splits[p["source"]]
        for s in sentences(p["text"], per_passage):
            # the other book comes from the same split: a test book's text never reaches training
            data[part] += pairs_for(p, s, by_split[part], rng, negatives)
            todo[part].append({"passage_id": p["id"], "source": p["source"], "kind": p["kind"],
                               "sentence": s, "passage": p["text"]})
    return data, todo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "muhawir.db")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--passages", type=int, default=3000)
    ap.add_argument("--negatives", type=int, default=3, help="wrong pairs per true one")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    data, todo = build(load(args.db, args.passages, args.seed), args.seed, negatives=args.negatives)
    args.out.mkdir(parents=True, exist_ok=True)
    for part in data:
        with open(args.out / f"{part}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in data[part])
        with open(args.out / f"sft_{part}.jsonl", "w", encoding="utf-8") as f:  # the same pairs as conversations
            f.writelines(json.dumps(chat_example(r), ensure_ascii=False) + "\n" for r in data[part])
    with open(args.out / "rewrite_todo.jsonl", "w", encoding="utf-8") as f:
        for part, rows in todo.items():
            f.writelines(json.dumps({**r, "split": part}, ensure_ascii=False) + "\n" for r in rows)
    for part, rows in data.items():
        kinds = Counter(r["type"] for r in rows)
        print(f"{part:5} {len(rows):6} pairs  supported {kinds['faithful']}  " +
              " ".join(f"{k}={v}" for k, v in sorted(kinds.items()) if k != "faithful"))
    print(f"to reword: {sum(len(v) for v in todo.values())} sentences -> {args.out / 'rewrite_todo.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
