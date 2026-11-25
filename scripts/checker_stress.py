"""Stress test for the support check: how often does a model approve a sentence its passage does NOT support?

The support check (generate.py, CHECK_PROMPT) is Muhawir's guard against answers that cite a real passage but change
its meaning. With a small local model doing that check, this script measures it before the model is trusted. FREE: it
calls only a model served by Ollama on this server, and reads the real passages from data/muhawir.db.

For each sampled passage, a sentence is taken from the passage word for word (it must be APPROVED), and wrong versions
of it are made by code (each must be REJECTED):
  negated           the sentence presented as wrong («من الخطأ قول: …»)
  added_consensus   a claim of scholarly consensus the passage does not make, added at the end
  other_passage     the true sentence, cited against a passage from another book
  number            a number changed (digits, or a number word)
  ruling_flip       a ruling turned to another (حلال/حرام، يجوز/لا يجوز، واجب/مستحب …)
  school_swap       a school or imam replaced by another
  condition_dropped an exception or condition clause removed (إلا، إذا، بشرط، ما لم …)
Each case goes through the same check the service uses (ModelGenerator.check_support), one sentence per call.

    .venv/bin/python scripts/checker_stress.py --model qwen3:8b --per-type 4 --out ~/stress-qwen3.json

The result: per type, how many were approved / rejected / unusable; the false-approval rate (wrong sentences approved)
must be near zero, and the faithful sentences should mostly be approved. The JSON keeps every case for a human to read.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muhawir.corpus import Passage  # noqa: E402
from muhawir.generate import ModelGenerator, ollama_call  # noqa: E402
from muhawir.verify import Claim  # noqa: E402

KINDS = ("fiqh", "hadith", "tafsir")
_SENTENCE_END = re.compile(r"(?<=[.؟!؛\n])\s+")

# Closed grammatical or legal sets used only to MAKE wrong test sentences (never by the service itself)
_NUMBER_WORDS = ["واحد", "اثنان", "اثنين", "ثلاث", "ثلاثة", "أربع", "أربعة", "خمس", "خمسة", "ست", "ستة", "سبع",
                 "سبعة", "ثمان", "ثمانية", "تسع", "تسعة", "عشر", "عشرة", "مائة", "ألف"]
_RULINGS = [("لا يجوز", "يجوز"), ("يجوز", "لا يجوز"), ("حرام", "حلال"), ("حلال", "حرام"), ("محرم", "مباح"),
            ("واجب", "مستحب"), ("مستحب", "واجب"), ("فرض", "سنة"), ("سنة", "فرض"), ("مكروه", "واجب"),
            ("يحرم", "يستحب"), ("يجب", "لا يجب"), ("لا يجب", "يجب"), ("باطل", "صحيح"), ("صحيح", "باطل")]
_SCHOOLS = [("الحنفية", "الشافعية"), ("الشافعية", "الحنفية"), ("المالكية", "الحنابلة"), ("الحنابلة", "المالكية"),
            ("أبي حنيفة", "الشافعي"), ("أبو حنيفة", "الشافعي"), ("الشافعي", "مالك"), ("مالك", "أحمد"),
            ("أحمد", "أبي حنيفة")]
_CONDITION = re.compile(r"[،,]?\s*(?:إلا|إذا|إذ|إن كان|إن كانت|بشرط|شريطة|ما لم|ما دام|متى)\b[^،,.؟!]*")


def _word_swap(text: str, pairs) -> str | None:
    """Replace the first whole-word occurrence of any pair's left side with its right side."""
    for old, new in pairs:
        m = re.search(rf"(?<![\w]){re.escape(old)}(?![\w])", text)
        if m:
            return text[:m.start()] + new + text[m.end():]
    return None


def negated(s: str, *_) -> str:
    return f"من الخطأ قول: «{s.rstrip('.')}»."


def added_consensus(s: str, *_) -> str:
    return s.rstrip(".،, ") + "، وهذا محل إجماع عند جميع العلماء بلا خلاف."


def number(s: str, *_) -> str | None:
    m = re.search(r"\d+", s)
    if m:
        return s[:m.start()] + str(int(m.group()) + 1) + s[m.end():]
    words = [(w, _NUMBER_WORDS[(i + 3) % len(_NUMBER_WORDS)]) for i, w in enumerate(_NUMBER_WORDS)]
    return _word_swap(s, words)


def ruling_flip(s: str, *_) -> str | None:
    return _word_swap(s, _RULINGS)


def school_swap(s: str, *_) -> str | None:
    return _word_swap(s, _SCHOOLS)


def condition_dropped(s: str, *_) -> str | None:
    out = " ".join(_CONDITION.sub("", s, count=1).split())
    return out if out != " ".join(s.split()) and len(out.split()) >= 6 else None


DISTORTIONS = {"negated": negated, "added_consensus": added_consensus, "number": number,
               "ruling_flip": ruling_flip, "school_swap": school_swap, "condition_dropped": condition_dropped}


def first_sentence(text: str, low: int = 8, high: int = 40) -> str | None:
    """The first sentence of a passage with between ``low`` and ``high`` words, word for word."""
    for part in _SENTENCE_END.split(text):
        part = part.strip()
        if low <= len(part.split()) <= high:
            return part
    return None


def sample(db: Path, n: int, seed: int) -> list[Passage]:
    """``n`` passages of the given kinds, picked the same way for the same seed, mid-length, with a usable sentence."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, source_id, location, kind, grade, text, keywords FROM passages "
        f"WHERE kind IN ({','.join('?' * len(KINDS))}) AND length(text) BETWEEN 150 AND 1500", KINDS).fetchall()
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows:
        p = Passage(id=r[0], source_id=r[1], location=r[2], text=r[5], kind=r[3], grade=r[4], keywords=r[6])
        if first_sentence(p.text):
            out.append(p)
        if len(out) >= n:
            break
    return out


def build_cases(passages: list[Passage], per_type: int) -> list[dict]:
    """The faithful sentence of each passage, then the wrong versions, at most ``per_type`` of each type."""
    cases, counts = [], {"faithful": 0, "other_passage": 0, **{k: 0 for k in DISTORTIONS}}
    for i, p in enumerate(passages):
        s = first_sentence(p.text)
        if counts["faithful"] < per_type:
            cases.append({"type": "faithful", "expect": "approve", "text": s, "passage": p})
            counts["faithful"] += 1
        other = next((q for q in passages[i + 1:] + passages[:i] if q.source_id != p.source_id), None)
        if other and counts["other_passage"] < per_type:
            cases.append({"type": "other_passage", "expect": "reject", "text": s, "passage": other})
            counts["other_passage"] += 1
        for name, make in DISTORTIONS.items():
            wrong = make(s)
            if wrong and wrong != s and counts[name] < per_type:
                cases.append({"type": name, "expect": "reject", "text": wrong, "passage": p})
                counts[name] += 1
    return cases


def run_case(gen: ModelGenerator, case: dict) -> dict:
    p = case["passage"]
    start = time.perf_counter()
    verdict = gen.check_support([Claim(case["text"], (p.id,))], {p.id: p}, "")
    seconds = time.perf_counter() - start
    if verdict is None or len(verdict) != 1:
        outcome = "unusable"
    else:
        outcome = "approve" if verdict[0] else "reject"
    reasons = list(getattr(gen, "last_check_reasons", []) or [])
    return {"type": case["type"], "expect": case["expect"], "outcome": outcome, "seconds": round(seconds, 1),
            "text": case["text"], "passage_id": p.id, "passage": p.text, "reason": reasons[0] if reasons else ""}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for t in ["faithful", "other_passage", *DISTORTIONS]:
        part = [r for r in rows if r["type"] == t]
        if part:
            out[t] = {k: sum(r["outcome"] == k for r in part) for k in ("approve", "reject", "unusable")}
            out[t]["n"] = len(part)
    wrong = [r for r in rows if r["expect"] == "reject" and r["outcome"] != "unusable"]
    right = [r for r in rows if r["expect"] == "approve" and r["outcome"] != "unusable"]
    out["false_approval_rate"] = round(sum(r["outcome"] == "approve" for r in wrong) / len(wrong), 3) if wrong else None
    out["faithful_approval_rate"] = round(sum(r["outcome"] == "approve" for r in right) / len(right), 3) if right else None
    out["median_seconds"] = sorted(r["seconds"] for r in rows)[len(rows) // 2] if rows else None
    return out


def render(summary: dict, model: str) -> str:
    lines = [f"{model}", f'{"type":18} {"n":>3} {"approve":>8} {"reject":>7} {"unusable":>9}  expected']
    for t, v in summary.items():
        if isinstance(v, dict):
            lines.append(f'{t:18} {v["n"]:>3} {v["approve"]:>8} {v["reject"]:>7} {v["unusable"]:>9}  '
                         f'{"approve" if t == "faithful" else "reject"}')
    lines.append(f'false approvals (wrong sentences approved): {summary["false_approval_rate"]}  (want ~0)')
    lines.append(f'faithful sentences approved: {summary["faithful_approval_rate"]}  (want high)')
    lines.append(f'median seconds per check: {summary["median_seconds"]}')
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="an Ollama model name, e.g. qwen3:8b or gemma4:e4b")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "muhawir.db")
    ap.add_argument("--per-type", type=int, default=4, help="cases of each type (8 types)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    # a wide pool, so the rarer cases (a ruling word, a school's name) find sentences too
    passages = sample(args.db, max(args.per_type * 150, 300), args.seed)
    cases = build_cases(passages, args.per_type)
    gen = ModelGenerator([("local", ollama_call(args.model, args.url, args.num_ctx, args.timeout))])
    rows = []
    for n, case in enumerate(cases, 1):
        row = run_case(gen, case)
        rows.append(row)
        mark = "✓" if row["outcome"] == row["expect"] else ("?" if row["outcome"] == "unusable" else "✗")
        print(f'{mark} {n:3}/{len(cases)} {row["type"]:17} {row["outcome"]:8} {row["seconds"]:6.1f}s', flush=True)
    summary = summarize(rows)
    print("\n" + render(summary, args.model))
    if args.out:
        args.out.write_text(json.dumps({"model": args.model, "summary": summary, "cases": rows},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
