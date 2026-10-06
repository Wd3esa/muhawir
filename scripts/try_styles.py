"""Ask the same questions in the four explanation styles and measure whether each answer fits its reader.

What is measured for each answer (counts only; the reviewer reads the report for the rest):
  - sentences, average and longest sentence in words;
  - hard words left in a children's answer (النصاب، الحول، المذاهب، الجمهور …);
  - lines that open by crediting a source («بحسب الشيخ …»), which weigh a short answer down;
  - how much the answer repeats the detailed («موسّع») answer word for word: a style that changes nothing.
An answer is flagged when it breaks its own style's limits (STYLE_LIMITS below, taken from generate.STYLE_GUIDE).

Usage (on the server):  .venv/bin/python scripts/try_styles.py --url http://127.0.0.1:8000 [--out styles.txt]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

QUESTIONS = ["ما هي الزكاة؟", "كيف أتوضأ؟", "ما معنى التوحيد؟", "لماذا نصوم رمضان؟", "ما هي الشفاعة؟",
             "هل لمس المرأة ينقض الوضوء؟"]
STYLES = ["kids", "youth", "newcomer", "extended"]
# (most sentences, longest sentence in words) per style, from the style guide in generate.py
STYLE_LIMITS = {"kids": (5, 14), "youth": (6, 28), "newcomer": (8, 30), "extended": (12, 45)}
HARD_FOR_KIDS = ["النصاب", "الحول", "المذاهب", "الجمهور", "الحنفية", "المالكية", "الشافعية", "الحنابلة", "مكروه",
                 "مستحب", "الإجماع", "القياس", "الاستنجاء", "الحدث", "نجاسة", "يجزئ", "تجزئ"]


def ask(url: str, question: str, style: str) -> tuple[dict, float]:
    body = json.dumps({"question": question, "style": style, "lang": "ar", "history": []}).encode()
    req = urllib.request.Request(f"{url}/api/ask", data=body, headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()), time.time() - started


def sentences(res: dict) -> list[str]:
    out = [c["text"] for c in res.get("claims", [])] + [f"{v['school']}: {v['text']}" for v in res.get("views", [])]
    return [s for s in out if s.strip()]


def words(text: str) -> list[str]:
    return re.findall(r"[\w؀-ۿ]+", text)


def overlap(a: str, b: str) -> float:
    """Share of a's three-word runs that appear in b."""
    def grams(t):
        w = words(t)
        return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / len(ga) if ga else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", default="muhawir_styles.txt")
    args = parser.parse_args()
    lines, flags = [], []

    def out(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    for q in QUESTIONS:
        answers = {}
        for style in STYLES:
            try:
                res, secs = ask(args.url, q, style)
            except Exception as exc:  # keep going
                out(f"\n=== {q} [{style}]\nERROR: {exc}")
                flags.append(f"{q} [{style}]: request error")
                continue
            sents = sentences(res)
            answers[style] = " ".join(sents)
            lens = [len(words(s)) for s in sents] or [0]
            hard = [w for w in HARD_FOR_KIDS if w in answers[style]] if style == "kids" else []
            credit = sum(s.startswith("بحسب") for s in sents)
            most, longest = STYLE_LIMITS[style]
            out(f"\n=== {q} [{style}] status: {res['status']} ({secs:.0f}s)")
            out(f"sentences: {len(sents)} · average words: {sum(lens) / len(lens):.0f} · longest: {max(lens)}"
                + (f" · hard words: {'، '.join(hard)}" if hard else "") + (f" · «بحسب…» lines: {credit}" if credit else ""))
            for s in sents:
                out(f"- {s}")
            if res["status"] != "answered":
                flags.append(f"{q} [{style}]: {res['status']}")
                continue
            if len(sents) > most:
                flags.append(f"{q} [{style}]: {len(sents)} sentences (limit {most})")
            if max(lens) > longest:
                flags.append(f"{q} [{style}]: a sentence of {max(lens)} words (limit {longest})")
            if hard:
                flags.append(f"{q} [kids]: hard words {'، '.join(hard)}")
            if credit and style in ("kids", "youth"):
                flags.append(f"{q} [{style}]: {credit} sentence(s) opening with «بحسب…»")
        for style in ("kids", "youth", "newcomer"):
            if style in answers and "extended" in answers:
                same = overlap(answers[style], answers["extended"])
                if same > 0.6:
                    flags.append(f"{q} [{style}]: {same:.0%} the same as the detailed answer")

    out("\n======== summary")
    out(f"{len(QUESTIONS)} questions × {len(STYLES)} styles; flagged: {len(flags)}")
    for f in flags:
        out(f"FLAG  {f}")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
