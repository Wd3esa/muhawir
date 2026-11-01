"""Compare the second reading one sentence at a time (today) with Check-groups (MUHAWIR_CHECK_GROUPS=1) on the same
sentences, before switching Check-groups on. PAID: it answers each question with the real model, then reads the same
checked sentences again in groups. Run it only with the owner's approval and a question list sized to the budget.

    .venv/bin/python scripts/compare_check_groups.py questions.txt --max-questions 20 > check-groups.json

questions.txt holds one question per line. The report lists, for each question, the sentences the two readings judged
differently; the one that matters is «accepted together, rejected alone»: a sentence the grouped reading would let
through. Check-groups is safe to switch on only if that list is empty or every case is reviewed and harmless.
No answer text is logged by the service; this report holds the sentences, so keep it on the server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muhawir import generate  # noqa: E402
from muhawir.generate import ModelGenerator, USAGE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("questions", type=Path)
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--style", default="youth")
    args = parser.parse_args()
    questions = [q.strip() for q in args.questions.read_text(encoding="utf-8").splitlines() if q.strip()]
    questions = questions[:args.max_questions]

    from muhawir.server import engine  # loads the real database and model settings from the environment
    generator = engine.generator
    if not isinstance(generator, ModelGenerator):
        print("needs the real model generator", file=sys.stderr)
        return 2

    recorded: list[tuple] = []
    original = ModelGenerator.check_support

    def recording(self, claims, passages, question="", *, memo=None):
        flags = original(self, claims, passages, question, memo=memo)
        if flags is not None and memo is None:  # a memo call reads its new sentences through a plain call: once only
            recorded.append((list(claims), dict(passages), question, list(flags)))
        return flags

    ModelGenerator.check_support = recording
    engine.store = None  # every question is answered afresh, not read from the answer cache (nothing is saved either)
    report, calls_alone, calls_together = [], 0, 0
    try:
        for question in questions:
            recorded.clear()
            before = USAGE["model_calls"]
            engine.ask(question, args.style, "ar", [])
            calls_alone += USAGE["model_calls"] - before
            differences = []
            for claims, passages, q, alone in list(recorded):
                generate.CHECK_GROUPS = True
                try:
                    before = USAGE["model_calls"]
                    together = original(generator, claims, passages, q)
                    calls_together += USAGE["model_calls"] - before
                finally:
                    generate.CHECK_GROUPS = False
                if together is None:
                    differences.append({"error": "the grouped reading could not run"})
                    continue
                for claim, a, t in zip(claims, alone, together):
                    if a != t:
                        differences.append({"sentence": claim.text, "passages": list(claim.passage_ids),
                                            "alone": a, "together": t,
                                            "risk": "accepted together, rejected alone" if t and not a else ""})
            report.append({"question": question, "differences": differences})
    finally:
        ModelGenerator.check_support = original
    risky = sum(1 for r in report for d in r["differences"] if d.get("risk"))
    json.dump({"questions": len(report), "accepted_together_rejected_alone": risky,
               "model_calls_answering": calls_alone, "model_calls_grouped_rereading": calls_together,
               "results": report}, sys.stdout, ensure_ascii=False, indent=1)
    print(f"\n{len(report)} questions; {risky} sentences accepted together but rejected alone", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
