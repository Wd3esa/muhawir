"""Compare the second reading one sentence at a time (today) with Check-groups (MUHAWIR_CHECK_GROUPS=1) on the same
sentences, before switching Check-groups on. PAID: it answers each question with the real model, then reads the same
checked sentences again in groups. Run it only with the owner's approval and a question list sized to the budget.

    .venv/bin/python scripts/compare_check_groups.py --mode all > check-all.json      # built-in 20 questions
    .venv/bin/python scripts/compare_check_groups.py questions.txt --mode groups > check-groups.json

--mode all reads every sentence of an answer in ONE request (MUHAWIR_READ_EACH_SENTENCE=0); --mode groups reads the
sentences citing the same passages together (MUHAWIR_CHECK_GROUPS=1). Since the audit fix of November 2026, every
sentence accepted in either mode must still quote words that code finds in its cited passage.

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


# varied questions whose answers have several sentences: rulings, how-to, meanings, hadith, verses, creed, seerah
BUILTIN = (
    "كيف أتوضأ؟", "ما حكم صلاة الوتر؟", "ما شروط وجوب الزكاة؟", "ما مبطلات الصيام؟", "ما حكم الربا وما أنواعه؟",
    "ما أركان الحج؟", "ما حكم قراءة القرآن للحائض؟", "ما فضل الصبر؟", "ما معنى التوكل على الله؟",
    "ما صحة حديث «إنما الأعمال بالنيات»؟", "هل ذكرت صفة الوضوء في القرآن؟", "ما تفسير آية الكرسي؟",
    "ما معنى الإحسان؟", "ما أقسام التوحيد؟", "كيف كانت غزوة بدر؟", "ما حكم بيع التقسيط؟",
    "ما كفارة اليمين؟", "ما آداب الدعاء؟", "ما حكم صلاة الجماعة؟", "ما الفرق بين الفرض والواجب؟")


def _service_settings() -> None:
    """The settings the service runs with (~/muhawir.env), when this shell has not loaded them; nothing is printed."""
    import os
    path = Path(os.environ.get("MUHAWIR_ENV_FILE") or Path.home() / "muhawir.env")
    if os.environ.get("LLM_PROVIDER") or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("questions", type=Path, nargs="?", help="one question per line (default: a built-in set)")
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--style", default="youth")
    parser.add_argument("--mode", choices=("all", "groups"), default="all")
    args = parser.parse_args()
    questions = ([q.strip() for q in args.questions.read_text(encoding="utf-8").splitlines() if q.strip()]
                 if args.questions else list(BUILTIN))
    questions = questions[:args.max_questions]
    _service_settings()

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
                saved = generate.CHECK_GROUPS, generate.READ_EACH_SENTENCE
                if args.mode == "all":
                    generate.READ_EACH_SENTENCE = False
                else:
                    generate.CHECK_GROUPS = True
                try:
                    before = USAGE["model_calls"]
                    together = original(generator, claims, passages, q)
                    calls_together += USAGE["model_calls"] - before
                finally:
                    generate.CHECK_GROUPS, generate.READ_EACH_SENTENCE = saved
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
    print(f"\n{len(report)} questions; {risky} sentences accepted together but rejected alone; "
          f"calls to read them one by one vs together: {calls_alone} answering, {calls_together} re-reading", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
