"""Measure the understanding step on the real model, for every phrasing in tests/test_verse_requests_matrix.py.

PAID but small: one understanding call per phrasing (about 36 short calls, about one cent). Run it only with the
owner's approval. Nothing else is called: no search by the model, no writing, no checking.

    .venv/bin/python scripts/eval_understanding.py

For each phrasing it shows whether the real model (1) asked for the verse text (form), and (2) pointed at the user's
own words or the place so that code finds the right verse in the server's real mushaf. Questions about a meaning must
not be taken for a request to show the bare verse. ✗ lines are what to fix, in the prompt, not in word lists.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from muhawir.generate import USAGE  # noqa: E402
from muhawir.server import engine  # noqa: E402  (the real database and model, as the service runs them)
from verse_phrasings import NOT_REQUESTS, PHRASINGS  # noqa: E402

# «I did not understand» in any words, after an answer: reexplain must be true (Muhawir explains again more clearly);
# a new question after an answer must not be taken for it
_ANSWER = [{"role": "user", "text": "ما حكم صلاة الوتر؟"},
           {"role": "assistant", "text": "صلاة الوتر سنة مؤكدة عند جمهور العلماء، ووقتها من بعد العشاء إلى طلوع الفجر."}]
REEXPLAIN = ["لم أفهم", "ما فهمت", "مافهمت شي", "ما وصلتني الفكرة", "مو واضح", "صعبة علي", "ممكن تبسطها", "وضح أكثر",
             "مش فاهم حاجة", "ما فهمتش", "I don't understand", "سمجھ نہیں آیا"]
NOT_REEXPLAIN = ["وما حكم صلاة الضحى؟", "شكرا", "متى وقتها بالضبط؟"]


def main() -> int:
    if os.environ.get("MUHAWIR_ALLOW_SYNTHETIC") == "1":
        print("runs on the real database and model only", file=sys.stderr)
        return 2
    understand = getattr(engine.generator, "understand", None)
    if understand is None:
        print("needs the model generator (LLM_PROVIDER=model)", file=sys.stderr)
        return 2
    failed, before = 0, USAGE.get("model_calls", 0)
    for question, _pointers, expected in PHRASINGS:
        u = understand(question, []) or {}
        request = u.get("answer_request") or {}
        verse = engine._requested_verse(request, question) if request.get("form") == "verse_text" else None
        got = verse.id if verse is not None else "nothing"
        ok = request.get("form") == "verse_text" and got == expected
        failed += not ok
        print(("✓ " if ok else "✗ ") + question + ("" if ok else
              f"\n    form={request.get('form')!r} quoted={request.get('quoted_text', '')!r} "
              f"place={request.get('surah', '')!r}:{request.get('ayah', 0)} → {got} (expected {expected})"))
    for question, _pointers in NOT_REQUESTS:
        u = understand(question, []) or {}
        form = (u.get("answer_request") or {}).get("form")
        ok = form != "verse_text"
        failed += not ok
        print(("✓ " if ok else "✗ ") + "(meaning, not the bare verse) " + question + ("" if ok else f"   → form={form!r}"))
    for message in REEXPLAIN:
        ok = bool((understand(message, _ANSWER) or {}).get("reexplain"))
        failed += not ok
        print(("✓ " if ok else "✗ ") + "(explain again) " + message)
    for message in NOT_REEXPLAIN:
        ok = not (understand(message, _ANSWER) or {}).get("reexplain")
        failed += not ok
        print(("✓ " if ok else "✗ ") + "(not «I did not understand») " + message)
    total = len(PHRASINGS) + len(NOT_REQUESTS) + len(REEXPLAIN) + len(NOT_REEXPLAIN)
    print(f"\n{total - failed} of {total} as expected; model calls: {USAGE.get('model_calls', 0) - before}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
