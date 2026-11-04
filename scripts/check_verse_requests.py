"""Free check on the server's real mushaf: the verses of tests/test_verse_requests_matrix.py found by code from the
pointers a correct understanding step returns (the user's quoted words, or surah and number). No model is called.
Whether the real model returns those pointers is measured separately by scripts/eval_understanding.py (paid, small).

    .venv/bin/python scripts/check_verse_requests.py

It prints how the stored mushaf spells a verse (Uthmani or plain), then one line per phrasing: ✓ when the expected
verse is shown, ✗ with what was found instead. Send the ✗ lines: each becomes a test first, then a fix.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muhawir.generate import ModelGenerator  # noqa: E402
from muhawir.pipeline import Muhawir  # noqa: E402
from muhawir.store import SqliteCorpus, SqliteRetriever  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from verse_phrasings import PHRASINGS  # noqa: E402


def main() -> int:
    db = Path(os.environ.get("MUHAWIR_DB") or ROOT / "data" / "muhawir.db")
    if not db.exists():
        print(f"no database at {db}", file=sys.stderr)
        return 2
    corpus = SqliteCorpus(db)
    engine = Muhawir(corpus, ModelGenerator([("unused", lambda *a, **k: "{}")]), SqliteRetriever(corpus))
    sample = corpus.passage("q:39:10")
    print("mushaf spelling (الزمر 10):", sample.text if sample else "missing")
    failed = 0
    for question, (_form, quoted, surah, ayah), expected in PHRASINGS:
        verse = engine._requested_verse({"quoted_text": quoted, "surah": surah, "ayah": ayah}, question)
        got = verse.id if verse is not None else "nothing"
        ok = got == expected
        failed += not ok
        print(("✓ " if ok else "✗ ") + question + ("" if ok else f"   → expected {expected}, got {got}"))
    print(f"\n{len(PHRASINGS) - failed} of {len(PHRASINGS)} as expected")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
