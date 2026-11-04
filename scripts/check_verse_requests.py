"""Free check on the server's real mushaf: every way of asking for a verse that tests/test_verse_requests_matrix.py
covers on test data, run against the real database. No model is called and nothing is sent anywhere.

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

# (what a user types, the verse that must be shown); real verses, so the real mushaf decides
CASES = [
    ("انما يوفى الصابرون اجرهم اكمل الآية واذكر السورة", "q:39:10"),
    ("إنما يوفى الصابرون أجرهم أكمل الآية", "q:39:10"),
    ("انما يوفى الصابرون اجرهم في اي سورة", "q:39:10"),
    ("اكمل الآية: انما يوفى الصابرون اجرهم", "q:39:10"),
    ("ما تكملة آية انما يوفى الصابرون اجرهم", "q:39:10"),
    ("في اي سورة قوله تعالى انما يوفى الصابرون اجرهم", "q:39:10"),
    ("اذكر الآية «استعينوا بالصبر والصلاة»", "q:2:45"),
    ("اكمل الآية يا ايها الذين امنوا استعينوا بالصبر والصلاة", "q:2:153"),
    ("ابي الاية اللي فيها الله لا اله الا هو الحي القيوم اكملها", "q:2:255"),
    ("اكمل: شهر رمضان الذي انزل فيه القرآن", "q:2:185"),
    ("اكمل الآية وإذ قال إبراهيم رب اجعل هذا بلدا آمنا", "q:2:126"),
    ("في أي سورة: الرحمن علم القرآن", "q:55:1"),
    ("اذكر الآية فاغسلوا وجوهكم وأيديكم إلى المرافق", "q:5:6"),
    ("الآية الخامسة من سورة البقرة", "q:2:5"),
    ("نص الآية الآية الخامسة من سورة البقرة", "q:2:5"),
    ("آية ٢٥٥ من البقرة", "q:2:255"),
    ("البقرة 5", "q:2:5"),
    ("سورة الزمر (١٠)", "q:39:10"),
    ("الزمر: 10", "q:39:10"),
]
NOT_REQUESTS = ["ما معنى انما يوفى الصابرون اجرهم بغير حساب", "ما تفسير الآية 5 من البقرة", "ما حكم الصبر على البلاء",
                "في اي سورة ذكرت قصة يوسف", "اكمل الشرح من فضلك", "صلاة الفجر 2 ركعة"]


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
    for question, expected in CASES:
        verse = engine._asked_verse(question) or engine._verse_by_reference(question)
        got = verse.id if verse is not None else "nothing"
        ok = got == expected
        failed += not ok
        print(("✓ " if ok else "✗ ") + question + ("" if ok else f"   → expected {expected}, got {got}"))
    for question in NOT_REQUESTS:
        verse = engine._asked_verse(question) or engine._verse_by_reference(question)
        ok = verse is None
        failed += not ok
        print(("✓ " if ok else "✗ ") + "(not a request) " + question + ("" if ok else f"   → wrongly showed {verse.id}"))
    print(f"\n{len(CASES) + len(NOT_REQUESTS) - failed} of {len(CASES) + len(NOT_REQUESTS)} as expected")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
