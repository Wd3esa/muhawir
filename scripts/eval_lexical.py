#!/usr/bin/env python3
"""Measure the dictionary lookup (muhawir/lexical.py) on a corpus database that holds «المفردات» (kind «gharib»).

    python scripts/eval_lexical.py data/muhawir.db

Prints, for «ما معنى X» questions about forms of words whose root is a heading of the dictionary: how often the entry
of the root is first / in the first three results; how many words that are in no book still get a passage that survives
the pipeline's coverage filter; and how often the gate keeps the true entry and an unrelated one.
Every case whose root is not a heading in the database is skipped, so the same list serves any edition."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muhawir import lexical  # noqa: E402
from muhawir.pipeline import HEADING_ALIGNMENT, MODEL_MIN_COVERAGE  # noqa: E402
from muhawir.store import SqliteCorpus, SqliteRetriever  # noqa: E402

CASES = [  # (a form as asked, the root heading of its entry)
    ("ألفوا", "لفى"), ("ألفينا", "لفى"), ("يؤفكون", "أفك"), ("استكبروا", "كبر"), ("رشدا", "رشد"), ("الرقيم", "رقم"),
    ("يتفكرون", "فكر"), ("أنبئكم", "نبأ"), ("ضالين", "ضل"), ("يصطفي", "صفو"), ("آتيناهم", "أتى"), ("أصحاب", "صحب"),
    ("فاستبقوا", "سبق"), ("يسبحون", "سبح"), ("قالوا", "قول"), ("مآب", "أوب"), ("آوى", "أوى"), ("نسوا", "نسي"),
    ("بعثناكم", "بعث"), ("ظلموا", "ظلم"), ("الرحمن", "رحم"), ("ساجدين", "سجد"), ("تشكرون", "شكر"),
    ("علمناه", "علم"), ("خلقناكم", "خلق"), ("اهتدوا", "هدى"), ("جاهلون", "جهل"), ("فاسقون", "فسق"), ("طغوا", "طغى"),
    ("آمنوا", "أمن"), ("أقرب", "قرب"), ("ينصركم", "نصر"), ("بيوتكم", "بيت"), ("المأوى", "أوى"), ("تفسدوا", "فسد"),
    ("يخرجون", "خرج"), ("كلمات", "كلم"), ("أحسنوا", "حسن"), ("الكرام", "كرم"), ("رقيب", "رقب"), ("خشيتم", "خشى"),
    ("وعاها", "وعي"), ("الكرب", "كرب"),
]
UNKNOWN = ["بلاغوزا", "كمبيوتر", "فيسبوك", "تلفزيون", "بطاطس", "جوال", "زقزفون", "مشمشاني", "ستيراكوتا", "برمجيات",
           "خوارزمية", "هيليكوبتر", "دردشة", "بيتزا"]


def main(path: str) -> None:
    corpus = SqliteCorpus(path)
    retriever = SqliteRetriever(corpus)
    entries = [corpus._row(r) for r in corpus.con.execute(
        "SELECT id, source_id, location, kind, grade, text, keywords, tokens, 0 FROM passages WHERE kind = 'gharib'")]
    headings = {p.keywords for p in entries}
    cases = [(w, r) for w, r in CASES if r in headings]
    print(f"{len(entries)} dictionary entries, {len(cases)} of {len(CASES)} cases usable")

    def results(question: str, gate: bool) -> list:
        hits = [h for h in retriever.search(question, k=20) if h.coverage >= MODEL_MIN_COVERAGE]
        if gate:
            hits = [h for h in hits if lexical.keeps(h.passage, question, retriever.df, heading_alignment=HEADING_ALIGNMENT)]
        return hits

    for frame in ("ما معنى {}", "ما معنى {} آباءهم"):
        t1 = t3 = 0
        for word, root in cases:
            found = [h.passage.keywords for h in results(frame.format(word), True)]
            t1 += root in found[:1]
            t3 += root in found[:3]
        print(f"{frame:22s} entry first {t1}/{len(cases)}   in first three {t3}/{len(cases)}")
    answered = sum(1 for w in UNKNOWN if results(f"ما معنى {w}", True))
    print(f"words in no book that still get a passage: {answered}/{len(UNKNOWN)}")

    random.seed(7)
    engaged = kept = wrong = tried = 0
    for word, root in cases:
        question = f"ما معنى {word}"
        if not any((t := lexical.tokenize(w)) and retriever.df(t[0]) <= lexical.RARE_DF for w in lexical.raw_terms(question)):
            continue
        engaged += 1
        kept += any(lexical.keeps(p, question, retriever.df, heading_alignment=HEADING_ALIGNMENT)
                    for p in entries if p.keywords == root)
        others = random.sample([p for p in entries if p.keywords != root], min(300, len(entries) - 1))
        wrong += sum(lexical.keeps(p, question, retriever.df, heading_alignment=HEADING_ALIGNMENT) for p in others)
        tried += len(others)
    if engaged:
        print(f"gate on rare words ({engaged} questions): true entry kept {kept}/{engaged}, "
              f"unrelated entries kept {wrong / tried:.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
