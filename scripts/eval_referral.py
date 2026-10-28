"""Which questions does Muhawir refer to a specialist? A labelled set, checked against the understanding step.

The cases rest on the books of adab al-fatwa (read from the Shamela text, 28 October 2026): ibn as-Salah, Adab al-Mufti
wa-l-Mustafti pp. 115-116 and ibn Hamdan, Sifat al-Mufti p. 198 (oaths, confessions and wording: only one who knows the
speaker's usage); ibn al-Qayyim, I'lam al-Muwaqqi'in 1/25-26 and 6/92 (the fact must be known; ask before ruling on a
breach of an oath); an-Nawawi, Adab al-Fatwa pp. 45-46 (a ruling with kinds: give each kind its ruling) and pp. 44-45,
63-65 (what one does not know: hold back, and say so). Nothing in them lists inheritance disputes or contemporary
matters; those two stay by the project owner's decision (27 October 2026), and are marked so below.

The rule (decision of the project owner, 27 October 2026): a question is referred only when its answer depends on
facts Muhawir cannot check and that change the ruling: a dispute over a right or an inheritance between particular
people, whether a divorce took place by particular words, a contemporary matter with the asker's private facts, or
a question that says its own details change the answer. A first-person question about worship, purity, fasting, zakat,
an oath's expiation, a missed prayer, the sick, the traveller or the aged has a known ruling and is answered.

Each question below carries the answer it should get ("refer" or "answer"). The script asks the same understanding
step the server uses (one short model call per question, 32 in all; the referral is read from the meaning, with no
word list since 28 October 2026), then prints every
disagreement. It changes nothing and writes no file.

Usage, in ~/muhawir with the service's environment loaded:
    set -a; source ~/muhawir.env; set +a
    python scripts/eval_referral.py --dry     # lists the questions, calls nothing
    python scripts/eval_referral.py           # one call per question
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


REFER, ANSWER = "refer", "answer"

QUESTIONS = [
    # first person, a known ruling: answered
    ("فاتتني صلاة الصبح وصحيت الساعة ١٠ ص اقرأ سرًا او جهراً", ANSWER),
    ("نسيت الوتر البارحة كيف أقضيه؟", ANSWER),
    ("أكلت لحم جمل وصليت، صلاتي صحيحة؟", ANSWER),
    ("حلفت على أمر ثم حنثت، هل تلزمني كفارة وما هي؟", ANSWER),
    ("نذرت أن أصوم يومًا ولم أفِ بنذري، ماذا أفعل؟", ANSWER),
    ("أنا مريض وسأجري عملية غدًا، هل يلزمني الصيام؟", ANSWER),
    ("أبوي كبير وما يقدر يصوم شنو عليه؟", ANSWER),
    ("أنا مسافر، هل أقصر الصلاة؟", ANSWER),
    ("أنا حائض هل أقرأ القرآن؟", ANSWER),
    ("هل علي زكاة في الذهب الذي ألبسه؟", ANSWER),
    ("ركبتي تؤلمني، هل يجوز لي أن أصلي جالسًا؟", ANSWER),
    ("أكلت وأنا صائم ناسيًا، هل يبطل صومي؟", ANSWER),
    ("أفطرت يومًا في رمضان عمدًا، ماذا يلزمني؟", ANSWER),
    ("بقدر أفطر إذا كنت مريض؟", ANSWER),
    ("Can I combine prayers while I am travelling?", ANSWER),
    # general: answered
    ("كيف أتوضأ؟", ANSWER),
    ("ما نصيب البنت من تركة أبيها؟", ANSWER),
    ("ما حكم قول الرجل لزوجته أنت طالق؟", ANSWER),
    ("ما أنواع الطلاق؟", ANSWER),
    ("حلفت ألا أكلم أخي ثم كلمته ناسيًا، هل علي كفارة؟", ANSWER),  # kinds with a ruling each: an-Nawawi pp. 45-46
    ("هل يجوز للوارث أن يأخذ من التركة قبل القسمة؟", ANSWER),
    # a dispute over a right or an inheritance between particular people: referred (the owner's decision, not from the books)
    ("أخي استولى على نصيبي من تركة أبينا، ماذا أفعل؟", REFER),
    ("توفي زوجي وترك أولادًا وأخوين وبيننا خلاف على قسمة البيت، كيف نقسمه؟", REFER),
    ("شريكي أخذ مال الشركة ويرفض القسمة، ما الحكم؟", REFER),
    ("طليقتي ترفض تسليمي ابني بعد انتهاء حضانتها، ماذا أفعل؟", REFER),
    ("اقترضت من صديقي ويقول إنني لم أسدد وليس عندي إثبات، ماذا أفعل؟", REFER),
    # whether a divorce took place by particular words, or what a particular oath wording binds: referred (ibn as-Salah)
    ("حلفت بالطلاق على أمر وبصيغة معينة ثم فعلته، هل وقع الطلاق؟", REFER),
    ("قلت لزوجتي في الغضب أنتِ طالق ثلاثًا، هل وقع الطلاق؟", REFER),
    ("مراتي قلتلها انتي طالق وانا متعصب", REFER),
    # a contemporary matter with the asker's private facts: referred
    ("عرض عليّ البنك تمويلًا عقاريًا بصيغة التورق المنظم وهذه شروط عقده، هل أوقّع عليه؟", REFER),
    ("أعمل في شركة تأمين وعقدي فيه بند كذا وكذا، هل عملي جائز؟", REFER),
    # the asker says that their own details change the answer: referred
    ("في حالتي الخاصة هل يجوز لي أن أفطر؟", REFER),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry", action="store_true", help="list the questions and call nothing")
    args = ap.parse_args()
    if args.dry:
        for q, want in QUESTIONS:
            print(f"{want:6} {q}")
        print(f"\n{len(QUESTIONS)} questions; no model was called")
        return 0
    from muhawir.generate import get_generator
    generator = get_generator()
    if not hasattr(generator, "understand"):
        sys.exit("the model is not configured: load ~/muhawir.env first (LLM_PROVIDER=model)")
    wrong, calls = [], 0
    for q, want in QUESTIONS:
        calls += 1
        got = REFER if (generator.understand(q, []) or {}).get("personal") else ANSWER
        how = "model"
        print(("ok  " if got == want else "MISS"), how, want, "<-", got, "|", q[:70])
        if got != want:
            wrong.append((q, want, got))
    print(f"\n{len(QUESTIONS) - len(wrong)} of {len(QUESTIONS)} as expected; {calls} model calls")
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
