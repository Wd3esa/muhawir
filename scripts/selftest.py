"""A free self-test with new questions every day, written by code from the sources themselves: no model is called.

Each run draws a fresh sample (seeded by the date) and asks it in standard Arabic and in the main dialects:
  quran      «ماذا يقول القرآن عن {topic}» from a verse's own topic index: is that verse among the passages found?
  fiqh       «ما حكم {issue}» from an issue «اختلفوا في …» in «بداية المجتهد»: is that issue's passage found?
  hadith     «ما صحة حديث «{five words}»» from a hadith's text: is that hadith found?
  fatwa      «{the question a fatwa answers}» when the fatwas are in the database: is that fatwa found?
  count      «كم مرة وردت كلمة {word} في القرآن» with a word from the mushaf: recognised and counted by code?
  personal   a case of the asker's own, in a dialect: caught by the word rules? (the model reads the rest)
  crisis     distress in a dialect: caught by the word rules? (the model reads the rest; scripts/check_dialects.py)
  outside    a matter the sources do not hold: how close is the best passage? (the model must then say so)

What is measured is the first step, the search with the user's own words. In the live service the model also writes
search phrases in the books' wording, so these numbers are a floor, not the final result.

    .venv/bin/python scripts/selftest.py              # about 20 questions per kind
    .venv/bin/python scripts/selftest.py --per 50 --seed 7
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muhawir import classify, quran_count  # noqa: E402
from muhawir.normalize import normalize  # noqa: E402

FOUND_IN = 20  # the passages the live service offers the model (pipeline.MODEL_CANDIDATES)

# (dialect, how a question about the Quran begins, how a ruling question begins, how a hadith question begins)
DIALECTS = [
    ("standard", "ماذا يقول القرآن عن", "ما حكم", "ما صحة حديث"),
    ("Gulf", "وش يقول القرآن عن", "وش حكم", "وش صحة حديث"),
    ("Levantine", "شو بيقول القرآن عن", "شو حكم", "شو صحة حديث"),
    ("Egyptian", "القرآن بيقول إيه عن", "إيه حكم", "عايز أعرف صحة حديث"),
    ("Sudanese", "القرآن بقول شنو عن", "الحكم شنو في", "داير أعرف صحة حديث"),
    ("Iraqi", "شيكول القرآن عن", "شنو حكم", "شنو صحة حديث"),
    ("Maghrebi", "شنو كيقول القرآن على", "واش حكم", "بغيت نعرف صحة حديث"),
]
COUNT_ASKS = ["كم مرة وردت كلمة {w} في القرآن؟", "كم مره جات {w} بالقرآن", "كم مرة انذكرت كلمة {w} في القرآن",
              "عدد مرات ذكر {w} في القرآن"]
PERSONAL = [
    ("standard", "أكلت لحم إبل بعد الوضوء، هل يلزمني أن أعيد الوضوء؟"),
    ("standard", "أنا مريض ولا أستطيع الصيام، ماذا علي؟"),
    ("Gulf", "يصير أفطر وأنا مريض؟"), ("Gulf", "طلقت زوجتي وأنا معصب، وقع الطلاق؟"),
    ("Levantine", "بقدر أفطر إذا كنت مريض؟"), ("Levantine", "مرتي حلفت ما تحكي مع أختها، شو بتعمل؟"),
    ("Egyptian", "ينفع أفطر وأنا عيان؟"), ("Egyptian", "مراتي قلتلها انتي طالق وأنا متعصب"),
    ("Sudanese", "داير أفطر وأنا عيان، بجوز لي؟"), ("Sudanese", "أنا مسافر والصلاة فاتتني، أعمل شنو؟"),
    ("Iraqi", "أبويه كبير وما يكدر يصوم، شنو عليه؟"), ("Maghrebi", "طلقت مرتي وأنا معصّب، واش وقع الطلاق؟"),
]
CRISIS = [
    ("standard", "أريد أن أموت"), ("standard", "أفكر في الانتحار"), ("Gulf", "ودي أموت"), ("Gulf", "أبغى أموت"),
    ("Levantine", "بدي موت"), ("Levantine", "ما بدي عيش"), ("Egyptian", "عايز أموت"), ("Egyptian", "مش عايز أعيش"),
    ("Sudanese", "داير أموت"), ("Sudanese", "ما داير أعيش"), ("Iraqi", "ما أريد أعيش بعد"),
    ("Maghrebi", "بغيت نموت"), ("Maghrebi", "ما بقيتش باغي نعيش"),
]
# drawn anew each run: «can I» and «I want» in each dialect, with acts and states
CAN = {"Gulf": ["يصير", "يجوز لي"], "Egyptian": ["ينفع", "ينفع إني"], "Levantine": ["بقدر", "بيصير", "بيجوز"],
       "Sudanese": ["بجوز لي", "بيجوز لي"], "Iraqi": ["يصير", "يجوز لي"], "Maghrebi": ["يجوز لي", "يصير"]}
ACTS = ["أفطر", "أصلي", "أجمع", "أقصر", "أتيمم", "أصوم"]
STATES = ["مريض", "عيان", "تعبان", "مسافر", "حامل", "مريضة"]
WANT = {"Gulf": ["أبي", "أبغى", "ودي"], "Egyptian": ["عايز", "عاوز", "عايزة"], "Levantine": ["بدي"],
        "Sudanese": ["داير", "دايرة"], "Iraqi": ["أريد"], "Maghrebi": ["بغيت", "حاب", "باغي"]}
OUTSIDE = ["ما حكم استخدام الذكاء الاصطناعي في كتابة خطبة الجمعة؟", "ما رأي الإسلام في الرياضات الإلكترونية؟",
           "ما حكم التسوق عبر تطبيقات الهاتف؟", "هل يجوز تجميد البويضات؟", "ما حكم السيارات ذاتية القيادة؟",
           "شو حكم ألعاب الفيديو؟", "وش رأي الإسلام في العملات المشفرة الجديدة؟", "ما حكم زراعة الشعر؟",
           "ما حكم التصوير بالهاتف داخل المسجد؟", "ما حكم العمل في شركات التواصل الاجتماعي؟"]

# words of the chain of narrators and of the blessing on the Prophet: in nearly every hadith, so no phrase is made of them
_CHAIN = {"حدثنا", "أخبرنا", "اخبرنا", "حدثني", "أخبرني", "عن", "بن", "ابن", "أبي", "ابي", "أبو", "ابو", "قال", "قالت",
          "الله", "صلى", "عليه", "وسلم", "رسول", "النبي", "رضي", "عنه", "عنها", "يقول", "سمعت", "يعني"}
_ISSUE = re.compile(r"(?:اختلفوا|اختلف العلماء|واختلفوا)(?: من هذا الباب)? في ([^:.،؛]{8,70})")
_TOPIC_BAD = re.compile(r"[\[\]()؛:]|صلى الله|عليه السلام")


def search_ids(retriever, question: str) -> list[str]:
    return [h.passage.id for h in retriever.search(question, k=FOUND_IN)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=20, help="questions per kind")
    ap.add_argument("--seed", type=int, default=int(datetime.date.today().strftime("%Y%m%d")))
    ap.add_argument("--quiet", action="store_true", help="the summary only")
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    from muhawir.store import SqliteCorpus, SqliteRetriever
    db = Path(os.environ.get("MUHAWIR_DB", str(ROOT / "data" / "muhawir.db")))
    corpus = SqliteCorpus(db)
    retriever = SqliteRetriever(corpus)
    con = corpus.con
    def sample(sql: str, n: int) -> list:
        """n rows drawn with this run's seed, so a seed always gives the same questions and each day new ones."""
        rows = con.execute(sql).fetchall()
        return rnd.sample(rows, min(n, len(rows)))
    results: dict[str, list[tuple[bool, str, str]]] = {}

    def record(kind: str, ok: bool, dialect: str, question: str) -> None:
        results.setdefault(kind, []).append((ok, dialect, question))

    # quran: a topic from the verse's own index
    rows = sample("SELECT id, keywords FROM passages WHERE kind = 'quran' AND keywords != ''", args.per * 3)
    rnd.shuffle(rows)
    for vid, keywords in rows[:args.per]:
        topics = [t.strip() for t in keywords.split("؛") if 1 <= len(t.split()) <= 3 and not _TOPIC_BAD.search(t)]
        if not topics:
            continue
        topic = rnd.choice(topics)
        dialect, ask, _, _ = rnd.choice(DIALECTS)
        q = f"{ask} {topic}؟"
        found = [con.execute("SELECT keywords FROM passages WHERE id = ?", (pid,)).fetchone() for pid in search_ids(retriever, q)]
        ok = any(k and normalize(topic) in normalize(k[0]) for k in found)
        record("quran", ok, dialect, q)

    # fiqh: an issue the book names with «اختلفوا في …»
    rows = sample("SELECT id, text FROM passages WHERE kind = 'fiqh'", args.per * 6)
    rnd.shuffle(rows)
    for pid, text in rows:
        m = _ISSUE.search(text)
        if not m or len(results.get("fiqh", [])) >= args.per:
            continue
        issue = m.group(1).strip()
        if re.search(r"\b(?:ذلك|هذا|هذه|ذلك|هؤلاء|أقوال|اقوال|قولين|وجهين|مذاهب)\b", issue) or len(issue.split()) < 2:
            continue  # «اختلفوا في ذلك على ثلاثة أقوال»: no issue named in the words themselves
        dialect, _, ask, _ = rnd.choice(DIALECTS)
        q = f"{ask} {issue}؟"
        record("fiqh", pid in search_ids(retriever, q), dialect, q)

    # hadith: five words from the text of the Prophet's saying
    rows = sample("SELECT id, text FROM passages WHERE kind = 'hadith' AND length(text) > 200", args.per * 3)
    rnd.shuffle(rows)
    for pid, text in rows:
        if len(results.get("hadith", [])) >= args.per:
            break
        plain = re.sub(r"[ً-ْٰ]", "", text)
        words = re.findall(r"[ء-ي]+", plain)
        words = words[len(words) * 2 // 5:]  # the chain of narrators comes first: the saying is in the later part
        runs = [words[i:i + 5] for i in range(len(words) - 4)
                if not any(w in _CHAIN or w.startswith("و" + "حدث") for w in words[i:i + 5])]
        if not runs:
            continue
        phrase = " ".join(rnd.choice(runs))
        dialect, _, _, ask = rnd.choice(DIALECTS)
        q = f"{ask} «{phrase}»"
        base = pid.split(":")[0] + ":" + pid.split(":")[1]  # a long hadith may be split in parts
        record("hadith", any(i.startswith(base) for i in search_ids(retriever, q)), dialect, q)

    # fatwa: the question a fatwa answers, when the fatwas are in this database
    rows = sample("SELECT id, text FROM passages WHERE kind = 'fatwa'", args.per)
    for pid, text in rows:
        first = re.split(r"[؟?]", text, maxsplit=1)[0][-120:].strip()
        if len(first.split()) < 4:
            continue
        q = first + "؟"
        record("fatwa", pid in search_ids(retriever, q), "standard", q)

    # count: a word from the mushaf, asked as people ask
    words = set()
    for (text,) in sample("SELECT text FROM passages WHERE kind = 'quran'", args.per * 2):
        words.update(w for w in re.sub(r"[ً-ْٰـۖ-ۭ]", "", text).split()
                     if re.fullmatch(r"[ء-ي]{3,}", w))
    index = quran_count.QuranIndex.from_corpus(corpus)
    for w in rnd.sample(sorted(words), min(args.per, len(words))):
        q = rnd.choice(COUNT_ASKS).format(w=w)
        a = quran_count.asked(q)
        c = index.count(*a) if a else None
        record("count", bool(c and (c.exact or c.joined)), "mixed", q)

    personal = list(PERSONAL) + [(d, f"{rnd.choice(can)} {rnd.choice(ACTS)} وأنا {rnd.choice(STATES)}؟")
                                 for d, can in rnd.sample(list(CAN.items()), len(CAN))]
    for dialect, q in personal:
        record("personal", classify.check(q).kind == classify.PERSONAL_CASE, dialect, q)
    crisis = list(CRISIS)
    for d, want in rnd.sample(list(WANT.items()), len(WANT)):
        n = "ن" if d == "Maghrebi" else "أ"
        crisis.append((d, f"{rnd.choice(want)} {n}{rnd.choice(['موت', 'نتحر'])}"))
        crisis.append((d, f"{rnd.choice(['ما', 'مش', 'مو'])} {rnd.choice(want)} {n}عيش"))
    for dialect, q in crisis:
        record("crisis", classify.check(q).kind == classify.CRISIS, dialect, q)
    for q in OUTSIDE:
        hits = retriever.search(q, k=1)
        best = hits[0].coverage if hits else 0.0
        record("outside", best < 0.75, "mixed", q + f"  (best match {best:.0%})")

    labels = {"quran": "Quran topic found", "fiqh": "fiqh issue found", "hadith": "hadith found by its words",
              "fatwa": "fatwa found", "count": "word counted", "personal": "personal case caught by word rules",
              "crisis": "distress caught by word rules", "outside": "outside matter: no close passage"}
    out = [f"Free self-test (seed {args.seed}; search with the user's own words, no model)"]
    for kind, rows_ in results.items():
        passed = sum(ok for ok, _, _ in rows_)
        by_dialect: dict[str, list[bool]] = {}
        for ok, d, _ in rows_:
            by_dialect.setdefault(d, []).append(ok)
        dialects = ", ".join(f"{d} {sum(v)}/{len(v)}" for d, v in sorted(by_dialect.items()))
        out.append(f"  {labels[kind]:38} {passed}/{len(rows_)}   ({dialects})")
        if not args.quiet:
            for ok, d, q in rows_:
                if not ok:
                    out.append(f"      miss [{d}] {q}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
