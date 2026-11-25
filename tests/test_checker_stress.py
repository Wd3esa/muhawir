import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import checker_stress as S  # noqa: E402

from muhawir.corpus import Passage  # noqa: E402

FIQH = ("لا يجوز بيع الطعام قبل قبضه عند الحنفية إلا إذا كان مما لا يكال ولا يوزن، وهذا قول جمهور أهل العلم. "
        "وفيه تفصيل آخر.")


def _p(pid, source, text=FIQH):
    return Passage(id=pid, source_id=source, location="باب", text=text, kind="fiqh")


def test_the_faithful_sentence_is_the_passage_own_words():
    s = S.first_sentence(FIQH)
    assert s and s in FIQH and 8 <= len(s.split()) <= 40


def test_each_distortion_changes_the_meaning_in_its_own_way():
    s = S.first_sentence(FIQH)
    assert S.ruling_flip(s).startswith("يجوز بيع")  # «لا يجوز» → «يجوز»
    assert "الشافعية" in S.school_swap(s) and "الحنفية" not in S.school_swap(s)
    dropped = S.condition_dropped(s)
    assert dropped and "إلا" not in dropped and len(dropped.split()) >= 6
    assert S.negated(s).startswith("من الخطأ قول: «") and "إجماع" in S.added_consensus(s)
    assert S.number("صلاة الظهر 4 ركعات") == "صلاة الظهر 5 ركعات"
    assert S.number("له ثلاث ركعات") != "له ثلاث ركعات"
    assert S.ruling_flip("نص بلا حكم ظاهر فيه أبدا") is None  # not applicable: no case is made


def test_cases_expect_the_faithful_approved_and_every_wrong_one_rejected():
    cases = S.build_cases([_p("a", "x"), _p("b", "y")], per_type=1)
    types = {c["type"]: c for c in cases}
    assert types["faithful"]["expect"] == "approve"
    assert all(c["expect"] == "reject" for c in cases if c["type"] != "faithful")
    assert types["other_passage"]["passage"].source_id == "y"  # the true sentence against another book
    assert sum(c["type"] == "faithful" for c in cases) == 1


def test_run_and_summary_count_false_approvals():
    class Gen:
        last_check_reasons = []

        def check_support(self, claims, passages, question):
            ok = "خطأ" not in claims[0].text  # approves everything except the negated one
            self.last_check_reasons = ["" if ok else "the second reading found: نفي"]
            return [ok]

    rows = [S.run_case(Gen(), c) for c in S.build_cases([_p("a", "x"), _p("b", "y")], per_type=1)]
    summary = S.summarize(rows)
    assert summary["faithful"]["approve"] == 1 and summary["negated"]["reject"] == 1
    wrong = [r for r in rows if r["expect"] == "reject"]
    assert summary["false_approval_rate"] == round((len(wrong) - 1) / len(wrong), 3)
    assert "false approvals" in S.render(summary, "m")


def test_an_unreadable_check_is_counted_apart():
    class Gen:
        def check_support(self, *a):
            return None

    row = S.run_case(Gen(), {"type": "faithful", "expect": "approve", "text": "جملة", "passage": _p("a", "x")})
    assert row["outcome"] == "unusable"
