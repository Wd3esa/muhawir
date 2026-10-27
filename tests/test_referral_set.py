"""The labelled referral set: the word list alone must never stop a question that should be answered."""
import importlib.util
from pathlib import Path

from muhawir import classify

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_referral.py"
spec = importlib.util.spec_from_file_location("eval_referral", SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_the_word_list_never_stops_a_question_that_should_be_answered():
    stopped = [q for q, want in ev.QUESTIONS
               if want == ev.ANSWER and classify.check(q).kind == classify.PERSONAL_CASE]
    assert stopped == []


def test_the_set_has_both_kinds_and_no_duplicates():
    questions = [q for q, _ in ev.QUESTIONS]
    assert len(set(questions)) == len(questions)
    assert {w for _, w in ev.QUESTIONS} == {ev.REFER, ev.ANSWER}
