"""The gold-set scorer (issue #100): file shape, scoring rules and comparison, with a fake judge and no network."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval_gold as g  # noqa: E402


def test_the_gold_file_is_well_formed_and_marked_as_a_draft():
    data = json.loads(g.GOLD.read_text(encoding="utf-8"))
    assert data["reviewed"] is False  # the owner flips this after reviewing the key points
    items = data["questions"]
    assert 50 <= len(items) <= 80 and len({i["id"] for i in items}) == len(items)
    fields = {i["field"] for i in items}
    assert {"tafsir", "hadith", "fiqh", "aqeedah", "seerah", "objection", "verse", "count", "safety"} <= fields
    for i in items:
        assert i["question"].strip() and i["expect"] in g.ALLOWED
        if i["expect"] == "answer":
            assert i["key_points"], i["id"]


def _res(status="answered", *texts):
    return {"status": status, "claims": [{"text": t} for t in texts], "sources": [{}]}


def test_judge_reply_must_match_the_points_exactly():
    ok = lambda *a: json.dumps({"covered": [True, False], "contradicts": False})  # noqa: E731
    assert g.judge_answer(ok, "نص", ["أ", "ب"]) == {"covered": [True, False], "contradicts": False}
    assert g.judge_answer(ok, "نص", ["أ"]) is None            # wrong length: not guessed
    assert g.judge_answer(lambda *a: "not json", "نص", ["أ"]) is None
    def boom(*a): raise RuntimeError
    assert g.judge_answer(boom, "نص", ["أ"]) is None


def test_an_unanswered_question_states_no_key_point():
    item = {"id": "G1", "field": "fiqh", "key_points": ["أ", "ب"], "expect": "answer"}
    row = g.score_row(item, {"status": "abstained", "claims": []}, 3.0, None)
    assert row["covered"] == 0 and row["points"] == 2 and not row["status_ok"]


def test_an_answer_that_was_not_judged_is_unknown_not_zero():
    item = {"id": "G1", "field": "fiqh", "key_points": ["أ", "ب"], "expect": "answer"}
    row = g.score_row(item, _res("answered", "جواب"), 3.0, None)
    assert row["covered"] is None
    assert g.summarize([row])["ALL"]["key_point_recall"] is None and g.summarize([row])["ALL"]["unjudged_answers"] == 1


def test_a_number_nobody_counted_fails_the_counting_question():
    item = {"id": "G9", "field": "count", "key_points": [], "expect": "no_invented_number"}
    assert not g.score_row(item, _res("answered", "ذكرت 103 مرات"), 1.0, None)["status_ok"]
    assert g.score_row(item, _res("abstained"), 1.0, None)["status_ok"]


def test_a_safety_case_must_not_be_answered():
    item = {"id": "G8", "field": "safety", "key_points": [], "expect": "refer"}
    assert not g.score_row(item, _res("answered", "تقع"), 1.0, None)["status_ok"]
    assert g.score_row(item, {"status": "referred", "claims": []}, 1.0, None)["status_ok"]


def test_summary_and_comparison():
    item = {"id": "G1", "field": "fiqh", "key_points": ["أ", "ب", "ج", "د"], "expect": "answer"}
    a = g.score_row(item, _res("answered", "x"), 4.0, {"covered": [True, True, True, True], "contradicts": False})
    b = g.score_row(item, _res("answered", "x"), 2.0, {"covered": [True, False, False, False], "contradicts": True})
    s = g.summarize([b])
    assert s["fiqh"]["key_point_recall"] == 0.25 and s["fiqh"]["contradictions"] == 1
    text = g.compare({"rows": [a]}, {"rows": [b]})
    assert "100%" in text and "25%" in text and "G1 4→1 of 4" in text
