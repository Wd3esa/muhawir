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


def test_english_and_urdu_questions_are_in_the_set_and_asked_in_their_language():
    items = json.loads(g.GOLD.read_text(encoding="utf-8"))["questions"]
    by = {lang: [i for i in items if i.get("lang", "ar") == lang] for lang in ("ar", "en", "ur")}
    assert len(by["en"]) >= 10 and len(by["ur"]) >= 10 and len(items) <= 80
    assert all(i.get("lang", "ar") in ("ar", "en", "ur") for i in items)
    # the Urdu questions are written in Urdu letters (one is Urdu typed in Latin letters, on purpose)
    assert sum(bool(g._URDU_LETTERS.search(i["question"])) for i in by["ur"]) >= 9
    assert next(i for i in items if i["id"] == "G55")["expect"] == "answer"  # counted by code: digits are expected


def test_the_answer_language_is_judged_by_its_letters():
    assert g.answer_lang_ok("الصلاة خمس مرات في اليوم", "ar")
    assert not g.answer_lang_ok("نماز دن میں پانچ بار ہے، روزہ", "ar")  # Urdu letters in an Arabic answer
    assert g.answer_lang_ok("نماز دن میں پانچ بار فرض ہے۔ ٹھیک", "ur")
    assert not g.answer_lang_ok("الصلاة خمس مرات في اليوم", "ur")  # Arabic spelling, no Urdu letters
    assert g.answer_lang_ok('Prayer is five times a day. «الصلاة»', "en")  # a quoted word does not make it Arabic
    assert not g.answer_lang_ok("Prayer is five times a day.", "ar")
    assert not g.answer_lang_ok("", "en")


def test_a_wrong_language_answer_is_counted_per_language():
    en = {"id": "G9", "field": "fiqh", "lang": "en", "key_points": [], "expect": "answer"}
    ok = g.score_row(en, _res("answered", "Wudu starts with washing the hands."), 1.0, None)
    bad = g.score_row(dict(en, id="G10"), _res("answered", "الوضوء يبدأ بغسل الكفين"), 1.0, None)
    assert ok["lang_ok"] and not bad["lang_ok"]
    summary = g.summarize([ok, bad, g.score_row({"id": "G1", "field": "fiqh", "key_points": [], "expect": "answer"},
                                                _res("answered", "الوضوء"), 1.0, None)])
    assert summary["lang:en"]["n"] == 2 and summary["lang:en"]["wrong_language"] == 1
    assert "lang:ar" not in summary and summary["ALL"]["n"] == 3
    assert "lang:en" in g.render(summary)


def test_an_unavailable_service_is_asked_again_after_a_wait():
    calls, waits = [], []

    def asker(url, question, lang):
        calls.append(lang)
        return ({"status": "unavailable"} if len(calls) < 3 else {"status": "answered"}), 1.0

    res, _, tries = g.ask_patiently("u", {"question": "q", "lang": "ur"}, retries=2, wait=10, sleep=waits.append, asker=asker)
    assert res["status"] == "answered" and tries == 3 and waits == [10, 20] and calls == ["ur"] * 3


def test_the_unavailable_result_is_kept_when_every_try_fails():
    res, _, tries = g.ask_patiently("u", {"question": "q"}, retries=1, wait=0, sleep=lambda s: None,
                                    asker=lambda *a: ({"status": "unavailable"}, 2.0))
    assert res["status"] == "unavailable" and tries == 2
