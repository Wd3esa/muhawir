"""Children's clarity review shares the relevance call and the single rewrite budget.

Fake model replies verify orchestration and evidence checks, not actual child comprehension.
"""
import json
from pathlib import Path

import pytest

from muhawir.corpus import load_corpus
from muhawir.generate import (AnswerReview, KIDS_REVIEW_PROMPT, ModelGenerator,
                              RELEVANCE_PROMPT, USAGE)
from muhawir.messages import TEXT
from muhawir.pipeline import Muhawir, Response, UNAVAILABLE
from muhawir.verify import Claim

CORPUS = load_corpus(Path(__file__).resolve().parent.parent / "data/synthetic_corpus.json")
QUESTION = "ماذا تحتاج النخلة في الصيف؟"
SIMPLE = {"text": "تحتاج النخلة إلى ماء كثير في الصيف.", "passage_ids": ["test-a:1"]}
LONG = [SIMPLE, {"text": "وتثمر التمر في آخره.", "passage_ids": ["test-a:1"]}]


def _model(drafts, reviews):
    drafts, reviews = iter(drafts), iter(reviews)
    seen = {"drafts": [], "reviews": [], "other": 0}

    def call(system, user, schema=None):
        if "verdict" in schema.get("properties", {}):
            seen["reviews"].append((system, user))
            reply = next(reviews)
            if isinstance(reply, Exception):
                raise reply
            return json.dumps(reply, ensure_ascii=False)
        if "absent" in schema.get("properties", {}):
            return '{"absent": []}'
        if "claims" not in schema.get("properties", {}):
            seen["other"] += 1
            return '{"queries": []}'
        seen["drafts"].append(user)
        return json.dumps({"abstain": False, "claims": next(drafts)}, ensure_ascii=False)

    gen = ModelGenerator([("fake", call)])
    return Muhawir(CORPUS, gen), seen


def _write(engine, style="kids", lang="ar"):
    return engine._write(QUESTION, [CORPUS.passage("test-a:1")], style, lang, False, {})


def _review(clarity="clear", verdict="yes", feedback=""):
    return {"verdict": verdict, "clarity": clarity, "feedback": feedback}


def test_clear_kids_answer_uses_one_combined_judge_and_no_rewrite():
    engine, seen = _model([[SIMPLE]], [_review()])
    before = USAGE["attempt_relevance"]
    result = _write(engine)
    assert result.kept[0].text == SIMPLE["text"]
    assert len(seen["drafts"]) == len(seen["reviews"]) == 1
    assert seen["reviews"][0][0] == KIDS_REVIEW_PROMPT
    assert USAGE["attempt_relevance"] == before + 1


def test_unclear_answer_is_replaced_by_shorter_checked_answer_with_specific_feedback():
    feedback = "كرر المعنى في جملتين؛ اشرح المعنى مرة واحدة بكلمات يومية."
    engine, seen = _model([LONG, [SIMPLE]], [_review("rewrite", feedback=feedback), _review()])
    result = _write(engine)
    assert [c.text for c in result.kept] == [SIMPLE["text"]]
    assert len(seen["drafts"]) == len(seen["reviews"]) == 2
    assert feedback in seen["drafts"][1]
    assert result.kept[0].passage_ids == ("test-a:1",)


def test_coverage_and_clarity_share_one_rewrite():
    engine, seen = _model([[SIMPLE], LONG], [
        _review("rewrite", "partly", "بسّط المصطلح قبل استعماله."), _review()
    ])
    result = _write(engine)
    assert not result.partial and len(result.kept) == 2
    assert "لا يغطي السؤال كله" in seen["drafts"][1]
    assert "بسّط المصطلح" in seen["drafts"][1]
    assert len(seen["drafts"]) == 2


@pytest.mark.parametrize("lang", ["ar", "en", "ur"])
def test_still_unclear_does_not_loop_or_claim_that_sources_are_missing(lang):
    bad = _review("rewrite", feedback="التعريف ما زال يكرر المعنى بدل شرحه.")
    engine, seen = _model([LONG, [SIMPLE]], [bad, bad])
    result = _write(engine, lang=lang)
    assert isinstance(result, Response) and result.status == UNAVAILABLE
    assert result.message == TEXT[lang]["simplify_unavailable"]
    assert not result.claims and not result.sources
    assert len(seen["drafts"]) == 2 and len(seen["reviews"]) == 2
    assert seen["other"] == 0


def test_shorter_rewrite_still_must_pass_citation_checks():
    unsupported = {"text": SIMPLE["text"], "passage_ids": ["not-retrieved"]}
    engine, seen = _model([LONG, [unsupported]], [_review("rewrite", feedback="احذف التكرار.")])
    result = _write(engine)
    assert isinstance(result, Response) and result.status == UNAVAILABLE
    assert result.message == TEXT["ar"]["simplify_unavailable"]
    assert len(seen["drafts"]) == 2 and len(seen["reviews"]) == 1


def test_known_unclear_answer_is_not_cleared_by_an_unavailable_second_review():
    engine, seen = _model([LONG, [SIMPLE]], [
        _review("rewrite", feedback="احذف التكرار."), {"verdict": "yes"}
    ])
    result = _write(engine)
    assert isinstance(result, Response) and result.message == TEXT["ar"]["simplify_unavailable"]
    assert len(seen["drafts"]) == len(seen["reviews"]) == 2


def test_off_topic_first_draft_is_not_rewritten_for_clarity():
    engine, seen = _model([LONG], [_review("rewrite", "no", "المعنى غير واضح.")])
    result = _write(engine)
    assert result.off_topic and not result.kept
    assert len(seen["drafts"]) == len(seen["reviews"]) == 1


def test_off_topic_rewrite_cannot_replace_the_answer_as_a_clear_one():
    engine, seen = _model([LONG, [SIMPLE]], [
        _review("rewrite", feedback="احذف التكرار."), _review(verdict="no")
    ])
    result = _write(engine)
    assert result.off_topic and not result.kept
    assert len(seen["drafts"]) == 2


def test_youth_uses_original_judge_and_does_not_request_clarity():
    engine, seen = _model([[SIMPLE]], [{"verdict": "yes"}])
    assert _write(engine, style="youth").kept
    assert seen["reviews"][0][0] == RELEVANCE_PROMPT
    assert len(seen["reviews"]) == len(seen["drafts"]) == 1


def test_missing_clarity_report_does_not_invent_rewrite_feedback():
    engine, seen = _model([[SIMPLE]], [_review("rewrite")])
    # An invalid clarity rejection without reasons is unknown, like an unavailable relevance judge.
    assert _write(engine).kept
    assert len(seen["drafts"]) == 1


def test_early_review_is_discarded_when_support_check_removes_a_sentence(monkeypatch):
    engine, seen = _model([LONG], [_review(), _review()])
    monkeypatch.setattr(engine.generator, "check_support", lambda *args: [True, False])
    # Prevent the support-repair branch here to isolate invalidation of the concurrent review.
    monkeypatch.setattr(engine, "_broken", lambda *args: False)
    assert len(_write(engine).kept) == 1
    assert len(seen["reviews"]) == 2
    assert "وتثمر" in seen["reviews"][0][1]
    assert "وتثمر" not in seen["reviews"][1][1]


def test_support_rewrite_already_uses_the_only_rewrite_budget(monkeypatch):
    engine, seen = _model([LONG, [SIMPLE]], [_review(), _review("rewrite", feedback="بسّط الشرح.")])
    checks = iter([[False, True], [True]])
    monkeypatch.setattr(engine.generator, "check_support", lambda *args: next(checks))
    result = _write(engine)
    assert isinstance(result, Response) and result.status == UNAVAILABLE
    assert len(seen["drafts"]) == 2 and len(seen["reviews"]) == 2


def test_combined_judge_returns_a_per_answer_value_not_thread_local_state():
    engine, _ = _model([], [_review("rewrite", feedback="اشرح المصطلح."), _review()])
    claims = [Claim(SIMPLE["text"], ("test-a:1",))]
    first = engine.generator.judge_kids_answer(QUESTION, claims)
    second = engine.generator.judge_kids_answer(QUESTION, claims)
    assert first == AnswerReview("yes", "rewrite", "اشرح المصطلح.")
    assert second == AnswerReview("yes", "clear", "")
