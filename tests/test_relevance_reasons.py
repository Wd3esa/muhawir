"""Reason capture is diagnostic, never a new model request or a new acceptance gate."""
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from muhawir import generate as g
from muhawir.verify import Claim


@pytest.mark.parametrize("verdict", ["yes", "partly", "no"])
def test_optional_reason_is_returned_in_the_same_call(verdict):
    seen = []
    def call(system, user, schema):
        seen.append(system)
        assert system == g.RELEVANCE_PROMPT
        assert "reason" in schema["properties"] and "reason" in system
        return json.dumps({"verdict": verdict, "reason": "  private diagnostic  "})
    review = g.ModelGenerator([("fake-reason", call)]).review_relevance("Question?", [Claim("Answer.", ())])
    assert review.verdict == verdict and review.reason == "private diagnostic"
    assert len(seen) == 1


@pytest.mark.parametrize("reason", [None, 1, [], {}, False])
def test_missing_or_bad_reason_does_not_invent_a_reason_or_trigger_a_retry(reason):
    seen = []
    def call(*args):
        seen.append(1)
        return json.dumps({"verdict": "no", "reason": reason})
    review = g.ModelGenerator([("fake-reason", call)]).review_relevance("Question?", [])
    assert review.verdict == "no" and review.reason == "" and len(seen) == 1


def test_legacy_replies_and_string_interface_remain_compatible():
    seen = []
    def call(*args):
        seen.append(1)
        return '{"verdict":"yes"}'
    generator = g.ModelGenerator([("fake-legacy-reason", call)])
    assert generator.judge_relevance("Question?", []) == "yes"
    assert len(seen) == 1


def test_reason_is_bounded_and_not_logged_or_reused_between_concurrent_requests(caplog):
    def call(system, user, schema):
        return json.dumps({"verdict": "partly", "reason": user + "PRIVATE" * 100})
    generator = g.ModelGenerator([("fake-concurrent-reason", call)])
    with ThreadPoolExecutor(max_workers=4) as pool:
        reviews = list(pool.map(lambda i: generator.review_relevance(f"Unique {i}?", []), range(12)))
    assert all(len(r.reason) == 240 for r in reviews)
    assert len({r.reason for r in reviews}) == 12
    assert "PRIVATE" not in caplog.text


def test_reason_is_not_added_to_the_childrens_clarity_contract():
    assert "reason" not in g.KIDS_REVIEW_SCHEMA["properties"]
    assert set(g.KIDS_REVIEW_SCHEMA["required"]) == {"verdict", "clarity", "feedback"}
