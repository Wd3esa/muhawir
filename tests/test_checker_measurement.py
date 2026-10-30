"""Diagnostic transport/accounting tests. All model replies below are synthetic."""
from dataclasses import asdict
from decimal import Decimal
import json

import pytest

from muhawir import generate as g, joint_review as j
from muhawir.corpus import Corpus, Passage, Source
from muhawir.verify import Claim
from scripts import measure_joint_review as m

pytestmark = pytest.mark.real_check


def response(content="{}", cost="0.001", **extra):
    return {"provider": "Together", "model": "fake", "usage": {"cost": cost},
            "choices": [{"finish_reason": "stop", "message": {"content": content}}], **extra}


def test_meter_counts_each_transport_attempt_and_retains_raw_usage():
    events = []
    meter = m.Meter(lambda *a: response(), events.append, Decimal("0.10"), maximum=2)
    for _ in range(2):
        assert meter.request("case", "current", g.CHECK_PROMPT, "private", {}) == "{}"
    with pytest.raises(m.StopRun, match="http_attempt_limit"):
        meter.request("case", "current", g.CHECK_PROMPT, "private", {})
    assert meter.attempts == 2 and meter.spent == Decimal("0.002")
    assert [e["kind"] for e in events] == ["attempt_started", "provider_response"] * 2
    assert events[1]["response"]["usage"]["cost"] == "0.001"


@pytest.mark.parametrize("bad", [None, True, -1, "NaN", "Infinity"])
def test_unknown_or_invalid_cost_stops_before_any_more_model_work(bad):
    events = []
    meter = m.Meter(lambda *a: response(cost=bad), events.append, Decimal("0.10"))
    with pytest.raises(m.StopRun, match="unknown_cost"):
        meter.request("case", "joint", j.REVIEW_PROMPT, "private", {})
    with pytest.raises(m.StopRun):
        meter.request("case", "joint", j.REVIEW_PROMPT, "private", {})
    assert meter.attempts == 1 and events[-1]["reported_cost_usd"] is None


def test_post_response_threshold_is_honest_about_overshoot():
    events = []
    meter = m.Meter(lambda *a: response(cost="0.11"), events.append, Decimal("0.10"))
    with pytest.raises(m.StopRun, match="post_response_cost_threshold"):
        meter.request("case", "joint", j.REVIEW_PROMPT, "private", {})
    assert meter.spent == Decimal("0.11") and meter.attempts == 1
    assert events[-1]["kind"] == "provider_response"


def test_failure_escapes_the_legacy_retry_loops_without_leaking_credentials():
    events = []
    def send(*args):
        exc = RuntimeError("PRIVATE KEY AND QUESTION")
        exc.response = type("Busy", (), {"status_code": 429})()
        raise exc
    meter = m.Meter(send, events.append, Decimal("0.10"))
    call = lambda *a: meter.request("case", "current", *a)
    with pytest.raises(m.StopRun):
        g.with_retry(call, g.CHECK_PROMPT, "private", {})
    assert meter.attempts == 1 and "PRIVATE" not in json.dumps(events)
    assert events[-1]["cost_status"] == "unknown"


@pytest.mark.parametrize("data", [None, [], {"usage": []},
    response(choices=[{"finish_reason": "length", "message": {"content": "{}"}}]),
    response(provider="Another provider"), response(choices=[]),
    response(choices=[{"finish_reason": "stop", "message": {"content": None}}])])
def test_malformed_and_truncated_response_is_recorded_then_stops(data):
    events = []
    meter = m.Meter(lambda *a: data, events.append, Decimal("0.10"))
    with pytest.raises(m.StopRun):
        meter.request("case", "joint", j.REVIEW_PROMPT, "private", {})
    assert meter.attempts == 1 and events[-1]["response"] == data


def toy_case():
    p = Passage("p", "s", "Synthetic location", "A synthetic box is blue.")
    source = Source("s", "Synthetic source", "Not religious evidence")
    return {"id": "toy", "family": "synthetic", "question": "What color is the box?",
            "expected": "accept", "local_only": False, "order": ["current", "joint"],
            "claims": [{**asdict(Claim("The box is blue.", ("p",))), "passage_ids": ["p"]}],
            "corpus": {"sources": [asdict(source)], "passages": [asdict(p)]}}


def test_actual_legacy_support_and_relevance_are_both_counted():
    case, events, systems = toy_case(), [], []
    def send(system, user, schema):
        systems.append(system)
        return response('{"problem":["none"]}' if system == g.CHECK_PROMPT else '{"verdict":"yes"}')
    meter = m.Meter(send, events.append, Decimal("0.10"))
    claims, corpus, _ = m.unpack(case)
    decision, detail = m.current_review(case, claims, corpus, meter)
    assert decision == "accept" and detail["relevance"] == "yes"
    assert systems == [g.CHECK_PROMPT, g.RELEVANCE_PROMPT] and meter.attempts == 2


def test_legacy_rereading_attempts_are_not_hidden(monkeypatch):
    case, events, seen = toy_case(), [], []
    def send(system, user, schema):
        seen.append(system)
        return response('{"problem":["addition"]}' if len(seen) == 1 else
                        '{"problem":["none"]}' if system == g.CHECK_PROMPT else '{"verdict":"yes"}')
    monkeypatch.setattr(g, "READ_EACH_SENTENCE", True)
    meter = m.Meter(send, events.append, Decimal("0.10"))
    claims, corpus, _ = m.unpack(case)
    decision, _ = m.current_review(case, claims, corpus, meter)
    assert decision == "accept" and meter.attempts == 3
    assert seen.count(g.CHECK_PROMPT) == 2


def test_an_unavailable_joint_judge_stops_batch_instead_of_counting_a_detection(monkeypatch):
    case = toy_case()
    case["order"] = ["joint", "current"]
    monkeypatch.setattr(m, "validate", lambda plan: None)
    events = []
    meter = m.Meter(lambda *a: response("{}"), events.append, Decimal("0.10"))
    status = m.run({"cases": [case]}, meter, events.append)
    result = next(e for e in events if e["kind"] == "case_result")
    assert result["decision"] == "unavailable"
    assert status == "stopped_unavailable_judge_no_retry_batch" and meter.attempts == 1


def test_summary_keeps_unavailable_and_missing_controls_separate(tmp_path):
    rows = [{"kind": "plan", "plan": {"scope": "paired_checker_diagnostic_not_release",
                                           "reference_status": "assistant_proposed_exposed_controls"}},
            {"kind": "attempt_started", "number": 1, "case_id": "test", "lane": "joint"},
            {"kind": "provider_response", "number": 1, "case_id": "test", "lane": "joint", "reported_cost_usd": "0.001",
             "response": response()},
            {"kind": "case_result", "case_id": "test", "lane": "joint", "expected": "reject", "decision": "unavailable", "seconds": 1},
            {"kind": "finished", "status": "stopped_unavailable_judge_no_retry_batch"}]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    report = m.summarize(path)
    joint = report["variants"]["joint"]
    assert joint["finished_controls"] == 1 and joint["expected_controls"] == 12
    assert joint["counts_against_proposed_references"] == {"unavailable": 1}
    assert joint["unknown_cost_attempts"] == 0 and joint["known_reported_usd"] == "0.001"
    assert report["variants"]["current"]["finished_controls"] == 0
    rows.insert(3, dict(rows[2]))
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="duplicate or unmatched response"):
        m.summarize(path)
