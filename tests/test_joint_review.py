"""Contract/orchestration tests with synthetic sources and fake model replies.

These test fail-closed mechanics and bounds, NOT the accuracy of an LLM judge.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from muhawir.corpus import Corpus, Passage, Source
from muhawir import joint_review as joint
from muhawir.verify import Claim


def context():
    corpus = Corpus({"s": Source("s", "Synthetic source", "Not religious evidence")}, [
        Passage("p1", "s", "1", "This sample rule applies only to blue boxes. The sample report is uncertain."),
        Passage("p2", "s", "2", "Green boxes have a separate sample rule."),
    ], synthetic=True)
    claims = [Claim("The sample rule applies only to blue boxes.", ("p1",))]
    return claims, corpus, {"p1", "p2"}


def request_for(claims=None):
    default, corpus, ids = context()
    return joint.prepare_review("What is the sample rule?", default if claims is None else claims, corpus, ids)


def positive(request):
    """Fake acceptance. It cannot establish that its own evidence semantically supports the claim."""
    body = json.loads(request.user)
    decisions = []
    passages = {p["id"]: p for p in body["passages"]}
    for c in body["claims"]:
        pid = c["passage_ids"][0]
        decisions.append({"id": c["id"], "verdict": "supported", "units": [{
            "text": c["displayed"], "evidence": [{"passage_id": pid, "quote": passages[pid]["text"]}],
            "reason": "Synthetic witness for contract testing only."}], "findings": []})
    return {"input_sha256": request.digest, "claims": decisions,
            "coverage": [{"part": "sample rule", "status": "answered",
                          "claim_ids": [c["id"] for c in decisions],
                          "passage_ids": sorted({p for c in body["claims"] for p in c["passage_ids"]}),
                          "reason": "Synthetic coverage."}],
            "checks": {k: {"status": "pass", "claim_ids": [], "reason": "Synthetic check."}
                       for k in joint.CHECKS}}


def negative(request, code="dropped_condition"):
    data = positive(request)
    data["claims"][0].update(verdict="unsupported", units=[],
                             findings=[{"code": code, "detail": "A material part is not supported."}])
    data["coverage"][0].update(status="missing", claim_ids=[])
    data["checks"]["conditions"].update(status="fail", claim_ids=["c001"])
    return data


def call_with(fn, seen):
    def call(system, user, schema):
        seen.append(system)
        return joint.Completion(json.dumps(fn(joint.ReviewRequest(user))), "stop")
    return call


def test_acceptance_requires_complete_decisions_and_real_witnesses_in_one_call():
    claims, corpus, ids = context()
    seen = []
    out = joint.evaluate("What is the sample rule?", claims, corpus, ids, call_with(positive, seen))
    assert out.status == "accepted_for_evaluation"
    assert out.claims == tuple(claims) and out.attempts == 1
    assert seen == [joint.REVIEW_PROMPT]


def test_model_receives_the_actual_output_contract_even_with_a_json_object_transport():
    claims, corpus, ids = context()
    def call(system, user, schema):
        body = json.loads(user)
        assert body["response_schema"] == schema == joint.REVIEW_SCHEMA
        assert "dropped_condition" in body["response_schema"]["properties"]["claims"]["items"][
            "properties"]["findings"]["items"]["properties"]["code"]["enum"]
        return joint.Completion(json.dumps(positive(joint.ReviewRequest(user))), "stop")
    out = joint.evaluate("Sample?", claims, corpus, ids, call)
    assert out.status == "accepted_for_evaluation" and out.attempts == 1


def test_misnested_rejection_remains_unavailable_instead_of_being_silently_repaired():
    request = request_for()
    data = negative(request)
    data["claims"][0]["units"] = [{
        "text": "The sample rule applies only to blue boxes.",
        "evidence": [{"passage_id": "p1", "quote": "only to blue boxes", "reason": "Condition."}],
        "findings": [{"code": "condition_omitted", "detail": "A condition was omitted."}],
    }]
    with pytest.raises(ValueError):
        joint.parse_review(json.dumps(data), request)


@pytest.mark.parametrize("code", joint.PROBLEMS)
def test_explicit_defect_is_never_outvoted_or_silently_deleted(code):
    claims, corpus, ids = context()
    seen = []
    out = joint.evaluate("Sample?", claims, corpus, ids,
                         call_with(lambda req: negative(req, code), seen))
    assert out.status == "rejected" and out.claims == () and out.attempts == 1
    assert len(seen) == 1


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(input_sha256="stale"),
    lambda d: d.update(claims=[]),
    lambda d: d["claims"].append(copy.deepcopy(d["claims"][0])),
    lambda d: d["claims"][0].update(id="unknown"),
    lambda d: d["claims"][0].update(verdict=True),
    lambda d: d["claims"][0].update(units=[]),
    lambda d: d["claims"][0]["units"][0].update(evidence=[]),
    lambda d: d["claims"][0]["units"][0].update(text="The sample rule applies."),
    lambda d: d["claims"][0]["units"][0]["evidence"][0].update(quote="Fabricated witness"),
    lambda d: d["claims"][0]["units"][0]["evidence"][0].update(quote="   "),
    lambda d: d["claims"][0]["units"][0]["evidence"][0].update(passage_id="p2"),
    lambda d: d["claims"][0].update(findings=[{"code": "wrong_inference", "detail": "Failure"}]),
    lambda d: d["claims"][0].update(verdict="uncertain"),
    lambda d: d.update(coverage=[]),
    lambda d: d["coverage"][0].update(claim_ids=["unknown"]),
    lambda d: d["coverage"][0].update(passage_ids=["p2"]),
    lambda d: d["coverage"][0].update(status="missing", passage_ids=[]),
    lambda d: d["checks"].pop("grading"),
    lambda d: d["checks"]["conditions"].update(status="fail"),
    lambda d: d.update(extra=True),
])
def test_bad_review_contract_fails_closed_without_retry(mutation):
    claims, corpus, ids = context()
    def answer(req):
        data = positive(req)
        mutation(data)
        return data
    seen = []
    out = joint.evaluate("Sample?", claims, corpus, ids, call_with(answer, seen), allow_repair=True)
    assert out.status == "unavailable" and out.attempts == 1 and not out.claims
    assert len(seen) == 1


@pytest.mark.parametrize("raw", ["not json", "{}", '{"input_sha256":NaN}',
                                  '{"claims":[],"claims":[]}', "x" * 128_001])
def test_malformed_reply_has_no_fallback_or_retry(raw):
    claims, corpus, ids = context()
    out = joint.evaluate("Sample?", claims, corpus, ids,
                         lambda *args: joint.Completion(raw, "stop"), allow_repair=True)
    assert out.status == "unavailable" and out.attempts == 1 and not out.claims


def test_timeout_does_not_leak_exception_or_retry():
    claims, corpus, ids = context()
    def fail(*args):
        raise TimeoutError("PRIVATE credential and question")
    out = joint.evaluate("Sample?", claims, corpus, ids, fail, allow_repair=True)
    assert out.error_type == "TimeoutError" and out.attempts == 1
    assert "PRIVATE" not in repr(out) and not out.claims


@pytest.mark.parametrize("finish", ["length", "content_filter", "error", None])
def test_even_parseable_json_is_rejected_if_completion_was_not_complete(finish):
    claims, corpus, ids = context()
    def call(system, user, schema):
        return joint.Completion(json.dumps(positive(joint.ReviewRequest(user))), finish)
    out = joint.evaluate("Sample?", claims, corpus, ids, call, allow_repair=True)
    assert out.status == "unavailable" and out.attempts == 1 and not out.claims


def test_unused_citation_cannot_be_used_as_source_decoration():
    claims, corpus, ids = context()
    claims[0] = Claim(claims[0].text, ("p1", "p2"))
    out = joint.evaluate("Sample?", claims, corpus, ids, call_with(positive, []))
    assert out.status == "unavailable" and out.attempts == 1


def test_all_displayed_words_must_be_covered_including_layout_and_speaker():
    claims, corpus, ids = context()
    claims[0] = Claim(claims[0].text, ("p1",), section="Important condition", label="Scope")
    def omit_layout(request):
        data = positive(request)
        data["claims"][0]["units"][0]["text"] = claims[0].text
        return data
    out = joint.evaluate("Sample?", claims, corpus, ids, call_with(omit_layout, []))
    assert out.status == "unavailable" and out.attempts == 1


@pytest.mark.parametrize("claim", [Claim("Blue.", ()), Claim("Blue.", ("missing",)),
                                   Claim("Blue.", ("p1", "p1")),
                                   Claim("«اقتباس مختلق»", ("p1",))])
def test_citation_and_quote_checks_happen_before_any_model_call(claim):
    _, corpus, ids = context()
    out = joint.evaluate("Sample?", [claim], corpus, ids, lambda *a: pytest.fail("unexpected call"))
    assert out.status == "unavailable" and out.attempts == 0


def test_no_truncation_on_oversized_context():
    claims, corpus, ids = context()
    corpus.passages.append(Passage("big", "s", "3", "x" * 240_000))
    corpus._by_id["big"] = corpus.passages[-1]
    out = joint.evaluate("Sample?", claims, corpus, ids | {"big"}, lambda *a: pytest.fail("unexpected call"))
    assert out.attempts == 0 and not out.claims


def test_insufficient_sources_cannot_trigger_repair_or_publish_partial_answer():
    claims, corpus, ids = context()
    def answer(req):
        data = positive(req)
        data["coverage"].append({"part": "second topic", "status": "insufficient",
                                 "claim_ids": [], "passage_ids": [], "reason": "No source."})
        return data
    out = joint.evaluate("Two parts?", claims, corpus, ids, call_with(answer, []), allow_repair=True)
    assert out.status == "needs_sources" and out.attempts == 1 and not out.claims


def test_repair_is_local_then_entire_new_draft_is_reverified():
    claims, corpus, ids = context()
    claims += [Claim("Green boxes have a separate sample rule.", ("p2",), section="Second part")]
    original_second = claims[1]
    seen = []
    def call(system, user, schema):
        seen.append(system)
        if system == joint.REPAIR_PROMPT:
            data = json.loads(user)
            assert data["response_schema"] == schema == joint.REPAIR_SCHEMA
            assert data["input_sha256"] == data["input"]["input_sha256"]
            assert data["editable_ids"] == ["c001"]
            replacement = dict(data["input"]["claims"][0])
            replacement.pop("id")
            replacement.pop("displayed")
            replacement["text"] = "Only blue boxes fall under this sample rule."
            return joint.Completion(json.dumps({"input_sha256": data["input"]["input_sha256"],
                               "changes": [{"id": "c001", "replacement": replacement}], "additions": []}), "stop")
        req = joint.ReviewRequest(user)
        data = negative(req) if len(seen) == 1 else positive(req)
        if len(seen) == 3:
            offered = json.loads(user)["claims"]
            assert len(offered) == 2 and offered[1]["text"] == original_second.text
        return joint.Completion(json.dumps(data), "stop")
    out = joint.evaluate("Both rules?", claims, corpus, ids, call, allow_repair=True)
    assert out.status == "accepted_for_evaluation" and out.attempts == 3
    assert out.claims[1] == original_second and out.claims[0] != claims[0]
    assert len(out.reviews) == 2
    assert seen == [joint.REVIEW_PROMPT, joint.REPAIR_PROMPT, joint.REVIEW_PROMPT]


@pytest.mark.parametrize("bad", ["unflagged", "new_source", "duplicate", "stale", "no_change"])
def test_repair_cannot_change_unflagged_content_or_bypass_checks(bad):
    claims, corpus, ids = context()
    claims += [Claim("Green boxes have a separate sample rule.", ("p2",))]
    def call(system, user, schema):
        if system == joint.REVIEW_PROMPT:
            return joint.Completion(json.dumps(negative(joint.ReviewRequest(user))), "stop")
        data = json.loads(user)
        replacement = dict(data["input"]["claims"][0])
        replacement.pop("id")
        replacement.pop("displayed")
        change = {"id": "c002" if bad == "unflagged" else "c001", "replacement": replacement}
        if bad == "new_source":
            replacement["passage_ids"] = ["not_retrieved"]
        return joint.Completion(json.dumps({"input_sha256": "stale" if bad == "stale" else data["input"]["input_sha256"],
                           "changes": [change, change] if bad == "duplicate" else [change], "additions": []}), "stop")
    out = joint.evaluate("Sample?", claims, corpus, ids, call, allow_repair=True)
    assert out.status == "unavailable" and out.attempts == 2 and not out.claims


def test_second_rejection_stops_at_three_callbacks_without_more_repair():
    claims, corpus, ids = context()
    def call(system, user, schema):
        if system == joint.REVIEW_PROMPT:
            return joint.Completion(json.dumps(negative(joint.ReviewRequest(user), "wrong_inference")), "stop")
        data = json.loads(user)
        replacement = dict(data["input"]["claims"][0])
        replacement.pop("id")
        replacement.pop("displayed")
        replacement["text"] = "This sample rule applies only to blue boxes, not other boxes."
        return joint.Completion(json.dumps({"input_sha256": data["input"]["input_sha256"],
                           "changes": [{"id": "c001", "replacement": replacement}], "additions": []}), "stop")
    out = joint.evaluate("Sample?", claims, corpus, ids, call, allow_repair=True)
    assert out.status == "rejected_after_repair" and out.attempts == 3 and not out.claims


def test_concurrent_reviews_do_not_share_decisions_or_mutable_state():
    def run(question):
        claims, corpus, ids = context()
        return joint.evaluate(question, claims, corpus, ids, call_with(positive, []))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, [f"Question {i}?" for i in range(12)]))
    assert all(r.status == "accepted_for_evaluation" and r.attempts == 1 for r in results)
    assert len({r.reviews[0].data["input_sha256"] for r in results}) == 12


def test_an_exact_witness_is_explicitly_not_a_semantic_proof():
    # Intentional dishonest fake judge: syntactic validation cannot discover this lie.
    # Keeping this limitation executable prevents claims that offline tests prove judge accuracy.
    claims, corpus, ids = context()
    claims[0] = Claim("An unrelated invented statement.", ("p1",))
    out = joint.evaluate("Sample?", claims, corpus, ids, call_with(positive, []))
    assert out.status == "accepted_for_evaluation"
