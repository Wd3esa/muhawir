"""Synthetic text only. These tests validate the review gate, not religious assertions."""
import hashlib
import json
import pytest

from scripts import prepare_quality_review as review


def report():
    p = {"id": "fake:1", "source": "Synthetic", "location": "test", "text": "The box is blue if the lamp is on."}
    return json.dumps({"results": [{"question": "What color is the box?", "gold": ["old:missing"],
                                   "missing_gold": ["old:missing"], "present_gold": [],
                                   "modes": {"exact": {"hits": [p]}, "ann": {"hits": [p]}}}]}).encode()


def approved():
    bundle = review.prepare(report())
    case = bundle["cases"][0]
    case.update(expected_status="answered", required_answer_parts=["Preserve the lamp condition."], claims=[
        {"text": "The box is blue if the lamp is on.", "passage_ids": ["fake:1"],
         "expected_supported": True, "expected_relevant": True, "defect": None, "rationale": "Literal conditional statement.",
         "evidence_quotes": [{"passage_id": "fake:1", "text": "blue if the lamp is on"}]},
        {"text": "The box is always blue.", "passage_ids": ["fake:1"],
         "expected_supported": False, "expected_relevant": True, "defect": "condition", "rationale": "Drops the condition.",
         "evidence_quotes": [{"passage_id": "fake:1", "text": "if the lamp is on"}]}])
    sign(case)
    return bundle


def sign(case):
    case["reviews"] = [{"reviewer": name, "approved": True, "date": "2026-10-29", "case_sha256": review.case_digest(case)}
                       for name in ("test-reviewer-A", "test-reviewer-B")]


def test_historical_gold_never_becomes_automatic_truth():
    bundle = review.prepare(report())
    case = bundle["cases"][0]
    assert case["historical_missing_ids"] == ["old:missing"]
    assert len(case["candidate_passages"]) == 1
    assert case["expected_status"] is None and case["claims"] == [] and case["reviews"] == []
    assert bundle["input_sha256"] == hashlib.sha256(report()).hexdigest()
    with pytest.raises(ValueError, match="awaits human"):
        review.validate(bundle)


def test_complete_record_can_be_frozen_but_edits_invalidate_signatures():
    bundle = approved()
    assert review.validate(bundle)["claims"] == 2
    bundle["cases"][0]["question"] += " changed"
    with pytest.raises(ValueError, match="stale"):
        review.validate(bundle)


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c["candidate_passages"][0].update(text="changed"), "changed source"),
    (lambda c: c["claims"][0]["evidence_quotes"][0].update(text="invented quote"), "nonliteral"),
    (lambda c: c["claims"][0].update(passage_ids=["missing"]), "missing cited"),
    (lambda c: c.update(split="holdout"), "hidden holdout"),
    (lambda c: c["reviews"].pop(), "two recorded"),
    (lambda c: c["claims"][1].update(defect=None), "defect"),
    (lambda c: c["claims"][0].update(expected_supported="true"), "human verdict"),
])
def test_invalid_or_incomplete_reviews_fail(mutation, match):
    bundle = approved()
    mutation(bundle["cases"][0])
    with pytest.raises(ValueError, match=match):
        review.validate(bundle)


def test_conflicting_source_snapshot_is_not_silently_replaced():
    data = json.loads(report())
    data["results"][0]["modes"]["ann"]["hits"][0]["text"] = "Different version."
    with pytest.raises(ValueError, match="conflicting snapshots"):
        review.prepare(json.dumps(data).encode())


def test_output_private_exclusive_and_owner_only(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "PRIVATE", tmp_path / "private")
    path = review.PRIVATE / "case.json"
    review.write_private(path, {"synthetic": True})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        review.write_private(path, {})
    with pytest.raises(ValueError, match="write only"):
        review.write_private(tmp_path / "public.json", {})


def test_duplicate_reviewers_or_changed_labels_cannot_pass():
    bundle = approved()
    case = bundle["cases"][0]
    case["reviews"][1]["reviewer"] = case["reviews"][0]["reviewer"]
    with pytest.raises(ValueError, match="independent"):
        review.validate(bundle)
    case["claims"][1]["expected_supported"] = True
    case["claims"][1]["defect"] = None
    case["reviews"][1]["reviewer"] = "another-reviewer"
    with pytest.raises(ValueError, match="stale"):
        review.validate(bundle)
