"""Prepare and validate a PRIVATE human-review bundle. No network or model imports.

A retrieval report supplies candidate evidence, never automatic truth labels.
Passing validation establishes completeness/integrity of a review record, not religious correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / "data" / "private" / "measurement"
DEFECTS = {"unsupported_detail", "negation", "condition", "merged_narration", "attribution", "off_topic"}


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def case_digest(case):
    # A reviewer signs all case inputs, labels and source snapshots. Editing any invalidates the signature.
    return digest({k: v for k, v in case.items() if k != "reviews"})


def prepare(raw):
    report = json.loads(raw)
    cases = []
    questions = set()
    for i, row in enumerate(report["results"], 1):
        question = row["question"]
        if not isinstance(question, str) or not question.strip() or question in questions:
            raise ValueError("missing or duplicated question; resolve question clusters before import")
        questions.add(question)
        passages = {}
        candidates = list(row.get("present_gold", []))
        for mode in row.get("modes", {}).values():
            candidates.extend(mode.get("hits", []))
        for item in candidates:
            evidence = {key: item[key] for key in ("id", "source", "location", "text")}
            if any(not isinstance(v, str) or not v.strip() for v in evidence.values()):
                raise ValueError("incomplete source snapshot")
            evidence["text_sha256"] = hashlib.sha256(evidence["text"].encode()).hexdigest()
            if evidence["id"] in passages and passages[evidence["id"]] != evidence:
                raise ValueError("conflicting snapshots for passage " + evidence["id"])
            passages[evidence["id"]] = evidence
        cases.append({"id": f"D{i:03d}", "cluster_id": f"D{i:03d}", "split": "development",
                      "question": question, "style": "youth", "lang": "ar", "history": [],
                      "candidate_passages": list(passages.values()),
                      "historical_reference_ids": row.get("gold", []),
                      "historical_missing_ids": row.get("missing_gold", []),
                      "expected_status": None, "required_answer_parts": [], "claims": [], "reviews": []})
    if not cases:
        raise ValueError("empty retrieval report")
    return {"schema_version": 1, "purpose": "development_grounding_review",
            "input_sha256": hashlib.sha256(raw).hexdigest(), "cases": cases}


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate(bundle):
    """Fail closed on pending labels, changed evidence, or stale/ambiguous reviewer attestations."""
    if bundle.get("schema_version") != 1 or bundle.get("purpose") != "development_grounding_review":
        raise ValueError("unsupported review bundle")
    cases = bundle.get("cases", [])
    if not cases:
        raise ValueError("no cases")
    ids = set()
    for case in cases:
        cid = case["id"]
        if not nonempty_text(cid) or cid in ids:
            raise ValueError("duplicate or invalid case ID")
        ids.add(cid)
        if case.get("split") != "development":
            raise ValueError(cid + ": previously exposed questions cannot become hidden holdout")
        if not all(nonempty_text(case.get(k)) for k in ("cluster_id", "question")):
            raise ValueError(cid + ": missing question/cluster")
        if case.get("style") not in ("youth", "kids", "extended", "newcomer") or case.get("lang") not in ("ar", "en", "ur"):
            raise ValueError(cid + ": invalid reader style/language")
        if case.get("expected_status") not in ("answered", "abstained", "referred", "declined"):
            raise ValueError(cid + ": expected status awaits human review")
        if not case.get("required_answer_parts") or not all(nonempty_text(p) for p in case["required_answer_parts"]):
            raise ValueError(cid + ": missing answer/abstention rubric")
        passages = {}
        for passage in case["candidate_passages"]:
            pid = passage["id"]
            if pid in passages:
                raise ValueError(cid + ": duplicate passage ID")
            if not all(nonempty_text(passage.get(k)) for k in ("id", "text", "source", "location")):
                raise ValueError(cid + ": incomplete passage")
            if hashlib.sha256(passage["text"].encode()).hexdigest() != passage["text_sha256"]:
                raise ValueError(cid + ": changed source text")
            passages[pid] = passage["text"]
        if not case.get("claims"):
            raise ValueError(cid + ": add supported and deliberately unsupported candidate claims for review")
        for claim in case["claims"]:
            verdict = claim.get("expected_supported")
            relevant = claim.get("expected_relevant")
            if type(verdict) is not bool or type(relevant) is not bool or not nonempty_text(claim.get("text")):
                raise ValueError(cid + ": claim awaits human verdict")
            if claim.get("defect") not in ({None} if verdict and relevant else DEFECTS):
                raise ValueError(cid + ": missing/inconsistent defect label")
            cited = claim.get("passage_ids", [])
            if not cited or any(pid not in passages for pid in cited):
                raise ValueError(cid + ": missing cited evidence")
            if not nonempty_text(claim.get("rationale")):
                raise ValueError(cid + ": missing reviewer rationale")
            quotes = claim.get("evidence_quotes", [])
            if not quotes:
                raise ValueError(cid + ": quote the support or the source condition that disproves the claim")
            for quote in quotes:
                if quote.get("passage_id") not in cited or not nonempty_text(quote.get("text")):
                    raise ValueError(cid + ": invalid quote citation")
                if quote["text"] not in passages[quote["passage_id"]]:
                    raise ValueError(cid + ": nonliteral quote")
        reviews = case.get("reviews", [])
        reviewers = set()
        for review in reviews:
            reviewer = review.get("reviewer")
            if not nonempty_text(reviewer) or reviewer in reviewers or review.get("approved") is not True:
                raise ValueError(cid + ": independent approvals required; adjudicate disagreements first")
            reviewers.add(reviewer)
            date.fromisoformat(review["date"])
            if review.get("case_sha256") != case_digest(case):
                raise ValueError(cid + ": reviewer signature is stale")
        if len(reviewers) < 2:
            raise ValueError(cid + ": two recorded human approvals required")
    claims = [claim for case in cases for claim in case["claims"]]
    return {"label_counts": dict(Counter("supported" if c["expected_supported"] else "unsupported" for c in claims)),
            "irrelevant_claims": sum(not c["expected_relevant"] for c in claims),
            "defect_counts": dict(Counter(c["defect"] for c in claims if c["defect"])),
            "cases": len(cases), "clusters": len({c["cluster_id"] for c in cases}),
            "claims": sum(len(c["claims"]) for c in cases), "bundle_sha256": digest(bundle),
            "scope": "review-record integrity only; not independent verification of reviewer expertise or truth"}


def write_private(path, value):
    path = path.resolve()
    if not path.is_relative_to(PRIVATE.resolve()):
        raise ValueError("write only under data/private/measurement; never publish source excerpts or review labels")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("report", type=Path)
    create.add_argument("--out", type=Path, default=PRIVATE / "development-review.json")
    for name in ("check", "freeze"):
        child = sub.add_parser(name)
        child.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        bundle = prepare(args.report.read_bytes())
        write_private(args.out, bundle)
        print(f"Prepared {len(bundle['cases'])} development cases; all verdicts pending; no quality result.")
        print("Input SHA256:", bundle["input_sha256"])
    else:
        bundle = json.loads(args.bundle.read_text())
        summary = validate(bundle)
        if args.command == "freeze":
            write_private(args.bundle.with_suffix(".lock.json"), {"review": summary, "bundle": bundle})
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
