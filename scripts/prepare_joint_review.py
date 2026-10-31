"""Prepare private joint-review inputs from a recorded draft comparison. No API calls.

Only current drafts are selected: the nonredundant candidate was rejected. These
four exposed development cases are not an independent quality evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muhawir.corpus import Corpus, Passage, Source
from muhawir.generate import parse_draft
from muhawir.joint_review import REVIEW_PROMPT, REVIEW_SCHEMA, prepare_review
from muhawir.verify import verify


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def corpus_from_case(case):
    """Recover the exact previously supplied texts, checking each recorded text digest.

    Source metadata is only what the recorded prompt exposed; no corpus query,
    inferred author, or replacement passage is introduced.
    """
    user = case["user"]
    if sha(user) != case["user_sha256"]:
        raise ValueError("changed recorded user prompt")
    boundary = "\n\n\n\nأسلوب الشرح:"
    if user.count(boundary) != 1:
        raise ValueError("unrecognized recorded prompt format")
    context = user.split(boundary)[0]
    starts = []
    ids = set()
    for e in case["evidence"]:
        if e["id"] in ids:
            raise ValueError("duplicate evidence id")
        ids.add(e["id"])
        marker = "\n\n[" + e["id"] + "] ("
        if context.count(marker) != 1:
            raise ValueError("ambiguous passage boundary")
        start = context.index(marker)
        body = context.index("\n", start + len(marker)) + 1
        header = context[start + 2:body].strip()
        starts.append((start, body, header, e))
    if not starts or [x[0] for x in starts] != sorted(x[0] for x in starts):
        raise ValueError("changed evidence order")
    sources, passages = {}, []
    for i, (_, body, header, e) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(context)
        text = context[body:end].rstrip("\n")
        matches = [text + suffix for suffix in ("", "\n", "\n\n")
                   if sha(text + suffix) == e["text_sha256"]]
        if len(matches) != 1:
            raise ValueError("changed evidence text")
        if not header.endswith(e["location"] + ")"):
            raise ValueError("changed source location")
        metadata = header[len("[" + e["id"] + "] ("):-(len(e["location"]) + 1)]
        metadata = metadata.removesuffix(" — ")
        kind = {"فتوى": "fatwa", "فقه": "fiqh", "حديث": "hadith", "آية": "quran"}.get(
            metadata.split(" — ")[0], "other")
        source = Source(e["source_id"], e["source"], metadata)
        previous = sources.get(source.id)
        if previous is not None and previous != source:
            raise ValueError("inconsistent recorded source metadata")
        sources[source.id] = source
        passages.append(Passage(e["id"], source.id, e["location"], matches[0], kind))
    return Corpus(sources, passages)


def prepare_recording(path):
    raw = path.read_bytes()
    if len(raw) > 20_000_000:
        raise ValueError("recording too large")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or rows[0].get("kind") != "plan" or rows[-1] != {
            "kind": "finished", "status": "completed_draft_only_no_quality_verdict"}:
        raise ValueError("incomplete or unexpected recording")
    plan = rows[0]["plan"]
    if sha(json.dumps(plan, ensure_ascii=False, indent=2)) != rows[0]["plan_sha256"]:
        raise ValueError("changed recorded plan")
    if plan["scope"] != "exploratory_draft_only_not_release" or len(plan["cases"]) != 4:
        raise ValueError("expected the four-case exploratory comparison")
    for name, system in plan["systems"].items():
        if sha(system) != plan["system_sha256"][name]:
            raise ValueError("changed recorded system prompt")
    responses, starts = {}, []
    for row in rows[1:-1]:
        key = (row.get("case_id"), row.get("variant"))
        if row["kind"] == "attempt_started":
            starts.append(key)
        elif row["kind"] == "response" and key not in responses:
            responses[key] = row
        else:
            raise ValueError("unexpected, failed or duplicate event")
    expected = [(c["id"], variant) for c in plan["cases"] for variant in c["order"]]
    if len(set(expected)) != 8 or starts != expected or set(responses) != set(expected):
        raise ValueError("recording attempts do not match the plan")
    out = []
    for case in plan["cases"]:
        row = responses[(case["id"], "current")]
        choices = row["response"]["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
            raise ValueError("truncated draft")
        raw_draft = choices[0]["message"]["content"]
        draft = json.loads(raw_draft)
        if draft["abstain"] is not False or not draft["claims"]:
            raise ValueError("not an answered draft")
        claims = parse_draft(raw_draft)
        if len(claims) != len(draft["claims"]) + len(draft["views"]):
            raise ValueError("draft parser discarded content")
        corpus = corpus_from_case(case)
        allowed = {p.id for p in corpus.passages}
        _, rejected = verify(claims, corpus, allowed)
        entry = {"case_id": case["id"], "variant": "current",
                 "original_response_id": row["response"]["id"],
                 "local_rejections": [{"text": r.claim.text, "reason": r.reason} for r in rejected]}
        if rejected:
            # Preserve the failed case, not a silently shortened answer or a paid reviewer request.
            entry.update(status="blocked_by_existing_code_check", request=None)
        else:
            request = prepare_review(case["question"], claims, corpus, allowed, case["style"])
            entry.update(status="prepared_not_executed",
                         request={"system": REVIEW_PROMPT, "user": request.user, "schema": REVIEW_SCHEMA})
        out.append(entry)
    return {"scope": "inactive_joint_review_inputs_not_quality_results",
            "recording_sha256": hashlib.sha256(raw).hexdigest(),
            "recorded_plan_sha256": rows[0]["plan_sha256"],
            "human_reference_status": "pending", "requests": out,
            "execution": "not_implemented_here_no_model_calls"}


def save_private(data, path):
    path = path.resolve()
    folder = (ROOT / "data/private/measurement").resolve()
    if not path.is_relative_to(folder):
        raise ValueError("output must be within data/private/measurement")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path,
                        default=ROOT / f"data/private/measurement/joint-review-{time.time_ns()}.json")
    args = parser.parse_args()
    data = prepare_recording(args.recording)
    save_private(data, args.out)
    blocked = sum(r["status"] == "blocked_by_existing_code_check" for r in data["requests"])
    print(f"Preserved four current drafts: {4 - blocked} reviewer inputs, {blocked} local rejections; no model calls.")
    print("No quality verdict. Independent reference review and paid-run approval remain pending.")
    print("Saved:", args.out.resolve())


if __name__ == "__main__":
    main()
