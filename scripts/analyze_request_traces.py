"""Read local content-free logs. No network, model calls, cost estimates or quality claims."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import json
import math
from pathlib import Path
import statistics

PREFIX = "muhawir_trace "


def read_records(path, skipped=None):
    return read_lines(Path(path).read_text().splitlines(), skipped)


def read_lines(lines, skipped=None):
    """The same strict parser for a private in-memory journal export.

    With ``skipped`` (a list) a trace line that is not valid JSON, such as one a journal export cut in two, is not an
    error: its line number is appended to ``skipped`` and the line is left out, and an event or completion whose request
    was lost that way is left out too (and listed). Without it every bad line stops the read, as before."""
    rows, ids, late, drained = [], set(), [], {}
    lenient = skipped is not None

    def load(text, number):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if not lenient:
                raise
            skipped.append(number)
            return None

    for number, line in enumerate(lines, 1):
        if "muhawir_trace_event " in line:
            item = load(line.split("muhawir_trace_event ", 1)[1], number)
            if item is not None:
                late.append(item)
            continue
        if "muhawir_trace_drained " in line:
            tail = load(line.split("muhawir_trace_drained ", 1)[1], number)
            if tail is not None:
                drained[tail["request_id"]] = tail
            continue
        if PREFIX not in line:
            continue
        row = load(line.split(PREFIX, 1)[1], number)
        if row is None:
            continue
        if row.get("schema_version") != 1:
            raise ValueError(f"line {number}: unsupported trace schema")
        if row["request_id"] in ids:
            raise ValueError(f"line {number}: duplicate request ID; do not count rotated logs twice")
        ids.add(row["request_id"])
        if not isinstance(row["duration_ns"], int) or row["duration_ns"] < 0:
            raise ValueError(f"line {number}: invalid duration")
        rows.append(row)
    if not rows:
        raise ValueError("no completed traces; absent/in-flight/killed requests are not zero-duration successes")
    by_id = {row["request_id"]: row for row in rows}
    for item in late:
        if item["request_id"] not in by_id:
            if lenient:
                skipped.append(f'event of request {item["request_id"]}')
                continue
            raise ValueError("late event without its request; export the complete log window")
        by_id[item["request_id"]]["events"].append(item["event"])
    if set(drained) - set(by_id):
        if not lenient:
            raise ValueError("background completion without its request; export the complete log window")
        skipped.extend(f"completion of request {i}" for i in sorted(set(drained) - set(by_id)))
        drained = {k: v for k, v in drained.items() if k in by_id}
    for row in rows:
        event_ids = [event["event_id"] for event in row["events"] if "event_id" in event]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate event ID; export each log event once")
        tail = drained.get(row["request_id"])
        row["background_complete"] = not row.get("pending_work_at_response", 0) or tail is not None
        if tail:
            row["events_dropped"] = tail["events_dropped"]
    return rows


def durations(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "median_s": None, "p95_s": None, "max_s": None}
    # Explicit nearest-rank definition; descriptive only, not a population guarantee.
    return {"n": len(values), "median_s": statistics.median(values) / 1e9,
            "p95_s": values[math.ceil(.95 * len(values)) - 1] / 1e9, "max_s": values[-1] / 1e9}


def span_profile(rows):
    """Where the time goes: per named stage (span), how many times it ran, its total seconds and its share of the
    requests' total time. Spans nest (a stage can contain another), so shares can add up to more than 100%: read each
    line on its own. Model calls are reported separately so waiting on a provider is visible apart from our own code."""
    total = sum(row["duration_ns"] for row in rows) or 1
    spans, model_ns = {}, 0
    for row in rows:
        for event in row["events"]:
            if event.get("kind") == "span" and "name" in event and "duration_ns" in event:
                entry = spans.setdefault(event["name"], {"runs": 0, "total_ns": 0})
                entry["runs"] += 1
                entry["total_ns"] += event["duration_ns"]
            elif event.get("kind") == "model" and "duration_ns" in event:
                model_ns += event["duration_ns"]
    ordered = sorted(spans.items(), key=lambda item: -item[1]["total_ns"])
    return {
        "requests": len(rows),
        "total_request_s": round(total / 1e9, 2),
        "model_calls_s": round(model_ns / 1e9, 2),
        "model_share": round(model_ns / total, 3),
        "spans": {name: {"runs": e["runs"], "total_s": round(e["total_ns"] / 1e9, 2),
                         "share": round(e["total_ns"] / total, 3)} for name, e in ordered},
    }


def loss_summary(rows):
    """What each step removed, summed over the answers that recorded it: counts only. «answers_with» says how many
    answers reached the step; a step that removes nothing shows no key."""
    stages = {}
    for row in rows:
        for event in row["events"]:
            if event.get("kind") != "loss":
                continue
            stage = stages.setdefault(event["stage"], {"answers_with": 0})
            stage["answers_with"] += 1
            for key, value in event.get("counts", {}).items():
                stage[key] = stage.get(key, 0) + value
    return stages


def summarize(rows):
    revisions = {row.get("code_revision") for row in rows}
    if len(revisions) > 1:
        raise ValueError("mixed code revisions; analyze each deployment separately")
    calls = [event for row in rows for event in row["events"] if event["kind"] == "model"]
    known = [Decimal(event["reported_cost_usd"]) for event in calls if event.get("reported_cost_usd") is not None]
    dropped = sum(row.get("events_dropped", 0) for row in rows)
    pending = sum(not row.get("background_complete", not row.get("pending_work_at_response", 0)) for row in rows)
    groups = {}
    # Never mix saved answers, newly computed replies, rejections, or streaming vs JSON.
    for row in rows:
        key = f'{row["endpoint"]}/{row.get("style")}/{row.get("lang")}/{row.get("as_question")}/{row["cache"]}/{row["http_status"]}/{row["status"]}'
        groups.setdefault(key, []).append(row["duration_ns"])
    per_request = []
    for row in rows:
        own = [event for event in row["events"] if event["kind"] == "model"]
        own_known = [Decimal(event["reported_cost_usd"]) for event in own if event.get("reported_cost_usd") is not None]
        per_request.append({"request_id": row["request_id"], "duration_s": row["duration_ns"] / 1e9,
                            "status": row["status"], "cache": row["cache"], "observed_transport_calls": len(own),
                            "reused_support_checks": sum(e.get("count", 0) for e in row["events"]
                                if e.get("kind") == "reuse" and e.get("name") == "support_positive"),
                            "known_reported_cost_usd": str(sum(own_known, Decimal(0))),
                            "calls_with_unknown_cost": len(own) - len(own_known)})
    return {
        "code_revision": next(iter(revisions)),
        "requests": per_request,
        "completed_requests": len(rows),
        "http_errors": sum(row["http_status"] >= 400 for row in rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "groups": {key: durations(values) for key, values in sorted(groups.items())},
        "loss": loss_summary(rows),
        "profile": span_profile(rows),
        "observed_transport_calls": len(calls),
        "transport_errors": sum(event["outcome"] != "ok" for event in calls),
        "calls_by_stage": dict(Counter(event["stage"] for event in calls)),
        "provider_models": sorted({f'{event["provider"]}/{event["model"]}' for event in calls}),
        "known_reported_cost_usd": str(sum(known, Decimal(0))),
        "calls_with_unknown_cost": len(calls) - len(known),
        "cost_complete_for_observed_calls": len(known) == len(calls) and not dropped and not pending,
        "events_dropped": dropped,
        "requests_with_unfinished_background_work": pending,
        "limitations": [
            "Server handler duration includes admission; excludes HTTP parsing, network and browser rendering.",
            "Only completed recorded requests; reconcile against a client manifest before reporting failure rate or throughput.",
            "p95 is empirical nearest rank; small samples do not establish a reliable tail or capacity.",
            "Spans overlap and nest; never add their totals to obtain wall time.",
            "Unknown costs are not zero; timeouts may still be billed. Reconcile provider billing separately.",
            "No quality labels, population confidence interval or sustainable capacity conclusion is inferred.",
            "Do not pool different code/config/corpus, cold/warm runs, or repeated-question clusters.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--skip-bad", action="store_true",
                        help="skip trace lines that are not valid JSON (for example cut by a journal export) and "
                             "report how many; the default stops at the first one")
    args = parser.parse_args()
    skipped: list | None = [] if args.skip_bad else None
    report = summarize(read_records(args.log, skipped))
    if skipped is not None:
        report["skipped_lines"] = {"count": len(skipped), "items": skipped[:20]}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
