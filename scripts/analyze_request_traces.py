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


def read_records(path):
    rows, ids, late, drained = [], set(), [], {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if "muhawir_trace_event " in line:
            late.append(json.loads(line.split("muhawir_trace_event ", 1)[1]))
            continue
        if "muhawir_trace_drained " in line:
            tail = json.loads(line.split("muhawir_trace_drained ", 1)[1])
            drained[tail["request_id"]] = tail
            continue
        if PREFIX not in line:
            continue
        row = json.loads(line.split(PREFIX, 1)[1])
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
            raise ValueError("late event without its request; export the complete log window")
        by_id[item["request_id"]]["events"].append(item["event"])
    if set(drained) - set(by_id):
        raise ValueError("background completion without its request; export the complete log window")
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
        key = f'{row["endpoint"]}/{row["cache"]}/{row["http_status"]}/{row["status"]}'
        groups.setdefault(key, []).append(row["duration_ns"])
    return {
        "code_revision": next(iter(revisions)),
        "completed_requests": len(rows),
        "http_errors": sum(row["http_status"] >= 400 for row in rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "groups": {key: durations(values) for key, values in sorted(groups.items())},
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
    args = parser.parse_args()
    print(json.dumps(summarize(read_records(args.log)), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
