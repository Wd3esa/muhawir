"""One batch, fake HTTP/journal only. No live questions or semantic quality claims."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
import threading

from muhawir.corpus import Corpus, Passage, Source
from scripts import run_acceptance_batch as batch
from scripts.analyze_request_traces import read_lines


def trace(rid, cost="0.001", cache="none", pending=0):
    return {"schema_version": 1, "request_id": rid, "code_revision": "code",
            "endpoint": "ask", "style": "youth", "lang": "ar", "as_question": False,
            "duration_ns": 100_000_000, "status": "answered", "cache": cache,
            "http_status": 200, "pending_work_at_response": pending, "events_dropped": 0,
            "events": [{"event_id": "ev-" + rid, "kind": "model", "provider": "fake", "model": "fake",
                        "stage": "model.draft", "outcome": "ok", "duration_ns": 10,
                        "reported_cost_usd": cost}]}


class Reply:
    status_code = 200
    def __init__(self, rid):
        self.rid = rid
    def json(self):
        return {"request_id": self.rid, "status": "answered", "claims": [], "sources": []}


class Client:
    def __init__(self, cost="0.001", failure=False, parallel=False):
        self.calls, self.records = [], []
        self.cost, self.failure = cost, failure
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(4) if parallel else None
    def post(self, path, json):
        assert path == "/api/ask" and json["history"] == []
        with self.lock:
            number = len(self.calls) + 1
            self.calls.append(json)
        if self.failure:
            raise TimeoutError("not a cancelled server request")
        if self.barrier and number > 1:
            self.barrier.wait(timeout=5)
        with self.lock:
            self.records.append(trace("r" + str(number), self.cost))
        return Reply("r" + str(number))
    def journal(self, invocation):
        with self.lock:
            return ["muhawir_trace " + json.dumps(row) for row in self.records]


def test_one_sequential_then_exactly_four_concurrent_without_resends(monkeypatch):
    monkeypatch.setattr(batch, "revision", lambda: "code")
    client = Client(parallel=True)
    snapshots = []
    results, rows, lines, status = batch.run(client, "invocation", client.journal,
                                           checkpoint=lambda d: snapshots.append(dict(d)))
    assert status == "completed_pending_quality_review"
    assert len(client.calls) == len(results) == len(rows) == 5
    assert client.calls[0]["question"] == batch.CASES[0]["question"]
    assert len({r["case"]["question"] for r in results}) == 5
    assert any("parallel_wall_seconds" in d for d in snapshots)
    assert snapshots[0]["status"] == "sequential_returned"


def test_first_transport_timeout_does_not_send_four_more_questions(monkeypatch):
    monkeypatch.setattr(batch, "revision", lambda: "code")
    client = Client(failure=True)
    results, rows, lines, status = batch.run(client, "invocation", client.journal)
    assert len(client.calls) == 1 and status == "stopped_incomplete_first_measurement"
    assert results[0]["error_class"] == "TimeoutError" and not rows


def test_a_parallel_timeout_is_not_retried_or_called_zero_cost(monkeypatch):
    monkeypatch.setattr(batch, "revision", lambda: "code")
    class OneTimeout(Client):
        def post(self, path, json):
            if json["question"] == batch.CASES[1]["question"]:
                with self.lock:
                    self.calls.append(json)
                raise TimeoutError("server work might still be running")
            return super().post(path, json)
    client = OneTimeout()
    results, rows, _, status = batch.run(client, "invocation", client.journal)
    assert len(client.calls) == len(results) == 5 and len(rows) == 4
    assert status == "completed_with_measurement_gaps"
    assert sum("error_class" in r for r in results) == 1


def test_unknown_cost_and_cost_threshold_stop_before_parallel_phase(monkeypatch):
    monkeypatch.setattr(batch, "revision", lambda: "code")
    for cost, expected in ((None, "stopped_incomplete_first_measurement"),
                           ("0.11", "stopped_after_first_reported_cost_threshold")):
        client = Client(cost=cost)
        _, _, _, status = batch.run(client, "invocation", client.journal, threshold=Decimal("0.10"))
        assert status == expected and len(client.calls) == 1


def test_other_visitors_trace_events_are_not_mixed_into_measurement():
    lines = ["muhawir_trace " + json.dumps(trace("ours")),
             "muhawir_trace " + json.dumps(trace("someone-else", "9"))]
    selected = batch.selected_lines(lines, {"ours"})
    assert len(selected) == 1
    assert read_lines(selected)[0]["request_id"] == "ours"


def test_background_costs_require_drain_completion_and_are_merged():
    row = trace("ours", pending=1)
    late = {"request_id": "ours", "event": {**row["events"][0], "event_id": "late", "reported_cost_usd": "0.002"}}
    lines = ["muhawir_trace " + json.dumps(row), "muhawir_trace_event " + json.dumps(late)]
    rows, _ = batch.capture("inv", {"ours"}, lambda _: lines, wait_seconds=0)
    assert not rows[0]["background_complete"] and len(rows[0]["events"]) == 2
    lines.append("muhawir_trace_drained " + json.dumps({"request_id": "ours", "events_dropped": 0}))
    rows, _ = batch.capture("inv", {"ours"}, lambda _: lines, wait_seconds=0)
    assert rows[0]["background_complete"]


def test_source_snapshots_keep_original_text_and_mark_unresolved_citations():
    corpus = Corpus({"s": Source("s", "Synthetic", "Synthetic source")},
                    [Passage("p", "s", "synthetic", "Original synthetic text")])
    results = [{"response": {"claims": [{"text": "claim", "passage_ids": ["p", "missing"]}],
                             "views": [], "sources": [{"passage_id": "p"}]}}]
    snapshots = batch.evidence(results, corpus)
    assert snapshots["p"]["passage"]["text"] == "Original synthetic text"
    assert snapshots["missing"] is None
    assert results[0]["citation_ids_missing_from_cards"] == ["missing"]


def test_cache_groups_remain_separate():
    from scripts.analyze_request_traces import summarize
    rows = read_lines(["muhawir_trace " + json.dumps(trace("a", cache="memory")),
                       "muhawir_trace " + json.dumps(trace("b", cache="none"))])
    assert len(summarize(rows)["groups"]) == 2


def test_configuration_export_has_no_credentials_or_full_environment():
    config = batch.public_config(b"MUHAWIR_TRACE=1\0OPENAI_COMPAT_MODEL=fake/model\0"
                                 b"OPENAI_COMPAT_API_KEY=PRIVATE\0HOME=/private\0SECRET=PRIVATE\0")
    assert config == {"MUHAWIR_TRACE": "1", "OPENAI_COMPAT_MODEL": "fake/model"}
    assert "PRIVATE" not in json.dumps(config)
