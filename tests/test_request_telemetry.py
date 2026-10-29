"""No paid calls: exercise transport retries, nested workers and real API handlers with fake IO."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import httpx
import pytest

from muhawir import generate, telemetry
from scripts.analyze_request_traces import read_records, summarize


def records(caplog):
    return [json.loads(r.message.split("muhawir_trace ", 1)[1]) for r in caplog.records
            if r.message.startswith("muhawir_trace ")]


def test_nested_workers_keep_requests_separate_and_log_no_content(monkeypatch, caplog):
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    meeting = threading.Barrier(2)
    def post(*args, **kwargs):
        meeting.wait(timeout=3)
        return httpx.Response(200, json={"choices": [{"message": {"content": "secret answer"}}],
                                       "usage": {"cost": .000123, "prompt_tokens": 17,
                                                 "completion_tokens": 3,
                                                 "prompt_tokens_details": {"cached_tokens": 8}}},
                              request=httpx.Request("POST", "https://openrouter.ai"))
    monkeypatch.setattr(httpx, "post", post)
    call = generate.openai_compatible_call("https://openrouter.ai/api/v1", "test/model", "secret key")
    def request(i):
        trace = telemetry.start("ask")
        with telemetry.bind(trace):
            with telemetry.span("request-" + str(i)):
                with telemetry.ThreadPoolExecutor(max_workers=1) as outer:
                    def nested():
                        with telemetry.ThreadPoolExecutor(max_workers=1) as inner:
                            return inner.submit(call, "secret prompt", "secret question", {}).result()
                    assert outer.submit(nested).result() == "secret answer"
            trace.finish("answered", 200)
            trace.finish("answered", 200)  # one completion only
        return trace.id
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(request, range(2)))
    rows = records(caplog)
    assert {r["request_id"] for r in rows} == set(ids)
    for row in rows:
        events = [e for e in row["events"] if e["kind"] == "model"]
        assert len(events) == 1
        assert events[0]["reported_cost_usd"] == "0.000123"
        assert events[0]["tokens_cached"] == 8
        assert events[0]["duration_ns"] > 0
        assert events[0]["stage"] in ("request-0", "request-1")
    assert "secret" not in caplog.text
    assert telemetry._current.get() is None


def test_transport_retry_and_timeout_are_not_free_successes(monkeypatch, caplog):
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    attempts = 0
    def post(*a, **k):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, request=httpx.Request("POST", "http://fake"))
        raise httpx.ReadTimeout("secret request content")
    monkeypatch.setattr(httpx, "post", post)
    call = generate.openai_compatible_call("https://openrouter.ai/api/v1", "test")
    trace = telemetry.start("ask")
    with telemetry.bind(trace):
        with pytest.raises(httpx.ReadTimeout):
            generate.with_retry(call, generate.CHECK_PROMPT, "secret", {}, sleep=lambda _: None)
        trace.finish("unavailable", 200)
    row = records(caplog)[0]
    events = [e for e in row["events"] if e["kind"] == "model"]
    assert len(events) == 2
    assert [e["http_status"] for e in events] == [429, None]
    assert all(e["reported_cost_usd"] is None and e["outcome"] == "error" for e in events)
    result = summarize([row])
    assert result["calls_with_unknown_cost"] == 2
    assert not result["cost_complete_for_observed_calls"]
    assert result["calls_by_stage"] == {"model.support": 2}
    assert "secret" not in caplog.text


@pytest.mark.parametrize("cost", [None, "NaN", "Infinity", -1, True, "invalid"])
def test_invalid_billing_value_is_unknown(monkeypatch, caplog, cost):
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    @telemetry.model_call("openrouter", "fake")
    def call():
        telemetry.response_metadata(None, {"cost": cost}, usd_cost=True)
    trace = telemetry.start("ask")
    with telemetry.bind(trace):
        call()
        trace.finish("answered", 200)
    assert records(caplog)[0]["events"][0]["reported_cost_usd"] is None


def test_off_by_default(monkeypatch, caplog):
    monkeypatch.delenv("MUHAWIR_TRACE", raising=False)
    assert telemetry.start("ask") is None
    @telemetry.model_call("fake", "fake")
    def call():
        return "ok"
    assert call() == "ok"
    assert records(caplog) == []


@pytest.mark.parametrize("path", ["/api/ask", "/api/ask_stream"])
def test_real_handlers_include_id_and_record_background_work(monkeypatch, caplog, path):
    from fastapi.testclient import TestClient
    from muhawir import server
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    monkeypatch.setattr(server, "_allow_question", lambda _: True)
    monkeypatch.setattr(server, "usage_of", lambda _: None)
    def run(body):
        with telemetry.span("fake_pipeline"):
            telemetry.cache_hit("disk")
            return {"status": "answered"}
    monkeypatch.setattr(server, "_run_ask", run)
    response = TestClient(server.app).post(path, json={"question": "secret question"})
    assert response.status_code == 200
    reply = response.json() if path == "/api/ask" else [json.loads(s)["reply"] for s in response.text.splitlines() if '"reply"' in s][0]
    row = records(caplog)[0]
    assert reply["request_id"] == row["request_id"]
    assert row["cache"] == "disk"
    assert row["style"] == "youth" and row["lang"] == "ar" and row["as_question"] is False
    assert {e["name"] for e in row["events"]} == {"admission", "fake_pipeline"}
    assert "secret question" not in caplog.text


def test_rejected_admission_is_recorded(monkeypatch, caplog):
    from fastapi.testclient import TestClient
    from muhawir import server
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    monkeypatch.setattr(server, "_allow_question", lambda _: False)
    for path in ("/api/ask", "/api/ask_stream"):
        assert TestClient(server.app).post(path, json={"question": "secret"}).status_code == 429
    assert len(records(caplog)) == 2
    assert all(r["http_status"] == 429 and r["status"] == "rejected" for r in records(caplog))


def test_report_rejects_duplicate_ids_and_separates_cache(tmp_path):
    row = {"schema_version": 1, "request_id": "one", "endpoint": "ask", "cache": "none",
           "duration_ns": 10_000_000_000, "http_status": 200, "status": "answered", "events": []}
    cached = {**row, "request_id": "two", "cache": "memory", "duration_ns": 1_000_000}
    rejected = {**row, "request_id": "three", "http_status": 429, "status": "rejected"}
    p = tmp_path / "trace.log"
    p.write_text("unrelated log\n" + "\n".join("prefix muhawir_trace " + json.dumps(r) for r in (row, cached, rejected)))
    report = summarize(read_records(p))
    assert report["completed_requests"] == 3 and report["http_errors"] == 1
    assert len(report["groups"]) == 3
    assert report["groups"]["ask/None/None/None/none/200/answered"]["p95_s"] == 10
    p.write_text(p.read_text() + "\nmuhawir_trace " + json.dumps(row))
    with pytest.raises(ValueError, match="duplicate"):
        read_records(p)


def test_overflow_is_explicit_not_complete_cost(monkeypatch, caplog):
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    monkeypatch.setattr(telemetry, "MAX_EVENTS", 1)
    trace = telemetry.start("ask")
    trace.add({"kind": "span"})
    trace.add({"kind": "model"})
    trace.finish("answered", 200)
    row = records(caplog)[0]
    assert row["events_dropped"] == 1
    assert not summarize([row])["cost_complete_for_observed_calls"]


def test_work_finishing_after_reply_remains_billable_and_correlated(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    release, entered = threading.Event(), threading.Event()
    @telemetry.model_call("openrouter", "fake")
    def call():
        entered.set()
        assert release.wait(timeout=3)
        telemetry.response_metadata(None, {"cost": "0.05"}, usd_cost=True)
    trace = telemetry.start("ask")
    with telemetry.bind(trace):
        with telemetry.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(call)
            assert entered.wait(timeout=3)
            trace.finish("unavailable", 200)
            release.set()
            future.result()
    row = records(caplog)[0]
    assert row["pending_work_at_response"] == 2  # submitted job + physical call
    assert not summarize([row])["cost_complete_for_observed_calls"]
    path = tmp_path / "log"
    path.write_text("\n".join(r.message for r in caplog.records))
    result = summarize(read_records(path))
    assert result["known_reported_cost_usd"] == "0.05"
    assert result["observed_transport_calls"] == 1
    assert result["cost_complete_for_observed_calls"]
    assert result["requests_with_unfinished_background_work"] == 0


def test_mixed_deployments_are_not_pooled():
    with pytest.raises(ValueError, match="mixed code"):
        summarize([{"code_revision": "old"}, {"code_revision": "new"}])


def test_duplicate_late_event_is_rejected(tmp_path):
    row = {"schema_version": 1, "request_id": "one", "duration_ns": 1, "events": []}
    late = {"request_id": "one", "event": {"kind": "model", "event_id": "same"}}
    p = tmp_path / "log"
    p.write_text("muhawir_trace " + json.dumps(row) + "\n" +
                 ("muhawir_trace_event " + json.dumps(late) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate event"):
        read_records(p)
