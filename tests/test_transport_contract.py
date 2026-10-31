"""Offline transport contracts: completed envelopes, billing, schema opt-in and pool lifetime."""
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from muhawir import generate, telemetry, transport
from scripts.analyze_request_traces import summarize


def install(monkeypatch, body, status=200):
    sent = []
    def post(url, **kwargs):
        sent.append((url, kwargs))
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))
    monkeypatch.setattr(transport, "post", post)
    return sent


def envelope(reason="stop", content="{}", **message):
    return {"provider": "Together", "choices": [{"finish_reason": reason,
            "message": {"content": content, **message}}],
            "usage": {"cost": "0.00123", "prompt_tokens": 13, "completion_tokens": 7}}


@pytest.mark.parametrize("reason", ["length", "content_filter", "tool_calls", "error", "unexpected", None])
def test_invalid_finish_keeps_billing_and_is_not_retried(monkeypatch, caplog, reason):
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    sent = install(monkeypatch, envelope(reason))
    call = generate.openai_compatible_call("https://openrouter.ai/api/v1", "fake")
    trace = telemetry.start("ask")
    with telemetry.bind(trace):
        with pytest.raises(generate.IncompleteModelResponse):
            generate.with_retry(call, generate.CHECK_PROMPT, "private question", {}, sleep=lambda _: None)
        trace.finish("unavailable", 200)
    assert len(sent) == 1
    row = next(json.loads(r.message.split("muhawir_trace ", 1)[1]) for r in caplog.records
               if r.message.startswith("muhawir_trace "))
    event = row["events"][0]
    assert event["outcome"] == "error"
    assert event["reported_cost_usd"] == "0.00123"
    assert event["tokens_in"] == 13
    assert event["provider"] == "openrouter"
    assert event["upstream_provider"] == "Together"
    expected = "missing" if reason is None else "unknown" if reason == "unexpected" else reason
    assert event["finish_reason"] == expected
    summary = summarize([row])
    assert summary["finish_reasons"] == {expected: 1}
    assert summary["upstream_providers"] == {"Together": 1}
    assert summary["calls_with_unknown_cost"] == 0
    assert "private question" not in caplog.text


@pytest.mark.parametrize("content,extra", [(None, {}), ("", {}), ("  ", {}),
    ([{"type": "text", "text": "{}"}], {}), ("{}", {"refusal": "declined"}),
    ("{}", {"tool_calls": [{"id": "fake"}]}), ("{}", {"function_call": {"name": "fake"}})])
def test_stop_is_insufficient_without_plain_answer(monkeypatch, content, extra):
    install(monkeypatch, envelope(content=content, **extra))
    with pytest.raises(generate.IncompleteModelResponse):
        generate.openai_compatible_call("http://fake/v1", "fake")("system", "user", {})


@pytest.mark.parametrize("choices", [[], None, "bad", [None], [{}]])
def test_missing_choice_fails_closed_with_billing(monkeypatch, caplog, choices):
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    body = envelope(); body["choices"] = choices
    install(monkeypatch, body)
    trace = telemetry.start("ask")
    with telemetry.bind(trace):
        with pytest.raises(generate.IncompleteModelResponse):
            generate.openai_compatible_call("https://openrouter.ai/api/v1", "fake")("s", "u", {})
        trace.finish("unavailable", 200)
    assert '"reported_cost_usd":"0.00123"' in caplog.text.replace(" ", "")


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("base", ["https://openrouter.ai/api/v1", "http://fake/v1"])
def test_schema_mode_is_explicit_and_passes_exact_schema(monkeypatch, strict, base):
    sent = install(monkeypatch, envelope())
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    assert generate.openai_compatible_call(base, "fake", structured_outputs=strict)("s", "u", schema) == "{}"
    body = sent[0][1]["json"]
    assert body["response_format"] == ({"type": "json_schema", "json_schema": {
        "name": "muhawir_reply", "strict": True, "schema": schema}} if strict else {"type": "json_object"})
    assert body.get("provider") == ({"require_parameters": True} if strict and "openrouter.ai" in base else None)


def test_strict_provider_error_does_not_downgrade(monkeypatch):
    sent = install(monkeypatch, {"error": {"message": "unsupported schema"}}, status=400)
    call = generate.openai_compatible_call("https://openrouter.ai/api/v1", "fake", structured_outputs=True)
    with pytest.raises(httpx.HTTPStatusError):
        generate.with_retry(call, "s", "u", {}, sleep=lambda _: None)
    assert len(sent) == 1
    assert sent[0][1]["json"]["response_format"]["type"] == "json_schema"


@pytest.mark.parametrize("primary,backup", [("0", "0"), ("1", "0"), ("0", "1")])
def test_environment_keeps_schema_modes_independent(monkeypatch, primary, backup):
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "model")
    for prefix, mode in [("OPENAI_COMPAT", primary), ("OPENAI_COMPAT2", backup)]:
        monkeypatch.setenv(prefix + "_BASE_URL", "http://fake/v1")
        monkeypatch.setenv(prefix + "_MODEL", "fake")
        monkeypatch.setenv(prefix + "_STRUCTURED_OUTPUTS", mode)
    options = []
    monkeypatch.setattr(generate, "openai_compatible_call", lambda *a, **k: options.append(k))
    monkeypatch.setattr(generate, "ModelGenerator", lambda calls: calls)
    generate.get_generator()
    assert [o["structured_outputs"] for o in options] == [primary == "1", backup == "1"]


@pytest.mark.parametrize("reason", ["end_turn", "max_tokens", "refusal", None])
def test_anthropic_finish_contract(monkeypatch, reason):
    install(monkeypatch, {"stop_reason": reason, "content": [{"type": "text", "text": "{}"}]})
    call = generate.anthropic_call("key", "fake")
    if reason == "end_turn":
        assert call("s", "u", {}) == "{}"
    else:
        with pytest.raises(generate.IncompleteModelResponse): call("s", "u", {})


@pytest.mark.parametrize("reason", ["STOP", "MAX_TOKENS", "SAFETY", None])
def test_gemini_finish_contract(monkeypatch, reason):
    install(monkeypatch, {"candidates": [{"finishReason": reason, "content": {"parts": [
        {"thought": True, "text": "private reasoning"}, {"text": "{}"}]}}]})
    call = generate.gemini_call("key", "fake")
    if reason == "STOP":
        assert call("s", "u", {}) == "{}"
    else:
        with pytest.raises(generate.IncompleteModelResponse): call("s", "u", {})


def test_pool_shared_across_factories_and_threads_without_sharing_credentials(monkeypatch):
    transport.close()
    original = httpx.Client
    created, requests = [], []
    def factory(**kwargs):
        def handle(request):
            requests.append(request)
            return httpx.Response(200, json=envelope())
        obj = original(transport=httpx.MockTransport(handle), **kwargs)
        created.append(obj)
        return obj
    monkeypatch.setattr(httpx, "Client", factory)
    first = generate.openai_compatible_call("http://fake/v1", "a", "key-a", timeout=2)
    second = generate.openai_compatible_call("http://fake/v1", "b", "key-b", timeout=7)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(lambda f: f("s", "u", {}), [first, second] * 8)) == ["{}"] * 16
        assert len(created) == 1
        assert sorted(r.headers["authorization"] for r in requests) == ["Bearer key-a"] * 8 + ["Bearer key-b"] * 8
        for request in requests:
            expected = 2 if request.headers["authorization"] == "Bearer key-a" else 7
            assert request.extensions["timeout"]["read"] == expected
        transport.close()
        assert created[0].is_closed
        first("s", "u", {})
        assert len(created) == 2 and not created[1].is_closed
    finally:
        transport.close()


def test_server_shutdown_closes_pool(monkeypatch):
    from muhawir import server
    from fastapi.testclient import TestClient
    closed = []
    monkeypatch.setattr(transport, "close", lambda: closed.append(True))
    with TestClient(server.app):
        assert not closed
    assert closed == [True]


@pytest.mark.real_check
def test_well_formed_positive_json_cannot_bypass_incomplete_transport(monkeypatch):
    from muhawir.corpus import Passage
    from muhawir.verify import Claim
    passage = Passage("synthetic:1", "synthetic", "test", "The box is blue.", "other")
    claim = Claim(passage.text, (passage.id,))
    positive = {"on_topic": [True], "missing": [[]], "evidence": [passage.text], "problem": ["none"]}
    assert generate._valid_support_reply(positive, 1)
    sent = install(monkeypatch, envelope("length", json.dumps(positive)))
    call = generate.openai_compatible_call("http://fake/v1", "fake")
    monkeypatch.setattr(generate, "READ_EACH_SENTENCE", True)
    gen = generate.ModelGenerator([("fake-incomplete", call)])
    memo = {}
    assert gen.check_support([claim], {passage.id: passage}, memo=memo) is None
    assert memo == {} and len(sent) == 2  # existing bounded unusable-reading retry, no approval
