"""Muhawir over MCP: the sources for other assistants' models. Synthetic data; no model call, no network."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("mcp")
from muhawir import mcp_server  # noqa: E402

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}}


def _client(monkeypatch, token="", ask=False, per_minute=120, asked=None):
    from muhawir import server
    monkeypatch.setattr(mcp_server, "ENABLED", True)
    monkeypatch.setattr(mcp_server, "TOKEN", token)
    monkeypatch.setattr(mcp_server, "ASK_ALLOWED", bool(token) or ask)
    monkeypatch.setattr(mcp_server, "PER_MINUTE", per_minute)
    run_ask = lambda q, s, l: (asked.append((q, s, l)) if asked is not None else None) or {  # noqa: E731
        "status": "answered", "message": "", "claims": [], "sources": [], "debug_only": "x"}
    app = FastAPI()
    mcp_server.mount(app, mcp_server.build(server.engine, run_ask))
    return TestClient(app)


def _payload(response):
    """The tool's result: its structured content when given, else the JSON text it returned."""
    result = response.json()["result"]
    if "structuredContent" in result:
        data = result["structuredContent"]
        return data.get("result", data) if isinstance(data, dict) and set(data) == {"result"} else data
    return __import__("json").loads(result["content"][0]["text"])


def _call(client, method, params=None, headers=HEADERS, id=2):
    return client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}})


def test_mcp_is_off_unless_switched_on(monkeypatch):
    from muhawir import server
    monkeypatch.setattr(mcp_server, "ENABLED", False)
    assert mcp_server.build(server.engine, lambda *a: {}) is None


def test_the_free_tools_work_and_asking_muhawir_needs_a_token_or_a_switch(monkeypatch):
    with _client(monkeypatch) as client:
        assert client.post("/mcp", headers=HEADERS, json=INIT).status_code == 200
        names = [t["name"] for t in _call(client, "tools/list").json()["result"]["tools"]]
        assert names == ["search_sources", "read_passage", "find_quote"]
        found = _call(client, "tools/call", {"name": "search_sources", "arguments": {"query": "النخلة في الصيف"}})
        result = _payload(found)
        assert result[0]["passage_id"] == "test-a:1" and result[0]["source"] and result[0]["location"]
        read = _call(client, "tools/call", {"name": "read_passage", "arguments": {"passage_id": "test-a:2"}})
        passage = _payload(read)
        assert passage["before"]["passage_id"] == "test-a:1" and passage["text"].startswith("يظهر القمر")


def test_a_token_guards_every_request_and_opens_ask_muhawir(monkeypatch):
    asked = []
    with _client(monkeypatch, token="s3cret", asked=asked) as client:
        assert client.post("/mcp", headers=HEADERS, json=INIT).status_code == 401
        assert client.post("/mcp", headers={**HEADERS, "Authorization": "Bearer wrong"}, json=INIT).status_code == 401
        auth = {**HEADERS, "Authorization": "Bearer s3cret"}
        names = [t["name"] for t in _call(client, "tools/list", headers=auth).json()["result"]["tools"]]
        assert "ask_muhawir" in names
        reply = _call(client, "tools/call", {"name": "ask_muhawir",
                                             "arguments": {"question": "سؤال", "style": "bad", "lang": "xx"}}, headers=auth)
        answer = _payload(reply)
        assert answer["status"] == "answered" and "debug_only" not in answer
        assert asked == [("سؤال", "youth", "ar")]  # unknown style and language fall back to the defaults


def test_all_tool_calls_share_one_budget_per_minute(monkeypatch):
    with _client(monkeypatch, per_minute=1) as client:
        args = {"name": "read_passage", "arguments": {"passage_id": "test-a:1", "around": False}}
        assert not _call(client, "tools/call", args).json()["result"]["isError"]
        assert _call(client, "tools/call", args, id=3).json()["result"]["isError"]
