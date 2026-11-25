"""The local model slot (Ollama on the server): no key, thinking off, the context window set, last in line."""
import pytest

from muhawir import budget, generate


class _Reply:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def _clear(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_MODEL", "OPENAI_COMPAT2_BASE_URL",
              "OPENAI_COMPAT2_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL", "OLLAMA_MODEL", "OLLAMA_URL",
              "OLLAMA_NUM_CTX", "OLLAMA_SCHEMA", "MUHAWIR_ANSWER_DEADLINE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "model")


def test_request_turns_thinking_off_sets_the_window_and_the_schema(monkeypatch):
    sent = {}

    def post(url, json, timeout):
        sent.update(url=url, body=json, timeout=timeout)
        return _Reply({"message": {"content": '{"ok": true}'}, "prompt_eval_count": 4000, "eval_count": 30})

    monkeypatch.setattr("httpx.post", post)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    call = generate.ollama_call("qwen3:8b", "http://localhost:11434/", num_ctx=16384, timeout=900)
    assert call("system", "user", schema) == '{"ok": true}'
    body = sent["body"]
    assert sent["url"] == "http://localhost:11434/api/chat"  # Ollama's own API, not its chat-completions copy
    assert body["think"] is False and body["stream"] is False and body["format"]["required"] == ["ok"]
    assert body["options"]["num_ctx"] == 16384 and body["options"]["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_schema_can_be_switched_off(monkeypatch):
    sent = {}
    monkeypatch.setattr("httpx.post", lambda url, json, timeout: sent.update(json) or _Reply({"message": {"content": "{}"}}))
    generate.ollama_call("m", schema_on=False)("s", "u", {"type": "object"})
    assert sent["format"] == "json"


def test_local_model_alone_is_enough_and_gets_a_longer_answer_deadline(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setattr(budget, "ANSWER_SECONDS", 120.0)
    gen = generate.get_generator()
    assert [name for name, _ in gen.calls] == ["local"]
    assert budget.ANSWER_SECONDS == generate.LOCAL_ANSWER_DEADLINE


def test_an_owner_deadline_is_kept(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("MUHAWIR_ANSWER_DEADLINE", "300")
    monkeypatch.setattr(budget, "ANSWER_SECONDS", 300.0)
    generate.get_generator()
    assert budget.ANSWER_SECONDS == 300.0


def test_local_model_comes_after_the_hosted_ones(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("GEMINI_MODEL", "g")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setattr(budget, "ANSWER_SECONDS", 120.0)
    assert [name for name, _ in generate.get_generator().calls] == ["gemini", "local"]


def test_no_model_at_all_is_still_an_error(monkeypatch):
    _clear(monkeypatch)
    with pytest.raises(RuntimeError, match="OLLAMA_MODEL"):
        generate.get_generator()


def test_every_unbounded_list_gets_a_maximum_and_a_set_one_is_kept():
    schema = {"type": "object", "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
        "request": {"type": "object", "properties": {"parts": {"type": "array", "maxItems": 6, "items": {}}}},
        "answer": {"type": "array", "items": {"type": "object", "properties": {"ids": {"type": "array"}}}}}}
    out = generate.bounded_schema(schema, 20)
    assert out["properties"]["queries"]["maxItems"] == 20
    assert out["properties"]["request"]["properties"]["parts"]["maxItems"] == 6
    assert out["properties"]["answer"]["maxItems"] == 20
    assert out["properties"]["answer"]["items"]["properties"]["ids"]["maxItems"] == 20
    assert "maxItems" not in schema["properties"]["queries"]  # the shared schema is not changed


def test_the_real_understanding_schema_is_bounded_in_the_request(monkeypatch):
    sent = {}
    monkeypatch.setattr("httpx.post", lambda url, json, timeout: sent.update(json) or _Reply({"message": {"content": "{}"}}))
    generate.ollama_call("m", max_items=10)("s", "u", generate.UNDERSTAND_SCHEMA)
    assert sent["format"]["properties"]["queries"]["maxItems"] == 10
