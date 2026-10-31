from fastapi.testclient import TestClient

from muhawir.server import app

client = TestClient(app)


def test_health():
    body = client.get("/api/health").json()
    assert body["ok"] and body["generator"] == "extractive" and body["synthetic"] is True


def test_health_identifies_all_application_code_without_a_model_call(monkeypatch):
    from muhawir import server, telemetry
    monkeypatch.setattr(server, "credit_remaining", lambda: None)
    def unexpected(*args, **kwargs):
        raise AssertionError("health must not generate an answer")
    monkeypatch.setattr(server.engine.generator, "generate", unexpected)
    body = client.get("/api/health").json()
    assert body["code_revision"] == telemetry.CODE_REVISION
    assert len(body["code_revision"]) == 64
    assert body["pipeline_revision"] == server.PIPELINE_REVISION


def test_ask_answers_with_sources():
    body = client.post("/api/ask", json={"question": "ماذا تحتاج النخلة في الصيف؟"}).json()
    assert body["status"] == "answered" and body["sources"]


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200 and "مُحاور" in r.text


def test_health_shows_the_credit_left_on_an_openrouter_key(monkeypatch):
    import httpx
    from muhawir import server

    class Reply:
        status_code = 200
        def json(self):
            return {"data": {"limit": 5, "usage": 4.2, "limit_remaining": 0.8}}
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Reply())
    server._credit.update(at=0.0, value=None)
    assert server.credit_remaining() == 0.8
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://ollama.com/v1")
    assert server.credit_remaining() is None  # not OpenRouter: unknown


def test_one_visitor_asking_too_fast_is_told_to_wait(monkeypatch):
    from fastapi.testclient import TestClient
    from muhawir import server
    monkeypatch.setattr(server, "ASK_LIMITS", ((60, 3),))
    monkeypatch.setattr(server, "_visitor_calls", server.defaultdict(server.deque))
    client = TestClient(server.app)
    codes = [client.post("/api/ask", json={"question": "ماذا تحتاج النخلة في الصيف؟"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_a_visitor_is_told_when_four_fifths_of_a_limit_are_used(monkeypatch):
    """22 October 2026: «استخدمت 80% من حد الأسئلة… يتجدد بعد…», like a usage notice."""
    import time
    from muhawir import server
    monkeypatch.setattr(server, "ASK_LIMITS", ((60, 10), (3600, 200)))
    now = time.monotonic()
    monkeypatch.setitem(server._visitor_calls, "9.9.9.9", __import__("collections").deque([now - 5] * 7))
    assert server.usage_of("9.9.9.9") is None  # 70%: nothing yet
    server._visitor_calls["9.9.9.9"].append(now)
    u = server.usage_of("9.9.9.9")
    assert u["used"] == 8 and u["limit"] == 10 and 50 <= u["resets_in"] <= 56


def test_ask_stream_sends_stages_then_the_reply():
    import json
    lines = [json.loads(x) for x in client.post(
        "/api/ask_stream", json={"question": "ماذا تحتاج النخلة في الصيف؟"}).text.splitlines() if x.strip()]
    assert lines[0] == {"stage": "understanding"}
    assert "reply" in lines[-1] and lines[-1]["reply"]["status"]


def test_ask_stream_refuses_when_over_the_limit():
    codes = [client.post("/api/ask_stream", json={"question": "ماذا تحتاج النخلة في الصيف؟"}).status_code for _ in range(30)]
    assert 429 in codes


def test_ready_reports_parts():
    r = client.get("/api/ready")
    body = r.json()
    assert set(body["parts"]) == {"corpus", "vectors", "models", "disk"}
    assert r.status_code == (200 if body["ok"] else 503)


def test_breaker_pauses_a_failing_model_when_another_can_answer():
    from muhawir import generate
    now = [0.0]
    calls = []

    def failing(system, user, schema):
        calls.append(1)
        raise RuntimeError("down")

    guarded = generate.with_breaker("t-down", failing, others=1, clock=lambda: now[0])
    for _ in range(generate.BREAKER_FAILURES):
        try:
            guarded("s", "u", {})
        except RuntimeError:
            pass
    try:
        guarded("s", "u", {})
    except generate.CircuitOpen:
        pass
    else:
        raise AssertionError("not paused")
    assert len(calls) == generate.BREAKER_FAILURES
    now[0] += generate.BREAKER_COOLDOWN + 1  # after the cooldown it is tried again
    try:
        guarded("s", "u", {})
    except RuntimeError:
        pass
    assert len(calls) == generate.BREAKER_FAILURES + 1


def test_breaker_never_pauses_the_only_model():
    from muhawir import generate

    def failing(system, user, schema):
        raise RuntimeError("down")

    guarded = generate.with_breaker("t-only", failing, others=0)
    for _ in range(generate.BREAKER_FAILURES + 2):
        try:
            guarded("s", "u", {})
        except RuntimeError:
            pass
        except generate.CircuitOpen:
            raise AssertionError("the only model was paused")


def test_ask_stream_sends_stages_then_the_reply_and_no_unchecked_text():
    import json
    from muhawir import server
    server._visitor_calls.clear()
    lines = [json.loads(x) for x in client.post(
        "/api/ask_stream", json={"question": "ماذا تحتاج النخلة في الصيف؟"}).text.splitlines() if x.strip()]
    assert "reply" in lines[-1]
    assert all(item.get("stage") != "found" for item in lines)  # passages are shown only as the reply's checked sources


def test_tashkeel_result_is_kept_and_a_failure_names_its_reason(monkeypatch):
    from muhawir import server, tashkeel
    server._vowelled.clear()
    calls = []

    def fake(texts, *a, **k):
        calls.append(1)
        return {"texts": list(texts), "rejected": 0, "automatic": True}

    monkeypatch.setattr(tashkeel, "diacritize", fake)
    body = {"texts": ["قول"], "passage_ids": []}
    assert client.post("/api/tashkeel", json=body).status_code == 200
    assert client.post("/api/tashkeel", json=body).status_code == 200
    assert len(calls) == 1  # the second answer comes from the kept result

    def busy(*a, **k):
        raise tashkeel.Unavailable("Tashkeel rate limit")

    monkeypatch.setattr(tashkeel, "diacritize", busy)
    r = client.post("/api/tashkeel", json={"texts": ["نص آخر"], "passage_ids": []})
    assert r.status_code == 503 and "rate limit" in r.json()["detail"]


def test_ask_stream_names_the_books_it_searches():
    import json
    from muhawir import server
    server._visitor_calls.clear()
    server.engine._cache.clear()  # a reply kept from an earlier test would skip the search
    lines = [json.loads(x) for x in client.post(
        "/api/ask_stream", json={"question": "ماذا تحتاج النخلة في الصيف؟"}).text.splitlines() if x.strip()]
    names = [item["data"] for item in lines if item.get("stage") == "source"]
    assert names and all(isinstance(n, str) and n for group in names for n in group)
def test_book_answer_assets_are_served():
    page = client.get("/")
    css = client.get("/book-answer.css?v=question-divider-6")
    assert page.status_code == 200 and 'id="explanationPicker"' in page.text
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert ".msg.plain-answer" in css.text
    for name in ("Amiri-Regular.ttf", "Amiri-Bold.ttf"):
        font = client.get("/fonts/" + name)
        assert font.status_code == 200 and font.content[:4] == b"\x00\x01\x00\x00"


