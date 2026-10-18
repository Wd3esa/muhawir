from fastapi.testclient import TestClient

from muhawir.server import app

client = TestClient(app)


def test_health():
    body = client.get("/api/health").json()
    assert body["ok"] and body["generator"] == "extractive" and body["synthetic"] is True


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
