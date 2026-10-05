"""Optional neural speech and its browser fallback contract."""
import asyncio
import xml.etree.ElementTree as ET

import httpx
import pytest
from fastapi.testclient import TestClient

from muhawir import tts
from muhawir.server import _tts_calls, app

client = TestClient(app)


@pytest.mark.parametrize("missing", ["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION"])
def test_missing_azure_settings_returns_503(monkeypatch, missing):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")
    monkeypatch.delenv(missing)
    response = client.post("/tts", json={"text": "An answer", "voice": "female", "lang": "en"})
    assert response.status_code == 503


@pytest.mark.parametrize("lang,voice,expected", [
    ("ar", "male", "ar-SA-HamedNeural"),
    ("ar", "female", "ar-SA-ZariyahNeural"),
    ("en", "male", "en-US-AndrewNeural"),
    ("en", "female", "en-US-AvaNeural"),
])
def test_ssml_escapes_text_and_replaces_verses(lang, voice, expected):
    text = "A & B < C > D ﴿سورة واختبار﴾ end"
    output = tts.ssml(text, voice, lang)
    root = ET.fromstring(output)
    spoken = "".join(root.itertext())
    assert "سورة واختبار" not in spoken
    assert ("(آية)" if lang == "ar" else "(verse)") in spoken
    assert "A & B < C > D" in spoken
    assert "&amp;" in output and "&lt;" in output and "&gt;" in output
    assert expected in output
    assert "سورة واختبار" not in tts.replace_verses("﴿سورة واختبار", lang)


def test_long_text_rejected_before_synthesis(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")
    response = client.post("/tts", json={"text": "x" * 3001, "voice": "male", "lang": "en"})
    assert response.status_code == 422


def test_cache_hit_does_not_call_azure_again(monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")
    monkeypatch.setattr(tts, "CACHE_DIR", tmp_path / "tts-cache")
    calls = []

    def azure(request):
        calls.append(request)
        assert request.url.host == "eastus.tts.speech.microsoft.com"
        assert request.headers["Ocp-Apim-Subscription-Key"] == "test-key"
        assert request.headers["X-Microsoft-OutputFormat"] == tts.OUTPUT_FORMAT
        assert "سورة" not in request.content.decode("utf-8")
        return httpx.Response(200, content=b"ID3mock-audio", headers={"content-type": "audio/mpeg"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(tts.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(azure)))
    first = asyncio.run(tts.synthesize("شرح ﴿سورة﴾ موجز", "male", "ar"))
    second = asyncio.run(tts.synthesize("شرح ﴿سورة﴾ موجز", "male", "ar"))
    assert first == second == b"ID3mock-audio"
    assert len(calls) == 1
    assert len(list(tts.CACHE_DIR.glob("*.mp3"))) == 1


def test_rate_limit_per_ip(monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")
    monkeypatch.setattr(tts, "CACHE_DIR", tmp_path / "tts-cache")
    async def fake_azure(text, voice, lang):
        return b"ID3mock-audio"
    monkeypatch.setattr(tts, "_request_azure", fake_azure)
    _tts_calls.clear()
    body = {"text": "An answer", "voice": "female", "lang": "en"}
    assert all(client.post("/tts", json=body).status_code == 200 for _ in range(20))
    assert client.post("/tts", json=body).status_code == 429
    _tts_calls.clear()
