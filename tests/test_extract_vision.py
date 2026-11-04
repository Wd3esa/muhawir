"""A photo Tesseract cannot read (handwriting) goes to the vision model when Gemini is configured and not turned off."""
import io

import httpx
import pytest
from PIL import Image

from muhawir import extract


def _photo() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (1000, 1000), "white").save(out, "PNG")
    return out.getvalue()


def _fake_post(text):
    def post(url, headers=None, json=None, timeout=None):
        assert json["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/jpeg"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]},
                              request=httpx.Request("POST", url))
    return post


def test_blank_photo_is_read_by_the_vision_model_when_gemini_is_set(monkeypatch):
    monkeypatch.delenv("MUHAWIR_VISION_OCR", raising=False); monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setattr(httpx, "post", _fake_post("اللهم يا فارج الهم ويا كاشف الغم فرّج همي ويسّر أمري"))
    got = extract.extract(_photo())
    assert got["text"].startswith("اللهم يا فارج الهم") and got["method"] == "ocr"


def test_turned_off_no_picture_leaves_the_server(monkeypatch):
    monkeypatch.setenv("MUHAWIR_VISION_OCR", "0"); monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("the picture was sent to a model"))
    with pytest.raises(extract.ExtractError) as err:
        extract.extract(_photo())
    assert str(err.value) == "no_text" or err.value.args[0] == "no_text"


def test_a_failing_vision_call_falls_back_to_no_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m")
    def boom(*a, **k): raise httpx.ConnectError("down")
    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(extract.ExtractError):
        extract.extract(_photo())


def test_a_failing_vision_call_reports_vision_failed_not_unreadable(monkeypatch):
    monkeypatch.delenv("MUHAWIR_VISION_OCR", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(httpx.ReadTimeout("The read operation timed out")))
    with pytest.raises(extract.ExtractError) as err:
        extract.extract(_photo())
    assert err.value.code == "vision_failed"


def test_a_flash_model_is_asked_without_thinking_and_retried_plain_on_a_refusal(monkeypatch):
    monkeypatch.delenv("MUHAWIR_VISION_OCR", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    seen = []
    def post(url, headers=None, json=None, timeout=None):
        seen.append("thinkingConfig" in json["generationConfig"])
        request = httpx.Request("POST", url)
        if len(seen) == 1:
            return httpx.Response(400, json={"error": "no thinking here"}, request=request)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "اللهم يا فارج الهم ويا كاشف الغم"}]}}]}, request=request)
    monkeypatch.setattr(httpx, "post", post)
    assert extract.extract(_photo())["text"].startswith("اللهم")
    assert seen == [True, False]


def test_a_busy_model_service_is_asked_again(monkeypatch):
    import time
    monkeypatch.delenv("MUHAWIR_VISION_OCR", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = []
    def post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        request = httpx.Request("POST", url)
        if len(calls) < 3:
            return httpx.Response(503, json={"error": {"status": "UNAVAILABLE"}}, request=request)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "اللهم يا فارج الهم ويا كاشف الغم"}]}}]}, request=request)
    monkeypatch.setattr(httpx, "post", post)
    assert extract.extract(_photo())["text"].startswith("اللهم") and len(calls) == 3


def test_a_service_that_stays_busy_gives_up_with_its_status(monkeypatch):
    import time
    monkeypatch.delenv("MUHAWIR_VISION_OCR", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(httpx, "post", lambda url, **k: httpx.Response(503, json={}, request=httpx.Request("POST", url)))
    with pytest.raises(extract.ExtractError) as err:
        extract.extract(_photo())
    assert err.value.code == "vision_failed" and err.value.note.startswith("m HTTP 503") and "gemini-2.5-flash HTTP 503" in err.value.note


def test_a_slow_answer_is_asked_once_more(monkeypatch):
    monkeypatch.delenv("MUHAWIR_VISION_OCR", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m")
    calls = []
    def post(url, headers=None, json=None, timeout=None):
        calls.append(json["generationConfig"])
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "اللهم يا فارج الهم ويا كاشف الغم"}]}}]}, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", post)
    assert extract.extract(_photo())["text"].startswith("اللهم") and len(calls) == 2
    assert "thinkingConfig" in calls[0]  # a model of any name is asked not to think first


def test_an_overloaded_model_is_replaced_by_the_next(monkeypatch):
    import time
    monkeypatch.delenv("MUHAWIR_VISION_OCR", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "big")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    urls = []
    def post(url, headers=None, json=None, timeout=None):
        urls.append(url)
        request = httpx.Request("POST", url)
        if "/big:" in url:
            return httpx.Response(503, json={}, request=request)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "اللهم يا فارج الهم ويا كاشف الغم"}]}}]}, request=request)
    monkeypatch.setattr(httpx, "post", post)
    assert extract.extract(_photo())["text"].startswith("اللهم")
    assert "/big:" in urls[0] and "gemini-2.5-flash" in urls[-1]


def test_when_every_gemini_model_is_busy_the_compat_endpoint_reads_the_picture(monkeypatch):
    import time
    monkeypatch.delenv("MUHAWIR_VISION_OCR", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "big")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://openrouter.ai/api/v1"); monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "x")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    def post(url, headers=None, json=None, timeout=None):
        request = httpx.Request("POST", url)
        if "googleapis" in url:
            return httpx.Response(503, json={}, request=request)
        assert url.endswith("/chat/completions") and json["messages"][0]["content"][1]["type"] == "image_url"
        return httpx.Response(200, json={"choices": [{"message": {"content": "اللهم يا فارج الهم ويا كاشف الغم"}}]}, request=request)
    monkeypatch.setattr(httpx, "post", post)
    assert extract.extract(_photo())["text"].startswith("اللهم")
