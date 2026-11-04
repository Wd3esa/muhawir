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


TEXT = "اللهم يا فارج الهم ويا كاشف الغم فرّج همي ويسّر أمري"


def _gemini_sse(*pieces):
    """A streamed Gemini answer: one server-sent event per piece of text."""
    import json
    return "".join("data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": p}]}}]}, ensure_ascii=False)
                   + "\n\n" for p in pieces).encode()


def _compat_sse(*pieces):
    import json
    return ("".join("data: " + json.dumps({"choices": [{"delta": {"content": p}}]}, ensure_ascii=False) + "\n\n"
                    for p in pieces) + "data: [DONE]\n\n").encode()


def _serve(monkeypatch, handler):
    """Every vision request goes to `handler(request) -> httpx.Response`; returns the list of requests seen."""
    seen = []
    def handle(request):
        seen.append(request)
        return handler(request)
    monkeypatch.setattr(extract, "_client", lambda: httpx.Client(transport=httpx.MockTransport(handle)))
    return seen


@pytest.fixture(autouse=True)
def _gemini(monkeypatch):
    monkeypatch.delenv("MUHAWIR_VISION_OCR", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_VISION_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("vision reading must stream, not wait for the whole page"))


def test_blank_photo_is_read_by_the_vision_model_when_gemini_is_set(monkeypatch):
    seen = _serve(monkeypatch, lambda r: httpx.Response(200, content=_gemini_sse(TEXT)))
    got = extract.extract(_photo())
    assert got["text"].startswith("اللهم يا فارج الهم") and got["method"] == "ocr"
    assert ":streamGenerateContent?alt=sse" in str(seen[0].url)


def test_a_long_page_arriving_in_pieces_is_read_whole(monkeypatch):
    # the cause of the live ReadTimeout: a full page took longer than one wait; streamed, only silence counts
    pieces = [w + " " for w in TEXT.split()]
    _serve(monkeypatch, lambda r: httpx.Response(200, content=_gemini_sse(*pieces)))
    assert extract.extract(_photo())["text"].startswith(TEXT)


def test_turned_off_no_picture_leaves_the_server(monkeypatch):
    monkeypatch.setenv("MUHAWIR_VISION_OCR", "0"); monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m")
    _serve(monkeypatch, lambda r: pytest.fail("the picture was sent to a model"))
    with pytest.raises(extract.ExtractError) as err:
        extract.extract(_photo())
    # «no_text» where Tesseract has Arabic data, «ocr_unavailable» where it has none (the CI machine): either way the
    # picture never left the server
    assert err.value.code in ("no_text", "ocr_unavailable")


def test_a_failing_vision_call_reports_vision_failed_not_unreadable(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("down")
    _serve(monkeypatch, handler)
    with pytest.raises(extract.ExtractError) as err:
        extract.extract(_photo())
    assert err.value.code == "vision_failed" and "m ConnectError" in err.value.note


def test_a_model_silent_too_long_is_reported_by_name(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("silent")
    _serve(monkeypatch, handler)
    with pytest.raises(extract.ExtractError) as err:
        extract.extract(_photo())
    assert err.value.code == "vision_failed" and "m ReadTimeout" in err.value.note


def test_a_flash_model_is_asked_without_thinking_and_retried_plain_on_a_refusal(monkeypatch):
    import json
    thinking = []
    def handler(request):
        thinking.append("thinkingConfig" in json.loads(request.content)["generationConfig"])
        return httpx.Response(400, json={}) if len(thinking) == 1 else httpx.Response(200, content=_gemini_sse(TEXT))
    _serve(monkeypatch, handler)
    assert extract.extract(_photo())["text"].startswith("اللهم") and thinking == [True, False]


def test_a_busy_model_is_replaced_by_the_next_at_once(monkeypatch):
    seen = _serve(monkeypatch, lambda r: httpx.Response(503, json={}) if "/m:" in str(r.url)
                  else httpx.Response(200, content=_gemini_sse(TEXT)))
    assert extract.extract(_photo())["text"].startswith("اللهم")
    assert "/m:" in str(seen[0].url) and "gemini-2.5-flash" in str(seen[-1].url)


def test_a_slow_model_does_not_hold_up_the_next(monkeypatch):
    import threading
    import time
    monkeypatch.setattr(extract, "VISION_HEDGE", 0.2)
    release = threading.Event()
    def handler(request):
        if "/m:" in str(request.url):
            release.wait(3)  # still thinking
            return httpx.Response(200, content=_gemini_sse("متأخر"))
        return httpx.Response(200, content=_gemini_sse(TEXT))
    _serve(monkeypatch, handler)
    started = time.monotonic()
    assert extract.extract(_photo())["text"].startswith("اللهم")
    assert time.monotonic() - started < 2
    release.set()


def test_when_every_gemini_model_fails_the_compat_endpoint_reads_the_picture(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://openrouter.ai/api/v1"); monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "x")
    def handler(request):
        if "googleapis" in str(request.url):
            return httpx.Response(503, json={})
        import json
        body = json.loads(request.content)
        assert str(request.url).endswith("/chat/completions") and body["stream"] is True
        assert body["messages"][0]["content"][1]["type"] == "image_url"
        return httpx.Response(200, content=_compat_sse(*[w + " " for w in TEXT.split()]))
    _serve(monkeypatch, handler)
    assert extract.extract(_photo())["text"] == TEXT


def test_when_every_model_fails_each_is_named_with_its_status(monkeypatch):
    _serve(monkeypatch, lambda r: httpx.Response(503, json={}))
    with pytest.raises(extract.ExtractError) as err:
        extract.extract(_photo())
    assert err.value.code == "vision_failed"
    assert "m HTTP 503" in err.value.note and "gemini-2.5-flash HTTP 503" in err.value.note
