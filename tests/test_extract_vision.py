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
