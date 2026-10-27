"""Text from uploaded files and photos: nothing is stored, no model is called."""
import io
import shutil
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from muhawir import extract, quote_check, server
from muhawir.normalize import normalize

FIX = Path(__file__).parent / "fixtures"
client = TestClient(server.app)
needs_ocr = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is not installed")
SECOND = "الصلاة عماد الدين وهي أول ما يحاسب عليه العبد يوم القيامة"
VERSE = "قُلْ يَا عِبَادِيَ الَّذِينَ أَسْرَفُوا عَلَىٰ أَنفُسِهِمْ لَا تَقْنَطُوا مِن رَّحْمَةِ اللَّهِ"


def _reset():
    server._visitor_calls.clear()


def read(name):
    return extract.extract((FIX / name).read_bytes())


@pytest.mark.parametrize("name,kind", [("doc.docx", "zip"), ("doc.odt", "zip"), ("page.html", "text"), ("plain.txt", "text")])
def test_text_formats_give_the_paragraphs_exactly(name, kind):
    r = read(name)
    assert r["kind"] == kind and VERSE in r["text"] and SECOND in r["text"]
    assert "color" not in r["text"] and "var x" not in r["text"]  # style and script are not text


def test_windows_1256_text_is_decoded():
    assert SECOND in read("win1256.txt")["text"]


@needs_ocr
def test_a_photo_is_read_and_a_vowelled_verse_still_finds_its_verse():
    r = read("photo.png")
    assert r["method"] == "ocr" and "عماد الدين" in normalize(r["text"])
    # the vowelled line is read with a few mistakes; the quote-check's tolerance still reaches the verse
    verse_line = next(p for p in extract.pieces(r["text"]) if "عبادي" in normalize(p))
    match = quote_check.check_passage(normalize(verse_line).split(), "q:1", VERSE)
    assert match is not None and match.status == "close" and match.coverage >= 0.7


@needs_ocr
def test_a_pdf_with_text_is_read():
    r = read("layer.pdf")
    assert r["kind"] == "pdf" and "عماد الدين" in normalize(r["text"])


def test_the_kind_is_found_from_the_bytes_not_the_name_or_declared_type():
    assert extract.sniff(b"%PDF-1.7 ...") == "pdf" and extract.sniff(b"PK\x03\x04....") == "zip"
    assert extract.sniff((FIX / "photo.png").read_bytes()) == "image" and extract.sniff("نص".encode()) == "text"


def test_unreadable_and_unsupported_files_are_refused_with_a_code():
    for data, code in [(b"", "unreadable"), (b"\x00\x01\x02" * 500, "unsupported"), (b"   \n  ", "no_text"),
                       (b"PK\x03\x04garbage", "unreadable")]:
        with pytest.raises(extract.ExtractError) as e:
            extract.extract(data)
        assert e.value.code == code


def test_a_zip_that_expands_hugely_is_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", "<w:p><w:r><w:t>x</w:t></w:r></w:p>" + " " * (extract.MAX_UNZIPPED + 10))
    with pytest.raises(extract.ExtractError) as e:
        extract.extract(buffer.getvalue())
    assert e.value.code == "unreadable"


def test_long_text_is_cut_and_marked():
    r = extract.extract(("كلمة عربية " * 5000).encode())
    assert r["truncated"] and len(r["text"]) == extract.MAX_TEXT


def test_pieces_keep_lines_apart_and_cut_long_ones_at_sentence_ends():
    text = "جملة أولى قصيرة هنا. جملة ثانية أطول قليلا من الأولى. كلمتان. " + "كلمة " * 200 + "\nسطر ثان منفصل تماما"
    out = extract.pieces(text, size=120)
    assert out and all(len(p.split()) >= 3 and len(p) <= 125 for p in out)
    assert not any(p == "كلمتان." for p in out) and out[-1] == "سطر ثان منفصل تماما"
    assert extract.pieces("سطر واحد طويل\nسطر آخر مستقل") == ["سطر واحد طويل", "سطر آخر مستقل"]


def test_the_endpoint_returns_text_and_pieces():
    _reset()
    r = client.post("/api/extract", content=(FIX / "doc.docx").read_bytes())
    body = r.json()
    assert r.status_code == 200 and VERSE in body["text"] and body["pieces"] and body["kind"] == "zip"


def test_the_endpoint_maps_errors_to_codes_and_statuses():
    _reset()
    assert client.post("/api/extract", content=b"\x00\x01" * 800).json()["detail"] == "unsupported"
    _reset()
    big = client.post("/api/extract", content=b"a" * (extract.MAX_UPLOAD + 1))
    assert big.status_code == 413 and big.json()["detail"] == "too_large"
    _reset()
    assert client.post("/api/extract", content=b"").status_code == 422


def test_a_busy_server_says_try_again(monkeypatch):
    _reset()
    for _ in range(2):
        server.OCR_SLOTS.acquire()
    try:
        r = client.post("/api/extract", content=(FIX / "plain.txt").read_bytes())
        assert r.status_code == 429 and r.json()["detail"] == "busy"
    finally:
        for _ in range(2):
            server.OCR_SLOTS.release()


def test_the_upload_is_not_kept(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _reset()
    client.post("/api/extract", content=(FIX / "photo.png").read_bytes())
    assert list(tmp_path.iterdir()) == []
