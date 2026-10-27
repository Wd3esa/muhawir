"""Quote-check: a pasted quotation is located in the stored passages without any model."""
from fastapi.testclient import TestClient

from muhawir import quote_check, server

client = TestClient(server.app)


def _reset():
    server._visitor_calls.clear()


def test_exact_quotation_is_located_with_its_source_and_verbatim_excerpt():
    _reset()
    body = client.post("/api/verify_text", json={"text": "النخلة شجرة تحتاج إلى ماء كثير في الصيف"}).json()
    assert body["status"] == "exact"
    first = body["matches"][0]
    assert first["passage_id"] == "test-a:1" and first["location"] and first["source_name"]
    assert first["excerpt"].startswith("النخلة شجرة") and first["coverage"] == 1.0


def test_diacritics_and_letter_variants_do_not_hide_a_quotation():
    _reset()
    body = client.post("/api/verify_text", json={"text": "النَّخْلَةُ شَجَرَةٌ تَحْتَاجُ اِلَى مَاءٍ كَثِيرٍ"}).json()
    assert body["status"] == "exact" and body["matches"][0]["passage_id"] == "test-a:1"


def test_a_quotation_with_wrong_edges_is_close_not_exact():
    _reset()
    body = client.post("/api/verify_text", json={"text": "قال النبي النخلة شجرة تحتاج إلى ماء كثير في الصيف وغيره"}).json()
    assert body["status"] == "close"
    assert 0.6 <= body["matches"][0]["coverage"] < 1.0


def test_unknown_text_is_not_found_and_short_text_is_refused():
    _reset()
    assert client.post("/api/verify_text", json={"text": "كلام لا يوجد في أي مصدر محفوظ عندنا إطلاقا"}).json() == {"status": "not_found", "matches": []}
    assert client.post("/api/verify_text", json={"text": "الماء"}).status_code == 422


def test_overlong_text_is_rejected():
    _reset()
    assert client.post("/api/verify_text", json={"text": "ماء " * 400}).status_code == 422


def test_quote_check_calls_no_model(monkeypatch):
    _reset()
    def boom(*a, **k):
        raise AssertionError("a model was called")
    monkeypatch.setattr(server.engine, "ask", boom)
    assert client.post("/api/verify_text", json={"text": "النخلة شجرة تحتاج إلى ماء كثير"}).status_code == 200


def test_longest_run_finds_the_unbroken_piece():
    assert quote_check._longest_run(list("abcde"), list("xbcdy")) == (3, 1, 1)


def test_a_phrase_of_common_words_is_called_generic_not_not_found(monkeypatch):
    """«أن رسول الله صلى الله عليه وسلم كان»-like text: in the wording of many passages, so no exact one is named."""
    from types import SimpleNamespace
    _reset()
    fake = [SimpleNamespace(passage=SimpleNamespace(id=f"x:{i}", text="كلام آخر تماما لا يطابق"), score=1.0, coverage=0.9)
            for i in range(12)]
    monkeypatch.setattr(server.engine.retriever, "search", lambda q, k=3: fake[:k])
    body = client.post("/api/verify_text", json={"text": "قال رسول الله صلى الله عليه وسلم كان"}).json()
    assert body == {"status": "too_generic", "matches": []}


def test_few_close_passages_stay_not_found(monkeypatch):
    from types import SimpleNamespace
    _reset()
    fake = [SimpleNamespace(passage=SimpleNamespace(id=f"x:{i}", text="كلام آخر تماما لا يطابق"), score=1.0, coverage=0.3)
            for i in range(12)]
    monkeypatch.setattr(server.engine.retriever, "search", lambda q, k=3: fake[:k])
    assert client.post("/api/verify_text", json={"text": "كلام لا يوجد في أي مصدر محفوظ عندنا إطلاقا"}).json()["status"] == "not_found"


def _hit(pid, kind, source_id, text, location="ص1"):
    from types import SimpleNamespace
    return SimpleNamespace(passage=SimpleNamespace(id=pid, text=text, kind=kind, source_id=source_id, location=location, grade=""),
                           score=1.0, coverage=1.0)


def test_the_quran_comes_first_then_hadith_then_the_other_sources():
    quote = "قل يا عبادي الذين اسرفوا على انفسهم"
    hits = [_hit("t:1", "tafsir", "tabari", "(" + quote + ")"), _hit("h:1", "hadith", "sahih-bukhari", quote),
            _hit("q:1", "quran", "quran", quote), _hit("f:1", "fatwa", "fatwas", quote)]
    class Fake:
        def search(self, q, k=3):
            return hits
    order = [h.passage.kind for _m, h in quote_check.find_quote(Fake(), quote)]
    assert order == ["quran", "hadith", "tafsir", "fatwa"]


def test_the_same_wording_twice_in_one_book_is_one_card_with_its_other_places(monkeypatch):
    _reset()
    quote = "النخلة شجرة تحتاج إلى ماء كثير"
    hits = [_hit("t:1", "tafsir", "test-a", quote, "ص306"), _hit("t:2", "tafsir", "test-a", quote, "ص307")]
    monkeypatch.setattr(server.engine.retriever, "search", lambda q, k=3: hits)
    monkeypatch.setattr(server.engine.corpus, "source_of", lambda p: type("S", (), {"id": "test-a", "name": "كتاب", "url": ""})())
    body = client.post("/api/verify_text", json={"text": quote}).json()
    assert len(body["matches"]) == 1 and body["matches"][0]["location"] == "ص306" and body["matches"][0]["also"] == ["ص307"]


def test_window_cuts_at_word_edges_and_keeps_the_run_intact():
    text = "كلمة " * 100 + "النخلة شجرة تحتاج ماء" + " كلمة" * 100
    start = text.index("النخلة")
    end = start + len("النخلة شجرة تحتاج ماء")
    w = quote_check.window(text, start, end, reach=40)
    assert w["match"] == "النخلة شجرة تحتاج ماء" and w["cut_lead"] and w["cut_trail"]
    assert w["lead"].startswith("كلمة") and w["trail"].endswith("كلمة") and len(w["lead"]) <= 41
    short = quote_check.window("نص قصير جدا", 0, 3, reach=40)
    assert short["lead"] == "" and not short["cut_lead"] and not short["cut_trail"]


def test_neighbour_edges_are_short_and_marked():
    long_text = "كلمة " * 200
    assert quote_check.edge(long_text, 50).endswith("…") and len(quote_check.edge(long_text, 50)) <= 52
    assert quote_check.edge(long_text, 50, tail=True).startswith("…")
    assert quote_check.edge("قصير", 50) == "قصير"


def test_verify_text_returns_the_window_and_the_neighbours():
    _reset()
    body = client.post("/api/verify_text", json={"text": "الماء يتجمد عند درجة الصفر المئوية"}).json()
    first = body["matches"][0]
    assert first["window"]["match"].startswith("الماء يتجمد") and "before" in first and "after" in first
    assert first["before"] is None or first["before"]["text"]


def test_the_page_scripts_are_valid_javascript():
    """A script that does not parse leaves the whole page dead; the other tests never run the page's JavaScript."""
    import re
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path
    node = shutil.which("node")
    if node is None:
        import pytest
        pytest.skip("node is not installed")
    page = (Path(server.__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    for i, code in enumerate(re.findall(r"<script>([\s\S]*?)</script>", page)):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(code)
        result = subprocess.run([node, "--check", f.name], capture_output=True, text=True)
        assert result.returncode == 0, f"script {i}: {result.stderr[:300]}"


VERSE = "قُلْ يَا عِبَادِيَ الَّذِينَ أَسْرَفُوا عَلَىٰ أَنفُسِهِمْ لَا تَقْنَطُوا مِن رَّحْمَةِ اللَّهِ"


def test_a_quotation_with_a_wrong_and_a_misspelt_word_is_close_and_shows_the_true_wording():
    from muhawir.normalize import normalize
    m = quote_check.check_passage(normalize("قل يا عبادي الذين اسرفو في انفسهم").split(), "q:1", VERSE)
    assert m is not None and m.status == "close" and m.coverage >= 0.8
    assert m.excerpt.startswith("قُلْ يَا عِبَادِيَ") and "أَنفُسِهِمْ" in m.excerpt


def test_scattered_common_words_in_a_long_passage_are_not_a_fuzzy_match():
    from muhawir.normalize import normalize
    filler = " ".join(["كلمة"] * 60)
    text = f"قل {filler} يا {filler} عبادي {filler} الذين {filler} انفسهم"
    assert quote_check.check_passage(normalize("قل يا عبادي الذين اسرفو في انفسهم").split(), "q:2", text) is None


def test_one_letter_apart_words_need_four_letters():
    assert quote_check._same_word("اسرفو", "اسرفوا") and quote_check._same_word("يغفر", "تغفر")
    assert not quote_check._same_word("في", "من") and not quote_check._same_word("على", "علي")
