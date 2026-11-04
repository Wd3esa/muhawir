"""Domain routing (November 2026): the understanding step names the question's field, and that field's books are
searched first, three for each passage from another field. Synthetic data and no paid model call."""
import json

from muhawir import pipeline
from muhawir.corpus import in_domain, parse_corpus
from muhawir.generate import UNDERSTAND_SCHEMA, ModelGenerator
from muhawir.pipeline import Muhawir

_TEXT = "الوضوء غسل الوجه واليدين ومسح الرأس وغسل الرجلين وهو شرط لصحة الصلاة عند أهل العلم كلهم بلا خلاف بينهم في ذلك."
CORPUS = parse_corpus({"synthetic": True, "sources": [
    {"id": "tafsir-x", "name": "تفسير", "about": "تفسير"}, {"id": "fiqh-x", "name": "فقه", "about": "فقه"},
    {"id": "extra-zad-maad", "name": "زاد المعاد", "about": "سيرة وفقه"}], "passages": [
    *[{"id": f"t:{i}", "source_id": "tafsir-x", "location": f"ت {i}", "kind": "tafsir", "text": _TEXT + " الوضوء"} for i in range(12)],
    *[{"id": f"f:{i}", "source_id": "fiqh-x", "location": f"ف {i}", "kind": "fiqh", "text": _TEXT} for i in range(12)],
    {"id": "z:1", "source_id": "extra-zad-maad", "location": "هديه في الوضوء", "kind": "other", "text": _TEXT}]})


def _engine():
    return Muhawir(CORPUS, ModelGenerator([("m", lambda *a, **k: "{}")]))


def test_the_understanding_step_is_asked_for_the_field():
    request = UNDERSTAND_SCHEMA["properties"]["answer_request"]
    assert "domain" in request["required"] and "fiqh" in request["properties"]["domain"]["enum"]


def test_a_book_added_as_other_belongs_to_its_field():
    zad = next(p for p in CORPUS.passages if p.id == "z:1")
    assert in_domain(zad, "fiqh") and in_domain(zad, "seerah") and not in_domain(zad, "quran")


def test_the_field_books_come_first_with_one_other_passage_after_each_three(monkeypatch):
    monkeypatch.setattr(pipeline, "MODEL_CANDIDATES", 8)
    plain, _ = _engine()._gather("الوضوء", "", ["الوضوء"])
    routed, _ = _engine()._gather("الوضوء", "", ["الوضوء"], domain="fiqh")
    assert [p.kind for p in plain][:4] == ["tafsir"] * 4  # «الوضوء» twice ranks the tafsir passages first
    kinds = ["fiqh" if in_domain(p, "fiqh") else "other" for p in routed]
    assert kinds[:4] == ["fiqh", "fiqh", "fiqh", "other"] and kinds.count("fiqh") >= 6


def test_the_page_is_told_what_is_looked_for_and_where(monkeypatch):
    def call(system, user, schema=None):
        if '"question"' in json.dumps(schema or {}):
            return json.dumps({"question": "كيف أتوضأ؟", "translate": "", "answer_lang": "", "feeling": "", "personal": False,
                               "kind": "how", "reexplain": False, "recall": "", "queries": ["صفة الوضوء"],
                               "answer_request": {"form": "explanation", "requirements": [], "domain": "fiqh"}},
                              ensure_ascii=False)
        return "{}"
    engine = Muhawir(CORPUS, ModelGenerator([("m", call)]))
    events, seen = [], []
    monkeypatch.setattr(engine, "_gather", lambda *a, **k: (seen.append(k.get("domain")), ([], []))[1])
    token = pipeline._progress.set(lambda stage, data=None: events.append((stage, data)))
    try:
        engine.ask("كيف اتوضأ")
    finally:
        pipeline._progress.reset(token)
    assert seen == ["fiqh"]
    # the same question in plainer spelling is not a rewrite: the user's own words are shown
    assert ("looking", {"domain": "fiqh", "query": "كيف اتوضأ"}) in events
