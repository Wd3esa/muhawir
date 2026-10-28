"""The waiting status names selected sources, never discarded search hits."""
import pytest

from muhawir import pipeline
from muhawir.corpus import Corpus, Passage, Source
from muhawir.generate import ExtractiveGenerator
from muhawir.retrieve import Hit


@pytest.mark.parametrize("threads", [1, 4])
def test_source_progress_uses_filtered_context(monkeypatch, threads):
    monkeypatch.setattr(pipeline, "SEARCH_THREADS", threads)
    monkeypatch.setattr(pipeline, "MODEL_CANDIDATES", 3)
    text = "نص تجريبي طويل للاختبار فقط ولا يمثل مصدرا دينيا ولا يعتمد عليه في أي جواب"
    passages = [
        Passage("a", "chosen", "موضع أول", text),
        Passage("b", "confused", "باب القراض", text),
        Passage("c", "chosen", "موضع ثان", text),
        Passage("d", "overflow", "موضع آخر", text),
    ]
    corpus = Corpus({key: Source(key, name, "اختبار") for key, name in [
        ("chosen", "المصدر المختار"), ("confused", "المصدر المستبعد"), ("overflow", "خارج الحد")
    ]}, passages, synthetic=True)

    class Search:
        def search(self, query, k):
            return [Hit(p, 1.0, 1.0) for p in passages]

    engine = pipeline.Muhawir(corpus, ExtractiveGenerator(), Search())
    events = []
    token = pipeline._progress.set(lambda stage, data=None: events.append((stage, data)))
    try:
        offered, _ = engine._gather("ما حكم القروض", "ما حكم القروض", ["عبارة بحث"])
    finally:
        pipeline._progress.reset(token)
    assert [p.id for p in offered] == ["a", "c"]
    assert events == [("source", ["المصدر المختار"])]


def test_empty_search_does_not_report_sources():
    class Search:
        def search(self, query, k):
            return []

    engine = pipeline.Muhawir(Corpus({}, [], synthetic=True), ExtractiveGenerator(), Search())
    events = []
    token = pipeline._progress.set(lambda stage, data=None: events.append((stage, data)))
    try:
        assert engine._gather("سؤال", "سؤال", [])[0] == []
    finally:
        pipeline._progress.reset(token)
    assert events == []
