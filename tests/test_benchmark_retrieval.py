"""Fair timing excludes cold query encoding and preserves evidence for manual review."""
from types import SimpleNamespace

import numpy as np
import pytest

from muhawir.corpus import Corpus, Passage, Source
from muhawir.retrieve import Hit
from muhawir.vectors import HybridRetriever, VectorIndex
from scripts.benchmark_retrieval import paired_search, prepare_queries, review_report


def test_embeddings_are_computed_once_per_distinct_question_outside_search():
    called = []
    def embed(q):
        called.append(q)
        return [len(q)]
    ticks = iter([0, 2, 2, 5])
    vectors, times = prepare_queries(embed, [{"q": "one"}, {"q": "two"}, {"q": "one"}],
                                      clock=lambda: next(ticks))
    assert called == ["one", "two"]
    assert times == [2, 3] and vectors == {"one": [3], "two": [3]}


def test_every_mode_is_warmed_before_timing_and_search_order_alternates():
    calls, warmed = [], set()
    def retriever(name):
        def search(q, depth):
            calls.append((name, q))
            warmed.add((name, q))
            return []
        return SimpleNamespace(search=search)
    modes = {name: retriever(name) for name in ("a", "b")}
    questions = [{"q": q, "gold": []} for q in ("one", "two")]
    def clock():
        assert len(warmed) == 4
        return len(calls) * .1
    result = paired_search(questions, modes, 10, 2, clock)
    # The four warmups precede all recorded timing; paired tests run in both orders.
    assert len(calls) == 12
    assert calls[4:6] == [("a", "one"), ("b", "one")]
    assert calls[6:8] == [("b", "two"), ("a", "two")]
    assert all(len(data["seconds"]) == 4 for data in result.values())
    assert all(len(ranks) == 2 for data in result.values() for ranks in data["ranks"])
    assert all(abs(t - .1) < 1e-8 for data in result.values() for t in data["seconds"])


def test_report_keeps_missing_gold_and_full_texts_including_different_candidates():
    a = Passage("a", "book", "p. 1", "نص مصطنع أول لاختبار التقرير")
    b = Passage("b", "book", "p. 2", "نص مصطنع آخر لاختبار التقرير")
    corpus = Corpus({"book": Source("book", "كتاب الاختبار", "مصطنع")}, [a, b], synthetic=True)
    questions = [{"q": "سؤال", "gold": ["a", "missing"]}]
    modes = {"exact": SimpleNamespace(search=lambda *args: [Hit(a, 1, 1)]),
             "ANN": SimpleNamespace(search=lambda *args: [Hit(b, 1, 1)])}
    result = paired_search(questions, modes, 10, 2)
    row = review_report(corpus, questions, result)["results"][0]
    assert row["missing_gold"] == ["missing"]
    assert row["present_gold"][0]["text"] == a.text
    assert row["modes"]["exact"]["gold_ranks"] == [1, 1]
    assert row["modes"]["ANN"]["gold_ranks"] == [None, None]
    assert row["modes"]["ANN"]["hits"][0] == {
        "id": "b", "source": "كتاب الاختبار", "location": "p. 2", "text": b.text}


def test_ann_fallback_must_not_be_reported_as_ann_speed():
    index = VectorIndex(["a"], np.array([[1., 0.]], dtype=np.float16), "local:fake@1",
                        lambda _: [1., 0.], ann=SimpleNamespace(search=lambda *args, **kw: []))
    corpus = Corpus({"book": Source("book", "مصطنع", "اختبار")},
                    [Passage("a", "book", "p.1", "نص مصطنع للاختبار")], synthetic=True)
    retriever = HybridRetriever(SimpleNamespace(search=lambda *args: [], corpus=corpus), index)
    with pytest.raises(RuntimeError, match="ANN failed"):
        paired_search([{"q": "test", "gold": []}], {"USearch": retriever}, 10, 1)


@pytest.mark.parametrize("questions,depth,repeats", [([], 10, 1), ([{"q": "q"}], 9, 1),
                                                    ([{"q": "q"}], 10, 0)])
def test_invalid_benchmark_is_rejected(questions, depth, repeats):
    with pytest.raises(ValueError):
        paired_search(questions, {"exact": SimpleNamespace()}, depth, repeats)
