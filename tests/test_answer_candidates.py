"""Question-only source parts must never be offered as evidence for an answer."""
import numpy as np

from muhawir.corpus import Passage
from muhawir.generate import ExtractiveGenerator
from muhawir.pipeline import Muhawir
from muhawir.store import SqliteCorpus, SqliteRetriever, build_db
from muhawir.vectors import HybridRetriever, VectorIndex
from muhawir.verify import Claim, verify


def _corpus(tmp_path):
    data = {
        "synthetic": True,
        "sources": [{"id": "bayyinat", "name": "بينات", "about": "أسئلة وأجوبة"}],
        "passages": [
            {"id": "by:question", "source_id": "bayyinat", "location": "المسألة (1) — السؤال (ص1)",
             "kind": "qa", "text": "السؤال: لماذا فرض الله الصيام؟ وما الحكمة منه؟"},
            {"id": "by:gist", "source_id": "bayyinat", "location": "المسألة (1) — مضمون السؤال (ص1)",
             "kind": "qa", "text": "مضمون السؤال: يسأل عن الحكمة من فرض الصيام."},
            {"id": "by:answer", "source_id": "bayyinat", "location": "المسألة (1) — مختصر الإجابة (ص2)",
             "kind": "qa", "text": "فرض الله الصيام ليتعلم المؤمن التقوى والصبر."},
        ],
    }
    path = tmp_path / "bayyinat.db"
    build_db(data, path)
    return SqliteCorpus(path)


def test_search_and_answer_context_exclude_questions_but_keep_answers(tmp_path):
    corpus = _corpus(tmp_path)
    search = SqliteRetriever(corpus)
    hits = search.search("لماذا فرض الله الصيام؟", k=3)
    assert [h.passage.id for h in hits] == ["by:answer"]
    # The original questions remain available in source browsing; only answer context changes.
    assert corpus.passage("by:question").text.startswith("السؤال:")
    context, _ = Muhawir(corpus, ExtractiveGenerator(), search)._gather(
        "لماذا فرض الله الصيام؟", "لماذا فرض الله الصيام؟", []
    )
    assert [p.id for p in context] == ["by:answer"]


def test_vector_fusion_does_not_reintroduce_question_only_parts(tmp_path):
    corpus = _corpus(tmp_path)
    # The vector rank favors the question, yet it may only return answer material.
    vectors = VectorIndex(
        ["by:question", "by:gist", "by:answer"],
        np.array([[1, 0, 0], [0.9, 0.1, 0], [0.8, 0.2, 0]], dtype=np.float16),
        "local:example/model@revision", lambda _: [1, 0, 0],
    )
    hybrid = HybridRetriever(SqliteRetriever(corpus), vectors)
    assert [h.passage.id for h in hybrid.search("لماذا فرض الله الصيام؟", k=3)] == ["by:answer"]
    hybrid._disabled = True
    assert [h.passage.id for h in hybrid.search("لماذا فرض الله الصيام؟", k=3)] == ["by:answer"]


def test_verifier_rejects_question_as_citation_even_if_present_in_allowed_set(tmp_path):
    corpus = _corpus(tmp_path)
    claims = [Claim("لماذا فرض الله الصيام؟", ("by:question",)),
              Claim("فرض الله الصيام ليتعلم المؤمن التقوى والصبر.", ("by:answer",))]
    kept, rejected = verify(claims, corpus, {"by:question", "by:answer"})
    assert kept == [claims[1]]
    assert [item.claim for item in rejected] == [claims[0]]


def test_only_bayyinat_question_parts_are_affected():
    from muhawir.bayyinat import is_answer_part

    other = Passage("x", "some-other-book", "المسألة — السؤال (ص1)", "السؤال نص في هذا المصدر")
    assert is_answer_part(other)
