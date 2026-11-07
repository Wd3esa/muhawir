"""General pilgrimage questions are procedures, even when the understanding model calls them definitions."""
from muhawir.corpus import parse_corpus
from muhawir.generate import ModelGenerator, build_user_prompt
from muhawir.pipeline import Muhawir, Response, ABSTAINED, _pilgrimage_overview, fit_style


def test_short_procedure_request_covers_both_rites_without_catching_individual_rulings():
    assert _pilgrimage_overview("صفة الحج والعمرة") == ("umrah", "hajj")
    assert _pilgrimage_overview("خطوات العمرة") == ("umrah",)
    assert _pilgrimage_overview("مناسك الحج") == ("hajj",)
    assert not _pilgrimage_overview("ما حكم طواف الوداع في الحج؟")
    assert not _pilgrimage_overview("ما معنى العمرة؟")


def test_rites_survive_a_definition_rewrite_and_search_each_stage(monkeypatch):
    corpus = parse_corpus({"synthetic": True, "sources": [{"id": "test", "name": "تجريبي", "about": "تجريبي"}],
                           "passages": [{"id": "p1", "source_id": "test", "location": "أ", "kind": "fiqh",
                                         "text": "نص تجريبي لا يستخدم في الجواب."}]})
    generator = ModelGenerator([("fake", lambda *args, **kwargs: "{}")])
    generator.understand = lambda *args: {
        "question": "ما هو الحج؟", "kind": "what", "queries": ["تعريف الحج"], "personal": False,
        "answer_request": {"form": "explanation", "requirements": ["تعريف الحج"], "domain": "fiqh"}}
    engine = Muhawir(corpus, generator)
    seen = {}

    def capture(question, *args, **kwargs):
        seen.update(question=question, **kwargs)
        return Response(ABSTAINED, "مصادر غير كافية", synthetic=True)

    monkeypatch.setattr(engine, "_ask", capture)
    engine.ask("صفة الحج والعمرة")  # no network or /api/ask; _ask is replaced before generation
    assert seen["question"] == "صفة الحج والعمرة" and seen["kind"] == "how"
    assert seen["answer_request"]["domain"] == "fiqh"
    assert seen["answer_request"]["rituals"] == ["umrah", "hajj"]
    assert len(seen["answer_request"]["requirements"]) == 2
    assert "تعريف الحج" not in seen["answer_request"]["requirements"]
    assert any("طواف العمرة" in q for q in seen["queries"])
    assert any("عرفة" in q for q in seen["queries"])
    assert "تعريف الحج" in seen["queries"]


def test_writer_is_told_to_complete_ordered_steps_from_sources():
    prompt = build_user_prompt("صفة الحج والعمرة", [], "youth", "ar", False, kind="how",
                               answer_request={"form": "explanation", "rituals": ["umrah", "hajj"],
                                               "requirements": ["خطوات العمرة", "خطوات الحج"]})
    assert "لا تعريفه" in prompt and "فلا تخلط أعمال الأنساك المختلفة" in prompt
    assert "حدّ الجمل التقريبي" in prompt and "لا تستكمل خطوة لا مصدر لها" in prompt


def test_complete_ritual_steps_are_not_cut_to_general_style_limit():
    steps = list(range(16))
    assert len(fit_style(steps, "youth", True)) == 10
    assert fit_style(steps, "youth", True, ["hajj"]) == steps[:12]
    assert fit_style(steps, "youth", True, ["umrah", "hajj"]) == steps
