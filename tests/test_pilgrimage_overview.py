"""A question about steps or a list (how to perform a rite) is answered completely, by a general rule and not by word
lists (issue #100): the understanding step names a part for each rite, the search covers each part, the writer is told to
give every stage the sources support, and the length ceiling does not cut a procedure short."""
import muhawir.pipeline as pipeline
from muhawir import generate
from muhawir.generate import build_user_prompt
from muhawir.pipeline import fit_style


def test_no_word_lists_for_any_rite_remain_in_the_code():
    for name in ("_HAJJ_SEARCH", "_UMRAH_SEARCH", "_pilgrimage_overview", "_pilgrimage_queries"):
        assert not hasattr(pipeline, name)


def test_the_understanding_step_is_told_to_make_a_part_for_each_rite_and_a_search_phrase_for_each_stage():
    assert "مطلبًا لكل عبادة" in generate.UNDERSTAND_PROMPT and "«خطوات …» لا «تعريف …»" in generate.UNDERSTAND_PROMPT
    assert "لكل مرحلة من مراحلها عبارة بحث" in generate.QUERY_RULES


def test_writer_is_told_to_complete_ordered_steps_from_sources_for_any_how_question():
    for question in ("صفة الحج والعمرة", "كيف أتوضأ؟", "ما شروط الصلاة؟"):
        prompt = build_user_prompt(question, [], "youth", "ar", False, kind="how",
                                   answer_request={"form": "explanation", "requirements": ["خطوات"]})
        assert "لا تعريفًا" in prompt and "لا تحذف مرحلة لازمة" in prompt and "لا تخلط أعمال نوع بأعمال آخر" in prompt
        assert "لا تستكمل خطوة لا مصدر لها" in prompt
    other = build_user_prompt("ما معنى الصمد؟", [], "youth", "ar", False, kind="what")
    assert "لا تحذف مرحلة لازمة" not in other


def test_a_procedure_or_a_multi_part_answer_is_not_cut_to_the_prose_length():
    steps = list(range(30))
    assert len(fit_style(steps, "youth", False)) == pipeline.STYLE_CEILING["youth"][0]
    assert len(fit_style(steps, "youth", True)) == pipeline.STYLE_CEILING["youth"][1]
    assert fit_style(steps, "youth", False, parts=2) == steps[:pipeline.STYLE_CEILING["youth"][1]]
    assert len(fit_style(list(range(16)), "youth", True)) == 16  # a full rite of sixteen stages loses nothing
    assert len(fit_style(list(range(12)), "youth", False)) == 12  # nor does a prose answer of twelve sentences
