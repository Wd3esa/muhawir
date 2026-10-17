"""A verse quoted with one word in the wrong form gets the note «نص الآية في المصحف ...», however the question is
wrapped: the request may sit inside the same quotation marks (a committee-style test, 17 October 2026). Synthetic
data and fake model calls only: the «verse» below is made-up test text, not from the Quran."""
import json

from muhawir.corpus import Corpus, Passage, Source
from muhawir.generate import ModelGenerator
from muhawir.pipeline import Muhawir

VERSE = "يَا أَيُّهَا النَّاسُ اشْرَبُوا مَاءً كَثِيرًا إِنَّ النَّخْلَ مَعَ الصَّائِمِينَ"
WRONG_END = "اشربوا ماء كثيرا إن النخل مع الصائمون"


def muhawir() -> Muhawir:
    corpus = Corpus({"test-q": Source("test-q", "مصدر تجريبي للآيات", "مصطنع للاختبار فقط", "")},
                    [Passage("test-q:1", "test-q", "سورة تجريبية، الآية 1", VERSE, "quran", "", ""),
                     Passage("test-q:2", "test-q", "فقرة تجريبية", "الماء يغلي عند مئة درجة.", "other", "", "")],
                    True)

    def call(system, user, schema=None):
        if '"question"' in json.dumps(schema or {}):
            return json.dumps({"question": "ما معنى الآية؟", "translate": "", "answer_lang": "ar", "queries": []},
                              ensure_ascii=False)
        return json.dumps({"abstain": True, "claims": []})
    return Muhawir(corpus, ModelGenerator([("m", call)]))


def test_the_verse_alone_in_quotation_marks_is_found():
    found = muhawir()._misquoted(f"اذكر الآية: «{WRONG_END}»")
    assert found and found[0].id == "test-q:1"


def test_the_request_inside_the_same_quotation_marks_does_not_hide_the_verse():
    m = muhawir()
    for question in (f"«اذكر الآية: {WRONG_END}»", f"«اكتب لي الآية {WRONG_END}»", f"«{WRONG_END} هل هذا صحيح»"):
        found = m._misquoted(question)
        assert found and found[0].id == "test-q:1", question


def test_the_note_shows_the_verse_as_it_is():
    res = muhawir().ask(f"«اذكر الآية: {WRONG_END}»")
    assert "سورة تجريبية" in res.note and VERSE in res.note


def test_a_verse_quoted_correctly_gets_no_note():
    assert muhawir()._misquoted("«اذكر الآية: اشربوا ماء كثيرا إن النخل مع الصائمين»") is None


def test_a_verse_asked_for_is_shown_whole_with_no_model_and_an_offer_to_explain():
    calls = []
    m = muhawir()
    m.generator = ModelGenerator([("m", lambda *a, **k: calls.append(1) or "{}")])
    res = m.ask(f"«اذكر الآية: {WRONG_END}»")
    assert res.status == "answered" and not calls
    assert res.claims[0]["text"].startswith(f"﴿{VERSE}﴾") and res.claims[0]["passage_ids"] == ["test-q:1"]
    assert "تنبيه" in res.note


def test_a_verse_quoted_without_its_first_and_is_still_found():
    res = muhawir().ask("اذكر الآية: ايها الناس اشربوا ماء كثيرا")
    assert res.status == "answered" and res.claims[0]["passage_ids"] == ["test-q:1"]
