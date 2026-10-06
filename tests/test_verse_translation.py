"""A verse is never translated by the model (Codex audit, 6 October 2026). Synthetic data and fake model calls only:
the «verse» below is made-up test text, not from the Quran."""
import json

from muhawir.corpus import Corpus, Passage, Source
from muhawir.generate import ModelGenerator
from muhawir.pipeline import ABSTAINED, TRANSLATED, Muhawir

VERSE = "يَا أَيُّهَا النَّخْلُ اشْرَبْ مَاءً كَثِيرًا فِي الصَّيْفِ"
APPROVED = "O palm, drink much water in summer (approved test translation)."
WRONG = "A wrong translation made up by the model."


def corpus() -> Corpus:
    return Corpus({"test-q": Source("test-q", "مصدر تجريبي للآيات", "مصطنع للاختبار فقط", "")},
                  [Passage("test-q:1", "test-q", "سورة تجريبية، الآية 1", VERSE, "quran", "", ""),
                   Passage("test-q:2", "test-q", "فقرة تجريبية", "الماء يغلي عند مئة درجة.", "other", "", "")],
                  True, {"test-q:1": APPROVED})


def muhawir(translate: str):
    seen = {"model_translated": 0}

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if '"question"' in keys:
            return json.dumps({"question": "" if translate else "ما معنى الآية؟", "translate": translate, "answer_lang": "en", "queries": []},
                              ensure_ascii=False)
        if '"translation"' in keys:
            seen["model_translated"] += 1
            return json.dumps({"translation": WRONG})
        return json.dumps({"abstain": True, "claims": []})
    return Muhawir(corpus(), ModelGenerator([("m", call)])), seen


def test_a_verse_gets_the_approved_translation_with_its_source_never_the_models():
    m, seen = muhawir(VERSE)
    res = m.ask(f"ترجم هذه الآية: {VERSE}")
    assert res.status == TRANSLATED and res.message == APPROVED and WRONG not in res.message
    assert [s["passage_id"] for s in res.sources] == ["test-q:1"] and "سورة تجريبية" in res.note
    assert seen["model_translated"] == 0


def test_a_long_unvocalised_quote_of_a_verse_is_found_without_being_called_a_verse():
    m, seen = muhawir("يا ايها النخل اشرب ماء كثيرا")
    res = m.ask("ترجم: يا ايها النخل اشرب ماء كثيرا")
    assert res.status == TRANSLATED and res.message == APPROVED and seen["model_translated"] == 0


def test_text_called_a_verse_but_not_in_the_sources_is_not_translated():
    m, seen = muhawir("قل يا نخل نم طويلا")
    res = m.ask("ترجم الآية: قل يا نخل نم طويلا")
    assert res.status == ABSTAINED and not res.sources and seen["model_translated"] == 0


def test_a_short_common_phrase_that_happens_to_occur_in_a_verse_is_translated_as_language():
    m, seen = muhawir("ماء كثيرا")
    res = m.ask("ترجم ماء كثيرا إلى الإنجليزية")
    assert res.status == TRANSLATED and res.message == WRONG and seen["model_translated"] == 1


def test_a_verse_quoted_with_one_word_wrong_gets_a_note_with_the_verse_as_it_is():
    m, _ = muhawir("")
    res = m.ask("ما معنى قوله تعالى: «يا ايها النخل اشرب ماء كثيرون»؟")
    assert "اشْرَبْ مَاءً كَثِيرًا" in res.note and "كثيرون" in res.note and "سورة تجريبية" in res.note


def test_a_verse_quoted_correctly_gets_no_note():
    m, _ = muhawir("")
    res = m.ask("ما معنى قوله تعالى: «يا ايها النخل اشرب ماء كثيرا»؟")
    assert "تنبيه" not in res.note
