"""Urdu as a third language (22 October 2026): told apart from Arabic by its own letters, its distress phrases and
greetings caught by code, its verses quoted from the approved Urdu translation, and its voices offered by name.
Synthetic data only; no model is called."""
import json

from muhawir import attribution, classify, translation, tts
from muhawir.corpus import parse_corpus
from muhawir.messages import LANGS, TEXT
from muhawir.pipeline import script_lang
from muhawir.store import SqliteCorpus, build_db


def test_urdu_is_told_apart_from_arabic_and_english():
    assert script_lang("اسلام کے ارکان کیا ہیں؟", "ar") == "ur"
    assert script_lang("ما هي أركان الإسلام؟", "en") == "ar"
    assert script_lang("What are the pillars of Islam?", "ar") == "en"
    assert script_lang("123", "ur") == "ur"  # no letters: the page language


def test_every_fixed_message_has_its_urdu_text():
    assert "ur" in LANGS and set(TEXT["ur"]) == set(TEXT["ar"])
    assert set(TEXT["ur"]["no_reason"]) == set(TEXT["ar"]["no_reason"])


def test_urdu_distress_is_caught_by_code_and_a_question_about_suicide_is_not():
    for said in ("میں خودکشی کرنا چاہتا ہوں", "میں مرنا چاہتی ہوں", "اب جینا نہیں چاہتا", "دل کرتا ہے خود کو مار دوں"):
        assert classify.check(said).kind == classify.CRISIS, said
    assert classify.check("خودکشی کا اسلام میں کیا حکم ہے؟").kind is None
    assert classify.mentions_suicide("خودکشی کا اسلام میں کیا حکم ہے؟")


def test_urdu_greetings_and_thanks_get_the_short_reply():
    assert classify.is_small_talk("السلام علیکم") and classify.is_thanks("بہت شکریہ")


def test_an_urdu_ruling_is_credited_in_urdu():
    assert attribution.gives_ruling("یہ حرام ہے۔", "ur") and attribution.says_who("علما کے مطابق یہ حرام ہے۔", "ur")


def _urdu_file():
    verses = [{"chapter": 1, "verse": i, "text": f"آیت ﻻ {i}"} for i in range(1, translation.VERSES + 1)]
    return json.dumps({"quran": verses}, ensure_ascii=False)


def test_the_urdu_translation_is_kept_beside_each_verse_in_ordinary_letters(tmp_path):
    corpus = {"synthetic": True, "sources": [{"id": "q", "name": "م", "about": "م"}],
              "passages": [{"id": "q:1:1", "source_id": "q", "location": "الفاتحة 1", "kind": "quran", "text": "بسم الله"}]}
    translation.add_urdu_to_corpus(corpus, _urdu_file())
    build_db(corpus, tmp_path / "m.db")
    for c in (parse_corpus(corpus), SqliteCorpus(tmp_path / "m.db")):
        assert c.translation("q:1:1", "ur") == "آیت لا 1"  # «ﻻ» written as the ordinary letters
        assert c.translation("q:1:1", "en") == "" and "جوناگڑھی" in c.translation_name_for("ur")


def test_each_language_offers_named_voices_women_first():
    voices = tts.public_list()
    assert set(voices) == {"ar", "en", "ur"}
    for lang, offered in voices.items():
        assert offered[0]["gender"] == "female" and any(v["gender"] == "male" for v in offered)
    assert ("ur", "uzma") in tts.VOICES and tts.VOICES[("ar", "female")][1] == "ar-SA-ZariyahNeural"
    assert "(آیت)" in tts.replace_verses("﴿بسم الله﴾", "ur")
