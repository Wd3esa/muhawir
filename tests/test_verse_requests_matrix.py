"""Every way people ask for a verse (November 2026). Two layers, so nothing depends on guessing phrasings:

1. The understanding step (the model, already called for every question) reads any phrasing and returns pointers:
   form (verse_text…), quoted_text (the user's own words of the verse, as typed) and surah/ayah when named by place.
   Whether the real model returns them is MEASURED on the server by scripts/eval_understanding.py over PHRASINGS.
2. Code (this file, free): given those pointers, the verse is read from the mushaf, on the in-memory search and on the
   SQLite store, against an Uthmani mushaf crowded by commentaries; a text the user did not type is never shown.

A live failure is added to PHRASINGS first (with what a correct understanding returns), then fixed: the list only grows."""
import json

import pytest

from muhawir.corpus import parse_corpus
from muhawir.generate import ModelGenerator
from muhawir.pipeline import ANSWERED, Muhawir
from muhawir.store import SqliteCorpus, SqliteRetriever, build_db

# Uthmani spelling, as the mushaf may be stored («ٱلصَّٰبِرُونَ» for «الصابرون»): typed words must still find it
ZUMAR_10 = ("قُلۡ يَٰعِبَادِ ٱلَّذِينَ ءَامَنُواْ ٱتَّقُواْ رَبَّكُمۡۚ لِلَّذِينَ أَحۡسَنُواْ فِي هَٰذِهِ ٱلدُّنۡيَا حَسَنَةٞۗ وَأَرۡضُ ٱللَّهِ وَٰسِعَةٌۗ "
            "إِنَّمَا يُوَفَّى ٱلصَّٰبِرُونَ أَجۡرَهُم بِغَيۡرِ حِسَابٖ")
BAQARA_5 = "أُولَٰئِكَ عَلَىٰ هُدًى مِّن رَّبِّهِمْ ۖ وَأُولَٰئِكَ هُمُ الْمُفْلِحُونَ"
BAQARA_153 = "يَٰٓأَيُّهَا ٱلَّذِينَ ءَامَنُواْ ٱسۡتَعِينُواْ بِٱلصَّبۡرِ وَٱلصَّلَوٰةِۚ إِنَّ ٱللَّهَ مَعَ ٱلصَّٰبِرِينَ"
DATA = {"synthetic": True, "sources": [
    {"id": "quranpedia-hafs", "name": "القرآن الكريم", "about": "المصحف"},
    {"id": "tafsir-x", "name": "تفسير", "about": "تفسير"}, {"id": "fiqh-x", "name": "فقه", "about": "فقه"}],
    "passages": [
    {"id": "q:2:1", "source_id": "quranpedia-hafs", "location": "سورة البقرة، الآية 1", "kind": "quran", "text": "الم"},
    {"id": "q:2:5", "source_id": "quranpedia-hafs", "location": "سورة البقرة، الآية 5", "kind": "quran", "text": BAQARA_5},
    {"id": "q:2:153", "source_id": "quranpedia-hafs", "location": "سورة البقرة، الآية 153", "kind": "quran", "text": BAQARA_153},
    {"id": "q:39:1", "source_id": "quranpedia-hafs", "location": "سورة الزمر، الآية 1", "kind": "quran",
     "text": "تَنزِيلُ الْكِتَابِ مِنَ اللَّهِ الْعَزِيزِ الْحَكِيمِ"},
    {"id": "q:39:10", "source_id": "quranpedia-hafs", "location": "سورة الزمر، الآية 10", "kind": "quran", "text": ZUMAR_10},
    # commentaries and fiqh passages that quote the same words many times, to outrank the verse in a plain search
    *[{"id": f"t:{i}", "source_id": "tafsir-x", "location": f"سورة الزمر، الآية 10 ({i})", "kind": "tafsir",
       "text": "قوله تعالى إنما يوفى الصابرون أجرهم بغير حساب أي إنما يوفى الصابرون أجرهم بغير حساب يوم القيامة "
               "والصابرون هم الذين صبروا فيوفى الصابرون أجرهم"} for i in range(40)],
    *[{"id": f"f:{i}", "source_id": "fiqh-x", "location": f"باب الصبر {i}", "kind": "fiqh",
       "text": "استعينوا بالصبر والصلاة إن الله مع الصابرين، وأولئك على هدى من ربهم وأولئك هم المفلحون، والصبر واجب"}
      for i in range(40)]]}

from verse_phrasings import NOT_REQUESTS, PHRASINGS, V  # noqa: E402

def _understanding(form, quoted, surah, ayah, question="سؤال"):
    def call(system, user, schema=None):
        if '"question"' in json.dumps(schema or {}):
            return json.dumps({"question": question, "translate": "", "answer_lang": "", "feeling": "", "personal": False,
                               "kind": "", "reexplain": False, "recall": "", "queries": ["الصبر"],
                               "answer_request": {"form": form, "requirements": [], "domain": "quran",
                                                  "quoted_text": quoted, "surah": surah, "ayah": ayah}},
                              ensure_ascii=False)
        return json.dumps({"plan": "", "abstain": True, "as_list": False, "follow_up": "", "claims": [], "views": []})
    return call


@pytest.fixture(scope="module", params=["memory", "sqlite"])
def store(request, tmp_path_factory):
    if request.param == "memory":
        corpus = parse_corpus(DATA)
        return corpus, None
    path = tmp_path_factory.mktemp("matrix") / "m.db"
    build_db(DATA, path)
    corpus = SqliteCorpus(path)
    return corpus, SqliteRetriever(corpus)


def _ask(store, question, pointers):
    corpus, retriever = store
    engine = Muhawir(corpus, ModelGenerator([("m", _understanding(*pointers, question=question))]), retriever)
    engine._asked_verse = lambda q: None  # the word shortcut is off: the general path alone must find it
    return engine.ask(question)


@pytest.mark.parametrize("question,pointers,expected", PHRASINGS)
def test_every_way_of_asking_for_a_verse_shows_it_from_the_understanding_pointers(store, question, pointers, expected):
    res = _ask(store, question, pointers)
    assert res.status == ANSWERED and res.claims and res.claims[0]["passage_ids"] == [expected]


@pytest.mark.parametrize("question,pointers", NOT_REQUESTS)
def test_a_question_about_meaning_is_not_answered_with_the_bare_verse(store, question, pointers):
    res = _ask(store, question, pointers)
    assert not (res.claims and res.claims[0]["text"].startswith("﴿") and len(res.claims) == 1 and res.status == ANSWERED)


def test_a_text_the_user_did_not_type_is_never_shown(store):
    # the model «helpfully» writes a verse the user only described: not in the user's words, so not used
    res = _ask(store, "اذكر الآية التي فيها الصبر على البلاء", (V, "إنما يوفى الصابرون أجرهم بغير حساب", "", 0))
    assert not (res.claims and res.claims[0]["passage_ids"] == ["q:39:10"])


def test_the_live_failure_shows_the_verse_and_its_surah(store):
    res = _ask(store, "انما يوفى الصابرون اجرهم اكمل الآية واذكر السورة", PHRASINGS[0][1])
    assert res.claims[0]["passage_ids"] == ["q:39:10"] and "سورة الزمر" in res.claims[0]["text"]
