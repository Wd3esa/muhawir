"""Every way people ask for a verse (November 2026): the fragment before or after the request, with or without marks,
with or without «الآية», in dialect, misspelt hamzas, asking for the surah, by its place. Each phrasing must show the
verse from the mushaf at once, with no model, on the in-memory search and on the SQLite store the server uses, while
commentaries quoting the same words try to crowd it out. Questions about the meaning must NOT be taken for a request.

When a live question fails, its phrasing is added to PHRASINGS first, then the code is fixed: the list only grows."""
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

# (what the user typed, the verse that must be shown)
PHRASINGS = [
    # the fragment before the request (the live failure of 4 November 2026)
    ("انما يوفى الصابرون اجرهم اكمل الآية واذكر السورة", "q:39:10"),
    ("إنما يوفى الصابرون أجرهم أكمل الآية", "q:39:10"),
    ("انما يوفى الصابرون اجرهم.. كمل الاية", "q:39:10"),
    ("انما يوفى الصابرون اجرهم في اي سورة", "q:39:10"),
    ("انما يوفى الصابرون اجرهم وين توجد", "q:39:10"),
    ("«إنما يوفى الصابرون أجرهم» أكمل", "q:39:10"),
    # the fragment after the request
    ("اكمل الآية: انما يوفى الصابرون اجرهم", "q:39:10"),
    ("أكمل الاية انما يوفى الصابرون", "q:39:10"),
    ("اكمل: انما يوفى الصابرون اجرهم", "q:39:10"),
    ("كمل الايه انما يوفى الصابرون اجرهم واذكر اسم السورة", "q:39:10"),
    ("ما تكملة آية انما يوفى الصابرون اجرهم", "q:39:10"),
    ("ما بقية الآية إنما يوفى الصابرون أجرهم", "q:39:10"),
    ("في اي سورة قوله تعالى انما يوفى الصابرون اجرهم", "q:39:10"),
    ("في أي سورة: إنما يوفى الصابرون أجرهم بغير حساب", "q:39:10"),
    ("ما اسم السورة التي فيها انما يوفى الصابرون اجرهم", "q:39:10"),
    ("اذكر الآية «استعينوا بالصبر والصلاة»", "q:2:153"),
    ("لو سمحت اكتب لي آية استعينوا بالصبر والصلاة ان الله مع الصابرين", "q:2:153"),
    ("ابي الاية اللي فيها استعينوا بالصبر والصلاة اكملها", "q:2:153"),
    # plain spelling against the Uthmani mushaf
    ("اكمل الآية يا ايها الذين امنوا استعينوا بالصبر", "q:2:153"),
    ("اذكر آية استعينوا بالصبر والصلاة إن الله مع الصابرين", "q:2:153"),
    # by its place
    ("الآية الخامسة من سورة البقرة", "q:2:5"),
    ("نص الآية الآية الخامسة من سورة البقرة", "q:2:5"),
    ("آية ٥ من البقرة", "q:2:5"),
    ("البقرة 5", "q:2:5"),
    ("سورة الزمر (١٠)", "q:39:10"),
    ("الزمر: 10", "q:39:10"),
]

# questions about a verse's meaning, its ruling, or ordinary sentences: NOT a request to show a verse
NOT_REQUESTS = [
    "ما معنى انما يوفى الصابرون اجرهم بغير حساب",
    "ما تفسير الآية 5 من البقرة",
    "ما حكم الصبر على البلاء",
    "في اي سورة ذكرت قصة يوسف",
    "اكمل الشرح من فضلك",
    "صلاة الفجر 2 ركعة",
]


def _call(system, user, schema=None):
    if '"question"' in json.dumps(schema or {}):
        return json.dumps({"question": "", "translate": "", "answer_lang": "", "feeling": "", "personal": False, "kind": "",
                           "reexplain": False, "recall": "", "queries": [],
                           "answer_request": {"form": "explanation", "requirements": [], "domain": ""}})
    return "{}"


@pytest.fixture(scope="module", params=["memory", "sqlite"])
def engine(request, tmp_path_factory):
    if request.param == "memory":
        return Muhawir(parse_corpus(DATA), ModelGenerator([("m", _call)]))
    path = tmp_path_factory.mktemp("matrix") / "m.db"
    build_db(DATA, path)
    corpus = SqliteCorpus(path)
    return Muhawir(corpus, ModelGenerator([("m", _call)]), SqliteRetriever(corpus))


def _shown(engine, question):
    verse = engine._asked_verse(question) or engine._verse_by_reference(question)
    return verse.id if verse is not None else None


@pytest.mark.parametrize("question,expected", PHRASINGS)
def test_every_way_of_asking_for_a_verse_shows_it(engine, question, expected):
    assert _shown(engine, question) == expected


@pytest.mark.parametrize("question", NOT_REQUESTS)
def test_a_question_about_meaning_or_an_ordinary_sentence_is_not_a_verse_request(engine, question):
    assert _shown(engine, question) is None


def test_the_live_failure_is_answered_with_the_verse_and_its_surah(engine):
    res = engine.ask("انما يوفى الصابرون اجرهم اكمل الآية واذكر السورة")
    assert res.status == ANSWERED and res.claims[0]["passage_ids"] == ["q:39:10"]
    assert "سورة الزمر" in res.claims[0]["text"]
