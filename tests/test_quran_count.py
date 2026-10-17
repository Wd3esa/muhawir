"""Counting a word in the Quran is done by code over every verse, never by a model (17 October 2026: a platform
answered «الرشد» twice; it is three times). Synthetic verses only: the text below is made up, not from the Quran."""
import json

from muhawir.corpus import Corpus, Passage, Source
from muhawir.generate import ModelGenerator
from muhawir.pipeline import ABSTAINED, ANSWERED, Muhawir
from muhawir.quran_count import QuranIndex, asked, reply

VERSES = [
    ("q:1:1", "سورة التجربة، الآية 1", "قَدْ تَبَيَّنَ النُّورُ مِنَ الظُّلْمَةِ ۖ وَالنُّورُ قَرِيبٌ"),
    ("q:1:2", "سورة التجربة، الآية 2", "يَهْدِي إِلَى النُّورِ فَآمَنَّا بِهِ"),
    ("q:2:1", "سورة المثال، الآية 1", "وَجَعَلْنَا لَهُ نُورَهُ وَالنَّارَ"),
    ("q:2:2", "سورة المثال، الآية 2", "خَلَقَ السَّمَاوَاتِ وَالْأَرْضَ بِالنُّورِ"),
]


def corpus() -> Corpus:
    return Corpus({"test-q": Source("test-q", "مصدر تجريبي للآيات", "مصطنع للاختبار فقط", "")},
                  [Passage(i, "test-q", loc, text, "quran", "", "") for i, loc, text in VERSES], True)


def test_counting_questions_are_recognised_and_others_are_not():
    assert asked("كم مره وردت كلمة النور في القرآن؟") == ("النور", "", False)
    assert asked("كم مرة ذكرت «النور» في سورة المثال") == ("النور", "المثال", False)
    assert asked("كم مرة ورد جذر نور في القرآن")[2] is True
    assert asked("كم عدد سور القرآن") is None
    assert asked("ما معنى النور في القرآن") is None


def test_every_place_is_counted_and_joined_letters_apart():
    c = QuranIndex.from_corpus(corpus()).count("النور")
    assert [p.verse_id for p in c.exact] == ["q:1:1", "q:1:2"]
    assert sorted(p.word for p in c.joined) == ["بِالنُّورِ", "وَالنُّورُ"]


def test_a_surah_limits_the_count():
    c = QuranIndex.from_corpus(corpus()).count("النور", "المثال")
    assert not c.exact and [p.verse_id for p in c.joined] == ["q:2:2"]


def test_a_long_alif_left_out_is_still_found_but_not_a_first_letter():
    index = QuranIndex.from_corpus(corpus())
    c = index.count("السموات")
    assert [p.verse_id for p in c.exact] == ["q:2:2"] and c.alif_ignored
    assert not index.count("نر").exact  # «النار» is not «نر» with alifs added at its edges


def test_a_phrase_is_counted_with_letters_joined_to_its_first_word():
    c = QuranIndex.from_corpus(corpus()).count("النور فامنا به")
    assert [p.verse_id for p in c.exact] == ["q:1:2"]


def test_the_reply_cites_each_verse_and_asks_no_model():
    calls = []

    def call(system, user, schema=None):
        calls.append(1)
        return json.dumps({"abstain": True, "claims": []})
    m = Muhawir(corpus(), ModelGenerator([("m", call)]))
    res = m.ask("كم مرة وردت كلمة النور في القرآن؟")
    assert res.status == ANSWERED and not calls
    assert "مرتين" in res.message and "آيتين" in res.message
    assert [c["passage_ids"] for c in res.claims][:2] == [["q:1:1"], ["q:1:2"]]
    assert {s["passage_id"] for s in res.sources} == {"q:1:1", "q:1:2", "q:2:2"}
    assert "لا بنموذج لغوي" in res.note
    res = m.ask("كم مرة وردت كلمة قطار في القرآن؟")
    assert res.status == ABSTAINED and not res.sources and not calls


def test_a_root_question_says_roots_are_not_counted_yet():
    lead, _, _, _ = reply(QuranIndex.from_corpus(corpus()).count("النور", root_asked=True))
    assert lead.startswith("عدّ الجذر")
