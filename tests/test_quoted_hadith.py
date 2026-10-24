"""A fatwa that quotes a hadith comes after that hadith from its own book (21 October 2026: «اركان الاسلام» found
Ibn Baz's fatwa first, not «بُني الإسلام على خمس» in al-Bukhari and Muslim). Synthetic data only."""
from muhawir.corpus import parse_corpus
from muhawir.pipeline import Muhawir, _hadith_quotes

HADITH = "قال رسول الله صلى الله عليه وسلم بني الإسلام على خمس شهادة أن لا إله إلا الله وأن محمدا رسول الله وإقام الصلاة"
FATWA = ("سئل الشيخ عن أركان الإسلام فأجاب: أركان الإسلام خمسة كما قال النبي صلى الله عليه وسلم: "
         "بني الإسلام على خمس: شهادة أن لا إله إلا الله وأن محمدا رسول الله وإقام الصلاة.")
CORPUS = parse_corpus({"synthetic": True, "sources": [{"id": "s", "name": "مصدر تجريبي", "about": "مصطنع"}], "passages": [
    {"id": "z:1", "source_id": "s", "location": "فتوى", "kind": "fatwa", "text": FATWA},
    {"id": "b:8", "source_id": "s", "location": "حديث", "kind": "hadith", "text": HADITH},
    {"id": "b:9", "source_id": "s", "location": "حديث آخر", "kind": "hadith",
     "text": "قال رسول الله صلى الله عليه وسلم الإيمان بضع وسبعون شعبة والحياء شعبة من الإيمان"},
    {"id": "z:2", "source_id": "s", "location": "فتوى بلا حديث", "kind": "fatwa",
     "text": "سئل الشيخ عن الصلاة في الطائرة فأجاب بأنها تصح إذا استقبل القبلة وأتم الأركان"}]})


def _passages(*ids):
    return [CORPUS.passage(i) for i in ids]


def test_the_quoted_words_of_a_hadith_are_found_in_a_fatwa():
    assert _hadith_quotes(FATWA)[0].startswith("بني الإسلام على خمس")
    assert _hadith_quotes("قال: «الإيمان بضع وسبعون شعبة» والله أعلم") == ["الإيمان بضع وسبعون شعبة"]
    assert _hadith_quotes("قال الشيخ: نعم يجوز") == []  # fewer than four words is not a quotation


def test_the_hadith_a_fatwa_quotes_is_added_before_the_fatwa():
    out = Muhawir(CORPUS, None)._with_quoted_hadiths(_passages("z:1"))
    assert [p.id for p in out] == ["b:8", "z:1"]


def test_a_hadith_already_found_moves_up_before_the_fatwa_and_is_not_repeated():
    out = Muhawir(CORPUS, None)._with_quoted_hadiths(_passages("z:1", "b:9", "b:8"))
    assert [p.id for p in out] == ["b:8", "z:1", "b:9"]


def test_a_fatwa_that_quotes_no_hadith_is_left_as_it_is():
    out = Muhawir(CORPUS, None)._with_quoted_hadiths(_passages("z:2", "b:9"))
    assert [p.id for p in out] == ["z:2", "b:9"]


def test_a_hadith_quoted_word_for_word_from_its_passage_is_not_taken_for_pasting():
    from muhawir.normalize import normalize
    from muhawir.pipeline import _copies_a_source
    said = "روى البخاري ومسلم أن النبي ﷺ قال: «بني الإسلام على خمس شهادة أن لا إله إلا الله وأن محمدا رسول الله»"
    assert not _copies_a_source(said, (), (normalize(HADITH),))
    assert _copies_a_source(said, (), ())  # with no such hadith passage it is still a pasted quotation


def test_a_hadith_cited_from_both_sahih_books_is_called_agreed_upon():
    from muhawir.pipeline import _agreed_upon
    both = {"sahih-bukhari", "sahih-muslim"}
    assert _agreed_upon("روى البخاري ومسلم أن النبي ﷺ قال: «بني الإسلام على خمس»", both) == \
        "في الحديث المتفق عليه (رواه البخاري ومسلم) أن النبي ﷺ قال: «بني الإسلام على خمس»"
    assert _agreed_upon("وروى البخاري ومسلم أن النبي ﷺ قال كذا", both).startswith("وفي الحديث المتفق عليه")
    assert _agreed_upon("روى البخاري ومسلم أن النبي ﷺ قال كذا", {"sahih-bukhari"}).startswith("روى البخاري")
