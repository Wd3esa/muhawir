"""The Sunan, al-Muwatta and the Forty of al-Nawawi (21 October 2026): their hadiths carry the grading of named
scholars, translated into Arabic word by word, and a weak one is named as weak. Synthetic data only."""
from muhawir import sahihayn
from muhawir.corpus import Passage
from muhawir.pipeline import _hadith_notice


def test_a_grading_is_translated_word_by_word_and_never_guessed():
    t = sahihayn.translate_grade
    assert t("Hasan Sahih") == "حسن صحيح"
    assert t("Isnaad Hasan") == "إسناده حسن"
    assert t("Sahih Isnaad") == "صحيح الإسناد"
    assert t("Very Daif") == "ضعيف جدًا"
    assert t("Sahih Bukhari (1023) Sahih Muslim (894)") == "في صحيح البخاري في صحيح مسلم"
    assert t("Something Unknown") is None and t("-") is None


def test_each_grading_keeps_its_scholars_name_and_editors_are_left_out():
    grades = [{"name": "Al-Albani", "grade": "Daif"}, {"name": "Zubair Ali Zai", "grade": "Isnaad Hasan"},
              {"name": "Muhammad Fouad Abd al-Baqi", "grade": "Sahih"}, {"name": "Al-Albani", "grade": "Weird"}]
    assert sahihayn.translate_grades(grades) == "الألباني: ضعيف؛ زبير علي زئي: إسناده حسن"


def _book(grades):
    dump = {"hadiths": [{"arabicnumber": 1, "text": "قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات", "grades": grades}]}
    return sahihayn.build_book("abudawud", dump)[1][0]


def test_a_sunan_passage_carries_its_arabic_grade_or_says_it_has_none(monkeypatch):
    monkeypatch.setitem(sahihayn.MIN_HADITH, "abudawud", 1)  # a one-hadith file stands for the book here
    assert _book([{"name": "Al-Albani", "grade": "Sahih"}])["grade"] == "الألباني: صحيح"
    assert _book([])["grade"] == sahihayn.NO_GRADE
    assert _book([])["id"] == "ad:1"


def _p(grade, source="sunan-abi-dawud", kind="hadith"):
    return Passage(id="ad:1", source_id=source, location="x", kind=kind, text="نص", grade=grade)


def test_a_weak_hadith_is_named_weak_and_a_disputed_one_disputed():
    assert "ضعيف عند المحدثين" in _hadith_notice("روى أبو داود أن النبي ﷺ قال كذا.", [_p("الألباني: ضعيف؛ شعيب الأرناؤوط: ضعيف جدًا")])
    assert "اختلف المحدثون" in _hadith_notice("روى أبو داود حديثًا.", [_p("الألباني: ضعيف؛ زبير علي زئي: حسن")])


def test_sound_graded_sahih_book_and_already_warned_sentences_are_left_alone():
    assert _hadith_notice("روى أبو داود حديثًا.", [_p("الألباني: صحيح")]) == "روى أبو داود حديثًا."
    assert _hadith_notice("روى البخاري حديثًا.", [_p("", source="sahih-bukhari")]) == "روى البخاري حديثًا."
    assert _hadith_notice("وهو حديث ضعيف.", [_p("الألباني: ضعيف")]) == "وهو حديث ضعيف."
