"""al-Sakhawi's «المقاصد الحسنة» (21 October 2026): one passage per saying, named by its number and words, and a
saying never given as the Prophet's words ﷺ without a verdict on it. Synthetic text only."""
from muhawir import maqasid
from muhawir.corpus import Passage
from muhawir.pipeline import _saying_notice

FILE = """#META# 020.BookTITLE	:: المقاصد الحسنة في بيان كثير من الأحاديث المشتهرة على الألسنة
#META#Header#End#
### | تقديم
# كلام المحقق.
### | 1-
# من منهج المحقق.
### | الباب الأول: الأحاديث بحسب ترتيب الأحرف
### | حرف الهمزة
### | 125 ms060 -
# حديث: اطلبوا العلم ولو بالصين، البيهقي في الشعب (1) من حديث أنس مرفوعا به،
~~وهو ضعيف، بل قال ابن حبان: إنه باطل لا أصل له. PageV00P000
### | 126 -
# حديث: آفة الكذب النسيان، وسنده ضعيف.
### | الباب الثاني: الأحاديث بحسب ترتيب الأبواب
### | كتاب الإيمان.
# اطلبوا العلم ولو بالصين،
"""


def test_each_saying_is_one_passage_named_by_its_number_and_words():
    _, passages = maqasid.build(FILE, min_entries=1)
    assert [p["location"] for p in passages] == ["رقم 125: اطلبوا العلم ولو بالصين", "رقم 126: آفة الكذب النسيان"]
    assert passages[0]["kind"] == "takhrij" and passages[0]["id"] == "mq:1"
    assert "(1)" not in passages[0]["text"] and "باطل لا أصل له." in passages[0]["text"]
    assert "المحقق" not in " ".join(p["text"] for p in passages)


def _p(kind):
    return Passage(id="mq:1", source_id="maqasid-hasana", location="رقم 125", kind=kind, text="نص")


def test_a_saying_given_as_the_prophets_words_without_a_verdict_gets_one():
    said = "قال النبي ﷺ: اطلبوا العلم ولو بالصين."
    assert "المقاصد الحسنة" in _saying_notice(said, [_p("takhrij")])
    assert _saying_notice("يُروى «اطلبوا العلم ولو بالصين»، وقال السخاوي: ضعيف.", [_p("takhrij")]).endswith("ضعيف.")
    assert _saying_notice(said, [_p("hadith")]) == said
