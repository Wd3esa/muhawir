"""Who said a passage, from the source list (5 October 2026 evaluation)."""
from muhawir import attribution
from muhawir.corpus import Corpus, Passage, Source
from muhawir.pipeline import _named_views, _opens_with_conclusion, _says_almost_nothing
from muhawir.verify import Claim, verify

CORPUS = Corpus(
    sources={s: Source(s, s, "") for s in ("fatawa-ibn-baz", "fatawa-ibn-uthaymeen", "fatawa-arkan-islam",
                                           "bayyinat", "quranpedia-hafs", "bidayat-al-mujtahid")},
    passages=[
        Passage("z:1", "fatawa-ibn-baz", "ج1", "الاستماع إلى الأغاني والمعازف حرام لما فيه من الصد عن ذكر الله", "fatwa"),
        Passage("z:2", "fatawa-ibn-baz", "ج2", "يجب على المأموم أن يقرأ الفاتحة في سكتات الإمام", "fatwa"),
        Passage("u:1", "fatawa-ibn-uthaymeen", "ج1", "الصحيح أن البسملة يسر بها", "fatwa"),
        Passage("k:1", "fatawa-arkan-islam", "ص1", "الأفضل الإسرار بالبسملة", "fatwa"),
        Passage("by:1", "bayyinat", "ص65", "المسلمون لا يعبدون الكعبة بل يعبدون الله", "qa"),
        Passage("q:2:275", "quranpedia-hafs", "البقرة 275", "وأحل الله البيع وحرم الربا", "quran"),
        Passage("f:1", "bidayat-al-mujtahid", "الطهارة", "اختلفوا في مسح الرأس", "fiqh"),
    ])


def test_meta_lead_names_the_mufti():
    text = attribution.without_meta("الفتوى توضح أن هدف المسلمين إخلاص العبادة لله", ("z:1",), CORPUS)
    assert text == "بيّن الشيخ ابن باز أن هدف المسلمين إخلاص العبادة لله"
    text = attribution.without_meta("الشرح المختصر يبيّن أن المسلمين لا يعبدون الكعبة", ("by:1",), CORPUS)
    assert text == "بيّن كتاب «بينات» أن المسلمين لا يعبدون الكعبة"


def test_meta_lead_with_mixed_sources_is_left():
    text = "الفتوى توضح أن البسملة يسر بها"
    assert attribution.without_meta(text, ("z:1", "u:1"), CORPUS) == text


def test_meta_tail_removed():
    text = attribution.without_meta("يمسح الرأس ثم يمسح الأذنين، وهذا ما ورد في الفتوى.", ("z:2",), CORPUS)
    assert text == "يمسح الرأس ثم يمسح الأذنين."
    text = attribution.without_meta("وقت العصر يبدأ بعد الظهر، وهو ما ذكره الفقهاء في باب أوقات العصر.", ("f:1",), CORPUS)
    assert text == "وقت العصر يبدأ بعد الظهر."


def test_unattributed_ruling_is_credited():
    out = attribution.credited("الاستماع إلى الموسيقى حرام في الإسلام.", ("z:1",), CORPUS)
    assert out == "بحسب الشيخ ابن باز، الاستماع إلى الموسيقى حرام في الإسلام."
    # the two collections of one mufti name him once; two muftis are both named
    out = attribution.credited("الراجح أن البسملة يسر بها.", ("u:1", "k:1"), CORPUS)
    assert out.startswith("بحسب الشيخ ابن عثيمين، ")
    out = attribution.credited("الراجح أن البسملة يسر بها.", ("u:1", "z:2"), CORPUS)
    assert out.startswith("بحسب الشيخ ابن عثيمين والشيخ ابن باز، ")


def test_ruling_from_a_verse_alone_is_not_shown():
    assert attribution.credited("البيع حلال والربا حرام.", ("q:2:275",), CORPUS) is None


def test_attributed_or_plain_sentences_unchanged():
    for text in ("قال الشيخ ابن باز إن الأغاني حرام.", "اتفق الفقهاء على تحريم الخمر.",
                 "المسلمون يعبدون الله وحده."):
        assert attribution.credited(text, ("z:1",), CORPUS) == text


def test_english_ruling_credited():
    out = attribution.credited("Listening to music is forbidden.", ("z:1",), CORPUS, "en")
    assert out == "According to Sheikh Ibn Baz, listening to music is forbidden."


def test_view_named_after_the_mufti_is_kept():
    kept, rejected = verify([Claim("يجب على المأموم قراءة الفاتحة", ("z:2",), school="الشيخ ابن باز")], CORPUS, {"z:2"})
    assert kept and not rejected
    # another mufti's name on Ibn Baz's fatwa is still refused
    kept, rejected = verify([Claim("يجب على المأموم قراءة الفاتحة", ("z:2",), school="ابن عثيمين")], CORPUS, {"z:2"})
    assert not kept and "is not named" in rejected[0].reason
    # «الفتوى» is no name
    kept, rejected = verify([Claim("يجب على المأموم قراءة الفاتحة", ("z:2",), school="الفتوى (z:2)")], CORPUS, {"z:2"})
    assert not kept


def test_one_name_over_several_views_is_not_shown():
    views = [Claim("لا تجب", ("u:1",), school="الشيخ ابن عثيمين"), Claim("ركن", ("u:1",), school="الشيخ ابن عثيمين"),
             Claim("تجب", ("z:2",), school="الشيخ ابن باز")]
    assert _named_views(views) == []


def test_opening_conclusion_and_empty_sentence():
    assert _opens_with_conclusion("وبالتالي، توجّه المسلمين إلى الكعبة لتأكيد التوحيد")
    assert _opens_with_conclusion("Thus, to enter Islam one must declare the testimony.")
    assert not _opens_with_conclusion("الإسلام لم ينتشر بالسيف")
    assert _says_almost_nothing("سبب الاختلاف في ذلك.")
    assert not _says_almost_nothing("يغسل الوجه ثلاثًا.")


def test_prompt_names_the_mufti():
    from muhawir.generate import build_user_prompt
    prompt = build_user_prompt("ما حكم الأغاني", [CORPUS.passage("z:1"), CORPUS.passage("f:1")], "youth", "ar", False)
    assert "[z:1] (فتوى — الشيخ ابن باز — ج1)" in prompt
    assert "[f:1] (فقه — الطهارة)" in prompt


def test_preference_is_credited_even_with_a_hadith():
    out = attribution.credited("الراجح أن السنة الإسرار بها لحديث أبي هريرة أن النبي ﷺ قال…", ("u:1", "k:1"), CORPUS)
    assert out.startswith("بحسب الشيخ ابن عثيمين، الراجح")
    assert attribution.credited("الراجح عند الشيخ ابن عثيمين الإسرار.", ("u:1",), CORPUS) == "الراجح عند الشيخ ابن عثيمين الإسرار."


def test_meta_lead_with_inna():
    text = attribution.without_meta("الفتوى تقول إن العمل في البنوك الربوية غير جائز", ("z:1",), CORPUS)
    assert text == "بيّن الشيخ ابن باز أن العمل في البنوك الربوية غير جائز"


def test_short_sentence_that_says_its_thing_is_kept():
    for text in ("خمس صلوات.", "ماء كثير.", "يغسل الوجه ثلاثًا."):
        assert not _says_almost_nothing(text)


def test_a_sentence_saying_who_with_a_joining_letter_is_not_credited_again():
    for text in ("وقال في موضع آخر إن الغناء حرام.", "وسُئل عن الأغاني فأجاب بأنها حرام."):
        assert attribution.credited(text, ("z:1",), CORPUS) == text


def test_numbered_views_are_not_names():
    views = [Claim("لا تجب", ("u:1",), school="القول الأول"), Claim("ركن", ("u:1",), school="القول الثاني"),
             Claim("تجب", ("z:2",), school="الشيخ ابن باز")]
    assert _named_views(views) == []
