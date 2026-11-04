"""Requested texts (November 2026): a hadith asked for by its meaning («حديث عن الصبر»، «حديث بمعنى هذا»), a verse
named by its place («الآية الخامسة من سورة البقرة»، «الأنعام (٩٦)» in a photographed note), and the check that the
answer shows the text asked for. Synthetic data and no model call."""
from types import SimpleNamespace

from muhawir.corpus import parse_corpus
from muhawir.generate import ModelGenerator
from muhawir.pipeline import Muhawir
from muhawir.presentation import required_quote

_FIQH = "الصبر على البلاء واجب على المسلم وقد ذكر الفقهاء أن الصبر على البلاء من الإيمان والجزع عند البلاء محرم في قول عامة أهل العلم والصبر عند البلاء أجره عظيم."
CORPUS = parse_corpus({"synthetic": True, "sources": [
    {"id": "mushaf", "name": "المصحف", "about": "القرآن"},
    {"id": "fiqh-book", "name": "كتاب فقه", "about": "فقه"},
    {"id": "extra-riyad-salihin", "name": "رياض الصالحين", "about": "أحاديث"},
    {"id": "takhrij-book", "name": "كشف الخفاء", "about": "تخريج"}], "passages": [
    {"id": "q:2:1", "source_id": "mushaf", "location": "سورة البقرة، آية 1", "kind": "quran", "text": "الم"},
    {"id": "q:2:5", "source_id": "mushaf", "location": "سورة البقرة، آية 5", "kind": "quran",
     "text": "أولئك على هدى من ربهم وأولئك هم المفلحون"},
    {"id": "q:6:1", "source_id": "mushaf", "location": "سورة الأنعام، آية 1", "kind": "quran", "text": "الحمد لله الذي خلق السماوات والأرض"},
    {"id": "q:6:96", "source_id": "mushaf", "location": "سورة الأنعام، آية 96", "kind": "quran",
     "text": "فالق الإصباح وجعل الليل سكنا والشمس والقمر حسبانا ذلك تقدير العزيز العليم"},
    {"id": "q:36:1", "source_id": "mushaf", "location": "سورة يس، آية 1", "kind": "quran", "text": "يس"},
    {"id": "q:36:38", "source_id": "mushaf", "location": "سورة يس، آية 38", "kind": "quran",
     "text": "والشمس تجري لمستقر لها ذلك تقدير العزيز العليم"},
    *[{"id": f"f:{i}", "source_id": "fiqh-book", "location": f"باب الصبر {i}", "kind": "fiqh", "text": _FIQH} for i in range(30)],
    {"id": "r:1", "source_id": "extra-riyad-salihin", "location": "باب الصبر", "kind": "other",
     "text": "وعن أبي سعيد رضي الله عنه أن النبي صلى الله عليه وسلم قال: «وما أعطي أحد عطاء خيرا وأوسع من الصبر» متفق عليه."},
    {"id": "t:1", "source_id": "takhrij-book", "location": "حرف الميم", "kind": "takhrij",
     "text": "حديث: «من أخبر الناس بهذا الدعاء فرج الله همه» لا أصل له."}]})
BY_ID = {p.id: p for p in CORPUS.passages}


def _engine():
    return Muhawir(CORPUS, ModelGenerator([("m", lambda *a, **k: "{}")]))


def _claim(text, *ids):
    return SimpleNamespace(text=text, passage_ids=list(ids), school="")


# ---- the requested text is shown ------------------------------------------------------------------------------

def test_a_hadith_from_a_book_of_hadith_added_as_other_is_the_hadith_asked_for():
    assert required_quote(_claim("قال ﷺ: «وما أعطي أحد عطاء خيرا وأوسع من الصبر».", "r:1"), BY_ID, "hadith_text")


def test_a_hadith_shortened_with_an_ellipsis_is_still_found_word_for_word():
    assert required_quote(_claim("«وما أعطي أحد عطاء … وأوسع من الصبر»", "r:1"), BY_ID, "hadith_text")
    assert not required_quote(_claim("«وما أعطي أحد عطاء … أعظم من الشكر»", "r:1"), BY_ID, "hadith_text")


def test_a_saying_from_a_takhrij_book_is_never_shown_as_the_hadith_asked_for():
    assert not required_quote(_claim("«من أخبر الناس بهذا الدعاء فرج الله همه»", "t:1"), BY_ID, "hadith_text")


def test_words_of_a_fiqh_book_are_not_a_hadith_unless_it_quotes_the_prophet():
    assert not required_quote(_claim("«الصبر على البلاء واجب على المسلم»", "f:1"), BY_ID, "hadith_text")


def test_a_verse_is_shown_only_from_the_mushaf():
    assert required_quote(_claim("﴿أولئك على هدى من ربهم﴾", "q:2:5"), BY_ID, "verse_text")
    assert not required_quote(_claim("﴿أولئك على هدى من ربهم﴾", "r:1"), BY_ID, "verse_text")


# ---- a hadith asked for by its meaning is offered to the writer -----------------------------------------------

def test_a_hadith_wanted_by_its_meaning_is_offered_even_when_fiqh_passages_rank_above_it():
    plain, _ = _engine()._gather("حديث عن الصبر على البلاء", "", ["الصبر على البلاء"])
    wanted, _ = _engine()._gather("حديث عن الصبر على البلاء", "", ["الصبر على البلاء"], form="hadith_text")
    assert "r:1" not in [p.id for p in plain]  # thirty fiqh passages fill every place
    assert "r:1" in [p.id for p in wanted]


# ---- verses named by their place ------------------------------------------------------------------------------

def test_verses_noted_by_surah_and_number_in_a_photographed_note_are_offered_first():
    note = "اللهم يا فارج الهم ويا كاشف الغم. آيات تذهب الضيق: الأنعام (٩٦) يس (٣٨)"
    passages, _ = _engine()._gather(note, note, ["تفريج الهم"])
    assert [p.id for p in passages][:2] == ["q:6:96", "q:36:38"]


def test_a_verse_named_by_its_place_is_shown_at_once():
    engine = _engine()
    for q in ("الآية الخامسة من سورة البقرة", "نص الآية الآية الخامسة من سورة البقرة", "آية ٥ من البقرة",
              "اذكر الآية رقم 5 من سورة البقرة", "البقرة 5", "سورة الأنعام (٩٦)"):
        found = engine._verse_by_reference(q)
        assert found is not None and found.id in ("q:2:5", "q:6:96"), q
    res = engine.ask("الآية الخامسة من سورة البقرة")
    assert res.claims and res.claims[0]["passage_ids"] == ["q:2:5"]


def test_a_question_about_the_meaning_or_a_verse_past_the_end_is_not_a_place():
    engine = _engine()
    for q in ("ما تفسير الآية 5 من البقرة", "الآية 99 من البقرة", "الآية الخامسة من سورة الكهف", "ما حكم الصلاة"):
        assert engine._verse_by_reference(q) is None, q


def test_a_reference_inside_a_question_needs_a_mark_so_ordinary_words_are_not_taken_for_surahs():
    engine = _engine()
    assert [p.id for p in engine._referenced_verses("ما تفسير الآية الخامسة من سورة البقرة؟")] == ["q:2:5"]
    assert [p.id for p in engine._referenced_verses("اقرأ:\nالأنعام: 96، والبقرة (1)")] == ["q:6:96", "q:2:1"]
    for text in ("صلاة الأنعام 96 ركعة", "عمري يس 38 سنة", "ركعات الأنعام: 96", "الوقت: 5"):
        assert engine._referenced_verses(text) == [], text


def test_a_hadith_quoted_by_a_fatwa_counts_only_right_after_the_prophet_is_named():
    corpus = parse_corpus({"synthetic": True, "sources": [{"id": "fx", "name": "فتاوى", "about": "فتاوى"}], "passages": [
        {"id": "fx:1", "source_id": "fx", "location": "فتوى 1", "kind": "fatwa",
         "text": "قال النبي صلى الله عليه وسلم: «إنما الأعمال بالنيات وإنما لكل امرئ ما نوى» رواه البخاري. "
                 + "وهذا كلام طويل في شرح المسألة " * 20 + "قال ابن رجب: «النية هي قصد القلب ومريده»."}]})
    by_id = {p.id: p for p in corpus.passages}
    assert required_quote(_claim("«إنما الأعمال بالنيات»", "fx:1"), by_id, "hadith_text")
    assert not required_quote(_claim("«النية هي قصد القلب ومريده»", "fx:1"), by_id, "hadith_text")


# ---- «هل ذُكر ذلك في القرآن؟» ----------------------------------------------------------------------------------

def test_a_question_whether_something_is_in_the_quran_asks_for_the_verse():
    from muhawir.normalize import normalize
    from muhawir.pipeline import _ASKS_QURAN_EVIDENCE
    for q in ("هل ذكر ذلك في القرآن", "هل ذُكر الوضوء في القرآن الكريم؟", "ما الدليل من القرآن على الوضوء", "وين مذكور في القرآن"):
        assert _ASKS_QURAN_EVIDENCE.search(normalize(q)), q
    for q in ("ما حكم قراءة القرآن للحائض", "هل يجوز مس القرآن بغير وضوء", "كيف أتوضأ"):
        assert not _ASKS_QURAN_EVIDENCE.search(normalize(q)), q


def test_a_tafsir_passage_on_a_verse_brings_the_verse_itself_when_the_quran_is_asked_about():
    corpus = parse_corpus({"synthetic": True, "sources": [{"id": "mushaf", "name": "المصحف", "about": "القرآن"},
                                                          {"id": "tabari", "name": "تفسير الطبري", "about": "تفسير"}],
                           "passages": [
        {"id": "q:5:1", "source_id": "mushaf", "location": "سورة المائدة، الآية 1", "kind": "quran", "text": "يا أيها الذين آمنوا أوفوا بالعقود"},
        {"id": "q:5:6", "source_id": "mushaf", "location": "سورة المائدة، الآية 6", "kind": "quran",
         "text": "يا أيها الذين آمنوا إذا قمتم إلى الصلاة فاغسلوا وجوهكم وأيديكم إلى المرافق وامسحوا برءوسكم وأرجلكم إلى الكعبين"},
        {"id": "tb:5:6", "source_id": "tabari", "location": "سورة المائدة، الآية 6", "kind": "tafsir",
         "text": "يعني بقوله إذا قمتم إلى الصلاة إذا قمتم إليها وأنتم على غير طهر فاغسلوا وجوهكم بالماء في الوضوء"}]})
    engine = Muhawir(corpus, ModelGenerator([("m", lambda *a, **k: "{}")]))
    passages, _ = engine._gather("هل ذكر الوضوء في القرآن", "", ["الوضوء"], form="verse_evidence")
    assert passages[0].id == "q:5:6"
    by_id = {p.id: p for p in corpus.passages}
    assert required_quote(_claim("﴿إذا قمتم إلى الصلاة فاغسلوا وجوهكم﴾", "q:5:6"), by_id, "verse_evidence")


def test_the_writer_is_told_to_quote_the_verse_first_and_not_repeat_the_previous_answer():
    from muhawir.generate import VERSE_EVIDENCE_RULE, build_user_prompt
    prompt = build_user_prompt("هل ذكر ذلك في القرآن؟", [], "youth", "ar", False,
                               answer_request={"form": "verse_evidence", "requirements": []})
    assert VERSE_EVIDENCE_RULE in prompt and "هذه الآية" in VERSE_EVIDENCE_RULE


def test_the_verse_is_searched_first_even_when_the_understanding_step_calls_it_an_explanation(monkeypatch):
    import json
    def call(system, user, schema=None):
        if '"question"' in json.dumps(schema or {}):
            return json.dumps({"question": "هل ذكر الوضوء في القرآن؟", "translate": "", "answer_lang": "", "feeling": "",
                               "personal": False, "kind": "", "reexplain": False, "recall": "", "queries": ["الوضوء"],
                               "answer_request": {"form": "explanation", "requirements": []}}, ensure_ascii=False)
        return "{}"
    engine = Muhawir(CORPUS, ModelGenerator([("m", call)]))
    seen = []
    monkeypatch.setattr(engine, "_gather", lambda *a, **k: (seen.append(k.get("form")), ([], []))[1])
    engine.ask("هل ذكر ذلك في القرآن")
    assert seen and seen[0] == "verse_evidence"


def test_words_the_user_quoted_from_a_hadith_or_a_claimed_saying_are_offered_first():
    engine = _engine()
    note = "اللهم يا فارج الهم... قال (ص) من أخبر الناس بهذا الدعاء فرج الله همه"
    pinned = engine._pinned({"quoted_text": "من أخبر الناس بهذا الدعاء فرج الله همه"}, note)
    assert [p.id for p in pinned] == ["t:1"]  # the takhrij book says what it is
    asked = "هل يوجد حديث بمعنى وما أعطي أحد عطاء خيرا وأوسع من الصبر"
    assert [p.id for p in engine._pinned({"quoted_text": "وما أعطي أحد عطاء خيرا وأوسع من الصبر"}, asked)] == ["r:1"]
    # words the user did not write are never used to pick passages
    assert engine._pinned({"quoted_text": "وما أعطي أحد عطاء خيرا وأوسع من الصبر"}, "حديث عن الصبر") == []
