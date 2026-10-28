"""Fixes from real questions on the live site (19 October 2026): dialects in the fixed rules, a choice between
Arabic and English when another language is asked for, and the scholars' views note only when views are shown.
Fake model calls and synthetic data only."""
import json
from pathlib import Path

import pytest

from muhawir import classify
from muhawir.corpus import load_corpus, parse_corpus
from muhawir.generate import ModelGenerator
from muhawir.messages import TEXT
from muhawir.pipeline import ANSWERED, CHAT, REFERRED, Muhawir

CORPUS = load_corpus(Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json")
VIEWS = parse_corpus({"synthetic": True, "sources": [{"id": "s", "name": "مصدر تجريبي", "about": "مصطنع"}], "passages": [
    {"id": "v:1", "source_id": "s", "location": "أ", "kind": "fiqh",
     "text": "اختلفوا في سقي النخلة: فذهب مالك إلى أن النخلة تحتاج إلى ماء كثير، وذهب الشافعي إلى أنها تكفيها الأمطار."}]})


@pytest.mark.parametrize("text", ["داير أموت", "بدي موت", "عايز اموت", "عاوزه اموت", "بغيت نموت", "ما داير اعيش",
                                  "راح انتحر", "هنتحر", "بدي اقتل حالي", "ابغى اموت", "حابه نموت"])
def test_distress_in_the_main_dialects_gets_the_fixed_caring_reply(text):
    assert classify.check(text).kind == classify.CRISIS


def test_a_divorce_wording_in_a_dialect_is_referred_when_the_understanding_step_reads_it_so():
    assert _model(personal=True).ask("مراتي قلتلها انتي طالق وانا متعصب").status == REFERRED


@pytest.mark.parametrize("text", ["يصير أفطر وأنا مريض؟", "ينفع أفطر وأنا عيان؟", "بقدر أفطر إذا كنت مريض؟",
                                  "داير أفطر وأنا عيان، بجوز لي؟", "فاتتني صلاة الصبح وصحيت الساعة ١٠ أقرأ سرًا أو جهرًا؟"])
def test_a_first_person_question_with_a_known_ruling_is_answered_not_referred(text):
    assert classify.check(text).kind is None


@pytest.mark.parametrize("text", ["ما حكم الانتحار؟", "هل يجوز للمريض أن يفطر؟", "ما حكم قول الرجل لزوجته أنت طالق؟",
                                  "فاتتني الصلاة، ماذا أفعل؟", "ما هو الموت؟", "هل يصير الكافر مسلما بالشهادتين؟",
                                  "متى ينفع الدعاء؟", "داير المصدر وصحه حديث قردة وخنازير"])
def test_general_questions_and_the_word_dair_alone_are_still_answered(text):
    assert classify.check(text).kind is None


def _model(answer_lang="", views=False, feeling="", personal=False, understood="ما حكم سقي النخلة؟", kind="ruling"):
    pid = "v:1" if views else "test-a:1"
    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if '"question"' in keys:
            return json.dumps({"question": understood, "translate": "", "answer_lang": answer_lang,
                               "feeling": feeling, "personal": personal, "kind": kind, "reexplain": False, "recall": "",
                               "queries": ["النخلة"]},
                              ensure_ascii=False)
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if '"problem"' in keys:  # the second reading: one verdict per sentence shown to it
            return json.dumps({"supported": [True] * user.count("<<<", user.find("الجملة 1"))})
        if "plan" in keys or "claims" in keys:
            out = {"plan": "", "abstain": False, "as_list": False, "follow_up": "",
                   "claims": [{"section": "", "label": "", "text": "النخلة تحتاج إلى ماء كثير.", "passage_ids": [pid]}],
                   "views": [{"school": "مالك", "text": "تحتاج إلى ماء كثير.", "passage_ids": [pid]},
                             {"school": "الشافعي", "text": "تكفيها الأمطار.", "passage_ids": [pid]}] if views else []}
            return json.dumps(out, ensure_ascii=False)
        return json.dumps({"ok": True, "verdict": "supported", "reason": ""})
    return Muhawir(VIEWS if views else CORPUS, ModelGenerator([("m", call)]))


def test_another_language_gets_a_choice_between_arabic_english_and_urdu():
    res = _model(answer_lang="other").ask("أريد أحاديث عن الصيام بالإسبانية")
    assert res.status == CHAT and res.message == TEXT["ar"]["choose_lang"]
    assert res.choices == [TEXT["ar"]["choose_ar"], TEXT["ar"]["choose_en"], TEXT["ar"]["choose_ur"]]


def test_an_english_message_gets_the_choice_in_english():
    res = _model(answer_lang="other").ask("hadith about fasting in Spanish please", lang="ar")
    assert res.message == TEXT["en"]["choose_lang"]


def test_an_agreed_ruling_does_not_say_it_presents_the_scholars_views():
    res = _model().ask("ما حكم سقي النخلة؟")
    assert res.status == ANSWERED, res.why
    assert res.note == "" and not res.referral


def test_a_ruling_with_views_keeps_the_scholars_views_note_and_the_card():
    res = _model(views=True).ask("ما حكم سقي النخلة؟")
    assert res.status == ANSWERED, res.why
    assert res.views and res.note == "" and res.referral


# --- the understanding step reads distress and personal cases in any dialect, beyond the word lists ---------

def test_distress_the_word_list_does_not_hold_gets_the_caring_reply_from_the_understanding_step():
    text = "زهجت من الدنيا دي خلاص، داير أفارقها"
    assert classify.check(text).kind is None  # in no word list
    res = _model(feeling="crisis").ask(text)
    assert res.status == REFERRED and res.message == TEXT["ar"]["crisis"] and not res.referral


def test_a_personal_case_the_word_list_does_not_hold_is_referred_from_the_understanding_step():
    text = "أخويه استولى على نصيبي من تركة أبينا، شنو أسوي؟"
    assert classify.check(text).kind is None
    res = _model(personal=True).ask(text)
    assert res.status == REFERRED and res.message.startswith(TEXT["ar"]["personal_case"])


def test_a_general_question_stays_general():
    res = _model(personal=False).ask("ما حكم سقي النخلة؟")
    assert res.status == ANSWERED


# --- a hadith asked about in other words: the closest text, shown as it is -----------------------------------

HADITH = parse_corpus({"synthetic": True, "sources": [{"id": "s", "name": "كتاب تجريبي", "about": "مصطنع"}], "passages": [
    {"id": "h:1", "source_id": "s", "location": "كتاب تجريبي، رقم 1", "kind": "hadith",
     "text": "يكون في أمتي أقوام يغرسون النخل ويسقونه ماء كثيرا في الصيف حتى يثمر التمر"},
    {"id": "h:2", "source_id": "s", "location": "كتاب تجريبي، رقم 2", "kind": "hadith",
     "text": "الجمل يصبر على العطش أياما طويلة في الصحراء"}]})


def _abstaining(corpus):
    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if '"question"' in keys:
            return json.dumps({"question": "ما صحة حديث «سيغرس قوم من أمتي النخل ويسقونه في الصيف»؟", "translate": "",
                               "answer_lang": "", "feeling": "", "personal": False, "kind": "", "reexplain": False,
                               "recall": "", "queries": ["أمتي النخل الصيف"]}, ensure_ascii=False)
        if '"recall"' in keys:
            return json.dumps({"recall": "", "queries": ["النخل"]})
        return json.dumps({"plan": "", "abstain": True, "as_list": False, "claims": [], "views": [], "follow_up": ""})
    return Muhawir(corpus, ModelGenerator([("m", call)]))


def test_a_hadith_in_other_words_gets_the_closest_text_with_a_plain_note():
    res = _abstaining(HADITH).ask("داير صحة حديث «سيغرس قوم من أمتي النخل ويسقونه في الصيف»")
    assert [s["passage_id"] for s in res.sources] == ["h:1"]
    assert res.message.startswith("لم أجد هذا الحديث بلفظه") and "مختصًا في علم الحديث" in res.message


def test_no_closest_text_when_nothing_shares_enough_words_or_no_hadith_is_asked_about():
    assert not _abstaining(HADITH).ask("حديث «سيغرس قوم البرتقال في الشتاء»").sources
    assert not _abstaining(HADITH).ask("هل يسقي قوم من أمتي النخل في الصيف؟").sources


# --- the nightly review: a kept «not found» is answered again from scratch, on this server only --------------

def test_asking_again_replaces_a_kept_not_found(tmp_path):
    from muhawir.pipeline import AnswerStore
    night = {"second": False}

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if '"question"' in keys:
            return json.dumps({"question": "ما حاجة النخلة؟", "translate": "", "answer_lang": "", "feeling": "",
                               "personal": False, "kind": "", "reexplain": False, "recall": "", "queries": ["النخلة"]},
                              ensure_ascii=False)
        if '"recall"' in keys:
            return json.dumps({"recall": "", "queries": ["النخلة"]})
        if '"problem"' in keys:
            return json.dumps({"supported": [True] * user.count("<<<", user.find("الجملة 1"))})
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if not night["second"]:  # the first run, with its retries: nothing usable
            return json.dumps({"plan": "", "abstain": True, "as_list": False, "claims": [], "views": [], "follow_up": ""})
        return json.dumps({"plan": "", "abstain": False, "as_list": False, "views": [], "follow_up": "",
                           "claims": [{"section": "", "label": "", "text": "النخلة تحتاج إلى ماء كثير.",
                                       "passage_ids": ["test-a:1"]}]}, ensure_ascii=False)
    store = AnswerStore(tmp_path / "a.db", "v")
    m = Muhawir(CORPUS, ModelGenerator([("m", call)]), store=store)
    assert m.ask("ما حاجة النخلة؟").status != ANSWERED
    night["second"] = True
    assert m.ask("ما حاجة النخلة؟").status != ANSWERED  # the kept «not found» is served until asked again
    before, now = m.ask_again("ما حاجة النخلة؟")
    assert before is not None and before.status != ANSWERED and now.status == ANSWERED
    assert m.ask("ما حاجة النخلة؟").status == ANSWERED  # the new answer is the one kept


def test_ask_again_is_refused_through_the_web_proxy():
    from fastapi.testclient import TestClient
    from muhawir import server
    client = TestClient(server.app)
    r = client.post("/api/ask_again", json={"question": "ماذا تحتاج النخلة؟"}, headers={"X-Forwarded-For": "1.2.3.4"})
    assert r.status_code == 404
    r = client.post("/api/ask_again", json={"question": "ماذا تحتاج النخلة؟"})  # testclient is not 127.0.0.1
    assert r.status_code == 404


@pytest.mark.parametrize("text", ["ما أريد أعيش بعد", "ما بقيتش باغي نعيش", "مو داير أعيش", "ما ودي أعيش", "باغي نموت"])
def test_distress_in_iraqi_and_maghrebi_found_by_the_self_test(text):
    assert classify.check(text).kind == classify.CRISIS


# --- 19 October 2026: «ربا النسيئة: أن تبيع شيئًا بشيء من غير جنسه مع تأخير القبض» passed the checks ---------

def test_the_second_reading_can_name_a_dropped_condition():
    from muhawir.generate import CHECK_PROMPT, CHECK_SCHEMA, SYSTEM_PROMPT
    assert "dropped_condition" in CHECK_SCHEMA["properties"]["problem"]["items"]["enum"]
    assert "dropped_condition" in CHECK_PROMPT and "ربويًا بربوي" in CHECK_PROMPT
    assert "ولا تحذف عند التبسيط قيدًا" in SYSTEM_PROMPT and "من قال لمن" in SYSTEM_PROMPT


def test_a_sentence_the_reader_finds_with_a_dropped_condition_is_rejected_and_explained():
    from muhawir.pipeline import _feedback
    from muhawir.verify import Claim

    class Rejected:
        claim = Claim("ربا النسيئة أن تبيع شيئًا بشيء من غير جنسه مع تأخير القبض.", ["x:1"])
        reason = "the second reading found: dropped_condition"
    assert "حذفتَ قيدًا أو شرطًا" in _feedback([Rejected()])


def test_a_sentence_does_not_repeat_its_label():
    from muhawir.pipeline import _without_label
    assert _without_label("ربا الفضل أن يبيع ربويًا بجنسه متفاضلًا.", "ربا الفضل") == "أن يبيع ربويًا بجنسه متفاضلًا."
    assert _without_label("وربا النسيئة أن يبيع ربويًا بربوي نسيئة.", "ربا النسيئة") == "أن يبيع ربويًا بربوي نسيئة."
    assert _without_label("ربع العشر من المال.", "المقدار") == "ربع العشر من المال."


@pytest.mark.real_check
def test_one_reading_that_finds_a_dropped_condition_is_enough():
    """Two readings must agree to drop a doubtful sentence, but not one whose meaning was found changed."""
    from muhawir.verify import Claim
    from muhawir.corpus import Passage as P
    seen = {"n": 0}

    def call(system, user, schema=None):
        seen["n"] += 1
        problem = "dropped_condition" if seen["n"] == 1 else "none"  # the second reader would have let it pass
        return json.dumps({"on_topic": [True], "missing": [[]], "evidence": ["يبيع ربويا بربوي نسيئة"],
                           "problem": [problem]}, ensure_ascii=False)
    gen = ModelGenerator([("m", call)])
    passages = {"x:1": P("x:1", "s", "أ", "وربا النسيئة أن يبيع ربويا بربوي نسيئة من غير جنسه", "fatwa", "", "")}
    verdicts = gen.check_support([Claim("ربا النسيئة أن تبيع شيئا بشيء من غير جنسه مع تأخير القبض.", ["x:1"])], passages)
    assert verdicts == [False] and seen["n"] == 1
    assert "dropped_condition" in gen.last_check_reasons[0]


def test_a_saved_wrong_reply_can_be_removed(tmp_path, monkeypatch):
    import subprocess, sys as _sys
    from muhawir.pipeline import AnswerStore, Response
    store = AnswerStore(tmp_path / "a.db", "v")
    store.put(("ما هو الربا", "kids", "ar"), Response("answered", ""))
    store.put(("ما هي الزكاة", "kids", "ar"), Response("answered", ""))
    script = Path(__file__).resolve().parent.parent / "scripts" / "forget.py"
    out = subprocess.run([_sys.executable, str(script), "الربا", "--delete"], capture_output=True, text=True,
                         env={**__import__("os").environ, "MUHAWIR_STORE": str(tmp_path / "a.db")}).stdout
    assert "1 removed" in out and len(store) == 1


def test_tapping_a_language_choice_answers_the_question_before_in_that_language():
    m = _model(answer_lang="other")  # the model still reads «other» from the conversation
    first = m.ask("أريد أحاديث عن الصيام بالإسبانية")
    assert first.choices
    history = [{"role": "user", "text": "أريد أحاديث عن الصيام بالإسبانية"}, {"role": "assistant", "text": first.message}]
    res = m.ask("In English", lang="ar", history=history)
    assert res.status != CHAT and not res.choices  # never asked to choose again


@pytest.mark.parametrize("text", [
    "أريد أحاديث عن الصيام في رمضان",  # hadiths on a topic, not one hadith's wording
    "ما فضل الصدقة؟"])
def test_no_closest_hadith_for_a_topic(text):
    assert not _abstaining(HADITH).ask(text).sources


def test_a_ruling_turned_around_is_rejected_by_code():
    from muhawir.verify import Claim, verify
    corpus = parse_corpus({"synthetic": True, "sources": [{"id": "s", "name": "م", "about": "م"}], "passages": [
        {"id": "p:1", "source_id": "s", "location": "أ", "kind": "fiqh", "text": "يجب الوضوء من أكل لحم الإبل عند أحمد"},
        {"id": "p:2", "source_id": "s", "location": "ب", "kind": "fiqh",
         "text": "قال أحمد يجب الوضوء من لحم الإبل، وقال الجمهور لا يجب"}]})
    kept, rejected = verify([Claim("لا يجب الوضوء من لحم الإبل.", ("p:1",)),
                             Claim("لا يجب الوضوء من لحم الإبل عند الجمهور.", ("p:2",))], corpus, {"p:1", "p:2"})
    assert [c.passage_ids for c in kept] == [("p:2",)] and "turned around" in rejected[0].reason


_AFTER = [{"role": "user", "text": "ما حكم سقي النخلة؟"}, {"role": "assistant", "text": "النخلة تحتاج إلى ماء كثير."}]


@pytest.mark.parametrize("again", ["ما حكم سقي النخلة؟", "ما حكم سقي النخلة", "كيف أسقي النخلة؟"])
def test_a_question_asked_again_is_answered_not_met_with_an_offer_to_explain(again):
    # 20 October 2026: a visitor asked his question again and was told «يبدو أن جوابي لم يكن واضحًا…»
    res = _model(understood="").ask(again, history=_AFTER)
    assert res.status == ANSWERED and res.message != TEXT["ar"]["no_question_after_answer"]


@pytest.mark.parametrize("words", ["ما فهمت", "غبي"])
def test_a_message_with_no_question_after_an_answer_still_gets_the_offer(words):
    res = _model(understood="").ask(words, history=_AFTER)
    assert res.status == CHAT and res.message == TEXT["ar"]["no_question_after_answer"]


def test_an_answer_that_covers_only_part_of_the_question_says_so():
    """21 October 2026: «ما هي الكلمات النادرة في سورة الكهف؟» was answered from the tafsir of two words, faithfully and
    without a word that the sources hold no list. A reply judged partial after its one rewrite carries a notice."""
    judged = []

    class Judge(ModelGenerator):
        def judge_relevance(self, question, claims, passages=None):
            judged.append(1)
            return "partly"

    base = _model()
    res = Muhawir(CORPUS, Judge(base.generator.calls)).ask("ما حكم سقي النخلة؟")
    assert res.status == ANSWERED and TEXT["ar"]["partial_note"] in res.note
    assert len(judged) == 2  # judged, rewritten once, judged again


def test_an_answer_that_covers_the_question_has_no_partial_notice():
    class Judge(ModelGenerator):
        def judge_relevance(self, question, claims, passages=None):
            return "yes"

    res = Muhawir(CORPUS, Judge(_model().generator.calls)).ask("ما حكم سقي النخلة؟")
    assert res.status == ANSWERED and TEXT["ar"]["partial_note"] not in res.note


def test_the_relevance_judge_runs_once_while_the_support_check_runs():
    """Judge-early: the verdict read during the support check is used when the check keeps every sentence."""
    calls = []

    class Judge(ModelGenerator):
        def judge_relevance(self, question, claims, passages=None):
            calls.append(len(claims))
            return "yes"

    res = Muhawir(CORPUS, Judge(_model().generator.calls)).ask("ما حكم سقي النخلة؟")
    assert res.status == ANSWERED and len(calls) == 1


def test_phrases_are_searched_in_parallel_with_the_same_result(monkeypatch):
    from muhawir import pipeline
    one = _model()
    monkeypatch.setattr(pipeline, "SEARCH_THREADS", 1)
    a = [p.id for p in one._gather("ما حكم سقي النخلة؟", "", ["سقي النخلة", "ماء النخل", "حكم الري"])[0]]
    monkeypatch.setattr(pipeline, "SEARCH_THREADS", 4)
    b = [p.id for p in one._gather("ما حكم سقي النخلة؟", "", ["سقي النخلة", "ماء النخل", "حكم الري"])[0]]
    assert a == b


def test_a_question_in_other_words_reuses_the_reply_kept_for_its_plain_form():
    """Cache-by-meaning: the paraphrase costs the understanding step only; no search, writing or checking."""
    pipeline_calls = []
    engine = _model(understood="ما حكم سقي النخلة؟")
    orig = engine._ask

    def counting(*a, **k):
        pipeline_calls.append(1)
        return orig(*a, **k)

    engine._ask = counting
    first = engine.ask("هل يجوز سقي النخل؟")
    second = engine.ask("ما رأي الشرع في ري النخلة؟")
    assert first.status == ANSWERED and second.status == ANSWERED
    assert len(pipeline_calls) == 1 and second.understood == "ما حكم سقي النخلة؟"


def test_a_question_about_muhawir_itself_is_answered_as_a_turn_of_the_conversation():
    """28 October 2026: «لماذا ترد بهذه الطريقة؟» after a quotation was searched for and ended «not found»."""
    res = _model(kind="about", understood="لماذا يرد مُحاور بهذه الطريقة؟").ask(
        "لماذا ترد بهذه الطريقة؟", history=[{"role": "user", "text": "اطلبوا العلم ولو في الصين"},
                                             {"role": "assistant", "text": "اطلبوا العلم ولو بالصين"}])
    assert res.status == CHAT and res.message == TEXT["ar"]["about_muhawir"] and not res.claims


# --- 28 October 2026: replies in the message's language; a reply to Muhawir's question is read as one ---------------

def test_fixed_replies_are_in_the_language_of_the_message_not_the_page():
    m = _model()
    assert m.ask("thanks", lang="ar").message == TEXT["en"]["thanks"]
    assert m.ask("hello", lang="ar").message == TEXT["en"]["small_talk"]
    assert m.ask("شكرا", lang="en").message == TEXT["ar"]["thanks"]


def test_the_understanding_step_reads_a_short_reply_as_the_answer_to_muhawirs_question():
    from muhawir.generate import UNDERSTAND_PROMPT
    assert "ردًا عليه" in UNDERSTAND_PROMPT and "علامات الترقيم" in UNDERSTAND_PROMPT


def test_a_clear_question_put_in_other_words_is_not_asked_back():
    # 28 October 2026: «ما رأي بن باز في البيتكوين» was answered under «إذا كنت تقصد: ما رأي ابن باز…»
    res = _model(understood="ما حكم سقي النخلة في الصيف؟").ask("حكم سقي النخله")
    assert res.understood and res.unsure is False
