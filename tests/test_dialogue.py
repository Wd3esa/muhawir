"""Dialogue quality: search by phrase, one more search when nothing was found, clean text, tolerant
scholar names, translation requests, the strict second reading. Fake model calls and synthetic data only."""
import json
from pathlib import Path

import pytest

from muhawir import generate, pipeline, store
from muhawir.corpus import load_corpus, parse_corpus
from muhawir.generate import ModelGenerator, _has_evidence
from muhawir.pipeline import ABSTAINED, ANSWERED, TRANSLATED, Muhawir, _strip_ids
from muhawir.store import SqliteCorpus, SqliteRetriever, build_db
from muhawir.verify import Claim, is_named
from muhawir.corpus import Passage

CORPUS = load_corpus(Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json")
QUESTION = "ماذا تحتاج النخلة في الصيف؟"

PHRASES = {"synthetic": True, "sources": [{"id": "s", "name": "مصدر تجريبي", "about": "مصطنع"}], "passages": [
    {"id": "o", "source_id": "s", "location": "أ", "kind": "quran",
     "text": "تثمر النخلة التمر في آخر الصيف لأنها تحتاج إلى ماء كثير وتعطي ظلا للمسافرين في الطريق الطويل"},
    {"id": "c1", "source_id": "s", "location": "ب", "kind": "tafsir",
     "text": "قال المفسر إن النخلة تحتاج إلى ماء كثير ثم كرر أن النخلة تحتاج إلى ماء كثير وأن النخلة تحتاج إلى ماء كثير"},
    {"id": "c2", "source_id": "s", "location": "ج", "kind": "fiqh", "text": "ماء قليل وكثير من التمر عند الجمال"},
]}


@pytest.fixture()
def retriever(tmp_path):
    build_db(PHRASES, tmp_path / "p.db")
    return SqliteRetriever(SqliteCorpus(tmp_path / "p.db"))


# --- search ---------------------------------------------------------------------------------------

def test_the_original_text_comes_before_the_book_that_discusses_it(retriever):
    ids = [h.passage.id for h in retriever.search("تحتاج إلى ماء كثير", k=5)]
    assert ids[:2] == ["o", "c1"]  # c1 repeats the phrase three times and would win on score alone


def test_a_wrong_word_at_the_edge_of_a_recalled_phrase_still_finds_the_passage(retriever):
    ids = [h.passage.id for h in retriever.search("بالتأكيد تثمر النخلة التمر في آخر الصيف", k=5)]
    assert ids[0] == "o"


def test_a_phrase_found_in_too_many_passages_identifies_nothing(retriever, monkeypatch):
    monkeypatch.setattr(store, "MAX_PHRASE_MATCHES", 1)
    assert retriever._phrase_rows(["تحتاج", "ماء", "كثير"], 8) == []  # held by two passages: not selective


# --- one more search when the first found nothing usable ------------------------------------------

def _retrying_model(retry_queries, second_answer=True):
    seen = {"retry": 0, "answers": 0}

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if '"question"' in keys:
            return json.dumps({"question": "ما حاجة شجرة البلح؟", "translate": "", "answer_lang": "", "kind": "",
                               "reexplain": False, "recall": "", "queries": ["بلح"]}, ensure_ascii=False)
        if '"recall"' in keys:
            seen["retry"] += 1
            return json.dumps({"recall": "", "queries": retry_queries}, ensure_ascii=False)
        seen["answers"] += 1
        if "[test-a:1]" in user and second_answer:
            return json.dumps({"abstain": False, "claims": [{"text": "ماء كثير.", "passage_ids": ["test-a:1"]}]})
        return json.dumps({"abstain": True, "claims": []})
    return Muhawir(CORPUS, ModelGenerator([("m", call)])), seen


def test_when_nothing_is_found_the_model_is_asked_once_for_different_phrases():
    m, seen = _retrying_model(["النخلة ماء كثير الصيف"])
    res = m.ask("ما حاجة شجرة البلح؟")
    assert res.status == ANSWERED and res.sources[0]["passage_id"] == "test-a:1" and seen["retry"] == 1


def test_the_retry_does_not_loop_and_ends_in_an_honest_not_found():
    m, seen = _retrying_model(["ثعلب ذهبي"], second_answer=False)
    res = m.ask("ما حاجة شجرة البلح؟")
    assert res.status == ABSTAINED and seen["retry"] == 1


def test_no_retry_for_a_personal_case():
    m, seen = _retrying_model(["النخلة ماء كثير الصيف"])
    m.ask("هل يلزمني سقي شجرة البلح؟")
    assert seen["retry"] == 0


# --- text shown to the reader ---------------------------------------------------------------------

def test_ids_written_in_arabic_letters_or_unclosed_references_are_removed_from_the_text():
    assert _strip_ids("كما يذكر ف:565 أن الحول سنة.", {"f:565"}) == "كما يذكر أن الحول سنة."
    assert _strip_ids("في سورة الكهف (ق:18:51) خلق.", {"q:18:51"}) == "في سورة الكهف خلق."
    assert _strip_ids("وردت في t4:41:30:9543:1 وفي m:2643.02 أيضًا.", set()) == "وردت في وفي أيضًا."
    assert _strip_ids("وفيه قوله: «نص» (تفسير.", set()) == "وفيه قوله: «نص»."


def test_a_verse_reference_is_not_mistaken_for_an_id():
    text = "قال تعالى في سورة ق:16 إنه قريب، وفي الأنعام: 141 وفي ص:23 قصة، والوقت 5:30."
    assert _strip_ids(text, set()) == text


# --- names of scholars ----------------------------------------------------------------------------

PASSAGE = "فقال الشافعي وأصحابه بالوجوب، وقال أبو حنيفة وأصحابه بعدمه، وروي عن مالك."


def test_a_name_in_another_grammatical_case_or_with_his_companions_is_still_that_name():
    assert is_named("أبي حنيفة وأصحابه", PASSAGE) and is_named("الشافعي وأصحابه", PASSAGE)
    assert is_named("الشافعي وأحمد وغيرهم (مس الذكر)", PASSAGE) is False  # أحمد is not in this passage
    assert is_named("الشافعي ومالك وغيرهم (في المسألة)", PASSAGE)


def test_أبو_in_any_case_is_one_name_whatever_the_kunya():
    for kunya in ("حنيفة", "يوسف", "ثور", "عبيد"):
        for label in ("أبو", "أبي", "أبا", "ابو", "ابي", "ابا"):
            for written in ("أبو", "أبي", "أبا"):
                assert is_named(f"{label} {kunya}", f"وقال {written} {kunya} بعدمه.")


def test_a_name_joined_to_the_word_before_it_by_waw_is_still_named():
    # f:189 of «بداية المجتهد» (from the test report): the names follow «و» without a space
    passage = ("والذين قالوا يقتل منهم من أوجب قتله كفرا، وهو مذهب أحمد وإسحاق وابن المبارك، ومنهم من أوجبه حدا "
               "وهو مالك والشافعي وأبو حنيفة، وأصحابه، وأهل الظاهر ممن رأى حبسه وتعزيره حتى يصلي.")
    for label in ("أبي حنيفة", "أبا حنيفة", "أبو حنيفة", "أبي حنيفة وأصحابه", "إسحاق", "ابن المبارك", "مالك", "الشافعي"):
        assert is_named(label, passage), label
    assert is_named("وأبي حنيفة", passage)                    # a «و» joined in the label too
    assert is_named("أبو حنيفة", "قال مالك وأبي حنيفة: لا.")   # the passage in the genitive, joined by «و»
    assert not is_named("الثوري", passage)                    # a name the passage does not have stays rejected
    assert not is_named("أبي يوسف", passage)                  # same kunya word, different name


def test_a_school_the_passage_does_not_name_is_not_accepted():
    assert not is_named("الحنابلة", PASSAGE) and not is_named("المالكيون", PASSAGE) and not is_named("", PASSAGE)


# --- translation request --------------------------------------------------------------------------

def test_a_translation_request_is_translated_even_when_the_question_field_is_empty():
    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if '"question"' in keys:
            return json.dumps({"question": "", "translate": "التوحيد", "answer_lang": "en", "kind": "",
                               "reexplain": False, "recall": "", "queries": []}, ensure_ascii=False)
        return json.dumps({"translation": "Monotheism (Tawhid)"})
    res = Muhawir(CORPUS, ModelGenerator([("m", call)])).ask("ترجم كلمة التوحيد إلى الإنجليزية")
    assert res.status == TRANSLATED and res.message == "Monotheism (Tawhid)"


# --- the note on rulings --------------------------------------------------------------------------

def _kind_model(kind):
    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if '"question"' in keys:
            return json.dumps({"question": QUESTION, "translate": "", "answer_lang": "", "kind": kind,
                               "reexplain": False, "recall": "", "queries": []}, ensure_ascii=False)
        return json.dumps({"abstain": False, "claims": [{"text": "ماء كثير.", "passage_ids": ["test-a:1"]}]})
    return Muhawir(CORPUS, ModelGenerator([("m", call)]))


def test_a_ruling_answer_carries_a_fixed_notice_that_it_is_not_a_fatwa():
    assert "ليس فتوى" in _kind_model("ruling").ask(QUESTION).note
    assert _kind_model("what").ask(QUESTION).note == ""


# --- the strict second reading --------------------------------------------------------------------

def test_the_instructions_ask_for_search_phrases_in_the_sources_own_wording():
    p = generate.UNDERSTAND_PROMPT
    assert "بلفظها كما وردت" in p and "عناوين موضوعات" in p and "recall" in p
    assert "سؤال ضمني" in p  # a topic named without a question is a question about it
    assert generate.QUERY_RULES in generate.RETRY_PROMPT


def test_the_second_reading_must_quote_words_that_are_really_in_the_cited_passage():
    passages = {"a": Passage("a", "s", "ل", "تحتاج النخلة إلى ماء كثير في الصيف")}
    claim = Claim("تحتاج النخلة إلى ماء كثير.", ("a",))
    assert _has_evidence(claim, "تحتاج النخلة إلى ماء كثير", passages)
    assert not _has_evidence(claim, "", passages)                      # nothing quoted
    assert not _has_evidence(claim, "العدل صفة من صفات الله تعالى", passages)  # words that are not there
    assert _has_evidence(Claim("يغسل الوجه.", ("b",)), "فاغسلوا وجوهكم وأيديكم",
                         {"b": Passage("b", "s", "ل", "إذا قمتم إلى الصلاة فاغسلوا وجوهكم وأيديكم إلى المرافق")})
    assert _has_evidence(Claim("مثلًا، الماء للعطشان.", ("a",)), "", passages)  # an illustration quotes nothing


@pytest.mark.real_check
def test_a_sentence_whose_quoted_support_is_not_in_the_passage_is_dropped():
    answer = {"abstain": False, "claims": [{"text": "تحتاج النخلة إلى ماء كثير.", "passage_ids": ["test-a:1"]},
                                           {"text": "وهذا يدل على عدل الله.", "passage_ids": ["test-a:1"]}]}

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "problem" in keys:  # the second sentence is "approved" by a careless reader but quotes nothing real
            if "تحتاج النخلة" in user.split("الجملة 1: <<<", 1)[1].split(">>>", 1)[0]:
                return json.dumps({"missing": [[]], "evidence": ["تحتاج إلى ماء كثير في الصيف"], "supported": [True]},
                                  ensure_ascii=False)
            return json.dumps({"missing": [[]], "evidence": ["العدل صفة من صفات الله"], "supported": [True]},
                              ensure_ascii=False)
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if "queries" in keys:
            return '{"queries": []}'
        return json.dumps(answer, ensure_ascii=False)
    res = Muhawir(CORPUS, ModelGenerator([("m", call)])).ask(QUESTION)
    assert [c["text"] for c in res.claims] == ["تحتاج النخلة إلى ماء كثير."]


def test_instructions_forbid_added_conclusions_book_structure_and_neighbouring_topics():
    s, c = generate.SYSTEM_PROMPT, generate.CHECK_PROMPT
    assert "وهذا يدل على" in s and "ترتيب الكتاب" in s and "مسألة مجاورة" in s
    assert "لا يصف كيف تُؤدّى عبادة" in s
    assert "استنتاج أو تعليق أو تقييم" in c and "ترتيب الكتاب" in c and "مثال يصف كيف تُؤدّى عبادة" in c


# --- the answer must reply to the question asked --------------------------------------------------

def _judged(verdicts, answers=None, retry=None):
    """A model whose relevance judge says what `verdicts` lists, one per call (an Exception raises)."""
    verdicts, seen = iter(verdicts), {"answer_prompts": [], "judged": 0}
    answers = iter(answers or [{"abstain": False, "claims": [{"text": "ماء كثير.", "passage_ids": ["test-a:1"]}]}] * 3)

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if '"question"' in keys:
            return json.dumps({"question": QUESTION, "translate": "", "answer_lang": "", "kind": "", "reexplain": False,
                               "recall": "", "queries": []}, ensure_ascii=False)
        if "verdict" in keys:
            seen["judged"] += 1
            v = next(verdicts)
            if isinstance(v, Exception):
                raise v
            return json.dumps({"verdict": v})
        if '"recall"' in keys:
            return json.dumps({"recall": "", "queries": retry or []}, ensure_ascii=False)
        seen["answer_prompts"].append(user)
        return json.dumps(next(answers), ensure_ascii=False)
    return Muhawir(CORPUS, ModelGenerator([("m", call)])), seen


def test_a_sourced_answer_about_another_matter_is_not_shown(monkeypatch):
    monkeypatch.setattr(pipeline, "DEBUG", True)
    m, seen = _judged(["no"])
    res = m.ask(QUESTION)
    assert res.status == ABSTAINED and res.claims == [] and "another matter" in res.why


def test_an_off_topic_first_answer_triggers_the_second_search_and_a_better_answer():
    m, seen = _judged(["no", "yes"], retry=["الجمل العطش الصحراء"])  # reaches a passage not offered the first time
    res = m.ask(QUESTION)
    assert res.status == ANSWERED and seen["judged"] == 2 and len(seen["answer_prompts"]) == 2


def test_a_partial_answer_is_rewritten_once_and_told_what_is_missing():
    full = {"abstain": False, "claims": [{"text": "ماء كثير.", "passage_ids": ["test-a:1"]},
                                         {"text": "وتثمر التمر في آخره.", "passage_ids": ["test-a:1"]}]}
    first = {"abstain": False, "claims": [{"text": "ماء كثير.", "passage_ids": ["test-a:1"]}]}
    m, seen = _judged(["partly"], answers=[first, full])
    res = m.ask(QUESTION)
    assert [c["text"] for c in res.claims] == ["ماء كثير.", "وتثمر التمر في آخره."]
    assert len(seen["answer_prompts"]) == 2 and "لا يغطي السؤال كله" in seen["answer_prompts"][1]
    assert seen["judged"] == 1  # one rewrite only, and the rewrite is not judged again


def test_when_the_judge_cannot_run_the_checked_answer_stands():
    m, _ = _judged([RuntimeError("down")])
    assert m.ask(QUESTION).status == ANSWERED


def test_the_rewrite_request_says_why_each_sentence_was_rejected():
    from muhawir.verify import Rejected
    text = pipeline._feedback([Rejected(Claim("جملة منقولة.", ("a",)), "copied a source sentence instead of explaining it"),
                               Rejected(Claim("جملة زائدة.", ("a",)), "not supported by the cited passage")])
    assert "جملة منقولة.  ← نقلتَ نص المقطع" in text and "جملة زائدة.  ← فيها ما ليس في المقطع" in text
    assert "ولو بدا الجواب أقصر" in text and "لا يغطي السؤال كله" not in text
    assert "لا يغطي السؤال كله" in pipeline._feedback([], incomplete=True)


def test_a_short_quotation_inside_an_explanation_is_allowed_but_a_pasted_sentence_is_not():
    from muhawir.pipeline import _copies_a_source
    pasted = "حيث ذُكر: «تحتاج النخلة إلى ماء كثير في الصيف»"
    explained = "أخبرنا النبي ﷺ أن الناس سيظلون يتساءلون حتى يقولوا هذه العبارة: «هذا الله خالق كل شيء فمن خلق الله» وأمرنا بالاستعاذة."
    assert _copies_a_source(pasted) and not _copies_a_source(explained)
    assert not _copies_a_source("يخبرنا الله تعالى أنه خالق كل شيء.")


def test_an_everyday_example_that_likens_a_religious_matter_to_daily_life_is_dropped():
    from muhawir.pipeline import _compares_to_daily_life
    assert _compares_to_daily_life("مثلاً، إذا كان لديك حلوى كثيرة وتشاركها مع أصدقائك، فهذا يشبه الزكاة.")
    assert _compares_to_daily_life("مثلًا، في المدرسة نتبع خمس قواعد، وهذا يشبه أركان الإسلام.")
    assert not _compares_to_daily_life("مثلًا، من يترك كتابه في الصيف ثم يعود إليه بعد سنة يكون قد مرّ عليه حول.")
    assert not _compares_to_daily_life("الزكاة تشبه الضريبة عند بعض الناس.")  # not an example sentence: judged by the checks


def test_explaining_again_starts_with_a_short_human_line_and_the_next_turn_still_finds_the_answer():
    history = [{"role": "user", "text": QUESTION}, {"role": "assistant", "text": "تحتاج النخلة إلى ماء كثير في الصيف."}]

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if '"question"' in keys:
            return json.dumps({"question": QUESTION, "translate": "", "answer_lang": "", "kind": "", "reexplain": True,
                               "recall": "", "queries": []}, ensure_ascii=False)
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        return json.dumps({"abstain": False, "claims": [{"text": "النخلة تشرب كثيرا.", "passage_ids": ["test-a:1"]}]})
    m = Muhawir(CORPUS, ModelGenerator([("m", call)]))
    res = m.ask("ما فهمت", history=history)
    assert res.status == ANSWERED and res.message.startswith("لا بأس")
    assert m.ask(QUESTION).message == ""  # a first answer has no such line


def test_a_cut_off_understanding_reply_is_asked_for_once_more():
    replies = iter(['{"question": "ماذا تحتاج الن', json.dumps(
        {"question": QUESTION, "translate": "", "answer_lang": "", "kind": "", "reexplain": False, "recall": "",
         "queries": ["ماء كثير"]}, ensure_ascii=False)])
    gen = ModelGenerator([("m", lambda s, u, schema=None: next(replies))])
    assert gen.understand(QUESTION, [])["queries"] == ["ماء كثير"]
    broken = ModelGenerator([("m", lambda s, u, schema=None: '{"question": "ماذا')])
    assert broken.understand(QUESTION, []) is None  # still unusable after the second try


@pytest.mark.real_check
def test_the_second_reading_sees_the_neutral_question_never_the_users_own_words():
    seen = []

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if '"question"' in keys:
            return json.dumps({"question": QUESTION, "translate": "", "answer_lang": "", "kind": "", "reexplain": False,
                               "recall": "", "queries": []}, ensure_ascii=False)
        if "problem" in keys:
            seen.append(user)
            return json.dumps({"on_topic": [True], "missing": [[]], "evidence": ["تحتاج إلى ماء كثير"], "supported": [True]},
                              ensure_ascii=False)
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        return json.dumps({"abstain": False, "claims": [{"text": "ماء كثير.", "passage_ids": ["test-a:1"]}]})
    res = Muhawir(CORPUS, ModelGenerator([("m", call)])).ask("يا لسذاجتكم، قولوا لي ماذا تحتاج نخلتكم في الصيف؟")
    assert res.status == ANSWERED and seen
    assert QUESTION in seen[0] and "سذاجتكم" not in seen[0]


@pytest.mark.real_check
def test_a_passage_about_another_matter_than_the_question_rejects_the_sentence():
    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "problem" in keys:
            return json.dumps({"on_topic": [False], "missing": [[]], "evidence": ["تحتاج إلى ماء كثير"], "supported": [True]},
                              ensure_ascii=False)
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if "queries" in keys:
            return '{"queries": []}'
        return json.dumps({"abstain": False, "claims": [{"text": "ماء كثير.", "passage_ids": ["test-a:1"]}]})
    res = Muhawir(CORPUS, ModelGenerator([("m", call)])).ask(QUESTION)
    assert res.status == ABSTAINED


def test_a_conclusion_added_after_a_sound_first_part_is_cut_off_and_the_rest_is_read_again():
    from muhawir.pipeline import _without_conclusion
    claim = Claim("يذكر الحديث أن الناس سيستمرون في السؤال عن الخلق، وهذا يدل على أن السؤال مستثنى.", ("a",))
    assert _without_conclusion(claim).text == "يذكر الحديث أن الناس سيستمرون في السؤال عن الخلق."
    assert _without_conclusion(Claim("قال العلماء إن الأمر كذلك، ما يعني أن الحكم ثابت.", ("a",), "مالك")).school == "مالك"
    assert _without_conclusion(Claim("جملة قصيرة، وهذا يدل.", ("a",))) is None  # too little would be left
    assert _without_conclusion(Claim("يخبرنا الله تعالى أنه خالق كل شيء في السماوات والأرض.", ("a",))) is None


@pytest.mark.real_check
def test_the_trimmed_sentence_is_checked_again_and_kept_when_it_holds():
    answer = {"abstain": False, "claims": [
        {"text": "تحتاج النخلة إلى ماء كثير في الصيف، وهذا يدل على عدل الله.", "passage_ids": ["test-a:1"]}]}

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "problem" in keys:
            ok = "يدل على" not in user.split("الجملة 1: <<<", 1)[1].split(">>>", 1)[0]
            return json.dumps({"on_topic": [True], "missing": [[]], "evidence": ["تحتاج إلى ماء كثير"], "supported": [ok]},
                              ensure_ascii=False)
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if "queries" in keys:
            return '{"queries": []}'
        return json.dumps(answer, ensure_ascii=False)
    res = Muhawir(CORPUS, ModelGenerator([("m", call)])).ask(QUESTION)
    assert res.status == ANSWERED and [c["text"] for c in res.claims] == ["تحتاج النخلة إلى ماء كثير في الصيف."]


def _reading_model(says, default=True):
    """A model whose second reading says what `says` lists, one answer per reading, then `default`."""
    import threading
    lock, calls = threading.Lock(), {"n": 0}
    answers = iter(says)

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "problem" in keys:
            with lock:
                calls["n"] += 1
                ok = next(answers, default)
            return json.dumps({"on_topic": [True], "missing": [[]], "evidence": ["تحتاج إلى ماء كثير"],
                               "problem": ["none" if ok else "addition"]}, ensure_ascii=False)
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if "queries" in keys:
            return '{"queries": []}'
        return json.dumps({"abstain": False, "claims": [{"text": "ماء كثير.", "passage_ids": ["test-a:1"]}]})
    return Muhawir(CORPUS, ModelGenerator([("m", call)])), calls


@pytest.mark.real_check
def test_a_sentence_the_first_reading_accepts_is_read_once():
    m, calls = _reading_model([True])
    assert m.ask(QUESTION).status == ANSWERED and calls["n"] == 1


@pytest.mark.real_check
def test_a_sentence_is_dropped_only_when_two_readings_reject_it():
    m, calls = _reading_model([False, False], default=False)
    res = m.ask(QUESTION)
    assert res.status == ABSTAINED and calls["n"] >= 2  # (the rewrite asks again, and is rejected the same way)
    m, calls = _reading_model([False, True])  # a chance "no": the second reading keeps the sentence
    assert m.ask(QUESTION).status == ANSWERED and calls["n"] == 2


def _named_defects(defects):
    """A model whose second reading names the defect of each sentence of TWO ("none" = no defect)."""
    texts = [c["text"] for c in TWO_CLAIMS["claims"]]

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "problem" in keys:
            mine = defects[texts.index(user.split("الجملة 1: <<<", 1)[1].split(">>>", 1)[0])]
            return json.dumps({"on_topic": [True], "missing": [[]], "evidence": ["تحتاج إلى ماء كثير"], "problem": [mine]},
                              ensure_ascii=False)
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if "queries" in keys:
            return '{"queries": []}'
        return json.dumps(TWO_CLAIMS, ensure_ascii=False)
    return Muhawir(CORPUS, ModelGenerator([("m", call)]))


TWO_CLAIMS = {"abstain": False, "claims": [{"text": "تحتاج النخلة إلى ماء كثير.", "passage_ids": ["test-a:1"]},
                                           {"text": "لا تحتاج النخلة إلى الماء.", "passage_ids": ["test-a:1"]}]}


@pytest.mark.real_check
def test_a_sentence_is_rejected_only_when_the_reader_names_a_defect():
    res = _named_defects(["none", "distortion"]).ask(QUESTION)
    assert res.status == ANSWERED and [c["text"] for c in res.claims] == ["تحتاج النخلة إلى ماء كثير."]
    res = _named_defects(["none", "none"]).ask(QUESTION)
    assert len(res.claims) == 2


@pytest.mark.real_check
def test_the_reason_a_sentence_was_rejected_names_the_defect(monkeypatch):
    monkeypatch.setattr(pipeline, "DEBUG", True)
    res = _named_defects(["none", "conclusion"]).ask(QUESTION)
    assert "the second reading found: conclusion" in res.why


# --- ruling words ---------------------------------------------------------------------------------

def test_a_sentence_may_not_state_another_kind_of_ruling_than_the_passage_it_cites():
    from muhawir.verify import mismatched_rulings
    passage = "تجب الزكاة في الذهب والفضة والإبل."
    assert mismatched_rulings("تجوز الزكاة في الذهب.", [passage]) == {"permitted"}
    assert mismatched_rulings("يجب إخراج الزكاة من الذهب.", [passage]) == set()
    assert mismatched_rulings("وفرض على المسلم أن يخرجها.", [passage]) == set()  # «فرض» is the same kind as «تجب»
    assert mismatched_rulings("لا يجوز ترك الزكاة.", [passage]) == set()  # «not permitted» may paraphrase «obligatory»
    assert mismatched_rulings("تجوز الزكاة في الذهب.", ["الذهب معدن نفيس يُستخرج من الأرض."]) == set()  # nothing to compare


def test_verify_drops_a_sentence_with_another_kind_of_ruling_and_says_why():
    from muhawir.verify import verify
    corpus = parse_corpus({"synthetic": True, "sources": [{"id": "s", "name": "ت", "about": "ت"}], "passages": [
        {"id": "p", "source_id": "s", "location": "ل", "text": "تجب الزكاة في الذهب والفضة والإبل والغنم والبقر."}]})
    kept, rejected = verify([Claim("تجوز الزكاة في الذهب والفضة.", ("p",)), Claim("تجب الزكاة في الذهب والفضة.", ("p",))],
                            corpus, {"p"})
    assert [c.text for c in kept] == ["تجب الزكاة في الذهب والفضة."]
    assert rejected[0].reason.startswith("ruling word not in the cited passage")


# --- no conclusions of its own, resolved ids --------------------------------------------------------

def test_an_explicit_inference_at_the_end_of_a_sentence_is_cut_off_whatever_the_checks_say():
    from muhawir.pipeline import _without_inference
    cases = {
        "روى أنس بن مالك أن النبي ضحك ثم نزلت عليه سورة الكوثر، ما يدل على أن الوحي نزل استجابة لظرف.":
            "روى أنس بن مالك أن النبي ضحك ثم نزلت عليه سورة الكوثر.",
        "التفسير يوضح أن الله هو القادر على الخلق، مما يؤكد أنه ليس مخلوقًا بل هو من يخلق.":
            "التفسير يوضح أن الله هو القادر على الخلق.",
        "القرآن يطرح سؤالًا في سورة النحل عن الخلق، وبالتالي لا يُنسب إليه الخلق نفسه.":
            "القرآن يطرح سؤالًا في سورة النحل عن الخلق.",
        "ذكر الحديث أن للأبوين لكل منهما السدس، وهذا يؤكد أن حصة المرأة نصف حصة الرجل.":
            "ذكر الحديث أن للأبوين لكل منهما السدس.",
    }
    for text, expected in cases.items():
        assert _without_inference(Claim(text, ("a",), "")).text == expected
    plain = Claim("ذكر الطبري أن هذا الأمر واجب على كل مسلم بالغ عاقل.", ("a",))
    assert _without_inference(plain) is plain  # «هذا الأمر» is not an inference
    gloss = Claim("الحول، أي سنة كاملة تمر على المال عند صاحبه.", ("a",))
    assert _without_inference(gloss) is gloss  # a gloss («أي أن…») is only cut when the second reading rejected the sentence


def test_an_id_written_without_its_last_part_is_resolved_to_the_one_offered_id():
    from muhawir.pipeline import _resolve_ids
    allowed = {"t4:2:255:4947:1", "t4:2:255:4948:1", "q:2:255", "m:82.01", "m:83.01", "m:83.02", "b:4895"}
    ids = lambda *cited: _resolve_ids(Claim("جملة.", cited), allowed).passage_ids  # noqa: E731
    assert ids("t4:2:255:4947") == ("t4:2:255:4947:1",)
    assert ids("m:82") == ("m:82.01",)           # cut at the dot of a Muslim id
    assert ids("t4:2:255") == ()                 # begins two offered ids: dropped
    assert ids("m:83") == ()
    assert ids("x:9") == ()                      # begins none: dropped
    assert ids("b:48") == ()                     # the start of a number is not a cut id: b:4895 is another hadith
    assert ids("q:2:255", "x:9", "m:82") == ("q:2:255", "m:82.01")  # the others stay
    claim = Claim("جملة.", ("q:2:255",))
    assert _resolve_ids(claim, allowed) is claim


def test_a_sentence_whose_only_ids_are_made_up_has_no_citation_and_is_not_shown():
    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if "queries" in keys:
            return '{"queries": []}'
        return json.dumps({"abstain": False, "claims": [
            {"text": "ماء كثير.", "passage_ids": ["test-a:99"]},
            {"text": "وتثمر في آخر الصيف.", "passage_ids": ["test-a:98", "test-a:1"]}]})
    res = Muhawir(CORPUS, ModelGenerator([("m", call)])).ask(QUESTION)
    assert [c["text"] for c in res.claims] == ["وتثمر في آخر الصيف."]
    assert [c["passage_ids"] for c in res.claims] == [["test-a:1"]]  # the made-up id is gone, the real one stays


def test_ids_and_source_numbers_are_taken_out_of_the_sentence_before_any_check():
    m = Muhawir(CORPUS, ModelGenerator([("m", lambda *a, **k: "{}")]))
    allowed = {"test-a:1", "b:7426"}
    text = ("كما ذكر المفسر (t4:48:26:9994:1) في حديث صحيح البخاري رقم 7296 وفي مسلم (رقم 726) "
            "وفي كتاب برقم 5641 وما رواه a460:108:1:4:1 وفي [m:82.01] وكذلك (b:7426) وهذا مذكور في ف:565.")
    [claim] = m._tidy([Claim(text, ("test-a:1",))], allowed)
    assert claim.text == "كما ذكر المفسر في حديث صحيح البخاري وفي مسلم وفي كتاب وما رواه وفي وكذلك وهذا مذكور في."
    for leftover in (":", "رقم", "7296", "726", "5641", "()"):
        assert leftover not in claim.text
    plain = Claim("أخبرنا النبي ﷺ أن الأعمال بالنيات، وعدد الأرقام كثير، وفي الأنعام: 141.", ("test-a:1",))
    assert m._tidy([plain], allowed) == [plain]  # nothing to take out: the same sentence, a verse number stays
    assert m._tidy([Claim("(رقم 7296)", ("test-a:1",))], allowed) == []  # nothing left of it


@pytest.mark.real_check
def test_the_second_reading_never_sees_ids_or_source_numbers_in_the_sentence():
    seen = []

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "problem" in keys:
            seen.append(user)
            return json.dumps({"on_topic": [True], "missing": [[]], "evidence": ["تحتاج إلى ماء كثير"],
                               "problem": ["none"]}, ensure_ascii=False)
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if "queries" in keys:
            return '{"queries": []}'
        return json.dumps({"abstain": False, "claims": [
            {"text": "تحتاج النخلة إلى ماء كثير كما في (t4:48:26:9994:1) وفي حديث البخاري رقم 7296.",
             "passage_ids": ["test-a:1"]}]}, ensure_ascii=False)
    res = Muhawir(CORPUS, ModelGenerator([("m", call)])).ask(QUESTION)
    assert res.status == ANSWERED and seen
    sentence = seen[0].split("الجملة 1: <<<", 1)[1].split(">>>", 1)[0]
    assert sentence == "تحتاج النخلة إلى ماء كثير كما في وفي حديث البخاري."


WEAK = {"synthetic": True, "sources": [{"id": "s", "name": "كتاب فقه", "about": "مصطنع"}], "passages": [
    {"id": "w1", "source_id": "s", "location": "الزكاة", "kind": "fiqh",
     "text": "وقد روي مرفوعا من حديث ابن عمر عن النبي: لا زكاة في مال حتى يحول عليه الحول، وسبب الاختلاف أنه لم يرد في ذلك حديث ثابت."}]}


def test_the_instructions_ask_to_say_that_a_hadith_is_weak_and_to_reject_a_sentence_that_hides_it():
    s, c = generate.SYSTEM_PROMPT, generate.CHECK_PROMPT
    for phrase in ("لم يرد فيه حديث ثابت", "لا يخلو من مقال", "ضعيف", "مرسل"):
        assert phrase in s and phrase in c
    assert "فيجب أن يقول جوابك ذلك صراحةً كلما استعملت هذا الحديث" in s
    assert "على أنه ثابت، أو ذكر هذا الحديث مع حذف هذا التنبيه" in c
    assert c.index("على أنه ثابت") > c.index('"distortion"')  # filed under the defect the reader must name


@pytest.mark.real_check
def test_a_sentence_that_gives_a_weak_hadith_as_established_is_rejected_and_the_one_that_warns_is_kept():
    seen = {"system": [], "user": []}
    warned = "ويُروى في ذلك حديث عن ابن عمر، لكنه لم يثبت."
    unwarned = "ثبت عن ابن عمر عن النبي أنه قال: لا زكاة في مال حتى يحول عليه الحول."

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "problem" in keys:  # a reader that applies the rule it is given
            seen["system"].append(system)
            seen["user"].append(user)
            sentence = user.split("الجملة 1: <<<", 1)[1].split(">>>", 1)[0]
            weak = "لم يرد في ذلك حديث ثابت" in user and "لم يثبت" not in sentence
            return json.dumps({"on_topic": [True], "missing": [[]], "evidence": ["لا زكاة في مال حتى يحول عليه الحول"],
                               "problem": ["distortion" if weak else "none"]}, ensure_ascii=False)
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if "queries" in keys:
            return '{"queries": []}'
        return json.dumps({"abstain": False, "claims": [{"text": warned, "passage_ids": ["w1"]},
                                                         {"text": unwarned, "passage_ids": ["w1"]}]}, ensure_ascii=False)
    res = Muhawir(parse_corpus(WEAK), ModelGenerator([("m", call)])).ask("ما الحول في الزكاة؟")
    assert [c["text"] for c in res.claims] == [warned]
    # the reader was given the rule, and saw the passage that says the hadith is not established beside each sentence
    assert generate.CHECK_PROMPT in seen["system"][0] and "لم يرد في ذلك حديث ثابت" in seen["user"][0]


# a460:108:1:4:1 (first paragraph): says Kaab ibn al-Ashraf disbelieved, and nothing about his tribe
KAAB = ("بأنه المنبتر من قومه ورده عليهم بأنهم خير من رسول الله - صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ - فأنزل الله السورة "
        "على رسوله بالمدينة مبشراً إياه بالكوثر ومخبره بأن مبغضه هو الأقطع وأنزل عليه الآية أيضاً في سورة النساء "
        "وأخبره بأن كعب بن الأشرف من الذين يؤمنون بالجبت والطاغوت.")


def test_the_instructions_reject_an_added_fact_about_a_person():
    c = generate.CHECK_PROMPT
    assert "قبيلته" in c and "قرابته" in c and "ولو كانت معروفة" in c
    assert c.index("قبيلته") < c.index("2. evidence")                  # in the list of missing details
    assert c.index("عن شخص", c.index('"addition"')) < c.index('"conclusion"')  # and in the definition of an addition


@pytest.mark.real_check
def test_a_fact_added_about_a_person_that_the_cited_passage_does_not_give_is_rejected():
    assert "قريش" not in KAAB  # the premise: the passage does not say which tribe he belonged to
    pid = "a460:108:1:4:1"
    passages = {pid: Passage(pid, "s", "سورة الكوثر، الآية 1", KAAB, "asbab")}
    kept = Claim("أخبر الله رسوله أن كعب بن الأشرف من الذين يؤمنون بالجبت والطاغوت.", (pid,))
    added = Claim("كعب بن الأشرف من قريش.", (pid,))

    def call(system, user, schema=None):  # a reader that applies the rule: a tribe the passage does not name is missing
        sentence = user.split("الجملة 1: <<<", 1)[1].split(">>>", 1)[0]
        passage = user.split("المقاطع:", 1)[1]
        missing = [t for t in ("قريش", "تميم", "الأوس", "الخزرج") if t in sentence and t not in passage]
        return json.dumps({"on_topic": [True], "missing": [missing], "problem": ["none"],
                           "evidence": ["كعب بن الأشرف من الذين يؤمنون بالجبت والطاغوت"]}, ensure_ascii=False)
    gen = ModelGenerator([("m", call)])
    assert gen.check_support([kept, added], passages, "ما سبب نزول سورة الكوثر؟") == [True, False]
    assert gen.last_check_reasons[1] == "key term not in the passage: قريش"


def test_the_instructions_say_never_to_write_ids_or_source_numbers_in_the_text():
    s = generate.SYSTEM_PROMPT
    assert "معرّفات المقاطع" in s and "t4:2:255:4947:1" in s
    assert "رقم الحديث" in s and "رقم الآية" in s and "رقم الصفحة" in s and "passage_ids وحدها" in s


def test_the_answer_shown_has_no_inference_tail_even_when_the_checks_accepted_the_sentence():
    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if "queries" in keys:
            return '{"queries": []}'
        return json.dumps({"abstain": False, "claims": [
            {"text": "تحتاج النخلة إلى ماء كثير في الصيف، وهذا يدل على عدل الله.", "passage_ids": ["test-a:1"]}]}, ensure_ascii=False)
    res = Muhawir(CORPUS, ModelGenerator([("m", call)])).ask(QUESTION)
    assert [c["text"] for c in res.claims] == ["تحتاج النخلة إلى ماء كثير في الصيف."]


# --- the understanding step runs twice and adds up its search phrases ---------------------------------

def _understanding(replies):
    replies = iter(replies)

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if '"question"' in keys:
            r = next(replies)
            if isinstance(r, Exception):
                raise r
            return json.dumps({"question": QUESTION, "translate": "", "answer_lang": "", "kind": r[0], "reexplain": False,
                               "recall": "", "queries": r[1]}, ensure_ascii=False)
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        return json.dumps({"abstain": False, "claims": [{"text": "ماء كثير.", "passage_ids": ["test-a:1"]}]})
    return Muhawir(CORPUS, ModelGenerator([("m", call)]))


def test_two_runs_of_the_understanding_step_add_their_search_phrases_and_keep_the_first_ones_reading():
    m = _understanding([("what", ["أ", "ب"]), ("why", ["ب", "ج"])])
    u = Muhawir._understood(m.generator.understand, QUESTION, [])
    assert u["queries"] == ["أ", "ب", "ج"] and u["kind"] == "what"


def test_if_one_run_of_the_understanding_step_fails_the_other_is_used():
    ok = ("what", ["أ"])
    for replies in ([RuntimeError("down"), ok], [ok, RuntimeError("down")]):
        m = _understanding(replies)
        assert Muhawir._understood(m.generator.understand, QUESTION, [])["queries"] == ["أ"]
    m = _understanding([RuntimeError("down"), RuntimeError("down")])
    assert Muhawir._understood(m.generator.understand, QUESTION, []) is None


def test_the_merged_search_phrases_are_capped():
    from muhawir import pipeline as pl
    many = [str(i) for i in range(10)]
    m = _understanding([("what", many), ("what", [str(i) for i in range(10, 20)])])
    assert len(Muhawir._understood(m.generator.understand, QUESTION, [])["queries"]) == pl.MAX_UNDERSTOOD_QUERIES


# --- a cut must leave a finished sentence -----------------------------------------------------------

def test_a_cut_that_would_leave_a_dangling_word_is_refused():
    from muhawir.pipeline import _without_inference
    dangling = Claim("لا تنفع الشفاعة إلا لمن وهذا يعني أن الإذن بيد الله وحده.", ("a",))
    assert _without_inference(dangling) is dangling  # «إلا لمن» cannot end a sentence
    ok = Claim("لا تنفع الشفاعة إلا لمن أذن له الله، وهذا يعني أن الإذن بيد الله وحده.", ("a",))
    assert _without_inference(ok).text == "لا تنفع الشفاعة إلا لمن أذن له الله."


def test_any_present_tense_verb_after_a_demonstrative_begins_an_inference():
    from muhawir.pipeline import _without_inference
    assert _without_inference(Claim("ورد نصف ما ترك الأزواج في الآية، وهذا يحدد نصيب المرأة بنصف نصيب الرجل.", ("a",))).text \
        == "ورد نصف ما ترك الأزواج في الآية."
    keep = Claim("ذكر العلماء أن للذكر مثل حظ الأنثيين، وهذا قول الجمهور في المسألة.", ("a",))
    assert _without_inference(keep) is keep  # «وهذا قول» is a statement about whose view it is, not an inference


# --- everyday examples are switched off -------------------------------------------------------------

def _with_example(text):
    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if "queries" in keys:
            return '{"queries": []}'
        return json.dumps({"abstain": False, "claims": [
            {"text": "تحتاج النخلة إلى ماء كثير في الصيف.", "passage_ids": ["test-a:1"]},
            {"text": text, "passage_ids": ["test-a:1"]}]}, ensure_ascii=False)
    return Muhawir(CORPUS, ModelGenerator([("m", call)]))


@pytest.mark.parametrize("example", ["مثلًا، إذا كان لديك لعبة كثيرة تعطي بعضها لأصدقائك.",
                                     "مثال: إذا كان لديك تمر تعطي صاعًا للفقراء.",
                                     "For example, if you have many toys you share them."])
def test_an_everyday_example_is_dropped_because_examples_are_switched_off(example, monkeypatch):
    monkeypatch.setattr(pipeline, "DEBUG", True)
    res = _with_example(example).ask(QUESTION)
    assert [c["text"] for c in res.claims] == ["تحتاج النخلة إلى ماء كثير في الصيف."]
    assert "examples are switched off" in res.why


def test_the_switch_can_turn_examples_back_on(monkeypatch):
    monkeypatch.setattr(pipeline, "ALLOW_EXAMPLES", True)
    res = _with_example("مثلًا، تحتاج النخلة إلى سقي كما يحتاج العطشان إلى الماء.").ask(QUESTION)
    assert len(res.claims) == 2
