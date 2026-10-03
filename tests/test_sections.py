"""Chapter and section openings of «بداية المجتهد» are offered first; pasted quotations are rejected."""
import json
from pathlib import Path

import pytest

from muhawir.corpus import Passage, load_corpus, parse_corpus
from muhawir.generate import ExtractiveGenerator, ModelGenerator
from muhawir.pipeline import ANSWERED, Muhawir
from muhawir.sections import SectionIndex

B = "bidayat-al-mujtahid"


def _corpus():
    def p(pid, kw, text):
        return {"id": pid, "source_id": B, "location": kw, "kind": "fiqh", "text": text, "keywords": kw}
    return parse_corpus({"synthetic": True, "sources": [{"id": B, "name": "بداية المجتهد", "about": "كتاب", "url": ""}],
        "passages": [
            p("f:1", "كتاب الزكاة؛ الجملة الأولى في معرفة من تجب عليه الزكاة", "كتاب الزكاة ينحصر في خمس جمل."),
            p("f:2", "كتاب الزكاة؛ الجملة الأولى في معرفة من تجب عليه الزكاة", "واختلفوا في مال اليتيم."),
            p("f:3", "كتاب الزكاة؛ الجملة الثانية في معرفة ما تجب فيه الزكاة من الأموال", "الذهب والفضة والإبل والبقر والغنم."),
            p("f:4", "كتاب القسمة؛ الباب الأول في أنواع القسمة", "أنواع القسمة ثلاثة."),
            p("f:5", "كتاب الصلاة؛ الجملة الأولى في معرفة وجوب الصلاة", "الصلاة واجبة."),
        ]})


def test_matching_chapter_opening_and_section_opening_come_first():
    ix = SectionIndex(_corpus())
    assert ix.match(["ما معنى الزكاة وما أنواعها؟", "أنواع الأموال التي تجب فيها الزكاة"]) == ["f:1", "f:3"]


def test_common_words_alone_do_not_pull_in_another_chapter():
    ix = SectionIndex(_corpus())
    assert "f:4" not in ix.match(["ما أنواع الزكاة؟"])  # «أنواع» alone does not bring «أنواع القسمة»
    assert ix.match(["من خلق الله؟"]) == []


# --- the passage that opens an issue comes before the evidence for it ---------------------------------

PRAYER = "كتاب الصلاة؛ الفصل الأول"
WUDU = "كتاب الوضوء؛ الباب الأول"


def _issues():
    """f:255 opens the issue of reading the basmala and gives the views; f:256 and f:257 give the evidence
    (the real passages, shortened). f:299 and f:300 belong to another heading."""
    def p(pid, kw, text):
        return {"id": pid, "source_id": B, "location": kw, "kind": "fiqh", "text": text, "keywords": kw}
    return parse_corpus({"synthetic": True, "sources": [{"id": B, "name": "بداية المجتهد", "about": "كتاب", "url": ""}],
        "passages": [
            p("f:254", PRAYER, "المسألة الثالثة ذهب قوم إلى أن التوجيه في الافتتاح واجب وقال مالك ليس بواجب."),
            p("f:255", PRAYER, "المسألة الرابعة اختلفوا في قراءة بسم الله الرحمن الرحيم في افتتاح القراءة، "
                               "فمنع ذلك مالك في المكتوبة، وقال أبو حنيفة والثوري وأحمد يقرؤها سرا."),
            p("f:256", PRAYER, "ومنها ما رواه مالك من حديث أنس قمت وراء أبي بكر وعمر وعثمان فكلهم كان لا يقرأ بسم الله."),
            p("f:257", PRAYER, "والسبب الثاني هل البسملة آية من أم الكتاب وحدها أو من كل سورة أم ليست آية."),
            p("f:299", WUDU, "المسألة الأولى اختلفوا في حكم النية في الوضوء فقال قوم بوجوبها."),
            p("f:300", WUDU, "واحتج من أوجبها بحديث إنما الأعمال بالنيات وهو حديث مشهور."),
        ]})


def _ids(passages):
    return [p.id for p in passages]


def _gather(monkeypatch, query):
    monkeypatch.setattr(Muhawir, "_neighbours", lambda self, passages: [])  # as when other passages used up the cap
    corpus = _issues()
    m = Muhawir(corpus, ExtractiveGenerator())
    return _ids(m._gather("ما حكم الجهر بالبسملة؟", "", [query])[0])


def test_an_evidence_passage_brings_the_passage_that_opens_its_issue_before_it(monkeypatch):
    # the basmala question: only f:256 (the evidence) matches the words; f:255 (the views) must come with it, first
    ids = _gather(monkeypatch, "قمت وراء أبي بكر وعمر وعثمان فكلهم يقرأ بسم الله")
    assert "f:256" in ids and "f:255" in ids and ids.index("f:255") < ids.index("f:256")


def test_the_opener_is_found_two_passages_back_too(monkeypatch):
    ids = _gather(monkeypatch, "هل البسملة آية من أم الكتاب وحدها أو من كل سورة")  # f:257; f:256 is not an opener
    assert "f:257" in ids and "f:255" in ids and ids.index("f:255") < ids.index("f:257")


def test_a_passage_that_opens_its_own_issue_or_sits_under_another_heading_adds_nothing(monkeypatch):
    assert _gather(monkeypatch, "اختلفوا في قراءة بسم الله الرحمن الرحيم في افتتاح القراءة") == ["f:255"]
    ids = _gather(monkeypatch, "واحتج من أوجبها بحديث إنما الأعمال بالنيات")  # f:300 -> its own heading's f:299
    assert ids == ["f:299", "f:300"] and "f:255" not in ids


def test_an_opener_already_in_the_list_moves_before_the_evidence_and_nothing_is_repeated():
    m = Muhawir(_issues(), ExtractiveGenerator())
    by = {p.id: p for p in _issues().passages}
    assert _ids(m._with_issue_openers([by["f:256"], by["f:257"], by["f:255"]])) == ["f:255", "f:256", "f:257"]
    assert _ids(m._with_issue_openers([by["f:254"], by["f:255"]])) == ["f:254", "f:255"]  # openers: nothing added
    assert _ids(m._with_issue_openers([])) == []


SYN = load_corpus(Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json")


@pytest.mark.real_check
def test_pasted_quotation_is_rejected_and_rewritten_in_own_words():
    prompts = []
    answers = iter([
        {"plan": "", "abstain": False, "as_list": False, "views": [], "claims": [
            {"text": "حيث ذُكر: «تحتاج النخلة إلى ماء كثير في الصيف»", "passage_ids": ["test-a:1"]}]},
        {"plan": "", "abstain": False, "as_list": False, "views": [], "claims": [
            {"text": "النخلة تشرب ماء كثيرا في الصيف.", "passage_ids": ["test-a:1"]}]}])

    def call(system, user, schema=None):
        keys = json.dumps(schema or {})
        if "problem" in keys:
            return json.dumps({"supported": [True] * user.count("الجملة ")})
        if "verdict" in keys:
            return '{"verdict": "yes"}'
        if '"question"' in keys or "queries" in keys:
            return '{"queries": []}'
        prompts.append(user)
        return json.dumps(next(answers), ensure_ascii=False)

    res = Muhawir(SYN, ModelGenerator([("m", call)])).ask("ماذا تحتاج النخلة في الصيف؟")
    assert res.status == ANSWERED and [c["text"] for c in res.claims] == ["النخلة تشرب ماء كثيرا في الصيف."]
    assert len(prompts) == 2 and "مراجعة لجواب سابق" in prompts[1]
