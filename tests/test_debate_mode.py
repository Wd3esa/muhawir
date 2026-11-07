"""Debate mode (issue #98): an objection, or pushback on the previous answer, is answered like a scholar in a calm debate.

The structure lives in the prompts (no training): the point of dispute first, then the evidence with how it proves the
point as the passage says it, then the counter-argument only when a passage makes it. Nothing loosens the support check."""
import json

from muhawir import generate
from muhawir.corpus import Passage
from muhawir.generate import KIND_GUIDE, SYSTEM_PROMPT, build_user_prompt
from muhawir.verify import Claim, verify


def _prompt(kind):
    p = Passage("q:1", "s", "سورة، الآية 1", "نص", "quran")
    return build_user_prompt("سؤال", [p], "youth", "ar", False, kind=kind)


def test_an_objection_gets_the_debate_structure():
    text = _prompt("objection")
    for part in ("تحرّر موضع الإشكال", "وجه دلالته", "إلزامًا", "النقطة الجديدة وحدها"):
        assert part in text


def test_other_kinds_do_not_get_it():
    assert "موضع الإشكال" not in _prompt("what") and "موضع الإشكال" not in _prompt("")


def test_the_limits_of_debate_are_stated():
    guide = KIND_GUIDE["objection"]
    assert "لا تستخف بالسؤال" in guide and "لا تكفّر أحدًا بعينه" in guide and "لا تحكم على حال السائل" in guide


def test_reasoning_and_counter_argument_must_come_from_the_passages():
    assert "كما ذكره المقطع، لا من عندك" in KIND_GUIDE["objection"]
    assert "وجه الدلالة والإلزام يُنقلان من المقاطع" in SYSTEM_PROMPT


def test_the_worked_debate_example_is_valid_json_with_example_ids_only():
    start = SYSTEM_PROMPT.index("المثال الثالث: اعتراض فيه إلزام")
    body = SYSTEM_PROMPT[SYSTEM_PROMPT.index('{"plan"', start):SYSTEM_PROMPT.index("\n\nأعد JSON فقط", start)]
    data = json.loads(body)
    assert not data["abstain"] and len(data["claims"]) == 4
    assert {i for c in data["claims"] for i in c["passage_ids"]} <= {"x7", "x8", "x9"}
    assert [c["label"] for c in data["claims"]][1:3] == ["من السنة", "من القرآن"]


def test_pushback_on_the_previous_answer_is_read_as_an_objection():
    assert "ردّ المستخدم على الجواب السابق برفضه أو التشكيك فيه" in generate.UNDERSTAND_PROMPT


def test_example_ids_are_still_refused_as_evidence():
    from test_chat import CORPUS
    kept, rejected = verify([Claim("نص.", ("x9",))], CORPUS, {"test-a:1"})
    assert not kept and rejected
