"""Actual checker plumbing, synthetic bodies, fake providers. No semantic accuracy claim."""
import json

import pytest

from muhawir import generate as g
from muhawir.corpus import Passage
from muhawir.verify import Claim

pytestmark = pytest.mark.real_check


@pytest.mark.parametrize("source_id,name", [
    ("fatawa-ibn-baz", "الشيخ ابن باز"),
    ("fatawa-ibn-uthaymeen", "الشيخ ابن عثيمين"),
    ("fatawa-arkan-islam", "الشيخ ابن عثيمين"),
])
def test_writer_and_actual_support_reader_receive_identical_catalog_captions(source_id, name):
    passage = Passage("synthetic:1", source_id, "موضع اصطناعي", "متن اصطناعي للاختبار فقط.", "fatwa")
    writing = g.build_user_prompt("سؤال تجريبي؟", [passage], "youth", "ar", False)
    seen = []
    def call(system, user, schema):
        assert system == g.CHECK_PROMPT
        seen.append(user)
        return json.dumps({"problem": ["none"]})
    result = g.ModelGenerator([("fake-source-metadata", call)])._check_once(
        [Claim("جملة اختبار.", (passage.id,))], {passage.id: passage}, "سؤال تجريبي؟")
    assert result == ([True], [""])
    block = g._passage_block(passage)
    assert block in writing and block in seen[0]
    assert name in block and name not in passage.text


def test_grading_and_location_are_available_but_do_not_become_verbatim_witnesses():
    passage = Passage("synthetic:1", "fatawa-ibn-baz", "موضع اصطناعي", "متن اصطناعي.", "fatwa", grade="ضعيف")
    claim = Claim("جملة اختبار.", (passage.id,))
    block = g._passage_block(passage)
    assert "موضع اصطناعي" in block and "حكم المحدث: ضعيف" in block
    assert not g._has_evidence(claim, "الشيخ ابن باز", {passage.id: passage})
    assert not g._has_evidence(claim, "ضعيف", {passage.id: passage})


def test_an_unknown_source_is_not_given_a_guessed_author():
    passage = Passage("synthetic:1", "unknown-source", "موضع اصطناعي", "متن اصطناعي.", "other")
    assert g._passage_block(passage) == "[synthetic:1] (نص — موضع اصطناعي)\nمتن اصطناعي."


def test_only_the_claims_own_sources_are_sent_to_the_support_reader():
    baz = Passage("synthetic:1", "fatawa-ibn-baz", "أ", "متن أول.", "fatwa")
    uthaymeen = Passage("synthetic:2", "fatawa-ibn-uthaymeen", "ب", "متن ثان.", "fatwa")
    def call(system, user, schema):
        assert "الشيخ ابن باز" in user and "الشيخ ابن عثيمين" not in user
        assert "متن ثان" not in user
        return '{"problem":["none"]}'
    out = g.ModelGenerator([("fake-source-isolation", call)])._check_once(
        [Claim("جملة اختبار.", (baz.id,))], {baz.id: baz, uthaymeen.id: uthaymeen})
    assert out == ([True], [""])
