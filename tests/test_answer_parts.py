"""Each part of a question has its own places among the offered passages, and a part the answer did not cover is
searched again on its own (issue #100). Synthetic corpus, fake model."""
import json

import pytest

from muhawir.corpus import parse_corpus
from muhawir.generate import ModelGenerator
from muhawir.pipeline import MODEL_CANDIDATES, Muhawir

pytestmark = pytest.mark.real_check

SOURCES = [{"id": "a", "name": "كتاب أ", "about": "تجريبي"}, {"id": "b", "name": "كتاب ب", "about": "تجريبي"}]
# the first part («الزكاة») has many passages that match any phrase about it; the second («النصاب») has two
FIRST = [{"id": f"a:{i}", "source_id": "a", "location": f"باب {i}", "kind": "fiqh",
          "text": f"تجب الزكاة على المسلم الحر إذا ملك المال وحال عليه الحول، وهذا بيان الزكاة رقم {i} في الشروط."} for i in range(40)]
SECOND = [{"id": "b:1", "source_id": "b", "location": "باب النصاب", "kind": "fiqh",
           "text": "نصاب الفضة مائتا درهم ونصاب الذهب عشرون مثقالا والواجب ربع العشر."},
          {"id": "b:2", "source_id": "b", "location": "باب المقدار", "kind": "fiqh",
           "text": "مقدار النصاب الذي تجب فيه الصدقة هو ما بلغ هذا القدر من الفضة والذهب."}]
CORPUS = parse_corpus({"synthetic": True, "sources": SOURCES, "passages": FIRST + SECOND})
PARTS = ["شروط وجوب الزكاة", "مقدار النصاب"]


def _engine(call=None):
    return Muhawir(CORPUS, ModelGenerator([("fake", call or (lambda *a, **k: "{}"))]))


def test_a_part_whose_evidence_ranks_low_for_the_shared_phrases_still_gets_its_places():
    engine = _engine()
    phrases = ["شروط الزكاة", "بيان الزكاة", "الزكاة على المسلم", "حال عليه الحول"]  # all about the first part
    alone, _ = engine._gather("شروط الزكاة ومقدار النصاب", "", phrases, domain="fiqh")
    parted, _ = engine._gather("شروط الزكاة ومقدار النصاب", "", phrases, domain="fiqh", parts=PARTS)
    assert not any(p.id.startswith("b:") for p in alone)  # the failure: twenty places all taken by the first part
    assert {"b:1", "b:2"} <= {p.id for p in parted}


def test_one_part_changes_nothing():
    engine = _engine()
    phrases = ["شروط الزكاة", "بيان الزكاة"]
    with_one = engine._gather("سؤال", "", phrases, domain="fiqh", parts=["شروط الزكاة"])[0]
    assert [p.id for p in with_one] == [p.id for p in engine._gather("سؤال", "", phrases, domain="fiqh")[0]]


def test_more_parts_offer_a_few_more_passages_and_never_unbounded():
    engine = _engine()
    phrases = ["شروط الزكاة", "بيان الزكاة", "الزكاة على المسلم", "حال عليه الحول", "تجب الزكاة"]
    many = [f"{w} الزكاة" for w in ("شروط", "بيان", "حكم", "وجوب", "مقدار", "أحكام")]
    one, _ = engine._gather("سؤال", "", phrases, domain="fiqh")
    six, _ = engine._gather("سؤال", "", phrases, domain="fiqh", parts=many)
    assert len(one) <= MODEL_CANDIDATES + 8 and len(six) <= MODEL_CANDIDATES + 12 + 8  # neighbours and openers aside


def test_an_uncovered_part_is_searched_again_and_the_rewrite_sees_what_was_found():
    seen = {"drafts": [], "reviews": 0}
    first = {"abstain": False, "claims": [
        {"text": "تجب الزكاة على المسلم الحر إذا ملك المال وحال عليه الحول.", "passage_ids": ["a:0"]}]}
    second = {"abstain": False, "claims": [*first["claims"],
        {"text": "نصاب الفضة مائتا درهم والواجب ربع العشر.", "passage_ids": ["b:1"]}]}

    def call(system, user, schema=None):
        props = (schema or {}).get("properties", {})
        if "claims" in props:
            seen["drafts"].append(user)
            return json.dumps(first if len(seen["drafts"]) == 1 else second, ensure_ascii=False)
        if "problem" in props:
            return json.dumps({"problem": ["none"] * user.count("الجملة ")})
        if "verdict" in props:
            seen["reviews"] += 1
            return json.dumps({"verdict": "partly", "missing_parts": [1], "reason": "النصاب"} if seen["reviews"] == 1
                              else {"verdict": "yes"})
        return json.dumps({"queries": []})

    engine = _engine(call)
    passages = [p for p in CORPUS.passages if p.id.startswith("a:")][:10]  # the first search offered only the first part
    request = {"answer_request": {"form": "explanation", "requirements": PARTS, "domain": "fiqh"}}
    written = engine._write("شروط وجوب الزكاة ومقدار النصاب", passages, "youth", "ar", False, request)
    assert len(seen["drafts"]) == 2 and "[b:1]" not in seen["drafts"][0] and "[b:1]" in seen["drafts"][1]
    assert "b:1" in written.allowed and any(c.passage_ids == ("b:1",) for c in written.kept)
