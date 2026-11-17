"""Repair, don't delete (issue #100): a sentence the support check rejects for going past its passage is written
again from that passage alone, checked like any other sentence, and put back where it stood. Fake model, synthetic data."""
import json
import threading

import pytest

from muhawir import telemetry
from muhawir.generate import ModelGenerator
from muhawir.pipeline import Muhawir
from test_fewer_calls import CORPUS, SIX

pytestmark = pytest.mark.real_check

THIRD = "ويصبر جذعها على الحر."


def _engine(draft, reject=("البارد",), repairs=None, repaired_ok=True):
    """`repairs` maps the start of a rejected sentence to (text, passage ids) written in its place."""
    seen = {"draft": 0, "repair": 0, "support": 0, "support_texts": []}
    lock = threading.Lock()

    def call(system, user, schema=None):
        props = (schema or {}).get("properties", {})
        with lock:
            if "claims" in props:
                seen["draft"] += 1
                return json.dumps({"abstain": False, "claims": draft, "follow_up": ""}, ensure_ascii=False)
            if "repairs" in props:
                seen["repair"] += 1
                rows = []
                for i, line in enumerate([l for l in user.splitlines() if l.startswith("الجملة ")], 1):
                    sentence = line.split("<<<", 1)[1].split(">>>", 1)[0]
                    for start, (text, ids) in (repairs or {}).items():
                        if sentence.startswith(start):
                            rows.append({"index": i, "text": text, "passage_ids": list(ids)})
                return json.dumps({"repairs": rows}, ensure_ascii=False)
            if "problem" in props:
                seen["support"] += 1
                sentences = [l.split("<<<", 1)[1].split(">>>", 1)[0] for l in user.splitlines() if l.startswith("الجملة ")]
                seen["support_texts"] += sentences
                return json.dumps({"problem": ["addition" if any(x in s for x in reject) else "none" for s in sentences]},
                                  ensure_ascii=False)
            if "verdict" in props:
                return json.dumps({"verdict": "yes"})
            return json.dumps({"queries": []})
    return Muhawir(CORPUS, ModelGenerator([("fake", call)])), seen


def _write(engine):
    return engine._write("ماذا تحتاج النخلة في الصيف؟", list(CORPUS.passages), "youth", "ar", False, {})


def _loss(engine):
    trace = telemetry.start("ask")
    with telemetry.bind(trace):
        written = _write(engine)
        trace.finish("answered", 200)
    return written, {e["stage"]: e["counts"] for e in trace.events if e["kind"] == "loss"}


def test_a_rejected_sentence_is_written_again_and_put_back_in_place(monkeypatch):
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    draft = list(SIX[:5])
    draft[2] = {"text": "ويصبر جذعها على الحر دائمًا وبلا ماء البارد.", "passage_ids": ["b:1"]}
    engine, seen = _engine(draft, repairs={"ويصبر": ("وجذعها يصبر على الحر.", ["b:1"])})
    written, loss = _loss(engine)
    assert [c.text for c in written.kept] == [SIX[0]["text"], SIX[1]["text"], "وجذعها يصبر على الحر.", SIX[3]["text"], SIX[4]["text"]]
    assert written.rejected == [] and loss["repair"] == {"asked": 1, "written": 1, "repaired": 1}
    assert seen["repair"] == 1 and seen["draft"] == 1  # no full rewrite was needed


def test_a_repair_that_is_rejected_again_stays_out(monkeypatch):
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    engine, seen = _engine(SIX, repairs={"والنخل يحب": ("والنخل يحب الماء البارد جدًا في الصيف.", ["b:2"])})
    written, loss = _loss(engine)
    assert [c.text for c in written.kept] == [c["text"] for c in SIX[:5]]
    assert len(written.rejected) == 1 and loss["repair"]["repaired"] == 0 and loss["repair"]["written"] == 1


def test_nothing_is_asked_when_nothing_is_rejected():
    engine, seen = _engine(SIX[:5])
    written = _write(engine)
    assert seen["repair"] == 0 and len(written.kept) == 5


def test_a_sentence_without_a_passage_to_be_written_from_is_not_repaired():
    draft = [*SIX[:2], {"text": "ويحب النخل الماء كثيرًا.", "passage_ids": []}]
    engine, seen = _engine(draft, reject=("شيء لا يوجد",), repairs={"ويحب": ("كلام", ["b:1"])})
    written = _write(engine)
    assert seen["repair"] == 0 and len(written.kept) == 2


def test_a_writer_that_cannot_repair_changes_nothing():
    engine, seen = _engine(SIX, repairs={})
    written = _write(engine)
    assert seen["repair"] == 1 and len(written.kept) == 5 and len(written.rejected) == 1


def test_most_of_the_answer_rejected_is_repaired_before_any_full_rewrite():
    draft = [{"text": f"جملة {i} عن النخل البارد.", "passage_ids": ["b:1"]} for i in range(3)] + [SIX[0]]
    repairs = {f"جملة {i}": (["تحتاج النخلة إلى ماء كثير في الصيف.", "وتثمر التمر في آخره.", "ويصبر جذعها على الحر."][i], ["b:1"])
               for i in range(3)}
    engine, seen = _engine(draft, repairs=repairs)
    written = _write(engine)
    assert seen["draft"] == 1 and seen["repair"] == 1 and len(written.kept) == 4  # before: a second full draft


def test_the_generator_alone_pairs_replies_with_sentences():
    from muhawir.verify import Claim
    calls = []
    def call(system, user, schema=None):
        calls.append(user)
        return json.dumps({"repairs": [{"index": 2, "text": "بديل", "passage_ids": ["b:2"]},
                                       {"index": 9, "text": "خارج النطاق", "passage_ids": ["b:2"]},
                                       {"index": 1, "text": "  ", "passage_ids": ["b:1"]}]})
    gen = ModelGenerator([("fake", call)])
    items = [(Claim("أ", ("b:1",), school="", section="قسم", label="تسمية"), "سبب"), (Claim("ب", ("b:2",)), "سبب")]
    out = gen.repair("سؤال", items, {p.id: p for p in CORPUS.passages})
    assert out[0] is None and out[1].text == "بديل" and out[1].passage_ids == ("b:2",)
    assert "الجملة 1" in calls[0] and "الجملة 2" in calls[0]
