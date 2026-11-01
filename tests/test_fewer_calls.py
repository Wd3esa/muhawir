"""Fewer model calls per answer, with every check kept (November 2026). Fake model calls and synthetic data only:
the counts are of requests the pipeline makes, not of any real model's verdicts."""
import asyncio
import json
import queue
import threading

import pytest

from muhawir import generate
from muhawir.corpus import parse_corpus
from muhawir.generate import ModelGenerator
from muhawir.pipeline import Muhawir

pytestmark = pytest.mark.real_check  # the second reading runs for real against the fake model

DATA = {"synthetic": True,
        "sources": [{"id": "q", "name": "نص تجريبي", "about": "مصطنع"},
                    {"id": "b", "name": "كتاب تجريبي", "about": "مصطنع"}],
        "passages": [
            {"id": "q:1:1", "source_id": "q", "location": "سورة تجريبية، الآية 1", "kind": "quran",
             "text": "والنخل باسقات لها طلع نضيد"},
            {"id": "b:1", "source_id": "b", "location": "باب 1", "kind": "other",
             "text": "النخلة شجرة تحتاج إلى ماء كثير في الصيف، وتثمر التمر في آخره، ويصبر جذعها على الحر."},
            {"id": "b:2", "source_id": "b", "location": "باب 2", "kind": "other",
             "text": "يسقي الفلاح النخل في الصباح الباكر، ويقطف الرطب إذا نضج."}]}
CORPUS = parse_corpus(DATA)
QUESTION = "ماذا تحتاج النخلة في الصيف؟"
SIX = [{"text": "تحتاج النخلة إلى ماء كثير في الصيف.", "passage_ids": ["b:1"]},
       {"text": "وتثمر التمر في آخر الصيف.", "passage_ids": ["b:1"]},
       {"text": "ويصبر جذعها على الحر.", "passage_ids": ["b:1"]},
       {"text": "ويسقيها الفلاح في الصباح الباكر.", "passage_ids": ["b:2"]},
       {"text": "ويقطف الرطب إذا نضج.", "passage_ids": ["b:2"]},
       {"text": "والنخل يحب الماء البارد جدًا.", "passage_ids": ["b:2"]}]


def _model(draft, reject=(), verdicts=("yes",)):
    """A fake model: one draft, support readings that reject the sentences containing any text in `reject`, and
    relevance verdicts in order. Every request is counted by kind."""
    seen = {"draft": 0, "support": 0, "support_sentences": [], "relevance": 0, "other": 0}
    verdicts = iter(list(verdicts) + [verdicts[-1]] * 5)
    lock = threading.Lock()

    def call(system, user, schema=None):
        props = (schema or {}).get("properties", {})
        with lock:
            if "claims" in props:
                seen["draft"] += 1
                return json.dumps({"abstain": False, "claims": draft, "follow_up": "هل تريد أن تعرف متى يُقطف الرطب؟"},
                                  ensure_ascii=False)
            if "problem" in props:
                seen["support"] += 1
                sentences = [line.split("<<<", 1)[1].split(">>>", 1)[0] for line in user.splitlines()
                             if line.startswith("الجملة ")]
                seen["support_sentences"].append(sentences)
                return json.dumps({"problem": ["addition" if any(x in s for x in reject) else "none"
                                               for s in sentences]}, ensure_ascii=False)
            if "verdict" in props:
                seen["relevance"] += 1
                return json.dumps({"verdict": next(verdicts)})
            seen["other"] += 1
            return json.dumps({"absent": []} if "absent" in props else {"queries": []})

    return Muhawir(CORPUS, ModelGenerator([("fake", call)])), seen


def _write(engine, style="youth"):
    return engine._write(QUESTION, list(CORPUS.passages), style, "ar", False, {})


def test_losing_one_sentence_of_six_costs_no_rewrite_when_the_answer_still_covers_the_question():
    engine, seen = _model(SIX, reject=("البارد",), verdicts=("yes",))
    result = _write(engine)
    assert seen["draft"] == 1 and len(result.kept) == 5 and not result.partial
    # six first readings, one second reading of the rejected sentence; no second draft, no second round
    assert seen["support"] == 7


def test_losing_two_sentences_of_six_now_waits_for_the_relevance_review():
    engine, seen = _model(SIX, reject=("البارد", "الباكر"), verdicts=("yes",))
    _write(engine)
    assert seen["draft"] == 1  # it was a full rewrite at a third or more


def test_an_answer_that_no_longer_covers_the_question_is_still_rewritten():
    # the early judge reads the draft, then the judge reads the checked answer: both find it incomplete
    engine, seen = _model(SIX, reject=("البارد", "الباكر"), verdicts=("partly", "partly", "yes"))
    _write(engine)
    assert seen["draft"] == 2


def test_losing_half_the_answer_or_its_first_sentence_is_still_rewritten():
    engine, seen = _model(SIX, reject=("البارد", "الباكر", "الرطب"))
    _write(engine)
    assert seen["draft"] == 2
    engine, seen = _model(SIX, reject=("كثير",))
    _write(engine)
    assert seen["draft"] == 2


def test_check_groups_are_off_by_default_and_read_same_passage_sentences_together(monkeypatch):
    engine, seen = _model(SIX)
    _write(engine)
    assert all(len(group) == 1 for group in seen["support_sentences"])
    monkeypatch.setattr(generate, "CHECK_GROUPS", True)
    engine, seen = _model(SIX, reject=("البارد",))
    result = _write(engine)
    assert sorted(len(g) for g in seen["support_sentences"])[-2:] == [3, 3]
    assert len(result.kept) == 5


def test_the_follow_up_is_dropped_after_a_partial_answer_and_rarely_asked_for():
    assert "واترك follow_up فارغًا في أغلب الأجوبة" in generate.SYSTEM_PROMPT
    assert '"follow_up": "هل تريد أن تعرف معنى سورة الإخلاص؟"' not in generate.SYSTEM_PROMPT
    engine, _ = _model(SIX, reject=("البارد", "الباكر", "الرطب"), verdicts=("partly",))
    res = engine.ask(QUESTION)
    assert res.follow_up == ""


def test_a_question_that_waits_for_a_slot_is_told_so_at_once(monkeypatch):
    from muhawir import server

    slot = threading.BoundedSemaphore(1)
    slot.acquire()
    monkeypatch.setattr(server, "_ask_slots", slot)
    monkeypatch.setattr(server, "_ask_waiters", threading.BoundedSemaphore(1))
    monkeypatch.setattr(server, "ASK_QUEUE_TIMEOUT", 2)
    monkeypatch.setattr(server, "_allow_question", lambda address: True)
    monkeypatch.setattr(server, "_run_ask", lambda body: {"status": "answered"})
    response = server.ask_stream(server.Ask(question="مرحبا"))
    lines: queue.Queue = queue.Queue()

    async def read():
        async for chunk in response.body_iterator:
            lines.put(chunk)

    reader = threading.Thread(target=lambda: asyncio.run(read()))
    reader.start()
    assert json.loads(lines.get(timeout=2)) == {"stage": "queued"}
    slot.release()
    reader.join(timeout=5)
    rest = []
    while not lines.empty():
        line = lines.get()
        if line.strip():
            rest.append(json.loads(line))
    assert {"stage": "understanding"} in rest and rest[-1]["reply"]["status"] == "answered"


def test_a_waiting_question_gives_up_with_429_when_no_slot_opens(monkeypatch):
    from muhawir import server

    slot = threading.BoundedSemaphore(1)
    slot.acquire()
    monkeypatch.setattr(server, "_ask_slots", slot)
    waiters = threading.BoundedSemaphore(1)
    monkeypatch.setattr(server, "_ask_waiters", waiters)
    monkeypatch.setattr(server, "ASK_QUEUE_TIMEOUT", .05)
    monkeypatch.setattr(server, "_allow_question", lambda address: True)
    response = server.ask_stream(server.Ask(question="مرحبا"))

    async def read():
        return [chunk async for chunk in response.body_iterator]
    lines = [json.loads(x) for x in asyncio.run(read()) if x.strip()]
    assert lines == [{"stage": "queued"}, {"error": 429}]
    assert waiters.acquire(blocking=False)  # the place in line was given back
    waiters.release()
    slot.release()
