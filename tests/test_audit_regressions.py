"""Regression checks for the deployment audit: fake sources and calls only."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from muhawir import attribution, pipeline, server, vectors
from muhawir.corpus import load_corpus, parse_corpus
from muhawir.generate import ExtractiveGenerator, ModelGenerator, parse_draft
from muhawir.verify import Claim, school_is_named, verify

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic_corpus.json"


@pytest.mark.parametrize("field", ["claims", "views"])
@pytest.mark.parametrize("value", [17, "wrong", {}])
def test_malformed_claim_collections_are_unusable(field, value):
    assert parse_draft(json.dumps({field: value})) == []


@pytest.mark.parametrize("step", ["expand", "understand", "retry_queries"])
@pytest.mark.parametrize("value", [None, 7, "bad"])
def test_bad_queries_fall_back_to_next_model(step, value):
    calls = []
    def bad(*args):
        calls.append("bad")
        return json.dumps({"queries": value})
    def good(*args):
        calls.append("good")
        return '{"queries":["النخلة"],"question":"النخلة"}'
    gen = ModelGenerator([("bad", bad), ("good", good)])
    result = getattr(gen, step)("النخلة", []) if step != "expand" else gen.expand("النخلة")
    assert (result["queries"] if step == "understand" else result) == ["النخلة"]
    assert calls == ["bad", "good"]


def test_non_string_kind_does_not_crash_understanding():
    gen = ModelGenerator([("fake", lambda *a: '{"queries":[],"kind":[]}')])
    assert gen.understand("النخلة", [])["kind"] == ""


def test_school_alias_does_not_hide_an_unmentioned_second_scholar():
    assert school_is_named("المالكية", ["قال مالك."])
    assert not school_is_named("مالك والشافعي", ["قال مالك."])
    assert school_is_named("مالك والشافعي", ["قال مالك والشافعي."])


def test_known_source_speaker_does_not_authorize_added_name():
    passage = SimpleNamespace(source_id="fatawa-ibn-baz")
    corpus = SimpleNamespace(passage=lambda pid: passage)
    assert attribution.names_speaker("الشيخ ابن باز", ["p"], corpus)
    assert not attribution.names_speaker("الشيخ ابن باز وابن عثيمين", ["p"], corpus)


@pytest.mark.parametrize("field", ["section", "label"])
def test_unfounded_quote_in_heading_is_rejected(field):
    corpus = load_corpus(SYNTHETIC)
    claim = Claim("تحتاج النخلة إلى الماء.", ("test-a:1",), **{field: "﴿نص مخترع﴾"})
    kept, rejected = verify([claim], corpus, {"test-a:1"})
    assert not kept and "quotation" in rejected[0].reason


@pytest.mark.real_check
def test_second_reading_sees_heading_and_label():
    seen = []
    def call(system, user, schema=None):
        seen.append(user)
        return '{"supported":[true],"problem":["none"]}'
    corpus = load_corpus(SYNTHETIC)
    claim = Claim("تحتاج النخلة الماء.", ("test-a:1",), section="عنوان الفحص", label="وسم الفحص")
    gen = ModelGenerator([("fake", call)])
    assert gen.check_support([claim], {"test-a:1": corpus.passage("test-a:1")}) == [True]
    assert "عنوان الفحص" in seen[0] and "وسم الفحص" in seen[0]


@pytest.mark.parametrize("field,value", [("style", "anything"), ("lang", "anything")])
def test_invalid_cache_variant_is_rejected(field, value):
    with pytest.raises(ValidationError):
        server.Ask(question="النخلة", **{field: value})


def test_missing_production_database_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("MUHAWIR_DB", str(tmp_path / "missing.db"))
    monkeypatch.delenv("MUHAWIR_CORPUS", raising=False)
    monkeypatch.delenv("MUHAWIR_ALLOW_SYNTHETIC", raising=False)
    monkeypatch.setattr(server, "get_generator", lambda: SimpleNamespace(name="fake-model"))
    with pytest.raises(RuntimeError, match="source database is missing"):
        server.build()


def test_disk_cache_reads_obey_memory_capacity(monkeypatch):
    corpus = load_corpus(SYNTHETIC)
    response = pipeline.Muhawir(corpus, ExtractiveGenerator()).ask("ماذا تحتاج النخلة في الصيف؟")
    engine = pipeline.Muhawir(corpus, ExtractiveGenerator(), store=SimpleNamespace(get=lambda key: response))
    monkeypatch.setattr(pipeline, "CACHE_SIZE", 2)
    for n in range(10):
        assert engine._kept((str(n), "youth", "ar")) is not None
    assert len(engine._cache) == 2
    assert next(iter(engine._cache))[0] == "8"


def test_lazy_embedder_is_loaded_once_during_concurrent_queries(monkeypatch):
    import time
    built = []
    class FakeClient:
        def embed_query(self, model, question):
            return [1.0]
    def load(name):
        built.append(name)
        time.sleep(0.01)
        return FakeClient()
    monkeypatch.setattr(vectors.LocalEmbedder, "from_name", load)
    embedder = vectors._LazyQueryEmbedder("local:fake")
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(embedder, range(8))) == [[1.0]] * 8
    assert built == ["local:fake"]


def test_busy_and_rate_limits_do_not_call_engine(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "engine", SimpleNamespace(ask=lambda *a: calls.append(a)))
    monkeypatch.setattr(server, "_ask_calls", __import__('collections').deque())
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(server, "_ask_slots", slots)
    with pytest.raises(HTTPException) as error:
        server.ask(server.Ask(question="النخلة"))
    assert error.value.status_code == 429 and not calls
    slots.release()
    monkeypatch.setattr(server, "ASK_PER_MINUTE", 1)
    server._ask_calls.append(__import__('time').monotonic())
    with pytest.raises(HTTPException) as error:
        server.ask(server.Ask(question="النخلة"))
    assert error.value.status_code == 429 and not calls


def test_answer_slot_is_released_after_failure(monkeypatch):
    def fail(*a):
        raise RuntimeError("fake failure")
    monkeypatch.setattr(server, "engine", SimpleNamespace(ask=fail))
    monkeypatch.setattr(server, "_ask_calls", __import__('collections').deque())
    slot = threading.BoundedSemaphore(1)
    monkeypatch.setattr(server, "_ask_slots", slot)
    with pytest.raises(RuntimeError):
        server.ask(server.Ask(question="النخلة"))
    assert slot.acquire(blocking=False)
    slot.release()


@pytest.mark.parametrize("label", ["مالك والشافعي", "مالك وأبو حنيفة"])
def test_composite_attribution_is_rejected_on_full_verifier_path(label):
    corpus = parse_corpus({"synthetic": True, "sources": [{"id": "s", "name": "مصدر تجريبي", "about": "مصطنع للاختبار"}],
        "passages": [{"id": "p", "source_id": "s", "location": "اختبار", "text": "قال مالك إن النخلة تحتاج إلى الماء."}]})
    kept, rejected = verify([Claim("تحتاج النخلة إلى الماء.", ("p",), school=label)], corpus, {"p"})
    assert not kept and "not named" in rejected[0].reason


def test_rejecting_older_answer_removes_its_persisted_history():
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the browser history regression")
    page = (SYNTHETIC.parents[1] / "muhawir/static/index.html").read_text()
    helper = "function forgetRejectedAnswer" + page.split("function forgetRejectedAnswer", 1)[1].split("function syncRegenerateButtons", 1)[0]
    scenario = r"""
const assert = require('node:assert/strict');
const res = {understood:'wrong question', message:'wrong answer', claims:[]};
const wrong = [{role:'user',text:'wrong question'}, {role:'assistant',text:'wrong answer'}];
const right = [{role:'user',text:'later question'}, {role:'assistant',text:'later answer'}];
const entry = {res};
const nextEntry = {history:[...wrong,...right]};
const conversation = {turns:[...wrong,...right],log:[entry,nextEntry]};
const context = {conversation,entry,question:'original'};
forgetRejectedAnswer(context,res);
const restored = JSON.parse(JSON.stringify(conversation));
assert.deepEqual(restored.turns,right);
assert.equal(restored.log.length,1);
assert.deepEqual(restored.log[0].history,right);
assert.equal(context.entry,null);
"""
    subprocess.run([node, "-e", helper + scenario], check=True, capture_output=True, text=True)
