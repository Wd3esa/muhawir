"""Real checker orchestration with synthetic passages and fake providers; no paid calls."""
import json
import threading

import pytest

from muhawir import generate, telemetry
from muhawir.corpus import Passage
from muhawir.generate import ModelGenerator
from muhawir.verify import Claim

pytestmark = pytest.mark.real_check
PASSAGE = Passage("fake:1", "fake", "synthetic", "A box can have different colors.", "other")
EXAMPLE = Claim("For example, a box might be blue.", ("fake:1",))
ORDINARY = Claim("A box can have different colors.", ("fake:1",))


@pytest.mark.parametrize("read_each", [True, False])
def test_religious_example_is_not_rescued_by_a_later_permissive_reading(monkeypatch, read_each):
    monkeypatch.setattr(generate, "READ_EACH_SENTENCE", read_each)
    counts = {"example": 0, "support": 0}
    guard = threading.Lock()
    def call(system, user, schema):
        with guard:
            if system == generate.EXAMPLE_PROMPT:
                counts["example"] += 1
                # A second answer would accept it. The first explicit finding must stand.
                return json.dumps({"religious": counts["example"] == 1})
            assert system == generate.CHECK_PROMPT
            counts["support"] += 1
            return '{"problem": ["none"]}'
    gen = ModelGenerator([("fake-example", call)])
    assert gen.check_support([ORDINARY, EXAMPLE], {PASSAGE.id: PASSAGE}) == [True, False]
    assert gen.last_check_reasons == ["", generate.RELIGIOUS_EXAMPLE]
    assert counts == {"example": 1, "support": 1}


def test_nonreligious_example_still_requires_support_and_topic_check():
    seen = []
    def call(system, user, schema):
        seen.append(system)
        if system == generate.EXAMPLE_PROMPT:
            return '{"religious": false}'
        assert system == generate.CHECK_PROMPT
        return '{"problem": ["none"], "on_topic": [false]}'
    gen = ModelGenerator([("fake-example-topic", call)])
    assert gen.check_support([EXAMPLE], {PASSAGE.id: PASSAGE}, "What color?") == [False]
    assert seen.count(generate.EXAMPLE_PROMPT) == 2  # generic support/topic retry is preserved
    assert seen.count(generate.CHECK_PROMPT) == 2


def test_ordinary_support_rejection_still_gets_its_second_reading():
    seen = []
    def call(system, user, schema):
        assert system == generate.CHECK_PROMPT
        seen.append(system)
        return json.dumps({"problem": ["addition" if len(seen) == 1 else "none"]})
    gen = ModelGenerator([("fake-ordinary-retry", call)])
    assert gen.check_support([ORDINARY], {PASSAGE.id: PASSAGE}) == [True]
    assert len(seen) == 2


@pytest.mark.parametrize("bad,expected_calls", [('{}', 2), ('not JSON', 4)])
def test_unusable_example_check_stays_fail_closed(bad, expected_calls):
    seen = []
    def call(system, user, schema):
        assert system == generate.EXAMPLE_PROMPT
        seen.append(system)
        return bad
    gen = ModelGenerator([("fake-invalid-example", call)])
    assert gen.check_support([EXAMPLE], {PASSAGE.id: PASSAGE}) is None
    assert len(seen) == expected_calls


@pytest.mark.parametrize("system,schema,label", [
    (generate.EXAMPLE_PROMPT, generate.EXAMPLE_SCHEMA, "example"),
    (generate.DEFINITION_PROMPT, generate.DEFINITION_SCHEMA, "definition"),
    (generate.EXPAND_PROMPT, generate.EXPAND_SCHEMA, "expand"),
    (generate.TRANSLATE_PROMPT.format(target="test"), generate.TRANSLATE_SCHEMA, "translation"),
])
def test_secondary_trace_label_is_precise_without_changing_health_counters(monkeypatch, caplog, system, schema, label):
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    for key in generate.USAGE:
        monkeypatch.setitem(generate.USAGE, key, 0)
    @telemetry.model_call("fake", "synthetic")
    def call(*args):
        return '{}'
    trace = telemetry.start("ask")
    with telemetry.bind(trace):
        generate.with_retry(call, system, "private synthetic input", schema)
        trace.finish("answered", 200)
    events = [e for e in trace.events if e["kind"] == "model"]
    assert len(events) == 1 and events[0]["stage"] == "model." + label
    assert generate.USAGE["attempt_other"] == 1
    assert sum(generate.USAGE[k] for k in generate.USAGE if k.startswith("attempt_")) == 1
    assert "private synthetic input" not in caplog.text
