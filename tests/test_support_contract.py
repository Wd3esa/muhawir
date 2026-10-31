"""Strict checker contract with fake IO; no paid calls or semantic accuracy claims."""
from copy import deepcopy
import json

import pytest

from muhawir import generate
from muhawir.corpus import Corpus, Passage, Source
from muhawir.generate import ModelGenerator, _valid_support_reply
from muhawir.pipeline import Muhawir, Response, UNAVAILABLE
from muhawir.verify import Claim

pytestmark = pytest.mark.real_check
P = Passage('synthetic:1', 'synthetic', 'test', 'A box can have different colors.', 'other')
C = Claim('A box can have different colors.', (P.id,))
GOOD = {'on_topic': [True], 'missing': [[]], 'evidence': [P.text], 'problem': ['none']}


def invalid_replies():
    for key in GOOD:
        row = deepcopy(GOOD); del row[key]
        yield pytest.param(row, id='missing-' + key)
        for bad in ([], [GOOD[key][0], GOOD[key][0]], None, 'not-array'):
            row = deepcopy(GOOD); row[key] = bad
            yield pytest.param(row, id='array-' + key + '-' + repr(bad))
    for key, values in {'on_topic': ['true', 1, None],
                        'missing': ['not-list', [1], [None]],
                        'evidence': [None, 1, {}],
                        'problem': ['NONE', 'unknown', ' none ', True, None]}.items():
        for value in values:
            row = deepcopy(GOOD); row[key] = [value]
            yield pytest.param(row, id='type-' + key + '-' + repr(value))
    yield pytest.param({'supported': [True]}, id='legacy-supported-only')
    yield pytest.param({'problem': ['none']}, id='problem-only-old-bypass')
    yield pytest.param({**GOOD, 'supported': [True]}, id='extra-legacy-field')
    yield pytest.param({**GOOD, 'extra': 'unexpected'}, id='unknown-field')
    yield pytest.param(None, id='not-object-null')
    yield pytest.param([GOOD], id='not-object-array')


@pytest.mark.parametrize('reply', list(invalid_replies()))
def test_malformed_approval_is_unusable_and_never_memoized(monkeypatch, reply):
    monkeypatch.setattr(generate, 'READ_EACH_SENTENCE', True)
    seen = []
    def provider(*args):
        seen.append(1)
        return json.dumps(reply)
    gen, memo = ModelGenerator([('fake-contract', provider)]), {}
    assert not _valid_support_reply(reply, 1)
    assert gen.check_support([C], {P.id: P}, 'What colors?', memo=memo) is None
    assert not memo and len(seen) == 2  # existing one unusable-reading retry only


def test_complete_positive_uses_one_call_and_can_be_reused(monkeypatch):
    monkeypatch.setattr(generate, 'READ_EACH_SENTENCE', True)
    seen = []
    gen = ModelGenerator([('fake-valid', lambda *args: seen.append(1) or json.dumps(GOOD))])
    memo = {}
    assert _valid_support_reply(GOOD, 1)
    assert gen.check_support([C], {P.id: P}, 'What colors?', memo=memo) == [True]
    assert gen.check_support([C], {P.id: P}, 'What colors?', memo=memo) == [True]
    assert len(seen) == 1 and gen.last_check_reasons == ['']


def test_empty_or_unrelated_witness_cannot_approve_ordinary_claim():
    for witness in ('', 'A different animal has no color.'):
        row = {**GOOD, 'evidence': [witness]}
        assert _valid_support_reply(row, 1)  # structurally valid, semantically unsupported
        gen = ModelGenerator([('fake-witness', lambda *args: json.dumps(row))])
        assert gen.check_support([C], {P.id: P}) == [False]


def test_complete_severe_rejection_is_not_outvoted():
    seen = []
    def provider(*args):
        seen.append(1)
        return json.dumps({**GOOD, 'problem': ['dropped_condition'], 'evidence': ['']})
    gen = ModelGenerator([('fake-condition', provider)])
    assert gen.check_support([C], {P.id: P}) == [False]
    assert 'dropped_condition' in gen.last_check_reasons[0] and len(seen) == 1


def test_missing_topic_is_invalid_even_without_question():
    row = {key: value for key, value in GOOD.items() if key != 'on_topic'}
    gen = ModelGenerator([('fake-no-topic', lambda *args: json.dumps(row))])
    assert gen.check_support([C], {P.id: P}) is None


def test_malformed_rejection_is_unavailable_not_a_supported_semantic_decision():
    row = {**GOOD, 'problem': ['dropped_condition'], 'missing': 'wrong-type'}
    gen = ModelGenerator([('fake-bad-reject', lambda *args: json.dumps(row))])
    assert gen.check_support([C], {P.id: P}) is None


def test_one_invalid_batch_element_invalidates_the_whole_reply():
    row = {key: values * 2 for key, values in GOOD.items()}
    row['evidence'][1] = None
    gen = ModelGenerator([('fake-batch', lambda *args: json.dumps(row))])
    assert gen._check_once([C, C], {P.id: P}) is None


def test_full_write_does_not_publish_an_answer_approved_by_incomplete_reply(monkeypatch):
    def provider(system, *args):
        if system == generate.CHECK_PROMPT:
            return '{"problem":["none"]}'
        return '{"verdict":"yes"}'
    gen = ModelGenerator([('fake-write-contract', provider)])
    monkeypatch.setattr(gen, 'generate', lambda *args, **kwargs: [C])
    corpus = Corpus({'synthetic': Source('synthetic', 'Synthetic', 'test only')}, [P], synthetic=True)
    result = Muhawir(corpus, gen)._write('What colors?', [P], 'youth', 'en', False, {})
    assert isinstance(result, Response) and result.status == UNAVAILABLE
    assert not result.claims


def test_invalid_reply_can_only_fall_back_to_a_complete_valid_reply():
    calls = []
    def bad(*args):
        calls.append('bad')
        return '{"problem":["none"]}'
    def good(*args):
        calls.append('good')
        return json.dumps(GOOD)
    gen = ModelGenerator([('bad', bad), ('good', good)])
    assert gen.check_support([C], {P.id: P}) == [True]
    assert calls == ['bad', 'good']


def test_invalid_but_billed_attempts_remain_in_request_costs(monkeypatch):
    from muhawir import telemetry
    monkeypatch.setenv('MUHAWIR_TRACE', '1')
    @telemetry.model_call('fake', 'contract')
    def provider(*args):
        telemetry.response_metadata(None, {'cost': '0.001', 'prompt_tokens': 10,
                                           'completion_tokens': 2}, usd_cost=True)
        return '{"problem":["none"]}'
    gen = ModelGenerator([('fake-cost', provider)])
    trace = telemetry.start('ask')
    with telemetry.bind(trace):
        assert gen.check_support([C], {P.id: P}, 'What colors?') is None
        trace.finish('unavailable', 200)
    model_events = [e for e in trace.events if e['kind'] == 'model']
    assert len(model_events) == 2
    assert all(e['reported_cost_usd'] == '0.001' for e in model_events)
    assert all(e['stage'] == 'model.support' for e in model_events)
