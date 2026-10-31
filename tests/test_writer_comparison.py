"""Offline behavioral checks for comparison controls, never a model-quality evaluation."""
from copy import deepcopy
from decimal import Decimal
import json

import pytest

from scripts import compare_writer_plans as tool


def context():
    return {'question': 'What color is the box?', 'model': 'test/model',
            'answer_request': {'requirements': ['State the color', 'Keep the condition']},
            'passages': [{'id': 'synthetic:1', 'source_id': 'synthetic', 'location': 'test',
                          'text': 'The box is blue when open.', 'kind': 'other', 'grade': 'test-grade',
                          'translation': 'Approved translation'}]}


def reply(candidate=False, cost='0.001', **options):
    draft = {'abstain': False, 'claims': [{'text': 'The box is blue when open.',
                                         'passage_ids': ['synthetic:1']}]}
    if candidate:
        draft['plan'] = [{'requirement': i, 'passage_ids': ['synthetic:1']} for i in range(2)]
    draft.update(options.pop('draft', {}))
    return {'provider': 'TestProvider', 'model': 'test/model', 'usage': {'cost': cost},
            'choices': [{'finish_reason': options.pop('finish', 'stop'),
                         'message': {'content': json.dumps(draft), **options}}]}


def test_builds_both_actual_writer_requests_with_full_evidence():
    plan = tool.prepare(context(), 'youth', 2)
    assert tool.validate(plan) == 4
    a, b = plan['payloads']['current'], plan['payloads']['candidate']
    for request in [a, b]:
        assert 'The box is blue when open.' in request['user']
        assert 'test-grade' in request['user']
    assert a['schema']['properties']['plan']['type'] == 'string'
    assert b['schema']['properties']['plan']['type'] == 'array'
    assert b['schema']['properties']['plan']['maxItems'] == 2
    assert 'State the color' in a['user'] and 'State the color' in b['user']
    assert a['system'] in b['system']


@pytest.mark.parametrize('mutation', ['payload', 'evidence', 'revision', 'runner', 'pairs'])
def test_changed_plans_fail_before_send(mutation):
    plan = tool.prepare(context(), 'youth', 2)
    if mutation == 'payload': plan['payloads']['candidate']['user'] += 'altered'
    if mutation == 'evidence': plan['case']['passages'][0]['text'] += ' altered'
    if mutation == 'revision': plan['code_sha256'] = 'bad'
    if mutation == 'runner': plan['runner_sha256'] = 'bad'
    if mutation == 'pairs': plan['repeats'] = 5
    with pytest.raises(ValueError): tool.validate(plan)


def test_balanced_pair_order_and_at_most_one_send_per_variant():
    plan = tool.prepare(context(), 'youth', 2)
    sent, events = [], []
    def send(request, model):
        candidate = request['schema']['properties']['plan']['type'] == 'array'
        sent.append((candidate, model))
        return reply(candidate)
    assert tool.run(plan, send, events.append, Decimal('.10')) == 'completed_draft_only_quality_pending'
    assert sent == [(False, 'test/model'), (True, 'test/model'), (True, 'test/model'), (False, 'test/model')]
    assert sum(e['kind'] == 'started' for e in events) == 4
    assert sum(e['kind'] == 'response' for e in events) == 4
    assert all(e['semantic_quality'] == 'unreviewed' for e in events if e['kind'] == 'parsed')


@pytest.mark.parametrize('data', [reply(cost=None), reply(cost='NaN'), reply(cost=True),
                                  reply(cost='-1'), {'usage': 'bad'}, ['not-an-object']])
def test_unknown_cost_preserves_response_and_stops(data):
    events, sent = [], []
    def send(*_): sent.append(1); return data
    assert tool.run(tool.prepare(context(), 'youth', 2), send, events.append, Decimal('.1')) == 'stopped_unknown_cost'
    assert len(sent) == 1
    assert events[-1]['response'] == data and events[-1]['reported_cost_usd'] is None


@pytest.mark.parametrize('data', [reply(finish='length'), reply(refusal='no'),
    reply(tool_calls=[{'id': 'bad'}]), reply(draft={'claims': [{'text': 'invented', 'passage_ids': ['missing']}]}),
    reply(draft={'claims': []}), reply(draft={'claims': [{'text': 'uncited', 'passage_ids': []}]}),
    reply(draft={'claims': [{'bad': 'claim'}]})])
def test_invalid_answers_are_not_free_or_silently_graded(data):
    events = []
    status = tool.run(tool.prepare(context(), 'youth', 1), lambda *_: data, events.append, Decimal('.1'))
    assert status == 'stopped_invalid_draft_no_retry'
    assert [e['kind'] for e in events] == ['started', 'response', 'invalid_draft']
    assert events[1]['reported_cost_usd'] == '0.001'


def test_candidate_plan_validation_is_applied():
    data = reply(True, draft={'plan': [{'requirement': 0, 'passage_ids': ['missing']}]})
    with pytest.raises(ValueError, match='evidence plan'):
        tool.inspect_draft(data, 'candidate', tool.prepare(context(), 'youth', 1)['case'])


def test_transport_failure_has_no_retry_and_logs_no_private_exception():
    events, sent = [], []
    def send(*_): sent.append(1); raise TimeoutError('private credentials')
    result = tool.run(tool.prepare(context(), 'youth', 1), send, events.append, Decimal('.1'))
    assert result == 'stopped_transport_failure_no_retry' and len(sent) == 1
    assert 'private credentials' not in json.dumps(events)
    assert events[-1]['reported_cost_usd'] is None


def test_threshold_is_post_response_not_a_hard_cap():
    events = []
    status = tool.run(tool.prepare(context(), 'youth', 2), lambda *_: reply(cost='.11'), events.append, Decimal('.10'))
    assert status == 'stopped_post_response_threshold'
    assert sum(e['kind'] == 'response' for e in events) == 1


def test_requires_actual_full_context_and_private_nonoverwriting_outputs(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='full saved writer passages'):
        tool.prepare({'question': 'q', 'model': 'test', 'sources': []}, 'youth', 2)
    monkeypatch.setattr(tool, 'PRIVATE', tmp_path / 'private')
    output = tool.write_private(tool.PRIVATE / 'plan.json', {'question': 'private'})
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError): tool.write_private(output, {})
    with pytest.raises(ValueError): tool.write_private(tmp_path / 'outside.json', {})


def recording():
    plan = tool.prepare(context(), 'youth', 2)
    events = [{'kind': 'plan', 'plan': plan, 'plan_sha256': tool.digest(plan)}]
    tool.run(plan, lambda request, _: reply(request['schema']['properties']['plan']['type'] == 'array'),
             events.append, Decimal('.10'))
    return events


def test_analysis_pairs_routes_and_blinds_only_review_content():
    events = recording()
    report, review = tool.analyze(events)
    assert report['known_reported_cost_usd'] == '0.004'
    assert report['unknown_cost_attempts'] == 0 and report['attempts_started'] == 4
    assert all(p['same_reported_route'] for p in report['pairs'])
    assert len(review['items']) == 4 and len(report['review_mapping']) == 4
    assert 'variant' not in json.dumps(review) and 'seconds' not in json.dumps(review)
    assert review['evidence'] == events[0]['plan']['case']['passages']
    assert review['evidence'][0]['grade'] == 'test-grade'
    assert all(item['verdicts']['meaning_supported'] is None for item in review['items'])


def test_analysis_does_not_hide_unfinished_or_unknown_billing():
    events = recording()[:2]  # process dies immediately after the first attempt starts
    report, review = tool.analyze(events)
    assert report['started_without_response'] == 1 and report['unknown_cost_attempts'] == 1
    assert not report['pairs'][0]['both_responses_present'] and not review['items']


def test_analysis_flags_different_routes_and_cache_counts():
    events = recording()
    first = next(e for e in events if e['kind'] == 'response')
    first['response']['provider'] = 'OtherProvider'
    first['response']['usage']['prompt_tokens_details'] = {'cached_tokens': 100}
    report, _ = tool.analyze(events)
    assert not report['pairs'][0]['same_reported_route']
    assert report['pairs'][0]['cached_tokens_differ']


@pytest.mark.parametrize('change', ['duplicate', 'plan', 'duration'])
def test_analysis_rejects_double_counted_or_modified_records(change):
    events = recording()
    response = next(e for e in events if e['kind'] == 'response')
    if change == 'duplicate': events.append(deepcopy(response))
    if change == 'plan': events[0]['plan']['case']['question'] += ' changed'
    if change == 'duration': response['seconds'] = float('nan')
    with pytest.raises(ValueError): tool.analyze(events)
