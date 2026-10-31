"""Offline safety/accounting/gate checks; human-label fixtures are not quality evidence."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts import compare_full_pipeline as tool


def manifest(n=2, holdout=False):
    return {'version': 1, 'partition': 'declared_holdout' if holdout else 'development',
        'independence': {'not_used_in_development': True, 'reviewer': 'fixture-only',
                         'development_questions_sha256': 'a' * 64},
        'success': {'critical_error_upper': .01, 'quality_harm_margin': .01, 'primary_metric': 'cost', 'alpha': .05},
        'cases': [{'id': str(i), 'cluster_id': str(i), 'question': 'Fixture question ' + str(i), 'style': 'youth', 'lang': 'ar',
                   'reference': {'expected_behavior': 'answer', 'requirements': ['Preserve the condition.'],
                      'review_status': 'approved', 'reviewer': 'fixture-only',
                      'evidence': [{'passage_id': 'test:1', 'text_sha256': 'a' * 64, 'passage_sha256': 'a' * 64}]}} for i in range(n)]}


def result(case='0', variant='current', cost='.01', **extra):
    return {'case_id': case, 'variant': variant, 'response': {'status': 'answered', 'claims': []},
            'measurement_complete': True, 'cache': 'none', 'client_seconds': 2 if variant == 'current' else 1,
            'reported_cost_usd': cost if variant == 'current' else str(Decimal(cost) / 2),
            'route_ids': [('fake', 'fake', 'fake')], **extra}


def judgments(rows):
    return [{'case_id': r['case_id'], 'variant': r['variant'], 'reviewer': 'fixture-only',
             'response_sha256': tool.digest(r['response']),
             **{label: label != 'critical_error' for label in tool.LABELS}} for r in rows]


@pytest.mark.parametrize('change', ['duplicate_id', 'duplicate_question', 'pending_reference', 'no_evidence', 'no_independence', 'no_margin'])
def test_manifest_does_not_turn_old_or_unreviewed_cases_into_holdout(change):
    m = manifest(2, True)
    if change == 'duplicate_id': m['cases'][1]['id'] = m['cases'][0]['id']
    if change == 'duplicate_question': m['cases'][1]['question'] = m['cases'][0]['question']
    if change == 'pending_reference': m['cases'][0]['reference']['review_status'] = 'pending'
    if change == 'no_evidence': m['cases'][0]['reference']['evidence'] = []
    if change == 'no_independence': m['independence'] = {}
    if change == 'no_margin': del m['success']['quality_harm_margin']
    with pytest.raises(ValueError): tool.validate_manifest(m)


def test_zero_failures_is_not_proof_of_zero_error():
    assert tool.upper_binomial(0, 1) == pytest.approx(.95)
    assert tool.upper_binomial(0, 298) > .01
    assert tool.upper_binomial(0, 299) <= .01
    assert tool.upper_binomial(1, 10) > tool.upper_binomial(0, 10)
    assert tool.upper_binomial(10, 10) == 1
    with pytest.raises(ValueError): tool.upper_binomial(0, 0)


def test_small_or_development_samples_cannot_pass_general_gate():
    for n, held in [(2, True), (300, False)]:
        rows = [result(c['id'], v) for c in manifest(n)['cases'] for v in ['current', 'candidate']]
        out = tool.verdict(manifest(n, held), rows, judgments(rows))
        assert out['general_improvement'] is False
    assert out['checks']['primary_metric_pass']


def test_gate_requires_complete_labels_sources_routes_and_paired_results():
    m = manifest(300, True)
    rows = [result(c['id'], v) for c in m['cases'] for v in ['current', 'candidate']]
    labels = judgments(rows)
    assert tool.verdict(m, rows, labels)['general_improvement']  # artificial gate fixture only
    assert not tool.verdict(m, rows, labels[:-1])['general_improvement']
    assert not tool.verdict(m, rows[:-1], labels)['general_improvement']
    rows[0]['cache'] = 'memory'
    assert not tool.verdict(m, rows, labels)['general_improvement']
    rows[0]['cache'] = 'none'; rows[0]['route_ids'] = [('different', 'different', 'different')]
    assert not tool.verdict(m, rows, labels)['general_improvement']


def test_critical_error_and_wrong_response_hash_cannot_be_approved():
    m = manifest(2, True); rows = [result(c['id'], v) for c in m['cases'] for v in ['current', 'candidate']]
    labels = judgments(rows); labels[1]['critical_error'] = True
    assert not tool.verdict(m, rows, labels)['checks']['no_observed_critical_error']
    labels[0]['response_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='different response'): tool.verdict(m, rows, labels)


def test_paired_order_all_statuses_and_cost_stop_are_preserved():
    calls, events = [], []
    def ask(v, case): calls.append((case['id'], v)); return result(case['id'], v)
    rows, status = tool.run(manifest(), ask, events.append, Decimal('.1'))
    assert calls == [('0', 'current'), ('0', 'candidate'), ('1', 'candidate'), ('1', 'current')]
    assert status == 'completed_pending_independent_review' and len(rows) == 4
    calls.clear()
    _, status = tool.run(manifest(), ask, events.append, Decimal('.005'))
    assert len(calls) == 1 and status == 'stopped_post_response_cost_threshold'


@pytest.mark.parametrize('change,expected', [('unknown', 'stopped_unknown_cost'), ('cache', 'stopped_measurement_gap_or_cache'),
                                           ('pending', 'stopped_measurement_gap_or_cache')])
def test_no_questions_after_incomplete_accounting(change, expected):
    calls = []
    def ask(v, case):
        calls.append(1); row = result(case['id'], v)
        if change == 'unknown': row['reported_cost_usd'] = None
        if change == 'cache': row['cache'] = 'disk'
        if change == 'pending': row['measurement_complete'] = False
        return row
    _, status = tool.run(manifest(), ask, lambda _: None, Decimal('.1'))
    assert status == expected and len(calls) == 1


def test_private_model_budget_caps_actual_http_and_never_saves_headers(monkeypatch):
    from muhawir import transport
    sent, events = [], []
    def post(url, **kw):
        sent.append(1)
        return httpx.Response(200, json={'usage': {'cost': '.001'}}, request=httpx.Request('POST', url))
    monkeypatch.setattr(transport, 'post', post)
    with tool.ModelHTTPBudget(2, Decimal('.1'), events.append) as budget:
        for _ in range(2): transport.post('https://openrouter.ai/api/v1/chat/completions', headers={'authorization':'secret-key'}, json={'model': 'fake'})
        with pytest.raises(tool.BudgetStopped): transport.post('https://openrouter.ai/api/v1/chat/completions')
        assert budget.sent == len(sent) == 2
    assert transport.post is post and 'secret-key' not in json.dumps(events)
    assert sum(e['kind'] == 'model_http_started' for e in events) == 2


def test_unknown_provider_billing_stops_parallel_following_calls(monkeypatch):
    from muhawir import transport
    monkeypatch.setattr(transport, 'post', lambda url, **kw: httpx.Response(200, json={}, request=httpx.Request('POST', url)))
    with tool.ModelHTTPBudget(10, Decimal('.1'), lambda _: None):
        transport.post('https://openrouter.ai/api/v1/chat/completions')
        with pytest.raises(tool.BudgetStopped): transport.post('https://openrouter.ai/api/v1/chat/completions')


def test_reference_fingerprints_and_actual_db_path_are_checked(tmp_path, monkeypatch):
    from muhawir.corpus import Passage
    p = Passage('test:1', 'test', 'p1', 'The condition is preserved.')
    corpus = SimpleNamespace(_path=tmp_path / 'external.db', passage=lambda pid: p if pid == p.id else None)
    corpus._path.write_bytes(b'database fixture')
    monkeypatch.setattr(tool, 'ROOT', tmp_path / 'other-checkout')
    assert str(corpus._path) in tool.source_fingerprints(corpus)
    m = manifest(1, True); m['cases'][0]['reference']['evidence'][0]['text_sha256'] = hashlib.sha256(p.text.encode()).hexdigest()
    from dataclasses import asdict
    m['cases'][0]['reference']['evidence'][0]['passage_sha256'] = tool.digest(asdict(p))
    tool.verify_references(m, corpus)
    m['cases'][0]['reference']['evidence'][0]['text_sha256'] = 'b' * 64
    with pytest.raises(ValueError): tool.verify_references(m, corpus)


def test_blinded_review_requires_matching_result_hash_and_completed_run():
    m = manifest(); rows = [result(c['id'], v) for c in m['cases'] for v in ['current', 'candidate']]
    events = [{'kind': 'manifest', 'manifest': m, 'manifest_sha256': tool.digest(m)}]
    for row in rows:
        events += [{'kind': 'started', 'case_id': row['case_id'], 'variant': row['variant']}, {'kind': 'result', 'result': row}]
    report, review = tool.analyze_recording(events)
    assert not report['execution_completed'] and not report['general_improvement']
    assert all(x['meaning_supported'] is None for x in review['items'])
    assert 'variant' not in json.dumps(review) and 'reported_cost_usd' not in json.dumps(review)
    events += [{'kind': 'finished', 'status': 'completed_pending_independent_review', 'sources_unchanged': True, 'code_unchanged': True}]
    assert tool.analyze_recording(events)[0]['execution_completed']
    events.append(events[1])
    with pytest.raises(ValueError, match='duplicate'): tool.analyze_recording(events)


@pytest.mark.real_check
def test_real_api_retrieval_writer_and_checker_run_for_both_arms_without_network(monkeypatch):
    from muhawir import server, pipeline, generate, telemetry
    from muhawir.corpus import Corpus, Passage, Source
    p = Passage('test-a:1', 'test-a', 'p1', 'تحتاج النخلة إلى ماء كثير في الصيف.', 'other')
    corpus = Corpus({'test-a': Source('test-a', 'Synthetic', '')}, [p], synthetic=True)
    calls = []
    @telemetry.model_call('fake', 'fake')
    def provider(system, user, schema):
        telemetry.response_metadata(None, {'cost': '.001'}, usd_cost=True)
        calls.append(system)
        if system == generate.UNDERSTAND_PROMPT:
            return json.dumps({'queries': [], 'answer_request': {'form': 'explanation', 'requirements': ['بيان الحاجة.']}})
        if system.startswith(generate.SYSTEM_PROMPT):
            return json.dumps({'abstain': False, 'claims': [{'text': p.text, 'passage_ids': [p.id]}],
                'plan': [{'requirement': 0, 'passage_ids': [p.id]}] if schema['properties']['plan']['type'] == 'array' else 'write'})
        if system == generate.CHECK_PROMPT:
            return json.dumps({'on_topic': [True], 'missing': [[]], 'evidence': [p.text], 'problem': ['none']})
        return '{"verdict":"yes","absent":[]}'
    current = pipeline.Muhawir(corpus, generate.ModelGenerator([('fake-full-current', provider)]))
    candidate = pipeline.Muhawir(corpus, generate.ModelGenerator([('fake-full-candidate', provider)], evidence_plan=True))
    monkeypatch.setattr(pipeline, 'CACHE_SIZE', 0)
    monkeypatch.setenv('MUHAWIR_TRACE', '1')
    monkeypatch.setattr(server, '_allow_question', lambda _: True)
    monkeypatch.setattr(server, '_ask_calls', __import__('collections').deque())
    case = {'id': 'fake-only', 'question': 'ماذا تحتاج النخلة في الصيف؟', 'style': 'youth', 'lang': 'ar'}
    original = server.engine
    with tool.IsolatedAPI(server, current, candidate) as ask:
        rows = [ask(v, case) for v in ['current', 'candidate']]
    assert server.engine is original
    for row in rows:
        assert row['measurement_complete'] and row['cache'] == 'none'
        assert row['response']['status'] == 'answered'
        assert row['source_snapshots'][p.id]['passage']['text'] == p.text
        events = [e for t in row['traces'] for e in t['events'] if e['kind'] == 'model']
        assert Decimal(row['reported_cost_usd']) == Decimal('.001') * len(events)
        assert any(e['stage'] == 'model.understand' for e in events)
    assert calls.count(generate.UNDERSTAND_PROMPT) >= 2
    assert calls.count(generate.CHECK_PROMPT) >= 2
    assert sum(system.startswith(generate.SYSTEM_PROMPT) for system in calls) == 2
    assert rows[0]['route_ids'] == rows[1]['route_ids']


def test_hybrid_query_caches_are_separate_but_local_weights_are_shared(monkeypatch):
    import numpy as np
    from muhawir import generate
    from muhawir.corpus import Corpus, Passage, Source
    from muhawir.retrieve import Retriever
    from muhawir.vectors import HybridRetriever, VectorIndex
    p = Passage('test:1', 'test', 'p1', 'Synthetic box.', 'other')
    corpus = Corpus({'test': Source('test', 'Synthetic', '')}, [p], synthetic=True)
    index = VectorIndex([p.id], np.array([[1., 0.]], dtype=np.float32), 'local:fake', query_embedder=lambda q: [1., 0.])
    old = HybridRetriever(Retriever(corpus), index)
    gen = generate.ModelGenerator([('fake-fresh', lambda *_: '{}')])
    monkeypatch.setenv('MUHAWIR_VECTORS', '1')
    monkeypatch.setattr(generate, 'get_generator', lambda: generate.ModelGenerator([('fake-next', lambda *_: '{}')]))
    server = SimpleNamespace(engine=SimpleNamespace(corpus=corpus, retriever=old, generator=gen))
    a, b = tool.isolated_engines(server)
    assert a.retriever is not b.retriever
    assert a.retriever.vectors is b.retriever.vectors
    a.retriever._query_vector('unique question')
    assert 'unique question' not in b.retriever._query_cache
    assert a.generator.evidence_plan is False and b.generator.evidence_plan is True


def writer_recording():
    from scripts import compare_writer_plans as writer
    saved = {'question': 'Synthetic box?', 'model': 'fake/model',
        'passages': [{'id': 'test:1', 'source_id': 'test', 'location': 'p1', 'text': 'A blue box.', 'kind': 'other'}]}
    plan = writer.prepare(saved, 'youth', 1)
    events = [{'kind': 'plan', 'plan': plan, 'plan_sha256': tool.digest(plan)}]
    def send(request, model):
        draft = {'abstain': False, 'claims': [{'text': 'A blue box.', 'passage_ids': ['test:1']}]}
        if request['schema']['properties']['plan']['type'] == 'array':
            draft['plan'] = [{'requirement': 0, 'passage_ids': ['test:1']}]
        return {'usage': {'cost': '.001'}, 'provider': 'fake', 'model': model,
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(draft)}}]}
    status = writer.run(plan, send, events.append, Decimal('.1'))
    events.append({'kind': 'finished', 'status': status})
    _, review = writer.analyze(events)
    review['reviewer'] = 'fixture-only'
    for item in review['items']:
        for key in item['verdicts']:
            if key != 'notes': item['verdicts'][key] = key != 'unnecessary_repetition'
    return events, review


def test_writer_stage_must_have_matching_complete_candidate_review():
    events, review = writer_recording()
    assert tool.writer_gate(events, review)['passed']
    bad = deepcopy(review); bad['items'][0]['response_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='different response'): tool.writer_gate(events, bad)
    bad = deepcopy(review); bad['items'][0]['verdicts']['question_covered'] = None
    with pytest.raises(ValueError, match='incomplete'): tool.writer_gate(events, bad)
    from scripts.compare_writer_plans import analyze
    mapping = analyze(events)[0]['review_mapping']
    candidate = next(x['id'] for x in mapping if x['variant'] == 'candidate')
    next(x for x in review['items'] if x['id'] == candidate)['verdicts']['conditions_preserved'] = False
    with pytest.raises(ValueError, match='has not passed'): tool.writer_gate(events, review)


def test_holdout_cannot_count_paraphrases_as_multiple_declared_clusters():
    m = manifest(2, True); m['cases'][1]['cluster_id'] = m['cases'][0]['cluster_id']
    with pytest.raises(ValueError, match='independent question'): tool.validate_manifest(m)


def test_parallel_budget_never_exceeds_explicit_model_http_starts(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from muhawir import transport
    barrier = threading.Barrier(3); sent = []
    def post(url, **kw):
        sent.append(1); barrier.wait(timeout=3)
        return httpx.Response(200, json={'usage': {'cost': '.001'}}, request=httpx.Request('POST', url))
    monkeypatch.setattr(transport, 'post', post)
    with tool.ModelHTTPBudget(3, Decimal('.1'), lambda _: None) as budget:
        def attempt(_):
            try: transport.post('https://openrouter.ai/api/v1/chat/completions'); return True
            except tool.BudgetStopped: return False
        with ThreadPoolExecutor(max_workers=10) as pool: outcomes = list(pool.map(attempt, range(10)))
        assert sum(outcomes) == len(sent) == budget.sent == 3
        assert budget.known_cost == Decimal('.003')


def accounting_recording():
    m = manifest(1)
    events = [{'kind': 'manifest', 'manifest': m, 'manifest_sha256': tool.digest(m)}]
    for sequence, variant in enumerate(['current', 'candidate'], 1):
        row = result('0', variant)
        rid = 'trace-' + variant
        row['response']['request_id'] = rid
        row['traces'] = [{'schema_version': 1, 'request_id': rid, 'endpoint': 'ask', 'style': 'youth',
            'lang': 'ar', 'as_question': False, 'code_revision': 'fixture', 'cache': 'none',
            'status': 'answered', 'http_status': 200, 'duration_ns': 1000, 'events_dropped': 0,
            'pending_work_at_response': 0, 'background_complete': True,
            'events': [{'kind': 'model', 'stage': 'model.draft', 'provider': 'fake', 'model': 'fake',
                        'outcome': 'ok', 'reported_cost_usd': row['reported_cost_usd']}]}]
        events += [{'kind': 'started', 'case_id': '0', 'variant': variant},
            {'kind': 'model_http_started', 'sequence': sequence},
            {'kind': 'model_http_response', 'sequence': sequence, 'reported_cost_usd': row['reported_cost_usd']},
            {'kind': 'result', 'result': row}]
    events.append({'kind': 'finished', 'status': 'completed_pending_independent_review',
                   'sources_unchanged': True, 'code_unchanged': True})
    return events


def test_independent_wire_trace_and_question_costs_reconcile():
    events = accounting_recording()
    assert tool.analyze_recording(events)[0]['accounting_verified']
    row = next(e['result'] for e in events if e['kind'] == 'result')
    row['reported_cost_usd'] = '.99'
    report = tool.analyze_recording(events)[0]
    assert not report['accounting_verified'] and not report['general_improvement']


@pytest.mark.parametrize('change', ['pending', 'unknown_wire', 'missing_wire', 'wrong_request'])
def test_incomplete_accounting_cannot_become_a_general_improvement(change):
    events = accounting_recording()
    row = next(e['result'] for e in events if e['kind'] == 'result')
    if change == 'pending': row['traces'][0]['background_complete'] = False
    if change == 'unknown_wire': next(e for e in events if e['kind'] == 'model_http_response')['reported_cost_usd'] = None
    if change == 'missing_wire': events = [e for e in events if not (e['kind'] == 'model_http_response' and e['sequence'] == 1)]
    if change == 'wrong_request': row['response']['request_id'] = 'wrong'
    report = tool.analyze_recording(events)[0]
    assert not report['accounting_verified'] and not report['general_improvement']
