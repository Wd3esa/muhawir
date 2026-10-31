"""Source-witness provenance and request-owned reuse; no network/provider calls."""
from dataclasses import replace
import json

import pytest

from muhawir import generate
from muhawir.corpus import Passage
from muhawir.generate import ModelGenerator, _has_evidence
from muhawir.verify import Claim

pytestmark = pytest.mark.real_check

P = Passage('s:1', 's', 'page 1', 'لا يجوز بيع الذهب بالذهب إلا مثلا بمثل يدا بيد.', 'fiqh')
C = Claim('يشترط في بيع الذهب بالذهب التماثل والتقابض.', (P.id,))


def test_witness_must_be_contiguous_in_one_cited_source():
    sources = {P.id: P, 's:2': replace(P, id='s:2', text='يجب الوفاء بالأمانة.')}
    assert _has_evidence(C, 'الذَّهَبِ بالذَّهَبِ، إلّا مِثلًا بِمثل', sources)
    for bad in ('يجوز بيع الذهب بالذهب دون تماثل', 'بيع الذهب حلال بلا شروط',
                'يدا بمثل الذهب', 'بيع الذهب بالذهب يجب الوفاء', 'بيع الذهب بالذهبب', ''):
        assert not _has_evidence(C, bad, sources)
    assert not _has_evidence(C, 'يجب الوفاء بالأمانة', sources)
    both = replace(C, passage_ids=(P.id, 's:2'))
    assert not _has_evidence(both, 'يدا بيد يجب الوفاء بالأمانة', sources)


def checker(monkeypatch, *, verdict='none'):
    monkeypatch.setattr(generate, 'READ_EACH_SENTENCE', True)
    seen = []
    def provider(system, user, schema):
        seen.append(user)
        return json.dumps({'problem': [verdict]})
    return ModelGenerator([('fake', provider)]), seen


def test_only_exact_positive_is_reused_with_original_order(monkeypatch):
    gen, seen = checker(monkeypatch)
    memo = {}
    assert gen.check_support([C], {P.id: P}, 'q', memo=memo) == [True]
    changed = replace(C, text=C.text + ' وهذا شرط.')
    assert gen.check_support([changed, C], {P.id: P}, 'q', memo=memo) == [True, True]
    assert len(seen) == 2
    assert gen.last_check_reasons == ['', '']


@pytest.mark.parametrize('change', ['claim', 'citation', 'source', 'source_id', 'grade', 'question', 'provider', 'policy'])
def test_changed_check_inputs_are_never_reused(monkeypatch, change):
    gen, seen = checker(monkeypatch)
    memo, sources, claim, question = {}, {P.id: P}, C, 'q'
    assert gen.check_support([claim], sources, question, memo=memo) == [True]
    if change == 'claim': claim = replace(C, label='تنبيه')
    elif change == 'citation':
        sources = {'s:2': replace(P, id='s:2')}
        claim = replace(C, passage_ids=('s:2',))
    elif change == 'source': sources = {P.id: replace(P, text=P.text + ' شرط آخر.')}
    elif change == 'source_id': sources = {P.id: replace(P, source_id='different-unknown-source')}
    elif change == 'grade': sources = {P.id: replace(P, grade='ضعيف')}
    elif change == 'question': question = 'different question'
    elif change == 'provider':
        old = gen.calls[0][1]
        gen.calls = [('different', lambda *args: old(*args))]
    elif change == 'policy': monkeypatch.setattr(generate, 'CHECK_PROMPT', generate.CHECK_PROMPT + ' ')
    assert gen.check_support([claim], sources, question, memo=memo) == [True]
    assert len(seen) == 2


def test_negative_results_retain_reasons_and_are_rechecked(monkeypatch):
    gen, seen = checker(monkeypatch, verdict='meaning')
    memo = {}
    assert gen.check_support([C], {P.id: P}, memo=memo) == [False]
    reason = gen.last_check_reasons[:]
    assert not memo
    first_calls = len(seen)
    assert gen.check_support([C], {P.id: P}, memo=memo) == [False]
    assert len(seen) == 2 * first_calls
    assert gen.last_check_reasons == reason


def test_failure_is_not_reused(monkeypatch):
    gen, seen = checker(monkeypatch)
    gen.calls = [('invalid', lambda *args: '{}')]
    memo = {}
    assert gen.check_support([C], {P.id: P}, memo=memo) is None
    assert not memo
    gen.calls = [('valid', lambda *args: seen.append(1) or '{"problem":["none"]}')]
    assert gen.check_support([C], {P.id: P}, memo=memo) == [True]
    assert len(seen) == 1


def test_batch_policy_does_not_reuse_individual_judgments(monkeypatch):
    gen, seen = checker(monkeypatch)
    memo = {}
    assert gen.check_support([C], {P.id: P}, memo=memo) == [True]
    monkeypatch.setattr(generate, 'READ_EACH_SENTENCE', False)
    assert gen.check_support([C], {P.id: P}, memo=memo) == [True]
    assert len(seen) == 2


def test_independent_requests_do_not_share_judgments(monkeypatch):
    gen, seen = checker(monkeypatch)
    for memo in ({}, {}):
        assert gen.check_support([C], {P.id: P}, memo=memo) == [True]
        assert gen.check_support([C], {P.id: P}, memo=memo) == [True]
    assert len(seen) == 2


def test_write_reuses_only_unchanged_accepted_claim_after_coverage_repair(monkeypatch):
    from muhawir.corpus import Corpus, Source
    from muhawir.pipeline import Muhawir
    monkeypatch.setattr(generate, 'READ_EACH_SENTENCE', True)
    counts = {'draft': 0, 'support': 0, 'review': 0}
    extra = Claim('ويشترط التقابض قبل التفرق.', (P.id,))
    def provider(system, user, schema):
        if system == generate.CHECK_PROMPT:
            counts['support'] += 1
            return '{"problem":["none"],"evidence":["مثلا بمثل يدا بيد"]}'
        if system == generate.RELEVANCE_PROMPT:
            counts['review'] += 1
            return json.dumps({'verdict': 'partly' if counts['review'] == 1 else 'yes',
                               'reason': 'التقابض يحتاج توضيحا'})
        if 'claims' in schema.get('properties', {}):
            counts['draft'] += 1
            claims = [C] if counts['draft'] == 1 else [C, extra]
            return json.dumps({'claims': [{'text': c.text, 'passage_ids': c.passage_ids} for c in claims]})
        return '{"absent":[]}'
    gen = ModelGenerator([('fake-pipeline', provider)])
    corpus = Corpus({'s': Source('s', 'Synthetic only', '')}, [P], synthetic=True)
    engine = Muhawir(corpus, gen)
    result = engine._write('ما شروط بيع الذهب بالذهب؟', [P], 'youth', 'ar', False, {})
    assert result.kept == [C, extra] and not result.partial
    assert counts == {'draft': 2, 'support': 2, 'review': 2}  # would be three support calls without reuse
    # Another answer attempt must perform both first checks again.
    engine._write('ما شروط بيع الذهب بالذهب؟', [P], 'youth', 'ar', False, {})
    assert counts['support'] == 4


def test_reuse_telemetry_reports_counts_without_content(monkeypatch, caplog):
    from muhawir import telemetry
    from scripts.analyze_request_traces import summarize
    monkeypatch.setenv('MUHAWIR_TRACE', '1')
    gen, seen = checker(monkeypatch)
    trace = telemetry.start('ask')
    with telemetry.bind(trace):
        memo = {}
        assert gen.check_support([C], {P.id: P}, memo=memo) == [True]
        assert gen.check_support([C], {P.id: P}, memo=memo) == [True]
        trace.finish('answered', 200)
    rows = [json.loads(r.message.split('muhawir_trace ', 1)[1]) for r in caplog.records
            if r.message.startswith('muhawir_trace ')]
    report = summarize(rows)
    assert report['requests'][0]['reused_support_checks'] == 1
    assert C.text not in caplog.text and P.text not in caplog.text
    assert report['observed_transport_calls'] == 0  # fake callable isn't a transport adapter
    assert len(seen) == 1


def test_mixed_rewrite_keeps_rejection_at_correct_position(monkeypatch):
    monkeypatch.setattr(generate, 'READ_EACH_SENTENCE', True)
    bad = replace(C, text='هذا استنتاج آخر ليس في المصدر.')
    def provider(system, user, schema):
        return json.dumps({'problem': ['meaning' if bad.text in user else 'none']})
    gen = ModelGenerator([('fake-mixed', provider)])
    memo = {}
    assert gen.check_support([C], {P.id: P}, memo=memo) == [True]
    assert gen.check_support([bad, C], {P.id: P}, memo=memo) == [False, True]
    assert 'meaning' in gen.last_check_reasons[0] and gen.last_check_reasons[1] == ''
    assert len(memo) == 1
