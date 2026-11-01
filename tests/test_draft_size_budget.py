"""Offline rejection of oversized candidate drafts, with complete synthetic evidence.

These checks prove contract/orchestration behavior; they do not measure model
brevity or religious accuracy. No private source text or recorded model IO is committed.
"""
from copy import deepcopy
import json
import re

import pytest

from muhawir import generate
from muhawir.corpus import Corpus, Passage, Source
from muhawir.pipeline import Muhawir, Response
from muhawir.writer_plan import (DraftTooLarge, OVERSIZED_DRAFT, STYLE_LIMITS,
                                  bounded_schema, check_draft_size, planned_prompt)
from scripts import compare_writer_plans as tool

P = Passage('synthetic:1', 'synthetic', 'Synthetic location',
            'The box can move only when empty. Its recorded label is provisional.', 'other',
            grade='Synthetic provisional grade', translation='Synthetic approved translation')


def draft(n, *, listing=False, views=0):
    return {'plan': [{'requirement': 0, 'passage_ids': [P.id]}], 'abstain': False,
            'as_list': listing, 'claims': [
                {'text': f'The box has synthetic detail {i}.', 'passage_ids': [P.id]}
                for i in range(n)],
            'views': [{'school': f'Synthetic author {i}', 'text': 'Only an empty box can move.',
                       'passage_ids': [P.id]} for i in range(views)], 'follow_up': ''}


@pytest.mark.parametrize('style', list(STYLE_LIMITS))
@pytest.mark.parametrize('listing', [False, True])
def test_count_boundaries_keep_the_full_draft_or_reject_it(style, listing):
    prose, items = STYLE_LIMITS[style]
    limit = items if listing else prose
    valid, invalid = draft(limit, listing=listing), draft(limit + 1, listing=listing)
    original = deepcopy(invalid)
    check_draft_size(valid, style)
    with pytest.raises(DraftTooLarge): check_draft_size(invalid, style)
    assert invalid == original  # never truncate away the last condition


@pytest.mark.parametrize('style', list(STYLE_LIMITS))
def test_named_views_are_bounded_and_not_silently_deleted(style):
    maximum = STYLE_LIMITS[style][1]
    check_draft_size(draft(1, views=maximum), style)
    with pytest.raises(DraftTooLarge): check_draft_size(draft(1, views=maximum + 1), style)


def test_duplicate_objects_cannot_hide_an_oversized_raw_response():
    data = draft(1)
    data['claims'] *= 112
    with pytest.raises(DraftTooLarge): check_draft_size(data, 'youth')


def test_invalid_list_flag_cannot_hide_a_clearly_oversized_answer():
    data = draft(112); data['as_list'] = 'true'
    with pytest.raises(DraftTooLarge): check_draft_size(data, 'youth')


def test_long_original_text_conditions_and_grades_are_not_cut_to_fit():
    data = draft(1)
    text = 'Synthetic quoted text. ' * 1000 + 'Only when empty; recorded label is provisional.'
    data['claims'][0]['text'] = text
    before = deepcopy(data)
    check_draft_size(data, 'youth')
    assert data == before


def examples(prompt):
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(prompt[m.start():])[0] for m in re.finditer(r'\{"plan": ', prompt)]


def test_one_plan_format_keeps_examples_and_religious_rules_intact():
    prompt = planned_prompt(generate.SYSTEM_PROMPT, 'youth')
    before, after = examples(generate.SYSTEM_PROMPT), examples(prompt)
    assert len(before) == len(after) == 3
    assert '"plan": "' not in prompt
    assert 'وماذا يقول كل منها باختصار' not in prompt
    assert 'واجمع المقاطع المتعلقة كلها' not in prompt
    # The first section is the full source, grade, attribution and religious policy.
    assert prompt.split('ثانيًا: كيف تشرح')[0] == generate.SYSTEM_PROMPT.split('ثانيًا: كيف تشرح')[0]
    for old, new in zip(before, after):
        assert {k:v for k,v in old.items() if k != 'plan'} == {k:v for k,v in new.items() if k != 'plan'}
        assert isinstance(new['plan'], list)
        assert set(new['plan'][0]['passage_ids']) == {p for c in new['claims'] for p in c['passage_ids']}


def test_changed_prompt_examples_fail_before_any_paid_call():
    with pytest.raises(ValueError): planned_prompt(generate.SYSTEM_PROMPT.replace('{"plan": ', '{"old_plan": ', 1), 'youth')


def test_schema_limits_do_not_mutate_the_current_writer():
    original = deepcopy(generate.SCHEMA)
    schema = bounded_schema(generate.SCHEMA, 'youth')
    assert schema['properties']['claims']['maxItems'] == 10
    assert schema['properties']['views']['maxItems'] == 10
    assert generate.SCHEMA == original and 'maxItems' not in generate.SCHEMA['properties']['claims']


@pytest.mark.parametrize('field,value', [('as_list', 'false'), ('claims', 'not-an-array'), ('views', {})])
def test_malformed_fields_cannot_bypass_the_bound(field, value):
    data = draft(1); data[field] = value
    with pytest.raises(ValueError): check_draft_size(data, 'youth')


def test_oversized_generation_stops_before_support_or_fallback(monkeypatch):
    calls = []
    def first(*args): calls.append('draft'); return json.dumps(draft(112))
    def fallback(*args): pytest.fail('oversized drafts must not trigger another provider')
    gen = generate.ModelGenerator([('fake-oversize', first), ('fake-fallback', fallback)], evidence_plan=True)
    monkeypatch.setattr(gen, 'check_support', lambda *args: pytest.fail('no sentence checks on oversized output'))
    corpus = Corpus({'synthetic': Source('synthetic', 'Synthetic', '')}, [P], synthetic=True)
    result = Muhawir(corpus, gen)._write('Describe the box.', [P], 'youth', 'en', False, {})
    assert isinstance(result, Response) and result.status == 'unavailable' and not result.claims
    assert calls == ['draft'] and gen.last_note == OVERSIZED_DRAFT and gen.last_evidence_plan == ()


def test_compact_draft_keeps_a_condition_and_its_full_input_metadata():
    seen = []
    data = draft(1); data['claims'][0]['text'] = 'Only when empty; recorded label is provisional.'
    def call(system, user, schema): seen.append((system,user,schema)); return json.dumps(data)
    gen = generate.ModelGenerator([('fake-compact', call)], evidence_plan=True)
    claims = gen.generate('Describe the moving condition.', [P], 'youth', 'en')
    assert claims[0].text == data['claims'][0]['text']
    assert P.text in seen[0][1] and P.grade in seen[0][1] and P.translation in seen[0][1]
    assert len(seen) == 1 and gen.last_evidence_plan


def test_current_writer_stays_unchanged_for_the_recorded_eight_claim_shape():
    data = draft(8); data['plan'] = 'ordinary current plan'
    gen = generate.ModelGenerator([('fake-current', lambda *args: json.dumps(data))])
    assert len(gen.generate('Describe the box.', [P], 'youth', 'en')) == 8


def test_evaluation_counts_the_cost_before_rejecting_oversized_candidate():
    saved = {'question': 'Describe the box.', 'model': 'fake/model', 'passages': [P.__dict__]}
    plan = tool.prepare(saved, 'youth', 2)
    sent, events = [], []
    def send(request, model):
        candidate = request['schema']['properties']['plan']['type'] == 'array'
        sent.append(candidate)
        data = draft(112 if candidate else 1)
        return {'provider':'fake', 'model':model, 'usage':{'cost':'.001'},
                'choices':[{'finish_reason':'stop','message':{'content':json.dumps(data)}}]}
    from decimal import Decimal
    assert tool.run(plan, send, events.append, Decimal('.03')) == 'stopped_invalid_draft_no_retry'
    assert sent == [False, True]
    assert sum(e['kind'] == 'response' for e in events) == 2
    assert events[-1]['error_class'] == 'DraftTooLarge'
    report, _ = tool.analyze([{'kind':'plan','plan':plan,'plan_sha256':tool.digest(plan)}, *events])
    assert Decimal(report['known_reported_cost_usd']) == Decimal('.002')


def test_historical_analysis_keeps_its_original_unbounded_contract():
    saved = {'question': 'Describe the box.', 'model': 'fake/model', 'passages': [P.__dict__]}
    plan = tool.prepare(saved, 'youth', 1)
    # Historical records had no candidate_budget_version. No old record is
    # rewritten in practice; this synthetic recording exercises compatibility.
    del plan['candidate_budget_version']
    data = {'provider':'fake', 'model':'fake/model', 'usage':{'cost':'.001'},
            'choices':[{'finish_reason':'stop','message':{'content':json.dumps(draft(8))}}]}
    events = [{'kind':'plan', 'plan':plan, 'plan_sha256':tool.digest(plan)},
              {'kind':'started', 'trial':0, 'variant':'candidate'},
              {'kind':'response', 'trial':0, 'variant':'candidate', 'response':data, 'seconds':1},
              {'kind':'parsed', 'trial':0, 'variant':'candidate'}]
    report, review = tool.analyze(events)
    assert report['pairs'][0]['candidate']['draft_parsed'] and len(review['items']) == 1
    with pytest.raises(ValueError): tool.validate(plan)  # never execute as a new bounded run
