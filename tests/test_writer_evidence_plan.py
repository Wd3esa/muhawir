"""Candidate writer protocol and exact deduplication, with synthetic sources only."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from muhawir import generate, telemetry
from muhawir.corpus import Corpus, Passage, Source
from muhawir.generate import ModelGenerator
from muhawir.pipeline import Muhawir
from muhawir.verify import Claim
from muhawir.writer_plan import EvidenceTask, planned_schema, read_plan, requirements_for

P = Passage('test:1', 'test', 'page 1', 'A box can have different colors. It must be empty before moving.', 'other', grade='Synthetic grading only')
Q = Passage('test:2', 'test', 'page 2', 'The box has a lid.', 'other')
C = Claim('A box can have different colors.', (P.id,))
REQUEST = {'form': 'explanation', 'requirements': ['Describe the box.', 'State the moving condition.']}
PLAN = [{'requirement': 0, 'passage_ids': [P.id]}, {'requirement': 1, 'passage_ids': [P.id]}]


def response(plan=PLAN):
    return {'plan': deepcopy(plan), 'abstain': False, 'as_list': False,
            'claims': [{'text': C.text, 'passage_ids': list(C.passage_ids)}], 'views': [], 'follow_up': ''}


def test_every_task_has_its_own_row_even_with_shared_source():
    assert read_plan(response(), 2, [P], [C]) == (EvidenceTask(0, (P.id,)), EvidenceTask(1, (P.id,)))
    assert read_plan(response(list(reversed(PLAN))), 2, [P], [C])[0].requirement == 0


@pytest.mark.parametrize('plan', [
    [], PLAN[:1], PLAN + PLAN[:1], [PLAN[0], PLAN[0]],
    [{'requirement': True, 'passage_ids': [P.id]}, PLAN[1]],
    [{'requirement': 2, 'passage_ids': [P.id]}, PLAN[1]],
    [{'requirement': -1, 'passage_ids': [P.id]}, PLAN[1]],
    [{'requirement': 0.0, 'passage_ids': [P.id]}, PLAN[1]],
    [{'requirement': 0, 'passage_ids': ['invented']}, PLAN[1]],
    [{'requirement': 0, 'passage_ids': [P.id, P.id]}, PLAN[1]],
    [{'requirement': 0, 'passage_ids': [1]}, PLAN[1]],
    [{'requirement': 0, 'passage_ids': P.id}, PLAN[1]],
    [{'requirement': 0, 'passage_ids': [P.id], 'claim': 'invented'}, PLAN[1]],
    ['bad', PLAN[1]], 'a free-text plan', None,
])
def test_incomplete_or_fabricated_plans_are_not_accepted(plan):
    assert read_plan({'plan': plan}, 2, [P], [C]) is None


def test_claim_citations_must_be_supplied_and_planned():
    assert read_plan(response(), 2, [P, Q], [Claim('A lid.', (Q.id,))]) is None
    assert read_plan(response(), 2, [P, Q], [Claim('A lid.', ())]) is None


def test_unresolved_task_is_not_silently_removed_or_called_verified():
    unresolved = [PLAN[0], {'requirement': 1, 'passage_ids': []}]
    assert read_plan({'plan': unresolved}, 2, [P], [C])[1] == EvidenceTask(1, ())
    # A mapping does not certify meaning. The downstream semantic check is still required.
    assert read_plan(response(), 2, [P], [replace(C, text='The box needs no moving condition.')]) is not None


def test_schema_keeps_claim_protocol_and_does_not_mutate_base():
    original = deepcopy(generate.SCHEMA)
    schema = planned_schema(generate.SCHEMA, 2, [P.id])
    assert generate.SCHEMA == original
    assert schema['properties']['claims'] == original['properties']['claims']
    assert schema['properties']['views'] == original['properties']['views']
    assert schema['properties']['plan']['minItems'] == schema['properties']['plan']['maxItems'] == 2
    assert schema['properties']['plan']['items']['properties']['passage_ids']['items']['enum'] == [P.id]


def test_candidate_uses_one_writer_call_and_preserves_full_source_metadata():
    seen = []
    def call(system, user, schema):
        seen.append((system, user, schema))
        return json.dumps(response())
    gen = ModelGenerator([('fake-plan', call)], evidence_plan=True)
    assert gen.generate('Describe and explain moving.', [P, Q], 'youth', 'en', answer_request=REQUEST) == [C]
    assert len(seen) == 1 and len(gen.last_evidence_plan) == 2
    assert generate.PLAN_INSTRUCTION in seen[0][0]
    assert P.text in seen[0][1] and Q.text in seen[0][1] and P.grade in seen[0][1]
    assert all(seen[0][1].count(task) == 1 for task in REQUEST['requirements'])
    assert seen[0][2]['properties']['plan']['type'] == 'array'


def test_existing_writer_remains_default_and_uses_unchanged_prompt():
    seen = []
    def call(system, user, schema):
        seen.append((system, user, schema))
        return json.dumps(response('ordinary free-text plan'))
    gen = ModelGenerator([('fake-default-plan', call)])
    assert gen.generate('Describe.', [P], 'youth', 'en') == [C]
    assert seen[0][0] == generate.SYSTEM_PROMPT and seen[0][2] is generate.SCHEMA
    assert gen.last_evidence_plan == ()


def test_broken_candidate_does_not_fall_back_to_unplanned_claims():
    for raw in ('not JSON', json.dumps(response('old free-text plan'))):
        gen = ModelGenerator([('fake-broken-plan', lambda *args: raw)], evidence_plan=True)
        assert gen.generate('Describe.', [P], 'youth', 'en', answer_request=REQUEST) == []
        assert gen.last_evidence_plan == () and 'unusable evidence plan' in gen.last_note


def test_no_sources_means_no_new_model_call():
    seen = []
    gen = ModelGenerator([('fake-empty-plan', lambda *args: seen.append(1))], evidence_plan=True)
    assert gen.generate('Describe.', [], 'youth', 'en') == [] and not seen


def test_plan_is_request_local_and_reset_after_invalid_rewrite():
    gen = ModelGenerator([('fake-reset-plan', lambda *args: json.dumps(response()))], evidence_plan=True)
    assert gen.generate('Describe.', [P], 'youth', 'en', answer_request=REQUEST) == [C]
    assert gen.last_evidence_plan
    gen.calls = [('fake-invalid-plan', lambda *args: '{}')]
    assert gen.generate('Describe.', [P], 'youth', 'en', answer_request=REQUEST) == []
    assert gen.last_evidence_plan == ()


def test_writer_stage_stays_precise_for_candidate_prompt(monkeypatch):
    monkeypatch.setenv('MUHAWIR_TRACE', '1')
    @telemetry.model_call('fake', 'planned-writer')
    def call(*args): return json.dumps(response())
    gen = ModelGenerator([('fake-trace-plan', call)], evidence_plan=True)
    trace = telemetry.start('ask')
    with telemetry.bind(trace):
        assert gen.generate('Describe.', [P], 'youth', 'en', answer_request=REQUEST) == [C]
        trace.finish('answered', 200)
    model_events = [event for event in trace.events if event['kind'] == 'model']
    assert len(model_events) == 1 and model_events[0]['stage'] == 'model.draft'


def test_no_requirements_keeps_whole_question_as_one_task():
    assert requirements_for('Whole question.', {}) == ('Whole question.',)
    assert requirements_for('Whole question.', {'requirements': 'not-an-array'}) == ('Whole question.',)
    assert requirements_for('Whole question.', {'requirements': ['', None]}) == ('Whole question.',)


def test_only_exact_duplicate_claims_are_removed_before_check():
    corpus = Corpus({'test': Source('test', 'Synthetic', '')}, [P, Q], synthetic=True)
    engine = Muhawir(corpus, ModelGenerator([('fake-dedup-plan', lambda *args: '{}')]))
    distinct = [replace(C, text='A box cannot have different colors.'),
                replace(C, passage_ids=(Q.id,)), replace(C, section='Other requirement'),
                replace(C, school='Another author'), replace(C, label='A different label')]
    assert engine._tidy([C, C, *distinct, C], {P.id, Q.id}) == [C, *distinct]


def test_cleanup_does_not_make_distinct_original_claims_eligible_for_deduplication():
    corpus = Corpus({'test': Source('test', 'Synthetic', '')}, [P], synthetic=True)
    engine = Muhawir(corpus, ModelGenerator([('fake-distinct-plan', lambda *args: '{}')]))
    first = replace(C, text='توضيح مدعوم رقم 1.')
    second = replace(C, text='توضيح مدعوم رقم 2.')
    assert len(engine._tidy([first, second], {P.id})) == 2


@pytest.mark.real_check
def test_full_pipeline_checks_only_one_copy_of_an_identical_claim():
    condition = Claim('The box must be empty before moving.', (P.id,))
    calls = {'draft': 0, 'support': 0, 'review': 0}
    def call(system, user, schema):
        if 'claims' in schema.get('properties', {}):
            calls['draft'] += 1
            data = response()
            data['claims'] = [{'text': claim.text, 'passage_ids': list(claim.passage_ids)}
                              for claim in [C, C, condition]]
            return json.dumps(data)
        if system == generate.CHECK_PROMPT:
            calls['support'] += 1
            return json.dumps({'on_topic': [True], 'missing': [[]], 'evidence': [P.text], 'problem': ['none']})
        if system == generate.RELEVANCE_PROMPT:
            calls['review'] += 1
            assert all(task in user for task in REQUEST['requirements'])
            return '{"verdict":"yes"}'
        return '{"absent":[]}'
    gen = ModelGenerator([('fake-pipeline-plan', call)], evidence_plan=True)
    corpus = Corpus({'test': Source('test', 'Synthetic', '')}, [P], synthetic=True)
    result = Muhawir(corpus, gen)._write('Describe and explain moving.', [P], 'youth', 'en', False,
                                        {'answer_request': REQUEST})
    assert result.kept == [C, condition]
    assert calls == {'draft': 1, 'support': 2, 'review': 1}


def test_unusable_plan_is_service_unavailable_not_source_absence():
    from muhawir.pipeline import Response, UNAVAILABLE
    gen = ModelGenerator([('fake-invalid-write-plan', lambda *args: '{"plan":"legacy"}')], evidence_plan=True)
    corpus = Corpus({'test': Source('test', 'Synthetic', '')}, [P], synthetic=True)
    result = Muhawir(corpus, gen)._write('Describe.', [P], 'youth', 'en', False, {'answer_request': REQUEST})
    assert isinstance(result, Response) and result.status == UNAVAILABLE and not result.claims


def test_valid_fallback_clears_previous_plan_failure():
    gen = ModelGenerator([('fake-invalid-first-plan', lambda *args: '{}'),
                          ('fake-valid-second-plan', lambda *args: json.dumps(response()))], evidence_plan=True)
    assert gen.generate('Describe.', [P], 'youth', 'en', answer_request=REQUEST) == [C]
    assert not gen.last_note and gen.last_evidence_plan


def test_plans_are_isolated_between_parallel_requests():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    gate = Barrier(2)
    def call(system, user, schema):
        gate.wait(timeout=3)
        passage = P if 'first task' in user else Q
        data = response([{'requirement': 0, 'passage_ids': [passage.id]}])
        data['claims'] = [{'text': passage.text, 'passage_ids': [passage.id]}]
        return json.dumps(data)
    gen = ModelGenerator([('fake-isolated-plan', call)], evidence_plan=True)
    def run(question):
        gen.generate(question, [P, Q], 'youth', 'en')
        return gen.last_evidence_plan
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(run, ['first task', 'second task']))
    assert first == (EvidenceTask(0, (P.id,)),) and second == (EvidenceTask(0, (Q.id,)),)
    assert gen.last_evidence_plan == ()


@pytest.mark.parametrize('lang', ['en', 'ur'])
def test_planning_preserves_original_verse_and_approved_translation(lang):
    verse = replace(P, kind='quran', translation='Synthetic approved translation.')
    seen = []
    def call(system, user, schema):
        seen.append(user)
        return json.dumps(response([{'requirement': 0, 'passage_ids': [P.id]}]))
    gen = ModelGenerator([('fake-translation-plan', call)], evidence_plan=True)
    assert gen.generate('Describe.', [verse], 'youth', lang) == [C]
    assert verse.text in seen[0] and verse.translation in seen[0]


def test_incomplete_planned_answer_does_not_leave_an_accepted_plan():
    data = response()
    data['claims'] = []  # not an intentional abstention
    gen = ModelGenerator([('fake-empty-claims-plan', lambda *args: json.dumps(data))], evidence_plan=True)
    assert gen.generate('Describe.', [P], 'youth', 'en', answer_request=REQUEST) == []
    assert gen.last_evidence_plan == () and 'unusable evidence plan' in gen.last_note
