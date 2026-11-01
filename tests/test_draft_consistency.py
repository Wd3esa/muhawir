"""Accepted draft text, layout and coverage must survive a rejected rewrite together.

Synthetic model replies exercise the real writer/reviewer/publication path; they
do not measure religious correctness or model quality.
"""
import json

import pytest

from muhawir import generate as g
from muhawir.corpus import Corpus, Passage, Source
from muhawir.messages import TEXT
from muhawir.pipeline import Muhawir


P = Passage('test:box', 'test', 'Synthetic box description',
            'This test describes a box with several documented features.', 'other')
CORPUS = Corpus({'test': Source('test', 'Synthetic source', 'Tests only')}, [P])
QUESTION = 'Describe the box and its lid.'
REQUEST = {'form': 'explanation', 'requirements': ['Describe the box.', 'Describe the lid.']}
CLAIMS = [{'text': f'The box has documented feature number {i}.', 'passage_ids': [P.id]}
          for i in range(7)]
FIRST_FOLLOW_UP = 'What else would you like to know about the box?'
SECOND_FOLLOW_UP = 'Would you like to discuss the lid?'


def engine(monkeypatch, replacement, reviews, *, first_as_list=True):
    first = {'claims': CLAIMS, 'as_list': first_as_list, 'follow_up': FIRST_FOLLOW_UP}
    drafts, reviews = iter([first, replacement]), iter(reviews)
    seen = {'drafts': [], 'reviews': []}

    def call(system, user, schema):
        if system in (g.RELEVANCE_PROMPT, g.KIDS_REVIEW_PROMPT):
            seen['reviews'].append(user)
            return json.dumps(next(reviews))
        if 'claims' in schema.get('properties', {}):
            seen['drafts'].append(user)
            reply = next(drafts)
            if isinstance(reply, Exception):
                raise reply
            return json.dumps(reply)
        return '{"absent": []}'

    gen = g.ModelGenerator([('fake-draft-consistency', call)])
    m = Muhawir(CORPUS, gen)
    monkeypatch.setattr(m, '_gather', lambda *args: ([P], []))
    monkeypatch.setattr(gen, 'check_support', lambda claims, *args: [True] * len(claims))
    return m, seen


@pytest.mark.parametrize('replacement', [
    {'claims': [{'text': 'A replacement with an invented source.', 'passage_ids': ['invented']}],
     'as_list': False, 'follow_up': SECOND_FOLLOW_UP},
    {'abstain': True, 'claims': [], 'as_list': False, 'follow_up': SECOND_FOLLOW_UP},
    RuntimeError('synthetic transport failure'),
])
def test_rejected_rewrite_keeps_original_list_follow_up_and_known_gaps(monkeypatch, replacement):
    m, seen = engine(monkeypatch, replacement, [
        {'verdict': 'partly', 'missing_parts': [1]},
        # A redundant re-review of unchanged text could wrongly clear known gaps.
        {'verdict': 'yes'},
    ])
    res = m._ask(QUESTION, 'youth', 'en', answer_request=REQUEST)
    assert res.status == 'answered'
    assert res.claims == CLAIMS and res.as_list
    assert res.follow_up == FIRST_FOLLOW_UP
    assert TEXT['en']['partial_note'] in res.note and REQUEST['requirements'][1] in res.note
    assert len(seen['drafts']) == 2 and len(seen['reviews']) == 1


def test_adopted_rewrite_publishes_and_reviews_its_own_list_and_follow_up(monkeypatch):
    replacement = {'claims': CLAIMS, 'as_list': True, 'follow_up': SECOND_FOLLOW_UP}
    m, seen = engine(monkeypatch, replacement, [
        {'verdict': 'partly', 'missing_parts': [1]}, {'verdict': 'yes'}], first_as_list=False)
    res = m._ask(QUESTION, 'youth', 'en', answer_request=REQUEST)
    assert res.claims == CLAIMS and res.as_list and res.follow_up == SECOND_FOLLOW_UP
    assert TEXT['en']['partial_note'] not in res.note
    assert CLAIMS[-1]['text'] not in seen['reviews'][0]
    assert CLAIMS[-1]['text'] in seen['reviews'][1]
    assert len(seen['drafts']) == len(seen['reviews']) == 2


def test_explicit_missing_requirement_cannot_be_certified_complete(monkeypatch):
    replacement = {'claims': CLAIMS, 'as_list': True}
    contradictory = {'verdict': 'yes', 'missing_parts': [1]}
    m, seen = engine(monkeypatch, replacement, [contradictory, contradictory])
    res = m._ask(QUESTION, 'youth', 'en', answer_request=REQUEST)
    assert TEXT['en']['partial_note'] in res.note and REQUEST['requirements'][1] in res.note
    assert len(seen['drafts']) == len(seen['reviews']) == 2
    feedback = seen['drafts'][1].split('مراجعة لجواب سابق:\n', 1)[1].split('السؤال (بيانات)', 1)[0]
    assert json.dumps([REQUEST['requirements'][1]], ensure_ascii=False) in feedback


def test_shorter_source_repair_cannot_replace_retained_draft_metadata(monkeypatch):
    replacement = {'claims': [CLAIMS[1]], 'as_list': False, 'follow_up': SECOND_FOLLOW_UP}
    m, seen = engine(monkeypatch, replacement, [{'verdict': 'yes'}, {'verdict': 'yes'}])
    monkeypatch.setattr(m.generator, 'check_support',
                        lambda claims, *args: [c.text != CLAIMS[0]['text'] for c in claims])
    res = m._ask(QUESTION, 'youth', 'en', answer_request=REQUEST)
    assert res.claims == CLAIMS[1:] and res.as_list and res.follow_up == FIRST_FOLLOW_UP
    # The original early review saw a subsequently rejected sentence, so it must
    # be invalidated even though the later rewrite was also not adopted.
    assert len(seen['drafts']) == len(seen['reviews']) == 2
    assert CLAIMS[0]['text'] not in seen['reviews'][1]


def test_out_of_range_missing_parts_do_not_invent_a_gap_or_a_rewrite(monkeypatch):
    m, seen = engine(monkeypatch, {}, [{'verdict': 'yes', 'missing_parts': [5, 999, -1, True]}])
    res = m._ask(QUESTION, 'youth', 'en', answer_request=REQUEST)
    assert res.claims == CLAIMS and TEXT['en']['partial_note'] not in res.note
    assert len(seen['drafts']) == len(seen['reviews']) == 1
