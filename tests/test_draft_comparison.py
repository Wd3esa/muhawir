"""Fake sends only; comparison controls do not grade religious correctness."""
from copy import deepcopy
from decimal import Decimal
import json
import unittest

from scripts.run_draft_comparison import cost_of, run, sha, validate


def plan():
    systems = {'current': 'original', 'candidate': 'original plus guard'}
    return {'scope': 'exploratory_draft_only_not_release', 'systems': systems,
            'system_sha256': {k: sha(v) for k, v in systems.items()},
            'temperature': 0, 'max_tokens': 8192,
            'cases': [{'id': str(i), 'user': 'same evidence ' + str(i),
                       'user_sha256': sha('same evidence ' + str(i)),
                       'order': ['current', 'candidate'] if i % 2 == 0 else ['candidate', 'current']}
                      for i in range(4)]}


def response(cost='0.001', finish='stop'):
    return {'usage': {'cost': cost}, 'choices': [{'finish_reason': finish,
            'message': {'content': json.dumps({'abstain': False, 'claims': []})}}]}


class DraftComparisonTests(unittest.TestCase):
    def test_eight_sends_balanced_orders_identical_user_context_and_logged_usage(self):
        p = plan(); sent = []; events = []
        def send(system, user):
            sent.append((system, user))
            return response()
        status = run(p, send, events.append, Decimal('0.10'))
        self.assertTrue(status.startswith('completed'))
        self.assertEqual(len(sent), 8)
        self.assertTrue(all(sent[i][1] == sent[i+1][1] for i in range(0, 8, 2)))
        self.assertEqual([s[0] for s in sent[:4]], ['original', 'original plus guard', 'original plus guard', 'original'])
        self.assertEqual(sum(e['kind'] == 'response' for e in events), 8)

    def test_changed_context_or_more_than_eight_attempts_rejected_before_send(self):
        p = plan(); p['cases'][0]['user'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'changed user'):
            validate(p)
        p = plan(); p['cases'].append(deepcopy(p['cases'][0]))
        with self.assertRaises(ValueError):
            validate(p)

    def test_unknown_cost_stops_without_continuing_or_treating_it_as_zero(self):
        sent = []
        def send(*_):
            sent.append(1)
            return response(None)
        self.assertEqual(run(plan(), send, lambda _: None, Decimal('0.10')), 'stopped_unknown_cost')
        self.assertEqual(len(sent), 1)
        for value in [None, True, 'NaN', '-1', 'Infinity']:
            self.assertIsNone(cost_of({'cost': value}))

    def test_error_is_logged_once_and_never_retried(self):
        sent = []; events = []
        def send(*_):
            sent.append(1)
            raise TimeoutError('private details')
        self.assertEqual(run(plan(), send, events.append, Decimal('0.10')), 'stopped_failure_no_retry')
        self.assertEqual(len(sent), 1)
        self.assertNotIn('private details', json.dumps(events))
        self.assertEqual(events[-1]['error_class'], 'TimeoutError')

    def test_budget_is_explicitly_a_post_response_stop(self):
        sent = []
        def send(*_):
            sent.append(1)
            return response('0.11')
        self.assertEqual(run(plan(), send, lambda _: None, Decimal('0.10')),
                         'stopped_after_reported_cost_threshold')
        self.assertEqual(len(sent), 1)

    def test_truncation_or_invalid_json_stops_and_preserves_response(self):
        for data in [response(finish='length'), {'usage': {'cost': '0.001'},
                     'choices': [{'finish_reason': 'stop', 'message': {'content': 'bad'}}]}]:
            events = []; sent = []
            def send(*_):
                sent.append(1)
                return data
            self.assertTrue(run(plan(), send, events.append, Decimal('0.10')).startswith('stopped'))
            self.assertEqual(len(sent), 1)
            self.assertTrue(any(e['kind'] == 'response' for e in events))
