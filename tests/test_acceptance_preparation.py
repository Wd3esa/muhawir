"""No model packages: synthetic evidence and temporary private paths only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_acceptance_review as prep
from scripts.prepare_quality_review import validate


class AcceptancePreparationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {'cases': [dict(
            id='R01', cluster_id='test', split='development', question='Color?',
            style='youth', lang='ar', evidence_variant='reviewer_frozen_context')]}
        self.card = {'id': 'fake:1', 'source': 'Test book', 'location': 'p1',
                     'text': 'The box is blue if the lamp is on.'}
        self.report = {'results': [{'question': 'Color?', 'present_gold': [self.card],
                                  'modes': {'ann': {'hits': [self.card]}}}]}
        self.pilot = {'question': 'Color?', 'passages': [dict(
            id='fake:1', source_id='test', location='p1', text=self.card['text'])],
            'results': {'current': {'parsed': {'claims': [dict(
                text='The box is always blue.', passage_ids=['fake:1'])], 'views': []}}}}

    def test_deduplicates_snapshots_but_never_approves_observed_claims(self):
        before = deepcopy(self.pilot)
        bundle = prep.prepare(self.catalog, [self.report], [self.pilot])
        c = bundle['cases'][0]
        self.assertEqual(len(c['candidate_passages']), 1)
        self.assertEqual(c['candidate_passages'][0]['source'], 'Test book')
        self.assertIsNone(c['claims'][0]['expected_supported'])
        self.assertEqual(c['reviews'], [])
        self.assertEqual(self.pilot, before)
        with self.assertRaisesRegex(ValueError, 'awaits human review'):
            validate(bundle)

    def test_ablation_does_not_inherit_normal_evidence_or_claims(self):
        self.catalog['cases'][0]['evidence_variant'] = 'reviewer_prepared_insufficient'
        c = prep.prepare(self.catalog, [self.report], [self.pilot])['cases'][0]
        self.assertEqual(c['candidate_passages'], [])
        self.assertEqual(c['claims'], [])
        self.assertEqual(c['evidence_state'], 'reviewer_ablation_pending')

    def test_report_conflict_is_rejected(self):
        other = deepcopy(self.report)
        other['results'][0]['present_gold'][0]['text'] = 'Changed text'
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            prep.prepare(self.catalog, [self.report, other], [])

    def test_pilot_conflict_is_rejected(self):
        self.pilot['passages'][0]['location'] = 'p2'
        with self.assertRaisesRegex(ValueError, 'conflict'):
            prep.prepare(self.catalog, [self.report], [self.pilot])

    def test_no_fuzzy_matching_or_automatic_abstention_for_missing_snapshots(self):
        self.catalog['cases'][0]['question'] = 'Other color question?'
        c = prep.prepare(self.catalog, [self.report], [])['cases'][0]
        self.assertEqual(c['evidence_state'], 'missing')
        self.assertIsNone(c['expected_status'])

    def test_private_output_refuses_overwrite_and_outside_path(self):
        with tempfile.TemporaryDirectory() as folder:
            private = Path(folder) / 'private'
            with patch.object(prep, 'PRIVATE', private):
                path = private / 'review.json'
                prep.write_private(path, {'test': True})
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(json.loads(path.read_text()), {'test': True})
                with self.assertRaises(FileExistsError):
                    prep.write_private(path, {})
                with self.assertRaises(ValueError):
                    prep.write_private(Path(folder) / 'outside.json', {})
