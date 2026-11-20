"""Assemble private, UNREVIEWED evidence from existing files; no model/network calls."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / 'data/private/measurement'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(catalog, reports, pilots):
    cases = deepcopy(catalog['cases'])
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('empty catalog or duplicate case ID')
    by_question = {}
    known = {}

    def register(p):
        card = {k: p[k] for k in ('id', 'source', 'location', 'text')}
        if any(not isinstance(v, str) or not v.strip() for v in card.values()):
            raise ValueError('incomplete source snapshot')
        card['text_sha256'] = digest(card['text'].encode())
        if card['id'] in known and known[card['id']] != card:
            raise ValueError('conflicting source snapshot: ' + card['id'])
        known[card['id']] = card
        return card

    for report in reports:
        for row in report['results']:
            cards = by_question.setdefault(row['question'], {})
            candidates = list(row.get('present_gold', []))
            for mode in row.get('modes', {}).values():
                candidates.extend(mode.get('hits', []))
            for p in candidates:
                cards[p['id']] = register(p)

    observed = {}
    for pilot in pilots:
        question = pilot['question']
        cards = by_question.setdefault(question, {})
        for p in pilot['passages']:
            existing = known.get(p['id'])
            if existing and (existing['text'] != p['text'] or existing['location'] != p['location']):
                raise ValueError('pilot/report snapshot conflict: ' + p['id'])
            card = existing or register({
                'id': p['id'], 'source': p['source_id'],
                'location': p['location'], 'text': p['text']})
            cards[p['id']] = card
        for variant, result in pilot['results'].items():
            claims = list((result.get('parsed') or {}).get('claims', []))
            claims += (result.get('parsed') or {}).get('views', [])
            for claim in claims:
                if not isinstance(claim.get('text'), str) or not claim.get('passage_ids'):
                    continue
                observed.setdefault(question, []).append({
                    'text': claim['text'], 'passage_ids': claim['passage_ids'],
                    'expected_supported': None, 'expected_relevant': None,
                    'defect': None, 'rationale': '', 'evidence_quotes': [],
                    'origin': 'observed_draft_not_truth', 'variant': variant})

    for case in cases:
        # Insufficiency/ablation evidence must be prepared separately by a reviewer.
        ablation = case.get('evidence_variant') == 'reviewer_prepared_insufficient'
        case['candidate_passages'] = [] if ablation else list(by_question.get(case['question'], {}).values())
        case['claims'] = [] if ablation else deepcopy(observed.get(case['question'], []))
        case['expected_status'] = None
        case['required_answer_parts'] = []
        case['reviews'] = []
        case['source_labels_require_review'] = True
        case['evidence_state'] = ('reviewer_ablation_pending' if ablation else
                                  'candidate_snapshots_only' if case['candidate_passages'] else 'missing')
    return {'schema_version': 1, 'purpose': 'development_grounding_review',
            'status': 'pending_human_review_not_quality_results', 'cases': cases}


def write_private(path, bundle):
    path = path.resolve()
    if not path.is_relative_to(PRIVATE.resolve()):
        raise ValueError('output must stay inside data/private/measurement')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'evaluation/reliability-first/cases.json')
    parser.add_argument('--report', type=Path, action='append', default=[])
    parser.add_argument('--pilot', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, default=PRIVATE / 'acceptance-review.json')
    args = parser.parse_args()
    files = [args.catalog, *args.report, *args.pilot]
    raw = {p: p.read_bytes() for p in files}
    bundle = prepare(json.loads(raw[args.catalog]),
                     [json.loads(raw[p]) for p in args.report],
                     [json.loads(raw[p]) for p in args.pilot])
    bundle['input_sha256'] = {p.name: digest(value) for p, value in raw.items()}
    if len(bundle['input_sha256']) != len(raw):
        raise ValueError('input basenames must be unique for provenance')
    write_private(args.out, bundle)
    for c in bundle['cases']:
        print(c['id'], 'snapshots:', len(c['candidate_passages']),
              'unreviewed claims:', len(c['claims']), c['evidence_state'])
    print('All verdicts pending; snapshots are not approved evidence. No paid calls.')
    print('Saved:', args.out.resolve())


if __name__ == '__main__':
    main()
