"""Build a private exploratory comparison plan from pinned local evidence; no API calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=ROOT / 'data/muhawir.db')
    args = parser.parse_args()
    from muhawir.generate import SYSTEM_PROMPT, build_user_prompt
    from muhawir.store import SqliteCorpus
    spec = json.loads((ROOT / 'experiments/nonredundant-draft/comparison-spec.json').read_text())
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    candidate = (ROOT / 'experiments/nonredundant-draft/candidate.txt').read_text()
    assert sha(SYSTEM_PROMPT) == spec['system_sha256']['current'], 'baseline prompt changed'
    assert sha(candidate) == spec['system_sha256']['candidate'], 'candidate changed'
    db = args.db.resolve()
    before = db.stat()
    digest = hashlib.sha256()
    with db.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == spec['corpus_sha256'], 'database differs from baseline'
    corpus = SqliteCorpus(db)
    cases = []
    for i, case in enumerate(spec['cases']):
        passages = []
        for expected in case['evidence']:
            p = corpus.passage(expected['id'])
            assert p is not None, expected['id']
            assert sha(p.text) == expected['text_sha256'], 'changed evidence: ' + p.id
            assert p.location == expected['location'] and p.source_id == expected['source_id'], 'changed metadata: ' + p.id
            passages.append(p)
        user = build_user_prompt(case['question'], passages, case['style'], 'ar', False)
        cases.append({**case, 'user': user, 'user_sha256': sha(user),
                      'order': ['current', 'candidate'] if i % 2 == 0 else ['candidate', 'current']})
    after = db.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'database changed during plan preparation'
    plan = {'scope': 'exploratory_draft_only_not_release', 'human_review_status': 'pending',
            'requested_model': spec['requested_model'], 'temperature': 0, 'max_tokens': 8192,
            'corpus_sha256': digest.hexdigest(), 'systems': {'current': SYSTEM_PROMPT, 'candidate': candidate},
            'system_sha256': spec['system_sha256'], 'cases': cases}
    folder = ROOT / 'data/private/measurement'
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / f'nonredundant-plan-{time.time_ns()}.json'
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as out:
        json.dump(plan, out, ensure_ascii=False, indent=2)
    print(f'Prepared {len(cases)} cases; {2 * len(cases)} HTTP attempts only if explicitly executed later.')
    print('No model calls. Human reference review and paid-run authorization remain pending.')
    print('Saved:', path)


if __name__ == '__main__':
    main()
