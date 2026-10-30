"""Draft-only exploratory comparison; offline by default, no retries or fallback."""
import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / 'data/private/measurement'


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate(plan):
    if plan.get('scope') != 'exploratory_draft_only_not_release':
        raise ValueError('not an exploratory draft plan')
    systems = plan['systems']
    if set(systems) != {'current', 'candidate'}:
        raise ValueError('need exactly two variants')
    if not systems['candidate'].startswith(systems['current']):
        raise ValueError('candidate must preserve the original prompt')
    if any(sha(value) != plan['system_sha256'][name] for name, value in systems.items()):
        raise ValueError('changed system prompt')
    cases = plan['cases']
    if not 1 <= len(cases) <= 4 or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('need 1-4 distinct cases; at most eight HTTP attempts')
    for case in cases:
        if sha(case['user']) != case['user_sha256']:
            raise ValueError('changed user prompt')
        if sorted(case['order']) != ['candidate', 'current']:
            raise ValueError('each case must compare both variants once')
    if plan.get('max_tokens') != 8192 or plan.get('temperature') != 0:
        raise ValueError('generation settings differ from the planned comparison')
    return len(cases) * 2


def cost_of(usage):
    value = usage.get('cost')
    if value is None or isinstance(value, bool):
        return None
    try:
        value = Decimal(str(value))
        return value if value.is_finite() and value >= 0 else None
    except Exception:
        return None


def run(plan, send, emit, stop_after_usd):
    validate(plan)
    spent = Decimal('0')
    for case in plan['cases']:
        for variant in case['order']:
            emit({'kind': 'attempt_started', 'case_id': case['id'], 'variant': variant})
            started = time.perf_counter_ns()
            try:
                data = send(plan['systems'][variant], case['user'])
                seconds = (time.perf_counter_ns() - started) / 1e9
                usage = data.get('usage') or {}
                cost = cost_of(usage)
                emit({'kind': 'response', 'case_id': case['id'], 'variant': variant,
                      'seconds': seconds, 'response': data,
                      'reported_cost_usd': str(cost) if cost is not None else None})
                if cost is None:
                    return 'stopped_unknown_cost'
                spent += cost
                if spent >= stop_after_usd:
                    return 'stopped_after_reported_cost_threshold'
                choice = data['choices'][0]
                raw = choice['message'].get('content') or ''
                parsed = json.loads(raw)
                if choice.get('finish_reason') != 'stop' or not isinstance(parsed, dict):
                    return 'stopped_incomplete_or_invalid_draft'
                if type(parsed.get('abstain')) is not bool or not isinstance(parsed.get('claims'), list):
                    return 'stopped_invalid_draft_fields'
            except Exception as exc:
                emit({'kind': 'failure', 'case_id': case['id'], 'variant': variant,
                      'error_class': type(exc).__name__,
                      'seconds': (time.perf_counter_ns() - started) / 1e9,
                      'cost_status': 'unknown_unless_recorded_in_response'})
                return 'stopped_failure_no_retry'
    return 'completed_draft_only_no_quality_verdict'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--stop-after-usd', default='0.10')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    count = validate(plan)
    threshold = Decimal(args.stop_after_usd)
    if not threshold.is_finite() or threshold <= 0:
        parser.error('stop-after-usd must be positive and finite')
    if not args.execute:
        print('OFFLINE PLAN CHECK: HTTP attempts:', count)
        print('Plan SHA256:', hashlib.sha256(raw).hexdigest())
        print('No network calls. Human review remains pending; not a release evaluation.')
        return
    base = os.environ['OPENAI_COMPAT_BASE_URL'].rstrip('/')
    parts = urlsplit(base)
    if parts.scheme != 'https' or parts.hostname != 'openrouter.ai' or parts.username or parts.password:
        parser.error('requires the configured HTTPS OpenRouter endpoint')
    model = os.environ['OPENAI_COMPAT_MODEL']
    if model != plan['requested_model']:
        parser.error('configured model differs from the plan')
    if os.environ.get('OPENAI_COMPAT_REASONING', '').strip().lower() not in ('off', 'none', 'false', '0'):
        parser.error('the plan requires reasoning off')
    path = (args.out or PRIVATE / f'nonredundant-draft-{time.time_ns()}.jsonl').resolve()
    if not path.is_relative_to(PRIVATE.resolve()):
        parser.error('output must stay inside data/private/measurement')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    import httpx  # only the explicit paid execution path needs this dependency
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        def emit(event):
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'kind': 'plan', 'plan_sha256': hashlib.sha256(raw).hexdigest(),
              'plan': plan, 'post_response_stop_usd': str(threshold),
              'budget_limit': 'post-response threshold; not a hard provider spending cap'})
        with httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=180,
                          trust_env=False, follow_redirects=False) as client:
            def send(system, user):
                response = client.post(base + '/chat/completions',
                    headers={'authorization': 'Bearer ' + os.environ['OPENAI_COMPAT_API_KEY']},
                    json={'model': model, 'temperature': 0, 'max_tokens': 8192,
                          'reasoning': {'enabled': False},
                          'response_format': {'type': 'json_object'},
                          'messages': [{'role': 'system', 'content': system},
                                       {'role': 'user', 'content': user}]})
                response.raise_for_status()
                return response.json()
            status = run(plan, send, emit, threshold)
        emit({'kind': 'finished', 'status': status})
    print(status)
    print('Saved:', path)


if __name__ == '__main__':
    main()
