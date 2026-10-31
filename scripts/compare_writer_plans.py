"""Pinned, draft-only development comparison. Offline unless --execute is explicit.

Uses the actual ModelGenerator request builder. No retrieval, checker, retries,
provider fallback, automatic quality grade, or production endpoint is invoked.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PRIVATE = ROOT / 'data/private/measurement'
GENERATION = {'temperature': 0, 'max_tokens': 8192, 'reasoning': {'enabled': False},
              'response_format': {'type': 'json_object'}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def revision():
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((ROOT / 'muhawir').glob('*.py'))})


def payload(case, candidate):
    from muhawir.corpus import Passage
    from muhawir.generate import ModelGenerator
    from muhawir.writer_plan import requirements_for
    captured = []
    def capture(system, user, schema):
        captured.append({'system': system, 'user': user, 'schema': schema})
        # Offline capture only: no response from this stub is included in any measurement.
        return json.dumps({'abstain': True, 'claims': [], 'plan': [
            {'requirement': i, 'passage_ids': []}
            for i, _ in enumerate(requirements_for(case['question'], case.get('answer_request')))]})
    generator = ModelGenerator([('offline-writer-capture', capture)], evidence_plan=candidate)
    generator.generate(case['question'], [Passage(**p) for p in case['passages']],
                       case['style'], case['lang'], answer_request=case.get('answer_request'))
    if len(captured) != 1:
        raise ValueError('writer must produce exactly one request')
    return captured[0]


def prepare(saved, style, repeats):
    """Only full saved contexts qualify; final answer citations are not writer context."""
    from muhawir.corpus import Passage
    if type(repeats) is not int or not 1 <= repeats <= 4:
        raise ValueError('one context, 1-4 pairs, at most eight HTTP attempts')
    if not isinstance(saved.get('question'), str) or not saved['question'].strip():
        raise ValueError('missing saved question')
    if not isinstance(saved.get('passages'), list) or not saved['passages']:
        raise ValueError('requires full saved writer passages, not answer/source cards')
    passages = [Passage(**p) for p in saved['passages']]
    if len({p.id for p in passages}) != len(passages) or any(not p.id or not p.text for p in passages):
        raise ValueError('empty or duplicated source context')
    if style not in {'kids', 'youth', 'extended', 'newcomer'}:
        raise ValueError('explicit supported style required')
    if not isinstance(saved.get('model'), str) or not saved['model'].strip():
        raise ValueError('missing requested model')
    case = {'question': saved['question'], 'style': style, 'lang': 'ar',
            'passages': [asdict(p) for p in passages],
            'answer_request': saved.get('answer_request')}
    plan = {'version': 1, 'scope': 'paired_writer_development_not_release',
            'input_sha256': digest(saved), 'code_sha256': revision(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'model': saved['model'], 'generation': GENERATION, 'case': case, 'repeats': repeats,
            'human_review': 'pending', 'payloads': {name: payload(case, candidate)
                for name, candidate in [('current', False), ('candidate', True)]}}
    plan['payload_sha256'] = digest(plan['payloads'])
    return plan


def validate(plan):
    if plan.get('version') != 1 or plan.get('scope') != 'paired_writer_development_not_release':
        raise ValueError('unsupported comparison plan')
    if type(plan.get('repeats')) is not int or not 1 <= plan['repeats'] <= 4:
        raise ValueError('at most eight HTTP attempts')
    if plan['code_sha256'] != revision() or plan['runner_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('code changed; prepare and review a new plan')
    if plan.get('generation') != GENERATION:
        raise ValueError('generation settings changed')
    expected = {name: payload(plan['case'], candidate)
                for name, candidate in [('current', False), ('candidate', True)]}
    if plan['payloads'] != expected or plan['payload_sha256'] != digest(expected):
        raise ValueError('writer payloads or evidence changed')
    if not isinstance(plan.get('model'), str) or not plan['model'].strip():
        raise ValueError('missing requested model')
    return 2 * plan['repeats']


def reported_cost(data):
    usage = data.get('usage') if isinstance(data, dict) else None
    value = usage.get('cost') if isinstance(usage, dict) else None
    if value is None or isinstance(value, bool):
        return None
    try:
        cost = Decimal(str(value))
        return cost if cost.is_finite() and cost >= 0 else None
    except (InvalidOperation, ValueError):
        return None


def inspect_draft(data, variant, case):
    from muhawir.corpus import Passage
    from muhawir.generate import load_json, parse_draft
    from muhawir.writer_plan import read_plan, requirements_for
    choices = data.get('choices')
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get('message') or {}
    if (choice.get('finish_reason') != 'stop' or not isinstance(message, dict)
            or message.get('refusal') or message.get('tool_calls') or message.get('function_call')):
        raise ValueError('incomplete model response')
    raw = message.get('content')
    parsed = load_json(raw)
    if not isinstance(parsed, dict) or type(parsed.get('abstain')) is not bool:
        raise ValueError('invalid draft fields')
    if not isinstance(parsed.get('claims'), list) or not isinstance(parsed.get('views', []), list):
        raise ValueError('invalid claim lists')
    claims = parse_draft(raw)
    # Never silently drop malformed claims when reporting evaluation success.
    if not parsed['abstain'] and len(claims) != len(parsed['claims']) + len(parsed.get('views', [])):
        raise ValueError('unparsed draft claims')
    if (parsed['abstain'] and (parsed['claims'] or parsed.get('views'))
            or not parsed['abstain'] and not claims):
        raise ValueError('inconsistent abstention or empty answer')
    allowed = {p['id'] for p in case['passages']}
    if any(not c.passage_ids or not set(c.passage_ids) <= allowed for c in claims):
        raise ValueError('unknown or missing citation')
    plan = None
    if variant == 'candidate':
        plan = read_plan(parsed, len(requirements_for(case['question'], case.get('answer_request'))),
                         [Passage(**p) for p in case['passages']], claims)
        if plan is None:
            raise ValueError('unusable evidence plan')
    return {'abstain': parsed['abstain'], 'claims': [asdict(c) for c in claims],
            'unresolved_requirements': sum(not row.passage_ids for row in plan) if plan is not None else None,
            'semantic_quality': 'unreviewed'}


def run(plan, send, emit, threshold):
    validate(plan)
    if not isinstance(threshold, Decimal) or not threshold.is_finite() or threshold <= 0:
        raise ValueError('positive finite post-response stop required')
    total = Decimal(0)
    for trial in range(plan['repeats']):
        order = ['current', 'candidate'] if trial % 2 == 0 else ['candidate', 'current']
        for variant in order:
            emit({'kind': 'started', 'trial': trial, 'variant': variant})
            started = time.perf_counter_ns()
            try:
                data = send(plan['payloads'][variant], plan['model'])
            except Exception as exc:
                emit({'kind': 'failure', 'trial': trial, 'variant': variant,
                      'error_class': type(exc).__name__, 'reported_cost_usd': None,
                      'seconds': (time.perf_counter_ns() - started) / 1e9})
                return 'stopped_transport_failure_no_retry'
            seconds = (time.perf_counter_ns() - started) / 1e9
            cost = reported_cost(data)
            emit({'kind': 'response', 'trial': trial, 'variant': variant, 'seconds': seconds,
                  'reported_cost_usd': str(cost) if cost is not None else None, 'response': data})
            if cost is None:
                return 'stopped_unknown_cost'
            total += cost
            try:
                checked = inspect_draft(data, variant, plan['case'])
            except Exception as exc:
                emit({'kind': 'invalid_draft', 'trial': trial, 'variant': variant,
                      'error_class': type(exc).__name__})
                return 'stopped_invalid_draft_no_retry'
            emit({'kind': 'parsed', 'trial': trial, 'variant': variant, **checked})
            if total >= threshold:
                return 'stopped_post_response_threshold'
    return 'completed_draft_only_quality_pending'


def analyze(events):
    """Describe paired recordings and build a separate blinded review; never infer accuracy."""
    if not events or events[0].get('kind') != 'plan':
        raise ValueError('missing recorded plan')
    recorded = events[0]
    plan = recorded['plan']
    if recorded['plan_sha256'] != digest(plan) or plan['payload_sha256'] != digest(plan['payloads']):
        raise ValueError('recorded plan changed')
    if plan.get('scope') != 'paired_writer_development_not_release':
        raise ValueError('not a writer comparison')
    starts, responses, parsed = set(), {}, {}
    for event in events[1:]:
        kind = event.get('kind')
        if kind not in {'started', 'response', 'parsed', 'failure', 'invalid_draft', 'finished'}:
            raise ValueError('unknown recording event')
        if kind == 'finished':
            continue
        key = (event['trial'], event['variant'])
        if (type(key[0]) is not int or not 0 <= key[0] < plan['repeats']
                or key[1] not in {'current', 'candidate'}):
            raise ValueError('unexpected trial or variant')
        if kind == 'started':
            if key in starts:
                raise ValueError('duplicate started attempt')
            starts.add(key)
        elif key not in starts:
            raise ValueError('result without a started attempt')
        if kind == 'response':
            if key in responses:
                raise ValueError('duplicate response')
            if type(event.get('seconds')) not in (int, float) or not 0 <= event['seconds'] < float('inf'):
                raise ValueError('invalid response duration')
            responses[key] = event
        if kind == 'parsed':
            if key not in responses or key in parsed:
                raise ValueError('missing response or duplicate parsed draft')
            # Re-parse the raw recording, never trust a precomputed approval/count.
            parsed[key] = inspect_draft(responses[key]['response'], key[1], plan['case'])
    pairs, review, mapping = [], [], []
    for trial in range(plan['repeats']):
        pair = {'trial': trial, 'both_responses_present': all((trial, v) in responses for v in ['current', 'candidate'])}
        for variant in ['current', 'candidate']:
            key = (trial, variant)
            if key not in responses:
                continue
            event, data = responses[key], responses[key]['response']
            usage = (data.get('usage') or {}) if isinstance(data, dict) else {}
            if not isinstance(usage, dict): usage = {}
            cost = reported_cost(data)
            details = usage.get('prompt_tokens_details')
            cached = details.get('cached_tokens') if isinstance(details, dict) else None
            pair[variant] = {'seconds': event['seconds'], 'reported_cost_usd': str(cost) if cost is not None else None,
                'provider': data.get('provider') if isinstance(data, dict) else None,
                'model': data.get('model') if isinstance(data, dict) else None,
                'tokens_in': usage.get('prompt_tokens'), 'tokens_out': usage.get('completion_tokens'),
                'cached_tokens': cached, 'draft_parsed': key in parsed}
            if key in parsed:
                slot = (['A', 'B'] if int(recorded['plan_sha256'][trial], 16) % 2 else ['B', 'A'])[['current', 'candidate'].index(variant)]
                review_id = f"P{trial + 1}-{slot}"
                review.append({'id': review_id, 'question': plan['case']['question'],
                    'style': plan['case']['style'], 'claims': parsed[key]['claims'],
                    'abstain': parsed[key]['abstain'], 'verdicts': {
                        'meaning_supported': None, 'conditions_preserved': None,
                        'attribution_and_grade_correct': None, 'question_covered': None,
                        'unnecessary_repetition': None, 'notes': ''}})
                mapping.append({'id': review_id, 'variant': variant, 'trial': trial})
        if pair['both_responses_present']:
            a, b = pair['current'], pair['candidate']
            pair['same_reported_route'] = all(a[k] and a[k] == b[k] for k in ['provider', 'model'])
            pair['candidate_minus_current_seconds'] = b['seconds'] - a['seconds']
            pair['cached_tokens_differ'] = a['cached_tokens'] != b['cached_tokens']
            pair['cache_counts_known'] = all(type(p['cached_tokens']) is int for p in [a, b])
            pair['candidate_minus_current_reported_usd'] = (
                str(Decimal(b['reported_cost_usd']) - Decimal(a['reported_cost_usd']))
                if all(p['reported_cost_usd'] is not None for p in [a, b]) else None)
        pairs.append(pair)
    costs = [reported_cost(e['response']) for e in responses.values()]
    report = {'scope': 'draft_only_descriptive_development', 'attempts_started': len(starts),
        'responses': len(responses), 'started_without_response': len(starts - responses.keys()),
        'known_reported_cost_usd': str(sum((c for c in costs if c is not None), Decimal(0))),
        'unknown_cost_attempts': len(starts) - sum(c is not None for c in costs),
        'quality_status': 'pending_independent_review', 'pairs': pairs, 'review_mapping': mapping,
        'planned_attempts': plan['repeats'] * 2,
        'finished_status': events[-1].get('status') if events[-1].get('kind') == 'finished' else 'incomplete_recording',
        'limitations': ['One previously exposed question; repeated trials are not independent questions.',
            'Draft calls only; excludes retrieval, checkers, rewrite and production network/browser latency.',
            'Routing, caching and time drift can confound timings; paired differences are descriptive.',
            'No general accuracy, population p95 or release approval is inferred.']}
    return report, {'scope': 'blinded_draft_review', 'evidence': plan['case']['passages'],
                    'answer_request': plan['case'].get('answer_request'),
                    'items': sorted(review, key=lambda r: r['id'])}


def write_private(path, value):
    path = path.resolve()
    if not path.is_relative_to(PRIVATE.resolve()):
        raise ValueError('output must stay in data/private/measurement')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='saved full context for preparation, or prepared plan for execution')
    parser.add_argument('--style', choices=['kids', 'youth', 'extended', 'newcomer'])
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--analyze', action='store_true', help='offline analysis of a saved result JSONL')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--stop-after-usd')
    args = parser.parse_args()
    if args.analyze:
        if args.execute:
            parser.error('analysis never executes model requests')
        events = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        report, review = analyze(events)
        stamp = time.time_ns()
        print('Report:', write_private(PRIVATE / f'writer-report-{stamp}.json', report))
        print('Blinded review:', write_private(PRIVATE / f'writer-review-{stamp}.json', review))
        print('Quality pending independent review; no release verdict.')
        return
    saved = json.loads(args.input.read_text())
    if not args.execute:
        if not args.style:
            parser.error('preparation requires --style; it is not inferred from an old answer')
        plan = prepare(saved, args.style, args.pairs)
        path = write_private(PRIVATE / f'writer-plan-{time.time_ns()}.json', plan)
        print('Offline only. Planned HTTP attempts:', validate(plan))
        print('Plan SHA256:', digest(plan))
        print('Saved:', path)
        return
    validate(saved)
    if args.plan_sha256 != digest(saved):
        parser.error('--execute requires the exact reviewed plan SHA256')
    try:
        threshold = Decimal(args.stop_after_usd or '')
    except InvalidOperation:
        parser.error('execution requires --stop-after-usd')
    if not threshold.is_finite() or threshold <= 0:
        parser.error('positive finite post-response threshold required; this is not a provider spending cap')
    base = os.environ.get('OPENAI_COMPAT_BASE_URL', '').rstrip('/')
    url = urlsplit(base)
    if url.scheme != 'https' or url.hostname != 'openrouter.ai' or url.username or url.password or url.query or url.fragment:
        parser.error('execution requires a configured HTTPS OpenRouter endpoint')
    if os.environ.get('OPENAI_COMPAT_MODEL') != saved['model'] or not os.environ.get('OPENAI_COMPAT_API_KEY'):
        parser.error('configured model/key missing or model differs from plan')
    import httpx
    path = PRIVATE / f'writer-results-{time.time_ns()}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        def emit(event):
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        emit({'kind': 'plan', 'plan_sha256': digest(saved), 'plan': saved,
              'post_response_threshold_usd': str(threshold),
              'limit_note': 'may exceed threshold by last response; no hard provider cap'})
        with httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=180,
                          trust_env=False, follow_redirects=False) as client:
            def send(request, model):
                response = client.post(base + '/chat/completions',
                    headers={'authorization': 'Bearer ' + os.environ['OPENAI_COMPAT_API_KEY']},
                    json={'model': model, **saved['generation'],
                          'messages': [{'role': 'system', 'content': request['system']},
                                       {'role': 'user', 'content': request['user']}]})
                response.raise_for_status()
                return response.json()
            status = run(saved, send, emit, threshold)
        emit({'kind': 'finished', 'status': status})
    print(status)
    print('Saved:', path)
    print('No semantic accuracy, production latency or release verdict is inferred.')


if __name__ == '__main__':
    main()
