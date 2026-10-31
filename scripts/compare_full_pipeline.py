"""Isolated full-pipeline paired evaluation; offline validation is the default.

Two writers in the same checkout, real retrieval/checking/rewrites and API
serialization. Never production deployment, browser/network latency or load testing.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.compare_writer_plans import digest, revision, write_private, reported_cost
from scripts.analyze_request_traces import read_lines, summarize

LABELS = ('meaning_supported', 'conditions_preserved', 'attribution_and_grade_correct',
          'question_covered', 'behavior_correct', 'critical_error')


def validate_manifest(manifest):
    if manifest.get('version') != 1 or manifest.get('partition') not in {'development', 'declared_holdout'}:
        raise ValueError('explicit development/declared_holdout partition required')
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 1000:
        raise ValueError('need 1-1000 predefined cases')
    ids, questions, clusters = set(), set(), set()
    for case in cases:
        if not isinstance(case.get('id'), str) or not case['id'] or case['id'] in ids:
            raise ValueError('distinct case IDs required')
        q = case.get('question')
        if not isinstance(q, str) or not q.strip() or ' '.join(q.split()).casefold() in questions:
            raise ValueError('distinct nonempty questions required')
        ids.add(case['id']); questions.add(' '.join(q.split()).casefold())
        if case.get('style') not in {'kids', 'youth', 'extended', 'newcomer'} or case.get('lang') not in {'ar', 'en', 'ur'}:
            raise ValueError('explicit supported style/language required')
        reference = case.get('reference')
        if not isinstance(reference, dict) or reference.get('expected_behavior') not in {'answer', 'abstain', 'refer', 'clarify', 'refuse'}:
            raise ValueError('expected behavior must be fixed before execution')
        if not isinstance(reference.get('requirements'), list) or not reference['requirements']:
            raise ValueError('predefined reference requirements required')
        if not all(isinstance(x, str) and x.strip() for x in reference['requirements']):
            raise ValueError('reference requirements must be nonempty strings')
        if manifest['partition'] == 'declared_holdout':
            cluster = case.get('cluster_id')
            if not isinstance(cluster, str) or not cluster.strip() or cluster in clusters:
                raise ValueError('holdout needs one independent question per declared cluster')
            clusters.add(cluster)
            if (reference.get('review_status') != 'approved' or not isinstance(reference.get('reviewer'), str)
                    or not reference['reviewer'].strip()):
                raise ValueError('holdout reference needs prior independent approval')
            if not isinstance(reference.get('evidence'), list):
                raise ValueError('reference evidence must be a list')
            if not reference['evidence'] and reference['expected_behavior'] == 'answer':
                raise ValueError('answer references need source evidence, not just an expected status')
            for evidence in reference['evidence']:
                if (not isinstance(evidence, dict) or not isinstance(evidence.get('passage_id'), str)
                        or not evidence['passage_id'] or not isinstance(evidence.get('text_sha256'), str)
                        or not re.fullmatch('[0-9a-f]{64}', evidence['text_sha256'])
                        or not isinstance(evidence.get('passage_sha256'), str)
                        or not re.fullmatch('[0-9a-f]{64}', evidence['passage_sha256'])):
                    raise ValueError('holdout evidence needs passage IDs, text and metadata SHA256')
    policy = manifest.get('success', {})
    for field in ['critical_error_upper', 'quality_harm_margin']:
        value = policy.get(field)
        if type(value) not in (float, int) or not 0 < value < 1:
            raise ValueError('predefine positive success margins below one')
    if policy.get('primary_metric') not in {'latency', 'cost'}:
        raise ValueError('predefine one primary metric; do not select the better result afterward')
    alpha = policy.get('alpha')
    if type(alpha) not in (float, int) or not 0 < alpha < .5:
        raise ValueError('predefine alpha')
    if manifest['partition'] == 'declared_holdout':
        attestation = manifest.get('independence', {})
        if (attestation.get('not_used_in_development') is not True
                or not isinstance(attestation.get('reviewer'), str) or not attestation['reviewer'].strip()
                or not isinstance(attestation.get('development_questions_sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', attestation['development_questions_sha256'])):
            raise ValueError('holdout needs an explicit independence attestation')
    return len(cases) * 2


def upper_binomial(k, n, alpha=.05):
    """Exact one-sided Clopper-Pearson upper bound; independent case units required."""
    if type(k) is not int or type(n) is not int or not 0 <= k <= n or n < 1 or not 0 < alpha < 1:
        raise ValueError('invalid binomial inputs')
    if k == n: return 1.0
    if k == 0: return -math.expm1(math.log(alpha) / n)
    lo, hi = 0.0, 1.0
    for _ in range(70):
        p = (lo + hi) / 2
        cdf = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                  + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k + 1))
        if cdf > alpha: lo = p
        else: hi = p
    return (lo + hi) / 2


def bootstrap_ratio(pairs, alpha, seed=20261031, draws=4000):
    """Paired percentile bootstrap of ratio of means, not a p95 latency estimate."""
    rng, ratios = random.Random(seed), []
    for _ in range(draws):
        selected = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        baseline, candidate = sum(p[0] for p in selected), sum(p[1] for p in selected)
        if baseline == 0:
            if candidate: return None
            ratios.append(1.0)
        else:
            ratios.append(candidate / baseline)
    return sorted(ratios)[math.ceil((1 - alpha) * draws) - 1]


def verdict(manifest, results, labels):
    """No automatic semantic grading; fail closed on missing/repeated/unreviewed cases."""
    validate_manifest(manifest)
    expected = {(c['id'], v) for c in manifest['cases'] for v in ['current', 'candidate']}
    by_key, reviewed = {}, {}
    for result in results:
        key = (result['case_id'], result['variant'])
        if key not in expected or key in by_key:
            raise ValueError('unexpected or duplicate result')
        by_key[key] = result
    for label in labels:
        key = (label['case_id'], label['variant'])
        if key not in expected or key in reviewed:
            raise ValueError('unexpected or duplicate human label')
        reviewed[key] = label
    n = len(manifest['cases'])
    report = {'partition': manifest['partition'], 'planned_cases': n, 'complete_results': len(by_key),
              'expected_results': len(expected), 'quality_status': 'pending_independent_review',
              'general_improvement': False}
    if set(by_key) != expected:
        report['reason'] = 'incomplete paired results'; return report
    if any(not row.get('measurement_complete') or row.get('cache') != 'none' for row in results):
        report['reason'] = 'measurement gaps or cached responses'; return report
    if set(reviewed) != expected or any(not isinstance(r.get('reviewer'), str) or not r['reviewer'].strip() or any(type(r.get(k)) is not bool for k in LABELS) for r in reviewed.values()):
        report['reason'] = 'missing independent human judgments'; return report
    for key, row in reviewed.items():
        if row.get('response_sha256') != digest(by_key[key]['response']):
            raise ValueError('human label belongs to a different response')
    acceptable = lambda row: all(row[k] for k in LABELS[:-1]) and not row['critical_error']
    harms, critical, perf = 0, 0, []
    statuses = {'current': Counter(), 'candidate': Counter()}
    quality_counts = {v: Counter() for v in statuses}
    for case in manifest['cases']:
        a, b = (case['id'], 'current'), (case['id'], 'candidate')
        harms += acceptable(reviewed[a]) and not acceptable(reviewed[b])
        critical += reviewed[b]['critical_error']
        for v in statuses:
            key = (case['id'], v); label = reviewed[key]
            status = by_key[key]['response'].get('status', 'unknown')
            statuses[v][status] += 1
            quality_counts[v]['cases'] += 1
            quality_counts[v]['acceptable_cases'] += acceptable(label)
            quality_counts[v]['critical_error_cases'] += label['critical_error']
            quality_counts[v]['coverage_failures'] += not label['question_covered']
            quality_counts[v]['unsupported_answer_cases'] += status == 'answered' and not all(label[k] for k in LABELS[:3])
            quality_counts[v]['false_reject_cases'] += (case['reference']['expected_behavior'] == 'answer'
                and status != 'answered' and not label['behavior_correct'])
            quality_counts[v]['false_accept_cases'] += (case['reference']['expected_behavior'] != 'answer'
                and status == 'answered' and not label['behavior_correct'])
        field = 'client_seconds' if manifest['success']['primary_metric'] == 'latency' else 'reported_cost_usd'
        values = tuple(float(by_key[key][field]) for key in [a, b])
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid performance measurement')
        perf.append(values)
    policy = manifest['success']; alpha = policy['alpha']
    harm_upper = upper_binomial(int(harms), n, alpha)
    critical_upper = upper_binomial(int(critical), n, alpha)
    ratio_upper = bootstrap_ratio(perf, alpha)
    routes_match = all(by_key[(c['id'], 'current')].get('route_ids') is not None
                       and by_key[(c['id'], 'current')]['route_ids'] == by_key[(c['id'], 'candidate')].get('route_ids')
                       for c in manifest['cases'])
    checks = {'same_reported_routes': routes_match, 'no_observed_critical_error': critical == 0,
              'critical_error_bound_pass': critical_upper <= policy['critical_error_upper'],
              'quality_noninferiority_pass': harm_upper <= policy['quality_harm_margin'],
              'primary_metric_pass': ratio_upper is not None and ratio_upper < 1}
    report.update(quality_status='human_labels_supplied', newly_harmed_cases=int(harms),
        candidate_critical_errors=int(critical), quality_harm_upper=harm_upper,
        critical_error_upper=critical_upper, primary_ratio_upper=ratio_upper, checks=checks,
        statuses={v: dict(s) for v, s in statuses.items()},
        quality_counts={v: dict(s) for v, s in quality_counts.items()},
        general_improvement=manifest['partition'] == 'declared_holdout' and all(checks.values()),
        limitations=['Independent sampling/reviewer declarations are supplied, not verified by code.',
            'Each bound is at its stated confidence individually, not a simultaneous confidence guarantee.',
            'Conservative noninferiority bounds new harms and does not offset them against benefits.',
            'Percentile bootstrap is approximate; corpus/traffic representativeness remains a human assessment.',
            'This measures a writer change inside the full pipeline, not all branch changes versus production.',
            'No browser/network, sustained load capacity or population p95 conclusion.'])
    return report


def run(manifest, ask, emit, threshold):
    validate_manifest(manifest)
    if not isinstance(threshold, Decimal) or not threshold.is_finite() or threshold <= 0:
        raise ValueError('positive finite post-response cost stop required')
    total, results = Decimal(0), []
    for i, case in enumerate(manifest['cases']):
        for variant in (['current', 'candidate'] if i % 2 == 0 else ['candidate', 'current']):
            emit({'kind': 'started', 'case_id': case['id'], 'variant': variant})
            try:
                result = {'case_id': case['id'], 'variant': variant, **ask(variant, case)}
            except Exception as exc:
                emit({'kind': 'failure', 'case_id': case['id'], 'variant': variant, 'error_class': type(exc).__name__})
                return results, 'stopped_failure_no_question_retry'
            emit({'kind': 'result', 'result': result}); results.append(result)
            if not result.get('measurement_complete') or result.get('cache') != 'none':
                return results, 'stopped_measurement_gap_or_cache'
            cost = reported_cost({'usage': {'cost': result.get('reported_cost_usd')}})
            if cost is None: return results, 'stopped_unknown_cost'
            total += cost
            if total >= threshold: return results, 'stopped_post_response_cost_threshold'
    return results, 'completed_pending_independent_review'


class BudgetStopped(RuntimeError):
    pass


class ModelHTTPBudget:
    """Bound real model HTTP starts across parallel workers; unknown billing stops new starts."""
    def __init__(self, limit, threshold, emit, *, retain_blocker_on_exit=False):
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError('explicit model HTTP cap must be 1-10000')
        self.limit, self.threshold, self.emit = limit, threshold, emit
        self.sent, self.known_cost, self.stopped = 0, Decimal(0), False
        self.lock = threading.Lock()
        self.active = None
        self.retain_blocker_on_exit = retain_blocker_on_exit

    def __enter__(self):
        from muhawir import transport
        self.transport, self.original = transport, transport.post
        def post(url, **kwargs):
            with self.lock:
                if self.stopped or self.sent >= self.limit or self.known_cost >= self.threshold:
                    self.stopped = True
                    raise BudgetStopped('evaluation transport limit reached')
                self.sent += 1
                sequence = self.sent
                self.emit({'kind': 'model_http_started', 'sequence': sequence, 'case_variant': self.active,
                           'request_body': kwargs.get('json'), 'started_unix_ns': time.time_ns()})
            try:
                response = self.original(url, **kwargs)
                try: body = response.json()
                except ValueError: body = None
            except BaseException:
                with self.lock: self.stopped = True
                raise
            from urllib.parse import urlsplit
            cost = reported_cost(body) if urlsplit(url).hostname == 'openrouter.ai' else None
            with self.lock:
                if cost is None: self.stopped = True
                else: self.known_cost += cost
                # Successful private model outputs explain checker failures; never record headers/keys/URLs.
                self.emit({'kind': 'model_http_response', 'sequence': sequence, 'http_status': response.status_code,
                           'response_body': body if response.status_code == 200 else None,
                           'reported_cost_usd': str(cost) if cost is not None else None})
            return response
        transport.post = post
        return self

    def __exit__(self, *args):
        with self.lock: self.stopped = True
        if not self.retain_blocker_on_exit: self.transport.post = self.original


def writer_gate(events, supplied_labels):
    from scripts.compare_writer_plans import analyze
    report, review = analyze(events)
    if events[0]['plan'].get('code_sha256') != revision():
        raise ValueError('writer pilot used different application code')
    if report['finished_status'] != 'completed_draft_only_quality_pending':
        raise ValueError('writer comparison has not completed')
    if report['unknown_cost_attempts'] or report['attempts_started'] != report['planned_attempts']:
        raise ValueError('writer comparison accounting is incomplete')
    if not isinstance(supplied_labels.get('reviewer'), str) or not supplied_labels['reviewer'].strip():
        raise ValueError('writer review needs an independent reviewer')
    expected = {row['id']: row for row in review['items']}
    mapping = {row['id']: row['variant'] for row in report['review_mapping']}
    reviewed = {}
    fields = ['meaning_supported', 'conditions_preserved', 'attribution_and_grade_correct',
              'question_covered', 'unnecessary_repetition']
    for row in supplied_labels.get('items', []):
        if row['id'] not in expected or row['id'] in reviewed:
            raise ValueError('unexpected or duplicate writer review')
        if row.get('response_sha256') != expected[row['id']]['response_sha256']:
            raise ValueError('writer judgment belongs to a different response')
        values = row.get('verdicts', {})
        if any(type(values.get(k)) is not bool for k in fields):
            raise ValueError('writer judgment is incomplete')
        reviewed[row['id']] = values
    if set(reviewed) != set(expected): raise ValueError('writer judgments missing')
    for key, label in reviewed.items():
        if mapping[key] == 'candidate' and (not all(label[k] for k in fields[:-1]) or label['unnecessary_repetition']):
            raise ValueError('candidate writer has not passed source/coverage/repetition review')
    return {'passed': True, 'writer_results_sha256': digest(events), 'writer_labels_sha256': digest(supplied_labels),
            'reviewer': supplied_labels['reviewer'], 'writer_plan_sha256': events[0]['plan_sha256']}


def review_bundle(manifest, results):
    items, mapping = [], []
    for row in results:
        slot = ['A', 'B'] if int(digest(row['case_id'])[0], 16) % 2 else ['B', 'A']
        review_id = row['case_id'] + '-' + slot[['current', 'candidate'].index(row['variant'])]
        case = next(c for c in manifest['cases'] if c['id'] == row['case_id'])
        items.append({'id': review_id, 'question': case['question'], 'reference': case['reference'],
            'response': {k: v for k,v in row['response'].items() if k not in {'request_id', 'usage'}},
            'source_snapshots': row.get('source_snapshots', {}), 'response_sha256': digest(row['response']),
            'reviewer': '', **{label: None for label in LABELS}})
        mapping.append({'id': review_id, 'case_id': row['case_id'], 'variant': row['variant']})
    return {'scope': 'blinded_full_pipeline_review', 'items': sorted(items, key=lambda r: r['id'])}, mapping


def analyze_recording(events, supplied_labels=None):
    if not events or events[0].get('kind') != 'manifest': raise ValueError('missing manifest')
    header = events[0]; manifest = header['manifest']
    if header['manifest_sha256'] != digest(manifest): raise ValueError('manifest changed')
    starts, results = set(), []
    expected = {(c['id'], v) for c in manifest['cases'] for v in ['current', 'candidate']}
    wire_starts, wire_responses = set(), {}
    for event in events[1:]:
        if event.get('kind') == 'started':
            key = (event['case_id'], event['variant'])
            if key not in expected or key in starts: raise ValueError('unexpected or duplicate question attempt')
            starts.add(key)
        if event.get('kind') == 'model_http_started':
            if event['sequence'] in wire_starts: raise ValueError('duplicate model HTTP start')
            wire_starts.add(event['sequence'])
        if event.get('kind') == 'model_http_response':
            seq = event['sequence']
            if seq not in wire_starts or seq in wire_responses: raise ValueError('unmatched model HTTP response')
            wire_responses[seq] = event
        if event.get('kind') == 'result':
            row = event['result']
            if (row['case_id'], row['variant']) not in starts: raise ValueError('result without question attempt')
            results.append(row)
    accounting_verified = wire_starts == set(wire_responses)
    wire_total = Decimal(0)
    for event in wire_responses.values():
        cost = reported_cost({'usage': {'cost': event.get('reported_cost_usd')}})
        if cost is None: accounting_verified = False
        else: wire_total += cost
    trace_total = Decimal(0)
    for row in results:
        traces = row.get('traces') or []
        if len(traces) != 1 or traces[0].get('request_id') != row['response'].get('request_id'):
            accounting_verified = False; continue
        measured = summarize(traces)
        cost = reported_cost({'usage': {'cost': row.get('reported_cost_usd')}})
        if (not measured['cost_complete_for_observed_calls'] or cost is None
                or cost != Decimal(measured['known_reported_cost_usd'])
                or row.get('cache') != traces[0].get('cache')):
            accounting_verified = False
        if cost is not None: trace_total += cost
    if wire_total != trace_total: accounting_verified = False
    review, mapping = review_bundle(manifest, results)
    by_id = {row['id']: row for row in mapping}
    labels = []
    for row in (supplied_labels or {}).get('items', []):
        if row['id'] not in by_id: raise ValueError('unknown blinded review ID')
        labels.append({**row, **by_id[row['id']]})
    complete = (starts == expected and events[-1].get('kind') == 'finished' and events[-1].get('sources_unchanged') is True
                and events[-1].get('code_unchanged') is True
                and events[-1].get('status') == 'completed_pending_independent_review')
    report = verdict(manifest, results, labels)
    report.update(execution_completed=complete, accounting_verified=accounting_verified, review_mapping=mapping,
                  question_attempts_started=len(starts),
                  actual_model_http_attempts=sum(e.get('kind') == 'model_http_started' for e in events))
    if not accounting_verified:
        report['general_improvement'] = False
        report['accounting_problem'] = 'HTTP billing, request traces and per-question totals do not fully reconcile'
    if header.get('writer_gate', {}).get('passed') is not True:
        report['general_improvement'] = False
        report['writer_gate_problem'] = 'writer comparison review has not passed'
    if not complete:
        report['general_improvement'] = False
        report['execution_problem'] = 'incomplete run or changed source/code fingerprints'
    return report, review


class IsolatedAPI:
    """One private TestClient, separate engine state, shared read-only retrieval resources."""
    def __init__(self, server, current, candidate, emit=None):
        self.server, self.engines = server, {'current': current, 'candidate': candidate}
        self.original_engine = server.engine
        self.lines, self.lock = [], threading.Lock()
        owner = self
        class Capture(logging.Handler):
            def emit(self, record):
                text = record.getMessage()
                if any(marker in text for marker in ['muhawir_trace ', 'muhawir_trace_event ', 'muhawir_trace_drained ']):
                    with owner.lock: owner.lines.append(text)
                    if emit is not None: emit({'kind': 'trace_record', 'line': text})
        self.handler = Capture()

    def __enter__(self):
        from fastapi.testclient import TestClient
        logging.getLogger('muhawir.telemetry').addHandler(self.handler)
        self.client = TestClient(self.server.app)
        try: self.client.__enter__()
        except BaseException:
            logging.getLogger('muhawir.telemetry').removeHandler(self.handler); raise
        return self

    def __exit__(self, *args):
        try: self.client.__exit__(*args)
        finally:
            self.server.engine = self.original_engine
            logging.getLogger('muhawir.telemetry').removeHandler(self.handler)

    def __call__(self, variant, case):
        self.server.engine = self.engines[variant]
        engine = self.server.engine
        if engine.store is not None: raise ValueError('evaluation must not read/write the production answer store')
        with self.lock: self.lines.clear()
        began = time.perf_counter_ns()
        response = self.client.post('/api/ask', json={'question': case['question'], 'style': case['style'],
                                                    'lang': case['lang'], 'history': []})
        seconds = (time.perf_counter_ns() - began) / 1e9
        data = response.json(); rid = data.get('request_id')
        deadline = time.monotonic() + 20
        rows, selected = [], []
        while rid:
            with self.lock:
                selected = [line for line in self.lines if json.loads(line.split(' ', 1)[1]).get('request_id') == rid]
            rows = read_lines(selected) if any('muhawir_trace ' in line for line in selected) else []
            if rows and rows[0]['background_complete']: break
            if time.monotonic() >= deadline: break
            time.sleep(.05)
        measured = summarize(rows) if rows else None
        cited = {pid for c in data.get('claims', []) + data.get('views', []) for pid in c.get('passage_ids', [])}
        sources = {}
        for pid in cited | {s['passage_id'] for s in data.get('sources', [])}:
            passage = engine.corpus.passage(pid)
            sources[pid] = {'passage': asdict(passage), 'source': asdict(engine.corpus.source_of(passage))} if passage else None
        model_events = [e for row in rows for e in row['events'] if e.get('kind') == 'model']
        route_ids = sorted({(e['provider'], e.get('upstream_provider', e['provider']), e['model']) for e in model_events})
        if any(e.get('provider') == 'openrouter' and not e.get('upstream_provider') for e in model_events):
            route_ids = None
        return {'response': data, 'http_status': response.status_code, 'client_seconds': seconds,
                'route_ids': route_ids,
                'cache': rows[0]['cache'] if rows else None, 'traces': rows, 'source_snapshots': sources,
                'measurement_complete': bool(response.status_code == 200 and measured
                    and measured['cost_complete_for_observed_calls'] and all(sources.values())),
                'reported_cost_usd': measured['known_reported_cost_usd'] if measured else None}


def isolated_engines(server):
    from muhawir import pipeline
    from muhawir.generate import get_generator
    prototype = server.engine
    retriever = prototype.retriever
    candidate_retriever = retriever
    if os.environ.get('MUHAWIR_VECTORS') == '1':
        from muhawir.vectors import HybridRetriever, is_local
        if (not isinstance(retriever, HybridRetriever) or retriever._disabled or retriever.vectors is None
                or not is_local(retriever.vectors.model_name)):
            raise ValueError('evaluation requires a functioning local vector index')
        # Warm only the shared local weights, then give each arm a fresh query cache.
        retriever.warm_up()
        if retriever._disabled: raise ValueError('local vector warm-up failed')
        args = (retriever.sqlite_retriever, retriever.vectors)
        options = {'bm25_pool_factor': retriever.bm25_pool_factor, 'ann_rerank_factor': retriever.ann_rerank_factor}
        retriever, candidate_retriever = HybridRetriever(*args, **options), HybridRetriever(*args, **options)
    current_generator = prototype.generator
    current_generator.evidence_plan = False
    candidate_generator = get_generator(); candidate_generator.evidence_plan = True
    return (pipeline.Muhawir(prototype.corpus, current_generator, retriever, store=None),
            pipeline.Muhawir(prototype.corpus, candidate_generator, candidate_retriever, store=None))


def source_fingerprints(corpus):
    paths = [ROOT / 'data' / name for name in ['vectors.npy', 'vectors_ids.json', 'vectors_model.txt',
                                             'vectors.usearch', 'vectors.usearch.json']]
    if not getattr(corpus, '_path', None): raise ValueError('real source database path required')
    db = Path(corpus._path).resolve()
    paths += [db, Path(str(db) + '-wal')]
    out = {}
    for path in paths:
        if not path.exists(): continue
        before = path.stat(); hashed = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): hashed.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('source/index changed while hashing')
        out[str(path.resolve())] = {'sha256': hashed.hexdigest(), 'bytes': after.st_size,
                                  'mtime_ns': after.st_mtime_ns}
    return out


def verify_references(manifest, corpus):
    for case in manifest['cases']:
        for evidence in case['reference'].get('evidence', []):
            passage = corpus.passage(evidence['passage_id'])
            if (passage is None or hashlib.sha256(passage.text.encode()).hexdigest() != evidence['text_sha256']
                    or evidence.get('passage_sha256') and digest(asdict(passage)) != evidence['passage_sha256']):
                raise ValueError('reference source missing or changed: ' + evidence['passage_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--writer-results', type=Path)
    parser.add_argument('--writer-labels', type=Path)
    parser.add_argument('--max-model-http-attempts', type=int)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--stop-after-usd')
    args = parser.parse_args()
    if args.analyze:
        if args.execute: parser.error('analysis is offline')
        events = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
        labels = json.loads(args.labels.read_text()) if args.labels else None
        report, review = analyze_recording(events, labels)
        folder = ROOT / 'data/private/measurement'; stamp = time.time_ns()
        print('Report:', write_private(folder / f'full-report-{stamp}.json', report))
        print('Blinded review:', write_private(folder / f'full-review-{stamp}.json', review))
        print('General improvement gate:', report['general_improvement'])
        return
    manifest = json.loads(args.manifest.read_text()); count = validate_manifest(manifest)
    if not args.execute:
        print('OFFLINE ONLY. Planned question HTTP requests:', count)
        print('Manifest SHA256:', digest(manifest))
        print('Internal model calls are not bounded by this question count. No general quality verdict.')
        return
    if args.manifest_sha256 != digest(manifest): parser.error('exact pre-reviewed manifest hash required')
    ready = {'passed': False, 'status': 'development_without_writer_review'}
    if manifest['partition'] == 'declared_holdout' or args.writer_results or args.writer_labels:
        if not args.writer_results or not args.writer_labels:
            parser.error('holdout execution requires completed writer results and human review')
        writer_events = [json.loads(line) for line in args.writer_results.read_text().splitlines() if line.strip()]
        ready = writer_gate(writer_events, json.loads(args.writer_labels.read_text()))
        pilot_question = ' '.join(writer_events[0]['plan']['case']['question'].split()).casefold()
        if manifest['partition'] == 'declared_holdout' and any(' '.join(c['question'].split()).casefold() == pilot_question for c in manifest['cases']):
            parser.error('writer pilot question cannot count as holdout')
    threshold = Decimal(args.stop_after_usd or '0')
    if not threshold.is_finite() or threshold <= 0: parser.error('explicit positive post-response USD stop required')
    if args.max_model_http_attempts is None or not 1 <= args.max_model_http_attempts <= 10000:
        parser.error('explicit --max-model-http-attempts required')
    if 'muhawir.server' in sys.modules: parser.error('requires a fresh isolated process')
    os.environ.update(MUHAWIR_STORE='0', MUHAWIR_CACHE='0', MUHAWIR_TRACE='1')
    if os.environ.get('MUHAWIR_VECTORS') == '1':
        from muhawir.vectors import is_local
        model_file = ROOT / 'data/vectors_model.txt'
        if not model_file.exists() or not is_local(model_file.read_text().strip()):
            parser.error('local embeddings required; remote embedding billing is outside this runner')
    from muhawir import server
    from muhawir.generate import ModelGenerator
    if server.engine.corpus.synthetic: parser.error('real execution requires the real pinned source corpus')
    if not isinstance(server.engine.generator, ModelGenerator): parser.error('model generator required')
    from scripts.run_acceptance_batch import public_config
    verify_references(manifest, server.engine.corpus)
    frozen = source_fingerprints(server.engine.corpus); code_before = revision()
    runner_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    current, candidate = isolated_engines(server)
    # This private in-process quality experiment does not test visitor quotas/capacity.
    server.ASK_PER_MINUTE = count + 1
    server.ASK_LIMITS = ((3600, count + 1),)
    folder = ROOT / 'data/private/measurement'; folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    # One marker per code+manifest prevents accidental paid repeats in the same checkout.
    identity = digest({'manifest': manifest, 'code': revision()})
    marker = folder / f'full-pipeline-{identity}.started'
    with os.fdopen(os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as out: out.write(identity)
    path = folder / f'full-pipeline-{time.time_ns()}.jsonl'
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        io_lock = threading.Lock()
        def emit(event):
            with io_lock:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n'); stream.flush()
                if not event['kind'].startswith('model_http_'): os.fsync(stream.fileno())
        emit({'kind': 'manifest', 'manifest': manifest, 'manifest_sha256': digest(manifest),
              'code_sha256': revision(), 'runner_sha256': runner_before,
              'source_fingerprints': frozen, 'writer_gate': ready,
              'config': public_config(b'\0'.join((k + '=' + v).encode() for k,v in os.environ.items())),
              'post_response_stop_usd': str(threshold), 'max_model_http_attempts': args.max_model_http_attempts, 'scope': 'same_build_writer_ablation_full_api_in_process',
              'measurement_conditions': {'concurrency': 1, 'answer_cache': 'disabled',
                  'query_caches': 'separate_per_arm', 'visitor_quota': 'isolated_test_allowance',
                  'timing_includes': 'API serialization, pipeline and private measurement IO',
                  'timing_excludes': 'startup, source hashing, visitor network and browser'}})
        with ModelHTTPBudget(args.max_model_http_attempts, threshold, emit, retain_blocker_on_exit=True) as budget:
            with IsolatedAPI(server, current, candidate, emit=emit) as api:
                def ask(variant, case):
                    budget.active = (case['id'], variant)
                    result = api(variant, case)
                    result['actual_model_http_attempts_cumulative'] = budget.sent
                    with budget.lock:
                        if not result['measurement_complete']: budget.stopped = True
                        result['measurement_complete'] = result['measurement_complete'] and not budget.stopped
                    return result
                try:
                    results, status = run(manifest, ask, emit, threshold)
                finally:
                    # Pending workers must never regain an unguarded transport during shutdown.
                    with budget.lock: budget.stopped = True
        stable = source_fingerprints(server.engine.corpus) == frozen
        emit({'kind': 'finished', 'status': status, 'sources_unchanged': stable, 'code_unchanged': revision() == code_before and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == runner_before,
              'quality': verdict(manifest, results, [])})
    print(status); print('Saved:', path)
    print('Quality pending; in-process API latency excludes visitor network/browser and load capacity.')


if __name__ == '__main__': main()
