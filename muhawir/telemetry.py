"""Opt-in, content-free per-request measurements. Never an LLM judge or a billing estimate."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor as Executor
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation
from functools import wraps
import json
import hashlib
from pathlib import Path
import logging
import os
import threading
import time
import uuid

log = logging.getLogger(__name__)
_current = ContextVar("muhawir_trace", default=None)
_stage = ContextVar("muhawir_trace_stage", default="other")
_attempt = ContextVar("muhawir_trace_attempt", default=None)
MAX_EVENTS = 4096
_code = hashlib.sha256()
for _path in sorted(Path(__file__).parent.glob("*.py")):
    _code.update(_path.name.encode())
    _code.update(_path.read_bytes())
CODE_REVISION = _code.hexdigest()


class Trace:
    def __init__(self, endpoint, style=None, lang=None, as_question=None):
        self.id = uuid.uuid4().hex
        self.started = time.perf_counter_ns()
        self.started_unix_ns = time.time_ns()
        self.endpoint = endpoint
        self.style, self.lang, self.as_question = style, lang, as_question
        self.events = []
        self.lock = threading.Lock()
        self.dropped = 0
        self.finished = False
        self.cache = "none"
        self.pending_work = 0

    def add(self, item):
        item = {**item, "event_id": uuid.uuid4().hex}
        with self.lock:
            if len(self.events) < MAX_EVENTS:
                self.events.append(item)
                late = self.finished
            else:
                self.dropped += 1
                return
        if late:
            log.warning("muhawir_trace_event %s", json.dumps({"request_id": self.id, "event": item}, allow_nan=False))

    def begin_work(self):
        with self.lock:
            self.pending_work += 1

    def end_work(self):
        with self.lock:
            self.pending_work -= 1
            drained = self.finished and self.pending_work == 0
            dropped = self.dropped
        if drained:
            log.warning("muhawir_trace_drained %s", json.dumps({"request_id": self.id, "events_dropped": dropped}))

    def finish(self, status, http_status):
        with self.lock:
            if self.finished:
                return
            self.finished = True
            record = {"schema_version": 1, "request_id": self.id, "endpoint": self.endpoint,
                      "pid": os.getpid(), "code_revision": CODE_REVISION,
                      "started_unix_ns": self.started_unix_ns,
                      "duration_ns": time.perf_counter_ns() - self.started,
                      "status": status, "http_status": http_status, "cache": self.cache,
                      "style": self.style, "lang": self.lang, "as_question": self.as_question,
                      "events_dropped": self.dropped, "pending_work_at_response": self.pending_work,
                      "events": list(self.events)}
        # No content, user identifiers, URL, credentials, exception message or prompt hashes.
        log.warning("muhawir_trace %s", json.dumps(record, separators=(",", ":"), allow_nan=False))


def start(endpoint, *, style=None, lang=None, as_question=None):
    if os.environ.get("MUHAWIR_TRACE") != "1":
        return None
    return Trace(endpoint, style, lang, as_question)


@contextmanager
def bind(trace, stage="other"):
    token, label = _current.set(trace), _stage.set(stage)
    try:
        yield
    finally:
        _stage.reset(label)
        _current.reset(token)


class ThreadPoolExecutor(Executor):
    """Propagate only telemetry, leaving existing pipeline thread-local state unchanged."""
    def submit(self, fn, /, *args, **kwargs):
        trace, stage = _current.get(), _stage.get()
        if trace is None:
            return super().submit(fn, *args, **kwargs)
        trace.begin_work()
        def run():
            try:
                with bind(trace, stage):
                    return fn(*args, **kwargs)
            finally:
                trace.end_work()
        try:
            future = super().submit(run)
        except BaseException:
            trace.end_work()
            raise
        future.add_done_callback(lambda f: trace.end_work() if f.cancelled() else None)
        return future


@contextmanager
def span(name):
    trace = _current.get()
    if trace is None:
        yield
        return
    started = time.perf_counter_ns()
    token = _stage.set(name)
    try:
        yield
    finally:
        _stage.reset(token)
        trace.add({"kind": "span", "name": name, "offset_ns": started - trace.started,
                   "duration_ns": time.perf_counter_ns() - started})


def support_reuse(count):
    """Count reused completed judgments without recording any answer or source text."""
    trace = _current.get()
    if trace is not None and count:
        trace.add({"kind": "reuse", "name": "support_positive", "count": count})


def cache_hit(where):
    trace = _current.get()
    if trace is not None:
        trace.cache = where


def response_metadata(response, usage, *, usd_cost=False):
    event = _attempt.get()
    if event is None:
        return
    event["http_status"] = getattr(response, "status_code", None)
    # Whitelist numeric usage fields. Never keep the provider's arbitrary response metadata.
    for output, value in (("tokens_in", usage.get("prompt_tokens", usage.get("input_tokens"))),
                          ("tokens_out", usage.get("completion_tokens", usage.get("output_tokens"))),
                          ("tokens_cached", (usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
                          ("reasoning_tokens", (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            event[output] = value
    value = usage.get("cost")
    if usd_cost and value is not None and not isinstance(value, bool):
        try:
            cost = Decimal(str(value))
            if cost.is_finite() and cost >= 0:
                event["reported_cost_usd"] = str(cost)
        except InvalidOperation:
            pass


def model_call(provider, model):
    """One event per actual transport invocation, including errors, retries and fallbacks."""
    def decorate(fn):
        @wraps(fn)
        def measured(*args, **kwargs):
            trace = _current.get()
            if trace is None:
                return fn(*args, **kwargs)
            started = time.perf_counter_ns()
            event = {"kind": "model", "stage": _stage.get(), "provider": provider, "model": model,
                     "offset_ns": started - trace.started, "reported_cost_usd": None,
                     "outcome": "ok", "http_status": None}
            trace.begin_work()
            token = _attempt.set(event)
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:
                event["outcome"] = "error"
                event["error_type"] = type(exc).__name__
                event["http_status"] = getattr(getattr(exc, "response", None), "status_code", event["http_status"])
                raise
            finally:
                _attempt.reset(token)
                event["duration_ns"] = time.perf_counter_ns() - started
                trace.add(event)
                trace.end_work()
        return measured
    return decorate
