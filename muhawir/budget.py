"""One time budget per answer (audit HIGH-007, November 2026).

Every model call of an answer (understanding, writing, the second reading, review, rewrite, a fallback model) takes its
wait from the same deadline: a call waits at most its own limit or the time left, whichever is shorter, and no call is
started once the time is up. Before, each call had its own limit (up to 300 s), so a slow provider could hold an answer
slot for many minutes. The deadline travels into the answer's parallel steps (telemetry.ThreadPoolExecutor).
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import time

ANSWER_SECONDS = float(os.environ.get("MUHAWIR_ANSWER_DEADLINE") or 120)
MIN_CALL = 2.0  # seconds: with less than this left, a call is not started
_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("muhawir_deadline", default=None)


class OutOfTime(Exception):
    """The answer's time budget is spent: no further model call is made."""


@contextlib.contextmanager
def answer_deadline(seconds: float | None = None):
    """The time budget for one answer. A budget already running (an answer inside an answer) is kept, not extended."""
    if _deadline.get() is not None:
        yield
        return
    token = _deadline.set(time.monotonic() + (ANSWER_SECONDS if seconds is None else seconds))
    try:
        yield
    finally:
        _deadline.reset(token)


def current() -> float | None:
    return _deadline.get()


def run_with(deadline: float | None, fn, *args, **kwargs):
    """Run fn under the given deadline (for a worker thread of the same answer)."""
    token = _deadline.set(deadline)
    try:
        return fn(*args, **kwargs)
    finally:
        _deadline.reset(token)


def call_timeout(own: float) -> float:
    """How long the next model call may wait: its own limit, or the time left in the answer; OutOfTime when too little."""
    deadline = _deadline.get()
    if deadline is None:
        return own
    left = deadline - time.monotonic()
    if left < MIN_CALL:
        raise OutOfTime()
    return min(own, left)


def left() -> float | None:
    deadline = _deadline.get()
    return None if deadline is None else deadline - time.monotonic()
