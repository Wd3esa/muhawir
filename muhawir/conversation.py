"""Deterministic, token-budgeted conversation context.

This module never generates religious claims, persists user information, or calls a model.
The caller controls retention; this only selects context to pass to the existing
source-grounded understanding pipeline.
"""
from __future__ import annotations

from collections.abc import Iterable

MAX_TURN_CHARS = 1200
MAX_CONTEXT_CHARS = 9000
RECENT_TURNS = 10
EARLIER_ANCHORS = 4


def select_context(
    history: Iterable[dict],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
    recent_turns: int = RECENT_TURNS,
    earlier_anchors: int = EARLIER_ANCHORS,
) -> list[dict[str, str]]:
    """Keep recent turns intact and a few earlier user topic anchors.

    Untrusted client history is strictly validated and truncated. No synthetic
    assistant statements or LLM-written summaries are introduced: an invented
    summary could otherwise contaminate religious question interpretation.
    """
    if max_chars < 1:
        return []
    clean: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            continue
        value = item.get("text")
        if not isinstance(value, str):
            continue
        value = value.strip()[:MAX_TURN_CHARS]
        if value:
            clean.append({"role": item["role"], "text": value})
    if not clean:
        return []

    recent = clean[-max(0, recent_turns):] if recent_turns else []
    earlier = clean[:-len(recent)] if recent else clean
    anchors = [turn for turn in earlier if turn["role"] == "user"][-max(0, earlier_anchors):] if earlier_anchors else []
    selected = anchors + recent
    # Preserve the latest dialogue when the budget is tight; anchors go first.
    while selected and sum(len(t["text"]) for t in selected) > max_chars:
        if anchors:
            anchors.pop(0)
            selected = anchors + recent
        elif recent:
            recent.pop(0)
            selected = recent
        else:
            break
    return selected
