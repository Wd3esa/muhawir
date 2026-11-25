"""One plain line per log record (November 2026).

Muhawir set no logging of its own, so the MCP library's start-up (`MCPServer`) configured the root logger with Rich,
which wraps every message at the console width (80 columns when the output is the journal). A long line was cut into
several journal lines: the model timing lines lost their token counts, and every `muhawir_trace` JSON line was split,
so scripts/analyze_request_traces.py found no complete trace. A handler set here first makes the library's
`logging.basicConfig` a no-op, and each record stays on one line however long it is."""
from __future__ import annotations

import logging
import sys

FORMAT = "%(levelname)s %(name)s: %(message)s"


def plain_logging(level: int = logging.INFO, stream=None) -> None:
    """Give the root logger one plain, unwrapped handler, unless it already has handlers (a test or a host's own)."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
