"""Read one uploaded file in a separate process (audit HIGH-005, November 2026).

The server starts this with the file's bytes on stdin and limits it: wall-clock time (the server kills it), memory and
processor time (set here, before anything is parsed). A PDF or picture made to exhaust its parser then ends this
process, not the web server. The reply on stdout is one JSON object: {"result": {...}} or {"code": ..., "note": ...}.

    python -m muhawir.extract_worker < file
"""
from __future__ import annotations

import json
import os
import sys


def _limit() -> None:
    try:
        import resource
    except ImportError:  # not on Linux: the server's wall-clock limit still applies
        return
    memory = int(os.environ.get("MUHAWIR_EXTRACT_MEMORY_MB", "1536")) * 1024 * 1024
    cpu = int(os.environ.get("MUHAWIR_EXTRACT_CPU_SECONDS", "90"))
    for kind, value in ((resource.RLIMIT_AS, memory), (resource.RLIMIT_CPU, cpu)):
        try:
            resource.setrlimit(kind, (value, value))
        except (ValueError, OSError):
            pass


def main() -> int:
    _limit()
    data = sys.stdin.buffer.read()
    from muhawir import extract
    try:
        reply = {"result": extract.extract(data)}
    except extract.ExtractError as exc:
        reply = {"code": exc.code, "note": exc.note}
    except MemoryError:
        reply = {"code": "too_large", "note": "memory"}
    sys.stdout.write(json.dumps(reply, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
