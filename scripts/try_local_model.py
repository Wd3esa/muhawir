"""Try a small local model (served by Ollama) on Muhawir's understanding step. FREE: no paid model is called.

For each gold question the model gets the real understanding prompt and schema (muhawir/generate.py) and must return
the JSON the pipeline needs. Reported per question: seconds, whether the reply is valid JSON that fits the schema,
and the kind / form / parts / search phrases it chose; then a summary. Nothing is written to the database or the service.

On the server (Ollama running, model pulled):
    ollama pull lfm3-omni
    .venv/bin/python scripts/try_local_model.py --model lfm3-omni --per-field 2

A reply that does not fit the schema is what the pipeline would reject: it counts as a failure here too.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muhawir.generate import UNDERSTAND_PROMPT, UNDERSTAND_SCHEMA  # noqa: E402


def pick(questions: list[dict], per_field: int) -> list[dict]:
    """The first ``per_field`` questions of every field, so each kind of question is tried."""
    seen: dict[str, int] = {}
    chosen = []
    for q in questions:
        seen[q["field"]] = seen.get(q["field"], 0) + 1
        if seen[q["field"]] <= per_field:
            chosen.append(q)
    return chosen


def fits(data, schema) -> str:
    """'' when ``data`` fits the schema, else the first problem (jsonschema when installed, else a key check)."""
    try:
        import jsonschema
        jsonschema.validate(data, schema)
        return ""
    except ImportError:
        missing = [k for k in schema["required"] if not isinstance(data, dict) or k not in data]
        return f"missing {missing}" if missing else ""
    except jsonschema.ValidationError as exc:
        return exc.message[:80]


def ask(client: httpx.Client, url: str, model: str, message: str, options: dict | None = None,
        schema: bool = True, stats: dict | None = None) -> tuple[object, float, str]:
    """(parsed reply or None, seconds, error text). ``options`` are Ollama's (num_ctx, num_predict, ...)."""
    start = time.perf_counter()
    try:
        r = client.post(f"{url}/api/chat", json={
            "model": model, "stream": False, "think": False,  # a reasoning model must answer, not think aloud
            "format": UNDERSTAND_SCHEMA if schema else "json",
            "options": {"temperature": 0, **(options or {})},
            "messages": [{"role": "system", "content": UNDERSTAND_PROMPT},
                         {"role": "user", "content": f"الرسالة: <<<{message}>>>"}]})
        r.raise_for_status()
        body = r.json()
        if stats is not None:  # Ollama's own counters: tokens read, tokens written, and nanoseconds spent on each
            stats.update({k: body.get(k, 0) for k in
                          ("prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")})
        text = body["message"]["content"]
        return json.loads(text), time.perf_counter() - start, ""
    except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
        return None, time.perf_counter() - start, f"{type(exc).__name__}: {str(exc)[:80]}"


def speed(stats: dict) -> str:
    """'  read N tok in X s, wrote M tok at Y tok/s': where the time went (reading the prompt or writing the reply)."""
    if not stats.get("eval_count"):
        return ""
    read_s = stats["prompt_eval_duration"] / 1e9
    write_s = max(stats["eval_duration"] / 1e9, 1e-9)
    return (f"\n      read {stats['prompt_eval_count']} tok in {read_s:.0f}s, "
            f"wrote {stats['eval_count']} tok at {stats['eval_count'] / write_s:.1f} tok/s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="lfm3-omni")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--per-field", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--num-ctx", type=int, default=8192,
                    help="context window in tokens; the prompt alone is about 10,000 Arabic characters, and Ollama's "
                         "default window may cut it")
    ap.add_argument("--num-predict", type=int, default=1024, help="longest reply in tokens")
    ap.add_argument("--repeat-penalty", type=float, default=None)
    ap.add_argument("--no-schema", action="store_true", help="ask for any JSON, not one forced to the schema")
    args = ap.parse_args()
    gold = json.loads((ROOT / "eval_sets" / "gold_questions.json").read_text(encoding="utf-8"))["questions"]
    questions = pick(gold, args.per_field)
    options = {"num_ctx": args.num_ctx, "num_predict": args.num_predict}
    if args.repeat_penalty:
        options["repeat_penalty"] = args.repeat_penalty
    print(f"options: {options}, schema {'off' if args.no_schema else 'on'}")
    seconds, good, rows = [], 0, []
    with httpx.Client(timeout=args.timeout) as client:
        for q in questions:
            stats: dict = {}
            data, took, error = ask(client, args.url, args.model, q["question"], options, not args.no_schema, stats)
            problem = error or fits(data, UNDERSTAND_SCHEMA)
            seconds.append(took)
            good += not problem
            req = (data or {}).get("answer_request", {}) if isinstance(data, dict) else {}
            print(f"{'✓' if not problem else '✗'} {q['id']:4} {q['field']:9} {took:5.1f}s  "
                  f"kind={(data or {}).get('kind', '?') if isinstance(data, dict) else '?'} "
                  f"form={req.get('form', '?')} parts={len(req.get('requirements', []) or [])} "
                  f"queries={len((data or {}).get('queries', []) or []) if isinstance(data, dict) else 0}"
                  + (f"  [{problem}]" if problem else "") + speed(stats), flush=True)
            if not problem:
                rows.append((q, data))
    print(f"\n{args.model}: {good}/{len(questions)} replies fit the schema; "
          f"median {statistics.median(seconds):.1f}s, slowest {max(seconds):.1f}s")
    for q, data in rows[:3]:
        print(f"\n{q['id']} {q['question']}\n  question: {data['question']}\n  queries: {data['queries']}\n"
              f"  parts: {data['answer_request']['requirements']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
