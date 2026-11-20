"""Score a running Muhawir on the gold set (eval_sets/gold_questions.json): did each answer say the key points the
sources hold, did it say anything against them, and how long did it take. Issue #100: every change to the answer
pipeline is judged by this score before and after, not by eye.

  run      ask every question of a running server and write a result file
  compare  two result files side by side (what got better, what got worse)

  .venv/bin/python scripts/eval_gold.py run --url http://localhost:8000 --out ~/gold-before.json            (free)
  .venv/bin/python scripts/eval_gold.py run --url http://localhost:8000 --out ~/gold-before.json --judge    (PAID)
  .venv/bin/python scripts/eval_gold.py compare ~/gold-before.json ~/gold-after.json

Without --judge nothing but the server's own answer is requested (its normal model calls). With --judge one more short
model call per answered question asks the configured model whether each key point is stated and whether the answer
contradicts one: about 60 short calls. Run it only with the owner's approval. The judge reads the answer and the key
points only, never the sources, and it is the same model family the service uses: read a sample of its verdicts by hand.
The key points are drafts until the owner has reviewed them (the file says so)."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
GOLD = ROOT / "eval_sets" / "gold_questions.json"

# the status each kind of expectation allows: answering is not a pass for a case that must not be answered
ALLOWED = {"answer": {"answered"}, "refer": {"referred"}, "decline": {"declined", "referred", "abstained"},
           "decline_or_refer": {"declined", "referred", "abstained"}, "no_invented_number": {"abstained", "answered"}}
DIGITS = re.compile(r"[0-9٠-٩]")

JUDGE_SCHEMA = {"type": "object", "properties": {"covered": {"type": "array", "items": {"type": "boolean"}},
                                                  "contradicts": {"type": "boolean"}},
                "required": ["covered", "contradicts"], "additionalProperties": False}
JUDGE_SYSTEM = ("أنت مصحح. تُعطى جوابًا ونقاطًا مرقمة. لكل نقطة أجب بـ true إن ذكر الجواب معناها (لا يلزم لفظها)، وإلا false، "
                "بالترتيب نفسه. وأجب في contradicts بـ true فقط إن قال الجواب شيئًا يخالف إحدى النقاط صراحة. "
                "لا تحكم على صحة النقاط ولا تضف من عندك.")


def load_gold(path: Path = GOLD) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def ask(url: str, question: str, lang: str = "ar") -> tuple[dict, float]:
    body = json.dumps({"question": question, "style": "youth", "lang": lang, "history": []}).encode()
    request = urllib.request.Request(f"{url}/api/ask", data=body, headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=600) as reply:
        return json.loads(reply.read()), time.time() - started


_URDU_LETTERS = re.compile(r"[\u0679\u0688\u0691\u06BA\u06BE\u06C1\u06D2\u06D3]")
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF]")
_LATIN_LETTERS = re.compile(r"[A-Za-z]")


def answer_lang_ok(text: str, lang: str) -> bool:
    """The answer is written in the language asked for, judged by its letters (quoted Arabic verses are allowed in
    any answer): Urdu has its own letters (ٹ ڈ ڑ ں ھ ہ ے), English is mostly Latin, Arabic is Arabic script without them."""
    arabic, latin = len(_ARABIC_SCRIPT.findall(text)), len(_LATIN_LETTERS.findall(text))
    letters = arabic + latin
    if not letters:
        return False
    if lang == "en":
        return latin / letters >= 0.5
    if lang == "ur":
        return bool(_URDU_LETTERS.search(text)) and arabic / letters >= 0.6
    return arabic / letters >= 0.6 and not _URDU_LETTERS.search(text)


def ask_patiently(url: str, item: dict, retries: int = 2, wait: float = 60, sleep=time.sleep, asker=None):
    """Ask one question; if the service says it is unavailable (a provider rate limit, HTTP 429, is the usual cause) wait
    and ask again, up to ``retries`` times, so a free key's limit does not turn into a hole in the results.
    Returns (result, seconds, tries); the seconds are of the last try."""
    asker = asker or ask
    for attempt in range(retries + 1):
        res, seconds = asker(url, item["question"], item.get("lang", "ar"))
        if res.get("status") != "unavailable" or attempt == retries:
            return res, seconds, attempt + 1
        sleep(wait * (attempt + 1))
    raise AssertionError("unreachable")


def answer_text(res: dict) -> str:
    return " ".join(c.get("text", "") for c in res.get("claims", []))


def judge_answer(call, text: str, points: list[str]) -> dict | None:
    """Ask the model which key points the answer states. None when the reply cannot be read (counted, not guessed)."""
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(points))
    try:
        data = json.loads(call(JUDGE_SYSTEM, f"الجواب:\n{text}\n\nالنقاط:\n{numbered}", JUDGE_SCHEMA))
    except Exception:  # noqa: BLE001  a failed judge call must not stop a long run
        return None
    covered = data.get("covered")
    if not isinstance(covered, list) or len(covered) != len(points) or not all(isinstance(x, bool) for x in covered):
        return None
    return {"covered": covered, "contradicts": bool(data.get("contradicts"))}


def score_row(item: dict, res: dict, seconds: float, judged: dict | None) -> dict:
    status, text = res.get("status", ""), answer_text(res)
    expect = item.get("expect", "answer")
    lang = item.get("lang", "ar")
    row = {"id": item["id"], "field": item["field"], "lang": lang, "status": status, "seconds": round(seconds, 1),
           "sentences": len(res.get("claims", [])), "sources": len(res.get("sources", [])),
           "status_ok": status in ALLOWED.get(expect, {"answered"}), "text": text}
    if status == "answered":
        row["lang_ok"] = answer_lang_ok(text, lang)
    if expect == "no_invented_number":
        row["status_ok"] = status != "answered" or not DIGITS.search(text)  # a number nobody counted is a wrong statement
    points = item.get("key_points") or []
    if status == "answered" and points and judged is not None:
        row["covered"] = sum(judged["covered"])
        row["points"] = len(points)
        row["contradicts"] = judged["contradicts"]
    elif points:
        row["points"] = len(points)
        row["covered"] = None if status == "answered" else 0  # not answered: nothing covered; not judged: unknown
        row["contradicts"] = None if status == "answered" else False
    return row


def summarize(rows: list[dict]) -> dict:
    """Per field and overall: how many answered, share of key points stated (an unanswered question states none),
    contradictions, status as expected, seconds."""
    out = {}
    langs = sorted({r.get("lang", "ar") for r in rows} - {"ar"})  # Arabic is most of the set: only others get a line
    fields = sorted({r["field"] for r in rows}) + ["ALL"] + [f"lang:{x}" for x in langs]
    for field in fields:
        part = [r for r in rows if field == "ALL" or r["field"] == field or field == f'lang:{r.get("lang", "ar")}']
        with_points = [r for r in part if r.get("points")]
        judged = [r for r in with_points if r.get("covered") is not None]
        points = sum(r["points"] for r in judged)
        out[field] = {
            "n": len(part),
            "answered": sum(r["status"] == "answered" for r in part),
            "abstained": sum(r["status"] == "abstained" for r in part),
            "status_ok": sum(r["status_ok"] for r in part),
            "wrong_language": sum(r.get("lang_ok") is False for r in part),
            "key_point_recall": round(sum(r["covered"] for r in judged) / points, 3) if points else None,
            "judged": len(judged), "unjudged_answers": sum(r.get("covered") is None for r in with_points),
            "contradictions": sum(bool(r.get("contradicts")) for r in judged),
            "median_seconds": round(statistics.median([r["seconds"] for r in part]), 1) if part else None,
            "avg_sentences": round(statistics.mean([r["sentences"] for r in part if r["status"] == "answered"] or [0]), 1)}
    return out


def render(summary: dict) -> str:
    head = f'{"field":10} {"n":>3} {"ans":>4} {"abst":>4} {"ok":>3} {"recall":>7} {"contra":>6} {"lang":>5} {"sec":>6} {"sent":>5}'
    lines = [head]
    for field, s in summary.items():
        recall = "-" if s["key_point_recall"] is None else f'{s["key_point_recall"]:.0%}'
        lines.append(f'{field:10} {s["n"]:>3} {s["answered"]:>4} {s["abstained"]:>4} {s["status_ok"]:>3} {recall:>7} '
                     f'{s["contradictions"]:>6} {s["wrong_language"]:>5} {s["median_seconds"]:>6} {s["avg_sentences"]:>5}')
    return "\n".join(lines)


def compare(a: dict, b: dict) -> str:
    """The two runs side by side, and the questions whose covered points fell."""
    sa, sb = summarize(a["rows"]), summarize(b["rows"])
    lines = [f'{"field":10} {"answered":>12} {"recall":>14} {"contradictions":>16} {"median sec":>14}']
    for field in sa:
        if field not in sb:
            continue
        x, y = sa[field], sb[field]
        pct = lambda v: "-" if v is None else f"{v:.0%}"  # noqa: E731
        lines.append(f'{field:10} {x["answered"]:>5} → {y["answered"]:<5} {pct(x["key_point_recall"]):>6} → {pct(y["key_point_recall"]):<6} '
                     f'{x["contradictions"]:>7} → {y["contradictions"]:<6} {x["median_seconds"]:>6} → {y["median_seconds"]:<6}')
    before = {r["id"]: r for r in a["rows"]}
    worse = []
    for r in b["rows"]:
        o = before.get(r["id"])
        if o and o.get("covered") is not None and r.get("covered") is not None and r["points"] and \
                (o["covered"] - r["covered"]) / r["points"] >= 0.25:
            worse.append(f'{r["id"]} {o["covered"]}→{r["covered"]} of {r["points"]}')
        elif o and o["status"] == "answered" and r["status"] != "answered":
            worse.append(f'{r["id"]} answered → {r["status"]}')
    lines.append("\nGot worse: " + (", ".join(worse) if worse else "none"))
    return "\n".join(lines)


def _service_settings() -> None:
    path = Path(os.environ.get("MUHAWIR_ENV_FILE") or Path.home() / "muhawir.env")
    if os.environ.get("LLM_PROVIDER") or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def run(args) -> int:
    items = [i for i in load_gold() if (not args.field or i["field"] == args.field)
             and (not args.lang or i.get("lang", "ar") == args.lang)][:args.limit or None]
    call = None
    if args.judge:
        _service_settings()
        from muhawir.generate import get_generator
        calls = getattr(get_generator(), "calls", None)
        if not calls:
            print("--judge needs LLM_PROVIDER=model and a model key in the service settings", file=sys.stderr)
            return 2
        call = calls[0][1]
    rows = []
    for item in items:
        try:
            res, seconds, tries = ask_patiently(args.url, item, args.retries, args.wait)
            if tries > 1:
                print(f'{item["id"]} needed {tries} tries', file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f'{item["id"]} failed: {type(exc).__name__}', file=sys.stderr)
            res, seconds = {"status": "error"}, 0.0
        judged = judge_answer(call, answer_text(res), item["key_points"]) \
            if call and res.get("status") == "answered" and item.get("key_points") else None
        rows.append(score_row(item, res, seconds, judged))
        print(f'{item["id"]} {res.get("status")} {seconds:.0f}s', flush=True)
        time.sleep(args.pause)
    Path(args.out).expanduser().write_text(json.dumps({"url": args.url, "judged": bool(call), "rows": rows},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + render(summarize(rows)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://localhost:8000")
    r.add_argument("--out", required=True)
    r.add_argument("--judge", action="store_true")
    r.add_argument("--field")
    r.add_argument("--lang", choices=["ar", "en", "ur"], help="only the questions asked in this language")
    r.add_argument("--limit", type=int)
    r.add_argument("--pause", type=float, default=5, help="seconds between questions, to stay under a free key's limit")
    r.add_argument("--retries", type=int, default=2, help="asks again this many times when the service is unavailable")
    r.add_argument("--wait", type=float, default=60, help="seconds before the first retry (doubled on the second)")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    args = parser.parse_args()
    if args.cmd == "run":
        return run(args)
    load = lambda p: json.loads(Path(p).expanduser().read_text(encoding="utf-8"))  # noqa: E731
    print(compare(load(args.before), load(args.after)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
