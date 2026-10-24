"""The nightly review, run on the server by a timer: what Muhawir could not answer in the last day, why, and
what was fixed on its own.

For each first question kept as «not found» in data/answers.db:
  1. diagnosis by code, at no cost: a language Muhawir does not answer in, a counting question not recognised,
     a hadith asked about, how well the search matched the user's own words, and the reason in the service log;
  2. with explicit --retry only, the question is answered once more from scratch by the live service
     (/api/ask_again, served on this machine only), through every check as usual. At most --max questions
     are retried (each costs what one answer costs); a question is not tried twice in a week;
  3. what is still not answered goes into the report with its diagnosis, for a person to decide. Nothing in the
     code, the instructions, the rules or the sources is changed by this script.

    .venv/bin/python scripts/nightly.py            # diagnose only: no model call, no cost
    .venv/bin/python scripts/nightly.py --retry --max 10  # explicit paid retries

The report goes to ~/muhawir-reports/<date>.txt and ~/muhawir-reports/latest.txt (English, questions in Arabic).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muhawir import quran_count  # noqa: E402
from muhawir.normalize import normalize  # noqa: E402

REPORTS = Path(os.environ.get("MUHAWIR_REPORTS", str(Path.home() / "muhawir-reports")))
WEEK = 7 * 24 * 3600
_OTHER_LANG = re.compile(
    r"بال(?:اسبان|فرنس|فرنساو|اورد|الماني|تركي|صيني|روس|ايطالي|اندونيس|ملاو|فارس|هند|برتغال|يابان|كوري|سواحل|بنغال)\w*|"
    r"\bin (?:spanish|french|urdu|german|turkish|chinese|russian|italian|indonesian|malay|persian|hindi|portuguese|"
    r"japanese|korean|swahili|bengali)\b", re.IGNORECASE)
_ASKS_COUNT = re.compile(r"كم\s*مر|عدد\s*مرات|how many times", re.IGNORECASE)


def service_url() -> str:
    if os.environ.get("MUHAWIR_URL"):
        return os.environ["MUHAWIR_URL"].rstrip("/")
    try:
        unit = subprocess.run(["systemctl", "cat", "muhawir"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"--port[ =](\d+)", unit)
        if m:
            return f"http://127.0.0.1:{m.group(1)}"
    except Exception:
        pass
    return "http://127.0.0.1:8000"


def post(url: str, body: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def log_reason(at: float) -> str:
    """The «not answered» line the service logged just before the reply was kept."""
    since = datetime.datetime.fromtimestamp(at - 180).strftime("%Y-%m-%d %H:%M:%S")
    until = datetime.datetime.fromtimestamp(at + 5).strftime("%Y-%m-%d %H:%M:%S")
    try:
        out = subprocess.run(["journalctl", "-u", "muhawir", "--since", since, "--until", until, "-o", "cat",
                              "--no-pager"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""
    lines = [ln.split("not answered:", 1)[1].strip() for ln in out.splitlines() if "not answered:" in ln]
    return lines[-1][:200] if lines else ""


def diagnose(question: str, retriever) -> list[str]:
    """Why a question may have gone unanswered, by code alone."""
    notes = []
    plain = normalize(question)
    if _OTHER_LANG.search(plain):
        notes.append("asks for a language Muhawir does not answer in (the Arabic/English buttons now handle this)")
    if _ASKS_COUNT.search(question) and not quran_count.asked(question):
        notes.append("looks like a counting question the counter did not recognise: worth a new pattern")
    if re.search(r"حديث|احاديث|hadith", plain):
        notes.append("asks about a hadith: the wording may differ from the sources (closest-hadith reply applies)")
    if retriever is not None:
        hits = retriever.search(question, k=3)
        if not hits or hits[0].coverage < 0.34:
            best = f"{hits[0].coverage:.0%}" if hits else "none"
            notes.append(f"the user's own words match the sources weakly (best match {best}): dialect, spelling or "
                         "a term the books word differently; the model's search phrases decide")
        else:
            notes.append("the sources match the user's words ("
                         + "; ".join(f"{h.passage.location} {h.coverage:.0%}" for h in hits[:2])
                         + "): the model judged the passages not enough")
    return notes


def run_tests() -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                             capture_output=True, text=True, timeout=600)
    except Exception as exc:
        return f"not run ({exc})"
    last = [ln for ln in out.stdout.splitlines() if ln.strip()][-1:] or ["no output"]
    return last[0] if "No module named pytest" not in out.stderr else "not run (pytest is not installed)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--max", type=int, default=10, help="questions answered again at most (each costs one answer)")
    ap.add_argument("--retry", action="store_true", help="retry unanswered questions with the model (costs credits)")
    ap.add_argument("--dry", action="store_true", help="deprecated alias for the default read-only diagnosis")
    args = ap.parse_args()
    if args.retry and args.dry:
        ap.error("--retry and --dry cannot be used together")

    store = Path(os.environ.get("MUHAWIR_STORE", str(ROOT / "data" / "answers.db")))
    db = Path(os.environ.get("MUHAWIR_DB", str(ROOT / "data" / "muhawir.db")))
    url = service_url()
    REPORTS.mkdir(parents=True, exist_ok=True)
    tried_file = REPORTS / ".tried.json"
    try:
        tried = json.loads(tried_file.read_text())
    except Exception:
        tried = {}
    now = time.time()
    tried = {k: t for k, t in tried.items() if now - t < WEEK}

    retriever = None
    if db.exists():
        from muhawir.store import SqliteCorpus, SqliteRetriever
        retriever = SqliteRetriever(SqliteCorpus(db))

    rows = []
    if store.exists():
        con = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
        for at, key, reply in con.execute("SELECT at, key, reply FROM replies WHERE at > ? ORDER BY at",
                                          (now - args.hours * 3600,)):
            try:
                r = json.loads(reply)
            except ValueError:
                continue
            if r.get("status") == "abstained":
                rows.append((at, key, json.loads(key), r))
        con.close()

    lines = [f"Muhawir nightly review, {datetime.datetime.now():%Y-%m-%d %H:%M}",
             f"Not answered in the last {args.hours:g} hours: {len(rows)}", ""]
    fixed = still = skipped = 0
    for n, (at, raw_key, (question, style, lang), r) in enumerate(rows, 1):
        when = datetime.datetime.fromtimestamp(at).strftime("%m-%d %H:%M")
        lines.append(f"{n}. [{when}, {style}, {lang}] {question}")
        if r.get("understood"):
            lines.append(f"   understood as: {r['understood']}")
        reason = log_reason(at)
        if reason:
            lines.append(f"   log: {reason}")
        for note in diagnose(question, retriever):
            lines.append(f"   - {note}")
        if not args.retry:
            lines.append("   (diagnosis only: not answered again)")
        elif raw_key in tried:
            skipped += 1
            lines.append("   already answered again this week: not repeated")
        elif fixed + still >= args.max:
            skipped += 1
            lines.append(f"   not answered again tonight (limit {args.max})")
        else:
            tried[raw_key] = now
            try:
                out = post(f"{url}/api/ask_again", {"question": question, "style": style, "lang": lang})
                status = out["now"]["status"]
                if status == "answered":
                    fixed += 1
                    lines.append("   FIXED: answered on a second try, through every check; the new answer replaces "
                                 "the kept «not found»")
                else:
                    still += 1
                    lines.append(f"   still {status} on a second try: FOR REVIEW")
            except Exception as exc:
                still += 1
                lines.append(f"   could not ask again: {exc}")
            time.sleep(2)
        lines.append("")
    tried_file.write_text(json.dumps(tried))

    try:
        health = get(f"{url}/api/health")
        use = health.get("usage") or {}
        credit = health.get("credit_usd")
        lines.append(f"Service: up. Since the last restart: {use.get('questions', '?')} questions, rated clear "
                     f"{use.get('rated_clear', '?')}, needs clarity {use.get('rated_unclear', '?')}. "
                     f"Model credit left: {'$%.2f' % credit if isinstance(credit, (int, float)) else 'unknown'}")
    except Exception as exc:
        lines.append(f"Service: NOT REACHABLE ({exc})")
    lines.append(f"Automated tests: {run_tests()}")
    try:  # new questions every night, written by code from the sources, in the main dialects: no model, no cost
        test = subprocess.run([sys.executable, str(ROOT / "scripts" / "selftest.py")], cwd=ROOT,
                              capture_output=True, text=True, timeout=900)
        lines.append("")
        lines.append(test.stdout.strip() or f"Self-test did not run: {test.stderr.strip()[-200:]}")
    except Exception as exc:
        lines.append(f"Self-test did not run: {exc}")
    lines.append("")
    lines.append(f"Summary: {len(rows)} not answered; {fixed} fixed on their own; {still} for review; "
                 f"{skipped} not retried.")
    report = "\n".join(lines) + "\n"
    (REPORTS / f"{datetime.date.today():%Y-%m-%d}.txt").write_text(report, encoding="utf-8")
    (REPORTS / "latest.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
