"""Hold short conversations with a running Muhawir, the way people use the site, and report what broke.

Single questions (scripts/evaluate.py) do not show what happens one step later. Here every conversation
starts with a question, then:
  1. taps the question Muhawir suggested (sent exactly as the button sends it),
  2. says it did not understand («ما فهمت» / "I didn't understand"),
  3. asks why («ولماذا؟» / "Why?").
A step is a failure when Muhawir finds no question in it ("chat"), cannot be reached ("unavailable"),
or returns an invalid reply; and a weak result when the tapped suggestion or the re-explanation finds
nothing in the sources ("abstained": Muhawir suggested a question it could not answer).

Usage (with the server running, on the server):
  .venv/bin/python scripts/try_dialogues.py --url http://127.0.0.1:8000 [--out dialogues.txt]
Exit code 1 when any step failed.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

# (opening question, style, language)
OPENINGS = [
    ("الزكاة", "youth", "ar"),
    ("ما معنى التوحيد؟", "newcomer", "ar"),
    ("كيف أتوضأ؟", "kids", "ar"),
    ("ما الفرق بين الفرض والسنة؟", "youth", "ar"),
    ("ما هي الشفاعة؟", "youth", "ar"),
    ("هل لمس المرأة ينقض الوضوء؟", "extended", "ar"),
    ("ما سبب نزول سورة الكوثر؟", "youth", "ar"),
    ("لماذا خلق الله الشر والمرض إذا كان رحيمًا؟", "youth", "ar"),
    ("الصيام", "kids", "ar"),
    ("Why would a merciful God allow suffering?", "newcomer", "en"),
]
NOT_UNDERSTOOD = {"ar": "ما فهمت", "en": "I didn't understand"}
WHY = {"ar": "ولماذا؟", "en": "Why?"}
FAIL = {"chat", "unavailable", "invalid"}


def ask(url: str, question: str, style: str, lang: str, history: list[dict]) -> tuple[dict, float]:
    body = json.dumps({"question": question, "style": style, "lang": lang, "history": history}).encode()
    req = urllib.request.Request(f"{url}/api/ask", data=body, headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()), time.time() - started


def said(res: dict) -> str:
    return " ".join([res.get("message") or ""] + [c["text"] for c in res.get("claims", [])]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", default="muhawir_dialogues.txt")
    args = parser.parse_args()
    lines: list[str] = []
    failed, weak, steps = [], [], 0

    def out(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    for n, (opening, style, lang) in enumerate(OPENINGS, 1):
        out(f"\n######## C{n:02d} [{style} · {lang}]")
        history: list[dict] = []
        plan = [("opening", opening)]
        i = 0
        while i < len(plan):
            step, text = plan[i]
            steps += 1
            try:
                res, secs = ask(args.url, text, style, lang, history)
            except Exception as exc:  # keep going: one failure must not hide the rest
                out(f"--- {step}: {text}\nERROR: {exc}")
                failed.append(f"C{n:02d} {step}: request error")
                break
            status = res.get("status", "?")
            out(f"--- {step}: {text}\nstatus: {status} ({secs:.0f}s)")
            if res.get("understood"):
                out(f"understood as: {res['understood']}")
            if said(res):
                out(said(res)[:700])
            if res.get("note"):
                out(f"note: {res['note']}")
            if status in FAIL:
                failed.append(f"C{n:02d} {step} «{text}»: {status}")
            elif status == "abstained" and step in ("tapped suggestion", "not understood"):
                weak.append(f"C{n:02d} {step} «{text}»: abstained")
            history = [{"role": "user", "text": res.get("understood") or text},
                       {"role": "assistant", "text": said(res)}]
            if step == "opening":
                if res.get("follow_up"):
                    plan.append(("tapped suggestion", res["follow_up"]))
                else:
                    out("(no suggested question)")
                plan += [("not understood", NOT_UNDERSTOOD[lang]), ("why", WHY[lang])]
            i += 1

    out("\n======== summary")
    out(f"{steps} steps in {len(OPENINGS)} conversations; failed: {len(failed)}; weak: {len(weak)}")
    for f in failed:
        out(f"FAILED  {f}")
    for w in weak:
        out(f"weak    {w}")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nreport written to {args.out}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
