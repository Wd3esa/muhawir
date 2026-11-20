"""Frozen positive/negative diagnostic; offline by default. Never calls /api/ask.

Measures current support+relevance versus the inactive joint reviewer on the same
drafts. This is a small, exposed, assistant-labelled diagnostic, not release proof.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muhawir import generate as g, joint_review as j
from muhawir.corpus import Corpus, Passage, Source
from muhawir.verify import Claim, verify
from scripts.prepare_joint_review import corpus_from_case, prepare_recording

PRIVATE = ROOT / "data/private/measurement"
MAX_HTTP = 48


def sha(data):
    return hashlib.sha256(data).hexdigest()


def code_hashes():
    files = sorted((ROOT / "muhawir").glob("*.py")) + [Path(__file__), ROOT / "scripts/prepare_joint_review.py"]
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in files}


def build_plan(recording):
    prepare_recording(recording)  # checks complete original recording and embedded evidence hashes
    rows = [json.loads(line) for line in recording.read_bytes().splitlines() if line.strip()]
    original = rows[0]["plan"]
    contexts = {c["id"]: corpus_from_case(c) for c in original["cases"]}
    cases = []

    def pair(name, question, origin, ids, good, bad, basis, local_only=False):
        corpus = contexts[origin]
        sources = {corpus.passage(pid).source_id for pid in ids}
        subset = {"sources": [asdict(corpus.sources[s]) for s in sorted(sources)],
                  "passages": [asdict(corpus.passage(pid)) for pid in ids]}
        if not any(basis in p["text"] for p in subset["passages"]):
            raise ValueError("reference basis not present")
        for label, texts in (("valid", good), ("mutant", bad)):
            claims = [{"text": text, "passage_ids": pids, "school": "", "section": "", "label": ""}
                      for text, pids in texts]
            cases.append({"id": f"{name}-{label}", "family": name, "question": question,
                          "corpus": subset, "claims": claims, "basis": basis,
                          "expected": "accept" if label == "valid" else "reject",
                          "local_only": local_only,
                          "order": ["current", "joint"] if (len(cases) + len(cases) // 2) % 2 == 0
                                   else ["joint", "current"]})

    pair("inference", "ما وجه الاستدلال بحديث تفاضل الصلاة في النص؟", "R08", ["u:5359"],
         [("ذكر ابن عثيمين أن وجه الاستدلال بحديث تفاضل الصلاة هو صحة صلاة الفرد.", ["u:5359"])],
         [("ذكر ابن عثيمين أن وجه الاستدلال بحديث تفاضل الصلاة هو وجوب صلاة الجماعة.", ["u:5359"])],
         "ولو كانت صلاة الفذ لا تصح ما كان فيها فضل")
    pair("condition", "كيف قيد ابن باز السعي في طلب الرزق في النص؟", "R01", ["z:1790"],
         [("ذكر ابن باز وجوب السعي في طلب الرزق والعمل مع القدرة على ذلك.", ["z:1790"])],
         [("ذكر ابن باز وجوب السعي في طلب الرزق والعمل ولو عجز الإنسان عن ذلك.", ["z:1790"])],
         "مع القدرة على ذلك")
    pair("grading", "ما حكم ابن باز على الحديث الذي ورد فيه ناكح يده؟", "R06", ["z:8127"],
         [("بيّن ابن باز أن الحديث الذي فيه السبعة ومنهم ناكح يده ضعيف غير صحيح.", ["z:8127"])],
         [("بيّن ابن باز أن الحديث الذي فيه السبعة ومنهم ناكح يده صحيح ثابت.", ["z:8127"])],
         "فهو ضعيف غير صحيح عند أهل العلم")
    pair("attribution", "ماذا قال صاحب الفتوى عن أخذ مريم بالأسباب؟", "R01", ["z:1790"],
         [("ذكر ابن باز أن مريم لم تترك الأسباب، بل هزت النخلة وأخذت بالأسباب.", ["z:1790"])],
         [("ذكر ابن عثيمين أن مريم لم تترك الأسباب، بل هزت النخلة وأخذت بالأسباب.", ["z:1790"])],
         "ومريم رحمة الله عليها لم تدع الأسباب")
    pair("disagreement", "ما الأقوال التي ينقلها النص في حكم صلاة الجماعة؟", "R08", ["f:289"],
         [("اختلف العلماء؛ فالجمهور يقولون سنة أو فرض كفاية، والظاهرية يقولون فرض متعين على كل مكلف.", ["f:289"])],
         [("اتفق العلماء على أن صلاة الجماعة فرض متعين على كل مكلف.", ["f:289"])],
         "فإن العلماء اختلفوا فيها")
    first = ("ذكر ابن باز أن العادة السرية منكر لا تجوز.", ["z:8127"])
    second = ("حذر ابن باز الصائمين وغيرهم في جميع الأوقات من الأفلام التي تعرض الصور العارية.", ["z:5863"])
    pair("coverage", "ما حكم العادة السرية ومشاهدة الأفلام الخليعة بحسب ابن باز؟", "R06",
         ["z:8127", "z:5863"], [first, second], [first],
         "الواجب على الصائمين وغيرهم")
    pair("composite_quote", "ما الحديث المذكور في النص عن السعي والاستعانة؟", "R01", ["z:1790"],
         [("ورد في الفتوى أن النبي ﷺ قال: «احرص على ما ينفعك واستعن بالله ولا تعجزن».", ["z:1790"])],
         [("ورد في الفتوى أن النبي ﷺ قال: «احرص على ما ينفعك واستعن بالله ولا تعجزن وهزي إليك بجذع النخلة».", ["z:1790"])],
         "احرص على ما ينفعك واستعن بالله ولا تعجزن", local_only=True)
    return {"scope": "paired_checker_diagnostic_not_release", "reference_status": "assistant_proposed_exposed_controls",
            "recording_sha256": sha(recording.read_bytes()), "code_sha256": code_hashes(),
            "prompt_sha256": {"support": sha(g.CHECK_PROMPT.encode()), "relevance": sha(g.RELEVANCE_PROMPT.encode()),
                              "joint": sha(j.REVIEW_PROMPT.encode())},
            "requested_model": original["requested_model"], "provider": "Together", "max_http_attempts": MAX_HTTP,
            "read_each_sentence": g.READ_EACH_SENTENCE, "parallel_checks": g.MAX_PARALLEL_CHECKS,
            "cases": cases}


def unpack(case):
    sources = {s["id"]: Source(**s) for s in case["corpus"]["sources"]}
    corpus = Corpus(sources, [Passage(**p) for p in case["corpus"]["passages"]])
    claims = [Claim(c["text"], tuple(c["passage_ids"]), c["school"], c["section"], c["label"])
              for c in case["claims"]]
    return claims, corpus, {p.id for p in corpus.passages}


def validate(plan):
    if plan["scope"] != "paired_checker_diagnostic_not_release" or plan["max_http_attempts"] != MAX_HTTP:
        raise ValueError("wrong experiment or attempt bound")
    if plan["code_sha256"] != code_hashes():
        raise ValueError("code differs from frozen plan")
    if plan["read_each_sentence"] != g.READ_EACH_SENTENCE or plan["parallel_checks"] != g.MAX_PARALLEL_CHECKS:
        raise ValueError("checker configuration differs from frozen plan")
    cases = plan["cases"]
    if len(cases) != 14 or len({c["id"] for c in cases}) != 14:
        raise ValueError("expected seven positive/negative control pairs")
    families = Counter(c["family"] for c in cases)
    if set(families.values()) != {2} or sum(c["expected"] == "accept" for c in cases) != 7:
        raise ValueError("unbalanced controls")
    for case in cases:
        claims, corpus, ids = unpack(case)
        _, rejected = verify(claims, corpus, ids)
        if case["local_only"]:
            if bool(rejected) != (case["expected"] == "reject"):
                raise ValueError("unexpected quotation preflight")
        else:
            if rejected:
                raise ValueError("semantic control unexpectedly fails local verification")
            j.prepare_review(case["question"], claims, corpus, ids)
        if sorted(case["order"]) != ["current", "joint"]:
            raise ValueError("unpaired control")


class StopRun(BaseException):
    """Not caught by legacy retry/fallback loops: never turn a network failure into more spending."""


def reported_cost(data):
    usage = data.get("usage") if isinstance(data, dict) else None
    value = usage.get("cost") if isinstance(usage, dict) else None
    if value is None or isinstance(value, bool):
        return None
    try:
        value = Decimal(str(value))
        return value if value.is_finite() and value >= 0 else None
    except Exception:
        return None


class Meter:
    def __init__(self, send, emit, threshold, maximum=MAX_HTTP):
        self.send, self.emit, self.threshold, self.maximum = send, emit, threshold, maximum
        self.lock = threading.Lock()
        self.attempts, self.spent, self.stopped = 0, Decimal(0), ""

    def request(self, case_id, lane, system, user, schema):
        with self.lock:
            if self.stopped:
                raise StopRun(self.stopped)
            if self.attempts >= self.maximum:
                self.stopped = "http_attempt_limit"
                raise StopRun(self.stopped)
            self.attempts += 1
            number = self.attempts
            self.emit({"kind": "attempt_started", "number": number, "case_id": case_id, "lane": lane,
                       "stage": "support" if system == g.CHECK_PROMPT else
                                "relevance" if system == g.RELEVANCE_PROMPT else "joint_review",
                       "system_sha256": sha(system.encode()), "user_sha256": sha(user.encode())})
        started = time.perf_counter_ns()
        try:
            data = self.send(system, user, schema)
        except Exception as exc:
            with self.lock:
                self.stopped = "transport_failure_cost_unknown"
                self.emit({"kind": "transport_failure", "number": number, "case_id": case_id, "lane": lane,
                           "seconds": (time.perf_counter_ns() - started) / 1e9,
                           "error_class": type(exc).__name__, "cost_status": "unknown",
                           "http_status": getattr(getattr(exc, "response", None), "status_code", None)})
            raise StopRun(self.stopped) from None
        seconds = (time.perf_counter_ns() - started) / 1e9
        cost = reported_cost(data)
        with self.lock:
            self.emit({"kind": "provider_response", "number": number, "case_id": case_id, "lane": lane,
                       "seconds": seconds, "response": data,
                       "reported_cost_usd": str(cost) if cost is not None else None})
            if cost is None:
                self.stopped = "unknown_cost"
            else:
                self.spent += cost
                if self.spent >= self.threshold:
                    self.stopped = "post_response_cost_threshold"
            choices = data.get("choices", []) if isinstance(data, dict) else []
            if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
                    or choices[0].get("finish_reason") != "stop"):
                self.stopped = self.stopped or "incomplete_provider_response"
            if not self.stopped and (not isinstance(choices[0].get("message"), dict)
                                    or not isinstance(choices[0]["message"].get("content"), str)):
                self.stopped = "invalid_provider_content"
            if not self.stopped and data.get("provider") != "Together":
                self.stopped = "provider_changed"
            if self.stopped:
                raise StopRun(self.stopped)
        return choices[0]["message"]["content"]


def current_review(case, claims, corpus, meter):
    def call(system, user, schema):
        if system not in (g.CHECK_PROMPT, g.RELEVANCE_PROMPT):
            raise StopRun("unexpected_legacy_stage")
        return meter.request(case["id"], "current", system, user, schema)
    generator = g.ModelGenerator([("paired-diagnostic", call)])
    by_id = {p.id: p for p in corpus.passages}
    flags = generator.check_support(claims, by_id, case["question"])
    if flags is None or len(flags) != len(claims):
        return "unavailable", {"reason": "support_check_unavailable"}
    if not all(flags):
        return "reject", {"support": flags, "reasons": generator.last_check_reasons}
    relevance = generator.judge_relevance(case["question"], claims, by_id)
    return ("accept" if relevance == "yes" else "reject" if relevance in ("partly", "no") else "unavailable"), {
        "support": flags, "relevance": relevance}


def run(plan, meter, emit):
    validate(plan)
    try:
        for case in plan["cases"]:
            claims, corpus, ids = unpack(case)
            if case["local_only"]:
                _, rejected = verify(claims, corpus, ids)
                emit({"kind": "local_result", "case_id": case["id"], "family": case["family"],
                      "expected": case["expected"], "decision": "reject" if rejected else "accept",
                      "reasons": [r.reason for r in rejected]})
                continue
            for lane in case["order"]:
                started, before = time.perf_counter_ns(), meter.attempts
                if lane == "current":
                    decision, detail = current_review(case, claims, corpus, meter)
                else:
                    def call(system, user, schema):
                        return j.Completion(meter.request(case["id"], lane, system, user, schema), "stop")
                    out = j.evaluate(case["question"], claims, corpus, ids, call, allow_repair=False)
                    decision = ("accept" if out.status == "accepted_for_evaluation" else
                                "unavailable" if out.status == "unavailable" else "reject")
                    if out.status == "needs_sources":
                        decision = "needs_sources"
                    elif any(any(c["verdict"] == "uncertain" for c in r.data["claims"]) or
                             any(c["status"] == "uncertain" for c in r.data["checks"].values())
                             for r in out.reviews):
                        decision = "uncertain"
                    detail = {"status": out.status, "error_type": out.error_type,
                              "reviews": [r.data for r in out.reviews]}
                emit({"kind": "case_result", "case_id": case["id"], "family": case["family"], "lane": lane,
                      "expected": case["expected"], "decision": decision, "detail": detail,
                      "seconds": (time.perf_counter_ns() - started) / 1e9,
                      "http_attempts": meter.attempts - before})
                if decision == "unavailable":
                    return "stopped_unavailable_judge_no_retry_batch"
        return "completed_diagnostic_not_release"
    except StopRun as exc:
        return "stopped_" + str(exc)


def save(path, data):
    path = path.resolve()
    if not path.is_relative_to(PRIVATE.resolve()):
        raise ValueError("output must stay within data/private/measurement")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def summarize(path):
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    if not rows or rows[0].get("kind") != "plan":
        raise ValueError("missing diagnostic plan")
    plan = rows[0]["plan"]
    if plan["scope"] != "paired_checker_diagnostic_not_release":
        raise ValueError("wrong recording")
    if sum(r["kind"] == "plan" for r in rows) != 1:
        raise ValueError("mixed recordings")
    starts = [r for r in rows if r["kind"] == "attempt_started"]
    if [r["number"] for r in starts] != list(range(1, len(starts) + 1)):
        raise ValueError("duplicate or missing attempt numbers")
    by_number = {r["number"]: r for r in starts}
    seen = set()
    for response in (r for r in rows if r["kind"] in ("provider_response", "transport_failure")):
        number = response["number"]
        if number not in by_number or number in seen:
            raise ValueError("duplicate or unmatched response")
        if any(response[key] != by_number[number][key] for key in ("case_id", "lane")):
            raise ValueError("crossed response identity")
        if response["kind"] == "provider_response":
            recorded = response["reported_cost_usd"]
            actual = reported_cost(response["response"])
            if (recorded is None) != (actual is None) or (recorded is not None and Decimal(recorded) != actual):
                raise ValueError("reported cost was altered")
        seen.add(number)
    variants = {}
    results = [r for r in rows if r["kind"] == "case_result"]
    if len({(r["case_id"], r["lane"]) for r in results}) != len(results):
        raise ValueError("duplicate result")

    def tokens(response, key):
        data = response["response"]
        usage = data.get("usage") if isinstance(data, dict) else None
        value = usage.get(key) if isinstance(usage, dict) else None
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    for lane in ("current", "joint"):
        counts = Counter()
        selected = [r for r in results if r["lane"] == lane]
        for result in selected:
            if result["decision"] in ("unavailable", "uncertain", "needs_sources"):
                counts[result["decision"]] += 1
            else:
                correct = result["decision"] == result["expected"]
                counts[("correct_" if correct else "false_") +
                       ("accept" if result["decision"] == "accept" else "reject")] += 1
        responses = [r for r in rows if r["kind"] == "provider_response" and r["lane"] == lane]
        starts = [r for r in rows if r["kind"] == "attempt_started" and r["lane"] == lane]
        known = sum((Decimal(r["reported_cost_usd"]) for r in responses
                     if r["reported_cost_usd"] is not None), Decimal(0))
        variants[lane] = {"finished_controls": len(selected), "expected_controls": 12,
                          "counts_against_proposed_references": dict(counts), "http_attempts": len(starts),
                          "known_reported_usd": str(known),
                          "unknown_cost_attempts": len(starts) - sum(r["reported_cost_usd"] is not None for r in responses),
                          "mean_finished_control_seconds": sum(r["seconds"] for r in selected) / len(selected) if selected else None,
                          "known_input_tokens": sum(tokens(r, "prompt_tokens") or 0 for r in responses),
                          "known_output_tokens": sum(tokens(r, "completion_tokens") or 0 for r in responses),
                          "responses_missing_token_usage": sum(tokens(r, "prompt_tokens") is None or
                                                               tokens(r, "completion_tokens") is None for r in responses)}
    return {"status": rows[-1].get("status") if rows[-1]["kind"] == "finished" else "incomplete_recording",
            "reference_status": plan["reference_status"], "variants": variants,
            "local_controls_finished": sum(r["kind"] == "local_result" for r in rows),
            "interpretation": "small exposed diagnostic; no general reliability or production latency claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "report"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-after-usd", default="0.10")
    args = parser.parse_args()
    if args.mode == "report":
        print(json.dumps(summarize(args.input), ensure_ascii=False, indent=2))
        return
    if args.mode == "prepare":
        plan = build_plan(args.input)
        validate(plan)
        path = args.out or PRIVATE / "checker-diagnostic-plan.json"
        save(path, plan)
        print("Prepared 6 semantic control pairs and 1 local quotation pair. No model calls.")
        print("HTTP attempt ceiling:", MAX_HTTP, "(including legacy rereads; repairs disabled)")
        print("References are assistant-proposed development controls, not independent certification.")
        print("Plan:", path.resolve())
        return
    raw = args.input.read_bytes()
    plan = json.loads(raw)
    validate(plan)
    if not args.execute:
        print("OFFLINE CHECK OK. No model calls. Plan SHA256:", sha(raw))
        return
    threshold = Decimal(args.stop_after_usd)
    if not threshold.is_finite() or threshold <= 0 or threshold > Decimal("0.10"):
        parser.error("post-response threshold must be positive and no more than USD 0.10")
    base = os.environ["OPENAI_COMPAT_BASE_URL"].rstrip("/")
    url = urlsplit(base)
    if url.scheme != "https" or url.hostname != "openrouter.ai" or url.username or url.password:
        parser.error("requires configured HTTPS OpenRouter endpoint")
    model = os.environ["OPENAI_COMPAT_MODEL"]
    if model != plan["requested_model"]:
        parser.error("configured model differs from plan")
    if os.environ.get("OPENAI_COMPAT_REASONING", "").lower().strip() not in ("off", "none", "false", "0"):
        parser.error("reasoning must be off")
    path = (args.out or PRIVATE / "checker-diagnostic-results.jsonl").resolve()
    if not path.is_relative_to(PRIVATE.resolve()):
        parser.error("output must stay private")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    import httpx
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        lock = threading.Lock()
        def emit(event):
            with lock:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        emit({"kind": "plan", "plan_sha256": sha(raw), "plan": plan,
              "post_response_stop_usd": str(threshold),
              "budget_note": "not a hard provider spending cap; already in-flight calls may finish"})
        marker = PRIVATE / ("checker-diagnostic-" + sha(raw) + ".started")
        os.close(os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        with httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=120,
                          trust_env=False, follow_redirects=False) as client:
            def send(system, user, schema):
                response = client.post(base + "/chat/completions", headers={
                    "authorization": "Bearer " + os.environ["OPENAI_COMPAT_API_KEY"]}, json={
                    "model": model, "temperature": 0, "max_tokens": 8192,
                    "provider": {"order": [plan["provider"]], "allow_fallbacks": False},
                    "reasoning": {"enabled": False}, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
                response.raise_for_status()
                return response.json()
            meter = Meter(send, emit, threshold)
            status = run(plan, meter, emit)
        emit({"kind": "finished", "status": status, "http_attempts": meter.attempts,
              "known_reported_cost_usd": str(meter.spent)})
    print("Status:", status)
    print("HTTP attempts:", meter.attempts, "Known reported USD:", meter.spent)
    print("Results:", path)
    print("Do not repeat the batch; analyze the saved responses first.")
    print(json.dumps(summarize(path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
