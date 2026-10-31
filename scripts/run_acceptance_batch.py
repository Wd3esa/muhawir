"""One explicit five-question production batch: answer/source review and measured latency/cost.

One sequential question followed by four simultaneous different questions. No
HTTP retries, cache deletion or automatic answer regrading. Default is offline.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from decimal import Decimal
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyze_request_traces import read_lines, summarize

PRIVATE = ROOT / "data/private/measurement"
CASES = [
    {"id": "B1", "style": "youth", "question": "بيّن معنى التوكل على الله وعلاقته بالأسباب المأذون فيها، مع نسبة القول لصاحبه.",
     "checks": ["تعريف بلا تكرار", "قيد الأسباب المأذون فيها", "صحة نسبة القول"]},
    {"id": "B2", "style": "youth", "question": "اذكر الأقوال المنقولة في حكم صلاة الجماعة، وانسب كل قول إلى صاحبه دون ترجيح.",
     "checks": ["حفظ الخلاف وأصحابه", "عدم نقل دليل الصحة إلى الوجوب", "صلة التفاصيل بالسؤال"]},
    {"id": "B3", "style": "extended", "question": "ما المقصود بالحول في الزكاة، وما الأموال التي يشترط فيها، وما الاستثناءات؟",
     "checks": ["المعنى والأموال والاستثناءات", "عدم تصحيح رواية ضعّفها المصدر", "مطابقة الاقتباس"]},
    {"id": "B4", "style": "youth", "question": "ما حكم العادة السرية ومشاهدة الإباحية، وما درجة الحديث الذي فيه ناكح يده حسب المصدر؟",
     "checks": ["تغطية المسألتين", "حفظ درجة الرواية", "عدم نقل أحكام مسائل مجاورة"]},
    {"id": "B5", "style": "kids", "question": "اشرح لطفل معنى الأمانة، مع مثال قصير من حياته، بالاعتماد على المصادر.",
     "checks": ["وضوح المعنى للصغار", "مثال لا يضيف حكمًا شرعيًا", "سند المعنى والشرح"]},
]
CONFIG_KEYS = frozenset(("MUHAWIR_TRACE", "MUHAWIR_PARALLEL_CHECKS", "MUHAWIR_SEARCH_THREADS",
                        "MUHAWIR_ASK_CONCURRENCY", "MUHAWIR_ASK_QUEUE_SIZE", "MUHAWIR_READ_EACH_SENTENCE",
                        "MUHAWIR_VECTORS", "MUHAWIR_VECTOR_INDEX", "MUHAWIR_USEARCH_EXPANSION_SEARCH",
                        "OPENAI_COMPAT_MODEL", "OPENAI_COMPAT_REASONING", "OPENAI_COMPAT_REASONING_CHECK",
                        "OMP_NUM_THREADS", "MKL_NUM_THREADS"))


def public_config(raw):
    out = {}
    for entry in raw.split(b"\0"):
        key, sep, value = entry.partition(b"=")
        name = key.decode("utf-8", errors="replace")
        if sep and name in CONFIG_KEYS:
            out[name] = value.decode("utf-8", errors="replace")
    return out


def revision():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "muhawir").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def service():
    result = subprocess.check_output(["systemctl", "show", "muhawir", "--property=InvocationID",
        "--property=MainPID", "--property=NRestarts", "--property=MemoryCurrent", "--property=MemoryPeak"], text=True)
    return dict(line.split("=", 1) for line in result.splitlines() if "=" in line)


def journal(invocation):
    return subprocess.check_output(["sudo", "-n", "journalctl", "_SYSTEMD_INVOCATION_ID=" + invocation,
        "--no-pager", "-o", "cat"], text=True, timeout=20).splitlines()


def selected_lines(lines, ids):
    selected = []
    for line in lines:
        for marker in ("muhawir_trace ", "muhawir_trace_event ", "muhawir_trace_drained "):
            if marker in line:
                row = json.loads(line.split(marker, 1)[1])
                if row["request_id"] in ids:
                    selected.append(line)
                break
    return selected


def capture(invocation, ids, read_journal=journal, wait_seconds=20):
    deadline, last, rows = time.monotonic() + wait_seconds, [], []
    while ids:
        last = selected_lines(read_journal(invocation), ids)
        rows = read_lines(last) if any("muhawir_trace " in line for line in last) else []
        if {r["request_id"] for r in rows} == ids and all(r["background_complete"] for r in rows):
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(.5)
    return rows, last


def ask(client, case):
    started = time.perf_counter_ns()
    item = {"case": case, "phase": "sequential" if case["id"] == "B1" else "parallel_4",
            "started_unix_ns": time.time_ns()}
    try:
        response = client.post("/api/ask", json={"question": case["question"], "style": case["style"],
                                               "lang": "ar", "history": []})
        item["http_status"] = response.status_code
        try:
            item["response"] = response.json()
        except ValueError:
            item["error_class"] = "NonJsonResponse"
    except Exception as exc:
        item["error_class"] = type(exc).__name__
    item["client_seconds"] = (time.perf_counter_ns() - started) / 1e9
    return item


def request_ids(results):
    return {r["response"]["request_id"] for r in results
            if isinstance(r.get("response"), dict) and r["response"].get("request_id")}


def evidence(results, corpus):
    out = {}
    for result in results:
        response = result.get("response") or {}
        if not isinstance(response, dict):
            continue
        cited = {pid for c in response.get("claims", []) + response.get("views", [])
                 for pid in c.get("passage_ids", [])}
        cards = {s["passage_id"] for s in response.get("sources", [])}
        result["citation_ids_missing_from_cards"] = sorted(cited - cards)
        for pid in cited | cards:
            if pid in out:
                continue
            p = corpus.passage(pid)
            out[pid] = None if p is None else {"passage": asdict(p), "source": asdict(corpus.source_of(p)),
                "text_sha256": hashlib.sha256(p.text.encode()).hexdigest()}
    return out


def run(client, invocation, read_journal=journal, threshold=Decimal("0.10"), checkpoint=lambda data: None):
    results = [ask(client, CASES[0])]
    checkpoint({"results": results, "status": "sequential_returned"})
    ids = request_ids(results)
    rows, lines = capture(invocation, ids, read_journal) if ids else ([], [])
    summary = summarize(rows) if rows else None
    # No new paid questions if the first request cannot be attributed/costed precisely.
    if (not summary or len(ids) != 1 or results[0].get("http_status") != 200 or
            not summary["cost_complete_for_observed_calls"] or
            summary["code_revision"] != revision()):
        return results, rows, lines, "stopped_incomplete_first_measurement"
    if Decimal(summary["known_reported_cost_usd"]) >= threshold:
        return results, rows, lines, "stopped_after_first_reported_cost_threshold"
    began = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(ask, client, case) for case in CASES[1:]]
        for future in as_completed(futures):
            results.append(future.result())
            checkpoint({"results": results, "status": "parallel_request_returned"})
    wall = (time.perf_counter_ns() - began) / 1e9
    ids = request_ids(results)
    rows, lines = capture(invocation, ids, read_journal) if ids else ([], [])
    checkpoint({"results": results, "parallel_wall_seconds": wall, "status": "requests_returned"})
    complete = (len(ids) == 5 and len(rows) == 5 and summarize(rows)["cost_complete_for_observed_calls"])
    return results, rows, lines, "completed_pending_quality_review" if complete else "completed_with_measurement_gaps"


def fingerprints():
    paths = [ROOT / "data" / name for name in ("muhawir.db", "vectors.npy", "vectors_ids.json",
                                             "vectors_model.txt", "vectors.usearch", "vectors.usearch.json")]
    out = {}
    for path in paths:
        if not path.exists():
            continue
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source/index file changed during fingerprinting")
        out[str(path.relative_to(ROOT))] = {"sha256": digest.hexdigest(), "bytes": after.st_size,
                                           "mtime_ns": after.st_mtime_ns}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("OFFLINE PLAN: 5 question requests, one sequential then four concurrent; no retries.")
        print("Post-response stop threshold USD 0.10 is not a provider spending cap.")
        print("Four concurrent requests may all finish and be charged. Quality verdicts stay pending.")
        print(json.dumps(CASES, ensure_ascii=False, indent=2))
        return
    import httpx
    from muhawir.store import SqliteCorpus
    before = service()
    if not before.get("InvocationID") or before.get("MainPID") in (None, "0"):
        raise ValueError("service is not running")
    config = public_config(Path(f"/proc/{before['MainPID']}/environ").read_bytes())
    if config.get("MUHAWIR_TRACE") != "1":
        raise ValueError("per-request tracing is not enabled; no questions sent")
    journal(before["InvocationID"])  # verify noninteractive read access before spending
    with httpx.Client(base_url="http://127.0.0.1:8000", transport=httpx.HTTPTransport(retries=0),
                      trust_env=False, follow_redirects=False, timeout=180) as client:
        h = client.get("/api/health")
        h.raise_for_status()
        health = h.json()
        if health.get("code_revision") != revision() or health.get("synthetic"):
            raise ValueError("running code/corpus differs from this checkout")
        corpus = SqliteCorpus(ROOT / "data/muhawir.db")
        if len(corpus.passages) != health.get("passages"):
            raise ValueError("corpus size differs from the running service")
        frozen = fingerprints()
        PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
        marker = PRIVATE / "acceptance-batch-v1.started"
        out = PRIVATE / f"acceptance-batch-{time.time_ns()}.json"
        with os.fdopen(os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w+") as stream:
            bundle = {"scope": "five_request_acceptance_and_latency_sample_not_capacity",
                      "quality_review_status": "pending", "cases": CASES, "health_before": health,
                      "service_before": before, "source_fingerprints": frozen,
                      "explicit_nonsecret_service_config": config,
                      "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "limitations": ["Small exposed sample; no general reliability, tail or sustained capacity estimate.",
                          "First request warm/cold state is unknown; sources are fingerprinted before timing starts.",
                          "Fingerprinting warms OS file caches; timing is full loopback HTTP JSON, not browser or visitor-network latency.",
                          "Cached and new answers must be analyzed separately; no cache is cleared.",
                          "Different questions/styles in the two phases do not establish a causal concurrency slowdown or a before/after improvement.",
                          "At most five question HTTP requests, not a bound on their internal model calls or billing.",
                          "USD 0.10 is checked after the first response, not a hard spending cap; four requests then run in flight.",
                          "Timeout does not cancel server work; unmatched requests have unknown cost, never zero.",
                          "MemoryPeak covers service lifetime, and other visitors may affect resource use."]}
            printed = set()
            def checkpoint(data):
                bundle.update(data)
                stream.seek(0)
                stream.truncate()
                json.dump(bundle, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
                for result in bundle.get("results", []):
                    cid = result["case"]["id"]
                    if cid not in printed:
                        reply = result.get("response") or {}
                        status = reply.get("status") if isinstance(reply, dict) else "invalid_response"
                        print(cid, "HTTP", result.get("http_status"), "status", status,
                              "seconds", round(result["client_seconds"], 3), flush=True)
                        printed.add(cid)
            checkpoint({"status": "preflight_complete", "results": []})
            os.close(os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            print("Batch started; do not repeat. File:", out, flush=True)
            try:
                results, rows, lines, status = run(client, before["InvocationID"], checkpoint=checkpoint)
                after = service()
                stable = before["InvocationID"] == after["InvocationID"] and before["MainPID"] == after["MainPID"]
                unchanged = all((ROOT / name).stat().st_size == item["bytes"] and
                                (ROOT / name).stat().st_mtime_ns == item["mtime_ns"] for name, item in frozen.items())
                summary = summarize(rows) if rows else None
                code_stable = revision() == health["code_revision"]
                phases = {}
                for phase in ("sequential", "parallel_4"):
                    own_ids = request_ids([r for r in results if r["phase"] == phase])
                    own_rows = [r for r in rows if r["request_id"] in own_ids]
                    phases[phase] = summarize(own_rows) if own_rows else None
                if summary:
                    summary.pop("groups")  # never pool sequential/concurrent timing distributions
                if not stable or not unchanged or not code_stable:
                    status = "invalid_environment_changed"
                checkpoint({"status": status, "service_after": after, "service_stable": stable,
                            "source_files_unchanged": unchanged, "code_files_unchanged": code_stable,
                            "traces": rows, "trace_lines": lines,
                            "measurement": summary, "phase_measurements": phases,
                            "cost_complete_for_batch": bool(summary and summary["cost_complete_for_observed_calls"] and
                                                            len(rows) == len(request_ids(results)) == len(results)),
                            "client_requests_without_trace_id": sum(not isinstance(r.get("response"), dict) or
                                                                     not r["response"].get("request_id") for r in results)})
                checkpoint({"source_snapshots": evidence(results, corpus)})
            except Exception as exc:
                checkpoint({"status": "stopped_no_retry", "error_class": type(exc).__name__})
        print("Status:", bundle["status"])
        print("Questions completed:", len(bundle.get("results", [])), "/ 5")
        if bundle.get("measurement"):
            print("Measurement:", json.dumps(bundle["measurement"], ensure_ascii=False))
        print("Quality: pending source-by-source review. Results:", out)


if __name__ == "__main__":
    main()
