# Scalability work against the October 24 server snapshot

The supplied audit describes an older revision and includes unmeasured performance claims. The deployed server snapshot has 94,431 passages, a validated 1,024-dimensional float16 BGE-M3 matrix, and an existing single-request batch verification path. No answer-model calls were made while implementing this change.

| Audit recommendation | Result |
| --- | --- |
| Batch second-reader calls | Already available with `MUHAWIR_READ_EACH_SENTENCE=0`. The production default remains individual readings: a prior batched reading accepted a claim that merged two hadith narrations. Switching the default requires a measured claim-level safety evaluation with the actual provider, and will consume model quota. JSON schema alone does not prevent a false positive. |
| Concurrency | Four active model requests by default; at most 12 additional requests may wait 10 seconds. `MUHAWIR_ASK_CONCURRENCY`, `MUHAWIR_ASK_QUEUE_SIZE`, and `MUHAWIR_ASK_QUEUE_TIMEOUT` are adjustable. A timed-out waiter does not consume the global model-request budget. Setting 16 active requests would multiply upstream usage and exceed an unmeasured VM/provider capacity. |
| Vector ANN | Optional USearch index from the **existing** matrix (`requirements-ann.txt`, `python -m muhawir.vectors build-ann`). It carries a full source fingerprint and falls back to exact vector search on absence, staleness or query error. Only enable with `MUHAWIR_VECTOR_INDEX=usearch` after measuring retrieval recall against the existing exact index. Neither command nor install re-embeds the corpus. |
| SQLite WAL | Enabled for the writable answer cache. The corpus is opened read-only, and its builder atomically replaces an SQLite file; it does not need a WAL sidecar. No Redis account or external network service is needed for a single-node cache. |
| SSE answer streaming | Deferred. The answer pipeline verifies a complete draft and drops unsupported sentences before display. Streaming draft tokens would bypass that safety gate. A mere heartbeat would not make a verified answer arrive faster. |
| Corpus expansion | Deferred until vetted source dumps, provenance checks and evaluation are available. It changes passage IDs and requires matching corpus/vector validation; a vector build must not be started automatically from a scalability audit. |

On a staging server, first compare the existing exact search with `HF_HUB_OFFLINE=1 MUHAWIR_VECTORS=1 .venv/bin/python scripts/eval_retrieval.py` (omit `--pipeline`: that option uses answer-model calls). Install the optional dependency with `.venv/bin/pip install -r requirements-ann.txt`, then build the ANN index in a separate process, without changing `MUHAWIR_VECTOR_INDEX`. Repeat the same retrieval evaluation with `MUHAWIR_VECTOR_INDEX=usearch`; inspect both hit rates and latency before configuring it on the service. A quick side-by-side result is not a substitute for the source-grounding checks.

For a single offline diagnostic report, run `.venv/bin/python scripts/eval_search_candidates.py` on Oracle. It checks which labeled passages still exist in the current database and reports each question's rank under BM25, exact hybrid, a wider BM25 candidate pool, USearch, and USearch with exact shortlist reranking. It uses the existing local embedding model, never calls the answer model, and does not alter the live setting. The production search defaults are unchanged; enable a candidate strategy only after its gains hold on a larger reviewed set, including Arabic, English and Urdu questions and alternate valid citations.

The new answer-cache WAL files (`answers.db-wal`, `answers.db-shm`) are runtime files and should remain alongside the cache database. `scripts/forget.py` is read-only by default and can remove cached replies matching a question phrase with `--delete`.

The fourfold BM25 candidate pool is experimental and **off by default** (`MUHAWIR_BM25_POOL_FACTOR=1`). The first October 24 offline evaluation requested 30 returned passages and found a labeled one in the top 10 for 4/20 questions with the wider pool, versus 3/20 previously. A subsequent real-corpus review requested **10 returned passages**, matching the visible review depth: both settings found labeled passages for 3/20 questions (9 reference IDs were absent). The wider pool moved a passage addressing “من خلق الله؟” from second to fifth, and general ablution guidance for “كيف أتوضأ؟” from fourth to sixth. Other passages improved: a definition of intercession moved from eighth to third. The apparent 30-passage gain does not establish a benefit at the displayed depth, nor does the timing in sequential, warmed runs prove a latency gain. Keep exact vectors; USearch alone also lost a labeled passage. A future reviewed, multilingual labeled set may justify opt-in experiments with `MUHAWIR_BM25_POOL_FACTOR=4`.

For a human review of more questions on the real corpus, run `.venv/bin/python scripts/review_search.py` on Oracle. It saves `/tmp/muhawir-search-review.json`, comparing the first 10 passages under the default pool (`old`) and experimental wider pool (`new`), with their book, location, text and currently labeled gold IDs. The default includes 20 labeled questions and 12 extra questions from the dialogue review; use `--questions /path/to/questions.txt` for additional questions, one per line. A reviewer must mark which passages actually answer each question; unlabeled questions are not included in the reported hit rate. This uses local embeddings offline and makes zero answer-model requests. Review custom questions for personal information before sharing the report.

## Paired, warmed retrieval timing

The October 29 Oracle comparison on 193,246 passages reported median 446 ms for exact search and 68 ms for USearch, with 3/20 reference hits in the top 10 under both. Nine reference IDs were absent, but every question still had at least one present reference. Those are missing reference passages, not evidence that nine books are missing. The measured ratio is provisional: `eval_search_candidates.py` computes and caches each question embedding during its exact run, then later ANN runs reuse those embeddings. Its previous warm-up only embedded the warm-up question, not the evaluated questions.

Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/benchmark_retrieval.py --repeats 3 --out /tmp/muhawir-retrieval-review.json` for a fairer comparison. It requires the existing local vectors and a matching USearch index, validates corpus/vector ID order, computes query embeddings once outside retrieval timing, warms all modes on the same questions, then alternates mode order across paired trials. It reports median and p95 **warmed retrieval** time separately from query embedding time, and refuses to label fallback-to-exact timing as ANN. It never imports the answer pipeline or changes the running service.

Gold-hit columns describe the first measured trial only; variation across trials is reported. The optional JSON preserves every trial's reference rank, missing reference IDs, full present-reference texts and full retrieved passages with their books and locations. Keep that report outside Git. Review the retrieved evidence and refresh the labels before deciding to enable ANN. Equal 3/20 hits do not establish that overall retrieval quality is acceptable. These isolated, warmed searches also do not measure the full multi-query question pipeline or production capacity under concurrency. No answer-model calls are made.

## October 29 ANN diagnosis and candidate rebuild

The original graph loaded with search expansion 64 even though its builder requested 96.
Increasing search expansion recovered most missing neighbors, but at 4096 `f:322` was
still absent from 160 results despite being the exact search's first neighbor. Its stored
vector was identical to the source row, yet searching by that vector itself did not find
its key. This establishes a failure to retrieve an existing vector, not a missing source
or a simple change of rank; it does not identify the underlying library/graph defect.

A separate graph built from the unchanged 193,246 vectors with shuffled insertion
(seed 42), connectivity 32, addition expansion 256 and two build threads found that
key first in self-search. After save/restore, with search expansion explicitly set to
1024, its warmed hybrid top-10 sets matched exact search for **200/200 passages across
20 questions**. Median retrieval was 79 ms versus 305 ms exact; p95 was 466 ms versus
690 ms. Exact shortlist reranking gave the same set overlap and median 80 ms. This
is a small, Arabic retrieval sample: it does not prove answer quality, multilingual
recall, complete ranking equivalence, full-answer latency or concurrent-user capacity.
The reference hit rate remained 3/20 at depth 10, with nine absent reference IDs.

`build-ann` now uses those candidate build settings, preserving source-row keys through
the shuffle. The loader explicitly reapplies search expansion 1024 after restore, so
serialization cannot silently return to the library's smaller search default. These
changes do not enable ANN automatically, repair an old graph, or modify embeddings.
Keep `MUHAWIR_VECTOR_INDEX=exact` until a validated rebuilt graph is installed. Retain
both the old graph and its manifest when replacing them, and keep the fingerprint
manifest bound to the exact source matrix, ID list and model. Reverting the service to
`MUHAWIR_VECTOR_INDEX=exact` restores exhaustive vector retrieval without re-embedding.

## CPU query-embedding contention

After installing the validated graph and enabling USearch, one new answer took 17.92 s
(search 5.6 s). Two simultaneous new questions completed in 16.22/27.34 s; four in
22.95/26.66/29.69/36.91 s. All answered, with no PID change or restart during the four
requests. The lifetime service memory peak rose to about 1.09 GiB. The four search
stages took 15–20.6 s. Different questions and rewrites prevent interpreting the
latency differences as a controlled scaling ratio. These bursts do not establish
sustained capacity, a crash threshold, or a production percentile.

An isolated diagnostic of 16 fixed Arabic phrases, on the same CPU with Torch's
intra-op and inter-op thread counts both 4, measured embedding-only wall time
2.09/6.91/10.21 s for 1/4/16 concurrent encoding calls. Retrieval with precomputed
embeddings measured 2.86/1.57/1.01 s respectively. These were single sequential
diagnostic runs, not repeated balanced trials, but they isolate encoding contention
as an actionable suspect rather than implying that all search latency is ANN time.

The shared local CPU embedder now serializes `encode` calls with an instance lock.
An encoding call already uses Torch's threads; up to four question workers each
issuing four phrase searches previously ran up to 16 encodings on that model.
Only encoding holds the lock: BM25, ANN and answer-model verification remain parallel.
GPU encoding is unchanged, and batch sizes, normalization, source vectors and query
wording are unchanged. There is no additional answer-model call. Repeat the isolated
diagnostic and new-question latency/concurrency measurements after deployment before
claiming an end-to-end improvement. This lock is per model instance and per process;
it does not coordinate multiple server workers.

## Reducing model calls without weakening the source check

The understanding step previously ran twice in parallel for every ordinary question and used the second result only to add search phrases. It now runs once when the first result contains usable phrases. A failed call or a result without search phrases gets a second attempt; the original two-pass behaviour is available with `MUHAWIR_UNDERSTAND_SECOND_PASS=1`. The second reader still checks every claim separately, including its existing second reading of a non-severe rejection. This saves **one understanding request on the ordinary successful path**, not a fixed percentage of all model requests. The two old requests were parallel, so response latency is not guaranteed to fall by the duration of a full request.

`/api/health` now exposes numeric `usage.attempt_understand`, `attempt_support`, `attempt_draft`, `attempt_relevance`, `attempt_search_retry`, and `attempt_other` since process startup. They count attempted model calls across providers, including provider retries; `usage.model_calls` remains the older OpenAI-compatible provider's successful-response count. Compare the attempted calls per question and answer quality before and after deploying the change. Do not send real questions solely for a benchmark without a model-budget decision.
