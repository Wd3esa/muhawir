"""HTTP API and the web page.

The server does not log question text. Optional speech audio is cached on disk.
Run: uvicorn muhawir.server:app
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .corpus import load_corpus
from .generate import USAGE, get_generator
from .pipeline import MAX_QUESTION_CHARS, STATS, AnswerStore, Muhawir, _progress
from .store import SqliteCorpus, SqliteRetriever
from . import extract, quote_check, tts, telemetry
from .normalize import normalize

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
PIPELINE_REVISION = hashlib.sha256((ROOT / "pipeline.py").read_bytes()).hexdigest()[:12]
DEFAULT_CORPUS = ROOT.parent / "data" / "synthetic_corpus.json"


DEFAULT_DB = ROOT.parent / "data" / "muhawir.db"


NOT_IN_ANSWERS = frozenset({"server.py", "tts.py", "build_data.py", "tashkeel.py", "tashkeel_worker.py", "extract.py",
                            "quote_check.py", "extra_sources.py", "telemetry.py"})


def answers_version(sources: Path, generator) -> str:
    """What a kept reply depends on: the code and prompts, the sources, and the models that wrote it."""
    h = hashlib.sha256()
    # what writes and checks a reply; not the page, the voice, the web server, the vowelling, the file reader, the quote
    # lookup or the source importers that run only when the database is built. A change to any other file starts the kept
    # replies afresh, and each question is then paid for again: deploy such changes together, not one by one
    for f in sorted(ROOT.glob("*.py")):
        if f.name not in NOT_IN_ANSWERS:
            h.update(f.read_bytes())
    if sources.exists():
        st = sources.stat()
        h.update(f"{sources.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    h.update(generator.name.encode())
    h.update(repr(sorted((k, v) for k, v in os.environ.items() if "MODEL" in k)).encode())
    return h.hexdigest()[:16]


def answer_store(sources: Path, generator) -> AnswerStore | None:
    """Replies kept on disk across restarts (data/answers.db); MUHAWIR_STORE=0 turns it off."""
    path = os.environ.get("MUHAWIR_STORE", str(ROOT.parent / "data" / "answers.db"))
    if path in ("", "0") or generator.name == "extractive":
        return None
    return AnswerStore(path, answers_version(sources, generator))


def build() -> Muhawir:
    """Use the SQLite database when it exists (built by muhawir.build_data), else a JSON corpus.

    Real religious sources are only served in model mode: keyword matching alone
    returns passages that share words with a question without answering it.
    """
    db = Path(os.environ.get("MUHAWIR_DB") or DEFAULT_DB)
    generator = get_generator()
    if db.exists() and not os.environ.get("MUHAWIR_CORPUS"):
        corpus = SqliteCorpus(db)
        retriever = SqliteRetriever(corpus)
        if os.environ.get("MUHAWIR_VECTORS") == "1":
            from .vectors import HybridRetriever, load_vectors

            # The wider pool remains opt-in: a 32-question source review found no
            # top-10 gold gain at the displayed depth and some direct answers fell.
            bm25_pool_factor = int(os.environ.get("MUHAWIR_BM25_POOL_FACTOR", "1"))
            retriever = HybridRetriever(retriever, load_vectors(), bm25_pool_factor=bm25_pool_factor)
            threading.Thread(target=retriever.warm_up, daemon=True).start()
        engine = Muhawir(corpus, generator, retriever, answer_store(db, generator))
    else:
        if generator.name != "extractive" and not os.environ.get("MUHAWIR_CORPUS") and os.environ.get("MUHAWIR_ALLOW_SYNTHETIC") != "1":
            raise RuntimeError("source database is missing; build it with python -m muhawir.build_data --download")
        corpus = load_corpus(os.environ.get("MUHAWIR_CORPUS") or DEFAULT_CORPUS)
        engine = Muhawir(corpus, generator)
    if not corpus.synthetic and generator.name == "extractive":
        raise RuntimeError("real sources need LLM_PROVIDER=model; extractive mode is for test data only")
    return engine


app = FastAPI(title="Muhawir", docs_url=None, redoc_url=None)
app.mount("/fonts", StaticFiles(directory=ROOT / "static" / "fonts"), name="fonts")


@app.get("/book-answer.css")
def book_answer_styles() -> FileResponse:
    """The stylesheet of the book-page view of an answer."""
    return FileResponse(ROOT / "static" / "book-answer.css", media_type="text/css")


engine = build()


class Turn(BaseModel):
    """One earlier message of the conversation, sent back by the browser."""
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(max_length=4000)


class Ask(BaseModel):
    """A question with the explanation style, the language and the recent conversation."""
    question: str = Field(max_length=MAX_QUESTION_CHARS * 2)
    style: Literal["kids", "youth", "extended", "newcomer"] = "youth"
    lang: Literal["ar", "en", "ur"] = "ar"
    history: list[Turn] = Field(default_factory=list, max_length=20)  # kept by the browser, not stored here
    as_question: bool = False  # «اسأل عنه» under a quotation found in the sources: answer it, do not look it up again


# Bound paid work across all visitors, including deployments behind a reverse proxy.
ASK_PER_MINUTE = max(1, int(os.environ.get("MUHAWIR_ASK_PER_MINUTE", "60")))
_ask_slots = threading.BoundedSemaphore(max(1, int(os.environ.get("MUHAWIR_ASK_CONCURRENCY", "4"))))
# Bound waiting requests separately. Raising generation concurrency on a small VM
# would multiply model calls and could exhaust the upstream provider's free quota.
ASK_QUEUE_SIZE = max(0, int(os.environ.get("MUHAWIR_ASK_QUEUE_SIZE", "12")))
ASK_QUEUE_TIMEOUT = max(0.0, float(os.environ.get("MUHAWIR_ASK_QUEUE_TIMEOUT", "10")))
_ask_waiters = threading.BoundedSemaphore(ASK_QUEUE_SIZE) if ASK_QUEUE_SIZE else None
_ask_calls: deque[float] = deque()
_ask_lock = threading.Lock()


# Questions per visitor, so one script cannot spend the prepaid model credit or keep the site busy for everyone.
# Generous for a person (a question takes 10 to 20 seconds); the address is the peer uvicorn reports, which behind
# the local Caddy proxy is the visitor's (uvicorn trusts X-Forwarded-For from 127.0.0.1 only).
ASK_LIMITS = ((60, 20), (3600, 200))  # (seconds, questions)
_visitor_calls: dict[str, deque[float]] = defaultdict(deque)
_visitor_lock = threading.Lock()


def _allow_question(address: str) -> bool:
    now = time.monotonic()
    longest = max(window for window, _ in ASK_LIMITS)
    with _visitor_lock:
        calls = _visitor_calls[address]
        while calls and calls[0] <= now - longest:
            calls.popleft()
        if any(sum(1 for t in calls if t > now - window) >= most for window, most in ASK_LIMITS):
            return False
        calls.append(now)
        if len(_visitor_calls) > 5000:  # forget visitors with nothing recent
            for key in [k for k, v in _visitor_calls.items() if not v]:
                del _visitor_calls[key]
        return True


WARN_AT = 0.8  # the visitor is told when four fifths of a limit are used


def usage_of(address: str) -> dict | None:
    """The limit this visitor is closest to, once four fifths of it are used (22 October 2026):
    {"used", "limit", "resets_in" (seconds until the oldest question in it leaves the window)}; else None."""
    now = time.monotonic()
    with _visitor_lock:
        calls = list(_visitor_calls.get(address, ()))
    worst = None
    for window, most in ASK_LIMITS:
        inside = [t for t in calls if t > now - window]
        if len(inside) >= WARN_AT * most and (worst is None or len(inside) / most > worst["used"] / worst["limit"]):
            worst = {"used": len(inside), "limit": most, "resets_in": int(inside[0] + window - now) + 1}
    return worst


def _admit(request) -> str:
    """Rate check and an answer slot, as for /api/ask; returns the visitor's address. The caller releases the slot."""
    address = request.client.host if request is not None and request.client else "unknown"
    if not _allow_question(address):
        raise HTTPException(status_code=429, detail="Too many questions; please wait a minute", headers={"Retry-After": "60"})
    # Never wait while holding _ask_lock: queued requests must not block rate checks.
    acquired = _ask_slots.acquire(blocking=False)
    if not acquired:
        if _ask_waiters is None or not _ask_waiters.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="All answer slots are busy; please try again shortly", headers={"Retry-After": "5"})
        try:
            acquired = _ask_slots.acquire(timeout=ASK_QUEUE_TIMEOUT)
        finally:
            _ask_waiters.release()
        if not acquired:
            raise HTTPException(status_code=429, detail="All answer slots are busy; please try again shortly", headers={"Retry-After": "5"})
    return address


@app.post("/api/ask")
def ask(body: Ask, request: Request = None) -> dict:  # type: ignore[assignment]  # called directly in tests
    """Answer a question: check the visitor's rate, take an answer slot (waiting briefly if all are busy), then run the pipeline."""
    trace = telemetry.start("ask")
    status, http_status = "error", 500
    with telemetry.bind(trace):
        try:
            with telemetry.span("admission"):
                address = _admit(request)
            try:
                reply = _run_ask(body)
                usage = usage_of(address)
                if usage:
                    reply["usage"] = usage
                status, http_status = reply.get("status", "unknown"), 200
                if trace:
                    reply["request_id"] = trace.id
                return reply
            finally:
                _ask_slots.release()
        except HTTPException as exc:
            status, http_status = "rejected", exc.status_code
            raise
        finally:
            if trace:
                trace.finish(status, http_status)



@app.post("/api/ask_stream")
def ask_stream(body: Ask, request: Request = None) -> StreamingResponse:  # type: ignore[assignment]
    """Server-ask stream: the same answer as /api/ask, sent as lines of JSON. {"stage": ...} lines tell which step the
    pipeline has reached (understanding, searching, writing, checking), so the page shows real progress at once;
    the last line is {"reply": ...}, the checked answer, or {"error": status} if it failed. No sentence is sent before
    every sentence has passed its check."""
    trace = telemetry.start("ask_stream")
    with telemetry.bind(trace):
        try:
            with telemetry.span("admission"):
                address = _admit(request)
        except BaseException as exc:
            if trace:
                trace.finish("rejected" if isinstance(exc, HTTPException) else "error",
                             exc.status_code if isinstance(exc, HTTPException) else 500)
            raise
    events: queue.Queue = queue.Queue()

    def work() -> None:
        with telemetry.bind(trace):
            token = _progress.set(lambda stage, data=None: events.put({"stage": stage, **({"data": data} if data else {})}))
            status, http_status = "error", 500
            try:
                events.put({"stage": "understanding"})
                reply = _run_ask(body)
                usage = usage_of(address)
                if usage:
                    reply["usage"] = usage
                status, http_status = reply.get("status", "unknown"), 200
                if trace:
                    reply["request_id"] = trace.id
                events.put({"reply": reply})
            except HTTPException as exc:
                status, http_status = "rejected", exc.status_code
                events.put({"error": exc.status_code})
            except Exception:
                log.exception("ask_stream failed")
                events.put({"error": 500})
            finally:
                _progress.reset(token)
                _ask_slots.release()
                if trace:
                    trace.finish(status, http_status)
                events.put(None)

    try:
        threading.Thread(target=work, daemon=True).start()
    except BaseException:
        _ask_slots.release()
        if trace:
            trace.finish("error", 500)
        raise

    def lines():
        while True:
            try:
                item = events.get(timeout=15)
            except queue.Empty:
                yield "\n"  # keeps the connection open through a proxy while a slow step runs
                continue
            if item is None:
                return
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


def _run_ask(body: Ask) -> dict:
    # Charge the global request budget only for a request admitted to a model slot.
    with _ask_lock:
        now = time.monotonic()
        while _ask_calls and _ask_calls[0] <= now - 60:
            _ask_calls.popleft()
        if len(_ask_calls) >= ASK_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Too many questions; please try again shortly", headers={"Retry-After": "60"})
        _ask_calls.append(now)
    return engine.ask(body.question, body.style, body.lang,
                      [t.model_dump() for t in body.history], answer_it=body.as_question).to_dict()


@app.post("/api/ask_again")
def ask_again(body: Ask, request: Request) -> dict:
    """For the nightly review on this server only (scripts/nightly.py): a question kept as «not found» is answered
    again from scratch, through every check, and the new reply replaces the kept one. Refused from any other address."""
    # a visit through the web proxy (Caddy) carries X-Forwarded-For: only a call made on the server itself is served
    proxied = any(h in request.headers for h in ("x-forwarded-for", "x-real-ip", "forwarded"))
    if proxied or request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=404)
    if not _ask_slots.acquire(timeout=120):
        raise HTTPException(status_code=429, detail="busy", headers={"Retry-After": "60"})
    try:
        before, now = engine.ask_again(body.question, body.style, body.lang)
        return {"before": before.status if before else "", "now": now.to_dict()}
    finally:
        _ask_slots.release()


class TashkeelRequest(BaseModel):
    """Texts to vowel and the ids of the passages whose own vowels may be copied."""
    texts: list[str] = Field(min_length=1, max_length=80)
    passage_ids: list[str] = Field(default_factory=list, max_length=50)


_vowelled: "OrderedDict[str, dict]" = OrderedDict()  # Tashkeel-cache: a text vowelled once is given again at once
_vowelled_lock = threading.Lock()


@app.post("/api/tashkeel")
def tashkeel_answer(body: TashkeelRequest) -> dict:
    """Display-only transformation; no generation, answer-cache write or corpus mutation."""
    from . import tashkeel
    if sum(map(len, body.texts)) > 6000 or any(len(p) > 200 for p in body.passage_ids):
        raise HTTPException(status_code=422, detail="Text too long")
    sources = []
    for passage_id in dict.fromkeys(body.passage_ids):
        passage = engine.corpus.passage(passage_id)
        if passage:
            sources.append(passage.text)
    if sum(map(len, sources)) > 100000:
        raise HTTPException(status_code=422, detail="Sources too long")
    flags = "".join(str(os.environ.get(k, "") == "1") for k in (
        "MUHAWIR_TASHKEEL_CONTEXT", "MUHAWIR_TASHKEEL_LEXICON", "MUHAWIR_TASHKEEL_PAIRS")) \
        + os.environ.get("MUHAWIR_TASHKEEL_MODEL_DIR", "")
    key = hashlib.sha256(json.dumps([body.texts, sorted(set(body.passage_ids)), flags], ensure_ascii=False).encode()).hexdigest()
    with _vowelled_lock:
        if key in _vowelled:
            _vowelled.move_to_end(key)
            return _vowelled[key]
    try:
        # the two ways of improving the tool's reading (a longer lead-in to each sentence; the vowelled books' agreement
        # on the body of a word) are switched on by the environment until they are measured: tashkeel_compare.py
        result = tashkeel.diacritize(
            body.texts, tuple(sources), getattr(engine, "vowelled_form", None),
            context=os.environ.get("MUHAWIR_TASHKEEL_CONTEXT") == "1",
            lexicon=getattr(engine, "vowel_stem", None) if os.environ.get("MUHAWIR_TASHKEEL_LEXICON") == "1" else None,
            pairs=getattr(engine, "vowel_pair", None) if os.environ.get("MUHAWIR_TASHKEEL_PAIRS") == "1" else None)
    except tashkeel.Unavailable as exc:
        # the reason is a short fixed phrase from tashkeel.py (busy, rate limit, no worker): the page tells it apart
        raise HTTPException(status_code=503, detail=f"Tashkeel unavailable: {exc}; original text preserved",
                            headers={"Retry-After": "5"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Text too long") from exc
    if not result.get("complete", True):
        return result  # some sentences are still unread: the page asks again, and the finished text is what is kept
    with _vowelled_lock:
        _vowelled[key] = result
        while len(_vowelled) > 300:
            _vowelled.popitem(last=False)
    return result


class Feedback(BaseModel):
    """A visitor's rating of an answer."""
    verdict: Literal["clear", "unclear"]


@app.post("/api/feedback")
def feedback(body: Feedback) -> dict:
    """«واضحة / تحتاج توضيحًا» under an answer: counted only, with no question, answer or visitor kept."""
    with _feedback_lock:
        FEEDBACK[body.verdict] += 1
    return {"ok": True}


FEEDBACK = {"clear": 0, "unclear": 0}
_feedback_lock = threading.Lock()


class SpeechRequest(BaseModel):
    """Text to read aloud with a chosen voice."""
    text: str = Field(min_length=1, max_length=3000)
    voice: str = Field(pattern="^[a-z]{2,20}$")  # a voice id from /api/voices, or «male» / «female»
    lang: Literal["ar", "en", "ur"]


_tts_calls: dict[str, deque[float]] = defaultdict(deque)
_tts_lock = threading.Lock()


@app.post("/tts")
async def read_aloud(body: SpeechRequest, request: Request) -> Response:
    if not tts.configured():
        raise HTTPException(status_code=503, detail="Natural voice is unavailable")
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Answer text is empty")
    if (body.lang, body.voice) not in tts.VOICES:
        raise HTTPException(status_code=422, detail="Unknown voice")
    # Use the server-observed peer, never an untrusted X-Forwarded-For header.
    address = request.client.host if request.client else "unknown"
    with _tts_lock:
        now = time.monotonic()
        calls = _tts_calls[address]
        while calls and calls[0] <= now - 60:
            calls.popleft()
        if len(calls) >= 20:
            raise HTTPException(status_code=429, detail="Too many speech requests")
        calls.append(now)
        if len(_tts_calls) > 2000:
            for key in list(_tts_calls):
                if not _tts_calls[key] or _tts_calls[key][-1] <= now - 60:
                    del _tts_calls[key]
    try:
        audio = await tts.synthesize(body.text, body.voice, body.lang)
    except tts.SpeechUnavailable as exc:
        raise HTTPException(status_code=502, detail="Natural voice is unavailable") from exc
    return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


_credit: dict = {"at": 0.0, "value": None}


def credit_remaining() -> float | None:
    """Credit left on the main model's OpenRouter key, in US dollars (None when unknown or not OpenRouter).
    Read from OpenRouter's key endpoint, at most every five minutes; it costs no model call. The credit ran out
    during testing on 6 October 2026 and the site stopped answering: this shows it coming."""
    base, key = os.environ.get("OPENAI_COMPAT_BASE_URL", ""), os.environ.get("OPENAI_COMPAT_API_KEY", "")
    if "openrouter.ai" not in base or not key:
        return None
    if time.time() - _credit["at"] < 300:
        return _credit["value"]
    value = None
    try:
        import httpx
        r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=5)
        left = r.json().get("data", {}).get("limit_remaining") if r.status_code == 200 else None
        value = round(float(left), 2) if left is not None else None
    except Exception:  # noqa: BLE001 - the health reply never fails because of this
        value = None
    _credit.update(at=time.time(), value=value)
    return value


@app.get("/api/voices")
def voices() -> dict:
    """The natural voices the page can offer for each language (empty when the speech service is not configured)."""
    return tts.public_list() if tts.configured() else {}


@app.get("/api/health")
def health() -> dict:
    """Service status: revision, generator, corpus size, usage counts and the credit left."""
    return {"ok": True, "pipeline_revision": PIPELINE_REVISION,
            "generator": engine.generator.name, "synthetic": engine.corpus.synthetic,
            "passages": len(engine.corpus.passages), "tts": tts.configured(), "credit_usd": credit_remaining(),
            "usage": {**{k: round(v, 1) for k, v in STATS.items()}, **USAGE,
                      "kept_replies": len(engine.store) if engine.store is not None else 0,
                      "rated_clear": FEEDBACK["clear"], "rated_unclear": FEEDBACK["unclear"]}}


@app.get("/api/ready")
def ready() -> Response:
    """Readiness for a monitor: 200 when every part works, 503 with the failing parts named. Reads no model, costs
    nothing. Parts: the corpus, the search vectors, each model's breaker, and free disk space."""
    import shutil
    from .generate import BREAKERS
    now = time.monotonic()
    parts = {"corpus": len(engine.corpus.passages) > 0}
    retriever = engine.retriever
    parts["vectors"] = not getattr(retriever, "_disabled", False)
    paused = sorted(name for name, state in BREAKERS.items() if state["open_until"] > now)
    parts["models"] = not paused
    try:
        parts["disk"] = shutil.disk_usage(ROOT).free > 200 * 1024 * 1024
    except OSError:
        parts["disk"] = False
    body = {"ok": all(parts.values()), "parts": parts, "paused_models": paused}
    return Response(json.dumps(body), media_type="application/json", status_code=200 if body["ok"] else 503)


# Reading a file or photo is work for the processor, so only a few run at once; the rest are told to try again shortly.
OCR_SLOTS = threading.BoundedSemaphore(int(os.environ.get("MUHAWIR_OCR_SLOTS", "2")))
_EXTRACT_ERRORS = {"too_large": 413, "unsupported": 415, "unreadable": 422, "no_text": 422, "encrypted": 422,
                   "ocr_unavailable": 501, "too_slow": 504}


@app.post("/api/extract")
async def extract_file(request: Request) -> dict:
    """The text of an uploaded file or photo (the bytes are the request body): nothing is stored and no model is called.
    The page puts the text in the «تحقق من نص» box or the question box. Errors carry a short code the page translates."""
    address = request.client.host if request.client else "unknown"
    if not _allow_question(address):
        raise HTTPException(status_code=429, detail="busy", headers={"Retry-After": "60"})
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > extract.MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="too_large")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > extract.MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="too_large")
        chunks.append(chunk)
    if not OCR_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="busy", headers={"Retry-After": "5"})
    try:
        result = await run_in_threadpool(extract.extract, b"".join(chunks))
    except extract.ExtractError as exc:
        raise HTTPException(status_code=_EXTRACT_ERRORS.get(exc.code, 422), detail=exc.code) from None
    finally:
        OCR_SLOTS.release()
    result["pieces"] = extract.pieces(result["text"])
    return result


class QuoteCheck(BaseModel):
    """A quotation pasted by the visitor, to be looked up in the sources."""
    text: str = Field(max_length=quote_check.MAX_QUOTE_CHARS)


@app.post("/api/verify_text")
def verify_text(body: QuoteCheck, request: Request = None) -> dict:  # type: ignore[assignment]
    """Quote-check: where does this quotation appear in the sources? No model is called, so no cost and no answer slot;
    only the visitor's rate limit applies. «exact» means every word of it, in order, inside one stored passage."""
    address = request.client.host if request is not None and request.client else "unknown"
    if not _allow_question(address):
        raise HTTPException(status_code=429, detail="Too many questions; please wait a minute", headers={"Retry-After": "60"})
    text = body.text.strip()
    if len(normalize(text).split()) < quote_check.MIN_QUOTE_WORDS:
        raise HTTPException(status_code=422, detail="Quotation too short to look up")
    return quote_check.lookup(engine.retriever, engine.corpus, text)


@app.get("/api/context/{passage_id}")
def context(passage_id: str) -> dict:
    """«اقرأ السياق» on a source card: the passage before and after the one cited, from the same book, as they are.
    Read from the database only: no model, no cost (22 October 2026)."""
    if len(passage_id) > 40 or engine.corpus.passage(passage_id) is None:
        raise HTTPException(status_code=404)
    before, after = engine.corpus.neighbours(passage_id)
    side = lambda p: {"passage_id": p.id, "location": p.location, "text": p.text} if p else None  # noqa: E731
    return {"before": side(before), "after": side(after)}


@app.get("/")
def index() -> FileResponse:
    """The page."""
    # no-cache: the browser asks again each time and gets the new page at once after an update (a stale copy kept
    # the old menus on a phone), yet still uses its copy when nothing changed
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})
