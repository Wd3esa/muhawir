"""HTTP API and the web page.

The server does not log question text. Optional speech audio is cached on disk.
Run: uvicorn muhawir.server:app
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .corpus import load_corpus
from .generate import get_generator
from .generate import USAGE
from .pipeline import MAX_QUESTION_CHARS, STATS, AnswerStore, Muhawir
from .store import SqliteCorpus, SqliteRetriever
from . import tts

ROOT = Path(__file__).resolve().parent
PIPELINE_REVISION = hashlib.sha256((ROOT / "pipeline.py").read_bytes()).hexdigest()[:12]
DEFAULT_CORPUS = ROOT.parent / "data" / "synthetic_corpus.json"


DEFAULT_DB = ROOT.parent / "data" / "muhawir.db"


def answers_version(sources: Path, generator) -> str:
    """What a kept reply depends on: the code and prompts, the sources, and the models that wrote it."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(ROOT.glob("*.py")):  # what writes and checks a reply; not the page, the voice or the web server
        if f.name not in ("server.py", "tts.py", "build_data.py", "tashkeel.py", "tashkeel_worker.py"):
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
engine = build()


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(max_length=4000)


class Ask(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS * 2)
    style: Literal["kids", "youth", "extended", "newcomer"] = "youth"
    lang: Literal["ar", "en"] = "ar"
    history: list[Turn] = Field(default_factory=list, max_length=20)  # kept by the browser, not stored here


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


@app.post("/api/ask")
def ask(body: Ask, request: Request = None) -> dict:  # type: ignore[assignment]  # called directly in tests
    if not _allow_question(request.client.host if request is not None and request.client else "unknown"):
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
    try:
        return _run_ask(body)
    finally:
        _ask_slots.release()


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
                      [t.model_dump() for t in body.history]).to_dict()


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
    texts: list[str] = Field(min_length=1, max_length=30)
    passage_ids: list[str] = Field(default_factory=list, max_length=50)


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
    try:
        return tashkeel.diacritize(body.texts, tuple(sources))
    except tashkeel.Unavailable as exc:
        raise HTTPException(status_code=503, detail="Tashkeel unavailable; original text preserved") from exc


class Feedback(BaseModel):
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
    text: str = Field(min_length=1, max_length=3000)
    voice: Literal["male", "female"]
    lang: Literal["ar", "en"]


_tts_calls: dict[str, deque[float]] = defaultdict(deque)
_tts_lock = threading.Lock()


@app.post("/tts")
async def read_aloud(body: SpeechRequest, request: Request) -> Response:
    if not tts.configured():
        raise HTTPException(status_code=503, detail="Natural voice is unavailable")
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Answer text is empty")
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


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "pipeline_revision": PIPELINE_REVISION,
            "generator": engine.generator.name, "synthetic": engine.corpus.synthetic,
            "passages": len(engine.corpus.passages), "tts": tts.configured(), "credit_usd": credit_remaining(),
            "usage": {**{k: round(v, 1) for k, v in STATS.items()}, **USAGE,
                      "kept_replies": len(engine.store) if engine.store is not None else 0,
                      "rated_clear": FEEDBACK["clear"], "rated_unclear": FEEDBACK["unclear"]}}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")
