"""HTTP API and the web page.

The server does not log question text. Optional speech audio is cached on disk.
Run: uvicorn muhawir.server:app
"""
from __future__ import annotations

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
from .pipeline import MAX_QUESTION_CHARS, Muhawir
from .store import SqliteCorpus, SqliteRetriever
from . import tts

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT.parent / "data" / "synthetic_corpus.json"


DEFAULT_DB = ROOT.parent / "data" / "muhawir.db"


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

            retriever = HybridRetriever(retriever, load_vectors())
            threading.Thread(target=retriever.warm_up, daemon=True).start()
        engine = Muhawir(corpus, generator, retriever)
    else:
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
    style: str = "youth"
    lang: str = "ar"
    history: list[Turn] = Field(default_factory=list, max_length=20)  # kept by the browser, not stored here


@app.post("/api/ask")
def ask(body: Ask) -> dict:
    return engine.ask(body.question, body.style, body.lang,
                      [t.model_dump() for t in body.history]).to_dict()


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


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "generator": engine.generator.name, "synthetic": engine.corpus.synthetic,
            "passages": len(engine.corpus.passages), "tts": tts.configured()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")
