"""Muhawir over MCP (November 2026): other AI assistants and models (Claude, ChatGPT, a local model…) can use
Muhawir's indexed sources through the Model Context Protocol, with their own model writing the answer.

On unless MUHAWIR_MCP=0. Served at /mcp (streamable HTTP, stateless). Four tools:

- search_sources   find passages for a question or phrase (no model call, no cost)
- read_passage     one passage in full, with the passages before and after it in its book (no model call)
- find_quote       where a quotation is in the sources, word for word or close (no model call)
- ask_muhawir      Muhawir's own checked answer (uses Muhawir's model: an answer slot and the question budget)

The three free tools are open to anyone with the address (the page's «ربط مع مساعد آخر» shows it). ask_muhawir spends
the model's credit, so it is offered only when MUHAWIR_MCP_TOKEN is set, and then answers only a request that carries
«Authorization: Bearer <token>» (or to anyone, with MUHAWIR_MCP_ASK=1, not advised on a public address).
MUHAWIR_MCP_PER_MINUTE limits all tool calls together (default 120).
"""
from __future__ import annotations

import collections
import contextlib
import contextvars
import hmac
import logging
import os
import threading
import time
from typing import Callable

import anyio

from . import bayyinat, quote_check
from .normalize import normalize

log = logging.getLogger(__name__)
ENABLED = os.environ.get("MUHAWIR_MCP", "1") != "0"
TOKEN = os.environ.get("MUHAWIR_MCP_TOKEN", "")
ASK_OPEN = os.environ.get("MUHAWIR_MCP_ASK", "") == "1"  # ask_muhawir without a token: not advised on a public address
ASK_ALLOWED = bool(TOKEN) or ASK_OPEN
_authorized: contextvars.ContextVar[bool] = contextvars.ContextVar("muhawir_mcp_authorized", default=False)
PER_MINUTE = max(1, int(os.environ.get("MUHAWIR_MCP_PER_MINUTE", "120")))
MAX_RESULTS = 12
MAX_TEXT = 1500  # characters of a passage in a search result; read_passage gives the whole passage
STYLES = ("kids", "youth", "extended", "newcomer")
LANGS = ("ar", "en", "ur")

INSTRUCTIONS = """Muhawir (مُحاور) gives access to a fixed library of Islamic sources: the Quran, hadith collections with
their stated gradings, tafsir, fiqh books and published fatwas. It does not answer from memory.

How to use it well:
1. Call search_sources with the question in Arabic (also try the sources' own terms, e.g. «القراض» for «المضاربة»).
2. Answer only from the passages returned. Quote verses and hadith word for word, inside «» or ﴿﴾, and name the
   source and place (the `source` and `location` fields) for every statement. Give a hadith's grading as stated.
3. If the passages do not answer the question, say that the sources available do not answer it. Do not fill the gap
   from memory.
4. Do not give a ruling on a person's own case (divorce, inheritance shares, disputes, oaths): present what the
   sources say in general and refer the person to a qualified scholar or official fatwa body.
5. read_passage gives the full passage and its neighbours; find_quote checks where a quotation appears.
6. ask_muhawir returns Muhawir's own answer, already checked sentence by sentence against its sources; it is slower
   and limited, so use it when a ready, verified answer is wanted.

يقدّم مُحاور مكتبة ثابتة من المصادر الإسلامية. أجب من المقاطع المعادة وحدها، وانقل الآيات والأحاديث بنصها، واذكر مصدر كل
جملة وموضعها، وقل «لم أجد في المصادر المتاحة ما يجيب عن هذا» إن لم تجب المقاطع، ولا تُفتِ في حالة شخصية."""


class _Budget:
    """All MCP tool calls together, per minute."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute, self.calls, self.lock = per_minute, collections.deque(), threading.Lock()

    def take(self) -> bool:
        with self.lock:
            now = time.monotonic()
            while self.calls and self.calls[0] <= now - 60:
                self.calls.popleft()
            if len(self.calls) >= self.per_minute:
                return False
            self.calls.append(now)
            return True


def _card(corpus, passage, text_limit: int | None = MAX_TEXT) -> dict:
    source = corpus.source_of(passage)
    text = passage.text if text_limit is None or len(passage.text) <= text_limit else passage.text[:text_limit] + "…"
    card = {"passage_id": passage.id, "source": source.name if source else "", "location": passage.location,
            "kind": passage.kind, "text": text}
    if passage.grade:
        card["grade"] = passage.grade
    url = getattr(source, "url", "") if source else ""
    if url:
        card["source_url"] = url
    return card


def build(engine, run_ask: Callable[[str, str, str], dict]):
    """The MCP server for this engine, or None when MCP is off. `run_ask(question, style, lang)` answers through the
    service's own answer slots and question budget, exactly as a visitor's question."""
    if not ENABLED:
        return None
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:  # MUHAWIR_MCP=1 without the package: the site still runs, without /mcp
        log.error("MUHAWIR_MCP=1 but the mcp package is not installed (pip install -r requirements.txt)")
        return None

    budget = _Budget(PER_MINUTE)
    server = MCPServer(name="muhawir", title="Muhawir — Islamic sources", instructions=INSTRUCTIONS,
                       website_url="https://muhawir.duckdns.org")

    def spend() -> None:
        if not budget.take():
            raise ValueError("Too many requests to Muhawir; wait a minute and try again.")

    @server.tool(description="Find passages in Muhawir's Islamic sources for a question or phrase (Arabic works best). "
                             "Returns up to `limit` passages with their source, location, kind and hadith grading.")
    async def search_sources(query: str, limit: int = 8) -> list[dict]:
        spend()
        query = query.strip()[:300]
        if not query:
            raise ValueError("Empty query.")
        limit = max(1, min(MAX_RESULTS, int(limit)))

        def run() -> list[dict]:
            hits = engine.retriever.search(query, k=limit * 2)
            out = [_card(engine.corpus, h.passage) for h in hits if bayyinat.is_answer_part(h.passage)]
            return out[:limit]
        return await anyio.to_thread.run_sync(run)

    @server.tool(description="Read one passage in full by its passage_id, with the passages just before and after it "
                             "in the same book (set around=false to skip them).")
    async def read_passage(passage_id: str, around: bool = True) -> dict:
        spend()
        passage = engine.corpus.passage(passage_id.strip()[:40])
        if passage is None:
            raise ValueError("No passage with this id.")
        out = _card(engine.corpus, passage, text_limit=None)
        if around:
            before, after = engine.corpus.neighbours(passage.id)
            out["before"] = _card(engine.corpus, before) if before else None
            out["after"] = _card(engine.corpus, after) if after else None
        return out

    @server.tool(description="Check where a quotation (a verse, a hadith, a scholar's words) appears in Muhawir's "
                             "sources: «exact» means every word, in order, inside one stored passage.")
    async def find_quote(text: str) -> dict:
        spend()
        text = text.strip()[:quote_check.MAX_QUOTE_CHARS]
        if len(normalize(text).split()) < quote_check.MIN_QUOTE_WORDS:
            raise ValueError("Quotation too short to look up.")
        return await anyio.to_thread.run_sync(lambda: quote_check.lookup(engine.retriever, engine.corpus, text))

    if ASK_ALLOWED:
        @server.tool(description="Muhawir's own answer, written by its model and checked sentence by sentence against "
                                 "its sources. Slower (15–40 s) and limited. style: kids, youth, extended or newcomer; "
                                 "lang: ar, en or ur.")
        async def ask_muhawir(question: str, style: str = "youth", lang: str = "ar") -> dict:
            if not (ASK_OPEN or _authorized.get()):
                raise ValueError("ask_muhawir needs the access token from the site's owner "
                                 "(Authorization: Bearer …). The other tools are open.")
            spend()
            question = question.strip()[:500]
            if not question:
                raise ValueError("Empty question.")
            style = style if style in STYLES else "youth"
            lang = lang if lang in LANGS else "ar"
            reply = await anyio.to_thread.run_sync(lambda: run_ask(question, style, lang))
            keep = ("status", "message", "claims", "sources", "note", "views", "follow_up", "referral")
            return {k: reply[k] for k in keep if k in reply}

    return server


def mount(app, server) -> None:
    """Serve `server` at /mcp on the FastAPI `app`, behind the optional bearer token, and run its session manager
    with the app (stateless: no session is kept between requests)."""
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = [h.strip() for h in os.environ.get("MUHAWIR_MCP_HOSTS", "").split(",") if h.strip()]
    security = TransportSecuritySettings(enable_dns_rebinding_protection=bool(hosts), allowed_hosts=hosts,
                                         allowed_origins=[f"https://{h}" for h in hosts])
    sub = server.streamable_http_app(streamable_http_path="/mcp", stateless_http=True, json_response=True,
                                     transport_security=security)
    for route in sub.routes:
        if getattr(route, "path", "") == "/mcp":
            route.app = _marked(route.app)
        app.router.routes.append(route)
    previous = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(asgi_app):
        async with server.session_manager.run():
            async with previous(asgi_app) as state:
                yield state
    app.router.lifespan_context = lifespan


def _marked(inner):
    """Note whether the request carries «Authorization: Bearer MUHAWIR_MCP_TOKEN»; only ask_muhawir needs it."""
    async def marked(scope, receive, send):
        given = dict(scope.get("headers") or []).get(b"authorization", b"") if scope["type"] == "http" else b""
        token = _authorized.set(bool(TOKEN) and hmac.compare_digest(given, f"Bearer {TOKEN}".encode()))
        try:
            await inner(scope, receive, send)
        finally:
            _authorized.reset(token)
    return marked
