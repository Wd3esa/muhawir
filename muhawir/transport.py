"""Process-owned HTTP connection pool; no model keys or prompts in shared configuration."""
from __future__ import annotations

import atexit
import threading

import httpx

_lock = threading.Lock()
_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    with _lock:
        if _client is None or _client.is_closed:
            _client = httpx.Client(limits=httpx.Limits(max_connections=64,
                                                     max_keepalive_connections=32,
                                                     keepalive_expiry=30))
        return _client


def post(url, **kwargs):
    # Pass credentials and timeouts per request, never mutate the shared client's defaults.
    return client().post(url, **kwargs)


def close():
    """Call after request workers drain; CLI processes also release the pool at exit."""
    global _client
    with _lock:
        previous, _client = _client, None
    if previous is not None:
        previous.close()


atexit.register(close)
