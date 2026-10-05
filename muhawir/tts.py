"""Azure neural read-aloud for the visible answer text only."""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import secrets
from pathlib import Path
from xml.sax.saxutils import escape

import httpx

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "tts-cache"
OUTPUT_FORMAT = "audio-24khz-48kbitrate-mono-mp3"
VOICES = {
    ("ar", "male"): ("ar-SA", "ar-SA-HamedNeural"),
    ("ar", "female"): ("ar-SA", "ar-SA-ZariyahNeural"),
    ("en", "male"): ("en-US", "en-US-AndrewNeural"),
    ("en", "female"): ("en-US", "en-US-AvaNeural"),
}
_cache_locks = [asyncio.Lock() for _ in range(32)]


class SpeechUnavailable(Exception):
    """Azure speech could not generate audio."""


def configured() -> bool:
    region = os.environ.get("AZURE_SPEECH_REGION", "")
    return bool(os.environ.get("AZURE_SPEECH_KEY")) and bool(re.fullmatch(r"[a-z0-9-]{2,32}", region))


def replace_verses(text: str, lang: str) -> str:
    """Do not synthesize text marked as a Quran verse, even if its end is missing: between ﴿﴾, or between {}
    (the form the writing step uses for a verse). The page also removes the quotations in a sentence that
    cites a verse, so this is the server's own second guard."""
    mark = "(آية)" if lang == "ar" else "(verse)"
    return re.sub(r"\{[^}]*(?:\}|$)", mark, re.sub(r"﴿[^﴾]*(?:﴾|$)", mark, text))


def ssml(text: str, voice: str, lang: str) -> str:
    locale, name = VOICES[(lang, voice)]
    return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="{locale}"><voice name="{name}">{escape(replace_verses(text, lang))}'
            '</voice></speak>')


async def _request_azure(text: str, voice: str, lang: str) -> bytes:
    region = os.environ["AZURE_SPEECH_REGION"]
    async with httpx.AsyncClient(timeout=40) as client:
        try:
            response = await client.post(
                f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
                headers={
                    "Ocp-Apim-Subscription-Key": os.environ["AZURE_SPEECH_KEY"],
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
                    "User-Agent": "Muhawir",
                },
                content=ssml(text, voice, lang).encode("utf-8"),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SpeechUnavailable("Azure speech request failed") from exc
        if not response.content or not response.headers.get("content-type", "").startswith("audio/"):
            raise SpeechUnavailable("Azure returned no audio")
        return response.content


async def synthesize(text: str, voice: str, lang: str) -> bytes:
    if not configured():
        raise SpeechUnavailable("Azure speech is not configured")
    if not text.strip() or len(text) > 3000 or (lang, voice) not in VOICES:
        raise ValueError("invalid speech request")
    digest = hashlib.sha256(f"{voice}\0{lang}\0{text}".encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{digest}.mp3"
    async with _cache_locks[int(digest[:2], 16) % len(_cache_locks)]:
        if cache_file.is_file():
            return cache_file.read_bytes()
        audio = await _request_azure(text, voice, lang)
        CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = CACHE_DIR / f".{digest}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as file:
                file.write(audio)
            os.replace(temporary, cache_file)
        finally:
            temporary.unlink(missing_ok=True)
        return audio
