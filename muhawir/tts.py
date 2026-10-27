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
# 48 kHz at 96 kbit/s (22 October 2026; was 24 kHz at 48): the voice no longer sounds muffled on phone speakers
OUTPUT_FORMAT = "audio-48khz-96kbitrate-mono-mp3"
RATE = "-6%"  # a little slower than the voices' own pace: religious explanation is listened to, not skimmed
# The voices a listener can choose, per language, women first: (id, Azure locale, Azure voice, gender, names).
# All are Azure's neural voices (recorded from real speakers); the first of each gender is the default for that gender.
VOICE_LIST = {
    "ar": [
        ("zariyah", "ar-SA", "ar-SA-ZariyahNeural", "female", {"ar": "زارية (السعودية)", "en": "Zariyah (Saudi)", "ur": "زاریہ (سعودی)"}),
        ("salma", "ar-EG", "ar-EG-SalmaNeural", "female", {"ar": "سلمى (مصر)", "en": "Salma (Egypt)", "ur": "سلمیٰ (مصر)"}),
        ("fatima", "ar-AE", "ar-AE-FatimaNeural", "female", {"ar": "فاطمة (الإمارات)", "en": "Fatima (UAE)", "ur": "فاطمہ (امارات)"}),
        ("hamed", "ar-SA", "ar-SA-HamedNeural", "male", {"ar": "حامد (السعودية)", "en": "Hamed (Saudi)", "ur": "حامد (سعودی)"}),
        ("shakir", "ar-EG", "ar-EG-ShakirNeural", "male", {"ar": "شاكر (مصر)", "en": "Shakir (Egypt)", "ur": "شاکر (مصر)"}),
        ("hamdan", "ar-AE", "ar-AE-HamdanNeural", "male", {"ar": "حمدان (الإمارات)", "en": "Hamdan (UAE)", "ur": "حمدان (امارات)"}),
    ],
    "en": [
        ("ava", "en-US", "en-US-AvaNeural", "female", {"ar": "آفا", "en": "Ava", "ur": "ایوا"}),
        ("emma", "en-US", "en-US-EmmaNeural", "female", {"ar": "إيما", "en": "Emma", "ur": "ایما"}),
        ("andrew", "en-US", "en-US-AndrewNeural", "male", {"ar": "أندرو", "en": "Andrew", "ur": "اینڈریو"}),
        ("brian", "en-US", "en-US-BrianNeural", "male", {"ar": "براين", "en": "Brian", "ur": "برائن"}),
    ],
    "ur": [
        ("uzma", "ur-PK", "ur-PK-UzmaNeural", "female", {"ar": "عظمى (باكستان)", "en": "Uzma (Pakistan)", "ur": "عظمیٰ (پاکستان)"}),
        ("asad", "ur-PK", "ur-PK-AsadNeural", "male", {"ar": "أسد (باكستان)", "en": "Asad (Pakistan)", "ur": "اسد (پاکستان)"}),
    ],
}
# (language, voice id) → (locale, Azure voice); «male» and «female» still name the first voice of each
VOICES = {(lang, vid): (locale, name) for lang, voices in VOICE_LIST.items() for vid, locale, name, _, _ in voices}
for _lang, _voices in VOICE_LIST.items():
    for _gender in ("female", "male"):
        _first = next(v for v in _voices if v[3] == _gender)
        VOICES[(_lang, _gender)] = (_first[1], _first[2])


def public_list() -> dict:
    """The voices the page offers: {"ar": [{"id", "gender", "names"}...], ...}."""
    return {lang: [{"id": vid, "gender": g, "names": names} for vid, _, _, g, names in voices]
            for lang, voices in VOICE_LIST.items()}
_cache_locks = [asyncio.Lock() for _ in range(32)]


class SpeechUnavailable(Exception):
    """Azure speech could not generate audio."""


def configured() -> bool:
    """True when Azure speech has a key and a valid region."""
    region = os.environ.get("AZURE_SPEECH_REGION", "")
    return bool(os.environ.get("AZURE_SPEECH_KEY")) and bool(re.fullmatch(r"[a-z0-9-]{2,32}", region))


def replace_verses(text: str, lang: str) -> str:
    """Do not synthesize text marked as a Quran verse, even if its end is missing: between ﴿﴾, or between {}
    (the form the writing step uses for a verse). The page also removes the quotations in a sentence that
    cites a verse, so this is the server's own second guard."""
    mark = {"ar": "(آية)", "ur": "(آیت)"}.get(lang, "(verse)")
    return re.sub(r"\{[^}]*(?:\}|$)", mark, re.sub(r"﴿[^﴾]*(?:﴾|$)", mark, text))


def ssml(text: str, voice: str, lang: str) -> str:
    """The SSML for this text with the voice of the language; quoted verses are dealt with by replace_verses."""
    locale, name = VOICES[(lang, voice)]
    return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="{locale}"><voice name="{name}"><prosody rate="{RATE}">{escape(replace_verses(text, lang))}'
            '</prosody></voice></speak>')


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
    locale, name = VOICES[(lang, voice)]  # the voice, format and pace are in the key: a better setting is not served old audio
    digest = hashlib.sha256(f"{name}\0{OUTPUT_FORMAT}\0{RATE}\0{lang}\0{text}".encode("utf-8")).hexdigest()
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
