"""Optional display-only diacritics. Never alters answers, retrieval or stored citations."""
from __future__ import annotations
import difflib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from collections import deque
from functools import lru_cache

MARKS = re.compile(r'[\u064b-\u0652\u0670]')
WORDS = re.compile(r'[\u0621-\u063a\u0641-\u064a\u0671][\u0621-\u063a\u0641-\u064a\u0671\u064b-\u0652\u0670]*')
QUOTES = re.compile(r'«[^»]*(?:»|$)|﴿[^﴾]*(?:﴾|$)|“[^”]*(?:”|$)|"[^"\n]*(?:"|$)')
_SLOT = threading.Lock()
_CALLS = deque()

class Unavailable(RuntimeError):
    pass


def bare(text: str) -> str:
    # Do NOT normalize alef/hamza, ya, ta marbuta, spaces or punctuation.
    return MARKS.sub('', text)


def protect(text: str, sources: tuple[str, ...]) -> dict[int, str]:
    words = list(WORDS.finditer(text))
    protected = {i: m.group() for i, m in enumerate(words) if MARKS.search(m.group())}
    for quote in QUOTES.finditer(text):
        for i, word in enumerate(words):
            if word.start() >= quote.start() and word.end() <= quote.end():
                protected[i] = word.group()
    keys = [bare(w.group()) for w in words]
    for source in sources:
        source_words = list(WORDS.finditer(source))
        source_keys = [bare(w.group()) for w in source_words]
        for block in difflib.SequenceMatcher(None, keys, source_keys, autojunk=False).get_matching_blocks():
            if block.size < 3 and block.size != len(source_keys):
                continue
            for offset in range(block.size):
                i = block.a + offset
                original = words[i].group()
                cited = source_words[block.b + offset].group()
                # Existing marks win; otherwise take the exact source word's marks.
                protected[i] = original if MARKS.search(original) else cited
    return protected


def merge(text: str, prediction: str, sources: tuple[str, ...]) -> tuple[str, bool]:
    # CATT may omit a final full stop; restore it only on an exact match.
    if text.endswith(".") and bare(prediction) == bare(text[:-1]):
        prediction += "."
    if bare(text) != bare(prediction):
        return text, False
    originals = list(WORDS.finditer(text))
    predictions = list(WORDS.finditer(prediction))
    if len(originals) != len(predictions):
        return text, False
    protected = protect(text, sources)
    pieces, pos = [], 0
    for i, (original, predicted) in enumerate(zip(originals, predictions)):
        pieces.extend((text[pos:original.start()], protected.get(i, predicted.group())))
        pos = original.end()
    pieces.append(text[pos:])
    result = ''.join(pieces)
    return (result, True) if bare(result) == bare(text) else (text, False)


@lru_cache(maxsize=32)
def _predict(texts: tuple[str, ...]) -> tuple[str, ...]:
    python = os.environ.get('MUHAWIR_TASHKEEL_PYTHON', str(Path.home() / 'tashkeel-test/venv/bin/python'))
    if not Path(python).is_file():
        raise Unavailable('Tashkeel environment unavailable')
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        result = subprocess.run(
            [python, str(Path(__file__).with_name('tashkeel_worker.py'))],
            input=json.dumps(texts, ensure_ascii=False), text=True,
            capture_output=True, timeout=45, check=True, env=env)
        output = json.loads(result.stdout)
        if not isinstance(output, list) or len(output) != len(texts) or not all(isinstance(x, str) for x in output):
            raise ValueError('Invalid worker response')
        if sum(map(len, output)) > 24000:
            raise ValueError('Oversized worker response')
        return tuple(output)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise Unavailable('Tashkeel unavailable') from exc


def diacritize(texts: list[str], sources: tuple[str, ...] = ()) -> dict:
    if len(texts) > 30 or sum(map(len, texts)) > 6000:
        raise ValueError('Text too long')
    if not _SLOT.acquire(blocking=False):
        raise Unavailable('Tashkeel busy')
    try:
        now = time.monotonic()
        while _CALLS and _CALLS[0] <= now - 60:
            _CALLS.popleft()
        if len(_CALLS) >= 20:
            raise Unavailable('Tashkeel rate limit')
        _CALLS.append(now)
        outputs = _predict(tuple(texts))
        merged = [merge(text, output, sources) for text, output in zip(texts, outputs)]
        return {'texts': [x[0] for x in merged], 'rejected': sum(not x[1] for x in merged), 'automatic': True}
    finally:
        _SLOT.release()
