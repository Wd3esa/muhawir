"""Optional display-only diacritics. Never alters answers, retrieval or stored citations."""
from __future__ import annotations
import difflib
import json
import logging
import os
from pathlib import Path
import re
import select
import subprocess
import threading
import time
import types
from collections import Counter, OrderedDict, deque

MARKS = re.compile(r'[\u064b-\u0652\u0670]')
WORDS = re.compile(r'[\u0621-\u063a\u0641-\u064a\u0671][\u0621-\u063a\u0641-\u064a\u0671\u064b-\u0652\u0670]*')
QUOTES = re.compile(r'«[^»]*(?:»|$)|﴿[^﴾]*(?:﴾|$)|“[^”]*(?:”|$)|"[^"\n]*(?:"|$)')
log = logging.getLogger(__name__)
_SLOT = threading.Lock()
_CALLS = deque()

class Unavailable(RuntimeError):
    """Raised when vowelling cannot be done now: no worker, busy, too slow or over its rate."""
    pass


def bare(text: str) -> str:
    # Do NOT normalize alef/hamza, ya, ta marbuta, spaces or punctuation.
    """The text without its vowel marks; letters, hamza, spaces and punctuation are left as they are."""
    return MARKS.sub('', text)


def protect(text: str, sources: tuple[str, ...]) -> dict[int, str]:
    """The words whose vowels must not come from the tool: words already vowelled, quoted words, and words matching a vowelled source."""
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
            # A source written without vowels (most tafsir and fiqh books) has none to give: the tool vowels those
            # words. Only a source that is itself vowelled (hadith, verses) is copied.
            cited_words = source_words[block.b:block.b + block.size]
            if sum(1 for w in cited_words if MARKS.search(w.group())) < 0.8 * block.size:
                continue
            for offset in range(block.size):
                i = block.a + offset
                original = words[i].group()
                cited = source_words[block.b + offset].group()
                # Existing marks win; otherwise take the exact source word's marks.
                protected[i] = original if MARKS.search(original) else cited
    return protected


def _keys(text: str) -> list[str]:
    return [bare(m.group()) for m in WORDS.finditer(text)]


_ENDING = re.compile(r'^(.*?)([\u064b-\u064d\u064e-\u0650]*)$', re.S)
_TANWEEN_ALIF = re.compile(r'^(.*)(\u064b)([اى])$', re.S)


def _split_ending(word: str) -> tuple[str, str, str]:
    """A vowelled word as (head, ending, tail): the case ending is the vowel on the last letter, or the tanween on the
    letter before a final alif («غَوْرًا» = «غَوْر» + «ً» + «ا»); the word is head + ending + tail."""
    tanween = _TANWEEN_ALIF.match(word)
    if tanween:
        return tanween.group(1), tanween.group(2), tanween.group(3)
    head, ending = _ENDING.match(word).groups()
    return head, ending, ''


def mostly_vowelled(text: str, share: float = 0.8) -> bool:
    """True when at least `share` of the words of a passage carry a vowel mark: its vowels are then the book's own
    (verses, hadith, and the vowelled commentaries), not a few marks left in an unvowelled text."""
    words = WORDS.findall(text)
    return bool(words) and sum(1 for w in words if MARKS.search(w)) >= share * len(words)


def build_lexicon(texts, min_count: int = 3, dominance: float = 0.9) -> dict[str, tuple[str, str, str]]:
    """What the vowelled books agree on. `texts` are passages whose own vowels are the book's (verses, hadith, vowelled commentaries). For
    each word, written without vowels, that the books vowel one way in at least `dominance` of at least `min_count`
    places: (head, tail, the ending they most often give it). The ending is kept apart because it changes with the
    word's place in the sentence, while the rest of the word does not. No word is named here: the books decide."""
    seen: dict[str, Counter] = {}
    endings: dict[tuple[str, str, str], Counter] = {}
    for text in texts:
        for word in WORDS.findall(text):
            if not MARKS.search(word):
                continue
            head, ending, tail = _split_ending(word)
            key = bare(word)
            seen.setdefault(key, Counter())[(head, tail)] += 1
            endings.setdefault((key, head, tail), Counter())[ending] += 1
    lexicon = {}
    for key, counted in seen.items():
        total = sum(counted.values())
        (head, tail), top = counted.most_common(1)[0]
        if total >= min_count and top >= dominance * total:
            lexicon[key] = (head, tail, endings[(key, head, tail)].most_common(1)[0][0])
    return lexicon


def build_pairs(texts, min_count: int = 3, dominance: float = 0.8) -> dict[tuple[str, str], tuple[str, str, str]]:
    """What the vowelled books agree on for a word after a given word. Many words are read two or three ways on their
    own («يحل») and only the word before decides («أن يحل» / «لا يحل»); the books' own pairs of words settle those
    where the single word cannot. Same shape as build_lexicon, with (previous word, word) written without vowels as
    the key and a lower `dominance`, as a pair is rarer than a word."""
    seen: dict[tuple[str, str], Counter] = {}
    endings: dict[tuple[tuple[str, str], str, str], Counter] = {}
    for text in texts:
        previous = None
        for word in WORDS.findall(text):
            key = bare(word)
            if previous is not None and MARKS.search(word):
                head, ending, tail = _split_ending(word)
                seen.setdefault((previous, key), Counter())[(head, tail)] += 1
                endings.setdefault(((previous, key), head, tail), Counter())[ending] += 1
            previous = key
    pairs = {}
    for key, counted in seen.items():
        total = sum(counted.values())
        (head, tail), top = counted.most_common(1)[0]
        if total >= min_count and top >= dominance * total:
            pairs[key] = (head, tail, endings[(key, head, tail)].most_common(1)[0][0])
    return pairs


def _correct_stem(word: str, lexicon, previous: str | None = None, pairs=None) -> str:
    """The tool's word with the rest of the word (everything but its ending) put right where the vowelled books agree
    on it: first as the word follows `previous` (build_pairs), else as the word alone (build_lexicon). The ending stays
    the tool's own, as it depends on the sentence; the books' most common ending is used only when the tool gave none."""
    key = bare(word)
    entry = None
    if pairs and previous:
        entry = pairs(previous, key) if callable(pairs) else pairs.get((previous, key))
    if not entry and lexicon:
        entry = lexicon(key) if callable(lexicon) else lexicon.get(key)
    if not entry:
        return word
    head, tail, common = entry
    own_head, own_ending, own_tail = _split_ending(word)
    if (own_head, own_tail) == (head, tail):
        return word if (not own_ending or common) else head + tail
    # the books never mark the last letter of this word (a particle, a word ending in a long vowel, a sukun): none is added
    fixed = head + ((own_ending or common) if common else '') + tail
    return fixed if bare(fixed) == bare(word) else word


def merge(text: str, prediction: str, sources: tuple[str, ...], lexicon=None, pairs=None) -> tuple[str, bool]:
    """Put the predicted vowels onto the words of `text`. Only the vowels of Arabic words are taken from the
    prediction: the letters must be the same, and everything between the words (punctuation, digits, spaces,
    Latin) stays exactly as in `text`, whatever the tool did to it. With a `lexicon` (build_lexicon) the tool's word
    is first put right where the vowelled books agree (`pairs`: as the word follows the one before it)."""
    originals = list(WORDS.finditer(text))
    predictions = list(WORDS.finditer(prediction))
    if _keys(text) != _keys(prediction):
        return text, False
    protected = protect(text, sources)
    pieces, pos = [], 0
    for i, (original, predicted) in enumerate(zip(originals, predictions)):
        word = protected.get(i)
        if word is None:
            if lexicon or pairs:
                word = _correct_stem(predicted.group(), lexicon, bare(originals[i - 1].group()) if i else None, pairs)
            else:
                word = predicted.group()
        pieces.extend((text[pos:original.start()], word))
        pos = original.end()
    pieces.append(text[pos:])
    result = ''.join(pieces)
    return (result, True) if bare(result) == bare(text) else (text, False)


def _difference(text: str, prediction: str) -> str:
    """For the log: the first place where the tool's words differ from the text's, with the code points."""
    a, b = _keys(text), _keys(prediction)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f'word {i}: {x!r} {[hex(ord(ch)) for ch in x]} became {y!r} {[hex(ord(ch)) for ch in y]}'
    return f'{len(a)} words became {len(b)}: {text[:120]!r} -> {prediction[:120]!r}'


_SENTENCE = re.compile(r'[^.؟!؛:\n]+[.؟!؛:\n]*|[.؟!؛:\n]+')


def chunks(text: str) -> list[str]:
    """The text cut after each sentence mark (joined again they give the text): the tool reads one sentence at a
    time, so a long answer is not cut short or thrown out whole because of one sentence."""
    return _SENTENCE.findall(text)


_WAIT = 15        # seconds a request may queue behind another one
_QUEUE = 3        # requests allowed to wait; more are turned away at once
_START = 40       # seconds for the worker to load its model
_RUN = 30         # seconds for one request once the model is loaded
_IDLE = 600       # the worker is stopped after this long without a request (frees its memory)
_proc = None
_idle_timer = None
_waiting = 0
_waiting_lock = threading.Lock()


def _stop() -> None:
    """Kill the worker (after a failure, a timeout or a long idle); the next request starts a new one."""
    global _proc
    proc, _proc = _proc, None
    if proc is not None:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass


def _line(proc, timeout: float) -> bytes:
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise subprocess.TimeoutExpired('tashkeel worker', timeout)
    line = proc.stdout.readline()
    if not line:
        raise OSError('Tashkeel worker ended')
    return line


def _start():
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc
    _stop()
    python = os.environ.get('MUHAWIR_TASHKEEL_PYTHON', str(Path.home() / 'tashkeel-test/venv/bin/python'))
    if not Path(python).is_file():
        raise Unavailable('Tashkeel environment unavailable')
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    proc = subprocess.Popen([python, str(Path(__file__).with_name('tashkeel_worker.py'))],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    _proc = proc
    if json.loads(_line(proc, _START)) != {"ready": True}:
        raise ValueError('Tashkeel worker did not start')
    return proc


def _arm_idle() -> None:
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()

    def stop_if_quiet():
        """Stop the worker unless a request is using it."""
        if _SLOT.acquire(blocking=False):  # never kill a worker that is answering
            try:
                _stop()
            finally:
                _SLOT.release()
    _idle_timer = threading.Timer(_IDLE, stop_if_quiet)
    _idle_timer.daemon = True
    _idle_timer.start()


_BATCH = 4        # sentences sent to the tool together
_BUDGET = 40      # seconds one request may spend reading sentences; what is left unread is asked again by the page
_DONE: OrderedDict[str, str] = OrderedDict()  # sentence -> its reading, kept so a repeated request is instant
_state = threading.local()


def _predict(texts: tuple[str, ...]) -> tuple[str, ...]:
    """Called with _SLOT held. The sentences are read a few at a time, each sentence once ever (the readings are kept).
    When the time budget runs out, or the worker fails after some sentences were read, the rest come back unread and
    `_state.partial` is set: the page asks again and only the unread ones are done. Nothing read at all is a failure."""
    todo = [x for x in dict.fromkeys(texts) if x not in _DONE]
    started = time.monotonic()
    for at in range(0, len(todo), _BATCH):
        if at and time.monotonic() - started > _BUDGET:
            _state.partial = True
            break
        batch = tuple(todo[at:at + _BATCH])
        try:
            proc = _start()
            proc.stdin.write((json.dumps(batch, ensure_ascii=False) + "\n").encode('utf-8'))
            proc.stdin.flush()
            output = json.loads(_line(proc, _RUN))
            if not isinstance(output, list) or len(output) != len(batch) or not all(isinstance(x, str) for x in output):
                raise ValueError('Invalid worker response')
            if sum(map(len, output)) > 24000:
                raise ValueError('Oversized worker response')
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            _stop()
            if at == 0 and not any(x in _DONE for x in texts):
                raise Unavailable('Tashkeel unavailable') from exc
            _state.partial = True
            break
        for sentence, out in zip(batch, output):
            _DONE[sentence] = out
        while len(_DONE) > 2000:
            _DONE.popitem(last=False)
    return tuple(_DONE.get(x, x) for x in texts)


_predict.cache_clear = _DONE.clear  # type: ignore[attr-defined]
_predict.cache_info = lambda: types.SimpleNamespace(currsize=len(_DONE))  # type: ignore[attr-defined]


def _lone_word(text: str) -> str | None:
    """The word when the whole text is one Arabic word (a heading), else None."""
    words = WORDS.findall(text)
    return words[0] if len(words) == 1 and not MARKS.search(text) and text.strip() == words[0] else None


_ARTICLE = re.compile(r'^ا[\u064b-\u0652\u0670]*ل[\u064b-\u0652\u0670]*')


def _without_case_ending(word: str) -> str:
    """A vowelled word without the vowel that ends it (its case ending, or the tanween): a heading has none."""
    word = re.sub(r'[\u064b-\u064d\u064e-\u0650]+$', '', word)
    return re.sub(r'\u064b(?=[اى]$)', '', word)


def _heading_form(word: str, vowelled_texts: list[str]) -> str | None:
    """A lone word (a heading) vowelled the way the same word is vowelled in the answer's own sentences, where the
    tool saw it with a sentence around it: without the article and without its case ending, which a heading does
    not have. None when the word is not in any of them. Works for any word; no list of words is involved."""
    for text in vowelled_texts:
        for found in WORDS.findall(text):
            if not MARKS.search(found):
                continue
            stem = found
            if bare(found) != word and bare(found) == 'ال' + word:
                stem = _ARTICLE.sub('', found)
                stem = re.sub(r'^(.[\u064b-\u0650\u0652\u0670]*)\u0651([\u064b-\u0652\u0670]*)', r'\1\2', stem)  # a sun letter's doubling came from the article
            if bare(stem) != word:
                continue
            return _without_case_ending(stem)
    return None


_CONTEXT_WORDS = 12


def _left_context(before: list[str]) -> str:
    """The last words that came before a sentence, plain: the tool reads a sentence better with what leads into it
    («ومعنى قوله تعالى:» before its verse). Sentence marks are dropped, because the tool drops a colon and the space
    after it and would glue two words."""
    words = WORDS.findall(re.sub(r'[.؟!؛:\n]+', ' ', ' '.join(before)))[-_CONTEXT_WORDS:]
    return ' '.join(bare(w) for w in words)


def _read(queries: tuple[tuple[str, str], ...]) -> dict[tuple[str, str], str]:
    """Each (context, sentence) read by the tool, as the sentence's own words vowelled. A sentence whose words the
    tool changed once the context was added is read again alone."""
    sent = tuple(f'{prefix} {c}' if prefix else c for prefix, c in queries)
    raw = _predict(sent) if sent else ()
    read: dict[tuple[str, str], str] = {}
    alone: list[str] = []
    for (prefix, c), asked, out in zip(queries, sent, raw):
        if not prefix:
            read[(prefix, c)] = out
        elif _keys(out) == _keys(asked):
            read[(prefix, c)] = ' '.join(WORDS.findall(out)[len(WORDS.findall(prefix)):])
        else:
            alone.append(c)
    if alone:
        alone = tuple(dict.fromkeys(alone))
        read.update({('', c): out for c, out in zip(alone, _predict(alone))})
    return read


def diacritize(texts: list[str], sources: tuple[str, ...] = (), form_of=None, context: bool = False,
               lexicon=None, pairs=None) -> dict:
    """Vowel the texts sentence by sentence; a sentence the tool changed beyond its marks keeps its original. Returns
    the texts and how many parts were left unchanged.

    A text that is one word alone (a heading) has no sentence for the tool to read, and its guess is the worst there
    («مُلْتَحِدًا» for «مُلْتَحَدًا»). `form_of(word)` gives the word as the mushaf writes it, when the mushaf writes
    it one way only; that form is used. Any other lone word takes the vowels it has in the answer's own sentences,
    and has none when it is in none of them, rather than a wrong guess."""
    if len(texts) > 80 or sum(map(len, texts)) > 6000:
        raise ValueError('Text too long')
    global _waiting
    with _waiting_lock:
        if _waiting >= _QUEUE:
            raise Unavailable('Tashkeel busy')
        _waiting += 1
    try:
        got = _SLOT.acquire(timeout=_WAIT)
    finally:
        with _waiting_lock:
            _waiting -= 1
    if not got:
        raise Unavailable('Tashkeel busy')
    try:
        now = time.monotonic()
        while _CALLS and _CALLS[0] <= now - 60:
            _CALLS.popleft()
        if len(_CALLS) >= 20:
            raise Unavailable('Tashkeel rate limit')
        _CALLS.append(now)
        known: dict[int, str] = {}
        later: list[int] = []  # lone words with no mushaf form: vowelled from the answer's own sentences afterwards
        if form_of is not None:
            for index, text in enumerate(texts):
                word = _lone_word(text)
                if word is not None:
                    form = form_of(word)
                    known[index] = _without_case_ending(form) if form and bare(form) == word else text
                    if known[index] == text:
                        later.append(index)
        parts = [[] if i in known else chunks(text) for i, text in enumerate(texts)]
        asked = [[(_left_context(cs[:j]) if context else '', c) for j, c in enumerate(cs)] for cs in parts]
        need = tuple(dict.fromkeys(q for qs in asked for q in qs if WORDS.search(q[1])))
        _state.partial = False
        predicted = _read(need)
        complete = not getattr(_state, 'partial', False)
        _arm_idle()
        results, rejected = [], 0
        for index, (text, cs) in enumerate(zip(texts, parts)):
            if index in known:
                results.append(known[index])
                continue
            pieces = []
            for j, c in enumerate(cs):
                out = predicted.get(asked[index][j], predicted.get(('', c)))
                if out is not None and _keys(out) != _keys(c):
                    rejected += 1  # the tool changed a letter or a word: this sentence stays as it was
                    log.warning('tashkeel rejected a sentence: %s', _difference(c, out))
                    out = None
                pieces.append(c if out is None else out)
            # joined with a space: the tool may drop a sentence's mark and the space after it, which would glue two words
            merged_text, ok = merge(text, ' '.join(pieces), sources, lexicon, pairs)
            if not ok:
                rejected += 1
                log.warning('tashkeel rejected a text as a whole: %r', text[:200])
            results.append(merged_text)
        for index in later:
            word = _lone_word(texts[index])
            sentences = [r for i, r in enumerate(results) if i not in known]
            results[index] = _heading_form(word, sentences) or texts[index]
        return {'texts': results, 'rejected': rejected, 'automatic': True, 'complete': complete}
    finally:
        _SLOT.release()
