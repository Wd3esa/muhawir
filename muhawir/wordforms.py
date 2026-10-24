"""Is a word in a sentence the same word as one in a source, in another form or another word for the same thing?

Used by the check on definitions (verify.invented_definition): a definition written in other words than its source
must not be dropped only because a word has another ending, another derived form, its verbal noun, or a synonym
(«الأساسات» and «الأركان», «يُعطى» and «إعطاء», «المحتاجين» and «حاجة»). Three ways, none of them a model call:

  1. the same stem once joined letters and common endings are removed («للفقراء» = «الفقراء»);
  2. the same root by the ISRI Arabic stemmer (NLTK, Apache-2.0), which gives the root of derived forms
     («مساعدة», «تساعد» → سعد);
  3. a synonym in the Arabic WordNet (AWN v2, Open Multilingual Wordnet, CC BY-SA 3.0), read from
     data/wn-data-arb.tab when that file is there («أساس» ~ «قاعدة»، «دعامة»، «ركيزة»).

None of these knows every synonym («ركن» and «أساس» are not linked in the WordNet), so a word still unmatched is not
judged here: the pipeline asks the model about that word only (Muhawir.judge_definition).
"""
from __future__ import annotations

import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from .normalize import normalize

SYNONYMS_FILE = Path(os.environ.get("MUHAWIR_SYNONYMS") or Path(__file__).resolve().parent.parent / "data" / "wn-data-arb.tab")
# Open Multilingual Wordnet data, pinned to one commit so every build reads the same synonyms
AWN_COMMIT = "406bf83b3c507a3d1f26e88252d5d66893fd36bf"
AWN_DOWNLOAD = (f"https://raw.githubusercontent.com/omwn/omw-data/{AWN_COMMIT}/wns/arb/wn-data-arb.tab",
                f"https://cdn.jsdelivr.net/gh/omwn/omw-data@{AWN_COMMIT}/wns/arb/wn-data-arb.tab")
AWN_FILE = "wn-data-arb.tab"

_ENDINGS = ("ات", "ون", "ين", "ان", "ها", "هم", "كم", "نا", "ه", "ة", "ي", "ك")


def stem(token: str) -> str:
    """A rough stem: a joined letter, the article and one common ending removed."""
    if token[:1] in "وفبلك" and len(token) > 4:
        token = token[1:]
    token = token[2:] if token.startswith("ال") and len(token) > 4 else token
    for end in _ENDINGS:
        if token.endswith(end) and len(token) - len(end) >= 3:
            return token[:-len(end)]
    return token


@lru_cache(maxsize=1)
def _isri():
    try:
        from nltk.stem.isri import ISRIStemmer
    except ImportError:  # the check still runs on stems and synonyms; the model judges the rest
        return None
    return ISRIStemmer()


_WEAK = str.maketrans("", "", "اويىءأإآئؤ")
_DERIVED = ("است", "مست", "ا", "م", "ت", "ي", "ن")


def skeleton(root: str) -> str:
    """The strong letters of a root, for words from a weak root: «يعط» and «عطء» → «عط», «محتاج» and «حاج» → «حج»
    (the «ت» of «افتعل» after the first letter is dropped too). Marked with «~» so it is compared only with another
    skeleton."""
    for prefix in _DERIVED:
        if root.startswith(prefix) and len(root) - len(prefix) >= 2:
            root = root[len(prefix):]
            break
    core = root.translate(_WEAK)
    if len(core) == 3 and core[1] == "ت":
        core = core[0] + core[2]
    return f"~{core}" if len(core) >= 2 else ""


@lru_cache(maxsize=50000)
def roots(token: str) -> frozenset[str]:
    """The word's stem, its root (ISRI) and the strong letters of each, all normalized."""
    found = {stem(token)}
    isri = _isri()
    if isri is not None:
        root = normalize(isri.stem(token))
        if len(root) >= 3:
            found.add(root)
    found |= {skeleton(r) for r in found}
    return frozenset(f for f in found if f)


@lru_cache(maxsize=1)
def _wordnet(path: str = "") -> tuple[dict, dict]:
    """(word → its synset ids, synset id → its words), single words only; empty when the file is missing."""
    file = Path(path) if path else SYNONYMS_FILE
    synsets: dict[str, set[str]] = defaultdict(set)
    words: dict[str, set[str]] = defaultdict(set)
    if not file.exists():
        return {}, {}
    for line in file.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if line.startswith("#") or len(parts) < 3 or not parts[1].endswith(":lemma"):
            continue
        word = normalize(parts[2])
        if word and " " not in word:
            synsets[word].add(parts[0])
            words[parts[0]].add(word)
    return dict(synsets), dict(words)


def synonyms(token: str) -> set[str]:
    """Words the Arabic WordNet gives as the same meaning, looked up as written and by stem."""
    by_word, by_synset = _wordnet()
    if not by_word:
        return set()
    found: set[str] = set()
    for form in {token, stem(token), f"ال{stem(token)}"}:
        for s in by_word.get(form, ()):
            found |= by_synset[s]
    return found - {token}


def _same_form(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))


def in_text(token: str, text_roots: set[str]) -> bool:
    """`token` is in a text whose words' roots are `text_roots` (from text_roots()), in any form or as a synonym."""
    mine = set(roots(token))
    if any(r == h if r.startswith("~") else (not h.startswith("~") and _same_form(r, h))
           for r in mine for h in text_roots):
        return True
    near = set().union(*(roots(s) for s in synonyms(token))) if synonyms(token) else set()
    return bool(near & text_roots)


def text_roots(tokens: list[str]) -> set[str]:
    return set().union(*(roots(t) for t in tokens)) if tokens else set()
