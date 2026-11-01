"""Word-level support for a dictionary question («ما معنى X»): is the asked word really in the entry?

BM25 matches whole normalized tokens. That fails for a dictionary question in three ways; the first two were seen live
(«ما معنى ألفوا آباءهم» was answered from the entry «أوى», which has nothing to do with the word):

  1. A word that every question about a meaning contains («معنى») is in a large share of the entries of a dictionary,
     so a word the book does not have still "covers" half of the question. `focus_terms` leaves such words out.
  2. The asked word is a form the book does not spell, though it explains it under its root («ألفوا» is explained in
     «لفى» as «ألفيت: وجدت»). `RootIndex` maps a word to the headings it may come from.
  3. A passage found by meaning (a vector) or by an expansion of the question may be about another word. `keeps`
     rejects a dictionary entry that does not hold the user's rare word as written, as its root, or in a close
     Smith-Waterman alignment.

The alignment is character-level only: it cannot know that two unrelated spellings mean the same, so it only rejects
or accepts a spelling, and never proves a meaning. It is also weak on its own (measured on «المفردات»: a threshold
that keeps most true forms lets a few percent of unrelated entries through), which is why it is the third test after
the exact word and the root, not the first.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from . import wordforms
from .normalize import STOPWORDS, normalize, tokenize

# Words that frame a question about a meaning and name no topic. Ignored when a question is checked against a passage.
FRAME_WORDS = frozenset(normalize(w) for w in """
معنى معاني المعنى معنا معناه معناها بمعنى يعني يعنى تعني تعنى المقصود مقصود المراد مراد تفسير تفسيره شرح اشرح فسر
توضيح كلمة كلمه لفظ لفظة لفظه
""".split())

MATCH, MISMATCH, GAP = 2, -1, -1
MIN_ALIGNMENT = 0.8   # strict alignment of the whole asked word to a whole word of the entry (see scripts/eval_lexical.py)
RARE_DF = 5           # a question word found in at most this many passages of the whole index is the word asked about
MAX_HEADING_PASSAGES = 6

_WEAK = frozenset("اويىءأإآئؤ")
_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل", "س", "ي", "ت", "ن", "م")
_SUFFIXES = ("ونا", "وها", "وهم", "ناه", "نها", "هما", "كما", "تم", "تن", "ون", "ين", "ان", "ات", "وا", "نا",
             "ها", "هم", "هن", "كم", "كن", "ني", "ته", "تا", "ت", "ه", "ا", "ي", "ك", "ن", "ة")


def sw_score(a: str, b: str) -> int:
    """Smith-Waterman local alignment score of two short strings (match +2, mismatch -1, gap -1)."""
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            diag = prev[j - 1] + (MATCH if a[i - 1] == b[j - 1] else MISMATCH)
            cur[j] = max(0, diag, prev[j] + GAP, cur[j - 1] + GAP)
            if cur[j] > best:
                best = cur[j]
        prev = cur
    return best


def alignment(term: str, word: str) -> float:
    """How much of `term` is found in `word`, 0..1 (1 = every letter of the term aligns)."""
    if not term or not word:
        return 0.0
    return sw_score(term, word) / (MATCH * len(term))


def strong(word: str) -> str:
    """The strong letters of a word: «ألفوا» → «لف». Two words of one root share them."""
    return "".join(c for c in normalize(word) if c not in _WEAK)


def cores(word: str) -> set[str]:
    """The word with a joined letter and/or a common ending taken off, in every combination (all normalized)."""
    word = normalize(word)
    starts = {word}
    for p in _PREFIXES:
        if word.startswith(p) and len(word) - len(p) >= 2:
            starts.add(word[len(p):])
    for p in _PREFIXES:  # a second prefix: «يتفكرون» = «ي» + «ت» + «فكر»
        for s in list(starts):
            if s.startswith(p) and len(s) - len(p) >= 3:
                starts.add(s[len(p):])
    found = set(starts)
    for core in starts:
        for end in _SUFFIXES:
            if core.endswith(end) and len(core) - len(end) >= 2:
                found.add(core[:-len(end)])
                for end2 in ("ك", "ه", "م", "كم", "هم", "نا"):  # «بعثناكم» = «بعث» + «نا» + «كم»
                    rest = core[:-len(end)]
                    if rest.endswith(end2) and len(rest) - len(end2) >= 2:
                        found.add(rest[:-len(end2)])
    return {c for c in found if len(c) >= 2}


def keys(word: str) -> set[str]:
    """Comparable root keys of a word: the strong letters of each of its cores (≥ 2), and the stemmer's skeletons."""
    out = {strong(c) for c in cores(word)}
    out = {k for k in out if len(k) >= 2}
    out |= {k for k in wordforms.roots(normalize(word)) if k.startswith("~")}
    return out


def heading_keys(heading: str) -> set[str]:
    """The same keys for a heading of the dictionary («لفى» → «لف»)."""
    out = {strong(heading)}
    skeleton = wordforms.skeleton(normalize(heading))
    if skeleton:
        out.add(skeleton)
    return {k for k in out if len(k) >= 2}


class RootIndex:
    """The headings of a dictionary arranged by root, looked up by the keys of a word."""

    def __init__(self, headings: Iterable[str]) -> None:
        self.by_key: dict[str, set[str]] = {}
        for heading in set(headings):
            for key in heading_keys(heading):
                self.by_key.setdefault(key, set()).add(heading)

    def candidates(self, word: str) -> set[str]:
        """Headings the word may come from."""
        found: set[str] = set()
        for key in keys(word):
            found |= self.by_key.get(key, set())
        return found


def is_meaning_question(question: str) -> bool:
    """The question asks what a word means (it holds a framing word such as «معنى»)."""
    return any(w in FRAME_WORDS for w in normalize(question).split())


def focus_terms(question: str) -> list[str]:
    """The tokens of a question that name the topic: no stopwords, no framing words."""
    return [t for t in dict.fromkeys(tokenize(question)) if t not in FRAME_WORDS]


def raw_terms(question: str) -> list[str]:
    """The question's topic words as normalized, with the article still on (the gate compares letters)."""
    return [w for w in dict.fromkeys(normalize(question).split())
            if len(w) >= 2 and w not in STOPWORDS and w not in FRAME_WORDS]


def supports(term: str, entry_words: set[str], heading: str = "", min_alignment: float = MIN_ALIGNMENT,
             heading_alignment: float = 0.0) -> bool:
    """The entry holds the asked word as written, as its heading, or in a close alignment with one of its words.

    `heading_alignment` > 0 asks a heading match to be backed by a (weaker) alignment with a word of the entry too:
    a stemmer finds a root for any string, so a word that is not Arabic at all can still "have" a root."""
    t = normalize(term)
    if t in entry_words or any(len(c) >= 3 and c in entry_words for c in cores(t)):
        return True
    close = lambda floor: any(abs(len(w) - len(t)) <= 2 and alignment(t, w) >= floor for w in entry_words)  # noqa: E731
    if heading and keys(t) & heading_keys(heading):
        return heading_alignment <= 0 or close(heading_alignment)
    return close(min_alignment)


def keeps(passage, user_text: str, df: Callable[[str], int] | None = None, **kw) -> bool:
    """False for a dictionary entry that is not about the rare word the user asked about; True for anything else.

    Only entries of a dictionary (kind «gharib») are judged: this check is measured on one. `df` gives the number of
    passages of the index that hold a token; without it nothing is rejected."""
    if getattr(passage, "kind", "") != "gharib" or df is None:
        return True
    terms = raw_terms(user_text)
    # the check is for a question that asks about a word the index barely knows («ألفوا» is in one verse);
    # a question of common words is the business of the keyword search alone
    if not any((tokens := tokenize(t)) and df(tokens[0]) <= RARE_DF for t in terms):
        return True
    # the entry must hold at least one of the user's words (as written, as its root, or aligned): a vector search or an
    # expanded phrase can bring an entry about another word («أوى» for «ألفوا»), which holds none of them
    words = set(normalize(passage.text).split())
    return any(supports(t, words, passage.keywords, **kw) for t in terms)
