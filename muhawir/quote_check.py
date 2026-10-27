"""Quote-check: where in the sources does a pasted quotation (a verse, a hadith, a jurist's sentence) appear?

Nothing here calls a model: the passages come from the same search the answers use, and the match is made on the
words themselves. A quotation is reported as exact only when every one of its words follows the others, in order,
inside one stored passage; a quotation recalled with wrong edges is reported as close, with the longest piece that
does appear. The excerpt shown is always cut verbatim from the stored passage, never from normalized text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import normalize

MIN_QUOTE_WORDS = 3        # fewer words than this cannot point to one passage
MAX_QUOTE_CHARS = 600
CLOSE_COVERAGE = 0.6       # share of the quotation's words found in one unbroken run
CLOSE_MIN_RUN = 4          # and the run itself must be this long
KIND_ORDER = {"quran": 0, "hadith": 1}   # the verse or hadith itself first, then commentaries, books and fatwas
PRIMARY_HADITH = ("sahih-bukhari", "sahih-muslim")  # then the other hadith books
GENERIC_CANDIDATES = 12    # a phrase is called generic when, of the best passages the search returns,
GENERIC_MIN = 8            # at least this many
GENERIC_COVERAGE = 0.8     # hold at least this share of its words (it is in the wording of many passages)
FUZZY_MIN_WORD = 4         # a word of at least this many letters may differ by one letter (a missing alef, a wrong letter)
FUZZY_COVERAGE = 0.7       # share of the quotation's words found, in order and close together, for a "close" match
FUZZY_MIN_FOUND = 4
FUZZY_MAX_STARTS = 150     # places tried in one passage
FUZZY_SLACK = 2            # words the passage may add inside the span of the quotation
_WORD = re.compile(r"\S+")

# Quote-in-box: a message typed or pasted in the question box is looked up as a quotation, with no model, only when it
# is long enough to point to one passage and asks nothing. Most people write without punctuation, so the test is on
# words: a first word that asks, requests or names what is asked about («ما، هل، اذكر، صحة، حديث…»), or a last word
# that asks about the text («صحيح، معناه، تفسيره…»), makes it a question. The lookup decides the rest: only a bare text
# found in the sources word for word is shown as a quotation; a nearly matching one is answered as a question.
QUOTE_IN_BOX_WORDS = 6
_ASK_FIRST = {normalize(w) for w in (
    "ما", "ماذا", "من", "متى", "اين", "أين", "كيف", "كم", "لماذا", "لم", "هل", "أي", "اي", "ماهو", "ماهي", "حكم",
    "معنى", "تفسير", "شرح", "شنو", "ايش", "إيش", "وش", "ليش", "ليه", "إيه", "شو", "منو", "وين", "ليه",
    "اذكر", "أذكر", "اشرح", "عدد", "عدّد", "بين", "بيّن", "وضح", "وضّح", "فسر", "فسّر", "أعطني", "اعطني", "عطني",
    "أريد", "اريد", "ابغى", "أبغى", "ابي", "أبي", "ودي", "ممكن", "لو", "أخبرني", "اخبرني", "قل", "علمني", "علّمني",
    "صحة", "صحه", "مدى", "درجة", "درجه", "تخريج", "مصدر", "حديث", "الحديث", "آية", "اية", "الآية", "قول", "مقولة",
    "عبارة", "دليل", "سبب", "فضل", "عن",
    "what", "who", "when", "where", "how", "why", "is", "are", "does", "do", "can", "explain", "tell", "list", "give")}
_ASK_LAST = {normalize(w) for w in (
    "صحيح", "صحيحه", "صحيحة", "صحته", "ضعيف", "معناه", "معناها", "تفسيره", "تفسيرها", "شرحه", "شرحها", "حكمه", "حكمها",
    "درجته", "درجتها", "سنده", "مصدره", "مصدرها", "true", "authentic", "mean", "meaning")}


def looks_like_quote(text: str) -> bool:
    """True when a message may be a quotation to look up rather than a question: enough words, and none of them at
    the start or the end asks or requests anything. The lookup then decides whether it is one."""
    words = normalize(text).split()
    if len(words) < QUOTE_IN_BOX_WORDS:
        return False
    return words[0] not in _ASK_FIRST and words[-1] not in _ASK_LAST and "هل" not in words[-3:]


@dataclass(frozen=True)
class QuoteMatch:
    """One stored passage that holds the quotation, wholly (exact) or in its longest unbroken piece (close)."""
    passage_id: str
    status: str            # "exact" or "close"
    coverage: float        # words of the longest unbroken run / words of the quotation
    excerpt: str           # verbatim from the stored passage, around the matched run
    run: int
    start: int = 0         # where the excerpt begins and ends in the stored passage's text
    end: int = 0


def _words(text: str) -> list[tuple[str, int, int]]:
    """(normalized word, start, end) for each word of the original text; words that normalize to nothing are skipped."""
    out = []
    for m in _WORD.finditer(text):
        w = normalize(m.group())
        if w:
            out.append((w, m.start(), m.end()))
    return out


def _longest_run(a: list[str], b: list[str]) -> tuple[int, int, int]:
    """Longest run of words found one after the other in both lists: (length, start in a, start in b)."""
    best = (0, 0, 0)
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best[0]:
                    best = (current[j], i - current[j], j - current[j])
        previous = current
    return best


def _same_word(a: str, b: str) -> bool:
    """Equal, or (for words of four letters or more) one letter apart: «اسرفو» and «اسرفوا», «يغفر» and «تغفر»."""
    if a == b:
        return True
    if len(a) < FUZZY_MIN_WORD or len(b) < FUZZY_MIN_WORD or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _fuzzy_span(quote_words: list[str], words: list[tuple[str, int, int]]) -> tuple[int, int, int] | None:
    """The best place in the passage for a quotation with a wrong or misspelt word: (found, first word, last word).
    The quotation's words are matched in order against a stretch of the passage no longer than the quotation plus a
    little, words one letter apart counting as equal; a stretch is tried only where one of the quotation's first three
    words occurs, so a long passage is not scanned everywhere."""
    n = len(quote_words)
    forms = [w for w, _s, _e in words]
    present = set(forms)
    if sum(1 for w in set(quote_words) if w in present) < FUZZY_COVERAGE / 2 * len(set(quote_words)):
        return None  # most of the quotation's words are not in this passage at all
    best = None
    tried = 0
    for start in range(len(forms)):
        if not any(_same_word(forms[start], quote_words[k]) for k in range(min(3, n))):
            continue
        tried += 1
        if tried > FUZZY_MAX_STARTS:
            break
        window = forms[start:start + n + FUZZY_SLACK]
        prev = [0] * (len(window) + 1)
        for i in range(1, n + 1):
            cur = [0] * (len(window) + 1)
            for c in range(1, len(window) + 1):
                cur[c] = max(prev[c], cur[c - 1], prev[c - 1] + 1 if _same_word(quote_words[i - 1], window[c - 1]) else 0)
            prev = cur
        found = prev[-1]
        end = next(c for c in range(len(window) + 1) if prev[c] == found)
        if best is None or (found, -end) > (best[0], -(best[2] - best[1] + 1)):
            best = (found, start, start + end - 1)
    return best


def check_passage(quote_words: list[str], passage_id: str, text: str) -> QuoteMatch | None:
    """How well one stored passage holds the quotation; None when it is not close enough to report."""
    words = _words(text)
    run, _qs, start = _longest_run(quote_words, [w for w, _s, _e in words])
    coverage = run / len(quote_words)
    if run == len(quote_words):
        status = "exact"
    elif coverage >= CLOSE_COVERAGE and run >= CLOSE_MIN_RUN:
        status = "close"
    else:
        fuzzy = _fuzzy_span(quote_words, words)
        if fuzzy is None or fuzzy[0] < FUZZY_MIN_FOUND or fuzzy[0] / len(quote_words) < FUZZY_COVERAGE:
            return None
        found, a, b = fuzzy
        return QuoteMatch(passage_id, "close", round(found / len(quote_words), 3), text[words[a][1]:words[b][2]], found,
                          words[a][1], words[b][2])
    first, last = words[start], words[start + run - 1]
    return QuoteMatch(passage_id, status, round(coverage, 3), text[first[1]:last[2]], run, first[1], last[2])


def window(text: str, start: int, end: int, reach: int = 260) -> dict:
    """The stored passage cut to `reach` characters either side of the matched run, at word edges, in three parts so
    the page can print the run in bold: {"lead", "match", "trail", "cut_lead", "cut_trail"}."""
    a, b = max(0, start - reach), min(len(text), end + reach)
    if a > 0:
        a = (text.find(" ", a, start) + 1) or a
    if b < len(text):
        b = text.rfind(" ", end, b) if text.rfind(" ", end, b) > end else b
    return {"lead": text[a:start], "match": text[start:end], "trail": text[end:b], "cut_lead": a > 0, "cut_trail": b < len(text)}


def edge(text: str, reach: int = 260, tail: bool = False) -> str:
    """The last (tail) or first `reach` characters of a neighbouring passage, at a word edge, with «…» where cut."""
    if len(text) <= reach:
        return text
    if tail:
        piece = text[-reach:]
        return "\u2026" + piece[piece.find(" ") + 1:]
    piece = text[:reach]
    return piece[:piece.rfind(" ")] + "\u2026" if " " in piece else piece + "\u2026"


def is_generic(retriever, quote: str) -> bool:
    """True when the quotation is made of words found together all over the sources («أن رسول الله صلى الله عليه
    وسلم كان»): no exact passage can be named for it, and «not found» would be the wrong thing to tell the visitor."""
    hits = retriever.search(quote, k=GENERIC_CANDIDATES)
    return sum(1 for h in hits if h.coverage >= GENERIC_COVERAGE) >= GENERIC_MIN


def find_quote(retriever, quote: str, limit: int = 8, candidates: int = 12) -> list[tuple[QuoteMatch, object]]:
    """Passages holding the quotation: exact ones first, then the Quran, then hadith (Bukhari and Muslim first), then
    the other sources, then by how much of the quotation each holds. Each item is (match, the search's Hit) so the
    caller can show the source and location."""
    quote_words = normalize(quote).split()
    if len(quote_words) < MIN_QUOTE_WORDS:
        return []
    found = []
    for rank, hit in enumerate(retriever.search(quote, k=candidates)):
        match = check_passage(quote_words, hit.passage.id, hit.passage.text)
        if match:
            found.append((match, hit, rank))
    found.sort(key=lambda item: (item[0].status != "exact", KIND_ORDER.get(item[1].passage.kind, 2),
                                 item[1].passage.source_id not in PRIMARY_HADITH, -item[0].coverage, item[2]))
    return [(m, h) for m, h, _r in found[:limit]]
