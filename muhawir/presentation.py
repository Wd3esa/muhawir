"""Formatting of already cited claims, independent of intent keyword matching.

No religious text is supplied here. Intent comes from the existing understanding
call; quoted words must occur in the cited original passage.
"""
from dataclasses import replace
import re
from .normalize import normalize


# the words that attribute what follows to the Prophet ﷺ in a book that quotes hadith among other text
_PROPHET = re.compile(r"(?:صلي الله عليه وسلم|ﷺ|رسول الله|النبي|عليه الصلاه والسلام)")
_ELLIPSIS = re.compile(r"\s*(?:…|\.{3,})\s*")
PROPHET_SPAN = 200  # characters before the quote in which the source names the Prophet ﷺ
# kinds whose text is never a source for showing a hadith: a takhrij book also lists sayings that are not hadith
_NOT_HADITH_TEXT = frozenset({"quran", "gharib", "takhrij"})


def _found_at(quote: str, text: str) -> int:
    """Where the quotation starts in the passage (normalised), or -1. A quotation shortened with «…» is found when
    each of its parts of two words or more is in the passage, in order."""
    parts = [p for p in (normalize(x) for x in _ELLIPSIS.split(quote)) if p]
    if not parts or (len(parts) > 1 and any(len(p.split()) < 2 for p in parts)):
        return -1
    start = at = text.find(parts[0])
    for part in parts[1:]:
        if at < 0:
            return -1
        at = text.find(part, at + 1)
    return start if at >= 0 else -1


def required_quote(claim, passages, form):
    """True when the sentence shows the text asked for (a verse or a hadith) word for word from a cited passage:
    a verse from the mushaf; a hadith from a book of hadith, or from another book that quotes it as the Prophet's
    words ﷺ (a fatwa, a commentary, رياض الصالحين)."""
    if form not in ('hadith_text', 'verse_text', 'verse_evidence'):
        return False
    form = 'verse_text' if form == 'verse_evidence' else form
    for quote in re.findall(r'«([^»]+)»|﴿([^﴾]+)﴾', claim.text):
        text = next((q for q in quote if q), '')
        if not normalize(text):
            continue
        for pid in claim.passage_ids:
            p = passages.get(pid)
            if p is None:
                continue
            body = normalize(p.text)
            at = _found_at(text, body)
            if at < 0:
                continue
            if form == 'verse_text':
                if p.kind == 'quran':
                    return True
                continue
            if p.kind == 'hadith':
                return True
            if p.kind not in _NOT_HADITH_TEXT and len(normalize(text).split()) >= 3 \
                    and _PROPHET.search(body[max(0, at - PROPHET_SPAN):at]):
                return True
    return False


def lead_with_requested_text(claims, passages, request):
    form = (request or {}).get('form')
    index = next((i for i, c in enumerate(claims) if not c.school and required_quote(c, passages, form)), None)
    if index is None or index == 0:
        return list(claims)
    return [claims[index], *claims[:index], *claims[index + 1:]]


def compact_provenance(claims, corpus, lang):
    """Remove only repeated book-only clauses tied to the same cited passage.

    A grading, different narrator, condition or differing source is not deleted.
    Narrator detail following a repeated book introduction is retained.
    """
    if lang != 'ar':
        return list(claims)
    seen, out = set(), []
    for claim in claims:
        text = claim.text
        for pid in claim.passage_ids:
            p = corpus.passage(pid)
            if p is None or p.kind != 'hadith':
                continue
            book = corpus.source_of(p).name
            # Only these literal catalog names have a short, unambiguous Arabic introduction.
            if book not in ('صحيح مسلم', 'صحيح البخاري'):
                continue
            short = book.split()[-1]
            if pid in seen:
                alias = '(?:' + re.escape(book) + '|' + re.escape(short) + ')'
                text = re.sub(r'(?<!\S)و?مصدره\s+(?:هو\s+)?' + alias +
                              r'(?:[.،]\s*|\s+(?=و?قد\s+أخرجه)|$)', '', text)
                text = re.sub(r'(?<!\S)و?(?:قد\s+)?أخرجه\s+' + alias + r'\s+عن\s+',
                              'والراوي ', text)
                text = re.sub(r'^(?:و?قد\s+)?(?:رواه|أخرجه)\s+' + alias + r'[.،]?\s*$', '', text)
            if re.search(r'(?:روى|رواه|أخرجه|صحيح)\s+(?:الإمام\s+)?' + re.escape(short), text):
                seen.add(pid)
        text = text.strip(' ،.') if not text.strip(' ،.') else text.strip()
        if text:
            out.append(replace(claim, text=text))
    return out


def retain_inline_qualification(claims, corpus):
    """Keep an explicit non-establishment notice with its uniquely quoted report.

    This is not grading a report. Copy the source's own sentence only when it
    contains one quoted report and the answer quotes that same report. Ambiguous
    passages remain the semantic support reader's responsibility.
    """
    out = []
    for claim in claims:
        text = claim.text
        for pid in claim.passage_ids:
            p = corpus.passage(pid)
            if p is None or p.kind not in ('fiqh', 'tafsir', 'takhrij'):
                continue
            quotes = re.findall(r'«([^»]+)»', p.text)
            if len(quotes) != 1 or len(quotes[0].split()) < 4 or normalize(quotes[0]) not in normalize(text):
                continue
            # A literal statement in the supplied source, not an inference about authenticity.
            notices = [sentence.strip() for sentence in re.split(r'[.\n؟]', p.text)
                       if re.search(r'لم\s+يرد\s+(?:في\s+ذلك|فيه)\s+حديث\s+ثابت', sentence)]
            for notice in notices:
                if normalize(notice) not in normalize(text):
                    text = text.rstrip(' .') + '. ويذكر المصدر: ' + notice + '.'
        out.append(replace(claim, text=text))
    return out
