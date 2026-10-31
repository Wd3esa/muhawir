"""Formatting of already cited claims, independent of intent keyword matching.

No religious text is supplied here. Intent comes from the existing understanding
call; quoted words must occur in the cited original passage.
"""
from dataclasses import replace
import re
from .normalize import normalize


def required_quote(claim, passages, form):
    kind = {'hadith_text': 'hadith', 'verse_text': 'quran'}.get(form)
    if not kind:
        return False
    for quote in re.findall(r'«([^»]+)»|﴿([^﴾]+)﴾', claim.text):
        text = next((q for q in quote if q), '')
        if not normalize(text):
            continue
        if any(p.kind == kind and normalize(text) in normalize(p.text)
               for pid in claim.passage_ids if (p := passages.get(pid)) is not None):
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
