"""Who said a passage, from the source list itself: never guessed from the text.

The evaluation of 5 October 2026 showed the model writing «الفتوى توضح أن…» and labelling views «الفتوى (z:4607)»:
a fatwa passage reached it marked only «فتوى», so it did not know whose fatwa it was, and the verifier then
dropped those views because «الفتوى» is no name. Each source with a known author now carries his name here: it is
shown beside the passage to the writing step, a view labelled with it is accepted (a fatwa does not name its
mufti in its own text), and a ruling that names no one is credited to the source it cites. The name comes from
the source the sentence cites, so naming him adds nothing that is not on the source card.

Sources with no single speaker (the Quran, the hadith collections, the books of asbab and sira) have none: a
ruling that cites only them is the writer's own reading of a text, not a scholar's word.
"""
from __future__ import annotations

import re

from .normalize import normalize, tokenize

# source id → (the person, as a view label and in «بيّن … أن»; English) — None when the source is a book
# that reports others' views (Ibn Rushd reports the jurists: a view is labelled with the jurist he names)
PERSON = {
    "fatawa-ibn-baz": ("الشيخ ابن باز", "Sheikh Ibn Baz"),
    "fatawa-ibn-uthaymeen": ("الشيخ ابن عثيمين", "Sheikh Ibn Uthaymeen"),
    "fatawa-arkan-islam": ("الشيخ ابن عثيمين", "Sheikh Ibn Uthaymeen"),
    "usul-al-sunna": ("الإمام أحمد", "Imam Ahmad"),
    "sharh-al-sunna-muzani": ("المزني", "al-Muzani"),
    "tawhid-ibn-khuzayma": ("ابن خزيمة", "Ibn Khuzayma"),
    "aqeeda-tahawiyya": ("الطحاوي", "al-Tahawi"),
}
# source id → (how the source is named when a sentence is credited to it; English)
CREDIT = {
    **PERSON,
    "bayyinat": ("كتاب «بينات»", "the book «Bayyinat»"),
    "bidayat-al-mujtahid": ("ما نقله ابن رشد في «بداية المجتهد»", "what Ibn Rushd reports in «Bidayat al-Mujtahid»"),
}


def person(source_id: str, lang: str = "ar") -> str:
    names = PERSON.get(source_id)
    return (names[1] if lang == "en" else names[0]) if names else ""


def label(source_id: str) -> str:
    """Who a passage is from, shown beside it to the writing step (a book that reports others' views has none)."""
    return person(source_id) or ("كتاب «بينات»" if source_id == "bayyinat" else "")


def credit(source_id: str, lang: str = "ar") -> str:
    names = CREDIT.get(source_id)
    return (names[1] if lang == "en" else names[0]) if names else ""


def _sources(passage_ids, corpus) -> list[str]:
    out = []
    for pid in passage_ids:
        p = corpus.passage(pid)
        if p is not None and p.source_id not in out:
            out.append(p.source_id)
    return out


def speaker(passage_ids, corpus, lang: str = "ar") -> str:
    """The one person every cited passage is from («الشيخ ابن عثيمين» for his two collections), or ""."""
    names = {person(s, lang) for s in _sources(passage_ids, corpus)}
    return names.pop() if len(names) == 1 and "" not in names else ""


def credits(passage_ids, corpus, lang: str = "ar") -> list[str]:
    """How each cited source with a known author is named, once each, in the order cited."""
    return list(dict.fromkeys(c for c in (credit(s, lang) for s in _sources(passage_ids, corpus)) if c))


def names_speaker(school: str, passage_ids, corpus) -> bool:
    """The label names the person all cited passages are from («ابن باز» or «الشيخ ابن باز» for a fatwa of his):
    a fatwa does not name its mufti in its own text, so the source list says who he is."""
    who = speaker(passage_ids, corpus)
    if not who:
        return False
    core = [t for t in tokenize(who) if t not in ("الشيخ", "الامام")]
    have = set(tokenize(school))
    return bool(core) and all(t in have for t in core)


# «الفتوى توضح أن…», «الشرح المختصر يبيّن أن…»: the kind of text as the subject, instead of who said it
_META_LEAD = re.compile(
    r"^\s*(?:و|ف)?(?:ال)?(?:فتوى|فتاوى|شرح(?:\s+المختصر)?|جواب(?:\s+التفصيلي)?|كتاب|مصدر|مقطع|نص)\s+"
    r"(?:ت|ي)(?:وضح|بيّ?ن|ؤكد|ذكر|قول|شير)\s+(?:إلى\s+)?(?:أنّ?|ان|إنّ?)\s+")
# «…، وهذا ما ورد في الفتوى», «…، وهو ما ذكره الفقهاء في باب أوقات العصر»: a closing remark about the text itself
_META_TAIL = re.compile(
    r"\s*[،,]?\s*(?:و|ف)?(?:هذا|هو|ذلك|قد)\s+(?:ما\s+)?(?:ورد|ذكر|ذكره|ذكرته|جاء|بينه|بيّ?نته|أوضحته|اوضحته|نصت\s+عليه)"
    r"(?:\s+ذلك)?(?:\s+(?:ال)?(?:فقهاء|علماء|فقه))?"
    r"\s+(?:في|عن)\s+(?:ال)?(?:فتوى|فتاوى|فقه|فقهاء|حديث|كتاب|شرح|مصدر|مقطع|نص|جواب|باب)[^.،,]*\.?\s*$")


def without_meta(text: str, passage_ids, corpus, lang: str = "ar") -> str:
    """The sentence with the text named as its own subject replaced by who said it, when that is one person or
    book («الفتوى توضح أن X» → «بيّن الشيخ ابن باز أن X»), and a closing «وهذا ما ورد في الفتوى» removed.
    Arabic only: the English answers did not show it."""
    if lang != "ar":
        return text
    tail = _META_TAIL.search(text)
    if tail and tail.start() > 0:
        text = text[:tail.start()].rstrip(" ،,") + "."
    lead = _META_LEAD.match(text)
    if lead:
        names = credits(passage_ids, corpus)
        if len(names) == 1:
            text = f"بيّن {names[0]} أن {text[lead.end():]}"
    return text


# a sentence that gives a ruling of its own («… حرام في الإسلام», «الراجح أن…»)
_RULING = frozenset(normalize(w) for w in (
    "حرام", "محرم", "يحرم", "تحرم", "حلال", "يجوز", "تجوز", "جائز", "لا يجوز", "واجب", "يجب", "تجب", "فرض",
    "مكروه", "يكره", "مستحب", "يستحب", "مباح", "يلزم", "يلزمه"))
# a sentence that prefers one view of its own («الراجح أن…», «الصحيح أن…»): a ruling in any answer
_PREFERS = frozenset(normalize(w) for w in ("الراجح", "الصحيح", "الأرجح", "والراجح", "والصحيح"))
_RULING_EN = re.compile(r"\b(?:forbidden|prohibited|haram|halal|permissible|permitted|obligatory|"
                        r"compulsory|disliked|recommended|the (?:correct|preferred|stronger) view)\b", re.IGNORECASE)
# the sentence already says whose words these are («قال…», «يرى…», «اتفق الفقهاء…», «عن النبي ﷺ…», «قوله تعالى»)
_SAYS = frozenset(normalize(w) for w in (
    "قال", "قالوا", "يقول", "يرى", "يرون", "رأى", "ذهب", "ذهبوا", "اتفق", "اتفقوا", "أجمع", "أجمعوا",
    "بحسب", "حسب", "وفق", "وفقا", "أفتى", "سئل", "أجاب", "روى", "النبي", "الرسول",
    "تعالى", "الشيخ", "الإمام", "الفقهاء", "العلماء", "الجمهور", "المذاهب", "الحنفية", "المالكية",
    "الشافعية", "الحنابلة", "مالك", "الشافعي", "أحمد", "حنيفة", "ﷺ"))
_SAYS_EN = re.compile(r"\b(?:according|said|says|held|holds|ruled|stated|states|replied|answered|reports?|"
                      r"narrated|scholars|jurists|the prophet|allah|sheikh|imam)\b|ﷺ", re.IGNORECASE)


def gives_ruling(text: str, lang: str = "ar") -> bool:
    if lang == "en":
        return bool(_RULING_EN.search(text))
    words = normalize(text).split()
    plain = set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}
    return bool(plain & _RULING)


def prefers(text: str) -> bool:
    words = normalize(text).split()
    return bool(words) and words[0] in _PREFERS


def says_who(text: str, lang: str = "ar") -> bool:
    if lang == "en":
        return bool(_SAYS_EN.search(text))
    words = set(normalize(text).split())
    # «وقال», «وسُئل», «فأجاب»: the same words with a joining letter in front
    words |= {w[1:] for w in words if w[:1] in "وف" and len(w) > 3}
    return bool(words & _SAYS) or "ﷺ" in text


def credited(text: str, passage_ids, corpus, lang: str = "ar") -> str | None:
    """A ruling sentence that does not say whose ruling it is, credited to the sources it cites
    («بحسب الشيخ ابن باز، …»). None when it cites no source with a known author (only verses or hadith):
    then the ruling is the writer's own reading of the text, and is not shown. Any other sentence as it is."""
    names = credits(passage_ids, corpus, lang)
    if prefers(text):  # «الراجح أن…»: whose preference it is, even when the sentence cites a hadith for it
        if any(all(t in set(tokenize(text)) for t in tokenize(n) if t not in ("الشيخ", "الامام")) for n in names):
            return text
    elif not gives_ruling(text, lang) or says_who(text, lang):
        return text
    if not names:
        return None
    if lang == "en":
        who = " and ".join(names)
        return f"According to {who}, {text[:1].lower() + text[1:]}"
    return f"بحسب {' و'.join(names)}، {text}"
