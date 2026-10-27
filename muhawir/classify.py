"""Fixed rules that stop a question before retrieval or generation.

Only the cases the evaluation list marks as level D (personal fatwa or case),
judging specific people, and attempts to override the rules are detected
here. Levels A to C are not guessed by rules: telling them apart needs the
approved material itself, which is not in the repository yet.

Note: these rules were written with the draft evaluation list visible, so
their result on that list is not independent evidence of accuracy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import normalize

PERSONAL_CASE = "personal_case"      # level D: general info only + referral
CRISIS = "crisis"                    # thoughts of suicide or self-harm: a fixed caring reply, no model
JUDGING_PEOPLE = "judging_people"    # out of scope: decline politely
OVERRIDE = "override_attempt"        # keep the rules, explain, refer
OUT_OF_SCOPE = "out_of_scope"        # contemporary financial rulings: refer to fatwa bodies (scope of version 1)
SMALL_TALK = "small_talk"            # greeting or thanks only: short fixed reply, no search

_GREETING = (r"السلام عليكم(?: ورحمه الله(?: وبركاته)?)?|عليكم السلام(?: ورحمه الله(?: وبركاته)?)?|"
             r"مرحبا|اهلا(?: وسهلا)?|صباح الخير|مساء الخير|hello|hi|hey|salam|assalamu alaikum|"
             r"السلام عل[یي][کك]م|اداب|خوش امد[یي]د")
_THANKS = (r"شكرا(?: جزيلا)?|جزاك(?:م)? الله(?: كل)? خيرا?|بارك الله فيك(?:م)?|فتح الله عليك(?:م)?|"
           r"احسن الله اليك(?:م)?|thank you(?: very much)?|thanks|jazakallah(?: khair)?|"
           r"ش[کك]ر[یي][ہه]|ب[ہه]ت ش[کك]ر[یي][ہه]|جزا[کك] الل[ہه](?: خ[یي]ر)?")


def _only(phrases: str) -> re.Pattern:
    return re.compile(rf"^(?:(?:و ?)?(?:{phrases})\s*)+$")


_SMALL_TALK, _THANKS_ONLY = _only(f"{_GREETING}|{_THANKS}"), _only(_THANKS)


def is_small_talk(question: str) -> bool:
    """True when the whole message is a greeting or other small talk."""
    return bool(_SMALL_TALK.match(normalize(question)))


def is_thanks(question: str) -> bool:
    """True when the whole message is only thanks."""
    return bool(_THANKS_ONLY.match(normalize(question)))

_PERSONAL = [re.compile(p) for p in (
    # Only what a scholar must hear the details of. A first-person question about worship, purity, fasting, zakat, an
    # oath's expiation or food has a known ruling and is answered from the sources (an own illness or journey, «هل علي»,
    # «يلزمني», «حلفت» are no longer referred); what stays is the wording of a divorce (whether it took place depends
    # on the words and the intention) and a question that says its own details change the answer. Disputes over an
    # inheritance or a right between people, and a contemporary matter with private facts, are told apart by the
    # understanding step, which sets `personal`.
    # The list rests on the books of adab al-fatwa, not on inference: ibn as-Salah, Adab al-Mufti pp. 115-116 and ibn Hamdan,
    # Sifat al-Mufti p. 198 (no fatwa on oaths, confessions and wording unless one knows the speaker's usage); ibn al-Qayyim,
    # I'lam al-Muwaqqi'in 1/25-26 (the fact must be known) and 6/92 (ask before ruling on a breach of an oath); an-Nawawi,
    # Adab al-Fatwa pp. 45-46 (a ruling with kinds is answered by giving each kind its ruling) and pp. 44-45, 63-65 (what
    # one does not know one holds back from: «I do not know», pointing to another mufti only if there is one).
    r"\bفي حالتي\b", r"\bطلقت\b", r"\bقلت ?لها (انتي|انت|انتِ) طالق\b",
    r"\bin my case\b", r"\bmy divorce\b",
)]

# matched on normalized text (أ/إ → ا, ؤ → و, ة → ه, ى → ي, no diacritics)
_CRISIS = [re.compile(p) for p in (
    # the person speaking about themselves; a question about the topic («ما حكم الانتحار») is answered normally
    r"\bسانتحر", r"\bساقتل نفسي\b", r"\bاريد ان انتحر", r"\bابي انتحر", r"\bابغي انتحر", r"\bودي انتحر",
    r"\bافكر (?:في |ب)(?:ال)?انتحار", r"\bافكر ان انتحر", r"\bافكر في قتل نفسي\b", r"\bاقتل نفسي\b", r"\bاقتل حالي\b", r"\bانهي حياتي\b", r"\bانهاء حياتي\b",
    r"\bاوذي نفسي\b", r"\bايذاء نفسي\b", r"\bاذي نفسي\b",
    r"\bلا اريد ان اعيش\b", r"\bما ابي اعيش\b", r"\bما بدي عيش\b", r"\bمش عايز اعيش\b",
    # the main dialects (19 October 2026): Sudanese «داير», Levantine «بدي», Egyptian «عايز / عاوز»,
    # Gulf «ودي / ابي», Maghrebi «بغيت / حاب»
    r"\b(?:ما |مش )?(?:داير|دايره|بدي|عايز|عايزه|عاوز|عاوزه|ودي|ابي|ابغي|بغيت|حاب|حابه) (?:ا|ن)?موت\b",
    r"\b(?:داير|دايره|بدي|عايز|عايزه|عاوز|عاوزه|ودي|ابغي|بغيت|حاب|حابه) (?:ا|ن)?نتحر\b",
    r"\b(?:داير|دايره|بدي|عايز|عايزه|عاوز|عاوزه|ابي|ابغي|ودي|بغيت|حاب|حابه) (?:اقتل|نقتل) (?:نفسي|حالي|روحي)\b",
    r"\b(?:ما|مش|مو|موش)(?: \S+)? (?:داير|دايره|عايز|عاوز|عاوزه|عايزه|ابي|ابغي|ودي|بدي|اريد|بغيت|باغي|باغيه|حاب|حابه|نبغي) (?:ا|ن)?عيش\b",
    r"\b(?:اريد|باغي|باغيه) (?:ا|ن)?(?:موت|نتحر)\b",
    r"\b(?:راح|رح|حا) انتحر\b", r"\b(?:هنتحر|حنتحر|بنتحر)\b", r"\bهموت نفسي\b", r"\bبقتل نفسي\b", r"\bنقتل روحي\b",
    r"\bاريد ان اموت\b", r"\bابغي اموت\b", r"\bابي اموت\b", r"\bنفسي اموت\b", r"\bتمنيت الموت\b",
    r"\bkill myself\b", r"\bend my life\b", r"\bwant to die\b", r"\bhurt myself\b",
    r"\bi (?:want|am going|m going|will|plan) to (?:commit suicide|end it all)\b", r"\bthinking (?:about|of) (?:suicide|killing myself)\b",
    r"\bdon ?t want to live\b",
    # Urdu (22 October 2026), on the same normalised text (آ → ا); ی/ي, ک/ك and ہ/ه are both allowed, as keyboards differ
    r"خود ?[کك]ش[یي] (?:[کك]ر|[کك]رنا|[کك]رنے|[کك]روں|[کك]ر لوں|[کك]رلوں)",
    r"(?:مرنا|مر جانا|مرجانا) چا[ہه]ت[اےی]",
    r"(?:ج[یي]نا|زند[ہه] ر[ہه]نا) ن[ہه][یي]ں چا[ہه]ت[اےی]",
    r"(?:خود|اپن[ےي] اپ) [کك]و (?:مار|ختم [کك]ر|نقصان پ[ہه]نچا)",
    r"اپن[یي] جان ل[ےي]", r"زندگ[یي] ختم [کك]ر",
)]

_JUDGING = [re.compile(p) for p in (
    # a named person ("هل فلان كافر"); not a category ("هل تارك الصلاة كافر") or a figure the sources name
    r"\bهل (?!ال|تارك|من\b|ما\b|كل\b|ابليس\b|فرعون\b|قارون\b|هامان\b)\S+( \S+){0,3} (كافر|مرتد|منافق|مبتدع)\b",
    r"\bis (?!it\b)\S+( \S+){0,3} (a )?(kafir|disbeliever|apostate|infidel)\b",
)]

_OVERRIDE = [re.compile(p) for p in (
    r"\bتجاهل\b.*\b(ال)?تعليمات", r"\bانس\b.*\b(ال)?تعليمات", r"\bبما تراه انت\b", r"\bرايك الشخصي\b",
    r"\bignore\b.*\binstructions\b", r"\bdisregard\b.*\b(rules|instructions)\b",
    r"\byour own (opinion|fatwa)\b",
)]


@dataclass(frozen=True)
class Gate:
    """The result of the safety check on a question: its kind, or None when it may proceed."""
    kind: str | None   # one of the constants above, or None when the question may proceed
    rule: str = ""     # the pattern that matched, for logs and tests


# Rulings on contemporary financial products need ijtihad by fatwa bodies and fiqh academies. Muhawir never
# judges them itself: it quotes the published fatwas in its sources, attributed to their authors, says plainly
# that this is no ruling on the product asked about, and refers to the official fatwa body (EVALUATION.md, Q25, Q26).
_OUT_OF_SCOPE = [re.compile(p) for p in (
    r"\b(تابي|تمارا|tabby|tamara)\b",  # buy-now-pay-later companies, named
    r"\b(التقسيط|بالتقسيط|تقسيط|الاقساط|اقساط)\b.*\b(حكم|حلال|حرام|يجوز|جائز|ربا|الربا|ربوي|ربويه)\b",
    r"\b(حكم|حلال|حرام|يجوز|جائز|ربا|الربا|ربوي|ربويه)\b.*\b(التقسيط|بالتقسيط|تقسيط|الاقساط|اقساط)\b",
    r"\b(buy now,? pay later|bnpl|instal?ments?)\b.*\b(halal|haram|permissible|allowed|riba|interest)\b",
    r"\b(حكم|حلال|حرام|يجوز|جائز)\b.*\b(البنوك?|بنكي[هة]?|المصارف|فوا[يئ]د|عملات رقمي[هة]|العملات الرقمي[هة]|الرقمي[هة]|بيتكوين|البيتكوين|كريبتو|التداول|تداول|الفوركس|فوركس|الاسهم|اسهم|التامين|تامين)\b",
    r"\b(البنوك?|بنكي[هة]?|المصارف|فوا[يئ]د|عملات رقمي[هة]|العملات الرقمي[هة]|بيتكوين|البيتكوين|كريبتو|التداول|تداول|الفوركس|فوركس|الاسهم|التامين)\b.*\b(حلال|حرام|يجوز|جائز)\b",
    r"\b(is|are)\b.*\b(bank interest|crypto|bitcoin|forex|stock trading|insurance)\b.*\b(halal|haram|permissible|allowed)\b",
    # the topic alone, with no question («العملات الرقمية», "bitcoin"): the model would turn it into a question of its
    # own, a ruling one or a «what is» one, so the same words got a referral one time and «not found» the next
    r"^(ما )?(حكم )?(عملات رقميه|العملات الرقميه|العمله الرقميه|عمله رقميه|بيتكوين|البيتكوين|بتكوين|كريبتو|الكريبتو"
    r"|الفوركس|فوركس|التداول|تداول الاسهم|تداول العملات|الاسهم|فوا[يئ]د البنوك|فوا[يئ]د البنك|الفوا[يئ]د البنكيه|التامين"
    r"|التامين التجاري|التقسيط|البيع بالتقسيط)( في الاسلام)?$",
    r"^(crypto|cryptocurrency|cryptocurrencies|bitcoin|forex|bank interest|insurance|stock trading)( in islam)?$",
)]


# the product a contemporary financial question names, and the words a passage must contain to speak about it.
# A fatwa on paper money or currency exchange says nothing about cryptocurrencies: quoted under that question it
# reads as an answer to it (live site, 6 October 2026). The specific products first; «التداول» alone is general.
_PRODUCTS = [
    (r"رقمي|بيتكوين|بتكوين|كريبتو|crypto|bitcoin", ("رقمي", "بيتكوين", "بتكوين", "كريبتو", "مشفر", "crypto", "bitcoin")),
    (r"فوركس|forex", ("فوركس", "الهامش", "forex")),
    (r"تامين|insurance", ("تامين", "insurance")),
    (r"تقسيط|اقساط|تابي|تمارا|tabby|tamara|bnpl|instal", ("تقسيط", "اقساط", "تابي", "تمارا", "مرابح", "tabby", "tamara", "instal")),
    (r"اسهم|سهم|stock", ("اسهم", "سهم", "شركات المساهم", "stock")),
    (r"بنك|البنوك|المصارف|مصرف|فوا[يئ]د|bank|interest", ("بنك", "بنوك", "مصرف", "مصارف", "فوايد", "bank", "interest")),
]
_TRADING = (r"تداول|trading", ("تداول", "trading"))


def product_words(question: str) -> tuple[str, ...]:
    """Words a passage must contain to speak of the product the question names; empty when none is named."""
    text = normalize(question)
    words = tuple(w for pattern, ws in _PRODUCTS if re.search(pattern, text) for w in ws)
    if not words and re.search(_TRADING[0], text):
        words = _TRADING[1]
    return words


def speaks_of(passage_text: str, words: tuple[str, ...]) -> bool:
    """True when the passage text contains any of the words."""
    text = normalize(passage_text)
    return any(w in text for w in words)


_COMPANY = re.compile(r"\b(تابي|تمارا|tabby|tamara)\b", re.IGNORECASE)


def names_company(text: str) -> bool:
    """A sentence naming a company asked about: in a quoted-fatwa reply it would apply the fatwa to that company."""
    return bool(_COMPANY.search(normalize(text)))


_SUICIDE_TOPIC = re.compile(r"انتحار|انتحر|\bsuicid|خود ?[کك]ش[یي]")


def mentions_suicide(question: str) -> bool:
    """The topic of suicide, e.g. a question about its ruling: answered, with a short caring line."""
    return bool(_SUICIDE_TOPIC.search(normalize(question)))


def check(question: str) -> Gate:
    """Check a question against the safety patterns in a fixed order and return the first kind that matches."""
    text = normalize(question)
    for kind, patterns in ((CRISIS, _CRISIS), (OVERRIDE, _OVERRIDE), (JUDGING_PEOPLE, _JUDGING),
                           (PERSONAL_CASE, _PERSONAL), (OUT_OF_SCOPE, _OUT_OF_SCOPE)):
        for pattern in patterns:
            if pattern.search(text):
                return Gate(kind, pattern.pattern)
    return Gate(None)
