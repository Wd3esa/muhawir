"""Question in, checked answer with source cards out.

Order: fixed rules -> retrieval -> sufficiency check -> generation ->
verifier -> response. The generator never sees a question that the rules
stopped, and never answers when retrieval found nothing sufficient.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace

from . import attribution, bayyinat, bidaya, classify, keyboard, quran_count, referrals, sahihayn, sections
from .asbab import AsbabIndex
from .corpus import Corpus, Passage
from .generate import Generator, is_example
from .messages import LANGS, STYLES, TEXT
from .normalize import normalize, tokenize
from .retrieve import Hit, Retriever, is_sufficient
from .sections import SectionIndex
from .verify import Rejected, invented_definition, verify

MODEL_CANDIDATES = int(os.environ.get("MUHAWIR_PASSAGES") or 20)  # passages offered to the model; fewer = faster on slow machines
MODEL_MIN_COVERAGE = 0.34  # loose filter: the model, not keyword overlap, decides

MAX_QUESTION_CHARS = 500
NEIGHBOUR_OF = 4      # fiqh passages whose neighbours are added
MAX_NEIGHBOURS = 4
MAX_UNDERSTOOD_QUERIES = 14  # bound any phrases combined from a fallback second pass
MAX_TOPIC_WORDS = 3   # a search phrase this short may name a chapter of the fiqh book
MIN_PASSAGE_WORDS = 4  # fewer words than this (e.g. a bare surah title) is not a passage to answer from
MAX_QUOTED_HADITHS = 2  # hadiths fetched from their own book because a fatwa offered to the model quotes them
MAX_QUOTE_WORDS = 12  # the opening words of a quoted hadith are looked up (narrations differ further on)
QUOTE_CANDIDATES = 10  # matches read for each quoted hadith
QUOTE_MIN_COVERAGE = 0.8  # a hadith passage must hold nearly all the quoted words to be the one quoted
SIMPLER = {"extended": "youth", "youth": "kids", "kids": "kids", "newcomer": "newcomer"}  # for "I did not understand"

ANSWERED, ABSTAINED, REFERRED, DECLINED, INVALID, CHAT, UNAVAILABLE, TRANSLATED = (
    "answered", "abstained", "referred", "declined", "invalid", "chat", "unavailable", "translated")
# replies kept in memory for a repeated question (Muhawir.ask): answers, referrals, refusals and «not found» only;
# never «unavailable» or a debug reply. MUHAWIR_CACHE=0 turns it off.
CACHE_SIZE = int(os.environ.get("MUHAWIR_CACHE", "500") or 0)
CACHE_SECONDS = 24 * 3600
# the same replies also kept on disk (AnswerStore) so that a restart or an update does not lose them; a reply is used
# only while the code, the prompt and the sources are those it was written with (`version`)
STORE_SECONDS = int(os.environ.get("MUHAWIR_STORE_DAYS", "30") or 0) * 24 * 3600
STORE_SIZE = 5000
# counts since the server started, shown by /api/health (numbers only: never a question or an answer)
STATS = {"questions": 0, "counted": 0, "verse_shown": 0, "from_memory": 0, "from_disk": 0, "waited_for_same": 0, "written": 0, "written_seconds": 0.0}
_stats_lock = threading.Lock()


def _stat(name: str, add: float = 1) -> None:
    with _stats_lock:
        STATS[name] += add
CACHED = frozenset({"answered", "referred", "declined", "abstained", "translated"})
_ARABIC = re.compile(r"[\u0600-\u06FF]")
_CHOOSE_LANG = {TEXT[lang]["choose_lang"] for lang in TEXT}
_CHOICES = {normalize(TEXT[lang][k]): v for lang in TEXT for k, v in (("choose_ar", "ar"), ("choose_en", "en"))}
# the message says the text is a verse («ترجم الآية…», «قوله تعالى», ﴿…﴾, "translate this ayah")
_VERSE_HINT = re.compile(r"[﴿﴾]|\bتعالى\b|\bالآي[ةه]\b|\bآي[ةه]\b|\bسور[ةه]\b|\bالقرآن\b|\b(?:verse|ayah|aya|surah|sura|quran|qur'an)\b",
                         re.IGNORECASE)
# a quotation in the question: «…», ﴿…﴾, "…", or what follows «قوله تعالى:» / «قال الله:» up to the end of the clause
_QUOTED = re.compile(r"«([^»]{6,200})»|﴿([^﴾]{6,200})﴾|\"([^\"]{6,200})\"|(?:تعالى|قال الله)\s*:?\s*([^«﴿\"؟?.،]{6,200})")


# Muhawir's own suggestion, sent back when tapped: «هل تريد أن تعرف أمثلة على السنن؟» is an offer to the user,
# so the understanding step found no question in it and asked the user to rephrase (live site, 6 October 2026)
_OFFER_AR = re.compile(r"^\s*(?:هل\s+)?(?:تريد|تريدين|تود|تودّ|تودين|تحب|تحبين|ترغب|ترغبين)\s+(?:في\s+)?(?:أن\s+|ان\s+)?"
                       r"(?:تعرف|تعرفي|تتعرف\s+على|تتعرفي\s+على|نتعرف\s+على|أشرح\s+لك|اشرح\s+لك|أخبرك\s+عن|أخبرك\s+ب)\s*(.+?)\s*[؟?]?\s*$")
_ASKS_AR = re.compile(r"^(?:ما|ماذا|كيف|لماذا|متى|أين|من|هل|كم|أي)\b")
_OFFER_EN = re.compile(r"^\s*(?:do you want|would you like|shall i tell you|want)\s+(?:to\s+)?(?:know|learn|hear|explain)?\s*(?:more\s+)?(?:about\s+)?(.+?)\s*\??\s*$",
                       re.IGNORECASE)
_ASKS_EN = re.compile(r"^(?:what|how|why|when|where|who|which|is|are|does|do|can)\b", re.IGNORECASE)


def as_question(text: str) -> str:
    """A tapped suggestion turned from an offer into the user's own question: «هل تريد أن تعرف أمثلة على السنن؟»
    → «ما أمثلة على السنن؟», «هل تريد أن تعرف كيف تُخرج الزكاة؟» → «كيف تُخرج الزكاة؟». Any other text as it is."""
    m = _OFFER_AR.match(text)
    if m:
        rest = m.group(1).strip()
        return f"{rest}؟" if _ASKS_AR.match(rest) else f"ما {rest}؟"
    m = _OFFER_EN.match(text)
    if m and re.match(r"^\s*(?:do you want|would you like|shall i tell you)", text, re.IGNORECASE):
        rest = m.group(1).strip()
        return f"{rest[:1].upper()}{rest[1:]}?" if _ASKS_EN.match(rest) else f"Tell me about {rest}."
    return text


# Each style's limits, as its instructions state them (generate.STYLE_GUIDE): sentences in a prose answer, items in
# a list of steps or kinds (which may need a few more), and for children the longest sentence kept when shorter ones
# are enough. The model often writes past them (style check of 6 October 2026: 13 sentences for a child, sentences of
# 24 words), so they are applied here, after the checks, keeping the order the answer was written in.
STYLE_LIMITS = {"kids": (5, 7), "youth": (6, 10), "newcomer": (8, 10), "extended": (12, 16)}
KIDS_LONGEST = 18


def fit_style(answer: list, style: str, listing: bool) -> list:
    prose, items = STYLE_LIMITS.get(style, STYLE_LIMITS["youth"])
    limit = items if listing else prose
    if style == "kids" and len(answer) > limit:
        # too many for a child: the longest sentences go first, but only as many as needed, so a sentence that
        # carries a needed condition is not dropped from an answer that already fits (19 October 2026)
        long = sorted((i for i, c in enumerate(answer) if len(c.text.split()) > KIDS_LONGEST),
                      key=lambda i: -len(answer[i].text.split()))
        drop = set(long[:len(answer) - limit])
        answer = [c for i, c in enumerate(answer) if i not in drop]
    return answer[:limit]


def _same_word(a: str, b: str) -> bool:
    """Two forms of one word («الصابرون»/«الصابرين»): the same letters but for the last one or two."""
    short = min(len(a), len(b))
    return short >= 4 and abs(len(a) - len(b)) <= 1 and a[:short - 2] == b[:short - 2]


# a request written before the verse inside the same quotation marks: \u00AB\u0627\u0630\u0643\u0631 \u0627\u0644\u0622\u064A\u0629: \u0648\u0627\u0633\u062A\u0639\u064A\u0646\u0648\u0627 \u0628\u0627\u0644\u0635\u0628\u0631 ...\u00BB
_ASK_BEFORE_VERSE = re.compile(
    r"^\s*(?:(?:\u0645\u0646 \u0641\u0636\u0644\u0643|\u0644\u0648 \u0633\u0645\u062D\u062A|\u0631\u062C\u0627\u0621[\u064B\u0627]?)\s+)?(?:\u0627\u0630\u0643\u0631|\u0623\u0630\u0643\u0631|\u0627\u0643\u062A\u0628|\u0623\u0643\u062A\u0628|\u0623\u0643\u0645\u0644|\u0627\u0643\u0645\u0644|\u0635\u062D\u062D|\u0635\u062D\u0651\u062D|\u0641\u0633\u0631|\u0641\u0633\u0651\u0631|\u0627\u0634\u0631\u062D|\u0645\u0627 \u0645\u0639\u0646\u0649|\u0645\u0627 \u062A\u0641\u0633\u064A\u0631)"
    r"\s*(?:\u0644\u064A\s+)?(?:\u0646\u0635\s+)?(?:\u0627\u0644\u0622\u064A[\u0629\u0647]|\u0627\u0644\u0627\u064A[\u0629\u0647]|\u0622\u064A[\u0629\u0647]|\u0627\u064A[\u0629\u0647]|\u0642\u0648\u0644\u0647 \u062A\u0639\u0627\u0644\u0649|\u0642\u0648\u0644 \u0627\u0644\u0644\u0647(?: \u062A\u0639\u0627\u0644\u0649)?)?\s*[:\uFF1A]?\s*")


# «اذكر الآية: …»، «اكتب لي آية …»، «ما نص الآية …»، «أكمل الآية …»: the verse itself is asked for, not its meaning
_ASKS_VERSE_TEXT = re.compile(
    r"(?:اذكر|أذكر|اكتب|أكتب|أكمل|اكمل|هات|ما\s+نص|نص)\s*(?:لي\s+)?(?:نص\s+)?(?:الآي[ةه]|الاي[ةه]|آي[ةه]|اي[ةه])")


# a hadith quoted in a fatwa: «…» or ((…)), or the words after «صلى الله عليه وسلم:» up to the end of the sentence
_HADITH_QUOTE = re.compile(r"«([^»]+)»|\(\(([^)]+)\)\)|(?:صلى الله عليه وسلم|ﷺ|عليه الصلاة والسلام)\s*(?:أنه\s+)?(?:قال\s*)?:\s*([^.«(\n]+)")


def _hadith_quotes(text: str) -> list[str]:
    """The opening words of each quotation of four words or more in `text`, in order, without repeats."""
    out: list[str] = []
    for m in _HADITH_QUOTE.finditer(text):
        words = next(g for g in m.groups() if g is not None).split()
        if len(words) >= 4:
            q = " ".join(words[:MAX_QUOTE_WORDS])
            if q not in out:
                out.append(q)
    return out


def _quote_parts(quoted: str) -> list[str]:
    """The quotation as written, then the verse part of it if a request came first inside the same marks
    (\u00AB\u0627\u0630\u0643\u0631 \u0627\u0644\u0622\u064A\u0629: \u0648\u0627\u0633\u062A\u0639\u064A\u0646\u0648\u0627 ...\u00BB): after a colon, or after the asking words, then shorter runs of its words from
    either end (four words or more), so the verse is found however the question was wrapped."""
    parts = [quoted]
    if re.search(r"[:\uFF1A]", quoted):
        parts.append(re.split(r"[:\uFF1A]", quoted, maxsplit=1)[1])
    parts.append(_ASK_BEFORE_VERSE.sub("", quoted))
    words = quoted.split()
    for n in range(len(words) - 1, 3, -1):
        parts += [" ".join(words[-n:]), " ".join(words[:n])]
    seen, out = set(), []
    for p in (p.strip() for p in parts):
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


# words that ask about a hadith rather than quote it («أريد مصدر وصحة حديث …»), and words in nearly every hadith
_HADITH_ASK_WORDS = {"حديث", "احاديث", "مصدر", "صحه", "وصحه", "صحيح", "سند", "درجه", "تخريج", "اريد", "داير", "دايره",
                     "عايز", "عاوز", "ابي", "ابغي", "بدي", "نص", "معني", "شرح", "هل", "ما", "شو", "ايش", "وش", "شنو", "قال",
                     "الله", "رسول", "نبي", "صلي", "عليه", "وسلم", "يقول", "كان", "hadith"}

def _without_label(text: str, label: str) -> str:
    """«ربا الفضل: ربا الفضل أن تبيع…» shows the label twice: the sentence keeps its words after the label."""
    if not label:
        return text
    words, start = text.split(), normalize(label).split()
    first = [normalize(w) for w in words[:len(start)]]
    if first and first[0].startswith("و") and first[0][1:] == (start[0] if start else ""):
        first[0] = first[0][1:]  # «وربا النسيئة» under the label «ربا النسيئة»
    if len(words) > len(start) + 2 and first == start:
        rest = " ".join(words[len(start):]).lstrip(":،, ")
        return rest or text
    return text


_HARAKAT = re.compile(r"[\u064B-\u0652\u0670]")


def _vocalised(text: str) -> bool:
    """Written with full diacritics, as verses are quoted (a plain Arabic sentence has few or none)."""
    letters = len(re.findall(r"[\u0621-\u064A]", text))
    return letters >= 6 and len(_HARAKAT.findall(text)) >= letters * 0.5

# a quotation of five words or more inside «» or "" or ﴿﴾: pasted from a source, not explained
_COPIED = re.compile(r'«(?:[^»\s]+\s+){4,}[^»]*»|"(?:[^"\s]+\s+){4,}[^"]*"|“(?:[^”\s]+\s+){4,}[^”]*”|﴿(?:[^﴾\s]+\s+){4,}[^﴾]*﴾')
MAX_QUOTED_SHARE = 0.5  # a sentence made mostly of a quotation is pasted, not explained
# Everyday examples («مثلًا…») from contemporary life, at most one per answer, to illustrate a general idea in
# the cited passage (the project owner's decision, 5 October 2026, after the user survey: 8 of 15 prefer answers
# with everyday examples; docs/SURVEY.md). They were off before because examples the model invented about acts of
# worship were wrong or likened worship to worldly things, so every example is still checked: it may state no
# religious information (a separate model check), may not compare a religious matter to daily life, and is never
# used in a ruling, a personal case, a contemporary financial matter or a reply to grief. It is shown marked as
# Muhawir's own illustration. Set False to switch them off again.
ALLOW_EXAMPLES = True

# Terms that look alike but name different matters. A chapter or passage headed by the first is offered only when
# the user's own words name it: the model turning «القروض» (loans) into «القراض» (a profit-sharing partnership) in
# its search phrases once brought a whole answer about the wrong contract (5 October 2026). Normalized spelling.
CONFUSABLE = (
    {"heading": ("القراض",), "user": ("قراض", "مقارض", "مقارضه", "مضاربه", "مضارب", "مضاربات",
                                      "qirad", "mudaraba", "mudarabah", "muqarada")},
)


# an example that ends by naming what the case is («…، فهذا مثال على المضاربة», «…, this is riba»): that names a
# particular case with a religious term, which is a ruling on it, so the example is dropped
_EXAMPLE_LABEL = re.compile(r"(?:[،,؛:]\s*[فو]?(?:هذا|هذه|ذلك|تلك)|[,;:]\s*(?:so\s+|and\s+)?(?:this|that)\s+is)"
                            r"(?:\s+\S+){1,6}\s*[.!]?\s*$", re.IGNORECASE)


def _labels_a_case(text: str) -> bool:
    return bool(_EXAMPLE_LABEL.search(text))


# a view must be named after the jurist or school that holds it, as the source names them: «بعض العلماء» is no name
_UNNAMED_SCHOOL = re.compile(r"^\s*(?:بعض|بعضهم|آخرون|اخرون|فريق|طائفة|قوم|غيرهم|some|others)\b|^\s*العلماء\b"
                            # «القول الأول», «الرأي الثاني»: a number, not a name
                            r"|^\s*(?:ال)?(?:قول|رأي|راي|مذهب|view|opinion)\s+(?:ال)?(?:اول|أول|ثاني|ثالث|رابع|خامس|first|second|third|fourth)",
                            re.IGNORECASE)
# a sentence about the scholars' disagreement: kept only when the question is about a ruling or the disagreement
_DISAGREEMENT = re.compile(r"(?:ا|ي|ت)ختلف\s+(?:ال)?(?:علماء|فقهاء)|خلاف\s+بين\s+(?:ال)?(?:علماء|فقهاء)|\bscholars\s+(?:differ|disagree)",
                           re.IGNORECASE)
# a connective that only makes sense after an earlier sentence: never the first word shown
_LEADING_LINK = re.compile(r"^\s*(?:كما|وكذلك|كذلك|وأيضًا|وأيضا|أيضًا|أيضا|ثم|وكما|also,?|moreover,?)\s+", re.IGNORECASE)


def _named_views(views: list) -> list:
    """Views whose school is a real name; with fewer than two named ones left, none (a lone view is no comparison)."""
    named = [v for v in views if not _UNNAMED_SCHOOL.search(v.school)]
    # one name over different views (a mufti listing the views of others, «بعض أصحاب مالك» three times): whose
    # each view is cannot be told, so none of them is shown
    count = {}
    for v in named:
        count[normalize(v.school)] = count.get(normalize(v.school), 0) + 1
    named = [v for v in named if count[normalize(v.school)] == 1]
    return named if len(named) >= 2 else []


def _without_leading_link(text: str) -> str:
    rest = _LEADING_LINK.sub("", text, count=1)
    return rest[:1].upper() + rest[1:] if rest != text and rest[:1].isascii() else rest


def _confused(passage: Passage, user_words: str) -> bool:
    """The passage is headed by a look-alike term the user did not use (see CONFUSABLE)."""
    heading = normalize(f"{passage.location} {passage.keywords}")
    words = set(tokenize(user_words))  # whole words: «الإقراض» (lending) is not «القراض»
    return any(any(h in heading for h in c["heading"]) and not words & set(c["user"]) for c in CONFUSABLE)
_LATIN = re.compile(r"[A-Za-z]{2,}")
# a question for the types or sections of something («أنواع الزكاة», «اشرح الزكاة وأنواعها», «أقسام الطلاق»), in any case
# or with a joining letter or the article; the singular «نوع» and the verb «أقسم» are not it
_ASKS_FOR_KINDS = re.compile(r"(?<!\w)[وفبل]?(?:ال)?(?:انواع|اقسام)")
# fixed replies (greetings, offers to explain again): never taken as "the previous answer"
_CANNED = {v for t in TEXT.values() for v in t.values() if isinstance(v, str)}
# words that open a question («كم», «ما حكم», «هل», «how»): a message with one of them, or a question mark, is a question
# («ما» and «لم» are left out: «ما فهمت» is no question, and is handled as a request to explain again)
_ASK_WORDS = {"كم", "ماذا", "متى", "اين", "كيف", "لماذا", "هل", "ماهو", "ماهي", "حكم", "شنو", "ايش", "وش", "ليش",
              "امتى", "فين", "وين", "قداش", "معنى", "what", "how", "why", "when", "where"}


def _still_a_question(message: str, turns: list[dict]) -> bool:
    """The understanding step found no question, yet the message is one: a question asked again word for word
    (20 October 2026: a visitor repeated his question and was told «يبدو أن جوابي لم يكن واضحًا…»), or a message
    with a question mark or a question word. It is answered, not met with an offer to explain again."""
    said = normalize(message)
    if not said:
        return False
    if any(t["role"] == "user" and normalize(t["text"]) == said for t in turns):
        return True
    words = said.split()
    return "؟" in message or "?" in message or (len(words) >= 2 and bool(set(words) & _ASK_WORDS))
ALL_MODELS_FAILED = "every model call failed"
DEBUG = os.environ.get("MUHAWIR_DEBUG") == "1"  # adds the reason for not answering to each response
log = logging.getLogger("muhawir")
MAX_HISTORY_TURNS = 6
MAX_TURN_CHARS = 600


@dataclass
class Response:
    status: str
    message: str
    claims: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    synthetic: bool = False
    note: str = ""
    views: list[dict] = field(default_factory=list)  # scholars' views as named in the sources
    understood: str = ""  # the follow-up question as rewritten for search, when it differs
    why: str = ""  # with MUHAWIR_DEBUG=1: why there is no answer (never contains the question)
    as_list: bool = False  # the answer lists types, kinds, conditions or steps: shown as a list
    follow_up: str = ""  # a short question Muhawir suggests to continue the dialogue (a tap asks it)
    referral: dict = field(default_factory=dict)  # who to ask, when Muhawir refers the user on (see referrals.py)
    choices: list[str] = field(default_factory=list)  # buttons that each send their text as the next message
    by_code: bool = False  # counted or quoted by code with no model (a word count, a verse shown): nothing to detail

    def to_dict(self) -> dict:
        return asdict(self)


_NUMBERS = r"(?::\d+(?:\.\d+)?)"
# an id written in the text in another form: Latin («t4:41:30:9543:1») or its first letter turned into
# Arabic («ف:565» for f:565, «ق:18:51» for q:18:51). «ق» alone needs two numbers, so «ق:16» (a verse) stays.
_LOOSE_ID = re.compile(
    rf"\s*[\[(]?(?<![\w:])(?:[a-z]\d*|[فتبمأ]\d*|ق\d*(?={_NUMBERS}{_NUMBERS})){_NUMBERS}{{1,5}}(?![\w:])[\])]?")
# the number of a hadith or a page written in the text: «رقم 7296», «برقم 7296»
_NUMBER_OF = re.compile(r"\s*(?<!\w)ب?رقم\s*:?\s*[0-9٠-٩]+(?:\.[0-9٠-٩]+)?")
_EMPTY_BRACKETS = re.compile(r"\s*[\[(]\s*[\])]")


@dataclass
class _Written:
    """One written and checked answer, with the passages it was written from."""
    passages: list
    allowed: set
    corpus: object
    kept: list
    rejected: list
    retried: bool = False  # written after a second search with new search phrases
    off_topic: bool = False  # the checked answer was about another matter than the question, so it was dropped
    partial: bool = False  # the answer is sourced and checked but covers only part of what was asked (said in a notice)


# why a sentence was rejected, told to the model in its own language when it is asked to write again
_REJECTED_BECAUSE = (
    ("copied a source", "نقلتَ نص المقطع بدل أن تشرحه بكلماتك"),
    ("quotation not found", "اقتبستَ نصًا لا يطابق المقطع"),
    ("not supported", "فيها ما ليس في المقطع المذكور (زيادة أو استنتاج أو تحريف)"),
    ("cites passages", "أسندتَها إلى مقطع غير موجود"),
    ("an example that", "كتبتَ مثالًا يشبّه أمرًا شرعيًا بشيء من الدنيا، فلا تكتب مثالًا هنا"),
    ("an everyday example", "كتبتَ مثالًا من عندك، ولا تُكتب أمثلة هنا"),
    ("definition not in", "عرّفتَ الشيء بكلمات ليست في المقطع؛ اكتب التعريف بألفاظ المقطع نفسها، أو اذكر ما ورد عنه بلا تعريف"),
    ("names a hadith book", "نسبتَ الحديث إلى كتاب (البخاري أو مسلم) لم تستند إلى مقطع منه؛ انسبه إلى الكتاب الذي في المقطع"),
    ("ruling word", "ذكرتَ نوعًا من الحكم (وجوبًا أو جوازًا أو تحريمًا أو استحبابًا…) غير الذي في المقطع"),
    ("no citation", "بلا مقطع تستند إليه"),
    ("is not named", "فيها اسم لم يرد في المقطع"),
    ("starts with a conclusion", "بدأتَها باستنتاج من عندك («وبالتالي»، «إذن»)"),
    ("says almost nothing", "جملة ناقصة لا تقول شيئًا"),
    ("dropped_condition", "حذفتَ قيدًا أو شرطًا من المقطع فصار الحكم أعمّ مما فيه؛ اذكر القيد كما في المقطع"),
    ("distortion", "غيّرتَ معنى المقطع (من قال لمن، أو النفي والإثبات، أو نسبة القول)"),
)


def _referral_kind(res: Response, question: str) -> str:
    """Which referral card a reply carries, if any: the official fatwa body, or a mosque."""
    texts = {lang: TEXT[lang] for lang in TEXT}
    if any(res.message.startswith(t["crisis"]) or res.message == t["care_note"] for t in texts.values()):
        return ""  # the fixed caring reply already points to people and the emergency number; no card
    if any(res.message in (t["out_of_scope"], t["contemporary_info"]) for t in texts.values()):
        return "finance"
    if res.status == REFERRED or (res.status == ABSTAINED and any(res.message == t["abstain"] for t in texts.values())):
        return "fatwa"  # a personal case, or nothing in the sources: a scholar (not for «no reason of revelation»)
    if res.status == ANSWERED and referrals.wants_to_become_muslim(question):
        return "newcomer"
    if res.status == ANSWERED and any(res.note == t["ruling_note"] for t in texts.values()):
        return "fatwa"  # scholars' views on a ruling: for one's own case, a fatwa body
    return ""


def _feedback(rejected: list, incomplete: bool = False) -> str:
    """What to tell the model before the one rewrite: each rejected sentence with the reason, and what to do."""
    parts = []
    if rejected:
        lines = []
        for r in rejected:
            why = next((ar for en, ar in _REJECTED_BECAUSE if r.reason.startswith(en) or en in r.reason), "لا تتفق مع المقاطع")
            lines.append(f"- {r.claim.text}  ← {why}")
        parts.append("رُفضت الجمل التالية، وبجانب كل جملة سبب رفضها:\n" + "\n".join(lines))
    if incomplete:
        parts.append("والجواب السابق لا يغطي السؤال كله: اذكر كل ما في المقاطع المتعلقة بالسؤال (كل العناصر إن كان المطلوب قائمة).")
    parts.append("اكتب الجواب كله من جديد جوابًا متصلًا مفهومًا، بما قالته المقاطع نفسها فقط، دون تفصيل أو شرح من عندك "
                 "لما لم تذكره المقاطع (ولو بدا الجواب أقصر)، واشرح النص المقتبس بكلماتك بدل نقله.")
    return "\n".join(parts)[:2500]


# where the model starts a conclusion of its own after a sentence that rests on the passage
MIN_TRIMMED_WORDS = 4  # fewest words a sentence keeps when its conclusion is cut off
_INFERRING = ("يدل", "يعني", "يؤكد", "يبين", "يثبت", "يوضح", "يظهر", "يفيد", "يوحي", "يشير", "يبرهن")
# explicit marks of an inference («…، مما يؤكد أن…», «…، وهذا يدل على…», «…، وبالتالي…»): the project's rule is
# that Muhawir draws no conclusion of its own, so everything from such a mark on is removed from every sentence
_INFERENCE_STARTS = tuple(tuple(normalize(w) for w in seq) for seq in (
    [(lead, verb) for lead in ("ما", "مما", "وهذا", "فهذا", "وذلك", "فذلك", "هذا", "ذلك") for verb in _INFERRING]
    + [("وبالتالي",), ("فبالتالي",), ("وعليه",), ("ومن", "ثم"), ("إذن",), ("فإذن",), ("فدل",)]))
# a wider set, used to rescue a sentence the second reading rejected: its first part may still be sound
_CONCLUSION_STARTS = _INFERENCE_STARTS + tuple(tuple(normalize(w) for w in seq) for seq in (
    ("وهذا",), ("وذلك",), ("فهذا",), ("فذلك",), ("أي", "أن")))


# after a demonstrative («وهذا», «وذلك»), any present-tense verb begins the model's own inference («وهذا يحدد…»)
_DEMONSTRATIVES = frozenset(normalize(w) for w in ("وهذا", "فهذا", "وذلك", "فذلك", "هذا", "ذلك"))
# a sentence may not end on one of these: the part before a cut would be left unfinished
_DANGLING = frozenset(normalize(w) for w in (
    "لمن", "من", "الذي", "التي", "الذين", "أن", "إن", "أنه", "أنها", "في", "على", "إلى", "عن", "ل", "ب", "ك", "و", "ف",
    "ثم", "أو", "لكن", "بل", "إلا", "إذا", "لو", "كما", "حتى", "مع", "عند", "هو", "هي", "ما", "لا", "لم", "لن"))


def _starts_inference(plain: list[str], k: int, starts) -> bool:
    if any(tuple(plain[k:k + len(seq)]) == seq for seq in starts):
        return True
    nxt = plain[k + 1] if k + 1 < len(plain) else ""
    return plain[k] in _DEMONSTRATIVES and len(nxt) >= 4 and nxt[0] in "يت"


def _cut_at(claim, starts):
    """The sentence up to the first of `starts` (after at least MIN_TRIMMED_WORDS words), as a new claim with the
    same sources; None if there is no such point, the part before it would end inside a quotation, or it would
    end on a word that needs something after it."""
    raw = claim.text.split()
    plain = [normalize(w) for w in raw]
    for k in range(MIN_TRIMMED_WORDS, len(raw)):
        if _starts_inference(plain, k, starts):
            head = " ".join(raw[:k]).rstrip(" ،,؛:")
            if head.count("«") == head.count("»") and normalize(head.split()[-1]) not in _DANGLING:
                return replace(claim, text=head + ".")
    return None


def _resolve_ids(claim, allowed: set[str]):
    """An id the model wrote without its last part («t4:2:255:4947» for «t4:2:255:4947:1», «m:82» for
    «m:82.01») is the one offered id it is the beginning of, cut at a «:» or «.». An id that is not offered and
    begins several offered ids, or none, is dropped: it supports nothing. A sentence left with no id has no
    citation and is rejected by the verifier."""
    fixed = []
    for pid in claim.passage_ids:
        if pid not in allowed:
            matches = [a for a in allowed if a.startswith(pid) and a[len(pid):len(pid) + 1] in (":", ".")]
            if len(matches) != 1:
                continue
            pid = matches[0]
        fixed.append(pid)
    ids = tuple(dict.fromkeys(fixed))
    return claim if ids == claim.passage_ids else replace(claim, passage_ids=ids)


def _without_ids(claim, allowed: set[str]):
    """The claim with passage ids and source numbers taken out of its text, so the checks read only what it says."""
    text = _strip_ids(claim.text, allowed)
    return claim if text == claim.text else replace(claim, text=text)


def _without_inference(claim):
    """The claim without an explicit inference of the model's own at its end; the claim itself if it has none."""
    return _cut_at(claim, _INFERENCE_STARTS) or claim


def _without_conclusion(claim):
    """The sentence up to the point where it starts a conclusion of its own, wider than `_without_inference`
    («…، وهذا قول…», «…، أي أن…»); None if there is no such point or too little would be left."""
    return _cut_at(claim, _CONCLUSION_STARTS)


# a sentence that opens with a conclusion of its own («وبالتالي، …», «Thus, …»): `_cut_at` keeps the first words of a
# sentence, so a conclusion at the very start is not cut but rejected (5 October 2026 evaluation, Q01 and Q18)
_OPENING_CONCLUSION = tuple(seq for seq in _INFERENCE_STARTS if seq[0] not in (normalize("ما"), normalize("مما")))
_OPENING_CONCLUSION_EN = re.compile(r"^\s*(?:thus|therefore|hence|consequently)\b", re.IGNORECASE)


def _opens_with_conclusion(text: str) -> bool:
    if _OPENING_CONCLUSION_EN.match(text):
        return True
    plain = [normalize(w) for w in text.split()]
    return any(tuple(plain[:len(seq)]) == seq for seq in _OPENING_CONCLUSION)


# a sentence that only points at something it does not say («سبب الاختلاف في ذلك.»): fewer than MIN_CONTENT_WORDS
# words besides particles, ending on a pointer («في ذلك») that stands for the rest (5 October 2026 evaluation, Q22).
# «ويزداد ذلك في الصيف» points back at the sentence before it and says something of it: kept.
# A short sentence that says its thing («خمس صلوات.», «يغسل الوجه ثلاثًا.») has no pointer and is kept.
MIN_CONTENT_WORDS = 3
_POINTERS = frozenset(normalize(w) for w in ("ذلك", "هذا", "هذه", "تلك", "this", "that"))
_EMPTY_WORDS = _POINTERS | frozenset(normalize(w) for w in (
    "في", "من", "على", "إلى", "عن", "هو", "هي", "و", "ف", "ثم", "أو", "the", "a", "an", "of", "in", "is", "it",
    "and", "to"))


def _unfinished(text: str) -> bool:
    """«أما من قامت عليه الحجة…» with no «فـ» answer after it: the sentence stops half-way (D15, 6 October)."""
    plain = normalize(text).split()
    if not plain or plain[0] not in (normalize("أما"), normalize("وأما")):
        return False
    return not any(w.startswith("ف") and len(w) > 2 for w in plain[2:])


def _says_almost_nothing(text: str) -> bool:
    plain = normalize(text).split()
    words = [w for w in plain if w not in _EMPTY_WORDS]
    return (len(words) < MIN_CONTENT_WORDS and bool(plain) and plain[-1] in _POINTERS) or _unfinished(text)


# words that turn an everyday example into a comparison with a religious matter («هذا يشبه الزكاة»)
_COMPARES = re.compile(r"(يشبه|تشبه|يشابه|تشابه|مثلما|كما لو|تماما مثل)")


def _compares_to_daily_life(text: str) -> bool:
    """An example («مثلًا…») that likens a religious matter to something in daily life. The model cannot
    tell where an everyday example stops explaining a word and starts saying what an act is like."""
    plain = normalize(text)
    return plain.startswith("مثلا") and bool(_COMPARES.search(plain))


def _asks_for_kinds(*questions: str) -> bool:
    """The question asks for the types or sections of something, so its answer is a list of them."""
    return any(_ASKS_FOR_KINDS.search(normalize(q)) for q in questions if q)


_BOTH_BOOKS = re.compile(r"^(و?)(?:روى|رواه|أخرج)\s+(?:الإمام\s+)?(?:البخاري\s+و(?:الإمام\s+)?مسلم|مسلم\s+والبخاري)\s+")


def _agreed_upon(text: str, source_ids: set[str]) -> str:
    """A hadith cited from both al-Bukhari and Muslim is named as the scholars name it, «متفق عليه»:
    «روى البخاري ومسلم أن النبي ﷺ قال…» → «وفي الحديث المتفق عليه (رواه البخاري ومسلم) أن النبي ﷺ قال…».
    Only when the sentence cites a passage of each book; any other sentence is left as written."""
    if not {"sahih-bukhari", "sahih-muslim"} <= source_ids or "متفق عليه" in text:
        return text
    return _BOTH_BOOKS.sub(lambda m: f"{m.group(1)}في الحديث المتفق عليه (رواه البخاري ومسلم) ", text, count=1)


_WEAK_GRADE = re.compile(r"ضعيف|منكر|موضوع|شاذ|باطل|معلول")
_SAHIH_BOOKS = {"sahih-bukhari", "sahih-muslim"}


def _hadith_notice(text: str, passages: list[Passage]) -> str:
    """A hadith of the Sunan or al-Muwatta that the scholars named in its card grade as weak is not left to be read
    as established: the system says so, as the sentence the model wrote may not (21 October 2026). Each scholar's
    grading is one part of the card's grade («الألباني: ضعيف؛ زبير علي زئي: إسناده حسن»); when all are weak the
    notice says so, when they differ it says they differ. A sentence that already says it is left as written."""
    if re.search(r"ضعيف|ضعف|اختلف|منكر|موضوع", text):
        return text
    weak = mixed = False
    for p in passages:
        if p.kind != "hadith" or p.source_id in _SAHIH_BOOKS or not p.grade or p.grade == sahihayn.NO_GRADE:
            continue
        verdicts = [bool(_WEAK_GRADE.search(part)) for part in p.grade.split("؛")]
        if all(verdicts):
            weak = True
        elif any(verdicts):
            mixed = True
    if weak:
        return f"{text.rstrip()} (وهذا الحديث ضعيف عند المحدثين الذين نُقل حكمهم في بطاقة المصدر)"
    if mixed:
        return f"{text.rstrip()} (واختلف المحدثون في الحكم على هذا الحديث، وأحكامهم في بطاقة المصدر)"
    return text


_SAYS_PROPHET = re.compile(r"قال\s+(?:النبي|رسول\s+الله)|عن\s+النبي|أن\s+النبي\s+ﷺ?\s*قال|حديث")
_NAMES_VERDICT = re.compile(r"السخاوي|ضعيف|موضوع|لا أصل له|باطل|صحيح|حسن|اختلف")


def _saying_notice(text: str, passages: list[Passage]) -> str:
    """A saying from «المقاصد الحسنة» (kind «takhrij») is one people repeat as a hadith, of any grade down to invented:
    a sentence that gives it as the Prophet's words ﷺ with no verdict on it gets one from the system (21 October 2026),
    whatever the model wrote. A sentence that rests also on a hadith book is left to the checks on hadith."""
    kinds = {p.kind for p in passages}
    if kinds != {"takhrij"} or not _SAYS_PROPHET.search(text) or _NAMES_VERDICT.search(text):
        return text
    return f"{text.rstrip()} (ذكره السخاوي في «المقاصد الحسنة» في الأقوال المشتهرة، وحكمه عليه في بطاقة المصدر)"


def _copies_a_source(text: str, translations: tuple[str, ...] = (), hadiths: tuple[str, ...] = ()) -> bool:
    """The sentence is mostly a long quotation (five words or more). A short quotation inside the model's own
    explanation is fine: the source card shows the text, and the sentence explains it. A quotation of a verse's
    approved English translation is what an English answer is asked to do, so it does not count, and neither does
    the words of the Prophet ﷺ quoted as they are in a hadith passage («بُني الإسلام على خمس…»): a hadith is quoted,
    not retold in other words (21 October 2026). `hadiths` are the hadith passages, normalized."""
    def approved(quote: str) -> bool:
        inner = " ".join(quote.strip('"“”').split()).lower()
        if inner.startswith("«") and hadiths:
            said = normalize(inner.strip("«»"))
            return bool(said) and any(said in h for h in hadiths)
        return bool(inner) and any(inner in t for t in translations)
    quoted = sum(len(m.group(0).split()) for m in _COPIED.finditer(text) if not approved(m.group(0)))
    return quoted > 0 and quoted / max(1, len(text.split())) > MAX_QUOTED_SHARE


def _strip_ids(text: str, ids: set[str]) -> str:
    """Remove passage ids and source numbers («رقم 7296») a model wrote into the answer text; the source
    cards already show where each sentence comes from."""
    for pid in sorted(ids, key=len, reverse=True):
        text = re.sub(rf"\s*[\[(]?(?<![\w:]){re.escape(pid)}(?![\w:])[\])]?", "", text)
    text = _LOOSE_ID.sub("", text)
    text = _EMPTY_BRACKETS.sub("", _NUMBER_OF.sub("", text))
    if text.count("(") > text.count(")"):  # a reference cut off in the middle, e.g. «(تفسير.»
        text = re.sub(r"\s*\([^()]*$", "", text).rstrip(" ،,:") + "."
    return re.sub(r"\s+([.،,؛])", r"\1", text).strip()


class AnswerStore:
    """Checked replies on disk (SQLite), keyed by the normalised question, style and language.

    Only questions asked with no conversation before them are kept; no visitor's name, address or history.
    A reply written by an older version of the code, the prompt or the sources is never returned."""

    def __init__(self, path, version: str) -> None:
        import sqlite3
        self.version = version
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        # The answer cache is written while other requests read it. WAL keeps readers
        # independent of the writer; SQLite still serializes writes via _lock.
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, version TEXT, at REAL, reply TEXT)")
        self._db.execute("DELETE FROM replies WHERE version != ?", (version,))
        self._db.commit()

    @staticmethod
    def _key(key: tuple) -> str:
        return json.dumps(key, ensure_ascii=False)

    def get(self, key: tuple) -> Response | None:
        with self._lock:
            row = self._db.execute("SELECT at, reply FROM replies WHERE key = ? AND version = ?",
                                   (self._key(key), self.version)).fetchone()
        if not row or time.time() - row[0] > STORE_SECONDS:
            return None
        try:
            return Response(**json.loads(row[1]))
        except (TypeError, ValueError):
            return None

    def put(self, key: tuple, res: Response) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO replies VALUES (?, ?, ?, ?)",
                             (self._key(key), self.version, time.time(), json.dumps(res.to_dict(), ensure_ascii=False)))
            self._db.execute("DELETE FROM replies WHERE key IN (SELECT key FROM replies ORDER BY at DESC LIMIT -1 OFFSET ?)",
                             (STORE_SIZE,))
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM replies").fetchone()[0]

    def delete(self, key: tuple) -> None:
        with self._lock:
            self._db.execute("DELETE FROM replies WHERE key = ?", (self._key(key),))
            self._db.commit()


class Muhawir:
    def __init__(self, corpus: Corpus, generator: Generator, retriever=None, store: AnswerStore | None = None) -> None:
        self.corpus = corpus
        self.retriever = retriever or Retriever(corpus)
        self.generator = generator
        self.asbab = AsbabIndex(corpus, self.retriever)
        self.sections = SectionIndex(corpus)
        self._cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()
        self.store = store
        self._inflight: dict = {}  # question key -> lock: the same question arriving twice at once is written once
        self._quran_index: quran_count.QuranIndex | None = None  # built on the first counting question
        self._quran_index_lock = threading.Lock()

    def _count_reply(self, word: str, surah: str, root_asked: bool, lang: str) -> Response:
        """«كم مرة وردت كلمة الرشد في القرآن»: counted by code over every verse, with each place listed and cited.
        No model is asked: a model counts only the places it happened to retrieve (quran_count)."""
        with self._quran_index_lock:
            if self._quran_index is None:
                self._quran_index = quran_count.QuranIndex.from_corpus(self.corpus)
        _stat("counted")
        c = self._quran_index.count(word, surah, root_asked)
        lead, lines, places, note = quran_count.reply(c, lang)
        if not self._quran_index.verses:
            return Response(ABSTAINED, TEXT[lang]["abstain"], synthetic=self.corpus.synthetic)
        claims = [{"text": f"{p.location}: {p.snippet}", "passage_ids": [p.verse_id]} for p in places]
        res = Response(ANSWERED if places else ABSTAINED, " ".join([lead, *lines]), claims=claims,
                       sources=self._cards([p.verse_id for p in places], lang=lang),
                       synthetic=self.corpus.synthetic, note=note, as_list=True, by_code=True)
        return res

    def _closest_hadith(self, question: str, passages: list[Passage], corpus, lang: str) -> Response | None:
        """A hadith asked about in words no passage holds («حديث سيسخط قوم من أمتي… قردة وخنازير», 19 October 2026):
        when the model found no answer but a hadith among the passages shares at least three of the quote's words
        and two fifths of them, that hadith is shown as it is in its source, with a plain note that the wording
        differs and that its authenticity is for a specialist. No model writes anything here."""
        plain = normalize(question)
        if not re.search(r"\bحديث\b|\bhadith\b", plain):
            return None  # «أحاديث عن الصيام» asks for hadiths on a topic, not about one hadith's wording
        quoted = [next(g for g in groups if g) for groups in _QUOTED.findall(question)]
        if quoted:
            text = " ".join(quoted)
        else:  # the hadith's words as the user wrote them: what follows «حديث»
            text = re.split(r"\bحديث\b|\bhadith\b", plain, maxsplit=1)[-1]
        words = [w for w in tokenize(text) if w not in _HADITH_ASK_WORDS]
        if len(set(words)) < (3 if quoted else 4):
            return None
        bare = lambda w: w[1:] if len(w) > 3 and w[0] in "وف" else w  # noqa: E731
        asked = {bare(w) for w in words}
        best, best_shared = None, 0
        for p in passages:
            if p.kind != "hadith":
                continue
            shared = len(asked & {bare(w) for w in tokenize(p.text)})
            if shared > best_shared:
                best, best_shared = p, shared
        if best is None or best_shared < 3 or best_shared < 0.4 * len(asked):
            return None
        t = TEXT[lang]
        source = corpus.source_of(best)
        grade = t["closest_hadith_grade"].format(grade=best.grade) if best.grade else ""
        where = best.location if best.location.startswith(source.name) else f"«{source.name}» ({best.location})"
        message = t["closest_hadith"].format(where=where, grade=grade)
        return Response(ABSTAINED, message, sources=self._cards([best.id], corpus, lang), synthetic=self.corpus.synthetic)

    def _abstain(self, question: str, t: dict, synthetic: bool) -> Response:
        """General abstain, or a precise one when the reasons-of-revelation source has no entry."""
        missing = self.asbab.missing(question)
        if missing:
            return Response(ABSTAINED, t["no_reason"][missing["what"]].format(**missing), synthetic=synthetic)
        return Response(ABSTAINED, t["abstain"], synthetic=synthetic)

    @staticmethod
    def _why(res: Response, reason: str, reply_start: str = "") -> Response:
        log.warning("not answered: %s", reason[:500])  # the reply itself is never logged
        if DEBUG:
            res.why = reason[:500] + (f" | reply began: {reply_start}" if reply_start else "")
        return res

    def _tidy(self, draft: list, allowed: set[str]) -> list:
        """Before any check (model drafts only): ids the model wrote without their last part are resolved and
        unknown ones dropped, passage ids and source numbers are taken out of the sentence text, and an explicit
        inference of its own at the end of a sentence is cut off. A sentence with nothing left is dropped."""
        if self.generator.name == "extractive":
            return draft
        tidy = [_without_inference(_without_ids(_resolve_ids(c, allowed), allowed)) for c in draft]
        return [c for c in tidy if c.text]

    def _checked(self, draft, corpus, allowed: set[str], passages: list[Passage], question: str = ""):
        """The two checks: in code (ids retrieved, quotes verbatim, schools named), then the model's
        second reading against the cited passages and the question. None when the second reading could not run."""
        shape = [Rejected(c, "starts with a conclusion of its own") for c in draft if _opens_with_conclusion(c.text)]
        shape += [Rejected(c, "says almost nothing") for c in draft
                  if not _opens_with_conclusion(c.text) and _says_almost_nothing(c.text)]
        kept, rejected = verify([c for c in draft if c not in [r.claim for r in shape]], corpus, allowed)
        rejected = shape + rejected
        # a definition of what was asked must say what the cited passages say: words the code cannot match in any
        # form or synonym are put to the model, and the sentence goes only when it confirms they are not there
        invented = {}
        judge = getattr(self.generator, "judge_definition", None)
        for c in kept:
            texts = [corpus.passage(pid).text for pid in c.passage_ids]
            suspects = invented_definition(c.text, texts, question)
            absent = judge(c.text, suspects, texts) if suspects and judge else None
            if absent:
                invented[id(c)] = absent
        rejected += [Rejected(c, f"definition not in the source's words: {', '.join(invented[id(c)])}")
                     for c in kept if id(c) in invented]
        kept = [c for c in kept if id(c) not in invented]
        if self.generator.name != "extractive":  # the model must explain, not paste the sources
            translations = tuple(" ".join(p.translation.split()).lower() for p in passages if p.translation)
            hadiths = tuple(normalize(p.text) for p in passages if p.kind == "hadith")
            copied = [c for c in kept if _copies_a_source(c.text, translations, hadiths)]
            rejected += [Rejected(c, "copied a source sentence instead of explaining it") for c in copied]
            kept = [c for c in kept if c not in copied]
            if not ALLOW_EXAMPLES:
                examples = [c for c in kept if is_example(c.text)]
                rejected += [Rejected(c, "an everyday example: examples are switched off") for c in examples]
                kept = [c for c in kept if c not in examples]
            labelled = [c for c in kept if is_example(c.text) and _labels_a_case(c.text)]
            rejected += [Rejected(c, "an example that names the case with a religious term") for c in labelled]
            kept = [c for c in kept if c not in labelled]
            compared = [c for c in kept if _compares_to_daily_life(c.text)]
            rejected += [Rejected(c, "an example that compares a religious matter to daily life") for c in compared]
            kept = [c for c in kept if c not in compared]
        check = getattr(self.generator, "check_support", None)
        if kept and check:
            by_id = {p.id: p for p in passages}
            flags = check(kept, by_id, question)
            if flags is None:
                return None
            flags, kept = list(flags), list(kept)
            why = list(getattr(self.generator, "last_check_reasons", None) or [])
            # a sentence rejected only because of a conclusion added after a sound first part
            # («…، وهذا يدل على…») is read once more without that tail
            cuts = {i: cut for i, (c, ok) in enumerate(zip(kept, flags)) if not ok
                    for cut in [_without_conclusion(c)] if cut is not None}
            again = check(list(cuts.values()), by_id, question) if cuts else None
            for (i, cut), ok in zip(cuts.items(), again or []):
                if ok:
                    kept[i], flags[i] = cut, True
            rejected += [Rejected(c, "not supported by the cited passage" + (f" ({why[i]})" if i < len(why) and why[i] else ""))
                         for i, (c, ok) in enumerate(zip(kept, flags)) if not ok]
            kept = [c for c, ok in zip(kept, flags) if ok]
        return kept, rejected

    @staticmethod
    def _broken(kept: list, rejected: list, draft: list) -> bool:
        """True when the checks removed the first sentence or a third or more of the answer."""
        first = draft[0] if draft else None
        return not kept or (first is not None and first not in kept) or len(rejected) * 2 >= len(kept)

    def _neighbours(self, passages: list[Passage], best: int = NEIGHBOUR_OF, limit: int = MAX_NEIGHBOURS) -> list[Passage]:
        """The passage before and after each of the first fiqh passages, when in the same section."""
        out, have = [], {p.id for p in passages}
        fiqh = [p for p in passages if p.kind == "fiqh" and p.id.startswith("f:") and p.id[2:].isdigit()][:best]
        for p in fiqh:
            n = int(p.id[2:])
            for pid in (f"f:{n - 1}", f"f:{n + 1}"):
                q = self.corpus.passage(pid)
                if q is not None and pid not in have and q.keywords == p.keywords and len(out) < limit:
                    out.append(q)
                    have.add(pid)
        return out

    def _issue_opener(self, p: Passage) -> Passage | None:
        """The passage that opens the issue (it starts with «المسألة…») which fiqh passage `p` continues:
        the one just before it, or the one before that, in the same heading. None when `p` opens its own
        issue, is not from «بداية المجتهد», or no such passage is there."""
        if p.kind != "fiqh" or not p.id.startswith("f:") or not p.id[2:].isdigit() or bidaya._ISSUE.match(p.text):
            return None
        n = int(p.id[2:])
        for pid in (f"f:{n - 1}", f"f:{n - 2}"):
            q = self.corpus.passage(pid)
            if q is not None and q.keywords == p.keywords and bidaya._ISSUE.match(q.text):
                return q
        return None

    def _with_issue_openers(self, passages: list[Passage]) -> list[Passage]:
        """An issue in «بداية المجتهد» states the views in its first passage and gives the evidence in the next
        ones. When only an evidence passage was found, the passage that opens its issue comes first, so the
        model reads what the disagreement is before the evidence for it; one already in the list moves up."""
        out: list[Passage] = []
        have: set[str] = set()
        for p in passages:
            opener = self._issue_opener(p)
            for q in (opener, p):
                if q is not None and q.id not in have:
                    out.append(q)
                    have.add(q.id)
        return out

    def _quoted_hadith(self, quote: str) -> Passage | None:
        """The hadith in «الصحيحان» (or another hadith book of the corpus) whose text holds `quote`: the best
        hadith match that has nearly all the quote's words. None when no hadith passage matches that closely."""
        # by words only (the quoted words in a row first): no call to the meaning model for each quotation
        words = getattr(self.retriever, "sqlite_retriever", self.retriever)
        for h in words.search(quote, k=QUOTE_CANDIDATES):
            if h.passage.kind == "hadith" and h.coverage >= QUOTE_MIN_COVERAGE:
                return h.passage
        return None

    def _with_quoted_hadiths(self, passages: list[Passage]) -> list[Passage]:
        """A fatwa that quotes a hadith comes after that hadith from its own book: the hadith is the source, the
        fatwa a scholar's word about it. «اركان الاسلام» found Ibn Baz's fatwa first (it has those words) and not
        «بُني الإسلام على خمس» in al-Bukhari and Muslim, which does not (21 October 2026). A hadith already in
        the list moves up before the fatwa; one not there is looked up by the quoted words and added."""
        out: list[Passage] = []
        have: set[str] = set()
        added = 0
        for p in passages:
            if p.kind == "fatwa" and added < MAX_QUOTED_HADITHS:
                for quote in _hadith_quotes(p.text):
                    h = self._quoted_hadith(quote)
                    if h is not None and h.id not in have:
                        out.append(h)
                        have.add(h.id)
                        added += 1
                        break
            if p.id not in have:
                out.append(p)
                have.add(p.id)
        return out

    @staticmethod
    def _understood(understand, question: str, turns: list[dict]) -> dict | None:
        """Understand once on the ordinary path. Retry only when the model fails or
        supplies no search phrases for a real question. An explicit second pass
        preserves the older, higher-query-recall behaviour when requested.

        The first pass always decides the question, personal/crisis flags and
        language; a second pass only contributes search phrases.
        """
        first = understand(question, turns)
        second_pass = os.environ.get("MUHAWIR_UNDERSTAND_SECOND_PASS") == "1"
        missing_search = (first and first.get("question") and not first.get("queries")
                          and not first.get("translate") and not first.get("feeling")
                          and first.get("lang") != "other" and not first.get("personal"))
        if not second_pass and first and not missing_search:
            return first
        second = understand(question, turns)
        if first and second:
            phrases = list(dict.fromkeys((first.get("queries") or []) + (second.get("queries") or [])))
            return {**first, "queries": phrases[:MAX_UNDERSTOOD_QUERIES]}
        return first or second

    def _gather(self, question: str, original: str, queries: list[str] | None) -> tuple[list[Passage], list[str]]:
        """The passages offered to the model, and the search phrases used to find them."""
        best: dict[str, object] = {}
        # the opening passage of the matching chapter and section of «بداية المجتهد» first:
        # it usually holds the overview, the definition or the list of kinds. Only the question and short
        # topic phrases are matched to chapter names: a quoted verse that happens to contain the word
        # «الصلاة» is not a question about the chapter on prayer
        topics = [q for q in (queries or []) if len(q.split()) <= MAX_TOPIC_WORDS]
        for pid in self.sections.match([original or question] + topics):
            p = self.corpus.passage(pid)
            if p is not None and bayyinat.is_answer_part(p):
                best[pid] = Hit(p, 0.0, 1.0)
        expand = getattr(self.generator, "expand", None)
        extra = queries if queries is not None else (expand(question) if expand else [])
        # the book's own name for a topic the user named otherwise («المضاربة» → «القراض»)
        said = set(tokenize(f"{original} {question}"))
        extra = extra + [f"ال{sections.SYNONYMS[t]}" for t in said if t in sections.SYNONYMS]
        queries = extra + [question]  # phrases in the sources' own wording first, the user's words last
        lists = [[h for h in self.retriever.search(query, k=MODEL_CANDIDATES) if bayyinat.is_answer_part(h.passage)
                  and h.coverage >= MODEL_MIN_COVERAGE
                  and (h.passage.kind == "quran" or len(h.passage.text.split()) >= MIN_PASSAGE_WORDS)]
                 for query in queries]
        # take turns between the phrases, so one long phrase with high scores cannot fill every place
        for rank in range(MODEL_CANDIDATES):
            for hits in lists:
                if rank < len(hits) and len(best) < MODEL_CANDIDATES:
                    best.setdefault(hits[rank].passage.id, hits[rank])
        # one issue in «بداية المجتهد» often runs over two or three passages in a row (the views in one,
        # the evidence in the next): add the neighbours of the best fiqh matches from the same section
        for p in self._neighbours([h.passage for h in best.values()]):
            best.setdefault(p.id, Hit(p, 0.0, 1.0))
        # and whatever fiqh passage was found, the passage that opens its issue (the views) comes before it
        offered = self._with_issue_openers([h.passage for h in best.values()])
        # and a hadith a fatwa quotes comes from its own book, before the fatwa
        offered = self._with_quoted_hadiths(offered)
        # the user's words, and the standalone question the understanding step made of a follow-up («وما شروطها؟»)
        return [p for p in offered if bayyinat.is_answer_part(p)
                and not _confused(p, f"{original} {question}")], queries

    def _write(self, question: str, passages: list[Passage], style: str, lang: str, personal: bool,
               extra: dict) -> "_Written | Response":
        """Draft an answer from the passages, check it twice, and rewrite it once if the checks broke it.
        A Response (service unavailable) instead when the model or the second reading could not run."""
        t, synthetic = TEXT[lang], self.corpus.synthetic
        if lang == "en":  # an English answer quotes the approved translation of a verse, never its own
            passages = [replace(p, translation=self._translation(p)) if p.kind == "quran" else p for p in passages]
        allowed = {p.id for p in passages}
        corpus = self.corpus
        if hasattr(self.generator, "last_note"):
            self.generator.last_note = self.generator.last_raw = ""
            self.generator.last_as_list = False
            self.generator.last_follow_up = ""
        draft = self._tidy(self.generator.generate(question, passages, style, lang, personal=personal, **extra), allowed)
        if getattr(self.generator, "last_note", "") == ALL_MODELS_FAILED:
            # the model could not be reached: say so honestly instead of "nothing found in the sources"
            return self._why(Response(UNAVAILABLE, t["unavailable"], synthetic=synthetic), ALL_MODELS_FAILED)
        result = self._checked(draft, corpus, allowed, passages, question)
        if result is None:  # the check could not run: show nothing unchecked, and say why honestly
            return self._why(Response(UNAVAILABLE, t["unavailable"], synthetic=synthetic),
                             "support check could not run")
        kept, rejected = result
        can_review = bool(getattr(self.generator, "check_support", None))
        rewritten = False

        def rewrite(feedback: str) -> None:
            """One full rewrite from the feedback, checked again; kept only if it leaves more sentences."""
            nonlocal kept, rejected, rewritten
            rewritten = True
            redraft = self.generator.generate(question, passages, style, lang, personal=personal, feedback=feedback,
                                              **extra)
            second = self._checked(self._tidy(redraft, allowed), corpus, allowed, passages, question) if redraft else None
            if second is not None and len(second[0]) > len(kept):
                kept, rejected = second

        if rejected and self._broken(kept, rejected, draft) and can_review:
            # the checks removed the start or most of the answer, so what is left would not read as one
            # answer: ask once for a full rewrite that avoids the rejected sentences, then check it again
            rewrite(_feedback(rejected))
        judge = getattr(self.generator, "judge_relevance", None)
        verdict = judge(question, kept, {p.id: p for p in passages}) if judge and kept else None
        partial = verdict == "partly"
        if partial and not rewritten and can_review:  # it answers only part of what was asked
            rewrite(_feedback(rejected, incomplete=True))
            # the rewrite may now cover the question: it is judged again, and a notice is kept only if it still does not
            again = judge(question, kept, {p.id: p for p in passages}) if judge and kept else None
            partial = again != "yes"
        if verdict == "no":  # sourced and checked, but about another matter than the question: not an answer
            return _Written(passages, allowed, corpus, [], rejected, off_topic=True)
        return _Written(passages, allowed, corpus, kept, rejected, partial=partial)

    def _retry(self, question: str, original: str, tried: list[str], style: str, lang: str, extra: dict,
               first: _Written | None) -> _Written | Response | None:
        """Nothing usable came out of the first search (no passage, or none that answered): ask the model once
        for different search phrases, search again, and write from the new passages. The first outcome
        stands unless this gives an answer."""
        more = getattr(self.generator, "retry_queries", lambda *_: [])(question, tried)
        if not more:
            return first
        passages, _ = self._gather(question, original, more)
        if all(p.id in (first.allowed if first else ()) for p in passages):
            return first
        second = self._write(question, passages, style, lang, False, extra)
        if isinstance(second, Response):
            # the model failed on the new passages: with nothing usable before, say the service failed, not that
            # the sources have nothing (Codex audit, 6 October 2026)
            return second if second.status == UNAVAILABLE and (first is None or not first.kept) else first
        if not second.kept:
            return first
        second.retried = True
        return second

    def _cards(self, passage_ids: list[str], corpus=None, lang: str = "ar") -> list[dict]:
        corpus = corpus or self.corpus
        cards = []
        for pid in dict.fromkeys(passage_ids):
            p = corpus.passage(pid)
            s = corpus.source_of(p)
            card = {"passage_id": p.id, "quote": p.text, "location": p.location,
                    "kind": p.kind, "grade": p.grade, "topics": p.keywords, "source_name": s.name,
                    "source_about": s.about, "source_url": s.url}
            translation = self._translation(p, corpus) if p.kind == "quran" and lang == "en" else ""
            if translation:
                card["translation"] = translation
                card["translation_name"] = getattr(corpus, "translation_name", "")
            cards.append(card)
        return cards

    def _verse(self, text: str) -> Passage | None:
        """The verse that holds `text` word for word (diacritics aside), or None."""
        words = normalize(text).split()
        if len(words) < 2:
            return None
        target = f" {' '.join(words)} "
        for hit in self.retriever.search(text, k=5):
            if hit.passage.kind == "quran" and target in f" {' '.join(normalize(hit.passage.text).split())} ":
                return hit.passage
        return None

    def _misquoted(self, question: str) -> tuple[Passage, str] | None:
        """A verse quoted in the question with a word wrong («إن الله مع الصابرون»): the verse it was meant to be,
        and the quote as written. Found with no model: the quote's words, all but one equal to a run of the verse's
        words, and the different word a form of the same word (the same first letters). The answer is not built on
        the wrong wording; the note shows the verse as it is (EVALUATION.md, Q11; guide p. 6)."""
        for groups in _QUOTED.findall(question):
            for i, quoted in enumerate(_quote_parts(next(g for g in groups if g))):
                words = normalize(quoted).split()
                if len(words) < 3 or (i == 0 and self._verse(quoted) is not None):
                    continue
                for verse in self._near_verses(quoted, search=i == 0):
                    verse_words = normalize(verse.text).split()
                    for start in range(len(verse_words) - len(words) + 1):
                        window = verse_words[start:start + len(words)]
                        wrong = [(a, b) for a, b in zip(words, window) if a != b]
                        if len(wrong) == 1 and _same_word(*wrong[0]):
                            return verse, quoted.strip()
        return None

    def _asked_verse(self, question: str) -> Passage | None:
        """The verse a message asks to be shown («اذكر الآية: …», «أكمل الآية …»), found in the mushaf by its words,
        or by the misquoted-verse check when one word is in the wrong form. None when no verse is asked for."""
        m = _ASKS_VERSE_TEXT.search(question)
        if not m:
            return None
        quoted = [next(g for g in groups if g) for groups in _QUOTED.findall(question)]
        after = question[m.end():].strip(" :：«»\"﴿﴾؟?.")
        tried = 0
        for text in [*quoted, after]:
            for part in _quote_parts(text):
                if len(normalize(part).split()) < 3 or tried >= 8:
                    continue
                tried += 1
                verse = self._verse(part) or self._verse_loosely(part)
                if verse is not None:
                    return verse
        fix = self._misquoted(question)
        return fix[0] if fix else None

    def _verse_loosely(self, text: str) -> Passage | None:
        """As _verse, with «و» or «ف» the quote left off its first word («كلوا واشربوا» for «وكلوا واشربوا»)."""
        words = normalize(text).split()
        if len(words) < 3:
            return None
        candidates = [h.passage for h in self.retriever.search(text, k=5) if h.passage.kind == "quran"]
        con, terms = getattr(self.corpus, "con", None), tokenize(text)
        if con is not None and terms:  # every verse holding all the quote's words, however the search ranked them
            # the first word may carry a «و» or «ف» the quote left off: it is checked below, not searched
            match = " AND ".join('"' + t.replace('"', '""') + '"' for t in (terms[1:] if len(terms) >= 3 else terms))
            rows = con.execute("SELECT p.id FROM passages_fts JOIN passages p ON p.rid = passages_fts.rowid "
                               "WHERE passages_fts MATCH ? AND p.kind = 'quran' LIMIT 50", (match,)).fetchall()
            candidates += [p for p in (self.corpus.passage(r[0]) for r in rows) if p is not None]
        for passage in candidates:
            verse_words = normalize(passage.text).split()
            for i in range(len(verse_words) - len(words) + 1):
                window = verse_words[i:i + len(words)]
                if window[1:] == words[1:] and window[0] in (words[0], "و" + words[0], "ف" + words[0]):
                    return passage
        return None

    def _verse_reply(self, verse: Passage, lang: str) -> Response:
        """The verse as it is in the mushaf, with its source card, and an offer to explain it. No model is asked."""
        res = Response(ANSWERED, "", claims=[{"text": f"﴿{verse.text.replace('۞', '').strip()}﴾ ({verse.location})",
                                         "passage_ids": [verse.id]}],
                       sources=self._cards([verse.id], lang=lang), synthetic=self.corpus.synthetic, by_code=True)
        parts = verse.id.split(":")
        surah = verse.location.split("،")[0].strip()
        if len(parts) == 3 and surah:
            res.follow_up = (f"ما تفسير الآية {parts[2]} من {surah}؟" if lang != "en"
                             else f"What does verse {parts[2]} of {surah} mean?")
        return res

    def _near_verses(self, quoted: str, search: bool = True) -> list[Passage]:
        """Verses that may hold the quote with one word in another form: with the SQLite store, every verse
        holding the quote's other search words and a word beginning like the varied one; otherwise the search."""
        # The live retriever may be HybridRetriever, which wraps SQLite and has no .corpus.
        # Use this engine's corpus so the near-verse lookup still works with vectors enabled.
        con = getattr(self.corpus, "con", None)
        found = ([h.passage for h in self.retriever.search(quoted, k=5) if h.passage.kind == "quran"]
                 if search or con is None else [])
        terms = tokenize(quoted)
        if con is None or not terms:
            return found
        quote = lambda t: '"' + t.replace('"', '""') + '"'  # noqa: E731
        for i, term in enumerate(terms):
            if len(term) < 4:
                continue
            match = " AND ".join([quote(t) for j, t in enumerate(terms) if j != i] + [quote(term[:-2]) + "*"])
            rows = con.execute("SELECT p.id FROM passages_fts JOIN passages p ON p.rid = passages_fts.rowid "
                               "WHERE passages_fts MATCH ? AND p.kind = 'quran' LIMIT 20", (match,)).fetchall()
            found += [p for p in (self.corpus.passage(r[0]) for r in rows) if p is not None]
        return list({p.id: p for p in found}.values())

    def _verse_translation(self, question: str, text: str, target: str, lang: str) -> Response | None:
        """A verse is never translated by the model: the approved translation is quoted with its source card,
        or Muhawir says it found no such verse (Codex audit, 6 October 2026: a model translation of «قل هو الله
        أحد» came back with no source and no check). None for any other text, which the model translates.
        Three or fewer words that merely occur in a verse («الله أكبر» is in al-Ankabut 45) are not taken for
        one unless the message says it is a verse or the text is written with the Quran's diacritics."""
        said_verse = bool(_VERSE_HINT.search(question) or _VERSE_HINT.search(text)) or _vocalised(text)
        verse = self._verse(text) if target == "en" else None
        if verse is not None and (said_verse or len(normalize(text).split()) >= 4):
            translation = self._translation(verse)
            if translation:
                return Response(TRANSLATED, translation, sources=self._cards([verse.id], lang="en"),
                                synthetic=self.corpus.synthetic,
                                note=TEXT[lang]["verse_translation_label"].format(where=verse.location))
            return Response(ABSTAINED, TEXT[lang]["verse_not_found"], synthetic=self.corpus.synthetic)
        if said_verse:
            return Response(ABSTAINED, TEXT[lang]["verse_not_found"], synthetic=self.corpus.synthetic)
        return None

    def _translation(self, passage: Passage, corpus=None) -> str:
        lookup = getattr(corpus or self.corpus, "translation", None)
        return lookup(passage.id) if lookup else ""

    def ask(self, question: str, style: str = "youth", lang: str = "ar",
            history: list[dict] | None = None) -> Response:
        """Answer one message. `history` (earlier turns) is used only to understand a follow-up;
        the answer itself still comes from retrieved passages alone. A referred reply names who to ask.

        A question asked before with no conversation before it, in the same style and language, gets the same checked
        reply from memory: no model call, no cost, at once. Each question costs 10 to 20 model calls (about one US
        cent); visitors and reviewers repeat the suggested questions and the test list (6 October 2026: the prepaid
        credit ran out). Kept in memory and, when the server gives a store, on disk (AnswerStore) so that it outlives a
        restart, until the code, the prompt or the sources change."""
        _stat("questions")
        key = None if history else (normalize(as_question(question or "")), style, lang)
        if not (key and key[0] and CACHE_SIZE):
            return self._answer_new(question, style, lang, history, None)
        kept = self._kept(key)
        if kept:
            return kept
        with self._cache_lock:
            lock = self._inflight.setdefault(key, [threading.Lock(), 0])
            lock[1] += 1
        try:
            with lock[0]:  # a second visitor asking the same question now waits for the first reply instead of paying again
                kept = self._kept(key, waited=True)
                return kept or self._answer_new(question, style, lang, history, key)
        finally:
            with self._cache_lock:
                lock[1] -= 1
                if not lock[1]:
                    self._inflight.pop(key, None)

    def ask_again(self, question: str, style: str = "youth", lang: str = "ar") -> tuple[Response | None, Response]:
        """The nightly review (scripts/nightly.py): a first question kept as «not found» is answered once more from
        scratch, through every check as usual, and the new reply replaces the kept one. Returns (kept before, now).
        The model's search phrases and reading differ from one run to the next, so a miss is often not repeated."""
        key = (normalize(as_question(question or "")), style, lang)
        before = self._kept(key)
        with self._cache_lock:
            self._cache.pop(key, None)
        if self.store is not None:
            self.store.delete(key)
        return before, self.ask(question, style, lang)

    def _kept(self, key: tuple, waited: bool = False) -> Response | None:
        """A reply kept in memory or on disk for this question, style and language."""
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < CACHE_SECONDS:
                self._cache.move_to_end(key)
                _stat("waited_for_same" if waited else "from_memory")
                return copy.deepcopy(hit[1])
        kept = self.store.get(key) if self.store is not None and STORE_SECONDS else None
        if kept:
            with self._cache_lock:
                self._cache[key] = (time.time(), copy.deepcopy(kept))
                self._cache.move_to_end(key)
                while len(self._cache) > CACHE_SIZE:
                    self._cache.popitem(last=False)
            _stat("waited_for_same" if waited else "from_disk")
        return kept

    def _answer_new(self, question: str, style: str, lang: str, history: list[dict] | None, key) -> Response:
        started = time.time()
        res = self._ask_once(question, style, lang, history)
        _stat("written")
        _stat("written_seconds", time.time() - started)
        if key and res.status in CACHED and not res.why:
            with self._cache_lock:
                self._cache[key] = (time.time(), copy.deepcopy(res))
                self._cache.move_to_end(key)
                while len(self._cache) > CACHE_SIZE:
                    self._cache.popitem(last=False)
            if self.store is not None and STORE_SECONDS:
                try:
                    self.store.put(key, res)
                except Exception:  # the disk is full or locked: the reply is still given
                    logging.getLogger(__name__).warning("answer store: could not save a reply", exc_info=True)
        return res

    def _ask_once(self, question: str, style: str, lang: str, history: list[dict] | None) -> Response:
        question = as_question(question)
        res = self._respond(question, style, lang, history)
        if res.status in (ANSWERED, ABSTAINED, REFERRED):
            fix = self._misquoted(question)
            if fix:
                verse, quoted = fix
                line = TEXT[lang if lang in LANGS else "ar"]["misquoted_verse"].format(
                    where=verse.location, verse=verse.text, quoted=quoted)
                res.note = f"{line} {res.note}".strip()
        kind = _referral_kind(res, question)
        if kind:
            reply_text = " ".join([res.message, *(c.get("text", "") for c in res.claims)])
            reply_lang = "ar" if _ARABIC.search(reply_text) else ("en" if reply_text.strip() else lang)
            res.referral = referrals.card(kind, reply_lang)
        return res

    def _respond(self, question: str, style: str, lang: str, history: list[dict] | None) -> Response:
        question = (question or "").strip()
        lang_ok = lang if lang in LANGS else "ar"
        if lang_ok == "ar" and keyboard.latin_only(question):
            # Arabic typed with the keyboard on English («hgvfh» → «الربا»): offered back as «هل تقصد: «الربا»؟» with
            # a button that asks it, only when every word of it is in the sources (an English question stays English)
            guess = keyboard.to_arabic(question).strip()
            hits = self.retriever.search(guess, k=1)
            if hits and hits[0].coverage >= 0.99:
                res = Response(CHAT, TEXT[lang_ok]["did_you_mean"].format(guess=guess), synthetic=self.corpus.synthetic)
                res.follow_up = guess
                return res
        if question and classify.check(question[:MAX_QUESTION_CHARS * 2]).kind == classify.CRISIS:
            # thoughts of suicide or self-harm: a fixed caring reply that points to people and help now; no model
            return Response(REFERRED, TEXT[lang_ok]["crisis"], synthetic=self.corpus.synthetic)
        if question and len(question) <= MAX_QUESTION_CHARS and classify.is_small_talk(question):
            reply = "thanks" if classify.is_thanks(question) else "small_talk"
            return Response(CHAT, TEXT[lang_ok][reply], synthetic=self.corpus.synthetic)
        counting = quran_count.asked(question) if question and len(question) <= MAX_QUESTION_CHARS else None
        if counting:
            return self._count_reply(*counting, lang_ok)
        if question and len(question) <= MAX_QUESTION_CHARS:
            verse = self._asked_verse(question)
            if verse is not None:
                _stat("verse_shown")
                return self._verse_reply(verse, lang_ok)
        turns = [{"role": t.get("role"), "text": str(t.get("text", ""))[:MAX_TURN_CHARS]}
                 for t in (history or []) if isinstance(t, dict) and t.get("role") in ("user", "assistant")]
        turns = turns[-MAX_HISTORY_TURNS:]
        if question and len(question) <= MAX_QUESTION_CHARS and classify.check(question).kind is None:
            missing = self.asbab.missing(question)  # needs no model: same answer in every style, at once
            if missing:
                return Response(ABSTAINED, TEXT[lang_ok]["no_reason"][missing["what"]].format(**missing),
                                synthetic=self.corpus.synthetic)
        understand = getattr(self.generator, "understand", None)
        understood, queries, previous, kind, feeling, personal = "", None, "", "", "", False
        # «بالعربية» / «In English» tapped under «بأي لغة تريد الجواب؟»: the language is the one tapped, decided here
        # and not by the model, and the question is never sent back to that choice again
        chosen = _CHOICES.get(normalize(question)) if turns and turns[-1]["role"] == "assistant" \
            and turns[-1]["text"].strip() in _CHOOSE_LANG else None
        if question and understand and len(question) <= MAX_QUESTION_CHARS:
            gate = classify.check(question)  # the user's own words are checked before any rewording
            if gate.kind not in (classify.JUDGING_PEOPLE, classify.OVERRIDE):
                u = self._understood(understand, question, turns)
                if u is None:  # understanding failed: do not spend another call on search phrases, answer directly
                    queries = []
                if u is not None:
                    reply_lang = "ar" if _ARABIC.search(question) else "en" if _LATIN.search(question) else lang_ok
                    if u.get("feeling") == "crisis":
                        # thoughts of suicide or self-harm said in any words or dialect, which the word list above
                        # cannot all hold: the same fixed caring reply, no answer from the sources
                        return Response(REFERRED, TEXT[reply_lang]["crisis"], synthetic=self.corpus.synthetic)
                    if chosen:
                        u = {**u, "lang": chosen}
                        if not u.get("question"):  # the model found no question in «بالعربية»: the one asked before
                            asked = [t["text"] for t in turns if t["role"] == "user"]
                            u["question"] = asked[-1] if asked else ""
                    if u.get("lang") == "other" and not chosen:
                        # a language Muhawir does not answer in («بالإسبانية»): say which it does and let the user
                        # choose; the choice is sent as the next message and the question is taken from the conversation
                        t_ = TEXT[reply_lang]
                        res = Response(CHAT, t_["choose_lang"], synthetic=self.corpus.synthetic)
                        res.choices = [t_["choose_ar"], t_["choose_en"]]
                        return res
                    if not u["question"] and not u.get("translate") and u.get("feeling") == "sad":
                        # sorrow without a question: comfort first
                        return Response(CHAT, TEXT[lang_ok]["comfort"], synthetic=self.corpus.synthetic)
                    # a language request: translate it, nothing more (the model may leave `question` empty for it)
                    if u.get("translate") and gate.kind is None:
                        text = u["translate"]
                        target = u.get("lang") if u.get("lang") in LANGS else ("en" if _ARABIC.search(text) else "ar")
                        verse = self._verse_translation(question, text, target, lang_ok)
                        if verse is not None:
                            return verse
                        out = getattr(self.generator, "translate", lambda *_: None)(text, target)
                        if out is None:
                            return Response(UNAVAILABLE, TEXT[lang_ok]["unavailable"], synthetic=self.corpus.synthetic)
                        return Response(TRANSLATED, out, synthetic=self.corpus.synthetic,
                                        note=TEXT[lang_ok]["translation_label"])
                    if not u["question"] and _still_a_question(question, turns):
                        u = {**u, "question": question}
                    if not u["question"]:  # no question in the message (e.g. only an insult): no judgement, an invitation
                        # right after an answer, it usually means the answer did not help: offer to explain again
                        after = any(t["role"] == "assistant" for t in turns)
                        reply = "no_question_after_answer" if after else "no_question"
                        return Response(CHAT, TEXT[lang_ok][reply], synthetic=self.corpus.synthetic)
                    queries = u["queries"]
                    kind = u.get("kind", "")
                    feeling = u.get("feeling", "")
                    personal = bool(u.get("personal"))
                    if u.get("reexplain"):  # "I did not understand": the same question, explained again more simply
                        previous = next((t["text"] for t in reversed(turns)
                                         if t["role"] == "assistant" and t["text"] not in _CANNED), "")
                        if previous:
                            style = SIMPLER.get(style, style)
                    if u.get("lang") in LANGS:  # e.g. "the meaning of Tawhid in English": answer in English
                        lang = u["lang"]
                    elif _LATIN.search(question) and not _ARABIC.search(question):
                        lang = "en"  # an English question gets an English answer, whatever the page language
                    elif _ARABIC.search(question) and not _LATIN.search(question):
                        lang = "ar"
                    if normalize(u["question"]) != normalize(question):
                        understood = u["question"]
        res = self._ask(understood or question, style, lang, original=question, queries=queries, previous=previous,
                        kind=kind, feeling=feeling, personal=personal)
        if classify.mentions_suicide(question) and res.status == ANSWERED:
            res.message = TEXT[lang_ok]["care_note"]  # the topic is answered, with a caring line above it
        if feeling == "sad" and res.status == ABSTAINED:
            # a grieving person is never told "not found": words of comfort, and an invitation to talk
            res = Response(CHAT, TEXT[lang_ok]["comfort"], synthetic=self.corpus.synthetic)
        if understood and res.status not in (INVALID,):
            res.understood = understood
        return res

    def _ask(self, question: str, style: str, lang: str, original: str = "",
             queries: list[str] | None = None, previous: str = "", kind: str = "", feeling: str = "",
             personal: bool = False) -> Response:
        lang = lang if lang in LANGS else "ar"
        style = style if style in STYLES else "youth"
        t = TEXT[lang]
        synthetic = self.corpus.synthetic
        question = (question or "").strip()
        if not question:
            return Response(INVALID, t["empty"], synthetic=synthetic)
        if len(question) > MAX_QUESTION_CHARS:
            return Response(INVALID, t["too_long"], synthetic=synthetic)

        gate = classify.check(question)
        if original and original != question:  # a rewritten follow-up: the user's own words decide too
            own = classify.check(original)
            if own.kind:
                gate = own
        if personal and gate.kind is None:
            # the understanding step read a case of the user's own in words no fixed rule holds (any dialect):
            # general information and who to ask, as for a case the rules found
            gate = classify.Gate(classify.PERSONAL_CASE, "understood as a personal case")
        if gate.kind in (classify.JUDGING_PEOPLE, classify.OVERRIDE):
            return Response(DECLINED, t[gate.kind], synthetic=synthetic)
        # a contemporary financial matter: the published fatwas and the scholars' words are quoted, attributed,
        # with no ruling of Muhawir's own on the product asked about; nothing found keeps the referral alone
        contemporary = gate.kind == classify.OUT_OF_SCOPE
        if contemporary and self.generator.strict_retrieval:  # no model to quote with care: refer, as before
            return Response(REFERRED, t["out_of_scope"], synthetic=synthetic)

        personal = gate.kind == classify.PERSONAL_CASE
        if not personal:  # decided before the model, so every style gets the same answer
            missing = self.asbab.missing(question)
            if missing:
                return Response(ABSTAINED, t["no_reason"][missing["what"]].format(**missing),
                                synthetic=synthetic)
        if self.generator.strict_retrieval:
            hits = self.retriever.search(question)
            passages = [h.passage for h in hits if bayyinat.is_answer_part(h.passage)
                        and is_sufficient([h]) and not _confused(h.passage, f"{original} {question}")]
        else:
            passages, queries = self._gather(question, original, queries)
        if contemporary:
            kind = "contemporary"
        extra = {k: v for k, v in (("previous", previous), ("kind", kind), ("feeling", feeling)) if v}
        written = self._write(question, passages, style, lang, personal, extra) if passages else None
        if isinstance(written, Response):  # the model could not be reached, or the check could not run
            return written
        if (written is None or not written.kept) and not personal and not self.generator.strict_retrieval:
            written = self._retry(question, original, queries or [], style, lang, extra, written)
            if isinstance(written, Response):
                return written
        if written is None:
            if personal:
                return Response(REFERRED, t["personal_case"], synthetic=synthetic)
            if contemporary:
                return self._why(Response(REFERRED, t["out_of_scope"], synthetic=synthetic), "search found no passage")
            return self._why(self._abstain(question, t, synthetic), "search found no passage")
        kept, rejected, passages, allowed, corpus = (written.kept, written.rejected, written.passages,
                                                     written.allowed, written.corpus)
        if contemporary:  # only what a passage says about the product asked about is quoted; the rest is not an answer
            words = classify.product_words(f"{original} {question}")
            if words:
                def _names_it(c) -> bool:
                    found = [corpus.passage(pid) for pid in c.passage_ids]
                    return any(p is not None and classify.speaks_of(p.text, words) for p in found)
                kept = [c for c in kept if _names_it(c)]
        offered = f"{len(passages)} passages offered"
        if not kept:
            reason = "; ".join(r.reason for r in rejected) or getattr(self.generator, "last_note", "") or "no claims"
            if written.off_topic:
                reason = "the answer was about another matter than the question" + (f"; {reason}" if rejected else "")
            raw = getattr(self.generator, "last_raw", "")
            if personal:
                return self._why(Response(REFERRED, t["personal_case"], synthetic=synthetic), f"{offered}; {reason}")
            if contemporary:
                return self._why(Response(REFERRED, t["out_of_scope"], synthetic=synthetic), f"{offered}; {reason}")
            closest = self._closest_hadith(original or question, passages, corpus, lang)
            if closest is not None:  # a hadith asked about in other words: the nearest text, shown as it is
                return self._why(closest, f"{offered}; {reason}; closest hadith shown")
            return self._why(self._abstain(question, t, synthetic), f"{offered}; {reason}", raw)
        # the dropped sentences themselves are shown only in the debug reply, never written to the log
        dropped = (f"{len(rejected)} sentence(s) dropped: " + " ## ".join(
            f"{r.reason} <{r.claim.text[:160]}>" for r in rejected)[:1500]) if rejected else ""
        if rejected:
            log.warning("some claims dropped: %s", "; ".join(r.reason for r in rejected)[:500])

        answer = [c for c in kept if not c.school]
        answer = [replace(c, text=_agreed_upon(c.text, {corpus.passage(pid).source_id for pid in c.passage_ids}))
                  for c in answer]
        answer = [replace(c, text=_saying_notice(_hadith_notice(c.text, ps), ps)) for c in answer
                  for ps in [[corpus.passage(pid) for pid in c.passage_ids]]]
        if contemporary:  # a fatwa is never applied to the company asked about: a sentence naming it is dropped
            answer = [c for c in answer if not classify.names_company(c.text)]
        # «(الآية)» marks a verse a book cites without its text: a sentence carrying it would show no verse
        answer = [c for c in answer if bayyinat.VERSE not in c.text]
        if contemporary or personal or kind == "ruling" or feeling == "sad":  # no invented example near a ruling or grief
            answer = [c for c in answer if not is_example(c.text)]
        if kind not in ("ruling", "contemporary") and style != "extended" and not personal:
            # a question on a meaning, kinds or wisdom gets what the sources agree on, not the disagreement
            answer = [c for c in answer if not _DISAGREEMENT.search(c.text)]
        # who said it, from the source list: «الفتوى توضح أن…» names the mufti, «وهذا ما ورد في الفتوى» goes; a ruling
        # that names no one is credited to the sources it cites, or dropped if it cites only verses or hadith
        answer = [replace(c, text=attribution.without_meta(c.text, c.passage_ids, corpus, lang)) for c in answer]
        rules = contemporary or personal or kind == "ruling"
        answer = [replace(c, text=t) for c in answer
                  for t in [c.text if is_example(c.text) or not (rules or attribution.prefers(c.text))
                            else attribution.credited(c.text, c.passage_ids, corpus, lang)] if t]
        if answer and _LEADING_LINK.match(answer[0].text):  # «كما تشمل…» cannot open an answer
            answer[0] = replace(answer[0], text=_without_leading_link(answer[0].text))
        listing = bool(getattr(self.generator, "last_as_list", False)) or kind == "how" or _asks_for_kinds(question, original)
        answer = fit_style(answer, style, listing)
        if not any(c.passage_ids and not is_example(c.text) for c in answer):  # an answer must rest on the sources: explanation alone is not shown
            if personal:
                return Response(REFERRED, t["personal_case"], synthetic=synthetic)
            if contemporary:
                return Response(REFERRED, t["out_of_scope"], synthetic=synthetic)
            return self._why(self._abstain(question, t, synthetic), f"{offered}; no sourced sentence (only views or Muhawir's own explanation)")
        claims = [{"text": _without_label(_strip_ids(c.text, allowed), c.label), "passage_ids": list(c.passage_ids),
                   **({"section": c.section} if c.section else {}), **({"label": c.label} if c.label else {}),
                   **({"example": True} if is_example(c.text) else {})}
                  for c in answer]
        named = _named_views([replace(c, text=attribution.without_meta(c.text, c.passage_ids, corpus, lang))
                              for c in kept if c.school])
        if style == "kids":  # the style for children names no disagreement between scholars (generate.STYLE_GUIDE)
            named = []
        views = [{"school": c.school, "text": _strip_ids(c.text, allowed), "passage_ids": list(c.passage_ids)}
                 for c in named]
        cards = self._cards([pid for c in answer + named for pid in c.passage_ids], corpus,
                            lang)
        note = t["translation_pending"] if lang == "en" and self.generator.name == "extractive" else ""
        if kind == "ruling":  # said by the system, not written by the model: a notice cannot cite a passage
            # «the scholars' views» only when views are shown: an agreed ruling («are the five prayers obligatory?»)
            # has no views to present
            note = t["ruling_note"] if views else t["ruling_note_agreed"]
        if written.partial:  # said by the system: what the sources hold covers only part of the question (21 October 2026)
            note = f"{note}\n{t['partial_note']}" if note else t["partial_note"]
        if contemporary:  # quoted fatwas, never Muhawir's own ruling on the product; the card names who to ask
            return Response(REFERRED, t["contemporary_info"], claims, cards, synthetic, note, views)
        if gate.kind == classify.PERSONAL_CASE:
            message = t["personal_case"] + "\n" + t["personal_case_info"]
            # the message already says this is no ruling on the case and to ask a qualified body:
            # the ruling notice would repeat it, so it is shown once only
            return Response(REFERRED, message, claims, cards, synthetic, "" if note in (t["ruling_note"], t["ruling_note_agreed"]) else note, views)
        # «I did not understand»: a short human line first, said by the system, with no religious content
        res = Response(ANSWERED, t["reexplain_lead"] if previous else "", claims, cards, synthetic, note, views)
        if DEBUG and (dropped or written.retried):
            res.why = ("answered after a second search | " if written.retried else "") + dropped
        # kinds, conditions, pillars or steps are always shown as a list, whatever the model marked: a "how" question
        # or one that asks for the types or sections of something (the model does not always call it "how")
        res.as_list = listing and len(claims) > 1
        res.follow_up = getattr(self.generator, "last_follow_up", "") or ""
        return res
