"""Question in, checked answer with source cards out.

Order: fixed rules -> retrieval -> sufficiency check -> generation ->
verifier -> response. The generator never sees a question that the rules
stopped, and never answers when retrieval found nothing sufficient.
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace

from . import attribution, bayyinat, bidaya, classify, keyboard, referrals, sections
from .asbab import AsbabIndex
from .corpus import Corpus, Passage
from .generate import Generator, is_example
from .messages import LANGS, STYLES, TEXT
from .normalize import normalize, tokenize
from .retrieve import Hit, Retriever, is_sufficient
from .sections import SectionIndex
from .verify import Rejected, verify

MODEL_CANDIDATES = int(os.environ.get("MUHAWIR_PASSAGES") or 20)  # passages offered to the model; fewer = faster on slow machines
MODEL_MIN_COVERAGE = 0.34  # loose filter: the model, not keyword overlap, decides

MAX_QUESTION_CHARS = 500
NEIGHBOUR_OF = 4      # fiqh passages whose neighbours are added
MAX_NEIGHBOURS = 4
MAX_UNDERSTOOD_QUERIES = 14  # search phrases kept after the understanding step has run twice
MAX_TOPIC_WORDS = 3   # a search phrase this short may name a chapter of the fiqh book
MIN_PASSAGE_WORDS = 4  # fewer words than this (e.g. a bare surah title) is not a passage to answer from
SIMPLER = {"extended": "youth", "youth": "kids", "kids": "kids", "newcomer": "newcomer"}  # for "I did not understand"

ANSWERED, ABSTAINED, REFERRED, DECLINED, INVALID, CHAT, UNAVAILABLE, TRANSLATED = (
    "answered", "abstained", "referred", "declined", "invalid", "chat", "unavailable", "translated")
_ARABIC = re.compile(r"[\u0600-\u06FF]")
# the message says the text is a verse («ترجم الآية…», «قوله تعالى», ﴿…﴾, "translate this ayah")
_VERSE_HINT = re.compile(r"[﴿﴾]|\bتعالى\b|\bالآي[ةه]\b|\bآي[ةه]\b|\bسور[ةه]\b|\bالقرآن\b|\b(?:verse|ayah|aya|surah|sura|quran|qur'an)\b",
                         re.IGNORECASE)
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


# why a sentence was rejected, told to the model in its own language when it is asked to write again
_REJECTED_BECAUSE = (
    ("copied a source", "نقلتَ نص المقطع بدل أن تشرحه بكلماتك"),
    ("quotation not found", "اقتبستَ نصًا لا يطابق المقطع"),
    ("not supported", "فيها ما ليس في المقطع المذكور (زيادة أو استنتاج أو تحريف)"),
    ("cites passages", "أسندتَها إلى مقطع غير موجود"),
    ("an example that", "كتبتَ مثالًا يشبّه أمرًا شرعيًا بشيء من الدنيا، فلا تكتب مثالًا هنا"),
    ("an everyday example", "كتبتَ مثالًا من عندك، ولا تُكتب أمثلة هنا"),
    ("ruling word", "ذكرتَ نوعًا من الحكم (وجوبًا أو جوازًا أو تحريمًا أو استحبابًا…) غير الذي في المقطع"),
    ("no citation", "بلا مقطع تستند إليه"),
    ("is not named", "فيها اسم لم يرد في المقطع"),
    ("starts with a conclusion", "بدأتَها باستنتاج من عندك («وبالتالي»، «إذن»)"),
    ("says almost nothing", "جملة ناقصة لا تقول شيئًا"),
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


def _copies_a_source(text: str, translations: tuple[str, ...] = ()) -> bool:
    """The sentence is mostly a long quotation (five words or more). A short quotation inside the model's own
    explanation is fine: the source card shows the text, and the sentence explains it. A quotation of a verse's
    approved English translation is what an English answer is asked to do, so it does not count."""
    def approved(quote: str) -> bool:
        inner = " ".join(quote.strip('"“”').split()).lower()
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


class Muhawir:
    def __init__(self, corpus: Corpus, generator: Generator, retriever=None) -> None:
        self.corpus = corpus
        self.retriever = retriever or Retriever(corpus)
        self.generator = generator
        self.asbab = AsbabIndex(corpus, self.retriever)
        self.sections = SectionIndex(corpus)

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
        if self.generator.name != "extractive":  # the model must explain, not paste the sources
            translations = tuple(" ".join(p.translation.split()).lower() for p in passages if p.translation)
            copied = [c for c in kept if _copies_a_source(c.text, translations)]
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

    @staticmethod
    def _understood(understand, question: str, turns: list[dict]) -> dict | None:
        """The understanding step, run twice at once. The model's recall of verses and hadith differs from one
        call to the next, so the second run only adds its search phrases to the first run's; everything else
        (the neutral question, its kind, the language) is the first run's, or the second's if the first failed."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = pool.map(lambda _: understand(question, turns), range(2))
        u = first or second
        if first and second:
            phrases = list(dict.fromkeys((first.get("queries") or []) + (second.get("queries") or [])))
            u = {**first, "queries": phrases[:MAX_UNDERSTOOD_QUERIES]}
        return u

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
            if p is not None:
                best[pid] = Hit(p, 0.0, 1.0)
        expand = getattr(self.generator, "expand", None)
        extra = queries if queries is not None else (expand(question) if expand else [])
        # the book's own name for a topic the user named otherwise («المضاربة» → «القراض»)
        said = set(tokenize(f"{original} {question}"))
        extra = extra + [f"ال{sections.SYNONYMS[t]}" for t in said if t in sections.SYNONYMS]
        queries = extra + [question]  # phrases in the sources' own wording first, the user's words last
        lists = [[h for h in self.retriever.search(query, k=MODEL_CANDIDATES) if h.coverage >= MODEL_MIN_COVERAGE
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
        # the user's words, and the standalone question the understanding step made of a follow-up («وما شروطها؟»)
        return [p for p in offered if not _confused(p, f"{original} {question}")], queries

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
        if verdict == "partly" and not rewritten and can_review:  # it answers only part of what was asked
            rewrite(_feedback(rejected, incomplete=True))
        if verdict == "no":  # sourced and checked, but about another matter than the question: not an answer
            return _Written(passages, allowed, corpus, [], rejected, off_topic=True)
        return _Written(passages, allowed, corpus, kept, rejected)

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
        the answer itself still comes from retrieved passages alone. A referred reply names who to ask."""
        res = self._respond(question, style, lang, history)
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
        turns = [{"role": t.get("role"), "text": str(t.get("text", ""))[:MAX_TURN_CHARS]}
                 for t in (history or []) if isinstance(t, dict) and t.get("role") in ("user", "assistant")]
        turns = turns[-MAX_HISTORY_TURNS:]
        if question and len(question) <= MAX_QUESTION_CHARS and classify.check(question).kind is None:
            missing = self.asbab.missing(question)  # needs no model: same answer in every style, at once
            if missing:
                return Response(ABSTAINED, TEXT[lang_ok]["no_reason"][missing["what"]].format(**missing),
                                synthetic=self.corpus.synthetic)
        understand = getattr(self.generator, "understand", None)
        understood, queries, previous, kind, feeling = "", None, "", "", ""
        if question and understand and len(question) <= MAX_QUESTION_CHARS:
            gate = classify.check(question)  # the user's own words are checked before any rewording
            if gate.kind not in (classify.JUDGING_PEOPLE, classify.OVERRIDE):
                u = self._understood(understand, question, turns)
                if u is None:  # understanding failed: do not spend another call on search phrases, answer directly
                    queries = []
                if u is not None:
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
                    if not u["question"]:  # no question in the message (e.g. only an insult): no judgement, an invitation
                        # right after an answer, it usually means the answer did not help: offer to explain again
                        after = any(t["role"] == "assistant" for t in turns)
                        reply = "no_question_after_answer" if after else "no_question"
                        return Response(CHAT, TEXT[lang_ok][reply], synthetic=self.corpus.synthetic)
                    queries = u["queries"]
                    kind = u.get("kind", "")
                    feeling = u.get("feeling", "")
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
                        kind=kind, feeling=feeling)
        if classify.mentions_suicide(question) and res.status == ANSWERED:
            res.message = TEXT[lang_ok]["care_note"]  # the topic is answered, with a caring line above it
        if feeling == "sad" and res.status == ABSTAINED:
            # a grieving person is never told "not found": words of comfort, and an invitation to talk
            res = Response(CHAT, TEXT[lang_ok]["comfort"], synthetic=self.corpus.synthetic)
        if understood and res.status not in (INVALID,):
            res.understood = understood
        return res

    def _ask(self, question: str, style: str, lang: str, original: str = "",
             queries: list[str] | None = None, previous: str = "", kind: str = "", feeling: str = "") -> Response:
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
            passages = [h.passage for h in hits if is_sufficient([h]) and not _confused(h.passage, f"{original} {question}")]
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
            return self._why(self._abstain(question, t, synthetic), f"{offered}; {reason}", raw)
        # the dropped sentences themselves are shown only in the debug reply, never written to the log
        dropped = (f"{len(rejected)} sentence(s) dropped: " + " ## ".join(
            f"{r.reason} <{r.claim.text[:160]}>" for r in rejected)[:1500]) if rejected else ""
        if rejected:
            log.warning("some claims dropped: %s", "; ".join(r.reason for r in rejected)[:500])

        answer = [c for c in kept if not c.school]
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
        if not any(c.passage_ids and not is_example(c.text) for c in answer):  # an answer must rest on the sources: explanation alone is not shown
            if personal:
                return Response(REFERRED, t["personal_case"], synthetic=synthetic)
            if contemporary:
                return Response(REFERRED, t["out_of_scope"], synthetic=synthetic)
            return self._why(self._abstain(question, t, synthetic), f"{offered}; no sourced sentence (only views or Muhawir's own explanation)")
        claims = [{"text": _strip_ids(c.text, allowed), "passage_ids": list(c.passage_ids),
                   **({"section": c.section} if c.section else {}), **({"label": c.label} if c.label else {}),
                   **({"example": True} if is_example(c.text) else {})}
                  for c in answer]
        named = _named_views([replace(c, text=attribution.without_meta(c.text, c.passage_ids, corpus, lang))
                              for c in kept if c.school])
        views = [{"school": c.school, "text": _strip_ids(c.text, allowed), "passage_ids": list(c.passage_ids)}
                 for c in named]
        cards = self._cards([pid for c in answer + named for pid in c.passage_ids], corpus,
                            lang)
        note = t["translation_pending"] if lang == "en" and self.generator.name == "extractive" else ""
        if kind == "ruling":  # said by the system, not written by the model: a notice cannot cite a passage
            note = t["ruling_note"]
        if contemporary:  # quoted fatwas, never Muhawir's own ruling on the product; the card names who to ask
            return Response(REFERRED, t["contemporary_info"], claims, cards, synthetic, note, views)
        if gate.kind == classify.PERSONAL_CASE:
            message = t["personal_case"] + "\n" + t["personal_case_info"]
            # the message already says this is no ruling on the case and to ask a qualified body:
            # the ruling notice would repeat it, so it is shown once only
            return Response(REFERRED, message, claims, cards, synthetic, "" if note == t["ruling_note"] else note, views)
        # «I did not understand»: a short human line first, said by the system, with no religious content
        res = Response(ANSWERED, t["reexplain_lead"] if previous else "", claims, cards, synthetic, note, views)
        if DEBUG and (dropped or written.retried):
            res.why = ("answered after a second search | " if written.retried else "") + dropped
        # kinds, conditions, pillars or steps are always shown as a list, whatever the model marked: a "how" question
        # or one that asks for the types or sections of something (the model does not always call it "how")
        listing = bool(getattr(self.generator, "last_as_list", False)) or kind == "how" or _asks_for_kinds(question, original)
        res.as_list = listing and len(claims) > 1
        res.follow_up = getattr(self.generator, "last_follow_up", "") or ""
        return res
