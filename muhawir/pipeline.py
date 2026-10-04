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
from dataclasses import asdict, dataclass, field

from . import bidaya, classify
from .asbab import AsbabIndex
from .corpus import Corpus, Passage
from .generate import Generator, is_example
from .messages import LANGS, STYLES, TEXT
from .normalize import normalize
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
# a quotation of five words or more inside «» or "" or ﴿﴾: pasted from a source, not explained
_COPIED = re.compile(r'«(?:[^»\s]+\s+){4,}[^»]*»|"(?:[^"\s]+\s+){4,}[^"]*"|“(?:[^”\s]+\s+){4,}[^”]*”|﴿(?:[^﴾\s]+\s+){4,}[^﴾]*﴾')
MAX_QUOTED_SHARE = 0.5  # a sentence made mostly of a quotation is pasted, not explained
# Everyday examples («مثلًا إذا كان لديك…») are switched off. In every full test run the examples the model wrote
# about acts of worship were wrong, likened the act to something worldly, or restated a ruling, and no check can
# tell what an invented example implies about religion. Set True to allow them again (the checks for them stay).
ALLOW_EXAMPLES = False
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
)


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
                return type(claim)(head + ".", claim.passage_ids, claim.school)
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
    return claim if ids == claim.passage_ids else type(claim)(claim.text, ids, claim.school)


def _without_ids(claim, allowed: set[str]):
    """The claim with passage ids and source numbers taken out of its text, so the checks read only what it says."""
    text = _strip_ids(claim.text, allowed)
    return claim if text == claim.text else type(claim)(text, claim.passage_ids, claim.school)


def _without_inference(claim):
    """The claim without an explicit inference of the model's own at its end; the claim itself if it has none."""
    return _cut_at(claim, _INFERENCE_STARTS) or claim


def _without_conclusion(claim):
    """The sentence up to the point where it starts a conclusion of its own, wider than `_without_inference`
    («…، وهذا قول…», «…، أي أن…»); None if there is no such point or too little would be left."""
    return _cut_at(claim, _CONCLUSION_STARTS)


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


def _copies_a_source(text: str) -> bool:
    """The sentence is mostly a long quotation (five words or more). A short quotation inside the model's own
    explanation is fine: the source card shows the text, and the sentence explains it."""
    quoted = sum(len(m.group(0).split()) for m in _COPIED.finditer(text))
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
        kept, rejected = verify(draft, corpus, allowed)
        if self.generator.name != "extractive":  # the model must explain, not paste the sources
            copied = [c for c in kept if _copies_a_source(c.text)]
            rejected += [Rejected(c, "copied a source sentence instead of explaining it") for c in copied]
            kept = [c for c in kept if c not in copied]
            if not ALLOW_EXAMPLES:
                examples = [c for c in kept if is_example(c.text)]
                rejected += [Rejected(c, "an everyday example: examples are switched off") for c in examples]
                kept = [c for c in kept if c not in examples]
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
        return self._with_issue_openers([h.passage for h in best.values()]), queries

    def _write(self, question: str, passages: list[Passage], style: str, lang: str, personal: bool,
               extra: dict) -> "_Written | Response":
        """Draft an answer from the passages, check it twice, and rewrite it once if the checks broke it.
        A Response (service unavailable) instead when the model or the second reading could not run."""
        t, synthetic = TEXT[lang], self.corpus.synthetic
        allowed = {p.id for p in passages}
        corpus = self.corpus
        if hasattr(self.generator, "last_note"):
            self.generator.last_note = self.generator.last_raw = ""
            self.generator.last_as_list = False
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
               first: _Written | None) -> _Written | None:
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
        if isinstance(second, Response) or not second.kept:
            return first
        second.retried = True
        return second

    def _cards(self, passage_ids: list[str], corpus=None) -> list[dict]:
        corpus = corpus or self.corpus
        cards = []
        for pid in dict.fromkeys(passage_ids):
            p = corpus.passage(pid)
            s = corpus.source_of(p)
            cards.append({"passage_id": p.id, "quote": p.text, "location": p.location,
                          "kind": p.kind, "grade": p.grade, "topics": p.keywords, "source_name": s.name,
                          "source_about": s.about, "source_url": s.url})
        return cards

    def ask(self, question: str, style: str = "youth", lang: str = "ar",
            history: list[dict] | None = None) -> Response:
        """Answer one message. `history` (earlier turns) is used only to understand a follow-up;
        the answer itself still comes from retrieved passages alone."""
        question = (question or "").strip()
        lang_ok = lang if lang in LANGS else "ar"
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
        understood, queries, previous, kind = "", None, "", ""
        if question and understand and len(question) <= MAX_QUESTION_CHARS:
            gate = classify.check(question)  # the user's own words are checked before any rewording
            if gate.kind not in (classify.JUDGING_PEOPLE, classify.OVERRIDE):
                u = self._understood(understand, question, turns)
                if u is None:  # understanding failed: do not spend another call on search phrases, answer directly
                    queries = []
                if u is not None:
                    # a language request: translate it, nothing more (the model may leave `question` empty for it)
                    if u.get("translate") and gate.kind is None:
                        text = u["translate"]
                        target = u.get("lang") if u.get("lang") in LANGS else ("en" if _ARABIC.search(text) else "ar")
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
        res = self._ask(understood or question, style, lang, original=question, queries=queries, previous=previous, kind=kind)
        if understood and res.status not in (INVALID,):
            res.understood = understood
        return res

    def _ask(self, question: str, style: str, lang: str, original: str = "",
             queries: list[str] | None = None, previous: str = "", kind: str = "") -> Response:
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
        if gate.kind == classify.OUT_OF_SCOPE:
            return Response(REFERRED, t["out_of_scope"], synthetic=synthetic)

        personal = gate.kind == classify.PERSONAL_CASE
        if not personal:  # decided before the model, so every style gets the same answer
            missing = self.asbab.missing(question)
            if missing:
                return Response(ABSTAINED, t["no_reason"][missing["what"]].format(**missing),
                                synthetic=synthetic)
        if self.generator.strict_retrieval:
            hits = self.retriever.search(question)
            passages = [h.passage for h in hits if is_sufficient([h])]
        else:
            passages, queries = self._gather(question, original, queries)
        extra = {k: v for k, v in (("previous", previous), ("kind", kind)) if v}
        written = self._write(question, passages, style, lang, personal, extra) if passages else None
        if isinstance(written, Response):  # the model could not be reached, or the check could not run
            return written
        if (written is None or not written.kept) and not personal and not self.generator.strict_retrieval:
            written = self._retry(question, original, queries or [], style, lang, extra, written)
        if written is None:
            if personal:
                return Response(REFERRED, t["personal_case"], synthetic=synthetic)
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
            return self._why(self._abstain(question, t, synthetic), f"{offered}; {reason}", raw)
        # the dropped sentences themselves are shown only in the debug reply, never written to the log
        dropped = (f"{len(rejected)} sentence(s) dropped: " + " ## ".join(
            f"{r.reason} <{r.claim.text[:160]}>" for r in rejected)[:1500]) if rejected else ""
        if rejected:
            log.warning("some claims dropped: %s", "; ".join(r.reason for r in rejected)[:500])

        answer = [c for c in kept if not c.school]
        if not answer:  # views alone, without a sourced answer, are not shown
            if personal:
                return Response(REFERRED, t["personal_case"], synthetic=synthetic)
            return self._why(self._abstain(question, t, synthetic), f"{offered}; only scholars' views, no sourced answer")
        claims = [{"text": _strip_ids(c.text, allowed), "passage_ids": list(c.passage_ids)} for c in answer]
        views = [{"school": c.school, "text": _strip_ids(c.text, allowed), "passage_ids": list(c.passage_ids)}
                 for c in kept if c.school]
        cards = self._cards([pid for c in answer + [v for v in kept if v.school] for pid in c.passage_ids], corpus)
        note = t["translation_pending"] if lang == "en" and self.generator.name == "extractive" else ""
        if kind == "ruling":  # said by the system, not written by the model: a notice cannot cite a passage
            note = t["ruling_note"]
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
        return res
