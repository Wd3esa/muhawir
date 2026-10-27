"""Verifier: every claim must cite retrieved passages, and every quotation
must appear verbatim in one of the cited passages.

A generator (model or extractive) returns a draft as a list of claims. Claims
that fail are dropped; if nothing survives, the pipeline abstains.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import attribution, bayyinat, wordforms
from .corpus import Corpus
from .normalize import normalize, tokenize

# quotations of the sources: Arabic text inside «» or ﴿﴾ (English "..." marks a word, not a quotation)
_QUOTES = re.compile(r"«([^»]+)»|﴿([^﴾]+)﴾")
_ARABIC = re.compile(r"[؀-ۿ]")
# words in a view's label that name nobody («his companions», «and others», «the school of»)
_LABEL_FILLER = frozenset(t for w in ("أصحابه", "أصحابهم", "أصحابهما", "غيرهم", "مذهب", "الإمام", "أصحاب")
                          for t in tokenize(w))
_CASE = {"ابي": "ابو", "ابا": "ابو"}  # أبو / أبي / أبا are one name in three grammatical cases
# the words that say what kind of ruling it is (normalized spelling). A sentence may not give a ruling of a
# different kind than the passage it cites: «تجوز» (permitted) for a passage that says «تجب» (obligatory).
_RULING_WORDS = {
    "obligatory": frozenset("تجب يجب واجب وجوب فرض يلزم لازم اوجب فريضه".split()),
    "permitted": frozenset("تجوز يجوز جايز مباح يباح اباح اجاز جاز يجيز".split()),
    "forbidden": frozenset("تحرم يحرم حرام محرم حرم ممنوع يمنع منع".split()),
    "disliked": frozenset("يكره تكره مكروه كراهه".split()),
    "recommended": frozenset("يستحب تستحب مستحب مسنون يسن ندب مندوب".split()),
    # an act of worship or contract made void («تبطل الصلاة», «ينقض الوضوء», «فاسد»)
    "void": frozenset("يبطل تبطل باطل باطله يفسد تفسد فاسد فاسده ينقض تنقض ناقض نواقض".split()),
}
_NEGATORS = frozenset("لا ليس ليست لم لن غير ما".split())
# the author's division of his book (Ibn Rushd's «جملة» is a group of chapters), copied into an answer as if it
# were a division of the worship itself: «تنقسم الزكاة إلى خمس جمل», «الجملة الأولى», «في هذا الباب»
_BOOK_STRUCTURE = re.compile(
    r"(?:^| )(?:جملتين|(?:ثلاث|اربع|خمس|ست|سبع|ثماني|ثمان|تسع|عشر) جمل)(?: |$)"
    r"|(?:^| )الجمله (?:الاولي|الثانيه|الثالثه|الرابعه|الخامسه|السادسه|الاخيره)(?: |$)"
    r"|(?:^| )(?:هذا الباب|هذه الجمله|هذا الفصل)(?: |$)")


def book_structure(text: str) -> bool:
    """The sentence speaks of the book's own division (its «جمل», chapters or sections)."""
    return bool(_BOOK_STRUCTURE.search(normalize(text)))


@dataclass(frozen=True)
class Claim:
    """One sentence of an answer with the passages it cites and its layout fields."""
    text: str
    passage_ids: tuple[str, ...]
    school: str = ""  # set for a scholar's or school's view; must be named in the cited passage
    section: str = ""  # heading of the part of the answer this sentence belongs to (layout only)
    label: str = ""  # a short bold word that leads the sentence, e.g. «المقدار» (layout only)


@dataclass(frozen=True)
class Rejected:
    """A claim dropped by the code checks, with the reason."""
    claim: Claim
    reason: str


# a school may be named by its founder or its followers: «الحنفية» is named when the passage says «أبو حنيفة»
_SCHOOLS = [("حنيفه", "حنفيه", "احناف"), ("مالك", "مالكيه"), ("شافعي", "شافعيه"),
            ("احمد", "حنبل", "حنابله")]


def school_names(school: str) -> list[str]:
    """Normalized names that count as naming this school in a passage (the school as written, plus its
    founder or followers for the four schools)."""
    own = normalize(school)
    names = [own] if own else []
    for group in _SCHOOLS:
        if any(g in own for g in group):
            names += list(group)
    return names


def quotes_in(text: str) -> list[str]:
    """The Arabic quotations in a text."""
    return [q for m in _QUOTES.finditer(text) for q in [next(g for g in m.groups() if g)] if _ARABIC.search(q)]


def school_is_named(label: str, texts: list[str]) -> bool:
    """Every named person must be present; aliases apply only to one school."""
    joined = " ".join(texts)
    if is_named(label, joined):
        return True
    words = set(_names(label)) - {"شيخ", "امام", "مذهب", "الشيخ", "الامام"}
    groups = (
        ({"حنفي", "حنفيه", "احناف", "ابو", "حنيفه"}, _SCHOOLS[0]),
        ({"مالك", "مالكي", "مالكيه"}, _SCHOOLS[1]),
        ({"شافعي", "شافعيه"}, _SCHOOLS[2]),
        ({"احمد", "بن", "حنبل", "حنبلي", "حنبليه", "حنابله"}, _SCHOOLS[3]),
    )
    have = set().union(*(_forms(t) for t in _names(joined)))
    return any(words and words <= aliases and bool(words & (aliases - {"ابو", "بن"}))
               and bool(have & set(names)) for aliases, names in groups)


def _names(text: str) -> list[str]:
    return [_CASE.get(t, t) for t in tokenize(text)]


def _forms(token: str) -> set[str]:
    """A name as it may be written in a text: as it is, and without a «و» joined to it
    («وأبو حنيفة» in «مالك والشافعي وأبو حنيفة»)."""
    forms = {token}
    if token.startswith("و") and len(token) > 2:
        forms.add(_CASE.get(token[1:], token[1:]))
    return forms


def is_named(school: str, text: str) -> bool:
    """Every name in a view's label is written in the passage. The label may differ from the passage
    only in grammatical case («أبي حنيفة» for «أبو حنيفة», and the same for any other «أبو» name),
    in a joining «و» on either side, in words that name nobody («وأصحابه»), and in a bracketed note.
    A school or scholar the passage does not name is not accepted."""
    label = re.sub(r"[(\[][^)\]]*[)\]]", " ", school)
    have = set().union(*(_forms(t) for t in _names(text)))
    words = [w for w in _names(label) if w not in _LABEL_FILLER and w[1:] not in _LABEL_FILLER]
    return bool(words) and all(_forms(w) & have for w in words)


def _rulings(text: str, positive_only: bool = False) -> set[str]:
    """The kinds of ruling a text speaks of. With `positive_only`, a ruling word right after a negation
    («لا يجوز») is left out: «not permitted» may be a paraphrase of «forbidden»."""
    found: set[str] = set()
    tokens = normalize(text).split()
    for i, token in enumerate(tokens):
        if positive_only and any(t in _NEGATORS for t in tokens[max(0, i - 2):i]):
            continue
        for word in {token, token[1:] if token[:1] in "وفبل" and len(token) > 3 else token,
                     token[2:] if token[:2] == "ال" and len(token) > 4 else token}:
            found.update(kind for kind, words in _RULING_WORDS.items() if word in words)
    return found


# a negation that surely negates the next ruling word («لا يجب», «ليس بواجب», «لم يوجب»); «ما» and «غير» are left
# out here, as «ما يجب» (what is obligatory) and «غير واجب» are read differently
_STRICT_NEGATORS = frozenset("لا ليس ليست لم لن".split())


def _polarity(text: str, negators=_NEGATORS, reach: int = 2) -> set[tuple[str, bool]]:
    """(kind of ruling, negated) for each ruling word in a text: negated when a negator is among the `reach`
    words before it."""
    found: set[tuple[str, bool]] = set()
    tokens = normalize(text).split()
    for i, token in enumerate(tokens):
        negated = any(t in negators or (t[:1] in "وف" and t[1:] in negators) for t in tokens[max(0, i - reach):i])
        for word in {token, token[1:] if token[:1] in "وفبل" and len(token) > 3 else token,
                     token[2:] if token[:2] == "ال" and len(token) > 4 else token}:
            found.update((kind, negated) for kind, words in _RULING_WORDS.items() if word in words)
    return found


def flipped_rulings(claim_text: str, passage_texts: list[str]) -> set[str]:
    """Kinds of ruling the sentence turns around: it says «لا يجب» where the passage says only «يجب», or «يجوز»
    where the passage says only «لا يجوز». A passage that states both (two views) flags nothing."""
    if not passage_texts:
        return set()
    claim = _polarity(claim_text, _STRICT_NEGATORS)
    plain = _polarity(claim_text)  # with every negator: a word after «ما» or «غير» is not read as a plain statement
    # in the passage a negation further back counts too («لا أرى أنه يجب»): the passage is read as saying both
    # rather than flag a sentence that may be right
    passages = set().union(*[_polarity(t) | _polarity(t, reach=4) for t in passage_texts])
    flipped = set()
    for kind, negated in claim:
        said = (kind, not negated) in passages and (kind, negated) not in passages
        if negated and said:
            flipped.add(kind)
        elif not negated and (kind, False) in plain and said:
            flipped.add(kind)
    return flipped


def mismatched_rulings(claim_text: str, passage_texts: list[str]) -> set[str]:
    """Kinds of ruling the sentence states that the cited passages do not state, when they state another
    kind. A passage that uses no ruling word at all says nothing either way, so nothing is flagged."""
    in_passages = set().union(*[_rulings(t) for t in passage_texts]) if passage_texts else set()
    return (_rulings(claim_text, positive_only=True) - in_passages) if in_passages else set()


# A definition the model made up: «أركان الإسلام هي الأساسات التي بُني عليها الإسلام» (21 October 2026) passed the
# second reading, which accepts a loose paraphrase. Most of the defining words have to be in the cited passages, in
# any form or as a synonym (wordforms: «المال» for «مالك», «إعطاء» for «يُعطى», «قاعدة» for «أساس»). The words still
# unmatched are only suspects: the pipeline asks the model whether they mean what the passage says before dropping.
_COPULA = frozenset(normalize("هو هي هم هما").split())
_DEFINING = frozenset(normalize("يقصد يُقصد المقصود المراد معنى تعريف ويقصد والمقصود والمراد ومعنى وتعريف").split())
_DEFINITION_FILLER = frozenset(normalize("""عليه عليها عليهم فيه فيها فيهم منه منها لها له لهم به بها بهم إليه إليها
    وهو وهي وهم أي يعني أيضا التي الذي الذين اللاتي شيء أمر الأمور كلها جميع بعض""").split())
MAX_SUBJECT_WORDS = 4      # «أركان الإسلام هي…», «صلاة الاستخارة هي…»: the thing defined is short
MIN_DEFINING_WORDS = 2     # a definition of one word («الحول هو سنة») is a gloss, not checked here
MIN_DEFINITION_FOUND = 0.75  # share of the defining words that must be in the cited passages


def invented_definition(claim_text: str, passage_texts: list[str], question: str = "") -> list[str]:
    """The defining words of a definition sentence («X هو/هي …», «يُقصد بـX …», «معنى X …») that are not in the
    cited passages, when too many of them are missing; [] when the sentence is no definition or is in their words.
    With `question`, only a definition of what the question asks about is checked: a plain gloss of another word
    on the way («الحول هو سنة كاملة» in an answer on zakat) is the simpler wording the answers are meant to use."""
    clause = re.split(r"[،,.؛:؟?!]", claim_text, maxsplit=1)[0]
    words = normalize(clause).split()
    if not words:
        return []
    if words[0] in _DEFINING:
        defining = words[2:]  # «يقصد بالركن …»: the word after the verb is the thing defined
        subject = words[1:2]
    else:
        at = next((i for i, w in enumerate(words) if w in _COPULA), None)
        if at is None or not 1 <= at <= MAX_SUBJECT_WORDS:
            return []
        subject, defining = words[:at], words[at + 1:]
    subject_tokens = tokenize(" ".join(subject))
    named = wordforms.text_roots(subject_tokens)
    if question and not any(wordforms.in_text(q, named) for q in tokenize(question)):
        return []
    content = [t for t in tokenize(" ".join(defining)) if t not in _DEFINITION_FILLER and not wordforms.in_text(t, named)]
    if len(content) < MIN_DEFINING_WORDS:
        return []
    have = wordforms.text_roots([t for text in passage_texts for t in tokenize(text)])
    missing = [t for t in content if not wordforms.in_text(t, have)]
    return missing if len(content) - len(missing) < MIN_DEFINITION_FOUND * len(content) else []


# «روى البخاري…» / «وفي صحيح مسلم…»: the book a sentence names must be the book of a passage it cites
_COLLECTORS = {"البخاري": "sahih-bukhari", "مسلم": "sahih-muslim"}
_NAMES_COLLECTOR = re.compile(r"(?:روى|رواه|رواها|أخرج|أخرجه|في صحيح|صحيح)\s+(?:الإمام\s+)?(البخاري|مسلم)"
                              r"(?:\s+و(?:الإمام\s+)?(البخاري|مسلم))?")


def wrong_collector(claim_text: str, source_ids: set[str]) -> list[str]:
    """The hadith books a sentence names («روى البخاري ومسلم») that none of its cited passages comes from."""
    named = {n for m in _NAMES_COLLECTOR.finditer(claim_text) for n in m.groups() if n}
    return sorted(n for n in named if _COLLECTORS[n] not in source_ids)


def verify(claims: list[Claim], corpus: Corpus,
           allowed_ids: set[str]) -> tuple[list[Claim], list[Rejected]]:
    """Check each claim against the passages it cites, in code; returns the claims kept and those rejected with the reason."""
    kept: list[Claim] = []
    rejected: list[Rejected] = []
    for claim in claims:
        if not claim.passage_ids:
            # every sentence must explain something a passage says: Muhawir's wording is its own,
            # but there is no sentence without a source
            rejected.append(Rejected(claim, "no citation"))
            continue
        unknown = [pid for pid in claim.passage_ids if pid not in allowed_ids]
        if unknown:
            rejected.append(Rejected(claim, f"cites passages that were not retrieved: {unknown}"))
            continue
        if any(not bayyinat.is_answer_part(corpus.passage(pid)) for pid in claim.passage_ids):
            rejected.append(Rejected(claim, "cites a question without its answer"))
            continue
        # compared without diacritics or punctuation: «الكوثر» matches الْكَوْثَرَ; the card shows the exact text
        displayed = " ".join(part for part in (claim.section, claim.label, claim.school, claim.text) if part)
        cited = [normalize(corpus.passage(pid).text) for pid in claim.passage_ids]
        bad = [q for q in quotes_in(displayed)
               if not normalize(q) or not any(normalize(q) in text for text in cited)]
        if bad:
            rejected.append(Rejected(claim, f"quotation not found verbatim: {bad}"))
            continue
        # a fatwa does not name its mufti in its own text: the source list says who he is («الشيخ ابن باز»)
        if claim.school and not attribution.names_speaker(claim.school, claim.passage_ids, corpus) and not school_is_named(
                claim.school, [corpus.passage(pid).text for pid in claim.passage_ids]):
            rejected.append(Rejected(claim, f"'{claim.school}' is not named in the cited passage"))
            continue
        if book_structure(displayed):
            rejected.append(Rejected(claim, "speaks of the book's own division, not of the religion"))
            continue
        texts = [corpus.passage(pid).text for pid in claim.passage_ids]
        books = wrong_collector(claim.text, {corpus.passage(pid).source_id for pid in claim.passage_ids})
        if books:
            rejected.append(Rejected(claim, f"names a hadith book it does not cite: {', '.join(books)}"))
            continue
        wrong = mismatched_rulings(displayed, texts)
        if wrong:
            rejected.append(Rejected(claim, f"ruling word not in the cited passage: {', '.join(sorted(wrong))}"))
            continue
        flipped = flipped_rulings(displayed, texts)
        if flipped:
            rejected.append(Rejected(claim, f"ruling turned around (negated or affirmed against the passage): "
                                            f"{', '.join(sorted(flipped))}"))
            continue
        kept.append(claim)
    return kept, rejected
