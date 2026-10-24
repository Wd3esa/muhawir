"""How many times a word is in the Quran: counted by code over the mushaf text, never by a model.

A model asked to count gathers the places it happened to retrieve and misses some (17 October 2026: one platform
answered «الرشد» twice; it is three times). Here every verse in the sources is read: the word as written (diacritics
aside), then the same word with letters joined to it (و ف ب ل ك before it, a pronoun after it) counted apart, each
place listed with its verse. Roots are not counted: telling «رَشَدًا» from «الرَّاشِدُونَ» as one root needs a
morphology list, which the sources do not have yet; the reply says so.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .normalize import normalize

# «كم مرة وردت كلمة الرشد في القرآن»، «عدد مرات ذكر الصلاة في القرآن»، «كم مره تكررت لفظة رحمة في سورة البقرة»
_ASKS_COUNT = re.compile(
    r"كم\s*(?:مر[ةه]|مرات|عدد)|عدد\s*(?:مرات|ورود|تكرار|ذكر)|كم\s*(?:وردت?|ذكرت?|تكررت?|جاءت?|جات|جت|اجت|انذكرت?|اتذكرت?)|"
    r"\bhow\s+(?:many\s+times|often)\b", re.IGNORECASE)
_IN_QURAN = re.compile(r"القرآن|القران|المصحف|سور[ةه]\s|\bqur'?an\b|\bkoran\b", re.IGNORECASE)
_ROOT = re.compile(r"جذر|مشتقات|مادة|تصاريف|اشتقاق|\broot\b|\bderivatives?\b", re.IGNORECASE)
_QUOTED_WORD = re.compile(r"[«\"“'(﴿]([^»\"”')﴾]{2,40})[»\"”')﴾]")
_NAMED_WORD = re.compile(
    r"(?:كلم[ةه]|لفظ[ةه]?|اسم|word)\s+([؀-ۿݐ-ݿ ]{2,40}?)(?=\s+(?:في|ف|بال|من|داخل|in)\b|[؟?.،,]|$)",
    re.IGNORECASE)
# after the asking words with no «كلمة»: «كم مرة ذكرت الصلاة في القرآن»
_AFTER_ASK = re.compile(
    r"(?:وردت?|ذكرت?|تكررت?|جاءت?|جات|جت|اجت|انذكرت?|اتذكرت?|ذكر|ورود|تكرار)\s+([؀-ۿ ]{2,40}?)(?=\s+(?:في|ف)\s)")
_SURAH = re.compile(r"(?:في|ف)\s+سور[ةه]\s+([؀-ۿ ]{2,30}?)(?=\s*(?:[؟?.،,]|$|\s+(?:في|من)\s))")
_FILLER = frozenset({"كلمه", "لفظ", "لفظه", "اسم", "الكلمه", "اللفظ", "جذر", "الجذر", "ماده", "مشتقات", "الجلاله", "word", "the",
                     "root"})
_BEFORE = ("", "و", "ف", "ب", "ل", "ك", "وب", "فب", "ول", "فل", "وك", "فك")
_AFTER = ("", "ه", "ها", "هم", "هما", "هن", "ك", "كم", "كما", "كن", "ي", "نا", "ني")  # pronouns only: a dual or plural is another word
SHOWN = 30  # places listed with their verse; the count covers them all


@dataclass
class Place:
    verse_id: str
    location: str
    word: str  # the word as written in the verse
    snippet: str  # the words around it, for the list


@dataclass
class Count:
    word: str
    surah: str
    exact: list[Place] = field(default_factory=list)
    joined: list[Place] = field(default_factory=list)  # with و ف ب ل ك before or a pronoun after
    alif_ignored: bool = False
    root_asked: bool = False


def asked(question: str) -> tuple[str, str, bool] | None:
    """The word (or words) a counting question is about, the surah it is limited to (or ""), and whether a root or
    its derivatives were asked for. None when the message does not ask how many times something is in the Quran."""
    q = " ".join((question or "").split())
    # the whole question pasted inside quotation marks («كم مرة وردت كلمة الرشد في القرآن؟»): the marks are not the word
    q = re.sub(r'^[«"“\']\s*(.*?)\s*[»"”\']\s*([؟?.]?)\s*$', r"\1\2", q)
    # slips of typing in the word: a stretching line («الرشـــد»), a digit or Latin letter stuck to it («الرشد2»,
    # «الرشدd»), symbols around or inside it («*الرشد*», «ال-رشد», «الرشد!!»)
    q = q.replace("\u0640", "")
    q = re.sub(r"(?<=[\u0621-\u064A\u064B-\u0652])[A-Za-z0-9\u0660-\u0669\u06F0-\u06F9]+"
               r"|[A-Za-z0-9\u0660-\u0669\u06F0-\u06F9]+(?=[\u0621-\u064A])", "", q)
    q = re.sub(r"(?<=[\u0621-\u064A])[-_*~^|/\\+=#@&%$!]+(?=[\u0621-\u064A])", "", q)
    q = re.sub(r"[-_*~^|/\\+=#@&%$!(){}\[\]<>]+", " ", q)
    q = " ".join(q.split())
    q = re.sub(r"\bب(القرآن|القران|المصحف)\b", r"في \1", q)  # «… رمضان بالقرآن» = «في القرآن»
    if not (_ASKS_COUNT.search(q) and _IN_QURAN.search(q)):
        return None
    surah = ""
    m = _SURAH.search(q)
    if m:
        surah = m.group(1).strip()
    word = ""
    for pattern in (_QUOTED_WORD, _NAMED_WORD, _AFTER_ASK):
        m = pattern.search(q)
        if m and normalize(m.group(1)) and normalize(m.group(1)) not in ("القران", "القرآن") \
                and not _ASKS_COUNT.search(m.group(1)):  # a quote holding the question itself is not the word
            word = m.group(1)
            break
    word = re.sub(r"(?:^|(?<=\s))ال\s+(?=[\u0621-\u064A])", "ال", word)  # «ال رشد»: the article typed apart
    words = [w for w in word.split() if normalize(w) and normalize(w) not in _FILLER]
    if not words or not any(re.search(r"[؀-ۿ]", w) for w in words):
        return None
    return " ".join(words), surah, bool(_ROOT.search(q))


class QuranIndex:
    """Every verse of the mushaf as words: the plain form for matching, the written form for showing."""

    def __init__(self, verses: list[tuple[str, str, str]]):
        self.verses = []
        for vid, location, text in verses:
            written = [w for w in text.split() if normalize(w)]
            self.verses.append((vid, location, [normalize(w) for w in written], written))

    @classmethod
    def from_corpus(cls, corpus) -> "QuranIndex":
        con = getattr(corpus, "con", None)
        if con is not None:
            rows = con.execute("SELECT id, location, text FROM passages WHERE kind = 'quran' ORDER BY rid").fetchall()
        else:
            rows = [(p.id, p.location, p.text) for p in corpus.passages if p.kind == "quran"]
        rows.sort(key=lambda r: _verse_order(r[0]))
        return cls(rows)

    def count(self, words: str, surah: str = "", root_asked: bool = False) -> Count:
        target = normalize(words).split()
        result = Count(words, surah, root_asked=root_asked)
        verses = [v for v in self.verses if not surah or _in_surah(v[1], surah)]
        self._match(verses, target, result, lambda w: w, words.split()[0] if words.split() else "")
        if not result.exact and not result.joined and len(target) == 1:
            # «السموات» for «السماوات»: the verse writes a long alif the question left out
            for vid, location, plain, written in verses:
                for i, token in enumerate(plain):
                    if _alif_added(token, target[0]):
                        left, right = max(0, i - 4), min(len(written), i + 5)
                        snippet = ("… " if left else "") + " ".join(written[left:right]) + (" …" if right < len(written) else "")
                        result.exact.append(Place(vid, location, written[i], snippet))
            result.alif_ignored = bool(result.exact)
        return result

    @staticmethod
    def _match(verses, target, result: Count, form, target_written: str = "") -> None:
        n = len(target)
        target = [form(w) for w in target]
        for vid, location, plain, written in verses:
            plain = [form(w) for w in plain]
            for i in range(len(plain) - n + 1):
                window = plain[i:i + n]
                exact = window == target
                joined = not exact and (_joined(window[0], target[0]) if n == 1 else (
                    window[1:-1] == target[1:-1]
                    and (window[0] == target[0] or _joined(window[0], target[0], after=False))
                    and (window[-1] == target[-1] or _joined(window[-1], target[-1], before=False))))
                if joined and n == 1 and _taa_marbuta(written[i]) and not _taa_marbuta(target_written):
                    # «الجنة» is not «الجن» with a pronoun: ة and ه are one letter once normalised, not in the mushaf
                    joined = False
                if exact and n == 1 and _taa_marbuta(target_written) and not _taa_marbuta(written[i]):
                    # «رحمة» written with ة is not «رَحِمَه» (he had mercy on him), nor «نعمة» «نِعَمَه». Typed with ه
                    # («رحمه», a common way of typing ة) both are counted and the reply lists each form apart
                    exact = False
                if not (exact or joined):
                    continue
                left, right = max(0, i - 4), min(len(written), i + n + 4)
                snippet = ("… " if left else "") + " ".join(written[left:right]) + (" …" if right < len(written) else "")
                (result.exact if exact else result.joined).append(
                    Place(vid, location, " ".join(written[i:i + n]), snippet))


def _taa_marbuta(word: str) -> bool:
    """The word as written ends with ة (diacritics aside)."""
    return re.sub(r"[\u064B-\u0652\u0670\u0640]", "", word or "").endswith("ة")


def _joined(token: str, word: str, before: bool = True, after: bool = True) -> bool:
    """`token` is `word` with letters joined to it: a conjunction or preposition before, a pronoun or ending after.
    «ال» is never added or removed: «رشد» and «الرشد» are counted as different words, as asked."""
    if token == word:
        return False
    for b in (_BEFORE if before else ("",)):
        if not token.startswith(b):
            continue
        rest = token[len(b):]
        if b in ("ل", "ول", "فل") and word.startswith("ال") and rest == "ل" + word[2:]:  # «للرشد» = ل + الرشد
            return True
        for a in (_AFTER if after else ("",)):
            if (b or a) and rest == word + a and len(word) >= 3:
                if a and word.endswith("ه"):  # «رحمه» + «ه»: the ة of the word is written ت before a pronoun
                    continue
                return True
            if a and word.endswith("ه") and rest == word[:-1] + "ت" + a and len(word) >= 3:  # «رحمته»
                return True
    return False


def _alif_added(token: str, word: str) -> bool:
    """`token` is `word` with one or more «ا» added inside it («السماوات» for «السموات»), and nothing else."""
    if len(token) <= len(word) or len(token) - len(word) > 2:
        return False
    i = 0
    for j, ch in enumerate(token):
        if i < len(word) and ch == word[i]:
            i += 1
        elif ch != "ا" or j == 0 or j == len(token) - 1:  # inside the word only: «أقطار» is not «قطار»
            return False
    return i == len(word)


def _in_surah(location: str, surah: str) -> bool:
    name = normalize(location.split("،")[0]).removeprefix("سوره ").strip()
    asked_name = normalize(surah).removeprefix("سوره ").strip()
    return name == asked_name or name == "ال" + asked_name or "ال" + name == asked_name


def _verse_order(vid: str) -> tuple:
    parts = vid.split(":")
    try:
        return (0, int(parts[1]), int(parts[2]))
    except (IndexError, ValueError):
        return (1, 0, 0)


_FINAL_MARKS = re.compile(r"[\u064B-\u0652\u0670\u06D6-\u06ED]+$")


def by_reading(places: list[Place]) -> list[tuple[str, int]]:
    """The places grouped by how the word is voweled, the ending aside: «الْجَنَّة» (the Garden) apart from
    «الْجِنَّة» (the jinn), «رَشَدًا» apart from «رُشْدًا»; most frequent first."""
    return Counter(_FINAL_MARKS.sub("", p.word) for p in places).most_common()


def times_ar(n: int) -> str:
    if n == 1:
        return "مرة واحدة"
    if n == 2:
        return "مرتين"
    return f"{n} مرات" if 3 <= n <= 10 else f"{n} مرة"


def verses_ar(n: int) -> str:
    if n == 1:
        return "آية واحدة"
    if n == 2:
        return "آيتين"
    return f"{n} آيات" if 3 <= n <= 10 else f"{n} آية"


def times_en(n: int) -> str:
    return "once" if n == 1 else ("twice" if n == 2 else f"{n} times")


def reply(c: Count, lang: str = "ar") -> tuple[str, list[str], list[Place], str]:
    """The lead line, the lines that follow it (one per reading or kind), the places to list (at most SHOWN, exact
    ones first), and a note on how it was counted. A word not found gives no places."""
    ar = lang != "en"
    where = (f"في سورة {c.surah}" if ar else f"in Surah {c.surah}") if c.surah else ("في القرآن الكريم" if ar else "in the Quran")
    total = len(c.exact)
    verses = len({p.verse_id for p in c.exact})
    lines: list[str] = []
    if total:
        lead = (f"وردت «{c.word}» {where} {times_ar(total)}، في {verses_ar(verses)}:"
                if ar else f"«{c.word}» occurs {where} {times_en(total)}, in {verses} verse{'s' if verses != 1 else ''}:")
        readings = by_reading(c.exact)
        if len(readings) > 1:
            lines.append(("بأكثر من ضبط، وقد تختلف الكلمة باختلافه: " if ar else "With more than one voweling, which may be different words: ")
                         + "، ".join(f"{w} ({n})" for w, n in readings))
    else:
        lead = (f"لم ترد «{c.word}» {where} بهذه الصيغة." if ar else f"«{c.word}» does not occur {where} in this form.")
    if c.joined:
        lines.append((f"ووردت متصلة بحروف (مثل الواو والباء أو ضمير) {times_ar(len(c.joined))} أخرى: " if ar
                      else f"With letters joined to it (a conjunction, preposition or pronoun), {times_en(len(c.joined))} more: ")
                     + "، ".join(f"{w} ({n})" for w, n in by_reading(c.joined)[:8]))
    if c.root_asked:
        lead = ("عدّ الجذر بكل مشتقاته غير متاح بعد، وهذا عدّ الكلمة بصيغتها المكتوبة. " if ar
                else "Counting a whole root with its derivatives is not available yet; this counts the word as written. ") + lead
    note = ("عُدّ بالبرنامج من نص المصحف كله (رواية حفص عن عاصم، بالرسم الإملائي)، لا بنموذج لغوي. تُعدّ الكلمة بحروفها، ويُبيَّن اختلاف ضبطها إن وُجد."
            if ar else "Counted by the program over the whole mushaf text (Hafs from Asim), not by a language model.")
    if c.alif_ignored:
        note += (" لم توجد الكلمة كما كُتبت، فعُدّت مع تجاهل كتابة الألف (مثل «السموات» و«السماوات»)." if ar
                 else " The word was not found as written, so it was counted with the long alif spelling aside.")
    return lead, lines, (c.exact + c.joined)[:SHOWN], note
