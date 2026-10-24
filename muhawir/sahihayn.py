"""Import Sahih al-Bukhari and Sahih Muslim (Arabic) from the open hadith-api data.

Source: https://github.com/fawazahmed0/hadith-api (released into the public domain,
"The Unlicense"), files editions/ara-bukhari.min.json and editions/ara-muslim.min.json.
The organizers' guide names «الصحيحان» first among hadith sources.

Numbering: each hadith keeps its standard number ("arabicnumber"): al-Bukhari as in
Fath al-Bari, Muslim as in Muhammad Fuad Abd al-Baqi's edition (further chains of the
same hadith, numbered 8.01, 8.02 in the data, are shown under their main number). Entries without a
standard number (mostly Muslim's introduction) are left out, so every card can be
looked up in a printed copy. The wording is unchanged; only invisible direction marks
(U+200E, U+200F) are removed and spaces collapsed. Very long hadiths are split at
sentence ends into parts of about MAX_PART characters, never inside a word.

No grading is added to the two Sahih books: both are accepted as authentic, and the card names the book
and number instead of inventing a grade.

The other books of the same data (the four Sunan, al-Muwatta, the Forty of al-Nawawi) hold hadiths of every grade,
and the data carries the grading of each by named scholars (al-Albani, Zubair Ali Zai, al-Arnaut...) in English.
Their gradings are translated into Arabic word by word (translate_grades), each kept under its scholar's name, and
a grading the table cannot translate completely is dropped, never guessed. A hadith with no translatable grading
says so on its card, and a weak, disputed or non-Prophetic one is named as such in any answer that uses it
(pipeline._hadith_notice).
"""
from __future__ import annotations

import re

BOOKS = {
    "bukhari": {"file": "ara-bukhari.min.json", "prefix": "b", "source_id": "sahih-bukhari",
                "name": "صحيح البخاري",
                "about": "«الجامع المسند الصحيح» للإمام محمد بن إسماعيل البخاري (ت 256هـ)، بالترقيم المتداول "
                         "(ترقيم فتح الباري). النص العربي من بيانات hadith-api المفتوحة "
                         "(github.com/fawazahmed0/hadith-api، ملكية عامة)."},
    "muslim": {"file": "ara-muslim.min.json", "prefix": "m", "source_id": "sahih-muslim",
               "name": "صحيح مسلم",
               "about": "«المسند الصحيح» للإمام مسلم بن الحجاج (ت 261هـ)، بترقيم محمد فؤاد عبد الباقي. "
                        "النص العربي من بيانات hadith-api المفتوحة "
                        "(github.com/fawazahmed0/hadith-api، ملكية عامة)."},
    "abudawud": {"file": "ara-abudawud.min.json", "prefix": "ad", "source_id": "sunan-abi-dawud", "graded": True,
                 "name": "سنن أبي داود",
                 "about": "«السنن» للإمام أبي داود سليمان بن الأشعث السجستاني (ت 275هـ). فيها الصحيح والحسن والضعيف، "
                          "فيُعرض مع كل حديث حكم المحدثين عليه كما في البيانات. النص من بيانات hadith-api المفتوحة "
                          "(github.com/fawazahmed0/hadith-api، ملكية عامة)."},
    "tirmidhi": {"file": "ara-tirmidhi.min.json", "prefix": "tm", "source_id": "jami-al-tirmidhi", "graded": True,
                 "name": "جامع الترمذي",
                 "about": "«الجامع» للإمام أبي عيسى محمد بن عيسى الترمذي (ت 279هـ). فيه الصحيح والحسن والضعيف، "
                          "فيُعرض مع كل حديث حكم المحدثين عليه كما في البيانات. النص من بيانات hadith-api المفتوحة "
                          "(github.com/fawazahmed0/hadith-api، ملكية عامة)."},
    "nasai": {"file": "ara-nasai.min.json", "prefix": "ns", "source_id": "sunan-al-nasai", "graded": True,
              "name": "سنن النسائي",
              "about": "«السنن الصغرى» للإمام أحمد بن شعيب النسائي (ت 303هـ). فيها الصحيح والضعيف، "
                       "فيُعرض مع كل حديث حكم المحدثين عليه كما في البيانات. النص من بيانات hadith-api المفتوحة "
                       "(github.com/fawazahmed0/hadith-api، ملكية عامة)."},
    "ibnmajah": {"file": "ara-ibnmajah.min.json", "prefix": "ib", "source_id": "sunan-ibn-majah", "graded": True,
                 "name": "سنن ابن ماجه",
                 "about": "«السنن» للإمام محمد بن يزيد ابن ماجه (ت 273هـ). فيها الصحيح والضعيف، "
                          "فيُعرض مع كل حديث حكم المحدثين عليه كما في البيانات. النص من بيانات hadith-api المفتوحة "
                          "(github.com/fawazahmed0/hadith-api، ملكية عامة)."},
    "malik": {"file": "ara-malik.min.json", "prefix": "ml", "source_id": "muwatta-malik", "graded": True,
              "name": "موطأ مالك",
              "about": "«الموطأ» للإمام مالك بن أنس (ت 179هـ)، برواية يحيى بن يحيى الليثي. فيه المرفوع والموقوف والمقطوع، "
                       "فيُعرض مع كل حديث حكم المحدثين عليه كما في البيانات. النص من بيانات hadith-api المفتوحة "
                       "(github.com/fawazahmed0/hadith-api، ملكية عامة)."},
    "nawawi": {"file": "ara-nawawi.min.json", "prefix": "nw", "source_id": "arbaeen-nawawi", "graded": False,
               "name": "الأربعون النووية",
               "about": "«الأربعون النووية» للإمام يحيى بن شرف النووي (ت 676هـ). النص من بيانات hadith-api المفتوحة "
                        "(github.com/fawazahmed0/hadith-api، ملكية عامة)."},
}
DOWNLOAD = ("https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{file}",
            "https://raw.githubusercontent.com/fawazahmed0/hadith-api/1/editions/{file}")
MAX_PART = 1500
MIN_HADITH = {"bukhari": 7000, "muslim": 5000, "abudawud": 5000, "tirmidhi": 3800, "nasai": 5500,
              "ibnmajah": 4200, "malik": 1700, "nawawi": 40}  # a truncated download is refused

_MARKS = re.compile("[‎‏]")
_SPACES = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.؟!])\s+|(?<=\.\")\s+")


# the scholars who grade in the data, as they are named in Arabic: a name not here is not shown
SCHOLARS = {
    "al-albani": "الألباني", "zubair ali zai": "زبير علي زئي", "shuaib al arnaut": "شعيب الأرناؤوط",
    "abu ghuddah": "أبو غدة", "ahmad muhammad shakir": "أحمد شاكر", "bashar awad maarouf": "بشار عواد معروف",
    "salim al-hilali": "سليم الهلالي",
}  # the two editors of the printed books are left out: they numbered and edited, they did not grade
# the words of a grading, in Arabic. «Isnaad» stands before a grade as «إسناده» and after it as «الإسناد».
_GRADE_PHRASES = (  # longest first, applied before the single words
    ("sahih bukhari and muslim", "في_الصحيحين"), ("bukhari and muslim", "في_الصحيحين"),
    ("sahih bukhari", "في_صحيح_البخاري"), ("sahih muslim", "في_صحيح_مسلم"), ("agreed upon", "متفق عليه"),
    ("very daif", "ضعيف_جدًا"), ("bukhari", "في_البخاري"), ("muslim", "في_مسلم"))
_GRADE_WORDS = {
    "sahih": "صحيح", "hasan": "حسن", "daif": "ضعيف", "mawdu": "موضوع", "munkar": "منكر", "shadh": "شاذ",
    "mutawatir": "متواتر", "mauquf": "موقوف", "muquf": "موقوف", "mawquf": "موقوف", "maqtu": "مقطوع",
    "lighairihi": "لغيره", "hadith": "", "batil": "باطل", "mursal": "مرسل", "malool": "معلول"}
_ISNAD = ("isnaad", "isnad", "sanad")


def translate_grade(grade: str) -> str | None:
    """One grading in Arabic («Hasan Sahih» → «حسن صحيح», «Isnaad Hasan» → «إسناده حسن», «Sahih Isnaad» → «صحيح
    الإسناد»). None when any word of it is not in the table: the grading is then left out, never guessed."""
    text = re.sub(r"[-–—,]", " ", re.sub(r"\([^)]*\)", " ", grade.lower()))  # «(1023)»: a number in another book
    for en, ar in _GRADE_PHRASES:
        text = text.replace(en, ar)
    out: list[str] = []
    for word in text.split():
        if word in _ISNAD:
            out.append("الإسناد" if out else "إسناده")
        elif word == "matn":
            out.append("المتن" if out else "متنه")
        elif word in _GRADE_WORDS:
            if _GRADE_WORDS[word]:
                out.append(_GRADE_WORDS[word])
        elif re.fullmatch(r"[\u0600-\u06FF()_]+", word):  # already translated by a phrase above
            out.append(word)
        else:
            return None
    return " ".join(out).replace("_", " ") if out else None


NO_GRADE = "لم يُنقل له حكم في بيانات المصدر"


def translate_grades(grades: list[dict]) -> str:
    """«الألباني: صحيح؛ زبير علي زئي: حسن»: each scholar's grading that can be translated, in the data's order."""
    parts = []
    for g in grades or []:
        who, how = SCHOLARS.get(str(g.get("name", "")).strip().lower()), translate_grade(str(g.get("grade", "")))
        if who and how:
            parts.append(f"{who}: {how}")
    return "؛ ".join(parts)


class ImportError_(ValueError):
    pass


def clean(text: str) -> str:
    return _SPACES.sub(" ", _MARKS.sub("", text)).strip()


def split_long(text: str, limit: int = MAX_PART) -> list[str]:
    """Whole sentences joined up to about `limit` characters; a hadith shorter than that stays whole."""
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + 1 + len(sentence) > limit:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        parts.append(current)
    return parts


def build_book(key: str, dump: dict) -> tuple[dict, list[dict]]:
    book = BOOKS[key]
    passages, seen = [], set()
    for h in dump.get("hadiths", []):
        number, text = h.get("arabicnumber"), clean(h.get("text") or "")
        if number in (None, "") or not text:
            continue
        number = str(number)
        if number in seen:
            raise ImportError_(f"{key}: hadith number {number} appears twice")
        seen.add(number)
        parts = split_long(text)
        for i, part in enumerate(parts, 1):
            # Muslim's sub-numbers (8.01, 8.02: further chains of hadith 8) are cited by the main number
            shown = number.split(".")[0]
            where = f"{book['name']}، رقم {shown}" + (f" (الجزء {i} من {len(parts)})" if len(parts) > 1 else "")
            passage = {"id": f"{book['prefix']}:{number}" + (f":{i}" if len(parts) > 1 else ""),
                       "source_id": book["source_id"], "location": where, "kind": "hadith", "text": part}
            if book.get("graded"):
                passage["grade"] = translate_grades(h.get("grades")) or NO_GRADE
            passages.append(passage)
    if len(seen) < MIN_HADITH[key]:
        raise ImportError_(f"{key}: only {len(seen)} numbered hadiths; the file looks incomplete")
    # no "open source" link: the data repository is credited in `about`; each card links to dorar to check the hadith
    source = {"id": book["source_id"], "name": book["name"], "about": book["about"], "url": ""}
    return source, passages


def add_to_corpus(corpus: dict, dumps: dict[str, dict]) -> dict:
    """Add the books in `dumps` ({"bukhari": ..., "muslim": ...}) to a corpus built by quranpedia.build_corpus."""
    for key, dump in dumps.items():
        source, passages = build_book(key, dump)
        corpus["sources"].append(source)
        corpus["passages"].extend(passages)
        corpus.setdefault("_provenance", {})[f"{key}_entries"] = len({p["id"].rsplit(":", 1)[0] if p["id"].count(":") > 1
                                                                       else p["id"] for p in passages})
    return corpus
