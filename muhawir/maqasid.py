"""Import «المقاصد الحسنة في بيان كثير من الأحاديث المشتهرة على الألسنة» by al-Sakhawi (d. 902 AH): 1,356 sayings
people repeat as hadith («آفة الكذب النسيان»، «اطلبوا العلم ولو بالصين»…), each with where it is reported and how
sound it is: sound, weak, without a chain, the words of a Companion or a later man, or invented. It answers «هل هذا
حديث؟» about a saying that is in none of the hadith books here.

Source: the OpenITI corpus (github.com/OpenITI/0925AH), version 0902Sakhawi.MaqasidHasana.Shamela0023177-ara1 (ed.
Muhammad Uthman al-Khisht, Dar al-Kitab al-Arabi, Beirut, 1st ed., 1405/1985), pinned to one commit so every build
gives the same passages. Licence: CC BY-NC-SA 4.0 (non-commercial use with attribution); cite OpenITI,
DOI 10.5281/zenodo.17767721.

The file is in OpenITI mARkdown: the entries of «الباب الأول» are headed "### | 266 - " and begin «حديث: <the saying>،»;
their lines (starting "# " or "~~" where the printed line broke) are joined into one text. Removed: page markers
(this file's are all "PageV00P000"), "ms155", and the editor's footnote numbers «(1)». The editor's introduction and
the index of «الباب الثاني» (the same sayings listed by topic) are left out. Each entry is one passage, cut at sentence
ends when long, named by its number and the saying («المقاصد الحسنة، رقم 2: آفة الكذب النسيان»). The kind «takhrij»
tells the writer the saying is not to be quoted as the Prophet's words ﷺ without al-Sakhawi's verdict on it.
"""
from __future__ import annotations

import re

from .bidaya import _PAGE, _clean
from .sahihayn import split_long

COMMIT = "4f0f4562d1e13083c593e3a0ca6a19a3eb33805e"
PATH = "data/0902Sakhawi/0902Sakhawi.MaqasidHasana/0902Sakhawi.MaqasidHasana.Shamela0023177-ara1"
FILE = "0902Sakhawi.MaqasidHasana.Shamela0023177-ara1"
DOWNLOAD = (f"https://raw.githubusercontent.com/OpenITI/0925AH/{COMMIT}/{PATH}",
            f"https://cdn.jsdelivr.net/gh/OpenITI/0925AH@{COMMIT}/{PATH}")
SOURCE_ID = "maqasid-hasana"
PREFIX = "mq"
KIND = "takhrij"
MAX_CHUNK = 1200
MIN_ENTRIES = 1300  # a truncated download is refused

SOURCE = {
    "id": SOURCE_ID,
    "name": "المقاصد الحسنة للسخاوي",
    "about": "«المقاصد الحسنة في بيان كثير من الأحاديث المشتهرة على الألسنة» للحافظ شمس الدين محمد بن عبد الرحمن "
             "السخاوي (ت 902هـ)، تحقيق محمد عثمان الخشت، دار الكتاب العربي، بيروت، الطبعة الأولى 1405هـ/1985م. يذكر "
             "الأقوال المشتهرة على الألسنة بوصفها أحاديث، ومن رواها، وحكمه عليها: صحيح أو ضعيف أو لا أصل له أو موضوع. "
             "فالقول المذكور فيه ليس حديثًا ثابتًا لمجرد وروده فيه. النص من مدونة OpenITI المفتوحة (github.com/OpenITI، "
             "رخصة CC BY-NC-SA 4.0، DOI 10.5281/zenodo.17767721).",
    "url": "https://shamela.ws/book/23177",
}

_ENTRY = re.compile(r"^([\d٠-٩]+)\s*-$")
_FOOTNOTE = re.compile(r"\s*\(\s*[\d٠-٩]+\s*\)")
_FIRST = "الباب الأول"
_END = "الباب الثاني"
_SAYING = re.compile(r"^حديث\s*[:.]\s*([^،؛.]+)")


class ImportError_(ValueError):
    pass


def _title(line: str) -> str:
    return _clean(_PAGE.sub("", line.lstrip("#").strip().lstrip("|").strip()))


def parse(text: str) -> list[dict]:
    """Entries in order: {"number", "saying", "text"}."""
    head, sep, body = text.partition("#META#Header#End#")
    if not sep or "المقاصد الحسنة" not in head:
        raise ImportError_("this is not the OpenITI file of «المقاصد الحسنة»")
    entries: list[dict] = []
    current: dict | None = None
    started = False
    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("###"):
            title = _title(line)
            if title.startswith(_FIRST):
                started = True
            elif title.startswith(_END) and started:
                break
            m = _ENTRY.match(title) if started else None
            current = {"number": int(m.group(1)), "pieces": []} if m else None
            if current is not None:
                entries.append(current)
            continue
        if line.startswith("#") and not line.startswith("# "):
            continue  # other OpenITI tags
        piece = _clean(_FOOTNOTE.sub("", _PAGE.sub("", line[2:] if line.startswith(("# ", "~~")) else line)))
        if current is not None and piece:
            current["pieces"].append(piece)
    out = []
    for e in entries:
        text = " ".join(e["pieces"])
        if not text:
            continue
        m = _SAYING.match(text)
        saying = " ".join(m.group(1).split()[:12]) if m else ""
        out.append({"number": e["number"], "saying": saying, "text": text})
    return out


def build(text: str, min_entries: int = MIN_ENTRIES) -> tuple[dict, list[dict]]:
    """Source record and passages: one entry each, cut at sentence ends when long."""
    entries = parse(text)
    if len(entries) < min_entries:
        raise ImportError_(f"only {len(entries)} entries from «المقاصد الحسنة»: the file looks incomplete")
    passages: list[dict] = []
    for n, e in enumerate(entries, 1):  # the id follows the order: the digital text numbers two entries 662
        parts = split_long(e["text"], MAX_CHUNK)
        where = f"رقم {e['number']}" + (f": {e['saying']}" if e["saying"] else "")
        for i, part in enumerate(parts, 1):
            passages.append({"id": f"{PREFIX}:{n}" + (f":{i}" if len(parts) > 1 else ""),
                             "source_id": SOURCE_ID, "kind": KIND, "location": where, "text": part,
                             "keywords": e["saying"]})
    return dict(SOURCE), passages


def add_to_corpus(corpus: dict, text: str) -> dict:
    """Add al-Sakhawi's «المقاصد الحسنة» to a corpus built by quranpedia.build_corpus."""
    source, passages = build(text)
    corpus["sources"].append(source)
    corpus["passages"].extend(passages)
    corpus.setdefault("_provenance", {})["maqasid_commit"] = COMMIT
    return corpus
