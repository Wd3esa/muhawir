"""Import «المفردات في غريب القرآن» by al-Raghib al-Isfahani (d. 502 AH): the meanings of the words of the Quran,
arranged by root («رشد»، «أبا»…), each with the verses that use the word. It answers what the Tafsir often does not
say plainly: what a rare word of a verse means («الرقيم»، «مرفقًا»، «رشدًا»), and how the forms of one root differ.

Source: the OpenITI corpus (github.com/OpenITI/0525AH), version 0502RaghibIsbahani.Mufradat.Shamela0023636-ara1
(ed. Safwan Adnan al-Dawudi, Dar al-Qalam and al-Dar al-Shamiyya, Damascus and Beirut, 1st ed., 1412 AH), pinned to
one commit so every build gives the same passages. Licence: CC BY-NC-SA 4.0 (non-commercial use with attribution);
cite OpenITI, DOI 10.5281/zenodo.17767721.

The file is in OpenITI mARkdown: "### | كتاب الراء" opens a letter, "### | رشد" opens the entry of a root, and the
lines of an entry begin with "# " or "~~" where the printed line broke (not where a paragraph begins), so the lines
of one entry are joined into one text. Only the markers are removed: "PageV01P354" (the END of a page), "ms610" and
the editor's footnote numbers «1»; the wording is unchanged. An entry longer than MAX_CHUNK characters is cut at
sentence ends into parts, each under the same root. The editor's introduction and indexes are left out, and so is a
heading the digital text made of a line of verse («477-»): its lines belong to the entry before it.
"""
from __future__ import annotations

import re

from .bidaya import _PAGE, _clean
from .sahihayn import split_long

COMMIT = "6290621b58e063ed1aa47ffd0a6fc1630191e7cb"
PATH = ("data/0502RaghibIsbahani/0502RaghibIsbahani.Mufradat/"
        "0502RaghibIsbahani.Mufradat.Shamela0023636-ara1")
FILE = "0502RaghibIsbahani.Mufradat.Shamela0023636-ara1"
DOWNLOAD = (f"https://raw.githubusercontent.com/OpenITI/0525AH/{COMMIT}/{PATH}",
            f"https://cdn.jsdelivr.net/gh/OpenITI/0525AH@{COMMIT}/{PATH}")
SOURCE_ID = "mufradat-raghib"
PREFIX = "g"
KIND = "gharib"
MAX_CHUNK = 1200
MIN_ENTRIES = 1500  # a truncated download is refused

SOURCE = {
    "id": SOURCE_ID,
    "name": "المفردات في غريب القرآن للراغب الأصفهاني",
    "about": "«المفردات في غريب القرآن» لأبي القاسم الحسين بن محمد الراغب الأصفهاني (ت 502هـ)، معجم لمعاني ألفاظ "
             "القرآن مرتب على الجذور، تحقيق صفوان عدنان الداودي، دار القلم والدار الشامية، دمشق وبيروت، الطبعة الأولى "
             "1412هـ. يُعتمد عليه لمعنى اللفظ القرآني، أما ما يذكره من أحاديث وأشعار فللاستشهاد اللغوي، ولم يُنقل هنا "
             "حكم المحقق عليها. النص من مدونة OpenITI المفتوحة (github.com/OpenITI، رخصة CC BY-NC-SA 4.0، "
             "DOI 10.5281/zenodo.17767721).",
    "url": "https://shamela.ws/book/23636",
}

_LETTER = re.compile(r"^كتاب\s+\S+$")
_FOOTNOTE = re.compile(r"«\s*[\d٠-٩]+\s*»")
_NOT_A_ROOT = re.compile(r"^[\d٠-٩\s\-–]*$")  # «477-»: a heading made of a line number
_END = "مقدمة للفهارس"


class ImportError_(ValueError):
    pass


def _title(line: str) -> str:
    return _clean(_PAGE.sub("", line.lstrip("#").strip().lstrip("|").strip())).strip(" :.")


def _text(line: str) -> str:
    return _clean(_FOOTNOTE.sub("", _PAGE.sub("", line)))


_WORD = re.compile(r"[\u0621-\u064A]+")
_ENDED = re.compile(r"[.؟?!:\]»)]\s*$")
_PUNCT = re.compile(r"[،:.؛/\[\]]")


def _glue(left: str, right: str, words: set[str]) -> str:
    """Join two pieces of text the digital edition cut apart: without a space when the word they make is a word of
    the book («واعت» + «بر» → «واعتبر»), with one otherwise («تعالى» + «جد» → «تعالى جد»)."""
    a, b = _WORD.findall(left[-30:]), _WORD.findall(right[:30])
    tight = a and b and left.rstrip() == left and _WORD.search(left[-1:]) and _WORD.match(right) \
        and a[-1] + b[0] in words
    return f"{left}{right}" if tight else f"{left} {right}"


def _is_cut(title: str, before: str, words: set[str]) -> bool:
    """A heading the digital text made of words in the middle of an entry, not a root: «ب عل» in «يج|ب عل|يه»
    (يجب عليه), «اف» in «الأعر|اف», «أسن» in «و|أسن الرجل» (وأسن), «رة، و» in «العش|رة، و|قيل». A real root comes
    after an entry that ended its sentence; a piece of text follows words cut in the middle and makes a word with them."""
    if _PUNCT.search(title):
        return True
    ended = bool(_ENDED.search(before))
    if " " in title:
        return not ended  # «إن وأن» is a real entry: it follows a finished sentence
    a, b = _WORD.findall(before[-30:]), _WORD.findall(title)
    return not ended and bool(a and b) and a[-1] + b[0] in words


def parse(text: str) -> list[dict]:
    """Entries in order: {"root", "letter", "text", "pages": [(volume, page), ...]}."""
    head, sep, body = text.partition("#META#Header#End#")
    if not sep or "المفردات في غريب القرآن" not in head:
        raise ImportError_("this is not the OpenITI file of «المفردات في غريب القرآن»")
    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    words = {w for ln in lines if not ln.startswith("#") or ln.startswith("# ") for w in _WORD.findall(_text(ln))}
    entries: list[dict] = []
    letter = ""
    current: dict | None = None
    started = glue_next = False
    page_ends: list[tuple[int, int]] = []

    def add(piece: str, tight: bool) -> None:
        if not current["pieces"]:
            current["pieces"].append(piece)
        elif tight:
            current["pieces"][-1] = _glue(current["pieces"][-1], piece, words)
        else:
            current["pieces"].append(piece)

    for line in lines:
        if line.startswith("###"):
            title = _title(line)
            page_ends += [(int(a), int(b)) for a, b in _PAGE.findall(line)]
            if title.startswith(_END):
                break
            if _LETTER.match(title):
                started, letter, current = True, title, None
                continue
            if not started:
                continue
            before = current["pieces"][-1] if current and current["pieces"] else ""
            if _NOT_A_ROOT.match(title.replace(".", "")):
                glue_next = False  # the number of a line of verse: the verse goes on in the same entry
            elif current is not None and before and _is_cut(title, before, words):
                add(title, True)
                glue_next = True
            else:
                current = {"root": title, "letter": letter, "pieces": [], "first": len(page_ends)}
                entries.append(current)
                glue_next = False
            continue
        if line.startswith("#") and not line.startswith("# "):
            continue  # other OpenITI tags
        piece = _text(line[2:] if line.startswith(("# ", "~~")) else line)
        page_ends += [(int(a), int(b)) for a, b in _PAGE.findall(line)]
        if current is not None and piece:
            add(piece, glue_next)
            current["last"] = len(page_ends)
        glue_next = False

    out = []
    for e in entries:
        if not e["pieces"]:
            continue
        # a page marker closes its page: the entry starts on the page after the last one closed before it
        first = page_ends[e["first"]] if e["first"] < len(page_ends) else None
        last = page_ends[min(e.get("last", e["first"]), len(page_ends) - 1)] if page_ends else None
        out.append({"root": e["root"], "letter": e["letter"], "text": " ".join(e["pieces"]),
                    "pages": [p for p in (first, last) if p]})
    return out


def _where(root: str, pages: list[tuple[int, int]]) -> str:
    where = f"مادة «{root}»"
    if pages:
        p1, p2 = pages[0][1], pages[-1][1]
        where += f" (ص{p1})" if p1 == p2 else f" (ص{p1}–{p2})"
    return where


def build(text: str, min_entries: int = MIN_ENTRIES) -> tuple[dict, list[dict]]:
    """Source record and passages: one entry per root, cut at sentence ends when long."""
    entries = parse(text)
    if len(entries) < min_entries:
        raise ImportError_(f"only {len(entries)} entries from «المفردات»: the file looks incomplete")
    passages: list[dict] = []
    for n, e in enumerate(entries, 1):
        parts = split_long(e["text"], MAX_CHUNK)
        for i, part in enumerate(parts, 1):
            passages.append({"id": f"{PREFIX}:{n}" + (f":{i}" if len(parts) > 1 else ""), "source_id": SOURCE_ID,
                             "kind": KIND, "location": _where(e["root"], e["pages"]), "text": part,
                             "keywords": e["root"]})
    return dict(SOURCE), passages


def add_to_corpus(corpus: dict, text: str) -> dict:
    """Add al-Raghib's «المفردات» to a corpus built by quranpedia.build_corpus."""
    source, passages = build(text)
    corpus["sources"].append(source)
    corpus["passages"].extend(passages)
    corpus.setdefault("_provenance", {})["mufradat_commit"] = COMMIT
    return corpus
