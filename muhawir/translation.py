"""The English translation of the meanings of the Quran shown beside a verse in English answers:
«The Noble Quran» by Muhammad Taqi-ud-Din al-Hilali and Muhammad Muhsin Khan, published by the King Fahd
Complex for the Printing of the Holy Quran (Madinah), the publisher of the Arabic text Muhawir uses.

Data: github.com/fawazahmed0/quran-api, edition «eng-muhammadtaqiudd» (taken from tanzil.net), pinned to
one commit so every build gives the same text. The repository is released under The Unlicense, but the
translation itself remains the work of its translators and publisher; it is shown as it is, attributed,
and never stored in the repository.

The translation is not a searchable passage: it is kept beside each verse (passage id «q:surah:ayah»), so
the search and the vector index are unchanged. In an English answer the writing step sees it next to the
Arabic verse and quotes it instead of translating the verse itself, and each verse card shows it.
"""
from __future__ import annotations

import json

COMMIT = "47ca096b0976443ba2eab2e45cdf0fb4096a2610"
PATH = "editions/eng-muhammadtaqiudd.min.json"
FILE = "quran-en-hilali-khan.min.json"
DOWNLOAD = (f"https://raw.githubusercontent.com/fawazahmed0/quran-api/{COMMIT}/{PATH}",
            f"https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@{COMMIT}/{PATH}")
VERSES = 6236
NAME = ("The Noble Quran, English translation of the meanings by Muhammad Taqi-ud-Din al-Hilali and "
        "Muhammad Muhsin Khan (King Fahd Complex, Madinah)")
NAME_AR = "ترجمة معاني القرآن الكريم إلى الإنجليزية، للهلالي ومحسن خان (مجمع الملك فهد)"


class ImportError_(ValueError):
    pass


def load(text: str) -> dict[str, str]:
    """{passage id «q:surah:ayah»: English text} for every verse; a wrong or short file is refused."""
    try:
        verses = json.loads(text)["quran"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ImportError_(f"not the quran-api translation file: {exc}") from exc
    out = {}
    for v in verses:
        if not (isinstance(v, dict) and isinstance(v.get("text"), str) and v["text"].strip()):
            raise ImportError_("a verse without text in the translation file")
        out[f"q:{int(v['chapter'])}:{int(v['verse'])}"] = v["text"].strip()
    if len(out) != VERSES:
        raise ImportError_(f"{len(out)} verses in the translation file, {VERSES} expected")
    return out


def add_to_corpus(corpus: dict, text: str) -> dict:
    """Attach the translation to a corpus built by quranpedia.build_corpus (only to its Quran verses)."""
    verses = load(text)
    ids = {p["id"] for p in corpus["passages"] if p.get("kind") == "quran"}
    corpus["_translations"] = {pid: t for pid, t in verses.items() if pid in ids}
    corpus["_translation_name"] = NAME
    corpus.setdefault("_provenance", {})["translation_commit"] = COMMIT
    return corpus
