"""Approved-source corpus: sources and the passages cut from them.

Every passage keeps its source, its location inside the source (surah and
ayah, or book and chapter) and, for hadith, the grading stated by the source.
A source also carries a one-line introduction, because the source card has
to tell the reader what the source is, not only its name.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KINDS = frozenset({"quran", "tafsir", "asbab", "hadith", "fiqh", "aqeedah", "seerah", "fatwa", "qa", "gharib", "takhrij", "other"})


class CorpusError(ValueError):
    """Raised when corpus data is malformed."""
    pass


@dataclass(frozen=True)
class Source:
    """A book or collection the passages come from."""
    id: str
    name: str
    about: str  # one line: what this source is
    url: str = ""


@dataclass(frozen=True)
class Passage:
    """One quotable piece of a source, with its location and, for hadith, the grading the source states."""
    id: str
    source_id: str
    location: str
    text: str
    kind: str = "other"
    grade: str = ""  # hadith grading as stated by the source, if any
    keywords: str = ""  # search-only terms (e.g. topic names); never shown as the quote
    translation: str = ""  # a verse's approved English translation, attached when needed (translation.py)


@dataclass
class Corpus:
    """All sources and passages held in memory, with the approved translations of the verses."""
    sources: dict[str, Source]
    passages: list[Passage]
    synthetic: bool = False
    translations: dict[str, str] = field(default_factory=dict)
    translation_name: str = ""
    other_translations: dict[str, dict[str, str]] = field(default_factory=dict)  # {"ur": {passage id: text}}
    other_translation_names: dict[str, str] = field(default_factory=dict)
    _by_id: dict[str, Passage] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {p.id: p for p in self.passages}

    def passage(self, passage_id: str) -> Passage | None:
        """The passage with this id, or None."""
        return self._by_id.get(passage_id)

    def source_of(self, passage: Passage) -> Source:
        """The source a passage belongs to."""
        return self.sources[passage.source_id]

    def translation(self, passage_id: str, lang: str = "en") -> str:
        """The approved translation of a verse in the given language; empty if there is none."""
        if lang == "en":
            return self.translations.get(passage_id, "")
        return self.other_translations.get(lang, {}).get(passage_id, "")

    def translation_name_for(self, lang: str = "en") -> str:
        """The name of the translation used for that language."""
        return self.translation_name if lang == "en" else self.other_translation_names.get(lang, "")

    def neighbours(self, passage_id: str) -> tuple[Passage | None, Passage | None]:
        """The passages just before and after this one in the same book (None at either end)."""
        p = self._by_id.get(passage_id)
        if p is None:
            return None, None
        i = self.passages.index(p)
        before = self.passages[i - 1] if i > 0 else None
        after = self.passages[i + 1] if i + 1 < len(self.passages) else None
        same = lambda q: q if q is not None and q.source_id == p.source_id else None  # noqa: E731
        return same(before), same(after)


def _require(obj: dict, keys: tuple[str, ...], where: str) -> None:
    for key in keys:
        value = obj.get(key)
        if not isinstance(value, str) or not value.strip():
            raise CorpusError(f"{where}: missing or empty '{key}'")


def parse_corpus(data: dict) -> Corpus:
    """Validate corpus data (a JSON object) and build a Corpus; raises CorpusError on any malformed entry."""
    if not isinstance(data, dict):
        raise CorpusError("corpus must be a JSON object")
    sources: dict[str, Source] = {}
    for i, raw in enumerate(data.get("sources", [])):
        _require(raw, ("id", "name", "about"), f"source #{i}")
        if raw["id"] in sources:
            raise CorpusError(f"duplicate source id '{raw['id']}'")
        sources[raw["id"]] = Source(raw["id"], raw["name"], raw["about"], raw.get("url", ""))

    passages: list[Passage] = []
    seen: set[str] = set()
    for i, raw in enumerate(data.get("passages", [])):
        _require(raw, ("id", "source_id", "location", "text"), f"passage #{i}")
        if raw["id"] in seen:
            raise CorpusError(f"duplicate passage id '{raw['id']}'")
        if raw["source_id"] not in sources:
            raise CorpusError(f"passage '{raw['id']}' refers to unknown source '{raw['source_id']}'")
        kind = raw.get("kind", "other")
        if kind not in KINDS:
            raise CorpusError(f"passage '{raw['id']}' has unknown kind '{kind}'")
        seen.add(raw["id"])
        passages.append(Passage(raw["id"], raw["source_id"], raw["location"], raw["text"],
                                kind, raw.get("grade", ""), raw.get("keywords", "")))
    if not passages:
        raise CorpusError("corpus has no passages")
    translations = data.get("_translations") or {}
    if not isinstance(translations, dict) or not all(isinstance(v, str) for v in translations.values()):
        raise CorpusError("_translations must map passage ids to text")
    unknown = [pid for pid in translations if pid not in seen]
    if unknown:
        raise CorpusError(f"translation for unknown passage '{unknown[0]}'")
    others = {lang: dict(v) for lang, v in (data.get("_other_translations") or {}).items()}
    return Corpus(sources, passages, bool(data.get("synthetic", False)), dict(translations),
                  str(data.get("_translation_name", "")), others, dict(data.get("_other_translation_names") or {}))


def load_corpus(path: str | Path) -> Corpus:
    """Read a corpus JSON file."""
    with open(path, encoding="utf-8") as fh:
        return parse_corpus(json.load(fh))
