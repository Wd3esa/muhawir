"""Disk-backed corpus and search (SQLite FTS5).

Same interface as the in-memory Corpus and Retriever, so the pipeline does not
change, but passages stay on disk: memory stays low and start-up is instant.
Text is tokenized with muhawir.normalize before indexing and querying, so the
Arabic matching rules are identical to the in-memory search.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .corpus import Passage, Source, parse_corpus
from .normalize import tokenize
from .retrieve import Hit

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources (id TEXT PRIMARY KEY, name TEXT NOT NULL, about TEXT NOT NULL, url TEXT NOT NULL);
CREATE TABLE passages (
    rid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, source_id TEXT NOT NULL REFERENCES sources(id),
    location TEXT NOT NULL, kind TEXT NOT NULL, grade TEXT NOT NULL, text TEXT NOT NULL,
    keywords TEXT NOT NULL, tokens TEXT NOT NULL);
CREATE TABLE translations (passage_id TEXT PRIMARY KEY, text TEXT NOT NULL);
CREATE VIRTUAL TABLE passages_fts USING fts5(tokens, content='passages', content_rowid='rid',
                                             tokenize='unicode61 remove_diacritics 0');
"""


def build_db(corpus_data: dict, path: str | Path) -> int:
    """Validate a corpus (same rules as the JSON corpus) and write it to a new SQLite file."""
    corpus = parse_corpus(corpus_data)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO meta VALUES (?, ?)", [
            ("synthetic", json.dumps(corpus.synthetic)),
            ("provenance", json.dumps(corpus_data.get("_provenance", {}), ensure_ascii=False)),
            ("translation_name", corpus.translation_name)])
        con.executemany("INSERT INTO translations VALUES (?, ?)", sorted(corpus.translations.items()))
        con.executemany("INSERT INTO sources VALUES (?, ?, ?, ?)",
                        [(s.id, s.name, s.about, s.url) for s in corpus.sources.values()])
        con.executemany(
            "INSERT INTO passages (id, source_id, location, kind, grade, text, keywords, tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(p.id, p.source_id, p.location, p.kind, p.grade, p.text, p.keywords,
              " ".join(tokenize(f"{p.text} {p.keywords}"))) for p in corpus.passages])
        con.execute("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')")
        con.commit()
    finally:
        con.close()
    tmp.replace(path)  # atomic: a half-built database is never served
    return len(corpus.passages)


class SqliteCorpus:
    """Read-only corpus backed by a database written by build_db."""

    def __init__(self, path: str | Path) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        meta = dict(self.con.execute("SELECT key, value FROM meta"))
        self.synthetic = json.loads(meta.get("synthetic", "false"))
        self.provenance = json.loads(meta.get("provenance", "{}"))
        self.sources = {r[0]: Source(*r) for r in self.con.execute("SELECT id, name, about, url FROM sources")}
        self.count = self.con.execute("SELECT count(*) FROM passages").fetchone()[0]
        self.translation_name = meta.get("translation_name", "")

    @property
    def passages(self) -> range:  # only its length is used outside this module
        return range(self.count)

    @staticmethod
    def _row(r) -> Passage:
        return Passage(id=r[0], source_id=r[1], location=r[2], text=r[5], kind=r[3], grade=r[4],
                       keywords=r[6])

    def passage(self, passage_id: str) -> Passage | None:
        r = self.con.execute("SELECT id, source_id, location, kind, grade, text, keywords "
                             "FROM passages WHERE id = ?", (passage_id,)).fetchone()
        return self._row(r) if r else None

    def source_of(self, passage: Passage) -> Source:
        return self.sources[passage.source_id]

    def translation(self, passage_id: str) -> str:
        """A verse's approved English translation, or "" (also for a database built before translations)."""
        try:
            r = self.con.execute("SELECT text FROM translations WHERE passage_id = ?", (passage_id,)).fetchone()
        except sqlite3.OperationalError:
            return ""
        return r[0] if r else ""


MAX_PHRASE_HITS = 8     # passages that contain the search words one after the other, listed before the rest
MAX_PHRASE_MATCHES = 60  # a phrase found in more passages than this (e.g. «قال رسول الله») identifies none of them
MIN_SUBPHRASE = 3        # shortest piece of a long phrase tried when the whole phrase is not found
# among passages holding the same phrase, the text itself comes before the books that discuss it
_ORIGINAL_FIRST = {"quran": 0, "hadith": 1}


class SqliteRetriever:
    """BM25 search through FTS5, returning the same Hit objects as Retriever.

    Passages that contain the search words in the same order (a phrase of a verse, a hadith or a
    jurist's sentence) come first, then the usual match on any of the words: a phrase in the
    source's own wording finds its passage even when much longer passages repeat its common words.
    A phrase recalled from memory is often wrong at its edges, so when the whole phrase is not found,
    its longest contiguous pieces are tried."""

    def __init__(self, corpus: SqliteCorpus) -> None:
        self.corpus = corpus

    def _match(self, match: str, limit: int) -> list:
        return self.corpus.con.execute(
            "SELECT p.id, p.source_id, p.location, p.kind, p.grade, p.text, p.keywords, p.tokens, "
            "bm25(passages_fts) FROM passages_fts JOIN passages p ON p.rid = passages_fts.rowid "
            "WHERE passages_fts MATCH ? ORDER BY bm25(passages_fts) LIMIT ?", (match, limit)).fetchall()

    def _phrase_rows(self, tokens: list[str], limit: int) -> list:
        """Passages holding the tokens one after the other: the whole run, else its longest pieces."""
        if len(tokens) < 2:
            return []
        shortest = len(tokens) if len(tokens) < MIN_SUBPHRASE + 1 else max(MIN_SUBPHRASE, (len(tokens) + 1) // 2)
        for size in range(len(tokens), shortest - 1, -1):
            found: dict[str, tuple] = {}
            for start in range(len(tokens) - size + 1):
                phrase = '"' + " ".join(t.replace('"', '""') for t in tokens[start:start + size]) + '"'
                rows = self._match(phrase, MAX_PHRASE_MATCHES + 1)
                if len(rows) <= MAX_PHRASE_MATCHES:
                    found.update((r[0], r) for r in rows if r[0] not in found)
            if found:
                return sorted(found.values(), key=lambda r: (_ORIGINAL_FIRST.get(r[3], 2), r[8]))[:limit]
        return []

    def search(self, question: str, k: int = 3) -> list[Hit]:
        ordered = tokenize(question)
        terms = list(dict.fromkeys(ordered))
        if not terms:
            return []
        quote = lambda t: '"' + t.replace('"', '""') + '"'  # noqa: E731
        rows = self._phrase_rows(ordered, min(k, MAX_PHRASE_HITS))
        seen = {r[0] for r in rows}
        rows += [r for r in self._match(" OR ".join(quote(t) for t in terms), k + len(rows)) if r[0] not in seen]
        hits = []
        for r in rows[:k]:
            doc_terms = set(r[7].split())
            coverage = sum(1 for t in terms if t in doc_terms) / len(terms)
            hits.append(Hit(SqliteCorpus._row(r), -r[8], coverage))  # FTS5 bm25: lower is better
        return hits
