"""Disk-backed corpus and search (SQLite FTS5).

Same interface as the in-memory Corpus and Retriever, so the pipeline does not
change, but passages stay on disk: memory stays low and start-up is instant.
Text is tokenized with muhawir.normalize before indexing and querying, so the
Arabic matching rules are identical to the in-memory search.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from . import bayyinat, lexical
from .corpus import Passage, Source, parse_corpus
from .normalize import normalize, tokenize
from .retrieve import Hit

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources (id TEXT PRIMARY KEY, name TEXT NOT NULL, about TEXT NOT NULL, url TEXT NOT NULL);
CREATE TABLE passages (
    rid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, source_id TEXT NOT NULL REFERENCES sources(id),
    location TEXT NOT NULL, kind TEXT NOT NULL, grade TEXT NOT NULL, text TEXT NOT NULL,
    keywords TEXT NOT NULL, tokens TEXT NOT NULL);
CREATE INDEX passages_source_rid ON passages(source_id, rid);
CREATE TABLE translations (passage_id TEXT PRIMARY KEY, text TEXT NOT NULL);
CREATE TABLE other_translations (lang TEXT NOT NULL, passage_id TEXT NOT NULL, text TEXT NOT NULL,
                                 PRIMARY KEY (lang, passage_id));
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
            ("translation_name", corpus.translation_name),
            *((f"translation_name_{lang}", name) for lang, name in sorted(corpus.other_translation_names.items()))])
        con.executemany("INSERT INTO translations VALUES (?, ?)", sorted(corpus.translations.items()))
        con.executemany("INSERT INTO other_translations VALUES (?, ?, ?)",
                        [(lang, pid, text) for lang, verses in sorted(corpus.other_translations.items())
                         for pid, text in sorted(verses.items())])
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
        self._path = str(path)
        self._local = threading.local()
        meta = dict(self.con.execute("SELECT key, value FROM meta"))
        self.synthetic = json.loads(meta.get("synthetic", "false"))
        self.provenance = json.loads(meta.get("provenance", "{}"))
        self.sources = {r[0]: Source(*r) for r in self.con.execute("SELECT id, name, about, url FROM sources")}
        self.count = self.con.execute("SELECT count(*) FROM passages").fetchone()[0]
        self.translation_name = meta.get("translation_name", "")
        self.translation_names = {k.removeprefix("translation_name_"): v for k, v in meta.items()
                                  if k.startswith("translation_name_")}

    @property
    def con(self) -> sqlite3.Connection:
        """This thread's own read-only connection: the searches run in several threads at once, and one connection
        shared between them fails with «bad parameter or other API misuse»."""
        con = getattr(self._local, "con", None)
        if con is None:
            con = self._local.con = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
        return con

    @property
    def passages(self) -> range:  # only its length is used outside this module
        """A range as long as the corpus; only its length is used outside this module."""
        return range(self.count)

    @staticmethod
    def _row(r) -> Passage:
        return Passage(id=r[0], source_id=r[1], location=r[2], text=r[5], kind=r[3], grade=r[4],
                       keywords=r[6])

    def passage(self, passage_id: str) -> Passage | None:
        """The passage with this id, read from the database, or None."""
        r = self.con.execute("SELECT id, source_id, location, kind, grade, text, keywords "
                             "FROM passages WHERE id = ?", (passage_id,)).fetchone()
        return self._row(r) if r else None

    def source_of(self, passage: Passage) -> Source:
        """The source a passage belongs to."""
        return self.sources[passage.source_id]

    def neighbours(self, passage_id: str) -> tuple[Passage | None, Passage | None]:
        """The passages just before and after this one in the same book (None at either end): the book's own order,
        as the passages were written into the database."""
        r = self.con.execute("SELECT rid, source_id FROM passages WHERE id = ?", (passage_id,)).fetchone()
        if r is None:
            return None, None
        cols = "SELECT id, source_id, location, kind, grade, text, keywords FROM passages WHERE source_id = ? AND "
        before = self.con.execute(cols + "rid < ? ORDER BY rid DESC LIMIT 1", (r[1], r[0])).fetchone()
        after = self.con.execute(cols + "rid > ? ORDER BY rid LIMIT 1", (r[1], r[0])).fetchone()
        return (self._row(before) if before else None), (self._row(after) if after else None)

    def translation(self, passage_id: str, lang: str = "en") -> str:
        """A verse's approved translation in English or another language («ur»), or "" (also for a database built
        before that translation was added)."""
        try:
            if lang == "en":
                r = self.con.execute("SELECT text FROM translations WHERE passage_id = ?", (passage_id,)).fetchone()
            else:
                r = self.con.execute("SELECT text FROM other_translations WHERE lang = ? AND passage_id = ?",
                                     (lang, passage_id)).fetchone()
        except sqlite3.OperationalError:
            return ""
        return r[0] if r else ""

    def translation_name_for(self, lang: str = "en") -> str:
        """The name of the translation used for that language."""
        return self.translation_name if lang == "en" else self.translation_names.get(lang, "")


MAX_PHRASE_HITS = 8     # passages that contain the search words one after the other, listed before the rest
MAX_PHRASE_PROBES = 60   # phrase searches per query at most (M-009)
MAX_PHRASE_MATCHES = 60  # a phrase found in more passages than this (e.g. «قال رسول الله») identifies none of them
HEADING_SCORE = 100.0    # score of an entry found through the root heading: above any BM25 score
MIN_SUBPHRASE = 3        # shortest piece of a long phrase tried when the whole phrase is not found
# among passages holding the same phrase, the text itself comes before the books that discuss it
_ORIGINAL_FIRST = {"quran": 0, "hadith": 1}
_PRIMARY_HADITH = ("sahih-bukhari", "sahih-muslim")  # then the Sunan and al-Muwatta, whose hadiths are of every grade


class SqliteRetriever:
    """BM25 search through FTS5, returning the same Hit objects as Retriever.

    Passages that contain the search words in the same order (a phrase of a verse, a hadith or a
    jurist's sentence) come first, then the usual match on any of the words: a phrase in the
    source's own wording finds its passage even when much longer passages repeat its common words.
    A phrase recalled from memory is often wrong at its edges, so when the whole phrase is not found,
    its longest contiguous pieces are tried."""

    def __init__(self, corpus: SqliteCorpus) -> None:
        self.corpus = corpus
        self._headings: lexical.RootIndex | None = None

    def df(self, token: str) -> int:
        """In how many passages of the index a (normalized) token is."""
        row = self.corpus.con.execute("SELECT count(*) FROM passages_fts WHERE passages_fts MATCH ?",
                                      ('"' + token.replace('"', '""') + '"',)).fetchone()
        return row[0]

    def _heading_rows(self, question: str) -> list:
        """The entries of a dictionary arranged by root («لفى») that the words of a question may come from.

        Only for a question about a meaning («ما معنى ألفوا»): the entry of the root explains a form the book does
        not spell. Their score is above any BM25 score, so they come first after the phrase matches."""
        if not lexical.is_meaning_question(question):
            return []
        if self._headings is None:
            self._headings = lexical.RootIndex(
                r[0] for r in self.corpus.con.execute("SELECT DISTINCT keywords FROM passages WHERE kind = 'gharib'"))
        roots: set[str] = set()
        for term in lexical.raw_terms(question):
            roots |= self._headings.candidates(term)
        if not roots:
            return []
        marks = ", ".join("?" for _ in roots)
        rows = self.corpus.con.execute(
            "SELECT p.id, p.source_id, p.location, p.kind, p.grade, p.text, p.keywords, p.tokens, 0 "
            f"FROM passages p WHERE p.kind = 'gharib' AND p.keywords IN ({marks})", sorted(roots)).fetchall()
        # the entry that holds a form closest to a word asked comes first, then the next root
        terms = lexical.raw_terms(question)

        def closeness(row) -> float:
            words = set(normalize(row[5]).split())
            return max((max((lexical.alignment(t, w) for w in words if abs(len(w) - len(t)) <= 2), default=0.0)
                        for t in terms), default=0.0)
        scored = sorted(((closeness(r), r) for r in rows), key=lambda x: -x[0])
        return [r[:8] + (-(HEADING_SCORE + c),) for c, r in scored[:lexical.MAX_HEADING_PASSAGES]]

    def _match(self, match: str, limit: int, originals: bool = False) -> list:
        only = f" AND p.kind IN ({', '.join(repr(k) for k in _ORIGINAL_FIRST)})" if originals else ""
        return self.corpus.con.execute(
            "SELECT p.id, p.source_id, p.location, p.kind, p.grade, p.text, p.keywords, p.tokens, "
            "bm25(passages_fts) FROM passages_fts JOIN passages p ON p.rid = passages_fts.rowid "
            f"WHERE passages_fts MATCH ?{only} ORDER BY bm25(passages_fts) LIMIT ?", (match, limit)).fetchall()

    def _phrase_rows(self, tokens: list[str], limit: int) -> list:
        """Passages holding the tokens one after the other: the whole run, else its longest pieces."""
        if len(tokens) < 2:
            return []
        shortest = len(tokens) if len(tokens) < MIN_SUBPHRASE + 1 else max(MIN_SUBPHRASE, (len(tokens) + 1) // 2)
        probes = 0  # at most MAX_PHRASE_PROBES searches, longest pieces first (audit M-009: a 100-word quote with no
        # match made 1,326 searches); the cap is reached only by long quotes, whose longest pieces are tried first
        for size in range(len(tokens), shortest - 1, -1):
            found: dict[str, tuple] = {}
            for start in range(len(tokens) - size + 1):
                if probes >= MAX_PHRASE_PROBES:
                    break
                probes += 1
                phrase = '"' + " ".join(t.replace('"', '""') for t in tokens[start:start + size]) + '"'
                rows = self._match(phrase, MAX_PHRASE_MATCHES + 1)
                if len(rows) > MAX_PHRASE_MATCHES:
                    # quoted in too many books to point to one passage: the verse or hadith itself still does
                    # («الحلال بين والحرام بين» is in dozens of fatwas and commentaries, and in three hadiths)
                    rows = self._match(phrase, MAX_PHRASE_MATCHES + 1, originals=True)
                if len(rows) <= MAX_PHRASE_MATCHES:
                    found.update((r[0], r) for r in rows if r[0] not in found)
            if found:
                return sorted(found.values(), key=lambda r: (_ORIGINAL_FIRST.get(r[3], 2), r[1] not in _PRIMARY_HADITH, r[8]))[:limit]
            if probes >= MAX_PHRASE_PROBES:
                break
        return []

    def search(self, question: str, k: int = 3) -> list[Hit]:
        """The best k passages for a question: phrase matches first, then full-text (FTS5) matches ranked by BM25."""
        if k <= 0:
            return []
        ordered = tokenize(question)
        terms = list(dict.fromkeys(ordered))
        if not terms:
            return []
        # words that frame a question about a meaning («معنى») are in most entries of a dictionary: they say
        # nothing about the topic, so they do not count as covered (fall back to all words if nothing else is left)
        focus = lexical.focus_terms(question) or terms
        quote = lambda t: '"' + t.replace('"', '""') + '"'  # noqa: E731
        # A Bayyinat objection is split into question, gist and answer passages.
        # Look slightly deeper so question-only parts do not consume answer slots.
        scan = k + 8
        rows = self._phrase_rows(ordered, min(scan, MAX_PHRASE_HITS))
        seen = {r[0] for r in rows}
        for r in self._heading_rows(question):
            if r[0] not in seen:
                rows.append(r)
                seen.add(r[0])
        rows += [r for r in self._match(" OR ".join(quote(t) for t in terms), scan + len(rows)) if r[0] not in seen]
        hits = []
        for r in rows:
            passage = SqliteCorpus._row(r)
            if not bayyinat.is_answer_part(passage):
                continue
            doc_terms = set(r[7].split())
            coverage = sum(1 for t in focus if t in doc_terms) / len(focus)
            if r[8] <= -HEADING_SCORE:
                coverage = 1.0  # found by the heading of its root: the word is in it under another form
            hits.append(Hit(passage, -r[8], coverage))  # FTS5 bm25: lower is better
            if len(hits) == k:
                break
        return hits
