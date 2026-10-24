"""Append locally reviewed, explicitly licensed books without changing built-in importers.

The manifest is private runtime data beside the source dumps, never downloaded or
committed. It is append-only so existing passage IDs retain their vector order.
This checks metadata and structure, not the truth of a copyright claim or text.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .corpus import CorpusError, KINDS

FILE = "approved-extra-sources.json"
_ID = re.compile(r"extra-[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_BASIS = {"public-domain", "license", "permission"}


def _required(record: dict, fields: tuple[str, ...], where: str) -> None:
    for field in fields:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise CorpusError(f"{where}: missing or empty {field}")


def append_reviewed(corpus: dict, folder: Path) -> int:
    """Validate before touching the corpus; return the number of added passages.

    A new book must be appended at the end of this file. Editing an older book
    changes the source fingerprint and requires a deliberate full vector rebuild.
    """
    path = folder / FILE
    if not path.exists():
        return 0
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusError(f"{FILE}: unreadable JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or not isinstance(manifest.get("sources"), list):
        raise CorpusError(f"{FILE}: expected schema_version 1 and a sources list")
    sources, passages = [], []
    seen_sources = {s["id"] for s in corpus["sources"]}
    seen_passages = {p["id"] for p in corpus["passages"]}
    for index, book in enumerate(manifest["sources"], 1):
        where = f"book #{index}"
        if not isinstance(book, dict):
            raise CorpusError(f"{where}: expected an object")
        _required(book, ("id", "name", "about", "url", "edition", "rights_holder", "rights_basis", "rights_document"), where)
        sid = book["id"]
        if not _ID.fullmatch(sid) or sid in seen_sources:
            raise CorpusError(f"{where}: invalid or duplicate id {sid!r}")
        if not book["url"].startswith("https://") or book["rights_basis"] not in _BASIS:
            raise CorpusError(f"{where}: expected HTTPS source URL and a specific rights basis")
        entries = book.get("passages")
        if not isinstance(entries, list) or not entries:
            raise CorpusError(f"{where}: passages must be a nonempty list")
        seen_sources.add(sid)
        sources.append({key: book[key].strip() for key in ("id", "name", "about", "url")})
        for number, item in enumerate(entries, 1):
            place = f"{where}, passage #{number}"
            if not isinstance(item, dict):
                raise CorpusError(f"{place}: expected an object")
            _required(item, ("location", "text", "kind"), place)
            kind = item["kind"]
            if kind not in KINDS or kind == "quran":
                raise CorpusError(f"{place}: unknown kind or Quran text (use the canonical Quran importer)")
            if kind == "hadith":
                _required(item, ("grade",), place)
            pid = f"{sid}:{number}"
            if pid in seen_passages:
                raise CorpusError(f"{place}: duplicate passage id {pid}")
            seen_passages.add(pid)
            passages.append({"id": pid, "source_id": sid, "location": item["location"].strip(),
                             "text": item["text"].strip(), "kind": kind,
                             "grade": str(item.get("grade", "")).strip(),
                             "keywords": str(item.get("keywords", "")).strip()})
    # All-or-nothing: a bad later book cannot leave half a corpus behind.
    corpus["sources"].extend(sources)
    corpus["passages"].extend(passages)
    corpus.setdefault("_provenance", {})["extra_sources"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sources": [{"id": b["id"], "edition": b["edition"],
                     "rights_holder": b["rights_holder"], "rights_basis": b["rights_basis"],
                     "rights_document": b["rights_document"]} for b in manifest["sources"]],
    }
    return len(passages)


def main() -> None:
    """Fast structural check without downloading or building the real corpus."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate a local reviewed-source manifest")
    parser.add_argument("folder", type=Path, help=f"folder containing {FILE}")
    args = parser.parse_args()
    if not (args.folder / FILE).is_file():
        parser.error(f"{FILE} not found")
    data = {"sources": [], "passages": [], "_provenance": {}}
    count = append_reviewed(data, args.folder)
    print(f"Valid structure: {len(data['sources'])} sources, {count} passages; "
          f"SHA-256 {data['_provenance']['extra_sources']['sha256']}")


if __name__ == "__main__":
    main()
