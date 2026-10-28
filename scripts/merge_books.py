"""Add the books that only a candidate database has to the live database, at the end.

The live passages keep their ids and order, so the vector index already built for them stays valid
(only the appended passages need new vectors). Offline, no model calls. Writes a new file and never
changes the two inputs.

    python scripts/merge_books.py data/muhawir.db data/muhawir-candidate.db data/muhawir-merged.db
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

_COLUMNS = "id, source_id, location, kind, grade, text, keywords, tokens"


def merge(live: Path, candidate: Path, out: Path) -> tuple[int, int, list[str]]:
    """Copy `live` to `out`, then append every source that only `candidate` has. Returns (live, added, sources)."""
    if out.exists():
        raise SystemExit(f"{out} already exists; remove it or choose another name")
    src = sqlite3.connect(f"file:{live}?mode=ro", uri=True)
    dst = sqlite3.connect(out)
    try:
        src.backup(dst)
    finally:
        src.close()
    try:
        before = [r[0] for r in dst.execute("SELECT id FROM passages ORDER BY rid")]
        dst.execute("ATTACH DATABASE ? AS k", (f"file:{candidate}?mode=ro",))
        new = [r[0] for r in dst.execute(
            "SELECT id FROM k.sources WHERE id NOT IN (SELECT id FROM main.sources) ORDER BY rowid")]
        for sid in new:
            dst.execute("INSERT INTO sources SELECT * FROM k.sources WHERE id = ?", (sid,))
            dst.execute(f"INSERT INTO passages ({_COLUMNS}) SELECT {_COLUMNS} FROM k.passages "
                        "WHERE source_id = ? ORDER BY rid", (sid,))
        # The rights record of the appended books travels with them.
        live_meta = json.loads(dst.execute("SELECT value FROM meta WHERE key = 'provenance'").fetchone()[0])
        cand_meta = json.loads(dst.execute("SELECT value FROM k.meta WHERE key = 'provenance'").fetchone()[0])
        added = [b for b in cand_meta.get("extra_sources", {}).get("sources", []) if b.get("id") in new]
        if added:
            live_meta["extra_sources_added"] = {"sha256": cand_meta["extra_sources"].get("sha256", ""), "sources": added}
            dst.execute("UPDATE meta SET value = ? WHERE key = 'provenance'",
                        (json.dumps(live_meta, ensure_ascii=False),))
        dst.execute("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')")
        dst.commit()
        after = [r[0] for r in dst.execute("SELECT id FROM passages ORDER BY rid")]
        if after[:len(before)] != before:
            raise SystemExit("the live passages changed order: not safe")
        return len(before), len(after) - len(before), new
    finally:
        dst.close()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print(__doc__)
        return 2
    live, added, sources = merge(Path(args[0]), Path(args[1]), Path(args[2]))
    print(f"live passages kept: {live}; added: {added} from {len(sources)} books: {', '.join(sources)}")
    print(f"total: {live + added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
