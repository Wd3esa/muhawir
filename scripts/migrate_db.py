"""Bring an existing database up to the current schema without rebuilding it. Safe to run more than once.

    .venv/bin/python scripts/migrate_db.py            # data/muhawir.db, or MUHAWIR_DB
    sudo systemctl restart muhawir

Adds the index the book-neighbour lookup needs (passages by source in book order; audit M-010). A database built
after November 2026 already has it. Nothing else is changed; the passages and their order stay as they are.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    db = Path(os.environ.get("MUHAWIR_DB") or ROOT / "data" / "muhawir.db")
    if not db.exists():
        print(f"no database at {db}", file=sys.stderr)
        return 2
    con = sqlite3.connect(db)
    try:
        had = con.execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'passages_source_rid'").fetchone()
        con.execute("CREATE INDEX IF NOT EXISTS passages_source_rid ON passages(source_id, rid)")
        con.commit()
    finally:
        con.close()
    print("index passages_source_rid: " + ("already there" if had else "added"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
