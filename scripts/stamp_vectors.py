"""Record which passages the vector index was made from (audit M-011). Run it once after building or downloading
vectors that you know match the database; the server then turns meaning search off if the two ever stop matching.

    .venv/bin/python scripts/stamp_vectors.py            # data/muhawir.db and data/vectors*.*
    sudo systemctl restart muhawir

No model is called. It reads every passage once (about a minute on the server).
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muhawir.vectors import CORPUS_STAMP, DATA_DIR, corpus_fingerprint, load_vectors  # noqa: E402


def main() -> int:
    db = Path(os.environ.get("MUHAWIR_DB") or ROOT / "data" / "muhawir.db")
    folder = Path(os.environ.get("MUHAWIR_VECTORS_DIR") or DATA_DIR)
    vectors = load_vectors(folder)
    if vectors is None or not db.exists():
        print("needs both the database and a complete vector index", file=sys.stderr)
        return 2
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        present = {r[0] for r in con.execute("SELECT id FROM passages")}
        missing = sum(1 for pid in vectors.ids if pid not in present)
        if missing:
            print(f"{missing} vector ids are not in the database: these vectors were made from other passages; "
                  "rebuild them instead of stamping", file=sys.stderr)
            return 1
        (folder / CORPUS_STAMP).write_text(corpus_fingerprint(con, vectors.ids) + "\n", encoding="utf-8")
    finally:
        con.close()
    print(f"stamped {len(vectors.ids)} vectors against {db.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
