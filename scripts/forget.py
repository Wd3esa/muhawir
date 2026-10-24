"""List or remove cached replies matching a question phrase, without loading the model.

Default is read-only. Set MUHAWIR_STORE to target a non-default answer cache.
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or forget cached answers by question phrase")
    parser.add_argument("phrase", help="part of the user's question")
    parser.add_argument("--delete", action="store_true", help="remove matching replies")
    args = parser.parse_args()
    if not args.phrase.strip():
        parser.error("phrase cannot be blank")
    path = Path(os.environ.get("MUHAWIR_STORE") or Path(__file__).resolve().parents[1] / "data" / "answers.db")
    if not path.is_file():
        parser.error(f"answer cache missing: {path}")
    con = sqlite3.connect(path)
    try:
        rows = con.execute("SELECT key FROM replies").fetchall()
        keys = [row[0] for row in rows if args.phrase in str(json.loads(row[0])[0])]
        if args.delete:
            with con:
                con.executemany("DELETE FROM replies WHERE key=?", [(key,) for key in keys])
        print(f"{len(keys)} {'removed' if args.delete else 'matching'}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
