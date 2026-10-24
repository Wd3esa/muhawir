"""List or remove cached replies matching all words of a question, without loading the model.

Default is read-only. Set MUHAWIR_STORE to target a non-default answer cache.
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muhawir.normalize import normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or forget cached answers by question phrase")
    parser.add_argument("phrase", help="words of the user's question (Arabic spelling variants are normalized)")
    parser.add_argument("--delete", action="store_true", help="remove matching replies")
    args = parser.parse_args()
    words = normalize(args.phrase).split()
    if not words:
        parser.error("phrase cannot be blank")
    path = Path(os.environ.get("MUHAWIR_STORE") or Path(__file__).resolve().parents[1] / "data" / "answers.db")
    if not path.is_file():
        parser.error(f"answer cache missing: {path}")
    con = sqlite3.connect(path)
    try:
        rows = con.execute("SELECT key FROM replies").fetchall()
        keys = []
        for (key,) in rows:
            try:
                question = normalize(str(json.loads(key)[0]))
            except (ValueError, TypeError, IndexError, KeyError):
                continue
            if all(word in question.split() for word in words):
                keys.append(key)
        if args.delete:
            with con:
                con.executemany("DELETE FROM replies WHERE key=?", [(key,) for key in keys])
        print(f"{len(keys)} {'removed' if args.delete else 'matching'}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
