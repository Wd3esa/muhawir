"""Join the live vector index with the vectors of the appended books, without embedding anything.

The live index covers the live passages; a second index (built elsewhere, e.g. on Kaggle) covers the
appended ones. This takes each passage's vector from whichever index has it and writes an index that
matches the merged database exactly, in order. Offline, no model calls. It refuses if the models differ,
if a vector is missing, or if the two indexes disagree on passages they both hold.

    python scripts/stitch_vectors.py data data/new-vectors data/muhawir-merged.db data/stitched
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

_CHECKED = 300       # shared passages compared between the two indexes
_AGREE = 0.98        # minimum cosine for a shared passage


def _load(folder: Path):
    ids = json.loads((folder / "vectors_ids.json").read_text(encoding="utf-8"))
    model = (folder / "vectors_model.txt").read_text(encoding="utf-8").strip()
    matrix = np.load(folder / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise SystemExit(f"{folder}: the vectors and their ids do not match")
    return ids, model, matrix


def stitch(live_dir: Path, other_dir: Path, db: Path, out_dir: Path) -> tuple[int, int, int]:
    live_ids, live_model, live = _load(live_dir)
    other_ids, other_model, other = _load(other_dir)
    if live_model != other_model or live.shape[1] != other.shape[1] or live.dtype != other.dtype:
        raise SystemExit(f"the two indexes differ in model or shape ({live_model} vs {other_model})")
    wanted = [r[0] for r in sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute(
        "SELECT id FROM passages ORDER BY rid")]
    if wanted[:len(live_ids)] != live_ids:
        raise SystemExit("the live index is not a prefix of the merged database")
    where_other = {pid: i for i, pid in enumerate(other_ids)}
    missing = [pid for pid in wanted[len(live_ids):] if pid not in where_other]
    if missing:
        raise SystemExit(f"{len(missing)} appended passages have no vector, first: {missing[0]}")
    shared = [pid for pid in live_ids if pid in where_other]
    step = max(1, len(shared) // _CHECKED)
    for pid in shared[::step][:_CHECKED]:
        a = np.asarray(live[live_ids.index(pid)], dtype=np.float32)
        b = np.asarray(other[where_other[pid]], dtype=np.float32)
        if float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) < _AGREE:
            raise SystemExit(f"the two indexes disagree on {pid}: not the same model or text")
    rows = np.asarray([other[where_other[pid]] for pid in wanted[len(live_ids):]], dtype=live.dtype)
    matrix = np.concatenate([np.asarray(live), rows]) if len(rows) else np.asarray(live)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vectors.npy", matrix, allow_pickle=False)
    (out_dir / "vectors_ids.json").write_text(json.dumps(wanted, ensure_ascii=False), encoding="utf-8")
    (out_dir / "vectors_model.txt").write_text(live_model + "\n", encoding="utf-8")
    return len(live_ids), len(rows), len(shared)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 4:
        print(__doc__)
        return 2
    kept, added, shared = stitch(*(Path(a) for a in args))
    print(f"live vectors kept: {kept}; added from the other index: {added}; "
          f"shared passages compared: {min(shared, _CHECKED)}; total: {kept + added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
