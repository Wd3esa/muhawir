import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("stitch_vectors", Path(__file__).resolve().parent.parent / "scripts" / "stitch_vectors.py")
stitch_vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stitch_vectors)


def _index(folder, ids, rows, model="m"):
    folder.mkdir()
    np.save(folder / "vectors.npy", np.asarray(rows, dtype=np.float16))
    (folder / "vectors_ids.json").write_text(json.dumps(ids))
    (folder / "vectors_model.txt").write_text(model + "\n")


def _db(path, ids):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE passages (rid INTEGER PRIMARY KEY, id TEXT)")
    con.executemany("INSERT INTO passages (id) VALUES (?)", [(i,) for i in ids])
    con.commit()
    con.close()


def test_stitch_takes_each_vector_from_the_index_that_has_it(tmp_path):
    _index(tmp_path / "live", ["a", "b"], [[1, 0], [0, 1]])
    _index(tmp_path / "new", ["a", "c", "d"], [[1, 0], [1, 1], [0, 2]])
    _db(tmp_path / "m.db", ["a", "b", "c", "d"])
    assert stitch_vectors.stitch(tmp_path / "live", tmp_path / "new", tmp_path / "m.db", tmp_path / "out") == (2, 2, 1)
    out = np.load(tmp_path / "out" / "vectors.npy")
    assert out.shape == (4, 2) and out[1].tolist() == [0, 1] and out[2].tolist() == [1, 1]
    assert json.loads((tmp_path / "out" / "vectors_ids.json").read_text()) == ["a", "b", "c", "d"]


def test_refuses_other_model_missing_vector_or_disagreement(tmp_path):
    _index(tmp_path / "live", ["a"], [[1, 0]])
    _db(tmp_path / "m.db", ["a", "c"])
    _index(tmp_path / "x", ["a"], [[1, 0]], model="other")
    with pytest.raises(SystemExit):
        stitch_vectors.stitch(tmp_path / "live", tmp_path / "x", tmp_path / "m.db", tmp_path / "o1")
    _index(tmp_path / "y", ["a"], [[1, 0]])
    with pytest.raises(SystemExit):  # the merged database wants "c" and no index has it
        stitch_vectors.stitch(tmp_path / "live", tmp_path / "y", tmp_path / "m.db", tmp_path / "o2")
    _index(tmp_path / "z", ["a", "c"], [[0, 1], [1, 1]])
    with pytest.raises(SystemExit):  # same id, different vector
        stitch_vectors.stitch(tmp_path / "live", tmp_path / "z", tmp_path / "m.db", tmp_path / "o3")
