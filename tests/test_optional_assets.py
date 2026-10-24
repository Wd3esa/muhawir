"""Optional assets are prepared from local source data without model calls."""
import gzip
import json
import zipfile

import numpy as np
import pytest

from scripts.prepare_optional_assets import ensure_ann, ensure_mushaf


def test_extracts_real_format_mushaf_without_download(tmp_path):
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    payload = gzip.compress(json.dumps({"data": {"id": 1, "surahs": []}}).encode())
    with zipfile.ZipFile(dumps / "mushafs-all.zip", "w") as archive:
        archive.writestr("mushafs/mushafs-1.json.gz", payload)

    path = ensure_mushaf(dumps, allow_download=False)

    assert path.read_bytes() == payload
    assert ensure_mushaf(dumps, allow_download=False) == path


def test_builds_and_reuses_ann_for_exact_existing_vectors(tmp_path, monkeypatch):
    pytest.importorskip("usearch")
    np.save(tmp_path / "vectors.npy", np.eye(4, dtype=np.float16))
    (tmp_path / "vectors_ids.json").write_text(json.dumps(["a", "b", "c", "d"]))
    (tmp_path / "vectors_model.txt").write_text("local:fake/model@abc")
    monkeypatch.delenv("MUHAWIR_VECTOR_INDEX", raising=False)

    assert ensure_ann(tmp_path) == 4
    index = tmp_path / "vectors.usearch"
    original_mtime = index.stat().st_mtime_ns
    assert ensure_ann(tmp_path) == 4
    assert index.stat().st_mtime_ns == original_mtime
