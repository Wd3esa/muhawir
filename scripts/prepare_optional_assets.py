"""Prepare optional real-data and ANN checks without changing the live service.

Run from the project's virtual environment. No embedding or answer-model calls are made.
The Quranpedia dump and generated ANN index stay in the ignored data directory.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def ensure_usearch() -> None:
    if importlib.util.find_spec("usearch") is None:
        print("Installing optional usearch dependency into the current Python environment", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "usearch>=2.16,<3"], check=True)
    import usearch.index  # noqa: F401 - verify that its native extension actually loads


def ensure_mushaf(folder: Path, *, allow_download: bool = True) -> Path:
    from muhawir.build_data import BASE, MUSHAFS_ZIP, MUSHAF_FILE

    folder.mkdir(parents=True, exist_ok=True)
    direct = folder / MUSHAF_FILE
    if not direct.is_file():
        archive = folder / MUSHAFS_ZIP
        if not archive.is_file():
            if not allow_download:
                raise FileNotFoundError(f"missing {archive}; download it from {BASE}")
            import httpx

            temporary_archive = archive.with_name(archive.name + ".part")
            print(f"Downloading Quranpedia's {MUSHAFS_ZIP}", flush=True)
            try:
                with httpx.stream("GET", BASE + MUSHAFS_ZIP, timeout=300, follow_redirects=True) as response:
                    response.raise_for_status()
                    with temporary_archive.open("wb") as stream:
                        for block in response.iter_bytes():
                            stream.write(block)
                with zipfile.ZipFile(temporary_archive) as source:
                    if source.testzip() is not None:
                        raise ValueError("Quranpedia archive is corrupt")
                os.replace(temporary_archive, archive)
            finally:
                temporary_archive.unlink(missing_ok=True)

        temporary_dump = direct.with_name(direct.name + ".part")
        try:
            with zipfile.ZipFile(archive) as source:
                members = [name for name in source.namelist() if Path(name).name == MUSHAF_FILE]
                if len(members) != 1:
                    raise ValueError(f"expected one {MUSHAF_FILE} in {archive}")
                with source.open(members[0]) as src, temporary_dump.open("wb") as dest:
                    shutil.copyfileobj(src, dest)
            with gzip.open(temporary_dump, "rt", encoding="utf-8") as src:
                mushaf = json.load(src)
            if mushaf.get("data", {}).get("id") != 1:
                raise ValueError("extracted file is not the Hafs mushaf")
            os.replace(temporary_dump, direct)
        finally:
            temporary_dump.unlink(missing_ok=True)
    print(f"QURANPEDIA_MUSHAF={direct.resolve()}", flush=True)
    return direct


def ensure_ann(folder: Path) -> int:
    from muhawir.vectors import build_ann, load_vectors

    ensure_usearch()
    os.environ["MUHAWIR_VECTOR_INDEX"] = "usearch"
    vectors = load_vectors(folder)
    if vectors is None:
        raise FileNotFoundError(f"complete vectors.npy, vectors_ids.json and vectors_model.txt required in {folder}")
    if vectors.ann is None:
        print("Building USearch from existing vectors (no model calls)", flush=True)
        build_ann(folder)
    verified = load_vectors(folder)
    if verified is None or verified.ann is None:
        raise RuntimeError("USearch index failed its source fingerprint/integrity check")
    print(f"USearch ready: {len(verified.ids)} vectors, {verified.matrix.shape[1]} dimensions", flush=True)
    return len(verified.ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dumps-dir", type=Path, default=ROOT / "data" / "dumps")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--no-download", action="store_true", help="only use an already downloaded Quranpedia ZIP")
    args = parser.parse_args(argv)
    ensure_mushaf(args.dumps_dir, allow_download=not args.no_download)
    ensure_ann(args.data_dir)
    print("Prepared only; MUHAWIR_VECTOR_INDEX was not changed in the service environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
