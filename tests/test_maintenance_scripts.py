"""Maintenance tools must preserve the cache unless explicitly told to modify it."""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forget_normalizes_and_never_deletes_without_flag(tmp_path, monkeypatch, capsys):
    tool = load_script("forget")
    cache = tmp_path / "answers.db"
    con = sqlite3.connect(cache)
    con.execute("CREATE TABLE replies (key TEXT PRIMARY KEY)")
    keys = [json.dumps(("ما حكم الصلاة", "youth", "ar"), ensure_ascii=False),
            json.dumps(("ما حكم الصيام", "youth", "ar"), ensure_ascii=False)]
    con.executemany("INSERT INTO replies VALUES (?)", [(key,) for key in keys])
    con.commit()
    monkeypatch.setenv("MUHAWIR_STORE", str(cache))
    monkeypatch.setattr(sys, "argv", ["forget.py", "حُكْم الصَّلَاة"])
    tool.main()
    assert "1 matching" in capsys.readouterr().out
    assert con.execute("SELECT COUNT(*) FROM replies").fetchone()[0] == 2
    monkeypatch.setattr(sys, "argv", ["forget.py", "حُكْم الصَّلَاة", "--delete"])
    tool.main()
    assert con.execute("SELECT key FROM replies").fetchone()[0] == keys[1]
    con.close()


def test_nightly_never_retries_without_explicit_flag(tmp_path, monkeypatch):
    tool = load_script("nightly")
    cache = tmp_path / "answers.db"
    con = sqlite3.connect(cache)
    con.execute("CREATE TABLE replies (at REAL, key TEXT, reply TEXT)")
    con.execute("INSERT INTO replies VALUES (?, ?, ?)",
                (tool.time.time(), json.dumps(("لم أجد النص", "youth", "ar"), ensure_ascii=False),
                 json.dumps({"status": "abstained"})))
    con.commit()
    con.close()
    monkeypatch.setenv("MUHAWIR_STORE", str(cache))
    monkeypatch.setenv("MUHAWIR_DB", str(tmp_path / "absent.db"))
    monkeypatch.setenv("MUHAWIR_URL", "http://127.0.0.1:8000")
    monkeypatch.setattr(tool, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(tool, "get", lambda *_: {})
    monkeypatch.setattr(tool, "run_tests", lambda: "skipped")
    monkeypatch.setattr(tool.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="self-test skipped"))
    monkeypatch.setattr(tool, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("paid retry")))
    monkeypatch.setattr(sys, "argv", ["nightly.py"])
    assert tool.main() == 0
    assert "diagnosis only" in (tool.REPORTS / "latest.txt").read_text()
