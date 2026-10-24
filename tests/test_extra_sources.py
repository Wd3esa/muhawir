import json
import subprocess
import sys
from pathlib import Path

import pytest

from muhawir.corpus import CorpusError
from muhawir.extra_sources import FILE, append_reviewed
from muhawir.store import SqliteCorpus, SqliteRetriever, build_db


def sample():
    return json.loads((Path(__file__).parents[1] / "data/synthetic_corpus.json").read_text())


def book(sid="extra-sample"):
    return {"id": sid, "name": "كتاب تجريبي", "about": "نص مصطنع للاختبار فقط", "url": "https://example.org/book",
            "edition": "test 1", "rights_holder": "test author", "rights_basis": "permission",
            "rights_document": "test-only permission", "passages": [
                {"location": "الباب الأول، ص 1", "text": "النخلة الكبيرة تثمر تمرًا حلوًا.", "kind": "other"}]}


def write(folder, sources):
    (folder / FILE).write_text(json.dumps({"schema_version": 1, "sources": sources}, ensure_ascii=False))


def test_opt_in_book_is_retrievable_and_keeps_existing_passage_order(tmp_path):
    corpus = sample()
    original = [p["id"] for p in corpus["passages"]]
    write(tmp_path, [book()])
    assert append_reviewed(corpus, tmp_path) == 1
    assert [p["id"] for p in corpus["passages"]][:len(original)] == original
    assert corpus["passages"][-1]["id"] == "extra-sample:1"
    assert corpus["_provenance"]["extra_sources"]["sources"][0]["rights_basis"] == "permission"
    db = tmp_path / "sample.db"
    build_db(corpus, db)
    stored = SqliteCorpus(db)
    assert stored.passage("extra-sample:1").location == "الباب الأول، ص 1"
    assert any(h.passage.id == "extra-sample:1" for h in SqliteRetriever(stored).search("النخلة الكبيرة", k=10))


@pytest.mark.parametrize("change", [
    lambda b: b.pop("rights_document"),
    lambda b: b.update(url="http://example.org/book"),
    lambda b: b["passages"][0].update(kind="quran"),
    lambda b: b["passages"][0].update(kind="hadith"),
    lambda b: b.update(id="test-a"),
])
def test_invalid_book_fails_without_partial_corpus(tmp_path, change):
    corpus = sample()
    before = json.dumps(corpus, sort_keys=True)
    invalid = book("extra-invalid")
    change(invalid)
    write(tmp_path, [book(), invalid])
    with pytest.raises(CorpusError):
        append_reviewed(corpus, tmp_path)
    assert json.dumps(corpus, sort_keys=True) == before


def test_no_manifest_leaves_existing_build_unchanged(tmp_path):
    corpus = sample()
    assert append_reviewed(corpus, tmp_path) == 0
    assert "extra_sources" not in corpus.get("_provenance", {})


def test_fast_validation_needs_no_real_corpus(tmp_path):
    write(tmp_path, [book()])
    result = subprocess.run([sys.executable, "-m", "muhawir.extra_sources", str(tmp_path)],
                            text=True, capture_output=True, check=True)
    assert "1 sources, 1 passages" in result.stdout
