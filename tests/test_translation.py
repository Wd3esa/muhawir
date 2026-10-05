"""The approved English translation: every verse, attached to the Quran passages, kept in the database."""
import json

import pytest

from muhawir import translation
from muhawir.store import SqliteCorpus, build_db


def _file(n=translation.VERSES):
    verses = [{"chapter": 1, "verse": i + 1, "text": f"verse {i + 1}"} for i in range(n)]
    return json.dumps({"quran": verses})


def test_a_short_or_wrong_file_is_refused():
    with pytest.raises(translation.ImportError_):
        translation.load(_file(10))
    with pytest.raises(translation.ImportError_):
        translation.load("{}")


def test_translations_are_stored_beside_the_verses_without_new_passages(tmp_path):
    corpus = {"synthetic": True, "sources": [{"id": "quran", "name": "القرآن", "about": "مصحف."}],
              "passages": [{"id": "q:1:1", "source_id": "quran", "kind": "quran", "location": "الفاتحة 1",
                            "text": "بسم الله الرحمن الرحيم"}]}
    translation.add_to_corpus(corpus, _file())
    assert corpus["_translations"] == {"q:1:1": "verse 1"}  # only the corpus's own verses
    count = build_db(corpus, tmp_path / "t.db")
    db = SqliteCorpus(tmp_path / "t.db")
    assert count == 1 and db.translation("q:1:1") == "verse 1" and db.translation("q:1:2") == ""
    assert "Hilali" in db.translation_name
