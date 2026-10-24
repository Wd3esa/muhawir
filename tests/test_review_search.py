"""Check that the review report preserves candidate evidence and missing labels."""
from muhawir.corpus import Passage, Source
from muhawir.retrieve import Hit
from scripts.review_search import export, questions_from_files


class FakeCorpus:
    def __init__(self):
        self.passages = {
            "answer": Passage("answer", "book", "p. 2", "نص المصدر الصحيح"),
            "other": Passage("other", "book", "p. 1", "نص مختلف"),
        }

    def passage(self, pid):
        return self.passages.get(pid)

    def source_of(self, passage):
        return Source("book", "الكتاب", "وصف")


class FakeRetriever:
    def __init__(self, corpus, ids):
        self.corpus, self.ids = corpus, ids

    def search(self, question, depth):
        return [Hit(self.corpus.passage(pid), 1.0, 1.0) for pid in self.ids[:depth]]


def test_export_compares_old_new_and_flags_stale_gold():
    corpus = FakeCorpus()
    report = export(corpus, FakeRetriever(corpus, ["other", "answer"]),
                    FakeRetriever(corpus, ["answer", "other"]),
                    [{"question": "ما الجواب؟", "gold": ["answer", "missing"]}])
    row = report["results"][0]
    assert (row["old_gold_rank"], row["new_gold_rank"]) == (2, 1)
    assert report["missing_gold_ids"] == ["missing"]
    assert row["new"][0]["source"] == "الكتاب"
    assert row["new"][0]["text"] == "نص المصدر الصحيح"


def test_custom_questions_are_deduplicated_against_gold(tmp_path):
    gold = tmp_path / "gold.json"
    gold.write_text('{"questions": [{"q": "سؤال قديم", "gold": ["answer"]}]}', encoding="utf-8")
    custom = tmp_path / "questions.txt"
    custom.write_text("# تعليق\nسؤال قديم\nسؤال جديد\n", encoding="utf-8")
    assert questions_from_files(gold, custom) == [
        {"question": "سؤال قديم", "gold": ["answer"]},
        {"question": "سؤال جديد", "gold": []},
    ]
