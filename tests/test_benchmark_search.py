"""The search benchmark's own arithmetic and plumbing (scripts/benchmark_search.py), with a fake embedding model and a
fake Muhawir: no download, no model. The real numbers come from running it on the server."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import benchmark_search as B  # noqa: E402

PASSAGES = [("2:1-5", "الم ذلك الكتاب لا ريب فيه هدى للمتقين"), ("2:153-157", "استعينوا بالصبر والصلاة"),
            ("39:10-10", "إنما يوفى الصابرون أجرهم بغير حساب"), ("112:1-4", "قل هو الله أحد")]


def test_scores_follow_the_shared_task_definitions():
    qrels = {"q1": {"a", "b"}, "q2": {"c"}, "q3": set()}  # q3 has no answer: left out of the averages
    ranked = {"q1": ["x", "a", "y", "b"], "q2": ["c"], "q3": ["a"]}
    s = B.scores(ranked, qrels)
    assert s["questions"] == 2
    assert s["MRR@10"] == pytest.approx((1 / 2 + 1) / 2)
    assert s["MAP@10"] == pytest.approx(((1 / 2 + 2 / 4) / 2 + 1) / 2)
    assert s["Recall@10"] == pytest.approx(1.0)


def test_rrf_merges_two_rankings_as_muhawir_does():
    first = lambda q, k: ["a", "b", "c"]  # noqa: E731
    second = lambda q, k: ["c", "a"]  # noqa: E731
    assert B.rrf(first, second)("q", 3)[0] == "a"  # high in both


class FakeEncoder:
    """Words as a bag-of-words vector: enough to rank the fake collection."""
    vocab = sorted({w for _, t in PASSAGES for w in t.split()})

    def encode(self, texts, batch_size=16, normalize_embeddings=True):
        out = np.array([[t.split().count(w) for w in self.vocab] for t in texts], dtype=float)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def test_a_dense_option_and_a_reranker_rank_the_collection():
    dense = B.dense_ranker("fake", PASSAGES, encoder=FakeEncoder())
    assert dense("إنما يوفى الصابرون أجرهم", 2)[0] == "39:10-10"
    cross = SimpleNamespace(predict=lambda pairs, batch_size=16: [len(set(q.split()) & set(t.split())) for q, t in pairs])
    rr = B.reranked(lambda q, k: ["2:1-5", "112:1-4", "2:153-157"], PASSAGES, cross=cross)
    assert rr("استعينوا بالصبر", 3)[0] == "2:153-157"


def test_muhawir_hits_are_mapped_to_the_official_passages():
    hits = [SimpleNamespace(passage=SimpleNamespace(id="q:39:10", kind="quran", location="سورة الزمر، الآية 10")),
            SimpleNamespace(passage=SimpleNamespace(id="tafsir-x:2:154:1:1", kind="tafsir", location="سورة البقرة، الآية 154")),
            SimpleNamespace(passage=SimpleNamespace(id="fatwa:1", kind="fatwa", location="فتوى")),
            SimpleNamespace(passage=SimpleNamespace(id="q:39:10", kind="quran", location="سورة الزمر، الآية 10"))]
    engine = SimpleNamespace(retriever=SimpleNamespace(search=lambda q, k: hits), _referenced_verses=lambda t: [])
    assert B.muhawir_ranker(PASSAGES, engine=engine)("سؤال", 10) == ["39:10-10", "2:153-157"]


def test_the_data_files_are_read_in_the_official_formats(tmp_path):
    (tmp_path / "Thematic_QPC").mkdir()
    (tmp_path / "qrels").mkdir()
    (tmp_path / "Thematic_QPC" / "QQA23_TaskA_QPC_v1.1.tsv").write_text("1:1-4\tبسم الله\n1:5-6\tإياك نعبد\n", encoding="utf-8")
    (tmp_path / "QQA23_TaskA_ayatec_v1.2_test.tsv").write_text("500\tسؤال؟\n501\tسؤال آخر؟\n", encoding="utf-8")
    (tmp_path / "qrels" / "QQA23_TaskA_ayatec_v1.2_qrels_test.gold").write_text("500\t0\t1:1-4\t1\n501\t0\t-1\t1\n", encoding="utf-8")
    assert B.load_passages(tmp_path)[0] == ("1:1-4", "بسم الله")
    assert B.load_questions(tmp_path, "test")["501"] == "سؤال آخر؟"
    assert B.load_qrels(tmp_path, "test") == {"500": {"1:1-4"}, "501": set()}
