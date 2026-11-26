import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import make_check_data as M  # noqa: E402

T = ("لا يجوز بيع الطعام قبل قبضه عند الحنفية إلا إذا كان مما لا يكال ولا يوزن، وهذا قول جمهور أهل العلم. "
     "ثم ذكر الفقهاء أن هذا الحكم يشمل كل ما يكال أو يوزن من الطعام عند أكثرهم في هذا الباب.")


def _passages(n=40):
    return [{"id": f"p{i}", "source": f"book{i % 8}", "kind": "fiqh", "text": T} for i in range(n)]


def test_sentences_are_taken_word_for_word_in_order():
    out = M.sentences(T)
    assert len(out) == 2 and all(s in T for s in out) and T.index(out[0]) < T.index(out[1])


def test_whole_books_are_split_and_test_books_never_appear_in_training():
    data, todo = M.build(_passages(), seed=1)
    books = {part: {r["source"] for r in rows} for part, rows in data.items()}
    assert books["test"] and books["train"] and not (books["test"] & books["train"])
    assert not (books["dev"] & books["train"])
    assert sum(len(v) for v in todo.values()) == sum(r["type"] == "faithful" for rows in data.values() for r in rows)


def test_each_true_pair_has_wrong_ones_of_other_kinds_with_label_zero():
    data, _ = M.build(_passages(), seed=1, negatives=3)
    rows = [r for part in data.values() for r in part]
    assert all(r["label"] == (1 if r["type"] == "faithful" else 0) for r in rows)
    kinds = {r["type"] for r in rows}
    assert {"faithful", "other_passage", "merged", "ruling_flip"} <= kinds
    merged = next(r for r in rows if r["type"] == "merged")
    assert merged["sentence"].count("،") >= 1 and merged["passage"] == T
    other = next(r for r in rows if r["type"] == "other_passage")
    assert other["passage_id"] != other.get("source")  # cited against another passage
