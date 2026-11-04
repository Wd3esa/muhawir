"""«الآية الخامسة من سورة البقرة»: a verse named by its place is shown from the mushaf with no model call."""
from types import SimpleNamespace

from muhawir.pipeline import Muhawir


class _Corpus:
    synthetic = True

    def __init__(self):
        self.p = {"q:2:1": SimpleNamespace(id="q:2:1", location="سورة البقرة، آية 1", text="الم"),
                  "q:2:5": SimpleNamespace(id="q:2:5", location="سورة البقرة، آية 5", text="أولئك على هدى من ربهم"),
                  "q:1:1": SimpleNamespace(id="q:1:1", location="سورة الفاتحة، آية 1", text="بسم الله")}

    def passage(self, pid):
        return self.p.get(pid)


def _find(question):
    return Muhawir._verse_by_reference(SimpleNamespace(corpus=_Corpus()), question)


def test_ordinal_and_number_forms():
    for q in ("الآية الخامسة من سورة البقرة", "آية 5 من البقرة", "اذكر الآية ٥ من سورة البقرة", "نص الآية رقم 5 من سورة البقرة"):
        assert _find(q).id == "q:2:5", q
    assert _find("الآية الأولى من الفاتحة").id == "q:1:1"


def test_other_requests_are_left_alone():
    for q in ("ما تفسير الآية 5 من البقرة", "ما حكم الصلاة", "الآية 99 من البقرة", "الآية الخامسة من سورة الكهف"):
        assert _find(q) is None, q
