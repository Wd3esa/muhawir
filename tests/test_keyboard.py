from muhawir import keyboard
from muhawir.pipeline import CHAT, Muhawir
from muhawir.retrieve import Hit


def test_arabic_typed_on_the_english_layout():
    assert keyboard.to_arabic("hgvfh") == "الربا"
    assert keyboard.to_arabic("hg.;hm") == "الزكاة"
    assert keyboard.to_arabic("lh i, hgj,pd]") == "ما هو التوحيد"


def test_only_latin_keys_count():
    assert keyboard.latin_only("hgvfh")
    assert not keyboard.latin_only("الربا")
    assert not keyboard.latin_only("ab")
    assert not keyboard.latin_only("hgvfh 123")


class _Retriever:
    def search(self, q, k=3):
        return [Hit(None, 1.0, 1.0 if q == "الربا" else 0.0)]


class _Corpus:
    synthetic = False


def _engine():
    engine = Muhawir.__new__(Muhawir)
    engine.retriever, engine.corpus = _Retriever(), _Corpus()
    return engine


def test_mistyped_arabic_is_offered_back_with_a_button():
    res = _engine()._respond("hgvfh", "youth", "ar", [])
    assert res.status == CHAT and "«الربا»" in res.message and res.follow_up == "الربا"
