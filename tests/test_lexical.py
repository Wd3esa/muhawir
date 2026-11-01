"""A question about the meaning of a word (live case 1 Nov 2026: «ما معنى ألفوا آباءهم» was answered from the entry
«أوى»). The entries below are short pieces of «المفردات» (al-Raghib), written here so that the test needs no download."""
import pytest

from muhawir import lexical
from muhawir.corpus import Passage
from muhawir.store import SqliteCorpus, SqliteRetriever, build_db

LAFA = "ألفيت: وجدت. قال الله: قالوا بل نتبع ما ألفينا عليه آباءنا [البقرة/ 170] ، وألفيا سيدها لدى الباب [يوسف/ 25]"
AWA = "المأوى مصدر أوى يأوي أويا ومأوى، تقول: أوى إلى كذا: انضم إليه يأوي أوي، وآويته: ضممته إليه قال تعالى: آوى إليه أخاه"
TAHR = "وقيل: إن ذلك اقتضى التطهير من حيث المعنى، وذلك أن الطاهر ضربان: ضرب لا يتغير ولا يستحيل ببقاء المعنى"
VERSE = "إنهم ألفوا آباءهم ضالين فهم على آثارهم يهرعون"


def _entry(pid, root, text, kind="gharib"):
    return {"id": pid, "source_id": "s", "kind": kind, "location": f"مادة «{root}»", "text": text, "keywords": root}


@pytest.fixture()
def retriever(tmp_path):
    passages = [_entry("g:1", "لفى", LAFA), _entry("g:2", "أوى", AWA), _entry("g:3", "طهر", TAHR),
                _entry("q:37:69", "", VERSE, kind="quran")]
    build_db({"sources": [{"id": "s", "name": "المفردات", "about": "معجم", "url": ""}],
              "passages": passages, "synthetic": False}, tmp_path / "c.db")
    return SqliteRetriever(SqliteCorpus(tmp_path / "c.db"))


def passage(pid, root, text, kind="gharib"):
    return Passage(pid, "s", root, text, kind, keywords=root)


def test_alignment_separates_a_form_of_the_word_from_another_word():
    asked = "الفوا"                                   # «ألفوا» as normalized
    assert lexical.alignment(asked, "ألفيت".replace("أ", "ا")) >= 0.55
    assert lexical.alignment(asked, "اوي") < 0.5      # «آوى»: the entry the live answer came from


def test_root_candidates_reach_the_heading_of_a_form_the_book_does_not_spell():
    index = lexical.RootIndex(["لفى", "أوى", "فكر", "كبر", "طهر"])
    assert "لفى" in index.candidates("ألفوا")
    assert "فكر" in index.candidates("يتفكرون")
    assert "كبر" in index.candidates("استكبروا")
    assert "أوى" not in index.candidates("ألفوا")


def test_question_words_do_not_count_as_covered(retriever):
    hits = retriever.search("ما معنى بلاغوزا", k=5)
    assert all(h.coverage < 0.34 for h in hits)       # «معنى» alone (it is in the entry «طهر») is not coverage


def test_the_entry_of_the_root_is_found_for_a_form_it_does_not_spell(retriever):
    hits = retriever.search("ما معنى ألفوا", k=5)
    assert hits[0].passage.keywords == "لفى" and hits[0].coverage == 1.0
    again = retriever.search("ما معنى الفوا اباءهم", k=5)     # as typed live, without hamza
    assert again[0].passage.keywords == "لفى"


def test_the_root_lookup_is_only_for_a_question_about_a_meaning(retriever):
    assert not any(h.passage.keywords == "لفى" for h in retriever.search("ألفوا آباءهم", k=5)
                   if h.score >= 100)


def test_gate_rejects_the_entry_of_another_word_and_keeps_the_right_one(retriever):
    question = "ما معنى الفوا اباءهم"
    assert retriever.df("فوا") <= lexical.RARE_DF
    assert not lexical.keeps(passage("g:2", "أوى", AWA), question, retriever.df, heading_alignment=0.6)
    assert lexical.keeps(passage("g:1", "لفى", LAFA), question, retriever.df, heading_alignment=0.6)


def test_gate_leaves_other_kinds_and_common_words_alone(retriever):
    question = "ما معنى الفوا اباءهم"
    assert lexical.keeps(passage("q:1", "", AWA, kind="quran"), question, retriever.df)
    assert lexical.keeps(passage("g:2", "أوى", AWA), "ما معنى الاوى", None)          # no df: nothing is rejected
    assert lexical.keeps(passage("g:3", "طهر", TAHR), "ما معنى المعنى", retriever.df)  # no rare word asked


def test_gate_rejects_a_heading_found_for_a_word_that_is_not_arabic(retriever):
    # a stemmer finds a «root» in any string: the heading must be backed by a word of the entry that is close to it
    entry = passage("g:9", "بتر", "البتر: قطع الشيء قبل تمامه، ومنه الأبتر")
    assert not lexical.keeps(entry, "ما معنى كمبيوتر", retriever.df, heading_alignment=0.6)


def test_existing_behaviour_of_other_questions_is_unchanged(retriever):
    hits = retriever.search("التطهير", k=3)
    assert hits and hits[0].passage.keywords == "طهر"


def test_pipeline_judges_the_users_words_not_the_models_rewrite(retriever):
    """Live case: the question was rewritten around «أوى» and its search phrase too; the entry of «أوى» must not be
    offered for what the user typed («ألفوا»)."""
    from muhawir.generate import ExtractiveGenerator
    from muhawir.pipeline import Muhawir
    engine = Muhawir(retriever.corpus, ExtractiveGenerator(), retriever)
    passages, _ = engine._gather("ما معنى أوى إلى في اللغة", "ما معنى الفوا اباءهم", ["معنى أوى إلى"])
    roots = [p.keywords for p in passages]
    assert "لفى" in roots
    assert "أوى" not in roots


def test_a_rewrite_that_lost_the_word_asked_is_not_trusted(retriever):
    user = "ما معنى الفوا اباءهم"
    assert lexical.loses_asked_word(user, ["ما معنى أوى إلى في اللغة", "معنى أوى إلى"], retriever.df)
    assert not lexical.loses_asked_word(user, ["ما معنى ألفوا آباءهم"], retriever.df)     # the same words
    assert not lexical.loses_asked_word("ما معنى الصرات", ["ما معنى الصراط"], retriever.df)  # a spelling correction
    # not a question about a meaning: the rewrite is the point of the understanding step (a dialect word in plain words)
    assert not lexical.loses_asked_word("ايش حكم البيتكوين", ["ما حكم العملات الرقمية"], retriever.df)


def test_whole_pipeline_with_a_model_that_rewrites_the_asked_word(retriever):
    """The live path (a model that writes, `_gather`), not the offline one: the model rewrote «ألفوا» into «أوى»."""
    from muhawir.generate import ExtractiveGenerator
    from muhawir.pipeline import Muhawir

    class Drifting(ExtractiveGenerator):
        strict_retrieval = False

        def understand(self, message, history):
            return {"question": "ما معنى أوى إلى في اللغة", "queries": ["معنى أوى إلى"], "lang": "", "translate": "",
                    "answer_request": {}, "reexplain": False, "kind": "", "feeling": "", "personal": False,
                    "unsure": False}

    engine = Muhawir(retriever.corpus, Drifting(), retriever)
    res = engine.ask("ما معنى ألفوا آباءهم")
    shown = " ".join(c["text"] for c in res.claims)
    assert "ألفيت: وجدت" in shown          # the entry of «لفى»
    assert "المأوى" not in shown            # not the entry of «أوى»


def test_asks_new_word_separates_a_new_word_from_a_request_to_explain_again():
    from muhawir.lexical import asks_new_word
    hist = "شجرة الخلد وملك لا يبلى. سياق الآية: هذه الآية تحكي وسوسة الشيطان لآدم"
    for msg in ("معنى الفو ابائهم", "ما معنى ألفوا آباءهم", "ما المقصود بكلمة ألفوا"):
        assert asks_new_word(msg, hist), msg
    for msg in ("لم أفهم", "اشرح لي بطريقة أبسط", "ما معنى هذا الكلام؟", "ما معنى شجرة الخلد؟",
                "ايش يعني هذا؟", "شو المقصود بهذا", "ما المراد بهذا الكلام"):
        assert not asks_new_word(msg, hist), msg


def test_strip_ids_keeps_verse_numbers_but_drops_hadith_numbers():
    from muhawir.pipeline import _strip_ids
    assert "٣٥" in _strip_ids("سورة البقرة، الآية رقم ٣٥", set())
    assert "7296" not in _strip_ids("رواه البخاري برقم 7296", set())


def test_every_element_the_page_script_looks_up_exists_in_the_page():
    """A lookup of an id that was removed from the HTML throws while the page starts, and nothing after it is wired
    (the theme switch and the assistant-connection dialog stopped answering that way)."""
    import re
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "muhawir" / "static" / "index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    used = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', html))
    assert not used - ids, sorted(used - ids)
