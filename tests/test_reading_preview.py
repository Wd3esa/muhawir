"""Reading-preview: while the answer is written, the page is sent the first passages being read (source text only,
cut at a word, with their references), before «writing», and never the model's draft."""
import json

from muhawir import pipeline
from muhawir.corpus import parse_corpus
from muhawir.generate import ModelGenerator
from muhawir.pipeline import Muhawir, preview_text

_TEXT = "الوضوء غسل الوجه واليدين ومسح الرأس وغسل الرجلين وهو شرط لصحة الصلاة عند أهل العلم كلهم بلا خلاف بينهم في ذلك."
CORPUS = parse_corpus({"synthetic": True, "sources": [{"id": "fiqh-x", "name": "كتاب فقه", "about": "فقه"}],
                       "passages": [{"id": f"f:{i}", "source_id": "fiqh-x", "location": f"باب {i}", "kind": "fiqh",
                                     "text": _TEXT} for i in range(5)]})


def test_preview_text_cuts_at_a_word_and_marks_the_cut():
    long = " ".join(["كلمة"] * 200)
    out = preview_text(long, 50)
    assert out.endswith("…") and len(out) <= 51 and not out[:-1].endswith(" ") and "كلم…" not in out
    assert preview_text("نص قصير") == "نص قصير"


def test_the_first_passages_are_sent_before_writing_and_no_draft_is_sent():
    draft = "جملة الكاتب التي لم تفحص بعد"

    def call(system, user, schema=None):
        if '"question"' in json.dumps(schema or {}):
            return json.dumps({"question": "كيف أتوضأ؟", "translate": "", "answer_lang": "", "feeling": "",
                               "personal": False, "kind": "how", "reexplain": False, "recall": "",
                               "queries": ["صفة الوضوء"],
                               "answer_request": {"form": "explanation", "requirements": [], "domain": "fiqh"}},
                              ensure_ascii=False)
        if '"claims"' in json.dumps(schema or {}):
            return json.dumps({"plan": "", "abstain": False, "as_list": False, "follow_up": "", "views": [],
                               "claims": [{"section": "", "label": "", "text": draft, "passage_ids": ["f:0"]}]},
                              ensure_ascii=False)
        return "{}"

    engine = Muhawir(CORPUS, ModelGenerator([("m", call)]))
    events = []
    token = pipeline._progress.set(lambda stage, data=None: events.append((stage, data)))
    try:
        engine.ask("كيف اتوضأ")
    finally:
        pipeline._progress.reset(token)
    stages = [s for s, _ in events]
    assert "reading" in stages and stages.index("reading") < stages.index("writing")
    shown = dict(events)["reading"]
    assert 1 <= len(shown) <= pipeline.READING_PREVIEW
    assert shown[0]["source"] == "كتاب فقه" and shown[0]["location"].startswith("باب") and shown[0]["text"] == _TEXT
    assert draft not in json.dumps(events, ensure_ascii=False)  # the unchecked draft never reaches the page early
