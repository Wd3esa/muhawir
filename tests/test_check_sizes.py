"""The support check for a local model: exact list sizes; «missing» terms always reject (the safe failure)."""
import json

import pytest

from muhawir import generate
from muhawir.corpus import Passage
from muhawir.generate import ModelGenerator, sized_check_schema
from muhawir.verify import Claim

P = Passage(id="p1", source_id="s", location="باب", kind="fiqh",
            text="ثم قال النبي صلى الله عليه وسلم إنما الأعمال بالنيات وإنما لكل امرئ ما نوى")
pytestmark = [pytest.mark.real_check, pytest.mark.raw_check_reply]  # the real check, the reply as sent

CLAIM = Claim("أخبر النبي ﷺ أن قبول الأعمال مرتبط بالنية التي يقصدها صاحبها", ("p1",))


def _reply(missing):
    return json.dumps({"on_topic": [True], "missing": [missing], "evidence": ["إنما الأعمال بالنيات"],
                       "problem": ["none"]}, ensure_ascii=False)


def test_sized_schema_asks_for_exactly_one_item_per_sentence():
    s = sized_check_schema(3)
    for field in ("on_topic", "missing", "evidence", "problem"):
        assert s["properties"][field]["minItems"] == 3 == s["properties"][field]["maxItems"]
    assert s["properties"]["missing"]["items"]["maxItems"] == 5
    assert "minItems" not in generate.CHECK_SCHEMA["properties"]["problem"]  # the shared schema is unchanged


def test_only_the_local_model_gets_the_sized_schema():
    seen = {}

    def call_for(name):
        def call(system, user, schema=None):
            seen[name] = schema
            raise RuntimeError("try the next")
        return call

    gen = ModelGenerator([("gemini", call_for("gemini")), ("local", call_for("local"))])
    gen.check_support([CLAIM], {"p1": P}, "")
    assert seen["gemini"] is generate.CHECK_SCHEMA
    assert seen["local"]["properties"]["problem"]["maxItems"] == 1


def test_a_missing_term_rejects_even_when_the_passage_has_it_elsewhere():
    """Safety first: the checker may name the changed detail as «missing»; it is not second-guessed by code."""
    gen = ModelGenerator([("local", lambda system, user, schema=None: _reply(["ثم"]))])
    assert gen.check_support([CLAIM], {"p1": P}, "") == [False]


def test_a_missing_term_the_passage_lacks_still_rejects():
    gen = ModelGenerator([("local", lambda system, user, schema=None: _reply(["الصيام"]))])
    assert gen.check_support([CLAIM], {"p1": P}, "") == [False]
    assert "الصيام" in gen.last_check_reasons[0]
