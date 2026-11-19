import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import try_local_model as T  # noqa: E402


class _Reply:
    def __init__(self, content): self.content = content
    def raise_for_status(self): pass
    def json(self): return {"message": {"content": self.content}}


class _Client:
    def __init__(self, content): self.content = content
    def post(self, *a, **k): return _Reply(self.content)


def test_picks_a_few_questions_of_every_field():
    qs = [{"field": f, "id": i} for i, f in enumerate("aabbbc")]
    assert [q["id"] for q in T.pick(qs, 1)] == [0, 2, 5]


def test_invalid_json_and_schema_misfit_are_failures_and_a_full_reply_is_not():
    data, _, error = T.ask(_Client("not json"), "u", "m", "q")
    assert data is None and error
    data, _, error = T.ask(_Client('{"question": "x"}'), "u", "m", "q")
    assert not error and T.fits(data, T.UNDERSTAND_SCHEMA)
    full = {"question": "q", "translate": "", "answer_lang": "ar", "feeling": "", "personal": False, "kind": "what",
            "unsure": False, "reexplain": False, "recall": "", "queries": ["a"],
            "answer_request": {"form": "explanation", "requirements": [], "domain": "", "quoted_text": "",
                               "surah": "", "ayah": 0}}
    data, _, error = T.ask(_Client(json.dumps(full)), "u", "m", "q")
    assert not error and T.fits(data, T.UNDERSTAND_SCHEMA) == ""


def test_options_reach_ollama_and_no_schema_asks_for_plain_json():
    sent = {}

    class Client:
        def post(self, url, json):
            sent.update(json)
            return _Reply('{"a": 1}')

    T.ask(Client(), "u", "m", "q", {"num_ctx": 8192}, schema=False)
    assert sent["options"] == {"temperature": 0, "num_ctx": 8192} and sent["format"] == "json"
    assert sent["think"] is False
    T.ask(Client(), "u", "m", "q")
    assert sent["format"] == T.UNDERSTAND_SCHEMA
