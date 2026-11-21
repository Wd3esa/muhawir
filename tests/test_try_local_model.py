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


def test_speed_line_says_where_the_time_went():
    stats = {"prompt_eval_count": 5000, "prompt_eval_duration": 50_000_000_000,
             "eval_count": 400, "eval_duration": 100_000_000_000}
    line = T.speed(stats)
    assert "read 5000 tok in 50s" in line and "wrote 400 tok at 4.0 tok/s" in line
    assert T.speed({}) == ""


def test_ask_fills_stats_from_ollama():
    class Client:
        def post(self, url, json):
            r = _Reply('{"a": 1}')
            r.json = lambda: {"message": {"content": '{"a": 1}'}, "eval_count": 7, "eval_duration": 2}
            return r

    stats: dict = {}
    T.ask(Client(), "u", "m", "q", stats=stats)
    assert stats["eval_count"] == 7 and stats["prompt_eval_count"] == 0


def test_gold_questions_can_be_picked_by_language():
    gold = [{"id": "a", "field": "f", "question": "q"}, {"id": "b", "field": "f", "lang": "en", "question": "q"}]
    only = lambda lang: [q["id"] for q in T.pick([q for q in gold if q.get("lang", "ar") == lang], 2)]  # noqa: E731
    assert only("ar") == ["a"] and only("en") == ["b"] and only("ur") == []
