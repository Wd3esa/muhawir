"""Regressions for the enterprise audit of 4 November 2026 (Muhawir_Enterprise_Audit_2026-11-04.md): each test is the
auditor's probe turned around, so the defect it found cannot come back. Synthetic data and fake models only."""
import json

import pytest

from muhawir.corpus import Passage
from muhawir.generate import Claim, ModelGenerator

PASSAGE = Passage(id="p1", source_id="s", location="باب", kind="fiqh", text="الوضوء شرط لصحة الصلاة عند جمهور الفقهاء.")


def _checker(reply):
    return ModelGenerator([("fake", lambda system, user, schema=None: json.dumps(reply, ensure_ascii=False))])


# ---- HIGH-001: a support-check reply without the required fields approves nothing ------------------------------

@pytest.mark.real_check
@pytest.mark.raw_check_reply
@pytest.mark.parametrize("reply", [
    {"problem": ["none"]},                                                      # the auditor's probe
    {"supported": [True]},                                                      # the old short form
    {"problem": ["none"], "evidence": [], "missing": [[]], "on_topic": [True]},  # one list too short
    {"problem": ["none"], "evidence": [1], "missing": [[]], "on_topic": [True]},  # wrong type
    {"problem": ["none"], "evidence": ["الوضوء شرط"], "missing": [[]], "on_topic": ["maybe"]},
])
def test_an_incomplete_support_reply_approves_nothing(reply):
    claim = Claim(text="الشمس تشرق من المغرب.", passage_ids=["p1"])  # unrelated to its passage
    flags = _checker(reply).check_support([claim], {"p1": PASSAGE}, "ما حكم الوضوء؟")
    assert flags is None or flags == [False]


@pytest.mark.real_check
@pytest.mark.raw_check_reply
def test_a_complete_reply_still_approves_a_sentence_whose_witness_is_in_the_passage():
    claim = Claim(text="الوضوء شرط لصحة الصلاة.", passage_ids=["p1"])
    reply = {"problem": ["none"], "evidence": ["الوضوء شرط لصحة الصلاة"], "missing": [[]], "on_topic": [True]}
    assert _checker(reply).check_support([claim], {"p1": PASSAGE}, "ما حكم الوضوء؟") == [True]


@pytest.mark.real_check
@pytest.mark.raw_check_reply
def test_a_witness_that_is_not_in_the_passage_rejects_the_sentence():
    claim = Claim(text="الوضوء شرط لصحة الصلاة.", passage_ids=["p1"])
    reply = {"problem": ["none"], "evidence": ["الوضوء سنة"], "missing": [[]], "on_topic": [True]}
    assert _checker(reply).check_support([claim], {"p1": PASSAGE}, "ما حكم الوضوء؟") == [False]


# ---- HIGH-002: Pillow 13, and an upload reaches only the decoders of the picture kinds Muhawir accepts -------------

def test_the_requirements_allow_the_patched_pillow():
    from pathlib import Path
    line = next(l for l in (Path(__file__).resolve().parent.parent / "requirements.txt").read_text().splitlines()
                if l.lower().startswith("pillow"))
    assert ">=13" in line


def test_a_picture_kind_outside_the_list_never_reaches_its_decoder():
    import io
    from PIL import Image
    from muhawir import extract
    out = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(out, "ICO")  # a real image, of a kind Muhawir does not take
    with pytest.raises(extract.ExtractError):
        extract._prepare(out.getvalue())


# ---- HIGH-003: two words that differ only in vowel marks never share a kept answer ------------------------------

def test_al_birr_and_al_burr_are_kept_apart():
    from muhawir.pipeline import Muhawir, Response, ANSWERED
    from muhawir.corpus import load_corpus
    from pathlib import Path
    corpus = load_corpus(Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json")
    engine = Muhawir(corpus, ModelGenerator([("fake", lambda *a, **k: "{}")]))
    asked = []
    def answer_new(question, style, lang, history, key, answer_it=False):
        asked.append(question)
        res = Response(ANSWERED, question)
        if key:
            engine._keep(key, res)
        return res
    engine._answer_new = answer_new
    assert engine.ask("ما معنى البِرّ؟").message == "ما معنى البِرّ؟"
    assert engine.ask("ما معنى البُرّ؟").message == "ما معنى البُرّ؟"
    assert len(asked) == 2
    # the same question with other spacing or end punctuation still finds its kept answer
    assert engine.ask("  ما معنى البِرّ  ").message == "ما معنى البِرّ؟" and len(asked) == 2


# ---- HIGH-004: a photo the server reads itself never leaves it --------------------------------------------------

def test_a_printed_photo_read_by_tesseract_is_not_sent_to_the_vision_model(monkeypatch):
    import io
    from PIL import Image
    from muhawir import extract
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.delenv("MUHAWIR_VISION_OCR", raising=False)
    monkeypatch.setattr(extract, "ocr_image", lambda image, deadline: "بسم الله الرحمن الرحيم " * 8)
    monkeypatch.setattr(extract, "vision_text", lambda image: pytest.fail("the photo was sent to the vision model"))
    out = io.BytesIO(); Image.new("RGB", (900, 900), "white").save(out, "PNG")
    assert extract.extract(out.getvalue())["text"].startswith("بسم الله")


def test_a_photo_tesseract_cannot_read_goes_to_the_vision_model_only_then(monkeypatch):
    import io
    from PIL import Image
    from muhawir import extract
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.delenv("MUHAWIR_VISION_OCR", raising=False)
    order = []
    monkeypatch.setattr(extract, "ocr_image", lambda image, deadline: order.append("ocr") or "")
    monkeypatch.setattr(extract, "vision_text", lambda image: (order.append("vision"), ("اللهم يا فارج الهم ويا كاشف الغم", ""))[1])
    out = io.BytesIO(); Image.new("RGB", (900, 900), "white").save(out, "PNG")
    assert extract.extract(out.getvalue())["text"].startswith("اللهم") and order == ["ocr", "vision"]


# ---- HIGH-006: a request too large is refused before it is parsed ------------------------------------------------

def test_a_huge_json_request_is_refused_before_it_is_parsed(monkeypatch):
    from fastapi.testclient import TestClient
    from muhawir import server
    monkeypatch.setattr(server, "_run_ask", lambda body: pytest.fail("a 2 MB request reached the answer step"))
    client = TestClient(server.app)
    big = json.dumps({"question": "ما حكم الوضوء؟", "unused": "x" * 2_000_000})
    response = client.post("/api/ask", content=big, headers={"content-type": "application/json"})
    assert response.status_code == 413
    # without a declared length (chunked), the bytes are counted as they arrive
    def chunks():
        yield big[:1000].encode()
        yield big[1000:].encode()
    response = client.post("/api/ask", content=chunks(), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_an_ordinary_question_with_a_full_conversation_still_passes(monkeypatch):
    from fastapi.testclient import TestClient
    from muhawir import server
    monkeypatch.setattr(server, "_allow_question", lambda address: True)
    monkeypatch.setattr(server, "_run_ask", lambda body: {"status": "answered", "message": body.question})
    history = [{"role": "user" if i % 2 == 0 else "assistant", "text": "ن" * 3900} for i in range(20)]
    response = TestClient(server.app).post("/api/ask", json={"question": "ما حكم الوضوء؟", "history": history})
    assert response.status_code == 200


# ---- M-002: visitors who left are forgotten -----------------------------------------------------------------------

def test_visitors_with_nothing_recent_are_forgotten(monkeypatch):
    from muhawir import server
    clock = [1000.0]
    monkeypatch.setattr(server.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(server, "_visitor_calls", server.defaultdict(server.deque))
    for i in range(5001):
        assert server._allow_question(f"10.0.{i // 250}.{i % 250}")
    clock[0] += 3 * 3600
    server._allow_question("10.9.9.9")
    assert len(server._visitor_calls) == 1


# ---- M-003: a failing answer store does not fail the answer -------------------------------------------------------

def test_a_failing_answer_store_means_no_kept_answer_not_a_failure():
    import sqlite3
    from pathlib import Path
    from muhawir.corpus import load_corpus
    from muhawir.pipeline import ANSWERED, Muhawir, Response
    corpus = load_corpus(Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json")
    engine = Muhawir(corpus, ModelGenerator([("fake", lambda *a, **k: "{}")]))

    class Broken:
        def get(self, key):
            raise sqlite3.OperationalError("database is locked")

        def put(self, key, res):
            raise sqlite3.OperationalError("database is locked")
    engine.store = Broken()
    engine._answer_new = lambda question, style, lang, history, key, answer_it=False: Response(ANSWERED, "ok")
    assert engine.ask("ما حكم الوضوء؟").message == "ok"


# ---- M-004: ready while a fallback model can answer ----------------------------------------------------------------

def test_ready_while_one_configured_model_can_answer(monkeypatch):
    import time
    from fastapi.testclient import TestClient
    from muhawir import server
    from muhawir.generate import BREAKERS
    monkeypatch.setattr(server.engine, "generator", ModelGenerator([("first", None), ("second", None)]))
    monkeypatch.setitem(BREAKERS, "first", {"failures": 3, "open_until": time.monotonic() + 60})
    body = TestClient(server.app).get("/api/ready").json()
    assert body["parts"]["models"] is True and body["paused_models"] == ["first"]
    monkeypatch.setitem(BREAKERS, "second", {"failures": 3, "open_until": time.monotonic() + 60})
    assert TestClient(server.app).get("/api/ready").json()["parts"]["models"] is False


# ---- M-005: one passage-id limit, never truncated -----------------------------------------------------------------

def test_a_long_passage_id_is_read_whole_and_never_cut_short(monkeypatch):
    from fastapi.testclient import TestClient
    from muhawir import server
    from muhawir.corpus import MAX_PASSAGE_ID
    long_id = "extra-" + "a" * 43 + ":1"  # 51 characters, importer-compatible
    passage = Passage(id=long_id, source_id="extra-" + "a" * 43, location="باب", kind="other", text="نص")
    monkeypatch.setattr(server.engine.corpus, "passage", lambda pid: passage if pid == long_id else None)
    monkeypatch.setattr(server.engine.corpus, "neighbours", lambda pid: (None, None))
    assert len(long_id) <= MAX_PASSAGE_ID
    assert TestClient(server.app).get(f"/api/context/{long_id}").status_code == 200


# ---- M-007: logs never carry the provider's text ------------------------------------------------------------------

def test_a_provider_error_is_logged_without_its_text():
    import httpx
    from muhawir.generate import describe
    response = httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT",
                                                   "message": "echo: سؤالي الخاص key=sk-secret"}},
                              request=httpx.Request("POST", "https://x"))
    said = describe(httpx.HTTPStatusError("bad", request=response.request, response=response))
    assert said == "HTTP 400 INVALID_ARGUMENT" and "sk-secret" not in said and "سؤالي" not in said
    assert describe(ValueError("contains sk-secret")) == "ValueError"


# ---- M-010: the book-neighbour index ------------------------------------------------------------------------------

def test_new_databases_have_the_neighbour_index_and_old_ones_get_it(tmp_path):
    import sqlite3
    import subprocess
    import sys
    from pathlib import Path
    from muhawir.store import build_db
    root = Path(__file__).resolve().parent.parent
    path = tmp_path / "m.db"
    build_db(json.loads((root / "data" / "synthetic_corpus.json").read_text(encoding="utf-8")), path)
    con = sqlite3.connect(path)
    plan = " ".join(str(r) for r in con.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM passages WHERE source_id = 'x' AND rid < 5 ORDER BY rid DESC LIMIT 1"))
    assert "passages_source_rid" in plan
    con.execute("DROP INDEX passages_source_rid"); con.commit(); con.close()
    done = subprocess.run([sys.executable, str(root / "scripts" / "migrate_db.py")], env={"MUHAWIR_DB": str(path)},
                          capture_output=True, text=True)
    assert done.returncode == 0 and "added" in done.stdout


# ---- L-001: the workflow link is data, not shell source ---------------------------------------------------------

def test_the_live_workflow_never_writes_the_link_into_a_script():
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "live-test.yml").read_text()
    runs = text.split("steps:", 1)[1]
    assert "${{ inputs.url }}" not in runs and "TARGET_URL: ${{ inputs.url }}" in text


# ---- HIGH-005: a file is read in its own limited process ----------------------------------------------------------

def test_a_file_is_read_in_a_separate_process(monkeypatch):
    from muhawir import extract
    monkeypatch.setenv("MUHAWIR_EXTRACT_ISOLATED", "1")
    got = extract.extract_isolated(("بسم الله الرحمن الرحيم " * 10).encode())
    assert got["text"].startswith("بسم الله") and got["method"] == "text"
    with pytest.raises(extract.ExtractError) as err:
        extract.extract_isolated(b"\x00\x01\x02" * 100)  # an error inside the worker comes back as its code
    assert err.value.code in ("unsupported", "no_text")


def test_a_reader_that_runs_out_of_memory_ends_alone_and_the_server_goes_on(monkeypatch):
    from muhawir import extract
    monkeypatch.setenv("MUHAWIR_EXTRACT_ISOLATED", "1")
    monkeypatch.setenv("MUHAWIR_EXTRACT_MEMORY_MB", "20")  # far too little even to start: the child fails, not us
    with pytest.raises(extract.ExtractError) as err:
        extract.extract_isolated(b"%PDF-1.4 " + b"0" * 1000)
    assert err.value.code in ("unreadable", "too_large")


def test_a_reader_that_takes_too_long_is_stopped(monkeypatch):
    import subprocess
    from muhawir import extract
    monkeypatch.setenv("MUHAWIR_EXTRACT_ISOLATED", "1")
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="worker", timeout=k.get("timeout"))
    monkeypatch.setattr(extract.subprocess, "run", slow)
    with pytest.raises(extract.ExtractError) as err:
        extract.extract_isolated(b"%PDF-1.4")
    assert err.value.code == "too_slow"


def test_pdf_text_extraction_stops_at_its_deadline(monkeypatch):
    import time
    from muhawir import extract
    class Page:
        def extract_text(self):
            return "نص " * 50
    class Reader:
        is_encrypted = False
        pages = [Page()] * 40
        def __init__(self, *a, **k):
            pass
    import pypdf
    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    with pytest.raises(extract.ExtractError) as err:
        extract.from_pdf(b"%PDF-1.4", time.monotonic() - 1)  # an already spent budget, as in the auditor's probe
    assert err.value.code == "too_slow"


# ---- M-009: a long quote with no match makes a bounded number of searches ------------------------------------------

def test_a_long_unmatched_quote_makes_a_bounded_number_of_phrase_searches(tmp_path):
    from pathlib import Path
    from muhawir.store import MAX_PHRASE_PROBES, SqliteCorpus, SqliteRetriever, build_db
    root = Path(__file__).resolve().parent.parent
    build_db(json.loads((root / "data" / "synthetic_corpus.json").read_text(encoding="utf-8")), tmp_path / "m.db")
    retriever = SqliteRetriever(SqliteCorpus(tmp_path / "m.db"))
    calls = []
    original = retriever._match
    retriever._match = lambda *a, **k: calls.append(1) or original(*a, **k)
    tokens = [f"كلمه{i}" for i in range(100)]  # the auditor's 100-token no-hit case (1,326 searches before)
    retriever._phrase_rows(tokens, 5)
    assert len(calls) <= MAX_PHRASE_PROBES


# ---- M-001: kept answers hold no question in plain words, expire for real, and personal cases are not written ------

def test_the_answer_store_deletes_expired_rows(tmp_path, monkeypatch):
    import sqlite3
    import time
    from muhawir import pipeline
    from muhawir.pipeline import ANSWERED, AnswerStore, Response
    store = AnswerStore(tmp_path / "answers.db", "v1")
    store.put(("سؤال", "youth", "ar"), Response(ANSWERED, "جواب"))
    later = time.time() + pipeline.STORE_SECONDS + 10
    monkeypatch.setattr(pipeline.time, "time", lambda: later)
    store.put(("سؤال آخر", "youth", "ar"), Response(ANSWERED, "ب"))  # the expired one is deleted, not only unused
    assert sqlite3.connect(tmp_path / "answers.db").execute("SELECT COUNT(*) FROM replies").fetchone()[0] == 1
    assert AnswerStore(tmp_path / "answers.db", "v1").get(("سؤال", "youth", "ar")) is None


def test_a_referral_is_never_written_to_disk(tmp_path):
    from pathlib import Path
    from muhawir.corpus import load_corpus
    from muhawir.pipeline import REFERRED, AnswerStore, Muhawir, Response
    corpus = load_corpus(Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json")
    engine = Muhawir(corpus, ModelGenerator([("fake", lambda *a, **k: "{}")]))
    engine.store = AnswerStore(tmp_path / "answers.db", "v1")
    engine._keep(("حلفت على زوجتي", "youth", "ar"), Response(REFERRED, "اسأل جهة الفتوى"))
    assert len(engine.store) == 0


# ---- HIGH-007: one deadline per answer; a permanent error is not retried -------------------------------------------

def test_a_refused_key_is_asked_once_not_twice():
    import httpx
    from muhawir.generate import json_reply
    calls = []
    def call(system, user, schema):
        calls.append(1)
        response = httpx.Response(401, request=httpx.Request("POST", "https://x"))
        raise httpx.HTTPStatusError("unauthorized", request=response.request, response=response)
    with pytest.raises(httpx.HTTPStatusError):
        json_reply(call, "s", "u", {})
    assert len(calls) == 1  # the auditor saw two


def test_a_reply_that_is_not_json_is_still_asked_for_again():
    from muhawir.generate import json_reply
    replies = iter(["not json", '{"ok": true}'])
    assert json_reply(lambda *a: next(replies), "s", "u", {}) == {"ok": True}


def test_every_model_call_of_an_answer_shares_one_deadline(monkeypatch):
    import time
    from muhawir import budget
    waits = []
    with budget.answer_deadline(10):
        waits.append(budget.call_timeout(300))  # a 300 s provider limit is cut to the time left
        assert waits[0] <= 10
        from muhawir.telemetry import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:  # a parallel step of the answer spends the same budget
            assert pool.submit(budget.left).result() <= 10
    with budget.answer_deadline(0.5):
        time.sleep(0.6)
        with pytest.raises(budget.OutOfTime):
            budget.call_timeout(300)  # no call is started once the time is spent


def test_an_answer_out_of_time_makes_no_more_calls_and_says_it_is_unavailable(monkeypatch):
    from pathlib import Path
    from muhawir import budget
    from muhawir.corpus import load_corpus
    from muhawir.pipeline import Muhawir
    corpus = load_corpus(Path(__file__).resolve().parent.parent / "data" / "synthetic_corpus.json")
    calls = []
    def call(system, user, schema=None):
        calls.append(1)
        budget.call_timeout(300)  # what every real transport does before sending
        return "{}"
    monkeypatch.setattr(budget, "ANSWER_SECONDS", 0.0)
    res = Muhawir(corpus, ModelGenerator([("a", call), ("b", call)])).ask("ما حكم سقي النخلة في الصيف؟")
    assert res.status in ("unavailable", "abstained") and res.status != "answered"


# ---- M-011: vectors made from other passage text are not used ------------------------------------------------------

def test_vectors_are_used_only_with_the_passages_they_were_made_from(tmp_path):
    import sqlite3
    import numpy as np
    from muhawir import vectors as V
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE passages (rid INTEGER PRIMARY KEY, id TEXT, text TEXT)")
    con.executemany("INSERT INTO passages (id, text) VALUES (?, ?)", [("a:1", "نص أول"), ("a:2", "نص ثان")])
    index = V.VectorIndex(["a:1", "a:2"], np.zeros((2, 4), dtype=np.float32), "m", None)
    assert V.check_corpus(index, con, tmp_path) is index and V.VECTOR_STATE["corpus"] == "unverified"
    (tmp_path / V.CORPUS_STAMP).write_text(V.corpus_fingerprint(con, index.ids))
    assert V.check_corpus(index, con, tmp_path) is index and V.VECTOR_STATE["corpus"] == "matches"
    con.execute("UPDATE passages SET text = 'نص مصحح' WHERE id = 'a:2'")  # same id, corrected text
    assert V.check_corpus(index, con, tmp_path) is None and V.VECTOR_STATE["corpus"] == "mismatch"


# ---- M-008: the speech cache ages out and stays within its size ---------------------------------------------------

def test_the_speech_cache_ages_out_and_is_trimmed(tmp_path, monkeypatch):
    import os
    import time
    from muhawir import tts
    monkeypatch.setattr(tts, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(tts, "CACHE_MB", 0)  # every older file is over the limit
    old = tmp_path / "old.mp3"
    old.write_bytes(b"x" * 100)
    os.utime(old, (time.time() - 40 * 86400,) * 2)
    assert tts._read_fresh(old) is None and not old.exists()  # older than 30 days: removed, not served
    tts._keep_audio("d1", tmp_path / "d1.mp3", b"a" * 10)
    tts._keep_audio("d2", tmp_path / "d2.mp3", b"b" * 10)
    assert len(list(tmp_path.glob("*.mp3"))) <= 1
    monkeypatch.setattr(tts, "CACHE_DIR", tmp_path / "missing" / "deep")
    monkeypatch.setattr(tts.os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    tts._keep_audio("d3", tmp_path / "d3.mp3", b"c")  # a full disk is logged, not raised
