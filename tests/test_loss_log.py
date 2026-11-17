"""The loss log (issue #100): every answer records, as counts only, what each step kept and removed, so a loss is
measured and not guessed. Fake model and synthetic data; nothing here says anything about real answers."""
import json

import pytest

from muhawir import telemetry
from test_fewer_calls import CORPUS, QUESTION, SIX, _model, _write

pytestmark = pytest.mark.real_check


def _traced(monkeypatch, engine, style="youth"):
    monkeypatch.setenv("MUHAWIR_TRACE", "1")
    trace = telemetry.start("ask")
    with telemetry.bind(trace):
        _write(engine, style)
        trace.finish("answered", 200)
    return {e["stage"]: e["counts"] for e in trace.events if e["kind"] == "loss"}, trace


def test_each_step_records_its_counts(monkeypatch):
    engine, _ = _model(SIX, reject=("البارد",))
    loss, _ = _traced(monkeypatch, engine)
    assert loss["draft"]["offered"] == len(CORPUS.passages) and loss["draft"]["drafted"] == 6
    assert loss["check"]["kept"] == 5 and loss["check"]["rejected"] == 1 and loss["check"]["why_not_supported"] == 1
    assert loss["review"]["verdict_yes"] == 1
    assert loss["display"]["kept"] == 5 and loss["display"]["shown"] == 5


def test_a_length_cut_is_visible(monkeypatch):
    engine, _ = _model(SIX)
    monkeypatch.setattr("muhawir.pipeline.STYLE_CEILING", {"youth": (4, 4), "kids": (4, 4)})
    loss, _ = _traced(monkeypatch, engine)
    assert loss["display"]["kept"] == 6 and loss["display"]["shown"] == 4 and loss["display"]["length_cut"] == 2


def test_nothing_but_counts_is_recorded(monkeypatch, caplog):
    engine, _ = _model(SIX, reject=("البارد",))
    _, trace = _traced(monkeypatch, engine)
    for event in (e for e in trace.events if e["kind"] == "loss"):
        assert set(event) == {"kind", "stage", "counts", "event_id"}
        assert all(isinstance(v, int) and not isinstance(v, bool) for v in event["counts"].values())
    text = json.dumps(trace.events, ensure_ascii=False)
    assert "النخلة" not in text and QUESTION not in text


def test_without_a_trace_nothing_happens():
    engine, _ = _model(SIX)
    _write(engine)  # no trace bound: the log must not raise


def test_the_analyzer_sums_the_loss_per_step(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from analyze_request_traces import loss_summary
    rows = [{"events": [{"kind": "loss", "stage": "check", "counts": {"kept": 5, "rejected": 1}}]},
            {"events": [{"kind": "loss", "stage": "check", "counts": {"kept": 3, "rejected": 0}},
                        {"kind": "model", "stage": "x"}]}]
    assert loss_summary(rows) == {"check": {"answers_with": 2, "kept": 8, "rejected": 1}}
