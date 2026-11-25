import io
import logging

from muhawir import logsetup


def test_a_long_record_stays_on_one_line_and_a_later_basicconfig_changes_nothing(monkeypatch):
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    stream = io.StringIO()
    logsetup.plain_logging(stream=stream)
    logging.basicConfig(handlers=[logging.NullHandler()])  # what a library (the MCP server) does at start-up
    assert len(root.handlers) == 1 and root.handlers[0].stream is stream
    payload = "x" * 5000
    logging.getLogger("muhawir.telemetry").warning("muhawir_trace %s", payload)
    out = stream.getvalue()
    assert out.count("\n") == 1 and out == f"WARNING muhawir.telemetry: muhawir_trace {payload}\n"


def test_existing_handlers_are_left_alone(monkeypatch):
    root = logging.getLogger()
    keep = logging.NullHandler()
    monkeypatch.setattr(root, "handlers", [keep])
    logsetup.plain_logging(stream=io.StringIO())
    assert root.handlers == [keep]
