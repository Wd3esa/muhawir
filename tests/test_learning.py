"""The classroom serves local teaching assets without running the answer engine."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from muhawir import server

ROOT = Path(__file__).resolve().parents[1]


def test_classroom_assets_do_not_call_the_answer_engine(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a classroom asset must not generate an answer")

    monkeypatch.setattr(server.engine, "ask", forbidden)
    client = TestClient(server.app)
    for path, media in [("/how-it-works", "text/html"),
                        ("/learning/style.css", "text/css"),
                        ("/learning/app.js", "javascript"),
                        ("/learning/content.json", "application/json")]:
        reply = client.get(path)
        assert reply.status_code == 200 and media in reply.headers["content-type"]
    assert "/how-it-works" in client.get("/").text


def test_all_components_and_scenarios_resolve_to_real_code():
    content = json.loads((ROOT / "muhawir/static/learn/content.json").read_text())
    ids = {node["id"] for node in content["nodes"]}
    assert len(ids) == len(content["nodes"])
    mapped = set()
    for node in content["nodes"]:
        for key in ("summary", "how", "why", "alternative", "exercise"):
            assert node[key]
        assert all((ROOT / path).is_file() for path in node["files"])
    for view in content["views"]:
        assert len(view["positions"]) == len(view["nodes"])
        mapped.update(view["nodes"])
        assert all(part in ids for edge in view["edges"] for part in edge)
    assert mapped == ids
    for scenario in content["scenarios"]:
        assert set(scenario["route"]) <= ids
    assert len(content["lessons"]) == 10


def test_learning_client_has_only_a_static_content_request():
    script = (ROOT / "muhawir/static/learn/app.js").read_text()
    assert script.count("fetch(") == 1
    assert 'fetch("/learning/content.json"' in script
    assert "/api/ask" not in script and "/tts" not in script
    assert "innerHTML" not in script


def test_learning_assets_cannot_serve_parent_files():
    client = TestClient(server.app)
    assert client.get("/learning/%2e%2e/server.py").status_code == 404
