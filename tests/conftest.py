"""Shared test setup.

Most tests use fake models that only know the understanding and answer steps. The second
reading (check_support) is accepted by default here; tests marked `real_check` exercise it.
"""
import os

import pytest

# Tests run on the synthetic corpus with fake models only. Set before any test module imports
# muhawir.server, which would otherwise load a real data/muhawir.db or a real model from the environment.
os.environ["MUHAWIR_DB"] = os.path.join(os.path.dirname(__file__), "no-database-for-tests.db")
for _name in ("MUHAWIR_CORPUS", "LLM_PROVIDER", "MUHAWIR_DEBUG", "MUHAWIR_DORAR"):
    os.environ.pop(_name, None)

from muhawir.generate import ModelGenerator  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "real_check: run ModelGenerator.check_support for real")
    config.addinivalue_line("markers", "raw_check_reply: send the fake model's support-check reply as it is")


@pytest.fixture(autouse=True)
def _accept_support_check(request, monkeypatch):
    if "real_check" not in request.keywords:
        monkeypatch.setattr(ModelGenerator, "check_support", lambda self, claims, passages, question="": [True] * len(claims))


@pytest.fixture(autouse=True)
def _complete_check_replies(request, monkeypatch):
    """Strict-check (audit HIGH-001): the support check now uses a reply only when every required field is there for
    every sentence. Fake models in older tests answer with the verdicts alone («supported»: [...] or «problem»: [...]).
    Here they are made to answer as a compliant model does: the same verdicts, each accepted sentence with the text of
    its cited passage as its witness, no missing term, on topic. Tests marked raw_check_reply see the reply unchanged."""
    if "raw_check_reply" in request.keywords:
        return
    import json
    import re
    from muhawir import generate

    original = generate.with_retry

    def complete(call, system, user, schema, *args, **kwargs):
        raw = original(call, system, user, schema, *args, **kwargs)
        if system != generate.CHECK_PROMPT:
            return raw
        try:
            data = generate.load_json(raw)
        except Exception:
            return raw
        if not isinstance(data, dict):
            return raw
        blocks = re.findall(r"الجملة \d+: <<<.*?>>>\nالمقاطع:\n(.*?)(?=\n\nالجملة \d+: <<<|\Z)", user, re.S)
        n = len(blocks)
        if not isinstance(data.get("problem"), list) and isinstance(data.get("supported"), list):
            data["problem"] = ["none" if (f is True or str(f).lower() == "true") else "addition" for f in data["supported"]]
        if isinstance(data.get("problem"), list) and len(data["problem"]) == n:
            if not isinstance(data.get("evidence"), list):
                data["evidence"] = []
                for block in blocks:  # the first cited passage's own text: certainly in that passage
                    lines = [line for line in block.split("\n") if line.strip() and not line.startswith("[")]
                    data["evidence"].append(lines[0] if lines else "")
            data.setdefault("missing", [[] for _ in range(n)])
            data.setdefault("on_topic", [True] * n)
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(generate, "with_retry", complete)
