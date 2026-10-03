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


@pytest.fixture(autouse=True)
def _accept_support_check(request, monkeypatch):
    if "real_check" not in request.keywords:
        monkeypatch.setattr(ModelGenerator, "check_support", lambda self, claims, passages, question="": [True] * len(claims))
