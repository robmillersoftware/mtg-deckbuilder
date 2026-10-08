"""Keep unit tests off the network: the repo-root .env holds real keys."""

import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def _no_external_keys(monkeypatch):
    for key in ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setattr(settings, key, None)
