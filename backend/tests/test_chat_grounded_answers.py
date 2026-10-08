"""Meta and matchup questions get an LLM answer to the user's message, grounded in the data."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.conversation import Conversation
from app.core.config import settings
from app.services import llm
from app.services.chat_service import GROUNDED_ANSWER_SYSTEM, ChatService

SNAPSHOTS = [SimpleNamespace(archetype="Dimir Aggro", meta_percentage=10.9),
             SimpleNamespace(archetype="Mono Green Aggro", meta_percentage=4.6)]


@pytest.fixture
def chat(monkeypatch):
    svc = ChatService.__new__(ChatService)
    result = MagicMock()
    result.scalars.return_value.all.return_value = SNAPSHOTS
    svc.db = MagicMock(execute=AsyncMock(return_value=result), commit=AsyncMock())
    svc._current_format = "standard"
    svc.seen = {}

    def complete(system, user, max_tokens=4096, model=None):
        svc.seen.update(system=system, user=user, model=model)
        return "Mono Green Aggro is a fast creature deck; cheap removal and blockers beat it."
    monkeypatch.setattr(llm, "is_configured", lambda: True)
    monkeypatch.setattr(llm, "complete", complete)
    return svc


def conversation(message, deck=None):
    conv = Conversation(messages=[{"role": "user", "content": message}], context={}, current_deck=deck)
    conv.id = uuid4()
    return conv


async def test_analyze_meta_answers_the_question_from_meta_data(chat):
    chat._archetype_cards = AsyncMock(side_effect=lambda a, f: [f"4.0x {a} Key Card (R Instant): Deal 3."])
    conv = conversation("what is a good card for my deck?")
    resp = await chat._handle_analyze_meta({}, conv)
    # the top decks' key cards are there to reason from; the canned closing question isn't
    assert "Dimir Aggro Key Card" in chat.seen["user"] and "Mono Green Aggro Key Card" in chat.seen["user"]
    assert "What direction interests you" not in chat.seen["user"]
    assert resp.response == "Mono Green Aggro is a fast creature deck; cheap removal and blockers beat it."
    assert "what is a good card for my deck?" in chat.seen["user"]
    assert "**Dimir Aggro** (10.9%)" in chat.seen["user"]
    assert chat.seen["system"] == GROUNDED_ANSWER_SYSTEM
    assert chat.seen["model"] == settings.ANSWER_MODEL
    assert conv.messages[-1]["content"] == resp.response


async def test_matchup_without_deck_includes_the_opponents_cards(chat, monkeypatch):
    chat._archetype_cards = AsyncMock(return_value=[
        "4x Llanowar Elves ({G} Creature — Elf Druid): {T}: Add {G}."])
    conv = conversation("what beats mono green?")
    resp = await chat._handle_matchup_query({"opponent_deck": "Mono Green Aggro"}, conv)
    assert resp.response.startswith("Mono Green Aggro is a fast creature deck")
    assert "Llanowar Elves" in chat.seen["user"] and "what beats mono green?" in chat.seen["user"]
    chat._archetype_cards.assert_awaited_once_with("Mono Green Aggro", "standard")
    assert "Tips against" not in chat.seen["user"]  # canned tips are fallback only


async def test_llm_failure_falls_back_to_the_data(chat, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(llm, "complete", boom)
    resp = await chat._handle_analyze_meta({}, conversation("hi there"))
    assert "Dimir Aggro" in resp.response  # the old template text


async def test_unconfigured_llm_falls_back_to_the_data(chat, monkeypatch):
    monkeypatch.setattr(llm, "is_configured", lambda: False)
    resp = await chat._handle_analyze_meta({}, conversation("hi there"))
    assert "Dimir Aggro" in resp.response
