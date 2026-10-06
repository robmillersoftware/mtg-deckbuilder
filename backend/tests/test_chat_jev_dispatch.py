"""process_message: confident Jev routes dispatch without an LLM tool call; otherwise the LLM decides."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.conversation import Conversation
from app.schemas.conversation import ChatResponse
from app.services import chat_routing, llm
from app.services.chat_service import TOOLS, ChatService

SHEOLDRED = {"name": "Sheoldred, the Apocalypse", "mana_cost": "{2}{B}{B}",
             "type_line": "Legendary Creature", "oracle_text": "Deathtouch", "colors": ["B"]}


def routed(action, confidence=0.9, probabilities=None):
    return chat_routing.Route(action, confidence, probabilities or {action: confidence},
                              {t["name"]: {"from": t["name"]} for t in TOOLS})


@pytest.fixture
def chat(monkeypatch):
    svc = ChatService.__new__(ChatService)
    svc.db = MagicMock(flush=AsyncMock(), commit=AsyncMock())
    conv = Conversation(messages=[], context={"strategy": "Mono-black midrange", "colors": ["B"]})
    conv.id = uuid4()
    svc._get_or_create_conversation = AsyncMock(return_value=conv)
    svc._meta_archetypes = AsyncMock(return_value=["Mono-Red Aggro"])
    svc.resolved = []

    async def resolve(message, format, check_legality=True):
        return svc.resolved if check_legality else []
    svc._resolve_card_mentions = resolve

    svc.dispatched = []

    async def dispatch(name, tool_input, conversation, user_id, ai_text=""):
        svc.dispatched.append((name, tool_input))
        return ChatResponse(response=f"ran {name}", conversation_id=conversation.id)
    svc._dispatch_tool = dispatch

    svc.llm_calls = []

    def chat_with_tools(**kwargs):
        svc.llm_calls.append("tools")
        return "llm text", []

    def complete(system, user, max_tokens=4096):
        svc.llm_calls.append("complete")
        svc.complete_args = (system, user)
        return "plain answer"
    monkeypatch.setattr(llm, "is_configured", lambda: True)
    monkeypatch.setattr(llm, "chat_with_tools", chat_with_tools)
    monkeypatch.setattr(llm, "complete", complete)

    svc.route = None

    async def route(**kwargs):
        svc.route_kwargs = kwargs
        return svc.route
    monkeypatch.setattr(chat_routing, "route", route)
    return svc


async def test_confident_tool_dispatches_without_llm(chat):
    chat.route = routed("suggest_core")
    resp = await chat.process_message("Build me a mono-black deck")
    assert resp.response == "ran suggest_core"
    assert chat.dispatched == [("suggest_core", {"from": "suggest_core"})]
    assert chat.llm_calls == []


async def test_route_state_comes_from_conversation(chat):
    chat.resolved = [SHEOLDRED]
    chat.route = routed("suggest_core")
    await chat.process_message("Build around Sheoldred")
    kw = chat.route_kwargs
    assert kw["card_names"] == ["Sheoldred, the Apocalypse"]
    assert kw["strategy"] == "Mono-black midrange"
    assert kw["summary"] == {"has_deck": False, "unique_nonland_cards": 0, "colors": ["B"]}
    assert kw["meta_archetypes"] == ["Mono-Red Aggro"]
    assert kw["history"][-1]["content"] == "Build around Sheoldred"
    assert kw["tools"] is TOOLS


async def test_forced_tool_overrides_reply(chat):
    chat.resolved = [SHEOLDRED]
    chat.route = routed("reply", 0.8, {"reply": 0.7, "get_matchup_info": 0.2, "suggest_core": 0.1})
    resp = await chat.process_message("Tell me about Sheoldred")
    assert chat.dispatched == [("get_matchup_info", {"from": "get_matchup_info"})]
    assert resp.response == "ran get_matchup_info"
    assert chat.llm_calls == []


async def test_confident_reply_uses_text_completion(chat):
    chat.route = routed("reply")
    resp = await chat.process_message("What does trample do?")
    assert resp.response == "plain answer"
    assert chat.llm_calls == ["complete"]
    system, user = chat.complete_args
    assert "no tools" in system
    assert user.endswith("user: What does trample do?")


async def test_low_confidence_uses_llm_tool_call(chat):
    chat.route = routed("suggest_core", confidence=chat_routing.ROUTE_CONFIDENCE - 0.01)
    resp = await chat.process_message("hmm, red maybe?")
    assert chat.llm_calls == ["tools"] and chat.dispatched == []
    assert resp.response == "llm text"


async def test_route_none_uses_llm_tool_call(chat):
    chat.route = None
    await chat.process_message("hello")
    assert chat.llm_calls == ["tools"]


async def test_empty_text_reply_falls_through_to_llm_tool_call(chat, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda system, user, max_tokens=4096: "")
    chat.route = routed("reply")
    await chat.process_message("What does trample do?")
    assert chat.llm_calls == ["tools"]


async def test_meta_query_filters_latest_snapshot_of_format(chat, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "k")
    captured = []

    async def execute(stmt):
        captured.append(str(stmt.compile(compile_kwargs={"literal_binds": True})))
        return MagicMock(scalars=lambda: MagicMock(all=lambda: ["A", "B", "A"]))
    chat.db.execute = execute
    assert await ChatService._meta_archetypes(chat, "modern") == ["A", "B"]
    sql = captured[0]
    assert "max(meta_snapshots.snapshot_date)" in sql and "'modern'" in sql
    assert "meta_snapshots.snapshot_date =" in sql


async def test_meta_query_skipped_without_jev_key(chat):
    chat.db.execute = AsyncMock(side_effect=AssertionError("queried"))
    assert await ChatService._meta_archetypes(chat, "modern") == []


async def test_meta_query_error_rolls_back_and_turn_uses_llm(chat, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "k")
    chat.db.execute = AsyncMock(side_effect=RuntimeError("db down"))
    chat.db.rollback = AsyncMock()
    assert await ChatService._meta_archetypes(chat, "modern") == []
    chat.db.rollback.assert_awaited_once()
    # full turn: real _meta_archetypes + real _route_with_jev, route unavailable
    chat._meta_archetypes = ChatService._meta_archetypes.__get__(chat)
    chat.route = None
    await chat.process_message("hello")
    assert chat.llm_calls == ["tools"]
