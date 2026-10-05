"""Chat handlers: only generate carries fit_flagged (iterate responses have none)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.services.chat_service import ChatService


def _deck(**kw):
    return SimpleNamespace(name="D", format="standard", commander=None,
                           main_deck=[], sideboard=[], archetype="aggro", **kw)


def _service(**stubs):
    s = ChatService.__new__(ChatService)
    s.db = MagicMock()
    s.deck_generator = SimpleNamespace(**stubs)
    s._current_format = "standard"
    return s


async def test_modification_handler_has_no_fit_flagged():
    result = SimpleNamespace(deck=_deck(), summary="done")  # no fit_flagged, like DeckIterateResponse
    s = _service(iterate=AsyncMock(return_value=result))
    conv = SimpleNamespace(id=uuid4(), current_deck={"main_deck": []})
    resp = await s._handle_deck_modification({"modification": "more burn"}, conv, None)
    assert "fit_flagged" not in resp.deck


async def test_generate_handler_carries_fit_flagged():
    cid = uuid4()
    result = SimpleNamespace(deck=_deck(), conversation_id=cid, strategy_summary="s",
                             fit_flagged=["Bad Card"])
    s = _service(generate=AsyncMock(return_value=result))
    conv = SimpleNamespace(id=cid)
    resp = await s._handle_generate_full_deck({"colors": ["R"]}, conv, None)
    assert resp.deck["fit_flagged"] == ["Bad Card"]
