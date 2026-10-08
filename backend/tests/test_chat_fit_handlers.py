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


async def test_requested_card_without_colors_does_not_borrow_the_top_decks_colors():
    # "build around Weapons Manufacturing" once got the top deck's U/B plus the card's R
    cid = uuid4()
    result = SimpleNamespace(deck=_deck(), conversation_id=cid, strategy_summary="s", fit_flagged=[])
    s = _service(generate=AsyncMock(return_value=result))
    s.db.execute = AsyncMock()
    conv = SimpleNamespace(id=cid)
    await s._handle_generate_full_deck({"colors": [], "specific_cards": ["Weapons Manufacturing"]}, conv, None)
    kwargs = s.deck_generator.generate.await_args.kwargs
    assert kwargs["colors"] is None  # the generator takes colors from the requested card
    assert kwargs["specific_cards"] == ["Weapons Manufacturing"]
    s.db.execute.assert_not_awaited()


async def test_generate_handler_starts_a_playtest(monkeypatch):
    from app.services import chat_service
    cid = uuid4()
    result = SimpleNamespace(deck=_deck(), conversation_id=cid, strategy_summary="s", fit_flagged=[],
                             assembly={"reference": None, "relatives": [], "colors": ["R"], "synergy": None,
                                       "requested": []})
    s = _service(generate=AsyncMock(return_value=result))
    sim_id = uuid4()
    start = AsyncMock(return_value=(SimpleNamespace(id=sim_id), None))
    monkeypatch.setattr(chat_service, "start_build", start)
    resp = await s._handle_generate_full_deck({"colors": ["R"]}, SimpleNamespace(id=cid), None)
    assert resp.simulation_id == sim_id and "Playtesting against the top 5 decks" in resp.response
    deck_arg, assembly_arg = start.await_args.args[1:3]
    assert deck_arg["name"] == "D" and assembly_arg == result.assembly

    start.return_value = (None, "Couldn't playtest this deck (the simulator isn't running), so this is the untested build.")
    resp = await s._handle_generate_full_deck({"colors": ["R"]}, SimpleNamespace(id=cid), None)
    assert resp.simulation_id is None and "simulator isn't running" in resp.response


async def test_generate_handler_survives_a_playtest_that_cannot_start(monkeypatch):
    from app.services import chat_service
    cid = uuid4()
    result = SimpleNamespace(deck=_deck(), conversation_id=cid, strategy_summary="s", fit_flagged=[],
                             assembly={"reference": None, "relatives": [], "colors": ["R"], "synergy": None,
                                       "requested": []})
    s = _service(generate=AsyncMock(return_value=result))
    monkeypatch.setattr(chat_service, "start_build", AsyncMock(side_effect=RuntimeError("redis down")))
    resp = await s._handle_generate_full_deck({"colors": ["R"]}, SimpleNamespace(id=cid), None)
    assert resp.simulation_id is None and resp.deck["name"] == "D" and "simulator isn't running" in resp.response
