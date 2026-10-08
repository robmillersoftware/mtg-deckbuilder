"""The simulations API: creating a test, reading it, stopping it."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.routes import simulation as api
from app.models.simulation import SimulationRun

DECK = {"name": "Mine", "main_deck": [{"card_name": "Forest", "quantity": 60}], "sideboard": []}
ALICE, BOB = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
NOW = datetime(2026, 10, 7)


def saved_run(**fields):
    return SimulationRun(**{"id": uuid4(), "kind": "test", "format": "standard", "deck": DECK,
                            "games_per_matchup": 50, "stop_requested": False, "created_at": NOW,
                            "updated_at": NOW, **fields})


def db_returning(obj=None):
    db = MagicMock(add=MagicMock(), commit=AsyncMock(), refresh=AsyncMock(), get=AsyncMock(return_value=obj))
    return db


async def test_create_needs_the_sim_worker(monkeypatch):
    monkeypatch.setattr(api, "sim_worker_running", lambda: False)
    with pytest.raises(HTTPException) as e:
        await api.create_simulation(api.SimulationCreate(deck=DECK), db_returning(), None)
    assert e.value.status_code == 503 and "isn't running" in e.value.detail


async def test_create_queues_a_test(monkeypatch):
    monkeypatch.setattr(api, "sim_worker_running", lambda: True)
    queued = []
    monkeypatch.setattr(api, "enqueue", queued.append)
    monkeypatch.setattr(api, "queue_position", lambda run_id: 1)
    db = db_returning()
    resp = await api.create_simulation(api.SimulationCreate(deck=DECK, opponents=["Dimir Aggro"], games=20), db, ALICE)
    run = db.add.call_args.args[0]
    assert (run.kind, run.user_id, run.opponents, run.games_per_matchup) == ("test", ALICE.id, ["Dimir Aggro"], 20)
    assert queued == [run.id] and resp.status == "queued" and resp.queue_position == 1


async def test_create_rejects_bad_input(monkeypatch):
    monkeypatch.setattr(api, "sim_worker_running", lambda: True)
    for body in (api.SimulationCreate(), api.SimulationCreate(deck={"main_deck": []}),
                 api.SimulationCreate(deck=DECK, format="cedh")):
        with pytest.raises(HTTPException) as e:
            await api.create_simulation(body, db_returning(), None)
        assert e.value.status_code == 400


@pytest.mark.parametrize("owner,caller,ok", [(None, None, True), (ALICE.id, ALICE, True),
                                             (ALICE.id, None, False), (ALICE.id, BOB, False)])
async def test_get_access(owner, caller, ok, monkeypatch):
    monkeypatch.setattr(api, "queue_position", lambda run_id: None)
    run = saved_run(user_id=owner, status="running")
    if ok:
        assert (await api.get_simulation(run.id, db_returning(run), caller)).id == run.id
    else:
        with pytest.raises(HTTPException) as e:
            await api.get_simulation(run.id, db_returning(run), caller)
        assert e.value.status_code == 404


async def test_stop_a_queued_run_stops_it_now(monkeypatch):
    monkeypatch.setattr(api, "queue_position", lambda run_id: None)
    removed = []
    monkeypatch.setattr(api.sim_queue, "remove", removed.append)
    run = saved_run(user_id=None, status="queued")
    resp = await api.stop_simulation(run.id, db_returning(run), None)
    assert run.stop_requested and resp.status == "stopped" and removed == [str(run.id)]
