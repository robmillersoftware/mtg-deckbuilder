"""The simulations API: creating a test, reading it, stopping it."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

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
                 api.SimulationCreate(deck=DECK, format="cedh"),
                 api.SimulationCreate(deck={"main_deck": [{"card_name": "", "quantity": 4}]}),
                 api.SimulationCreate(deck={"main_deck": [{"card_name": "Forest", "quantity": 0}]}),
                 api.SimulationCreate(deck={"main_deck": [{"card_name": "Forest"}]})):
        with pytest.raises(HTTPException) as e:
            await api.create_simulation(body, db_returning(), None)
        assert e.value.status_code == 400


async def test_create_limits_the_size_of_a_test(monkeypatch):
    monkeypatch.setattr(api, "sim_worker_running", lambda: True)
    monkeypatch.setattr(api, "enqueue", lambda run_id: None)
    monkeypatch.setattr(api, "queue_position", lambda run_id: 1)
    with pytest.raises(ValidationError):
        api.SimulationCreate(deck=DECK, games=101)
    names = [f"Deck {i}" for i in range(11)]
    with pytest.raises(HTTPException) as e:
        await api.create_simulation(api.SimulationCreate(deck=DECK, opponents=names, games=20), db_returning(), None)
    assert (e.value.status_code, e.value.detail) == (400, api.TOO_MANY_OPPONENTS)
    # The largest test the other limits allow, 10 opponents x 100 games, is exactly the cap.
    resp = await api.create_simulation(api.SimulationCreate(deck=DECK, opponents=names[:10], games=100),
                                       db_returning(), None)
    assert resp.status == "queued"
    # No opponents chosen counts as the gauntlet's 5.
    monkeypatch.setattr(api, "MAX_TOTAL_GAMES", 499)
    with pytest.raises(HTTPException) as e:
        await api.create_simulation(api.SimulationCreate(deck=DECK, games=100), db_returning(), None)
    assert (e.value.status_code, e.value.detail) == (400, api.TOO_MANY_GAMES)
    monkeypatch.setattr(api, "MAX_TOTAL_GAMES", 500)
    assert (await api.create_simulation(api.SimulationCreate(deck=DECK, games=100), db_returning(), None)).status == "queued"


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


async def test_stop_leaves_finished_runs_alone(monkeypatch):
    monkeypatch.setattr(api, "queue_position", lambda run_id: None)
    run = saved_run(user_id=None, status="completed")
    resp = await api.stop_simulation(run.id, db_returning(run), None)
    assert resp.status == "completed" and not run.stop_requested


async def test_archetypes_use_the_latest_snapshot_and_dedupe():
    rows = [SimpleNamespace(archetype="Dimir Aggro "), SimpleNamespace(archetype="dimir aggro"),
            SimpleNamespace(archetype="UW Control")]
    db = MagicMock(execute=AsyncMock(return_value=MagicMock(scalars=lambda: MagicMock(all=lambda: rows))))
    assert await api.list_archetypes("standard", db) == ["Dimir Aggro", "UW Control"]
    stmt = str(db.execute.call_args.args[0])
    assert "max(" in stmt and "snapshot_date" in stmt


async def test_stalled_run_is_reported_failed(monkeypatch):
    from datetime import timedelta
    monkeypatch.setattr(api, "queue_position", lambda run_id: None)
    run = saved_run(user_id=None, status="running", updated_at=datetime.utcnow() - timedelta(minutes=20))
    db = db_returning(run)
    resp = await api.get_simulation(run.id, db, None)
    assert resp.status == "failed" and resp.error == "The playtest stopped responding before it finished."
    db.commit.assert_awaited()


async def test_recently_updated_run_stays_running(monkeypatch):
    from datetime import timedelta
    monkeypatch.setattr(api, "queue_position", lambda run_id: None)
    run = saved_run(user_id=None, status="running", updated_at=datetime.utcnow() - timedelta(minutes=1))
    resp = await api.get_simulation(run.id, db_returning(run), None)
    assert resp.status == "running" and resp.error is None
