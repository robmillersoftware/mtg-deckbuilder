"""Running a deck test: progress, missing cards, stop, and failures."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.simulation import SimulationRun
from app.services import forge, sim_runs
from app.services.forge import OPPONENT, TESTED, GameRecord
from app.services.gauntlet import Opponent

OPPONENTS = [Opponent("A", 20.0, {"Island": 60}), Opponent("B", 10.0, {"Swamp": 60})]


def game(winner):
    return GameRecord(winner=winner, turns=7, own_turns={TESTED: 7, OPPONENT: 7}, mulligans={TESTED: 0, OPPONENT: 0},
                      casts={TESTED: [(2, "Bear")], OPPONENT: []}, lands={TESTED: [(1, "Forest")], OPPONENT: []},
                      log=["Turn 1: Tested"])


def make_run(**overrides):
    fields = dict(id=uuid4(), kind="test", status="running", format="standard", games_per_matchup=4,
                  stop_requested=False, deck={"name": "Mine", "main_deck": [
                      {"card_name": "Forest", "quantity": 20}, {"card_name": "Bear", "quantity": 4},
                      {"card_name": "Unknown Card", "quantity": 4}]})
    return SimulationRun(**{**fields, **overrides})


def fake_db():
    return MagicMock(commit=AsyncMock(), refresh=AsyncMock(), execute=AsyncMock())


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(sim_runs, "gauntlet", AsyncMock(return_value=OPPONENTS))
    monkeypatch.setattr(sim_runs, "land_names", AsyncMock(return_value={"Forest"}))
    monkeypatch.setattr(forge, "card_names", lambda: {"forest", "bear"})

    async def play(tested, matchups, games, seed, on_progress=None):
        assert "Unknown Card" in tested  # write_deck leaves it out; the job doesn't strip it
        out = {}
        for m in matchups:
            records = [game(TESTED), game(OPPONENT)] * (games // 2)
            out[m.key] = records
            if on_progress:
                await on_progress(m.key, records)
        return out
    monkeypatch.setattr(forge, "play", play)


async def test_run_test_reports_and_tracks_progress(wired):
    db, run = fake_db(), make_run()
    await sim_runs.run_test(db, run)
    assert run.status == "completed"
    assert run.report["overall"]["games"] == 8 and run.report["not_simulated"]["cards"] == ["Unknown Card"]
    assert [m["opponent"] for m in run.report["matchups"]] == ["A", "B"]
    p = run.progress
    assert p["games_done"] == p["games_planned"] == 8 and p["stage"] == "Done"
    assert p["matchups"][0] == {"opponent": "A", "share": 20.0, "wins": 2, "losses": 2, "draws": 0}
    assert any("Unknown Card" in e["text"] and "left out" in e["text"] for e in p["events"])


async def test_stop_keeps_the_games_played(wired, monkeypatch):
    db, run = fake_db(), make_run()
    calls = {"n": 0}

    async def refresh(obj, attrs=None):
        calls["n"] += 1
        obj.stop_requested = calls["n"] >= 1  # stop after the first batch
    db.refresh = refresh
    await sim_runs.run_test(db, run)
    assert run.status == "stopped" and run.report["stopped"] == "user"
    assert run.report["overall"]["games"] == 4  # only matchup A finished


async def test_no_opponents_is_a_user_facing_error(wired, monkeypatch):
    monkeypatch.setattr(sim_runs, "gauntlet", AsyncMock(return_value=[]))
    with pytest.raises(sim_runs.SimError, match="no recent decklists"):
        await sim_runs.run_test(fake_db(), make_run())


async def test_execute_marks_failures_in_plain_words(monkeypatch):
    run = make_run(status="queued")
    db = fake_db()
    db.get = AsyncMock(return_value=run)
    db.rollback = AsyncMock()
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=db)
    session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(sim_runs, "async_session_factory", lambda: session)
    monkeypatch.setattr(forge, "available", lambda: True)
    monkeypatch.setattr(sim_runs, "run_test", AsyncMock(side_effect=forge.ForgeError("Forge exited with code 1")))
    await sim_runs.execute(run.id)
    assert run.status == "failed"
    assert run.error == "Couldn't finish playtesting: the simulator failed while playing games."
    assert "Forge exited" not in run.error

    monkeypatch.setattr(forge, "available", lambda: False)
    run.status = "queued"
    await sim_runs.execute(run.id)
    assert run.status == "failed" and run.error == "Couldn't finish playtesting: the simulator isn't set up yet."


def test_deck_entry_conversions():
    assert sim_runs.main_of([{"card_name": "A", "quantity": "2"}, {"card_name": "A", "quantity": 1}]) == {"A": 3}
    assert sim_runs.entries_of({"A": 3, "B": 0}) == [{"card_name": "A", "quantity": 3}]
