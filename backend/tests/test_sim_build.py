"""Playtesting chat builds: starting the run, protection rules, replacements, the job."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.config import settings
from app.services import sim_build as sb

DECK = {"name": "Brew", "main_deck": [{"card_name": "Forest", "quantity": 60}], "sideboard": []}
ASSEMBLY = {"reference": None, "relatives": ["Boros Dwarves", "Jeskai Control"], "colors": ["W", "U", "R"],
            "synergy": {"type_contains": "Artifact", "min": 8}, "requested": ["Weapons Manufacturing"]}


class TestStartBuild:
    async def test_queues_a_build_with_its_options(self, monkeypatch):
        monkeypatch.setattr(sb, "sim_worker_running", lambda: True)
        queued = []
        monkeypatch.setattr(sb, "enqueue", queued.append)
        db = MagicMock(add=MagicMock(), commit=AsyncMock())
        conv, user = uuid4(), uuid4()
        run, note = await sb.start_build(db, DECK, ASSEMBLY, conv, user, "standard")
        assert note is None and queued == [run.id]
        assert (run.kind, run.status, run.conversation_id, run.user_id) == ("build", "queued", conv, user)
        assert run.options == {"requested": ["Weapons Manufacturing"], "archetypes": ["Boros Dwarves", "Jeskai Control"],
                               "synergy": {"type_contains": "Artifact", "min": 8}, "colors": ["W", "U", "R"]}

    async def test_reference_decks_protect_the_reference_lists(self, monkeypatch):
        monkeypatch.setattr(sb, "sim_worker_running", lambda: True)
        monkeypatch.setattr(sb, "enqueue", lambda run_id: None)
        run, _ = await sb.start_build(MagicMock(add=MagicMock(), commit=AsyncMock()), DECK,
                                      {**ASSEMBLY, "reference": "Boros Aggro"}, None, None, "standard")
        assert run.options["archetypes"] == ["Boros Aggro"]

    async def test_explains_when_it_cannot_playtest(self, monkeypatch):
        monkeypatch.setattr(sb, "sim_worker_running", lambda: False)
        run, note = await sb.start_build(MagicMock(), DECK, ASSEMBLY, None, None, "standard")
        assert run is None and "simulator isn't running" in note
        for args in ((DECK, None, None, None, "standard"), (DECK, ASSEMBLY, None, None, "cedh")):
            assert await sb.start_build(MagicMock(), *args) == (None, None)
        monkeypatch.setattr(settings, "FORGE_ENABLED", False)
        assert await sb.start_build(MagicMock(), DECK, ASSEMBLY, None, None, "standard") == (None, None)


def test_protect_rules():
    rows = {"Relic": SimpleNamespace(type_line="Artifact"), "Gadget": SimpleNamespace(type_line="Artifact Creature"),
            "Bolt": SimpleNamespace(type_line="Instant")}
    protect = sb.make_protect({"Engine"}, {"type_contains": "Artifact", "min": 8}, rows)
    main = {"Engine": 4, "Relic": 4, "Gadget": 5, "Bolt": 4}
    assert protect(main, "Engine")  # requested
    assert protect(main, "Relic")  # would leave 5 artifacts, under 8
    assert not protect({**main, "Gadget": 8}, "Relic")  # leaves 8
    assert not protect(main, "Bolt")
    assert not sb.make_protect(set(), None, rows)(main, "Relic")


async def test_staples_map_front_faces_back_to_deck_names():
    result = MagicMock()
    result.all.return_value = [SimpleNamespace(k="esper origins"), SimpleNamespace(k="forest")]
    db = MagicMock(execute=AsyncMock(return_value=result))
    got = await sb.staples(db, ["Jeskai Control "], "standard", ["Esper Origins // Summon: Esper Maduin", "Bolt"])
    assert got == {"Esper Origins // Summon: Esper Maduin"}
    sql, params = db.execute.call_args.args
    assert params == {"format": "standard", "archetypes": ["jeskai control"]}
    assert "HAVING COUNT(DISTINCT lists.id) * 2 > (SELECT COUNT(*) FROM lists)" in str(sql)
    assert await sb.staples(db, [], "standard", ["Bolt"]) == set()


async def test_replacement_picks_from_a_slot_like_the_cut_card(monkeypatch):
    cut = SimpleNamespace(name="Dud", type_line="Instant", cmc=2, roles=["removal_targeted"], mana_cost="{1}{R}",
                          oracle_text="Deal 1 damage.")
    pool = [SimpleNamespace(name="Unscripted"), SimpleNamespace(name="Shock")]
    seen = {}

    async def slot_pool(db, slot, colors, format, chosen, archetypes=()):
        seen["slot"], seen["chosen"] = slot, chosen
        return pool

    async def fill_slot(client, slot, pool_, state, format, copies_for, question=None, allow_none=True):
        seen["pool"], seen["state"] = [r.name for r in pool_], state
        return [("Shock", 4)]
    monkeypatch.setattr(sb, "slot_pool", slot_pool)
    monkeypatch.setattr(sb, "fill_slot", fill_slot)
    got = await sb.replacement(None, object(), {"Dud": 3, "Forest": 22}, "Dud", ["Dimir Aggro"], ["R"], "standard",
                               {"shock", "dud", "forest"}, {"Dud": cut})
    assert got == "Shock"
    slot = seen["slot"]
    assert (slot.role, slot.cmc_min, slot.cmc_max, slot.copies) == ("removal_targeted", 1, 3, 3)
    assert seen["pool"] == ["Shock"]  # Forge can't play the other
    assert seen["state"]["deck"]["losing_to"] == ["Dimir Aggro"] and "Dud" in seen["state"]["deck"]["cut"]
    assert await sb.replacement(None, None, {"Dud": 3}, "Dud", [], ["R"], "standard", set(), {"Dud": cut}) is None


async def test_start_build_marks_the_run_failed_when_enqueue_breaks(monkeypatch):
    monkeypatch.setattr(sb, "sim_worker_running", lambda: True)

    def boom(run_id):
        raise ConnectionError("redis down")
    monkeypatch.setattr(sb, "enqueue", boom)
    db = MagicMock(add=MagicMock(), commit=AsyncMock())
    with pytest.raises(ConnectionError):
        await sb.start_build(db, DECK, ASSEMBLY, None, None, "standard")
    run = db.add.call_args.args[0]
    assert run.status == "failed" and run.error == "Couldn't start the playtest."
    assert db.commit.await_count == 2


def test_protect_counts_cards_added_by_swaps():
    rows = {"Relic": SimpleNamespace(type_line="Artifact"), "Gadget": SimpleNamespace(type_line="Artifact"),
            "Bolt": SimpleNamespace(type_line="Instant")}
    protect = sb.make_protect(set(), {"type_contains": "Artifact", "min": 8}, rows)
    main = {"Relic": 4, "Gadget": 4, "Bolt": 4, "Shiny": 4}
    assert protect(main, "Relic")  # 4 left, under 8
    rows["Shiny"] = SimpleNamespace(type_line="Artifact Creature")  # a swap added it
    assert not protect(main, "Relic")  # 8 left
    assert protect({**main, "Gadget": 3}, "Relic")  # 7 left


def _records(n):
    from app.services import forge
    return [forge.GameRecord(winner=forge.TESTED if i % 2 else forge.OPPONENT, turns=6,
                             own_turns={forge.TESTED: 6, forge.OPPONENT: 6},
                             mulligans={forge.TESTED: 0, forge.OPPONENT: 0},
                             casts={forge.TESTED: [(2, "Bolt")], forge.OPPONENT: []},
                             lands={forge.TESTED: [(1, "Forest")], forge.OPPONENT: []}) for i in range(n)]


def _build_run():
    from datetime import datetime
    from app.models.simulation import SimulationRun
    deck = {"name": "Brew", "main_deck": [{"card_name": "Forest", "quantity": 24}, {"card_name": "Bolt", "quantity": 36}],
            "sideboard": []}
    return SimulationRun(id=uuid4(), kind="build", format="standard", deck=deck, status="running", games_per_matchup=8,
                         stop_requested=False, created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
                         options={"requested": ["Bolt"], "archetypes": ["A"], "synergy": None, "colors": ["R"]})


def _wire(monkeypatch, opponents):
    import contextlib
    from app.services.deck_plan import RECENT  # noqa: F401
    from app.services.gauntlet import Opponent
    ops = [Opponent("A", 60.0, {"Forest": 24, "Shock": 36}), Opponent("B", 40.0, {"Forest": 24, "Shock": 36})]
    monkeypatch.setattr(sb, "gauntlet", AsyncMock(return_value=ops if opponents else []))
    monkeypatch.setattr(sb.forge, "card_names", lambda: {"forest", "bolt"})

    async def play(main, matchups, games, seed, on_progress=None):
        out = {}
        for m in matchups:
            out[m.key] = _records(games)
            if on_progress:
                await on_progress(m.key, out[m.key])
        return out
    monkeypatch.setattr(sb.forge, "play", play)
    rows = [SimpleNamespace(name="Forest", type_line="Basic Land", roles=[], cmc=0),
            SimpleNamespace(name="Bolt", type_line="Instant", roles=[], cmc=1)]
    monkeypatch.setattr(sb, "requested_rows", AsyncMock(return_value=rows))
    monkeypatch.setattr(sb, "staples", AsyncMock(return_value=set()))
    monkeypatch.setattr(sb, "replacement", AsyncMock(return_value=None))

    @contextlib.asynccontextmanager
    async def session(_):
        yield None
    monkeypatch.setattr(sb.jev, "session", session)

    @contextlib.asynccontextmanager
    async def nested():
        yield
    return MagicMock(commit=AsyncMock(), refresh=AsyncMock(), begin_nested=nested)


async def test_run_build_wires_the_search_to_the_run(monkeypatch):
    db = _wire(monkeypatch, True)
    run = _build_run()
    await sb.run_build(db, run)
    assert run.status == "completed" and run.error is None
    assert sum(e["quantity"] for e in run.final_deck["main_deck"]) == 60
    assert run.report["overall"] and run.report["baseline"]
    assert run.progress["stage"] == "Done"


async def test_run_build_without_opponents_raises(monkeypatch):
    from app.services.sim_runs import SimError
    db = _wire(monkeypatch, False)
    with pytest.raises(SimError):
        await sb.run_build(db, _build_run())


async def test_run_build_leaves_a_reaped_run_failed(monkeypatch):
    db = _wire(monkeypatch, True)
    run = _build_run()

    async def reaped(r, attrs):
        r.status = "failed"
    db.refresh = reaped
    await sb.run_build(db, run)
    assert run.status == "failed" and run.final_deck is None


def test_merge_entries_keeps_card_objects():
    card = {"type_line": "Instant"}
    entries = [{"card_name": "Bolt", "quantity": 4, "card": card}, {"card_name": "Dud", "quantity": 4, "card": {}}]
    assert sb.merge_entries(entries, {"Bolt": 3, "Shock": 1}) == [
        {"card_name": "Bolt", "quantity": 3, "card": card}, {"card_name": "Shock", "quantity": 1}]


def _swapped_search(monkeypatch):
    from app.services.deck_search import SearchResult
    final = {"Forest": 24, "Bolt": 32, "Shock": 4}

    async def search(seed_main, opponents, *a, **k):
        return SearchResult(final, [], [], {o.archetype: _records(8) for o in opponents},
                            [{"cut": "Bolt", "add": "Shock", "copies": 4}], "no_improvement")
    monkeypatch.setattr(sb, "playtest", search)
    return final


@pytest.mark.parametrize("edited", [False, True])
async def test_run_build_writes_the_deck_back_unless_the_user_edited_it(monkeypatch, edited):
    db = _wire(monkeypatch, True)
    final = _swapped_search(monkeypatch)
    run = _build_run()
    run.conversation_id = uuid4()
    card = {"type_line": "Instant"}
    deck = {"name": "Brew", "format": "standard", "sideboard": [],
            "main_deck": [{"card_name": "Forest", "quantity": 24, "card": {"type_line": "Basic Land"}},
                          {"card_name": "Bolt", "quantity": 35 if edited else 36, "card": card}]}
    conversation = SimpleNamespace(current_deck=deck)
    db.get = AsyncMock(return_value=conversation)
    await sb.run_build(db, run)
    assert run.status == "completed" and sb.main_of(run.final_deck["main_deck"]) == final
    if edited:
        assert conversation.current_deck is deck
    else:
        assert conversation.current_deck["name"] == "Brew"
        assert conversation.current_deck["main_deck"] == [
            {"card_name": "Forest", "quantity": 24, "card": {"type_line": "Basic Land"}},
            {"card_name": "Bolt", "quantity": 32, "card": card}, {"card_name": "Shock", "quantity": 4}]


async def test_run_build_tallies_only_the_baseline_in_the_live_table(monkeypatch):
    db = _wire(monkeypatch, True)
    tallies = []
    real = sb.Progress.games

    async def games(self, key, records, tally=True):
        tallies.append(tally)
        await real(self, key, records, tally)
    monkeypatch.setattr(sb.Progress, "games", games)
    monkeypatch.setattr(sb, "replacement", AsyncMock(return_value="Shock"))
    run = _build_run()
    run.options["requested"] = []  # so Bolt can be swapped and a candidate gets played
    await sb.run_build(db, run)
    assert tallies[:2] == [True, True] and len(tallies) > 2 and not any(tallies[2:])
