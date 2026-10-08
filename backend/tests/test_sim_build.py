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
