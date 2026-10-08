"""The gauntlet: the top meta decks as recent real lists."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.services import gauntlet as g

SNAPSHOTS = [SimpleNamespace(archetype="Dimir Aggro", meta_percentage=10.9),
             SimpleNamespace(archetype="Boros Dragons ", meta_percentage=10.3),
             SimpleNamespace(archetype="No Lists", meta_percentage=9.0)]


def fake_db(lists):
    """execute(): first the snapshot query, then one list query per archetype tried."""
    snap = MagicMock()
    snap.scalars.return_value.all.return_value = SNAPSHOTS

    def result(row):
        r = MagicMock()
        r.first.return_value = None if row is None else SimpleNamespace(main_deck=row)
        return r
    return MagicMock(execute=AsyncMock(side_effect=[snap] + [result(x) for x in lists]))


async def test_meta_gauntlet_uses_shares_and_skips_archetypes_without_lists(monkeypatch):
    monkeypatch.setattr(g, "GAUNTLET_SIZE", 2)
    db = fake_db([[{"card_name": "Island", "quantity": 24}], [{"card_name": "Mountain", "quantity": "22"}]])
    got = await g.gauntlet(db, "standard")
    assert got == [g.Opponent("Dimir Aggro", 10.9, {"Island": 24}), g.Opponent("Boros Dragons", 10.3, {"Mountain": 22})]
    list_sql, params = db.execute.call_args_list[1].args
    assert params == {"format": "standard", "archetype": "Dimir Aggro"}
    snap_sql = str(db.execute.call_args_list[0].args[0])
    assert "max(" in snap_sql and "snapshot_date" in snap_sql
    assert "lower(trim(d.archetype)) = lower(trim(:archetype))" in str(list_sql)


async def test_chosen_archetypes_weigh_equally_and_skip_missing():
    db = fake_db([None, [], [{"card_name": "Forest", "quantity": 20}]])
    got = await g.gauntlet(db, "standard", ["No Lists", "no lists ", "Empty", "Mono Green", "mono green"])
    assert got == [g.Opponent("Mono Green", 1.0, {"Forest": 20})]


async def test_duplicate_lines_in_a_list_are_summed():
    db = fake_db([[{"card_name": "Forest", "quantity": 2}, {"card_name": "Bear", "quantity": 4},
                   {"card_name": "Forest", "quantity": "18"}]])
    got = await g.gauntlet(db, "standard", ["Mono Green"])
    assert got[0].main == {"Forest": 20, "Bear": 4}
