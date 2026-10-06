"""deck_fill: slot pools, Jev picks, copy rules and slot overflow."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.services import deck_fill as df
from app.services.deck_plan import Slot
from tests.jev_fake import FakeJev


def card(name, type_line="Instant", cmc=1, colors="R", roles=(), plays=10, mana_cost="{R}",
         oracle="Deal 2 damage.", identity=None):
    return SimpleNamespace(name=name, type_line=type_line, cmc=cmc, colors=list(colors), roles=list(roles),
                           plays=plays, mana_cost=mana_cost, oracle_text=oracle,
                           color_identity=list(identity if identity is not None else colors))


def fake_db(rows=()):
    result = MagicMock()
    result.all.return_value = list(rows)
    return MagicMock(execute=AsyncMock(return_value=result))


def ranks(*names):
    """FakeJev whose Choice puts `names` first, in order."""
    def answer(state, questions):
        probs = {n: 1.0 - i / 100 for i, n in enumerate(names)}
        return {"pick": {"choice": names[0], "confidence": 0.9, "probabilities": probs}}
    return FakeJev(answer=answer)


def no_copies(name, rank):
    return 0


class TestSlotPool:
    async def test_filters_and_params(self):
        db = fake_db([card("Shock")])
        slot = Slot("burn", 0, 1, 8, "Burn")
        rows = await df.slot_pool(db, slot, ["R", "W"], "standard", ["Lightning Strike"])
        assert [r.name for r in rows] == ["Shock"]
        stmt, params = db.execute.call_args[0]
        sql = str(stmt)
        assert params == {"format": "standard", "legality": "standard", "colors": ["R", "W"],
                          "cmc_min": 0, "cmc_max": 1, "chosen": ["Lightning Strike"],
                          "role": "burn", "type_contains": None}
        assert "e.date >= CURRENT_DATE - 14" in sql  # played in the window only
        assert "jsonb_array_elements(d.main_deck)" in sql
        assert "JOIN played p" in sql  # inner join: unplayed cards never qualify
        assert "c.legalities->>:legality = 'legal'" in sql
        assert "coalesce(nullif(c.colors, '{}'), c.color_identity, '{}') <@ CAST(:colors AS varchar[])" in sql
        assert "NOT LIKE '%Land%'" in sql
        assert "c.cmc BETWEEN :cmc_min AND :cmc_max" in sql
        assert "r.role = :role" in sql
        assert "NOT (c.name = ANY(CAST(:chosen AS varchar[])))" in sql
        assert "ORDER BY plays DESC, c.name" in sql and "LIMIT 255" in sql

    async def test_type_roles_and_type_contains(self):
        db = fake_db()
        await df.slot_pool(db, Slot("creature", 2, 2, 4, "x", "Equipment"), ["R"], "modern", [])
        sql = str(db.execute.call_args[0][0])
        assert "(c.type_line LIKE '%Creature%' OR c.type_line ILIKE '%' || :type_contains || '%')" in sql
        assert db.execute.call_args[0][1]["legality"] == "modern"

    def test_option_text(self):
        row = card("Shock", mana_cost="{R}", type_line="Instant", oracle="Shock deals 2 damage.\nDraw.", plays=7)
        assert df.option_text(row, "standard") == (
            "{R} Instant. Shock deals 2 damage. Draw. [Played in 7 recent Standard tournament decklists]")
        assert len(df.option_text(card("Long", oracle="x" * 500), "standard")) < 300


class TestFillSlot:
    async def test_brew_copies_follow_jev_rank(self):
        pool = [card(n) for n in ("A", "B", "C", "D", "E", "F")]
        jev = ranks("F", "E", "D", "C", "B", "A")
        picks = await df.fill_slot(jev, Slot("burn", 0, 1, 15, "Burn"), pool, {"deck": {}}, "standard",
                                   df.brew_copies)
        assert picks == [("F", 4), ("E", 4), ("D", 3), ("C", 2), ("B", 1), ("A", 1)]
        state, questions, _ = jev.calls[0]
        q = questions["pick"]
        assert q.instructions == {"question": df.SLOT_QUESTION, "slot": "Burn"}
        assert list(q.criteria) == ["A", "B", "C", "D", "E", "F"]

    async def test_reference_copies_capped_at_four_and_at_the_slot(self):
        pool = [card("A"), card("B"), card("C")]
        ref = {"A": 6, "B": 3}
        picks = await df.fill_slot(ranks("A", "B", "C"), Slot("burn", 0, 1, 6, ""), pool, {}, "standard",
                                   lambda n, r: ref.get(n) or df.brew_copies(n, r))
        assert picks == [("A", 4), ("B", 2)]

    async def test_at_least_one_copy(self):
        picks = await df.fill_slot(ranks("A"), Slot("burn", 0, 1, 2, ""), [card("A"), card("B")], {},
                                   "standard", no_copies)
        assert picks == [("A", 1), ("B", 1)]

    async def test_empty_pool_makes_no_jev_call(self):
        jev = ranks("A")
        assert await df.fill_slot(jev, Slot("burn", 0, 1, 4, ""), [], {}, "standard", df.brew_copies) == []
        assert jev.calls == []

    async def test_pool_runs_out(self):
        picks = await df.fill_slot(ranks("A"), Slot("burn", 0, 1, 12, ""), [card("A"), card("B")], {},
                                   "standard", df.brew_copies)
        assert picks == [("A", 4), ("B", 4)]


CARDS = [
    card("Bolt A", roles=["burn"], cmc=1), card("Bolt B", roles=["burn"], cmc=1),
    card("Bear A", "Creature", cmc=2, roles=["threat_cheap"]), card("Bear B", "Creature", cmc=2, roles=["threat_cheap"]),
    card("Bear C", "Creature", cmc=2, roles=["threat_cheap"]), card("Ogre", "Creature", cmc=3, roles=["threat_midrange"]),
    card("Giant", "Creature", cmc=6, roles=["threat_finisher"]), card("Wrath", "Sorcery", cmc=4, roles=["removal_mass"]),
]


def fake_pool(cards):
    calls = []

    async def pool(db, slot, colors, format, chosen):
        calls.append((slot.role, slot.copies, list(chosen)))
        return [c for c in cards
                if (slot.role == "any" or slot.role in c.roles)
                and slot.cmc_min <= c.cmc <= slot.cmc_max and c.name not in chosen
                and set(c.colors) <= set(colors)]
    pool.calls = calls
    return pool


class TestFillSlots:
    async def test_largest_first_and_shortfall_moves_to_same_role(self, monkeypatch):
        pool = fake_pool(CARDS)
        monkeypatch.setattr(df, "slot_pool", pool)
        build = df.Build()
        slots = [Slot("burn", 0, 1, 4, ""), Slot("threat_cheap", 2, 2, 16, ""), Slot("threat_cheap", 3, 4, 2, "")]
        short = await df.fill_slots(None, FakeJev(), slots, build, ["R"], "standard", dict, df.brew_copies)
        # threat_cheap 2-2 has 3 bears (4+4+3 = 11): 5 move to the other threat_cheap slot,
        # whose 3-4 pool has no threat_cheap cards, so 7 move to burn; burn has 2 cards (8).
        assert pool.calls[0][:2] == ("threat_cheap", 16)
        assert pool.calls[1][:2] == ("threat_cheap", 7)
        assert pool.calls[2][:2] == ("burn", 11)
        # 3 still short after the last slot: one catch-all slot over any played card
        assert pool.calls[3][:2] == ("any", 3)
        assert build.copies == {"Bear A": 4, "Bear B": 4, "Bear C": 3, "Bolt A": 4, "Bolt B": 4, "Ogre": 3}
        assert short == 0
        assert slots[1].copies == 16  # the plan's slots are not mutated

    async def test_returns_what_cannot_be_filled(self, monkeypatch):
        monkeypatch.setattr(df, "slot_pool", fake_pool([card("Bolt A", roles=["burn"])]))
        build = df.Build()
        short = await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], build, ["R"], "standard",
                                    dict, df.brew_copies)
        assert build.copies == {"Bolt A": 4} and short == 4

    async def test_cards_already_in_the_deck_are_excluded(self, monkeypatch):
        pool = fake_pool(CARDS)
        monkeypatch.setattr(df, "slot_pool", pool)
        build = df.Build()
        build.add("Bolt A", 4)  # a requested card
        await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 4, "")], build, ["R"], "standard", dict,
                            df.brew_copies)
        assert pool.calls[0][2] == ["Bolt A"]
        assert build.copies == {"Bolt A": 4, "Bolt B": 4}
