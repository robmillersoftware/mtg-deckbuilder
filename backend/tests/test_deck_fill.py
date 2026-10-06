"""deck_fill: slot pools, Jev picks, copy rules and slot overflow."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.services import deck_fill as df
from app.services.deck_plan import Plan, Slot
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


def requested_db(*rows):
    """execute().first() returns the next row (None: not found)."""
    results = [MagicMock(first=MagicMock(return_value=r)) for r in rows]
    return MagicMock(execute=AsyncMock(side_effect=results))


def plan(*slots, lands=22, nonbasic=None):
    return Plan(slots=list(slots), lands=lands, nonbasic_lands=nonbasic)


class TestReserveRequested:
    async def test_takes_four_copies_from_the_first_fitting_slot(self):
        p = plan(Slot("burn", 0, 1, 8), Slot("threat_cheap", 2, 2, 12))
        build = df.Build()
        db = requested_db(card("Shock", cmc=1, roles=["burn", "removal_targeted"]))
        colors = await df.reserve_requested(db, ["shock"], p, "standard", build)
        assert build.copies == {"Shock": 4} and colors == ["R"]
        assert [s.copies for s in p.slots] == [4, 12]
        stmt, params = db.execute.call_args[0]
        assert params == {"name": "shock", "legality": "standard"}
        assert "lower(split_part(c.name, ' // ', 1)) = lower(:name)" in str(stmt)  # front face finds a DFC
        assert df.CARD_COLORS + " AS colors" in str(stmt)

    async def test_legendary_is_one_copy_and_overflow_cuts_the_largest_slot(self):
        p = plan(Slot("burn", 0, 1, 2), Slot("threat_cheap", 2, 2, 12), Slot("removal_targeted", 0, 2, 6))
        build = df.Build()
        db = requested_db(card("Sephiroth, Fabled SOLDIER // Sephiroth, One-Winged Angel",
                               "Legendary Creature — Human // Legendary Creature — Angel", cmc=3,
                               colors="B", mana_cost=None, roles=["threat_midrange"]),  # SQL: identity
                          card("Big Burn", cmc=1, roles=["burn"]))
        colors = await df.reserve_requested(db, ["Sephiroth, Fabled SOLDIER", "Big Burn"], p, "standard", build)
        assert build.copies == {"Sephiroth, Fabled SOLDIER // Sephiroth, One-Winged Angel": 1, "Big Burn": 4}
        # Sephiroth fits no slot (threat_midrange 3): 1 off the largest. Big Burn: 2 from burn, 2 more off the largest.
        assert [s.copies for s in p.slots] == [0, 9, 6]
        assert colors == ["B", "R"]

    async def test_a_land_counts_toward_the_lands(self):
        p = plan(Slot("burn", 0, 1, 38), lands=22)
        build = df.Build()
        db = requested_db(card("Sacred Foundry", "Land — Mountain Plains", cmc=0, colors="RW",
                               mana_cost=None))  # SQL: a land's colors fall back to its identity
        colors = await df.reserve_requested(db, ["Sacred Foundry"], p, "standard", build)
        assert p.lands == 18 and p.slots[0].copies == 38
        assert colors == ["W", "R"]  # an off-color requested card brings its colors in

    async def test_unknown_or_illegal_names_and_repeats_are_skipped(self):
        p = plan(Slot("burn", 0, 1, 8))
        build = df.Build()
        db = requested_db(None, card("Shock"), card("Shock"))
        assert await df.reserve_requested(db, ["Nope", "Shock", "shock"], p, "standard", build) == ["R"]
        assert build.copies == {"Shock": 4} and p.slots[0].copies == 4


FLOODED_STRAND = ("{T}, Pay 1 life, Sacrifice this land: Search your library for a Plains or Island card, "
                  "put it onto the battlefield, then shuffle.")
BLOODSTAINED_MIRE = ("{T}, Pay 1 life, Sacrifice this land: Search your library for a Swamp or Mountain card, "
                     "put it onto the battlefield, then shuffle.")
EVOLVING_WILDS = ("{T}, Sacrifice this land: Search your library for a basic land card, "
                  "put it onto the battlefield tapped, then shuffle.")


class TestFits:
    def test_type_contains_or_role_and_cmc_band(self):
        slot = Slot("burn", 0, 2, 4, "x", type_contains="Saga")
        assert df._fits(slot, card("Saga", "Enchantment — Saga", cmc=2, roles=[]))
        assert df._fits(slot, card("Shock", cmc=1, roles=["burn"]))
        assert not df._fits(slot, card("Saga", "Enchantment — Saga", cmc=3, roles=[]))
        assert not df._fits(slot, card("Bear", "Creature", cmc=1, roles=[]))


class TestLandPool:
    async def test_fetchlands_must_find_a_deck_color(self):
        rows = [card(n, "Land", 0, "", identity="", mana_cost=None, oracle=o) for n, o in (
            ("Flooded Strand", FLOODED_STRAND), ("Bloodstained Mire", BLOODSTAINED_MIRE),
            ("Evolving Wilds", EVOLVING_WILDS), ("Sunbillow Verge", "{T}: Add {W}."))]
        got = await df.land_pool(fake_db(rows), ["R"], "standard", [])
        assert [r.name for r in got] == ["Bloodstained Mire", "Evolving Wilds", "Sunbillow Verge"]
        got = await df.land_pool(fake_db(rows), ["W", "U"], "standard", [])
        assert [r.name for r in got] == ["Flooded Strand", "Evolving Wilds", "Sunbillow Verge"]

    async def test_identity_subset_played_nonbasics(self):
        db = fake_db()
        await df.land_pool(db, ["R", "W"], "standard", ["Sacred Foundry"])
        stmt, params = db.execute.call_args[0]
        sql = str(stmt)
        assert params == {"format": "standard", "legality": "standard", "colors": ["R", "W"],
                          "chosen": ["Sacred Foundry"]}
        assert "coalesce(c.color_identity, '{}') <@ CAST(:colors AS varchar[])" in sql  # {} = colorless passes
        assert "LIKE '%Land%'" in sql and "c.type_line NOT LIKE 'Basic%'" in sql
        assert "JOIN played p" in sql and "LIMIT 255" in sql


class TestBasics:
    def test_pips_count_hybrid_and_fall_back_to_identity(self):
        rows = [card("A", mana_cost="{1}{R}{R}"), card("B", mana_cost="{R/W}{W}"),
                card("DFC", mana_cost=None, identity="B"), card("Artifact", mana_cost="{2}", colors="", identity=""),
                card("Front", mana_cost="{2}{R} // {1}{G}", identity="RG")]
        assert df.pips([(r, 1) for r in rows]) == {"W": 2, "U": 0, "B": 1, "R": 4, "G": 0}
        assert df.pips([(rows[0], 4)])["R"] == 8  # weighted by copies

    def test_split_by_pips_with_one_per_color(self):
        assert df.split_basics({"R": 30, "W": 0}, ["R", "W"], 10) == {"Plains": 1, "Mountain": 9}
        assert df.split_basics({"R": 6, "W": 2}, ["W", "R"], 9) == {"Plains": 3, "Mountain": 6}
        assert df.split_basics({}, ["U", "R"], 4) == {"Island": 2, "Mountain": 2}
        assert df.split_basics({"R": 1}, ["W", "U", "R"], 2) == {"Mountain": 2}  # too few for one each
        assert df.split_basics({"R": 1}, ["R"], 0) == {}

    def test_brew_nonbasics_by_color_count(self):
        assert [df.brew_nonbasics(c) for c in (["R"], ["R", "W"], ["U", "B", "R"], list("WUBRG"))] == [0, 4, 6, 6]


class TestFillLands:
    async def test_basics_are_weighted_by_copies(self, monkeypatch):
        monkeypatch.setattr(df, "land_pool", AsyncMock(return_value=[]))
        build = df.Build()
        build.add("Shock", 4, card("Shock", mana_cost="{R}{R}"))
        build.add("Get Lost", 1, card("Get Lost", mana_cost="{W}{W}", colors="W"))
        await df.fill_lands(None, FakeJev(), build, plan(lands=4, nonbasic=0), ["R", "W"], "standard",
                            dict, df.brew_copies)
        assert (build.copies["Mountain"], build.copies["Plains"]) == (3, 1)  # unweighted would be 2/2

    async def test_reference_nonbasics_then_basics_by_pips(self, monkeypatch):
        lands = [card("Sacred Foundry", "Land — Mountain Plains", 0, "", identity="RW", mana_cost=None),
                 card("Inspiring Vantage", "Land", 0, "", identity="RW", mana_cost=None)]
        monkeypatch.setattr(df, "land_pool", AsyncMock(return_value=lands))
        build = df.Build()
        build.add("Shock", 4, card("Shock", mana_cost="{R}"))
        build.add("Get Lost", 2, card("Get Lost", mana_cost="{1}{W}{W}", colors="W"))
        p = plan(lands=10, nonbasic=6)
        await df.fill_lands(None, ranks("Inspiring Vantage", "Sacred Foundry"), build, p, ["R", "W"],
                            "standard", dict, lambda n, r: 4)
        assert build.copies["Inspiring Vantage"] == 4 and build.copies["Sacred Foundry"] == 2
        assert build.copies["Mountain"] + build.copies["Plains"] == 4
        assert build.total() == 6 + 10


    async def test_requested_nonbasic_counts_and_mono_color_brew_gets_only_basics(self, monkeypatch):
        pool = AsyncMock(return_value=[])
        monkeypatch.setattr(df, "land_pool", pool)
        build = df.Build()
        build.add("Shock", 4, card("Shock"))
        await df.fill_lands(None, FakeJev(), build, plan(lands=22), ["R"], "standard", dict, df.brew_copies)
        pool.assert_not_called()  # brew rule: 0 nonbasics for one color
        assert build.copies == {"Shock": 4, "Mountain": 22}

        build = df.Build()
        build.add("Sacred Foundry", 4, card("Sacred Foundry", "Land", 0, "", identity="RW", mana_cost=None))
        await df.fill_lands(None, FakeJev(), build, plan(lands=18), ["R", "W"], "standard", dict, df.brew_copies)
        pool.assert_not_called()  # 2-color brew wants 4 nonbasics; the requested Foundry already is 4
        assert build.copies["Mountain"] + build.copies["Plains"] == 18
