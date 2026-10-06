"""deck_fill: slot pools, Jev picks, copy rules and slot overflow."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

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

    async def test_archetype_scope_is_bound(self):
        db = fake_db()
        await df.slot_pool(db, Slot("burn", 0, 1, 8, "Burn"), ["R"], "standard", [], ["Boros Aggro"])
        stmt, params = db.execute.call_args[0]
        assert "AND lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))" in str(stmt)
        assert params["archetypes"] == ["boros aggro"]
        db = fake_db()
        await df.slot_pool(db, Slot("burn", 0, 1, 8, "Burn"), ["R"], "standard", [])
        assert "archetype" not in str(db.execute.call_args[0][0])
        assert "archetypes" not in db.execute.call_args[0][1]

    async def test_a_set_of_archetypes_is_one_bound_list(self):
        db = fake_db()
        await df.slot_pool(db, Slot("burn", 0, 1, 8, "Burn"), ["R"], "standard", [],
                           ["Rakdos Aggro", " rakdos aggro ", "4/5C Control", "4/5c Control"])
        stmt, params = db.execute.call_args[0]
        assert params["archetypes"] == ["4/5c control", "rakdos aggro"]  # case and space duplicates merge
        assert str(stmt).count(":archetypes") == 1

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
        assert list(q.criteria) == ["A", "B", "C", "D", "E", "F", df.NO_FIT]

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

    async def test_pick_counts_when_probabilities_are_empty(self):
        jev = FakeJev(answer=lambda s, q: {"pick": {"choice": "B", "confidence": 0.9, "probabilities": {}}})
        picks = await df.fill_slot(jev, Slot("burn", 0, 1, 4, ""), [card("A"), card("B")], {}, "standard",
                                   df.brew_copies)
        assert picks == [("B", 4)]

    async def test_question_forbids_picks_that_work_against_the_plan(self):
        assert "never pick one that works against it" in df.SLOT_QUESTION

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


def fake_pool(cards, ref_names=()):
    """Pool over `cards`; with archetypes only `ref_names` qualify. `scopes`
    records each call's archetypes joined by ", " (None: the whole format)."""
    calls, scopes = [], []

    async def pool(db, slot, colors, format, chosen, archetypes=()):
        calls.append((slot.role, slot.copies, list(chosen)))
        scopes.append(", ".join(archetypes) or None)
        return [c for c in cards
                if (not archetypes or c.name in ref_names)
                and (slot.role == "any" or slot.role in c.roles)
                and slot.cmc_min <= c.cmc <= slot.cmc_max and c.name not in chosen
                and set(c.colors) <= set(colors)]
    pool.calls, pool.scopes = calls, scopes
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


class TestReferenceFill:
    async def test_reference_pool_first_then_format_for_the_shortfall(self, monkeypatch):
        pool = fake_pool(CARDS, ref_names={"Bolt A"})
        monkeypatch.setattr(df, "slot_pool", pool)
        build = df.Build()
        short = await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], build, ["R"], "standard", dict,
                                    df.brew_copies, ["Boros Aggro"])
        assert pool.scopes == ["Boros Aggro", None]
        assert pool.calls[0][:2] == ("burn", 8) and pool.calls[1][:2] == ("burn", 4)  # shortfall only
        assert pool.calls[1][2] == ["Bolt A"]  # chosen cards excluded from the format pool
        assert build.copies == {"Bolt A": 4, "Bolt B": 4} and short == 0

    async def test_full_reference_pool_skips_the_format(self, monkeypatch):
        pool = fake_pool(CARDS, ref_names={"Bolt A", "Bolt B"})
        monkeypatch.setattr(df, "slot_pool", pool)
        await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], df.Build(), ["R"], "standard", dict,
                            df.brew_copies, ["Boros Aggro"])
        assert pool.scopes == ["Boros Aggro"]

    async def test_a_set_of_relatives_is_one_scope(self, monkeypatch):
        pool = fake_pool(CARDS, ref_names={"Bolt A"})
        monkeypatch.setattr(df, "slot_pool", pool)
        await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], df.Build(), ["R"], "standard", dict,
                            df.brew_copies, ["Rakdos Aggro", "Boros Aggro"])
        assert pool.scopes == ["Rakdos Aggro, Boros Aggro", None]  # one pool over all of them, then the format

    async def test_brew_uses_the_format_pool_only(self, monkeypatch):
        pool = fake_pool(CARDS)
        monkeypatch.setattr(df, "slot_pool", pool)
        await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], df.Build(), ["R"], "standard", dict,
                            df.brew_copies)
        assert pool.scopes == [None]

    async def test_catch_all_is_also_two_stage(self, monkeypatch):
        pool = fake_pool([card("Bolt A", roles=["burn"]), card("Ogre", "Creature", cmc=3, roles=["x"])],
                         ref_names={"Bolt A"})
        monkeypatch.setattr(df, "slot_pool", pool)
        build = df.Build()
        short = await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], build, ["R"], "standard", dict,
                                    df.brew_copies, ["Boros Aggro"])
        assert build.copies == {"Bolt A": 4, "Ogre": 4} and short == 0  # Ogre only fits the catch-all
        assert pool.scopes == ["Boros Aggro", None, "Boros Aggro", None]
        assert pool.calls[2][:2] == ("any", 4)


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


class TestReserveOverflow:
    async def reserve(self, rows, p):
        build = df.Build()
        await df.reserve_requested(requested_db(*rows), [r.name for r in rows], p, "standard", build)
        return build, None

    async def test_requested_creatures_spill_into_lands(self):
        rows = [card(f"Req {i}", "Creature", 2, roles=["threat_cheap"]) for i in range(10)]
        p = plan(Slot("threat_cheap", 2, 2, 18), Slot("burn", 0, 1, 20), lands=22)
        build, _ = await self.reserve(rows, p)
        assert build.total() + sum(s.copies for s in p.slots) + p.lands == 60 and build.total() == 40

    async def test_requested_lands_spill_into_slots(self):
        rows = [card(f"Land {i}", "Land", 0, "", identity="R") for i in range(7)]
        p = plan(Slot("burn", 0, 1, 38), lands=22)
        build, _ = await self.reserve(rows, p)
        assert build.total() + sum(s.copies for s in p.slots) + p.lands == 60

    async def test_requests_alone_over_60_cut_the_last_requested(self):
        rows = [card(f"Req {i}", "Creature", 2, roles=["threat_cheap"]) for i in range(17)]
        p = plan(Slot("threat_cheap", 2, 2, 38), lands=22)
        build, _ = await self.reserve(rows, p)
        assert build.total() == 60 and set(build.copies) == {r.name for r in rows}
        assert build.copies["Req 0"] == 4 and build.copies["Req 16"] == 1


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
        monkeypatch.setattr(df, "brew_nonbasic_target",
                            AsyncMock(side_effect=lambda db, colors, format: df.brew_nonbasics(colors)))
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


class TestSummarize:
    async def test_template_without_llm(self):
        main = df.Build()
        assert await df.summarize(main, df.Build(), "Boros aggro", "Boros Aggro", ["W", "R"], "standard") == (
            "Boros Aggro", "Built from recent Standard tournament lists of Boros Aggro.")
        assert await df.summarize(main, df.Build(), "mono-red burn", None, ["R"], "standard") == (
            "Red Deck", "A Standard brew for: mono-red burn")

    async def test_llm_name_and_summary(self, monkeypatch):
        seen = {}

        def complete(system, user, max_tokens=4096):
            seen["user"] = user
            return 'Sure: {"name": "Red Rush", "strategy_summary": "Attack early."}'

        monkeypatch.setattr(df.llm, "is_configured", lambda: True)
        monkeypatch.setattr(df.llm, "complete", complete)
        main, side = df.Build(), df.Build()
        main.add("Shock", 4, card("Shock", mana_cost="{R}", oracle="Shock deals 2 damage to any target."))
        main.add("Mountain", 18)  # basics have no row
        side.add("Abrade", 2)
        assert await df.summarize(main, side, "burn", None, ["R"], "standard") == ("Red Rush", "Attack early.")
        assert "4 Shock ({R} Instant): Shock deals 2 damage to any target." in seen["user"]
        assert "18 Mountain" in seen["user"] and "Sideboard:\n2 Abrade" in seen["user"]
        assert "only on the cards' rules text" in df.SUMMARY_SYSTEM

    async def test_bad_llm_reply_uses_the_template(self, monkeypatch):
        monkeypatch.setattr(df.llm, "is_configured", lambda: True)
        monkeypatch.setattr(df.llm, "complete", lambda *a, **k: '{"name": ""}')
        assert (await df.summarize(df.Build(), df.Build(), "x", None, ["U", "R"], "standard"))[0] == "Blue-Red Deck"


MAIN_POOL = [
    card(f"Bear {i}", "Creature", cmc=2, roles=["threat_cheap"], colors="RW"[i % 2]) for i in range(20)
] + [card(f"Bolt {i}", cmc=1, roles=["burn"]) for i in range(10)] + [
    card(f"Hate {i}", "Enchantment", cmc=2, roles=["graveyard_hate"], colors="W") for i in range(8)]
RED = {c.name for c in MAIN_POOL if c.colors == ["R"]}
LANDS = [card("Sacred Foundry", "Land", 0, "", identity="RW", mana_cost=None),
         card("Inspiring Vantage", "Land", 0, "", identity="RW", mana_cost=None)]


def side_card(name, copies, colors="R"):
    row = card(name, "Instant", 2, colors)
    row.copies = copies
    return row


def fake_side_pool(by_scope):
    """`by_scope` keys: archetypes joined by ", ", or None for the whole format."""
    calls = []

    async def pool(db, colors, format, chosen, archetypes=()):
        key = ", ".join(archetypes) or None
        calls.append((key, list(chosen)))
        return [c for c in by_scope.get(key, []) if c.name not in chosen and set(c.colors) <= set(colors)]
    pool.calls = calls
    return pool


class TestSideboardPool:
    async def test_reference_sideboards(self):
        db = fake_db([side_card("Abrade", 2)])
        rows = await df.sideboard_pool(db, ["R", "W"], "standard", ["Shock"], ["Boros Aggro"])
        assert [r.name for r in rows] == ["Abrade"]
        stmt, params = db.execute.call_args[0]
        sql = str(stmt)
        assert params == {"format": "standard", "legality": "standard", "colors": ["R", "W"],
                          "chosen": ["Shock"], "archetypes": ["boros aggro"]}
        assert "jsonb_array_elements(d.sideboard)" in sql and "d.main_deck" not in sql
        assert "e.date >= CURRENT_DATE - 14" in sql and "JOIN side s" in sql  # played sideboard cards only
        assert "lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))" in sql
        assert df.CARD_COLORS + " <@ CAST(:colors AS varchar[])" in sql
        assert "GREATEST(1, ROUND(MAX(s.avg_copies)))::int AS copies" in sql
        assert "NOT (c.name = ANY(CAST(:chosen AS varchar[])))" in sql
        assert "ORDER BY plays DESC, c.name" in sql and "LIMIT 255" in sql

    async def test_format_sideboards_for_a_brew(self):
        db = fake_db()
        await df.sideboard_pool(db, ["R"], "standard", [])
        stmt, params = db.execute.call_args[0]
        assert "d.archetype" not in str(stmt) and "archetypes" not in params


class TestFillSideboard:
    async def test_a_short_sideboard_raises(self, monkeypatch):
        monkeypatch.setattr(df, "sideboard_pool", fake_side_pool({None: [side_card("Side 0", 4)]}))
        with pytest.raises(ValueError):
            await df.fill_sideboard(None, FakeJev(), df.Build(), df.Build(), ["R"], "standard", dict, None)

    async def test_reference_cards_then_format_top_up_to_exactly_15(self, monkeypatch):
        pool = fake_side_pool({
            "Boros Aggro": [side_card("Rest in Peace", 2, "W"), side_card("Abrade", 3), side_card("Duress", 2, "B")],
            None: [side_card("Abrade", 3), side_card("Sear", 6), side_card("Get Lost", 2, "W"),
                   side_card("Ghost Vessel", 1, ""), side_card("Pyroclasm", 4), side_card("Shock", 4)],
        })
        monkeypatch.setattr(df, "sideboard_pool", pool)
        main, side = df.Build(), df.Build()
        main.add("Shock", 4)
        jev = FakeJev()
        await df.fill_sideboard(None, jev, main, side, ["R", "W"], "standard",
                                lambda: {"deck": {"chosen": ["4x Shock"]}}, ["Boros Aggro"])
        assert pool.calls == [("Boros Aggro", ["Shock"]), (None, ["Shock", "Rest in Peace", "Abrade"])]
        # average copies (Sear's 6 capped at 4), the last pick cut to land on exactly 15
        assert side.copies == {"Rest in Peace": 2, "Abrade": 3, "Sear": 4, "Get Lost": 2, "Ghost Vessel": 1,
                               "Pyroclasm": 3}
        state, questions, _ = jev.calls[0]
        assert questions["pick"].instructions["question"] == df.SIDEBOARD_QUESTION
        assert state == {"deck": {"chosen": ["4x Shock"]}}

    async def test_brew_uses_the_format_sideboards(self, monkeypatch):
        pool = fake_side_pool({None: [side_card(f"Side {i}", 4) for i in range(6)]})
        monkeypatch.setattr(df, "sideboard_pool", pool)
        side = df.Build()
        await df.fill_sideboard(None, FakeJev(), df.Build(), side, ["R"], "standard", dict, None)
        assert pool.calls == [(None, [])]
        assert side.copies == {"Side 0": 4, "Side 1": 4, "Side 2": 4, "Side 3": 3}

    async def test_a_full_reference_pool_needs_no_top_up(self, monkeypatch):
        pool = fake_side_pool({"UW Control": [side_card(f"Side {i}", 3, "W") for i in range(5)]})
        monkeypatch.setattr(df, "sideboard_pool", pool)
        side = df.Build()
        await df.fill_sideboard(None, FakeJev(), df.Build(), side, ["W", "U"], "standard", dict, ["UW Control"])
        assert [c[0] for c in pool.calls] == ["UW Control"] and side.total() == 15


def wire(monkeypatch, reference_plan=None, archetypes=(("Boros Aggro", 5),), candidates=()):
    pool = fake_pool(MAIN_POOL)
    monkeypatch.setattr(df, "recent_archetypes", AsyncMock(return_value=list(archetypes)))
    monkeypatch.setattr(df, "relative_candidates", AsyncMock(return_value=list(candidates)))
    monkeypatch.setattr(df, "plan_from_decklists", AsyncMock(return_value=reference_plan))
    monkeypatch.setattr(df, "slot_pool", pool)
    monkeypatch.setattr(df, "land_pool", AsyncMock(return_value=LANDS))
    monkeypatch.setattr(df, "brew_nonbasic_target",
                        AsyncMock(side_effect=lambda db, colors, format: df.brew_nonbasics(colors)))
    side = fake_side_pool({"Boros Aggro": [side_card(f"Side {i}", 3, "RW"[i % 2]) for i in range(6)],
                           None: [side_card(f"Spare {i}", 2) for i in range(10)]})
    monkeypatch.setattr(df, "sideboard_pool", side)
    return pool, side


def reference_plan():
    return Plan(slots=[Slot("threat_cheap", 2, 2, 20, "Bears"), Slot("burn", 0, 1, 16, "Burn")],
                lands=24, nonbasic_lands=8, copies={"Bear 0": 4, "Bear 1": 2, "Sacred Foundry": 4},
                colors=["W", "R"], reference="Boros Aggro")


def total(entries):
    return sum(e["quantity"] for e in entries)


class TestAssemble:
    async def test_reference_deck_is_60_and_15(self, monkeypatch):
        _, side_pool = wire(monkeypatch, reference_plan())
        jev = FakeJev(answer=lambda state, q: (
            {"pick": {"choice": "Boros Aggro", "confidence": 0.9, "probabilities": {}}}
            if "none" in q["pick"].criteria else {}))
        deck = await df.assemble(None, "Boros aggro", ["R", "W"], [], "standard", True, "aggro", client=jev)
        assert total(deck["main_deck"]) == 60 and total(deck["sideboard"]) == 15
        assert deck["name"] == "Boros Aggro" and deck["reference"] == "Boros Aggro"
        assert deck["colors"] == ["W", "R"]
        main = {e["card_name"]: e["quantity"] for e in deck["main_deck"]}
        assert main["Bear 0"] == 4 and main["Bear 1"] == 2  # reference copies
        assert main["Sacred Foundry"] == 4
        side = {e["card_name"] for e in deck["sideboard"]}
        assert not side & set(main)  # copies stay within 4 across main and sideboard
        assert side_pool.calls[0][0] == "Boros Aggro" and set(side_pool.calls[0][1]) == set(main)
        state = jev.calls[-1][0]["deck"]
        assert state["plan"] == "Boros Aggro, a current Standard archetype" and state["colors"] == ["W", "R"]
        df.plan_from_decklists.assert_awaited_once_with(None, ["Boros Aggro"], "standard")

    async def test_brew_gets_a_format_sideboard_and_basic_land_split(self, monkeypatch):
        _, side_pool = wire(monkeypatch)
        brew = AsyncMock(return_value=Plan(slots=[Slot("threat_cheap", 2, 2, 18, "Bears"),
                                                  Slot("burn", 0, 1, 20, "Burn")], lands=22))
        monkeypatch.setattr(df, "plan_with_llm", brew)
        jev = FakeJev(answer=lambda state, q: (
            {"pick": {"choice": "none", "confidence": 0.9, "probabilities": {}}}
            if "none" in q["pick"].criteria else {}))
        deck = await df.assemble(None, "mono-red aggro", ["R"], [], "standard", True, "aggro", client=jev)
        main = {e["card_name"]: e["quantity"] for e in deck["main_deck"]}
        assert total(deck["main_deck"]) == 60 and total(deck["sideboard"]) == 15 and deck["reference"] is None
        assert [c[0] for c in side_pool.calls] == [None]  # the format's sideboard cards
        assert main["Mountain"] == 22 and "Sacred Foundry" not in main  # mono-color brew: basics only
        assert {n for n in main if n.startswith("Bear")} <= RED  # on-color only
        brew.assert_awaited_once_with("mono-red aggro", ["R"], "aggro")

    async def test_reference_colors_when_none_were_parsed(self, monkeypatch):
        wire(monkeypatch, reference_plan())
        jev = FakeJev(answer=lambda state, q: (
            {"pick": {"choice": "Boros Aggro", "confidence": 0.9, "probabilities": {}}}
            if "none" in q["pick"].criteria else {}))
        await df.assemble(None, "Build the best deck", [], [], "standard", False, "", client=jev)
        assert jev.calls[-1][0]["deck"]["colors"] == ["W", "R"]

    async def test_many_requested_cards_still_make_exactly_60(self, monkeypatch):
        wire(monkeypatch)
        monkeypatch.setattr(df, "plan_with_llm", AsyncMock(return_value=Plan(
            slots=[Slot("threat_cheap", 2, 2, 18, "Bears"), Slot("burn", 0, 1, 20, "Burn")], lands=22)))
        reqs = [card(f"Req {i}", "Creature", 2, roles=["threat_cheap"]) for i in range(10)] + [
            card(f"Req Land {i}", "Land", 0, "", identity="R", mana_cost=None) for i in range(7)]
        res = [MagicMock(first=MagicMock(return_value=r)) for r in reqs]
        db = MagicMock(execute=AsyncMock(side_effect=res))
        jev = FakeJev(answer=lambda state, q: {"pick": {"choice": "none", "confidence": 0.9, "probabilities": {}}}
                      if "none" in q["pick"].criteria else {})
        deck = await df.assemble(db, "x", ["R"], [r.name for r in reqs], "standard", True, "", client=jev)
        main = {e["card_name"]: e["quantity"] for e in deck["main_deck"]}
        assert total(deck["main_deck"]) == 60 and total(deck["sideboard"]) == 15
        assert {r.name for r in reqs} <= set(main)

    async def test_without_colors_a_requested_card_sets_them_before_planning(self, monkeypatch):
        # "build around Weapons Manufacturing": no colors stated, so the red card decides them
        wire(monkeypatch, candidates=[("Boros Aggro", 5, [("Bear 1", 4.0)])])
        monkeypatch.setattr(df, "plan_with_llm", AsyncMock(return_value=Plan(
            slots=[Slot("threat_cheap", 2, 2, 18, "Bears"), Slot("burn", 0, 1, 20, "Burn")], lands=22)))
        req = card("Weapons Manufacturing", "Enchantment", 2, "R", mana_cost="{1}{R}")
        db = MagicMock(execute=AsyncMock(return_value=MagicMock(first=MagicMock(return_value=req))))
        jev = FakeJev(answer=lambda state, q: {"relative": 0.3} if "relative" in q else (
            {"pick": {"choice": "none", "confidence": 0.9, "probabilities": {}}}
            if "none" in q["pick"].criteria else {}))
        deck = await df.assemble(db, "around Weapons Manufacturing", None, ["Weapons Manufacturing"],
                                 "standard", True, "", client=jev)
        df.relative_candidates.assert_awaited_once_with(db, ["R"], "standard")
        df.plan_with_llm.assert_awaited_once_with("around Weapons Manufacturing", ["R"], "")
        assert deck["colors"] == ["R"]

    async def test_brew_with_relatives_is_planned_and_filled_from_them(self, monkeypatch):
        candidates = [("Boros Aggro", 5, [("Bear 1", 4.0)]), ("Rakdos Aggro", 4, [("Bolt 0", 4.0)])]
        pool, side_pool = wire(monkeypatch, candidates=candidates)
        relatives_plan = Plan(slots=[Slot("threat_cheap", 2, 2, 18, "Bears"), Slot("burn", 0, 1, 20, "Burn")],
                              lands=22, copies={"Bolt 0": 2}, colors=["R"], reference="Boros Aggro")
        monkeypatch.setattr(df, "plan_from_decklists", AsyncMock(return_value=relatives_plan))
        llm_plan = AsyncMock()
        monkeypatch.setattr(df, "plan_with_llm", llm_plan)

        def answer(state, q):
            if "relative" in q:
                return {"relative": 0.9 if state["archetype"]["name"] == "Boros Aggro" else 0.2}
            return {"pick": {"choice": "none", "confidence": 0.9, "probabilities": {}}} \
                if "none" in q["pick"].criteria else {}
        jev = FakeJev(answer=answer)
        deck = await df.assemble(None, "mono-red aggro", ["R"], [], "standard", True, "aggro", client=jev)

        assert deck["reference"] is None and deck["relatives"] == ["Boros Aggro"]
        df.relative_candidates.assert_awaited_once_with(None, ["R"], "standard")
        df.plan_from_decklists.assert_awaited_once_with(None, ["Boros Aggro"], "standard", ["R"])
        llm_plan.assert_not_called()
        assert pool.scopes[0] == "Boros Aggro"  # relatives' cards first, then the format
        assert side_pool.calls[0][0] == "Boros Aggro"
        main = {e["card_name"]: e["quantity"] for e in deck["main_deck"]}
        assert total(deck["main_deck"]) == 60 and total(deck["sideboard"]) == 15
        assert main["Bolt 0"] == 2  # copies from the relatives' lists
        assert not {"Sacred Foundry", "Inspiring Vantage"} & set(main)  # brew nonbasic rule: 0 for one color
        slot_states = [state["deck"] for state, _, _ in jev.calls if "deck" in state]
        assert slot_states and all(
            s["plan"] == "mono-red aggro, built from the current archetypes closest to it" for s in slot_states)
        assert not any("Boros Aggro" in str(s) for s in slot_states)  # archetype names are not instructions

    async def test_no_relatives_uses_the_llm_plan(self, monkeypatch):
        pool, side_pool = wire(monkeypatch, candidates=[("Boros Aggro", 5, [("Bear 1", 4.0)])])
        monkeypatch.setattr(df, "plan_with_llm", AsyncMock(return_value=Plan(
            slots=[Slot("threat_cheap", 2, 2, 18, "Bears"), Slot("burn", 0, 1, 20, "Burn")], lands=22)))
        jev = FakeJev(answer=lambda state, q: {"relative": 0.3} if "relative" in q else (
            {"pick": {"choice": "none", "confidence": 0.9, "probabilities": {}}}
            if "none" in q["pick"].criteria else {}))
        deck = await df.assemble(None, "mono-red aggro", ["R"], [], "standard", True, "aggro", client=jev)
        assert deck["relatives"] == [] and total(deck["main_deck"]) == 60 and total(deck["sideboard"]) == 15
        df.plan_from_decklists.assert_not_called()
        df.plan_with_llm.assert_awaited_once_with("mono-red aggro", ["R"], "aggro")
        assert set(pool.scopes) == {None} and [c[0] for c in side_pool.calls] == [None]
        assert all(s["deck"]["plan"] == "mono-red aggro" for s, _, _ in jev.calls if "deck" in s)

    async def test_relatives_jev_failure_propagates(self, monkeypatch):
        wire(monkeypatch, candidates=[("Boros Aggro", 5, [("Bear 1", 4.0)])])
        jev = FakeJev(answer=lambda state, q: {"pick": "none"} if "pick" in q else {},
                      fail=lambda state: "archetype" in state)
        with pytest.raises(ExceptionGroup) as err:
            await df.assemble(None, "mono-red aggro", ["R"], [], "standard", True, "aggro", client=jev)
        assert err.group_contains(RuntimeError, match="jev unavailable")

    async def test_slow_jev_hits_the_deadline(self, monkeypatch):
        wire(monkeypatch, reference_plan())
        monkeypatch.setattr(df, "ASSEMBLY_DEADLINE", 0.05)
        with pytest.raises(TimeoutError):
            await df.assemble(None, "Boros aggro", ["R", "W"], [], "standard", True, "aggro",
                              client=FakeJev(delay=1.0))

    async def test_not_sixty_card_format(self):
        with pytest.raises(ValueError):
            await df.assemble(None, "x", ["B"], [], "cedh", False, client=FakeJev())

    async def test_no_jev_key(self):
        with pytest.raises(RuntimeError):
            await df.assemble(None, "x", ["R"], [], "standard")  # conftest blanks TYPESAFE_API_KEY

    async def test_no_recent_decklists(self, monkeypatch):
        wire(monkeypatch, archetypes=())
        with pytest.raises(ValueError):
            await df.assemble(None, "x", ["R"], [], "standard", client=FakeJev())

    async def test_brew_without_colors(self, monkeypatch):
        wire(monkeypatch)
        jev = FakeJev(answer=lambda state, q: {"pick": "none"})
        with pytest.raises(ValueError):
            await df.assemble(None, "something fun", [], [], "standard", client=jev)

    async def test_jev_failure_propagates(self, monkeypatch):
        wire(monkeypatch, reference_plan())
        with pytest.raises(ExceptionGroup) as err:  # jev.ask_many runs its asks in a TaskGroup
            await df.assemble(None, "x", ["R"], [], "standard", client=FakeJev(fail=lambda s: True))
        assert err.group_contains(RuntimeError, match="jev unavailable")



def ranks_with_no_fit(*names_then_none):
    """FakeJev ranking `names_then_none` in order; df.NO_FIT may appear among them."""
    def answer(state, questions):
        probs = {n: 1.0 - i / 100 for i, n in enumerate(names_then_none)}
        return {"pick": {"choice": names_then_none[0], "confidence": 0.9, "probabilities": probs}}
    return FakeJev(answer=answer)


class TestNoFit:
    async def test_cards_ranked_below_no_fit_are_not_taken(self):
        pool = [card(n) for n in ("Bolt", "Candy Trail", "Blade")]
        jev = ranks_with_no_fit("Bolt", df.NO_FIT, "Candy Trail", "Blade")
        picks = await df.fill_slot(jev, Slot("card_draw", 0, 3, 8, "Card draw"), pool, {}, "standard",
                                   df.brew_copies)
        assert picks == [("Bolt", 4)]
        assert df.NO_FIT in jev.calls[0][1]["pick"].criteria

    async def test_forced_slot_offers_no_no_fit(self):
        pool = [card(n) for n in ("Bolt", "Candy Trail")]
        jev = ranks_with_no_fit("Bolt", "Candy Trail")
        picks = await df.fill_slot(jev, Slot("any", 0, 99, 8, ""), pool, {}, "standard", df.brew_copies,
                                   allow_none=False)
        assert picks == [("Bolt", 4), ("Candy Trail", 4)]
        assert df.NO_FIT not in jev.calls[0][1]["pick"].criteria

    async def test_rejected_slot_overflows_and_catch_all_is_forced(self, monkeypatch):
        bolts = [card("Bolt A", roles=["burn"]), card("Bolt B", roles=["burn"])]
        trail = [card("Candy Trail", "Artifact", roles=["card_draw"])]

        async def pool(db, slot, colors, format, chosen, archetypes=()):
            return {"card_draw": trail, "burn": bolts, "any": bolts + trail}[slot.role]
        monkeypatch.setattr(df, "slot_pool", pool)

        def answer(state, questions):
            names = list(questions["pick"].criteria)
            order = [n for n in names if n != "Candy Trail"]
            if df.NO_FIT in names:  # slot Choice: Candy Trail ranks below "none of these fit"
                order = [n for n in order if n != df.NO_FIT] + [df.NO_FIT, "Candy Trail"]
            probs = {n: 1.0 - i / 100 for i, n in enumerate(order + ["Candy Trail"])}
            return {"pick": {"choice": order[0], "confidence": 0.9, "probabilities": probs}}
        build = df.Build()
        short = await df.fill_slots(None, FakeJev(answer=answer),
                                    [Slot("card_draw", 0, 3, 4, ""), Slot("burn", 0, 1, 4, "")],
                                    build, ["R"], "standard", dict, df.brew_copies)
        assert "Candy Trail" not in build.copies  # rejected in its slot; the burn slot took the 4
        assert build.copies == {"Bolt A": 4, "Bolt B": 4} and short == 0

    async def test_sideboard_never_offers_no_fit(self, monkeypatch):
        monkeypatch.setattr(df, "sideboard_pool", fake_side_pool({None: [side_card(f"Side {i}", 4) for i in range(5)]}))
        jev = FakeJev()
        await df.fill_sideboard(None, jev, df.Build(), df.Build(), ["R"], "standard", dict, None)
        assert all(df.NO_FIT not in q["pick"].criteria for _, q, _ in jev.calls)


class TestBrewNonbasicTarget:
    async def test_lower_quartile_of_recent_lists_with_the_same_color_count(self):
        db = MagicMock()
        counts = [(c,) for c in (5, 6, 3, 12, 4, 11, 12, 10)]  # sorted: 3 4 5 6 10 11 12 12
        db.execute = AsyncMock(return_value=MagicMock(all=lambda: counts))
        assert await df.brew_nonbasic_target(db, ["R"], "standard") == 5  # median would be 10
        stmt, params = db.execute.call_args[0]
        assert params == {"format": "standard", "n_colors": 1}

    async def test_falls_back_to_the_fixed_rule_without_data(self):
        db = MagicMock()
        db.execute = AsyncMock(return_value=MagicMock(all=lambda: []))
        assert await df.brew_nonbasic_target(db, ["R", "W"], "standard") == df.brew_nonbasics(["R", "W"])

    async def test_fill_lands_uses_it_for_brews(self, monkeypatch):
        lands = [card("Soulstone Sanctuary", "Land", 0, "", identity="", mana_cost=None),
                 card("Fabled Passage", "Land", 0, "", identity="", mana_cost=None)]
        monkeypatch.setattr(df, "land_pool", AsyncMock(return_value=lands))
        monkeypatch.setattr(df, "brew_nonbasic_target", AsyncMock(return_value=6))
        build = df.Build()
        build.add("Shock", 4, card("Shock"))
        useful = FakeJev(answer=lambda state, q: {"utility": 0.9} if "utility" in q else {})
        await df.fill_lands(None, useful, build, plan(lands=22), ["R"], "standard", dict, df.brew_copies)
        assert build.copies["Soulstone Sanctuary"] + build.copies.get("Fabled Passage", 0) == 6
        assert build.copies["Mountain"] == 16


# Real Standard oracle texts (2026-10-06)
LAND_TEXTS = {
    "Multiversal Passage": "As this land enters, choose a basic land type. Then you may pay 2 life. If you don't, it enters tapped.\nThis land is the chosen type.",
    "Starting Town": "This land enters tapped unless it's your first, second, or third turn of the game.\n{T}: Add {C}.\n{T}, Pay 1 life: Add one mana of any color.",
    "Fabled Passage": "{T}, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle. Then if you control four or more lands, untap that land.",
    "Secluded Courtyard": "As this land enters, choose a creature type.\n{T}: Add {C}.\n{T}: Add one mana of any color. Spend this mana only to cast a creature spell of the chosen type or activate an ability of a creature source of the chosen type.",
    "Soulstone Sanctuary": "{T}: Add {C}.\n{4}: This land becomes a 3/3 creature with vigilance and all creature types. It's still a land.",
    "Cavern of Souls": "As this land enters, choose a creature type.\n{T}: Add {C}.\n{T}: Add one mana of any color. Spend this mana only to cast a creature spell of the chosen type, and that spell can't be countered.",
    "Cori Mountain Monastery": "This land enters tapped unless you control a Plains or an Island.\n{T}: Add {R}.\n{3}{R}, {T}: Exile the top card of your library. Until the end of your next turn, you may play that card.",
    "Escape Tunnel": "{T}, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle.\n{T}, Sacrifice this land: Target creature with power 2 or less can't be blocked this turn.",
    "Maelstrom of the Spirit Dragon": "{T}: Add {C}.\n{T}: Add one mana of any color. Spend this mana only to cast a Dragon spell or an Omen spell.\n{4}, {T}, Sacrifice this land: Search your library for a Dragon card, reveal it, put it into your hand, then shuffle.",
}


class TestUtilityLands:
    async def test_keeps_lands_jev_judges_useful_beyond_mana(self):
        pool = [card(n, "Land", 0, "", identity="", mana_cost=None, oracle=t) for n, t in LAND_TEXTS.items()]
        useful = {"Soulstone Sanctuary", "Cavern of Souls", "Cori Mountain Monastery"}
        jev = FakeJev(answer=lambda state, q: {"utility": 0.9 if state["land"]["name"] in useful else 0.1})
        kept = await df.utility_lands(jev, pool)
        assert {r.name for r in kept} == useful
        assert len(jev.calls) == len(pool)
        state, questions, _ = jev.calls[0]
        assert state["land"]["oracle_text"] == pool[0].oracle_text
        assert questions["utility"].instructions == df.UTILITY_QUESTION

    async def test_empty_pool_makes_no_jev_call(self):
        jev = FakeJev()
        assert await df.utility_lands(jev, []) == [] and jev.calls == []

    async def test_one_color_decks_only_get_utility_lands(self, monkeypatch):
        pool = [card(n, "Land", 0, "", identity="", mana_cost=None, oracle=LAND_TEXTS[n])
                for n in ("Multiversal Passage", "Soulstone Sanctuary")]
        monkeypatch.setattr(df, "land_pool", AsyncMock(return_value=pool))
        jev = FakeJev(answer=lambda state, q: (
            {"utility": 0.9 if state["land"]["name"] == "Soulstone Sanctuary" else 0.1} if "utility" in q else {}))
        build = df.Build()
        build.add("Shock", 4, card("Shock"))
        await df.fill_lands(None, jev, build, plan(lands=22, nonbasic=4), ["R"], "standard", dict,
                            df.brew_copies)
        assert "Multiversal Passage" not in build.copies and build.copies["Soulstone Sanctuary"] == 4

    async def test_multi_color_decks_skip_the_check(self, monkeypatch):
        pool = [card("Multiversal Passage", "Land", 0, "", identity="", mana_cost=None,
                     oracle=LAND_TEXTS["Multiversal Passage"])]
        monkeypatch.setattr(df, "land_pool", AsyncMock(return_value=pool))
        jev = FakeJev()
        build = df.Build()
        build.add("Shock", 4, card("Shock"))
        await df.fill_lands(None, jev, build, plan(lands=22, nonbasic=4), ["R", "W"], "standard", dict,
                            df.brew_copies)
        assert build.copies["Multiversal Passage"] == 4
        assert all("utility" not in q for _, q, _ in jev.calls)
