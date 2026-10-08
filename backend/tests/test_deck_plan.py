"""deck_plan: reference archetype choice and slot plans."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.card import ROLE_DEFINITIONS
from app.services import deck_plan as dp
from app.services.deck_plan import Slot
from tests.jev_fake import FakeJev


def fake_db(rows=()):
    result = MagicMock()
    result.all.return_value = list(rows)
    return MagicMock(execute=AsyncMock(return_value=result))


def sql_of(db):
    stmt, params = db.execute.call_args[0]
    return str(stmt), params


class TestHelpers:
    def test_band_of(self):
        assert [dp.band_of(c) for c in (0, 1, 2, 3, 4, 5, 9)] == [
            (0, 1), (0, 1), (2, 2), (3, 3), (4, 4), (5, 99), (5, 99)]

    def test_describe(self):
        assert dp.describe("burn", 0, 1) == ROLE_DEFINITIONS["burn"] + " (mana value 0-1)"
        assert dp.describe("creature", 5, 99) == "A creature (mana value 5 or more)"
        assert dp.describe("noncreature", 3, 3) == "A noncreature spell (mana value 3)"

    def test_archetype_keys_lower_trim_and_dedupe(self):
        assert dp.archetype_keys(["Boros Aggro", " boros aggro", "4/5C Control", "4/5c Control "]) == [
            "4/5c control", "boros aggro"]
        assert dp.archetype_keys([]) == []
        assert dp.archetype_keys(["", "  ", "Boros Aggro"]) == ["boros aggro"]  # blanks match nothing
        # SQL trim() strips spaces only, so other whitespace must stay part of the key
        assert dp.archetype_keys(["Boros Aggro\t"]) == ["boros aggro\t"]

    def test_largest_remainder_hits_the_total(self):
        assert dp.largest_remainder([1.5, 1.5, 1.0], 4) == [2, 1, 1]
        assert sum(dp.largest_remainder([3.3, 7.1, 2.2, 9.9], 37)) == 37
        assert dp.largest_remainder([10, 0], 5) == [5, 0]


class TestRecentArchetypes:
    async def test_groups_names_case_insensitively_in_the_window(self):
        db = fake_db([("Dimir Aggro", 19), ("Boros Aggro", 2)])
        assert await dp.recent_archetypes(db, "standard") == [("Dimir Aggro", 19), ("Boros Aggro", 2)]
        sql, params = sql_of(db)
        assert params == {"format": "standard"}
        assert "e.date >= CURRENT_DATE - 14" in sql
        assert "GROUP BY lower(trim(d.archetype))" in sql


def picks(choice, confidence=0.9):
    return FakeJev(answer=lambda state, q: {
        "pick": {"choice": choice, "confidence": confidence, "probabilities": {choice: confidence}}})


ARCHETYPES = [("Dimir Aggro", 19), ("Boros Aggro", 2), ("Demone Scarso", 1)]


class TestChooseReference:
    async def test_picked_archetype(self):
        jev = picks("Boros Aggro")
        got = await dp.choose_reference(jev, "Build Boros aggro", ["R", "W"], ARCHETYPES, "standard")
        assert got == "Boros Aggro"
        state, questions, kwargs = jev.calls[0]
        assert state == {"request": "Build Boros aggro", "colors": ["R", "W"]}
        assert list(questions["pick"].criteria) == ["Dimir Aggro", "Boros Aggro", "none"]  # 1-list names out
        assert kwargs == {"timeout": dp.CHOICE_TIMEOUT}

    async def test_none_means_brew(self):
        assert await dp.choose_reference(picks("none"), "mono-red", ["R"], ARCHETYPES, "standard") is None

    async def test_low_confidence_means_brew(self):
        jev = picks("Dimir Aggro", confidence=0.4)
        assert await dp.choose_reference(jev, "blue black", ["U", "B"], ARCHETYPES, "standard") is None

    async def test_no_candidates_skips_jev(self):
        jev = picks("none")
        assert await dp.choose_reference(jev, "x", [], [("Rare Brew", 1)], "standard") is None
        assert jev.calls == []

    async def test_cap_and_reserved_none_name(self):
        many = [("None", 9)] + [(f"Deck {i}", 300 - i) for i in range(300)]
        jev = picks("Deck 0")
        await dp.choose_reference(jev, "x", [], many, "standard")
        criteria = jev.calls[0][1]["pick"].criteria
        assert len(criteria) == dp.MAX_OPTIONS
        assert "None" not in criteria and criteria["none"].startswith("None of these")

    async def test_jev_failure_raises(self):
        with pytest.raises(Exception):  # ask_many raises an ExceptionGroup
            await dp.choose_reference(FakeJev(fail=lambda s: True), "x", [], ARCHETYPES, "standard")


def entry(deck, name, qty, type_line, cmc, colors=(), role=None):
    return SimpleNamespace(deck_id=deck, name=name, qty=qty, type_line=type_line,
                           cmc=cmc, colors=None if colors is None else list(colors), role=role)


MOUNTAIN = "Basic Land — Mountain"
LISTS = [
    entry(1, "Hired Claw", 4, "Creature — Lizard Mercenary", 1, "R", "threat_cheap"),
    entry(1, "Burst Lightning", 4, "Instant", 1, "R", "burn"),
    entry(1, "Mystery Card", 1, None, None),  # decklist name with no card row
    entry(1, "Sacred Foundry", 4, "Land — Mountain Plains", 0),
    entry(1, "Mountain", 10, MOUNTAIN, 0),
    entry(2, "Hired Claw", 3, "Creature — Lizard Mercenary", 1, "R", "threat_cheap"),
    entry(2, "Burst Lightning", 4, "Instant", 1, "R", "burn"),
    entry(2, "Lightning Helix", 2, "Instant", 2, "RW", "removal_targeted"),
    entry(2, "Sacred Foundry", 4, "Land — Mountain Plains", 0),
    entry(2, "Mountain", 9, MOUNTAIN, 0),
]


class TestSlotSizes:
    def test_untagged_cards_fall_back_to_type(self):
        rows = [entry(1, "Bear", 4, "Creature — Bear", 2), entry(1, "Ponder", 2, "Sorcery", 1),
                entry(2, "Bear", 2, "Creature — Bear", 2)]
        assert dict(dp._slot_sizes(rows, 2)) == {("creature", (2, 2)): 3.0, ("noncreature", (0, 1)): 1.0}


class TestMergeSmall:
    def test_same_role_nearest_band_widens_the_range(self):
        got = dp._merge_small({("burn", (0, 1)): 6, ("burn", (3, 3)): 1, ("threat_cheap", (2, 2)): 8})
        assert got == [["burn", 0, 3, 7], ["threat_cheap", 2, 2, 8]]

    def test_same_band_largest_slot(self):
        got = dp._merge_small({("discard", (2, 2)): 1.5, ("threat_cheap", (2, 2)): 8,
                               ("burn", (2, 2)): 3, ("burn", (0, 1)): 4})
        assert ["threat_cheap", 2, 2, 9.5] in got and len(got) == 3

    def test_else_largest_slot(self):
        got = dp._merge_small({("tutor", (5, 99)): 1, ("burn", (0, 1)): 4, ("threat_cheap", (2, 2)): 8})
        assert got == [["burn", 0, 1, 4], ["threat_cheap", 2, 2, 9]]

    def test_a_lone_small_slot_stays(self):
        assert dp._merge_small({("burn", (0, 1)): 1}) == [["burn", 0, 1, 1]]


class TestPlanFromDecklists:
    async def test_slots_lands_and_copies(self):
        db = fake_db(LISTS)
        plan = await dp.plan_from_decklists(db, ["Boros Aggro"], "standard")
        sql, params = sql_of(db)
        assert params == {"format": "standard", "archetypes": ["boros aggro"]}
        assert "lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))" in sql
        assert dp.CARD_COLORS + " AS colors" in sql  # a DFC's colors fall back to its identity
        assert "lower(split_part(c.name, ' // ', 1)) = n.k" in sql  # DFCs listed by front face
        assert "r.role NOT LIKE 'land%'" in sql
        assert "l.sideboard" not in sql  # sideboards are not slot-planned

        assert plan.reference == "Boros Aggro"
        assert plan.lands == 14 and plan.nonbasic_lands == 4  # (14 + 13) / 2 rounds to 14
        # Averages: threat_cheap 0-1 3.5, burn 0-1 4, Helix 1, Mystery 0.5. Mystery joins its
        # band's largest slot (burn), then Helix the largest slot (burn): 3.5 vs 5.5, scaled to 46.
        assert {(s.role, s.cmc_min, s.cmc_max): s.copies for s in plan.slots} == {
            ("threat_cheap", 0, 1): 18, ("burn", 0, 1): 28}
        assert sum(s.copies for s in plan.slots) + plan.lands == dp.MAIN_SIZE
        assert plan.slots[0].description == dp.describe("threat_cheap", 0, 1)
        assert plan.copies["Hired Claw"] == 4 and plan.copies["Lightning Helix"] == 2
        assert plan.copies["Mountain"] == 10

    async def test_colors_played_by_at_least_half_the_lists(self):
        rows = [entry(d, "Shock", 4, "Instant", 1, "R", "burn") for d in (1, 2, 3)]
        rows.append(entry(3, "Get Lost", 1, "Instant", 2, "W", "removal_targeted"))
        plan = await dp.plan_from_decklists(fake_db(rows), ["Mono Red"], "standard")
        assert plan.colors == ["R"]

    async def test_no_lists(self):
        assert await dp.plan_from_decklists(fake_db([]), ["Gone"], "standard") is None


# Two relatives of a mono-red brew: lists 1-2 one archetype, list 3 another.
RELATIVE_LISTS = [
    entry(1, "Hired Claw", 4, "Creature — Lizard Mercenary", 1, "R", "threat_cheap"),
    entry(1, "Burst Lightning", 4, "Instant", 1, "R", "burn"),
    entry(1, "Lightning Helix", 4, "Instant", 2, "RW", "removal_targeted"),  # off-color for R
    entry(1, "Sacred Foundry", 4, "Land — Mountain Plains", 0, "RW"),
    entry(1, "Mountain", 18, MOUNTAIN, 0),
    entry(2, "Hired Claw", 2, "Creature — Lizard Mercenary", 1, "R", "threat_cheap"),
    entry(2, "Burst Lightning", 4, "Instant", 1, "R", "burn"),
    entry(2, "Lightning Helix", 4, "Instant", 2, "RW", "removal_targeted"),
    entry(2, "Mountain", 20, MOUNTAIN, 0),
    entry(3, "Hired Claw", 4, "Creature — Lizard Mercenary", 1, "R", "threat_cheap"),
    entry(3, "Fear of Missing Out", 4, "Enchantment Creature — Nightmare", 2, "BR", "threat_cheap"),
    entry(3, "Patchwork Beastie", 4, "Artifact Creature — Beast", 1, "", "threat_cheap"),  # colorless fits
    entry(3, "Mystery Card", 2, None, None, colors=None),  # no card row (SQL colors NULL): not counted
    entry(3, "Mountain", 24, MOUNTAIN, 0),
]


class TestPlanFromRelatives:
    async def plan(self):
        db = fake_db(RELATIVE_LISTS)
        plan = await dp.plan_from_decklists(db, ["Boros Aggro", "Rakdos Aggro"], "standard", ["R"])
        return plan, sql_of(db)

    async def test_one_query_over_all_the_relatives(self):
        _, (sql, params) = await self.plan()
        assert params == {"format": "standard", "archetypes": ["boros aggro", "rakdos aggro"]}

    async def test_only_on_color_spells_make_slots_totalling_60(self):
        plan, _ = await self.plan()
        # On-color per list: threat_cheap 0-1 Claw (4+2+4)/3 + Beastie 4/3 = 14/3; burn 0-1 8/3.
        # Helix and Fear of Missing Out are off-color; Mystery Card has no row. 38 nonland, 14:8.
        assert {(s.role, s.cmc_min, s.cmc_max): s.copies for s in plan.slots} == {
            ("threat_cheap", 0, 1): 24, ("burn", 0, 1): 14}
        assert sum(s.copies for s in plan.slots) + plan.lands == dp.MAIN_SIZE

    async def test_lands_copies_colors_and_label_from_the_relatives(self):
        plan, _ = await self.plan()
        assert plan.lands == 22  # (22 + 20 + 24) / 3 lands per list, Sacred Foundry included
        assert plan.nonbasic_lands is None  # brew rule (lower quartile by color count) at land time
        assert plan.copies["Hired Claw"] == 3  # (4 + 2 + 4) / 3 across the lists that play it
        assert plan.copies["Patchwork Beastie"] == 4
        assert plan.colors == ["R"]
        assert plan.reference == "Boros Aggro + Rakdos Aggro"  # a log label, never shown to Jev

    async def test_no_colors_keeps_the_reference_rules(self):
        plan = await dp.plan_from_decklists(fake_db(RELATIVE_LISTS), ["Boros Aggro", "Rakdos Aggro"], "standard")
        assert plan.nonbasic_lands == 1  # 4 Foundries over 3 lists
        assert ("removal_targeted", 2, 2) in {(s.role, s.cmc_min, s.cmc_max) for s in plan.slots}


def cand(archetype, lists, name, avg_copies, on_color=True):
    return SimpleNamespace(archetype=archetype, lists=lists, name=name, avg_copies=avg_copies, on_color=on_color)


CANDIDATE_ROWS = [
    # 8 on-color copies per list: exactly MIN_ON_COLOR_SPELLS, so a candidate
    cand("Boros Dragons", 18, "Lightning Helix", 4.0, on_color=False),
    cand("Boros Dragons", 18, "Burst Lightning", 4.0),
    cand("Boros Dragons", 18, "Hired Claw", 3.5),
    cand("Boros Dragons", 18, "Patchwork Beastie", 0.5),  # colorless counts as on-color
    # 7.9 on-color: out, however many lists it has
    cand("Dimir Aggro", 19, "Fear of Missing Out", 7.9),
    cand("Dimir Aggro", 19, "Get Lost", 4.0, on_color=False),
] + [cand("Rakdos Aggro", 4, f"Spell {i}", 1.0) for i in range(30)]  # 30 on-color spells


class TestRelativeCandidates:
    async def test_on_color_share_and_top_cards(self):
        db = fake_db(CANDIDATE_ROWS)
        got = await dp.relative_candidates(db, ["R"], "standard")
        assert [(a, n) for a, n, _ in got] == [("Boros Dragons", 18), ("Rakdos Aggro", 4)]
        # cards are the archetype's most-played spells of any color, so Jev sees off-color ones too
        assert got[0][2] == [("Lightning Helix", 4.0), ("Burst Lightning", 4.0), ("Hired Claw", 3.5),
                             ("Patchwork Beastie", 0.5)]
        assert len(got[1][2]) == dp.TOP_CARDS == 25
        sql, params = sql_of(db)
        assert params == {"format": "standard", "colors": ["R"], "min_lists": 2}
        assert "e.date >= CURRENT_DATE - 14" in sql
        assert "lower(trim(d.archetype)) AS a" in sql  # case and space duplicates are one archetype
        assert "HAVING COUNT(*) >= :min_lists" in sql
        assert "lower(split_part(c.name, ' // ', 1)) = n.k" in sql  # DFCs listed by front face
        assert dp.CARD_COLORS + " AS colors" in sql  # a DFC's colors fall back to its identity
        assert "colors <@ CAST(:colors AS varchar[])" in sql
        assert "NOT LIKE '%Land%'" in sql  # spells only

    async def test_no_colors_no_candidates(self):
        # 1.4 + 2.8 + 3.8 sums to 7.999999999999999 in floats: still exactly 8
        rows = [cand("Mono Red", 3, n, q) for n, q in (("A", 1.4), ("B", 2.8), ("C", 3.8))]
        assert [a for a, _, _ in await dp.relative_candidates(fake_db(rows), ["R"], "standard")] == ["Mono Red"]
        db = fake_db(CANDIDATE_ROWS)
        assert await dp.relative_candidates(db, [], "standard") == []
        db.execute.assert_not_called()


CANDIDATES = [(f"Deck {i}", 10 - i, [(f"Card {i}", 4.0)]) for i in range(7)]


def judges(probabilities):
    return FakeJev(answer=lambda state, q: {"relative": probabilities.get(state["archetype"]["name"], 0.0)})


class TestChooseRelatives:
    async def test_threshold_cap_and_order(self):
        jev = judges({"Deck 0": 0.5, "Deck 1": 0.9, "Deck 2": 0.49, "Deck 3": 0.7, "Deck 4": 0.8,
                      "Deck 5": 0.6, "Deck 6": 0.95})
        got = await dp.choose_relatives(jev, "mono-red aggro", ["R"], CANDIDATES)
        assert got == ["Deck 6", "Deck 1", "Deck 4", "Deck 3", "Deck 5"]  # 0.5 is in but past the cap of 5
        assert len(jev.calls) == len(CANDIDATES)

    async def test_state_holds_the_archetype_cards(self):
        jev = judges({})
        await dp.choose_relatives(jev, "mono-red aggro", ["R"], [("Boros Dragons", 18, [("Burst Lightning", 4.0),
                                                                                         ("Hired Claw", 3.5)])])
        state, questions, kwargs = jev.calls[0]
        assert state == {"request": "mono-red aggro", "colors": ["R"], "archetype": {
            "name": "Boros Dragons", "lists": 18, "cards": {"Burst Lightning": 4.0, "Hired Claw": 3.5}}}
        assert questions["relative"].instructions == dp.RELATIVE_QUESTION
        assert kwargs == {"timeout": dp.CHOICE_TIMEOUT}

    async def test_all_below_the_threshold_is_no_relatives(self):
        assert await dp.choose_relatives(judges({"Deck 0": 0.49}), "x", ["R"], CANDIDATES) == []

    async def test_no_candidates_skips_jev(self):
        jev = judges({})
        assert await dp.choose_relatives(jev, "x", ["R"], []) == [] and jev.calls == []

    async def test_jev_failure_raises(self):
        with pytest.raises(Exception):  # ask_many raises an ExceptionGroup
            await dp.choose_relatives(FakeJev(fail=lambda s: True), "x", ["R"], CANDIDATES)



LLM_PLAN = """Here you go:
[{"role": "threat_cheap", "cmc_min": 1, "cmc_max": 2, "type_contains": "Creature", "copies": 16,
  "description": "Cheap attackers"},
 {"role": "burn", "cmc_min": 1, "cmc_max": 3, "type_contains": null, "copies": 12, "description": "Burn"},
 {"role": "creature", "cmc_min": 3, "cmc_max": 4, "copies": 6, "description": "Top end"}]"""


class TestParseLlmPlan:
    def test_valid_plan_is_scaled_to_the_nonland_count(self):
        slots = dp.parse_llm_plan(LLM_PLAN, lands=22)
        assert [(s.role, s.cmc_min, s.cmc_max, s.type_contains) for s in slots] == [
            ("threat_cheap", 1, 2, "Creature"), ("burn", 1, 3, None), ("creature", 3, 4, None)]
        assert sum(s.copies for s in slots) == 38
        assert slots[0].description == "Cheap attackers"

    @pytest.mark.parametrize("bad", [
        "no json here",
        "[]",
        '[{"role": "land_basic", "copies": 4}]',
        '[{"role": "wizardry", "copies": 4}]',
        '[{"role": "burn", "copies": 0}]',
        '[{"role": "burn", "copies": "4"}]',
        '[{"role": "burn", "copies": true}]',
        '[{"role": "burn", "copies": 4, "cmc_min": 3, "cmc_max": 1}]',
        '[{"copies": 4}]',
        '["burn"]',
        '[{"role": "burn", "copies": 4, "cmc_min": 1.7}]',
        '[{"role": "burn", "copies": 4, "cmc_max": 2.5}]',
        '[{"role": "burn", "copies": 4, "cmc_min": -3}]',
        '[{"role": "burn", "copies": 4, "cmc_max": "3"}]',
        '[{"role": "burn", "copies": 4, "type_contains": 5}]',
    ])
    def test_invalid_plans(self, bad):
        assert dp.parse_llm_plan(bad, lands=22) is None


class TestPlanWithLlm:
    def test_land_counts(self):
        assert [dp.lands_for(a) for a in ("control", "Midrange", "aggro", "combo", "")] == [24, 23, 22, 22, 22]

    def test_hint_is_stripped(self):
        assert dp.lands_for("midrange ") == 23
        assert [s.role for s in dp.default_plan(" Control ")] == [r for r, *_ in dp.DEFAULT_PLANS["control"]]

    def test_description_is_capped_and_type_stripped(self):
        slots = dp.parse_llm_plan(
            '[{"role": "burn", "copies": 4, "description": "%s", "type_contains": "  "}]' % ("x" * 500), 22)
        assert len(slots[0].description) == 200 and slots[0].type_contains is None

    @pytest.mark.parametrize("arch,lands", [("aggro", 22), ("midrange", 23), ("control", 24)])
    def test_default_plans_total(self, arch, lands):
        assert sum(s.copies for s in dp.default_plan(arch)) == 60 - lands

    async def test_without_llm_uses_the_archetype_default(self):
        plan = await dp.plan_with_llm("mono-red aggro", ["R"], "aggro")
        assert plan.lands == 22 and plan.reference is None
        assert plan.nonbasic_lands is None  # brew rule applies at land time
        assert [s.role for s in plan.slots] == [r for r, *_ in dp.DEFAULT_PLANS["aggro"]]
        assert sum(s.copies for s in plan.slots) == 38

    async def test_llm_plan(self, monkeypatch):
        seen = {}

        def complete(system, user, max_tokens=4096):
            seen["user"] = user
            return LLM_PLAN

        monkeypatch.setattr(dp.llm, "is_configured", lambda: True)
        monkeypatch.setattr(dp.llm, "complete", complete)
        plan = await dp.plan_with_llm("UR control", ["U", "R"], "control")
        assert plan.lands == 24 and sum(s.copies for s in plan.slots) == 36
        assert plan.slots[0].description == "Cheap attackers"
        assert "UR control" in seen["user"] and "U, R" in seen["user"]

    async def test_bad_json_uses_the_default(self, monkeypatch):
        monkeypatch.setattr(dp.llm, "is_configured", lambda: True)
        monkeypatch.setattr(dp.llm, "complete", lambda *a, **k: '[{"role": "nonsense", "copies": 4}]')
        plan = await dp.plan_with_llm("control", ["U"], "control")
        assert [s.role for s in plan.slots] == [r for r, *_ in dp.DEFAULT_PLANS["control"]]
        assert sum(s.copies for s in plan.slots) == 36

    async def test_llm_error_uses_the_default(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(dp.llm, "is_configured", lambda: True)
        monkeypatch.setattr(dp.llm, "complete", boom)
        plan = await dp.plan_with_llm("tempo", ["U"], "tempo")  # no tempo default: midrange
        assert [s.role for s in plan.slots] == [r for r, *_ in dp.DEFAULT_PLANS["midrange"]]
        assert plan.lands == 22
