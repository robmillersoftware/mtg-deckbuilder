"""deck_plan: reference archetype choice and slot plans."""

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
