"""Tests for deck_fit: identity overrides, ranking, flagging, and Jev calls (stubbed)."""

from types import SimpleNamespace

import pytest

from app.services import deck_fit
from app.services.deck_fit import DeckIdentity, FitScore, IdentityOverrides


def fs(plan, syn=None, anti=0.0):
    return FitScore(plan_fit=plan, synergy=syn, anti_synergy=anti)


class TestCardPayload:
    def test_from_dict_and_object(self):
        d = deck_fit.card_payload({"card_name": "Murder", "mana_cost": "{1}{B}{B}",
                                   "type_line": "Instant", "oracle_text": "Destroy target creature."})
        o = deck_fit.card_payload(SimpleNamespace(name="Murder", mana_cost="{1}{B}{B}",
                                                  type_line="Instant", oracle_text="Destroy target creature."))
        assert d == o == {"name": "Murder", "mana_cost": "{1}{B}{B}",
                          "type_line": "Instant", "oracle_text": "Destroy target creature."}

    def test_truncates_long_oracle_text(self):
        p = deck_fit.card_payload({"card_name": "Wall", "oracle_text": "x" * 5000})
        assert len(p["oracle_text"]) == deck_fit.ORACLE_CHAR_LIMIT

    def test_none_fields_become_empty(self):
        p = deck_fit.card_payload({"card_name": "Blank"})
        assert p["type_line"] == "" and p["oracle_text"] == "" and p["mana_cost"] == ""

    def test_is_land(self):
        assert deck_fit.is_land({"type_line": "Basic Land — Swamp"})
        assert not deck_fit.is_land({"type_line": "Artifact Creature — Golem"})


class TestApplyOverrides:
    def test_tags_and_pins(self):
        ident = DeckIdentity(
            tags=["graveyard", "control"], key_cards=["A", "B"],
            overrides=IdentityOverrides(tags_on=["sacrifice"], tags_off=["control"],
                                        pinned=["C"], unpinned=["A"]),
        )
        out = deck_fit.apply_overrides(ident, ["A", "B", "C"])
        assert out.tags == ["graveyard", "sacrifice"]
        assert out.key_cards == ["B", "C"]

    def test_ignores_unknown_tag_and_pinned_card_not_in_deck(self):
        ident = DeckIdentity(tags=[], key_cards=[],
                             overrides=IdentityOverrides(tags_on=["not_a_tag"], pinned=["Gone"]))
        out = deck_fit.apply_overrides(ident, ["A"])
        assert out.tags == [] and out.key_cards == []

    def test_no_duplicates(self):
        ident = DeckIdentity(tags=["tokens"], key_cards=["A"],
                             overrides=IdentityOverrides(tags_on=["tokens"], pinned=["A"]))
        out = deck_fit.apply_overrides(ident, ["A"])
        assert out.tags == ["tokens"] and out.key_cards == ["A"]


class TestRank:
    def test_drops_anti_synergy_and_orders_by_total(self):
        fit = {"Good": fs(1.0, 1.0), "Meh": fs(0.3, 0.3), "Anti": fs(1.0, 1.0, anti=0.9)}
        assert deck_fit.rank(["Meh", "Anti", "Good"], fit, {}) == ["Good", "Meh"]

    def test_meta_frequency_breaks_fit_ties(self):
        fit = {"A": fs(0.5, 0.5), "B": fs(0.5, 0.5)}
        assert deck_fit.rank(["A", "B"], fit, {"b": 10, "a": 1}) == ["B", "A"]

    def test_missing_synergy_moves_weight_to_plan_fit(self):
        fit = {"A": fs(0.9), "B": fs(0.5)}
        assert deck_fit.rank(["B", "A"], fit, {}) == ["A", "B"]

    def test_ties_keep_input_order(self):
        fit = {"A": fs(0.5, 0.5), "B": fs(0.5, 0.5)}
        assert deck_fit.rank(["B", "A"], fit, {}) == ["B", "A"]


class TestFlagLowFit:
    def test_flags_low_average_and_anti_synergy_worst_first(self):
        fit = {"Fine": fs(0.8, 0.7), "Low": fs(0.1, 0.2), "Anti": fs(0.9, 0.9, anti=0.8),
               "Lowest": fs(0.0, 0.0)}
        assert deck_fit.flag_low_fit(fit) == ["Lowest", "Low", "Anti"]

    def test_limit(self):
        fit = {f"C{i}": fs(0.0, 0.0) for i in range(9)}
        assert len(deck_fit.flag_low_fit(fit)) == 5

    def test_empty(self):
        assert deck_fit.flag_low_fit({}) == []


def test_bucket():
    assert [deck_fit.bucket(n) for n in (0, 5, 6, 14, 15, 29, 30, 60)] == [0, 0, 1, 1, 2, 2, 3, 3]
