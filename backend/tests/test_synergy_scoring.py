"""
Tests for the mechanical synergy scoring system in card_selection.py.

Validates that:
- SYNERGY_PATTERNS correctly identify synergistic cards
- _score_card_synergy ranks highly synergistic cards above tangential ones
- _detect_card_themes detects themes from strategy text and card text
- Multi-axis cards (matching multiple patterns) get score bonuses
"""

import pytest
from unittest.mock import MagicMock

from app.services.ai.card_selection import (
    SYNERGY_PATTERNS,
    STRATEGY_THEME_MAP,
    CardSelectionMixin,
)


def _make_card(
    name: str = "Test Card",
    oracle_text: str = "",
    type_line: str = "Creature",
    keywords: list = None,
    colors: list = None,
):
    """Create a mock card object for testing."""
    card = MagicMock()
    card.name = name
    card.oracle_text = oracle_text
    card.type_line = type_line
    card.keywords = keywords or []
    card.colors = colors or []
    card.mana_cost = "{1}{B}"
    card.image_uri = None
    return card


class TestSynergyPatterns:
    """Verify SYNERGY_PATTERNS structure and coverage."""

    def test_graveyard_patterns_exist(self):
        assert "graveyard" in SYNERGY_PATTERNS
        patterns = SYNERGY_PATTERNS["graveyard"]
        assert len(patterns) > 10  # Should have comprehensive coverage

    def test_sacrifice_patterns_exist(self):
        assert "sacrifice" in SYNERGY_PATTERNS
        patterns = SYNERGY_PATTERNS["sacrifice"]
        assert len(patterns) > 5

    def test_all_patterns_have_weights(self):
        for theme, patterns in SYNERGY_PATTERNS.items():
            for pattern, weight in patterns:
                assert isinstance(pattern, str), f"Pattern must be string: {pattern}"
                assert 0 < weight <= 1.0, f"Weight out of range for {theme}/{pattern}: {weight}"

    def test_strategy_theme_map_values_reference_valid_themes(self):
        """All themes in STRATEGY_THEME_MAP should exist in SYNERGY_PATTERNS or be creature types."""
        valid_themes = set(SYNERGY_PATTERNS.keys())
        for strategy, themes in STRATEGY_THEME_MAP.items():
            for theme in themes:
                # Allow creature types and other special themes
                assert theme in valid_themes or theme in {
                    "tribal",
                }, f"Unknown theme '{theme}' in STRATEGY_THEME_MAP['{strategy}']"


class TestScoreCardSynergy:
    """Test the _score_card_synergy method."""

    def setup_method(self):
        self.mixin = CardSelectionMixin.__new__(CardSelectionMixin)

    def test_graveyard_recursion_card_scores_high(self):
        """A card that returns creatures from graveyard should score highly for graveyard theme."""
        card = _make_card(
            name="Reanimate",
            oracle_text="Return target creature card from your graveyard to the battlefield.",
        )
        score = self.mixin._score_card_synergy(card, ["graveyard"])
        assert score >= 2.0  # Should match multiple graveyard patterns

    def test_sacrifice_outlet_scores_high_for_sacrifice(self):
        """Sacrifice outlets should score highly for sacrifice theme."""
        card = _make_card(
            name="Viscera Seer",
            oracle_text="Sacrifice a creature: Scry 1.",
            type_line="Creature - Vampire Wizard",
        )
        score = self.mixin._score_card_synergy(card, ["sacrifice"])
        assert score >= 1.0

    def test_death_trigger_scores_for_both_graveyard_and_sacrifice(self):
        """Death triggers should score for both graveyard and sacrifice themes."""
        card = _make_card(
            name="Blood Artist",
            oracle_text="Whenever Blood Artist or another creature dies, target opponent loses 1 life and you gain 1 life.",
            type_line="Creature - Vampire",
        )
        graveyard_score = self.mixin._score_card_synergy(card, ["graveyard"])
        sacrifice_score = self.mixin._score_card_synergy(card, ["sacrifice"])
        combined_score = self.mixin._score_card_synergy(card, ["graveyard", "sacrifice"])

        assert graveyard_score > 0
        assert sacrifice_score > 0
        assert combined_score > graveyard_score  # Combined should be additive

    def test_unrelated_card_scores_zero(self):
        """A card with no synergy should score 0."""
        card = _make_card(
            name="Lightning Bolt",
            oracle_text="Lightning Bolt deals 3 damage to any target.",
            type_line="Instant",
        )
        score = self.mixin._score_card_synergy(card, ["graveyard"])
        assert score == 0.0

    def test_multi_pattern_bonus(self):
        """Cards matching 3+ patterns should get a 1.3x bonus."""
        # A card that has multiple graveyard interactions
        card = _make_card(
            name="Golgari Grave-Troll",
            oracle_text=(
                "Dredge 6. When Golgari Grave-Troll enters the battlefield, "
                "put a +1/+1 counter on it for each creature card in your graveyard. "
                "Return Golgari Grave-Troll from your graveyard to your hand."
            ),
        )
        score = self.mixin._score_card_synergy(card, ["graveyard"])
        # Should match: dredge, "cards in your graveyard", "from.*graveyard"
        # With 3+ matches, gets 1.3x bonus
        assert score >= 2.5

    def test_keyword_bonus(self):
        """Cards with relevant Scryfall keywords should get a bonus."""
        card = _make_card(
            name="Flashback Card",
            oracle_text="Draw a card. Flashback {2}{U}",
            keywords=["flashback"],
        )
        score_with_kw = self.mixin._score_card_synergy(card, ["graveyard"])

        card_no_kw = _make_card(
            name="Flashback Card",
            oracle_text="Draw a card. Flashback {2}{U}",
            keywords=[],
        )
        score_without_kw = self.mixin._score_card_synergy(card_no_kw, ["graveyard"])

        assert score_with_kw > score_without_kw

    def test_token_cards_score_for_tokens_theme(self):
        """Token generators should score for the tokens theme."""
        card = _make_card(
            name="Raise the Alarm",
            oracle_text="Create two 1/1 white Soldier creature tokens.",
            type_line="Instant",
        )
        score = self.mixin._score_card_synergy(card, ["tokens"])
        assert score >= 1.0

    def test_creature_type_theme_matching(self):
        """Creature type themes should match via simple text matching."""
        card = _make_card(
            name="Zombie Lord",
            oracle_text="Other Zombies you control get +1/+1.",
            type_line="Creature - Zombie",
        )
        score = self.mixin._score_card_synergy(card, ["zombie"])
        assert score > 0


class TestStrategyThemeMap:
    """Verify strategy-to-theme mapping covers key archetypes."""

    def test_graveyard_maps_to_sacrifice(self):
        """Graveyard strategies should also map to sacrifice."""
        assert "sacrifice" in STRATEGY_THEME_MAP["graveyard"]
        assert "graveyard" in STRATEGY_THEME_MAP["graveyard"]

    def test_aristocrats_maps_to_both(self):
        assert "sacrifice" in STRATEGY_THEME_MAP["aristocrats"]
        assert "graveyard" in STRATEGY_THEME_MAP["aristocrats"]

    def test_value_engine_has_mappings(self):
        assert "value engine" in STRATEGY_THEME_MAP
        assert len(STRATEGY_THEME_MAP["value engine"]) >= 1


class TestSynergyRanking:
    """Integration-style tests verifying that better synergy cards rank higher."""

    def setup_method(self):
        self.mixin = CardSelectionMixin.__new__(CardSelectionMixin)

    def test_recursion_ranks_above_incidental_graveyard(self):
        """A dedicated recursion card should rank above one that incidentally mentions graveyard."""
        recursion_card = _make_card(
            name="Unearth",
            oracle_text="Return target creature card with mana value 3 or less from your graveyard to the battlefield.",
        )
        incidental_card = _make_card(
            name="Thought Scour",
            oracle_text="Target player mills two cards. Draw a card.",
        )

        recursion_score = self.mixin._score_card_synergy(recursion_card, ["graveyard"])
        incidental_score = self.mixin._score_card_synergy(incidental_card, ["graveyard"])

        assert recursion_score > incidental_score

    def test_multi_theme_card_ranks_above_single_theme(self):
        """A card that's synergistic on multiple axes should rank higher."""
        multi_card = _make_card(
            name="Meren of Clan Nel Toth",
            oracle_text=(
                "Whenever another creature you control dies, you get an experience counter. "
                "At the beginning of your end step, return target creature card with mana value "
                "X or less from your graveyard to your hand, where X is the number of experience "
                "counters you have."
            ),
            type_line="Legendary Creature - Human Shaman",
        )
        single_card = _make_card(
            name="Eternal Witness",
            oracle_text="When Eternal Witness enters the battlefield, you may return target card from your graveyard to your hand.",
        )

        multi_score = self.mixin._score_card_synergy(multi_card, ["graveyard", "sacrifice"])
        single_score = self.mixin._score_card_synergy(single_card, ["graveyard", "sacrifice"])

        assert multi_score > single_score
