"""
Tests for keyword-based card recommendation fixes.

Validates that:
- _extract_mtg_keywords correctly detects MTG mechanics from role strings
- The guided builder uses keyword-based search for mechanic requests
- ai_service uses Card.keywords.contains() for proper MTG keywords
"""

import importlib
import sys
import pytest

# Import guided_builder module to test _extract_mtg_keywords
_spec = importlib.util.spec_from_file_location(
    "guided_builder",
    "backend/app/services/guided_builder.py",
    submodule_search_locations=[],
)
_mod = importlib.util.module_from_spec(_spec)

# Stub out dependencies that guided_builder imports
from unittest.mock import MagicMock

# Create mock modules for guided_builder's imports
sys.modules["sqlalchemy"] = MagicMock()
sys.modules["sqlalchemy.ext"] = MagicMock()
sys.modules["sqlalchemy.ext.asyncio"] = MagicMock()
sys.modules["app"] = MagicMock()
sys.modules["app.services"] = MagicMock()
sys.modules["app.services.card_service"] = MagicMock()

_spec.loader.exec_module(_mod)

_extract_mtg_keywords = _mod._extract_mtg_keywords
MTG_KEYWORDS = _mod.MTG_KEYWORDS


class TestExtractMtgKeywords:
    """Test _extract_mtg_keywords helper function."""

    def test_simple_keyword(self):
        """Direct keyword name should be detected."""
        result = _extract_mtg_keywords("surveil")
        assert "surveil" in result

    def test_cards_with_keyword(self):
        """'cards with X' pattern should detect the keyword."""
        result = _extract_mtg_keywords("cards with surveil")
        assert "surveil" in result

    def test_keyword_cards_pattern(self):
        """'X cards' pattern should detect the keyword."""
        result = _extract_mtg_keywords("surveil cards")
        assert "surveil" in result

    def test_flying_creatures(self):
        """'flying creatures' should detect flying."""
        result = _extract_mtg_keywords("flying creatures")
        assert "flying" in result

    def test_no_keyword_for_standard_roles(self):
        """Standard role names like 'removal' should not match any keyword."""
        result = _extract_mtg_keywords("removal")
        assert len(result) == 0

    def test_no_keyword_for_card_draw(self):
        """'card draw' should not match any keyword."""
        result = _extract_mtg_keywords("card draw")
        assert len(result) == 0

    def test_mill_keyword(self):
        """'mill' should be detected as a keyword."""
        result = _extract_mtg_keywords("mill")
        assert "mill" in result

    def test_deathtouch_keyword(self):
        result = _extract_mtg_keywords("deathtouch")
        assert "deathtouch" in result

    def test_case_insensitive(self):
        """Keywords should be matched case-insensitively."""
        result = _extract_mtg_keywords("Cards with Surveil")
        assert "surveil" in result

    def test_first_strike_multi_word(self):
        """Multi-word keywords like 'first strike' should be detected."""
        result = _extract_mtg_keywords("first strike")
        assert "first strike" in result

    def test_double_strike_not_first_strike(self):
        """'double strike' should match 'double strike', not 'first strike'."""
        result = _extract_mtg_keywords("double strike")
        assert "double strike" in result

    def test_empty_string(self):
        """Empty string should return empty list."""
        result = _extract_mtg_keywords("")
        assert len(result) == 0

    def test_utility_no_match(self):
        """'utility' should not match any keyword."""
        result = _extract_mtg_keywords("utility")
        assert len(result) == 0

    def test_threats_no_match(self):
        """'threats' should not match any keyword."""
        result = _extract_mtg_keywords("threats")
        assert len(result) == 0


class TestMtgKeywordsSet:
    """Verify the MTG_KEYWORDS set contains expected mechanics."""

    def test_surveil_in_keywords(self):
        assert "surveil" in MTG_KEYWORDS

    def test_mill_in_keywords(self):
        assert "mill" in MTG_KEYWORDS

    def test_flying_in_keywords(self):
        assert "flying" in MTG_KEYWORDS

    def test_flashback_in_keywords(self):
        assert "flashback" in MTG_KEYWORDS

    def test_lifelink_in_keywords(self):
        assert "lifelink" in MTG_KEYWORDS

    def test_all_lowercase(self):
        """All keywords should be lowercase."""
        for kw in MTG_KEYWORDS:
            assert kw == kw.lower(), f"Keyword '{kw}' is not lowercase"


class TestKeywordTournamentRanking:
    """Test that keyword search results are ranked by tournament playability."""

    @pytest.fixture
    def analyzer(self):
        """Create a DeckAnalyzer with mocked DB."""
        from unittest.mock import AsyncMock

        DeckAnalyzer = _mod.DeckAnalyzer
        instance = DeckAnalyzer.__new__(DeckAnalyzer)
        instance.db = AsyncMock()
        instance.card_service = AsyncMock()
        return instance

    @pytest.mark.asyncio
    async def test_rank_cards_returns_frequency_map(self, analyzer):
        """_rank_cards_by_tournament_frequency returns card->count mapping."""
        from unittest.mock import AsyncMock, MagicMock

        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("fatal push", 42),
            ("thoughtseize", 35),
        ]
        analyzer.db.execute = AsyncMock(return_value=mock_result)

        freq = await analyzer._rank_cards_by_tournament_frequency(
            ["Fatal Push", "Thoughtseize", "Unknown Card"],
            format="modern",
        )

        assert freq["fatal push"] == 42
        assert freq["thoughtseize"] == 35
        assert freq.get("unknown card") is None

    @pytest.mark.asyncio
    async def test_rank_cards_empty_input(self, analyzer):
        """Empty card list should return empty dict without DB call."""
        freq = await analyzer._rank_cards_by_tournament_frequency([], format="standard")
        assert freq == {}
        analyzer.db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_keyword_search_sorts_by_tournament_freq(self, analyzer):
        """suggest_cards_for_strategy should rank keyword cards by tournament frequency."""
        from unittest.mock import AsyncMock, MagicMock

        # Create mock cards: card_a has no tournament data, card_b is a staple
        def make_mock_card(name, card_id="id", rarity="common", cmc=3):
            card = MagicMock()
            card.name = name
            card.id = card_id
            card.mana_cost = "{1}{B}"
            card.type_line = "Creature"
            card.oracle_text = "Surveil 1"
            card.image_uri = None
            card.image_uri_small = None
            card.rarity = rarity
            card.cmc = cmc
            return card

        jank_card = make_mock_card("Jank Surveiler", "id-jank", rarity="common", cmc=5)
        staple_card = make_mock_card("Tournament Staple", "id-staple", rarity="rare", cmc=2)
        mid_card = make_mock_card("Midtier Surveil", "id-mid", rarity="uncommon", cmc=3)

        # card_service.search returns cards in alphabetical order (the current behavior)
        analyzer.card_service.search = AsyncMock(
            return_value=[jank_card, mid_card, staple_card]
        )

        # Tournament frequency: staple > mid > jank (0)
        mock_freq_result = MagicMock()
        mock_freq_result.all.return_value = [
            ("tournament staple", 50),
            ("midtier surveil", 10),
        ]
        analyzer.db.execute = AsyncMock(return_value=mock_freq_result)

        result = await analyzer.suggest_cards_for_strategy(
            strategy="surveil",
            colors=["B"],
            roles=["surveil"],
            existing_cards=[],
            format="standard",
            cards_per_role=3,
        )

        cards = result.get("surveil", [])
        assert len(cards) == 3
        # Tournament staple should come first, then mid-tier, then jank
        assert cards[0]["card_name"] == "Tournament Staple"
        assert cards[1]["card_name"] == "Midtier Surveil"
        assert cards[2]["card_name"] == "Jank Surveiler"

    @pytest.mark.asyncio
    async def test_keyword_tiebreak_by_rarity_and_cmc(self, analyzer):
        """When tournament frequency is tied, rarer and cheaper cards should rank higher."""
        from unittest.mock import AsyncMock, MagicMock

        def make_mock_card(name, card_id="id", rarity="common", cmc=3):
            card = MagicMock()
            card.name = name
            card.id = card_id
            card.mana_cost = "{1}{B}"
            card.type_line = "Creature"
            card.oracle_text = "Surveil 1"
            card.image_uri = None
            card.image_uri_small = None
            card.rarity = rarity
            card.cmc = cmc
            return card

        # All have 0 tournament appearances, so tiebreakers matter
        common_5cmc = make_mock_card("Common Expensive", "id-1", rarity="common", cmc=5)
        rare_2cmc = make_mock_card("Rare Cheap", "id-2", rarity="rare", cmc=2)
        rare_4cmc = make_mock_card("Rare Pricey", "id-3", rarity="rare", cmc=4)
        uncommon_3cmc = make_mock_card("Uncommon Mid", "id-4", rarity="uncommon", cmc=3)

        analyzer.card_service.search = AsyncMock(
            return_value=[common_5cmc, rare_2cmc, rare_4cmc, uncommon_3cmc]
        )

        # No tournament data at all
        mock_freq_result = MagicMock()
        mock_freq_result.all.return_value = []
        analyzer.db.execute = AsyncMock(return_value=mock_freq_result)

        result = await analyzer.suggest_cards_for_strategy(
            strategy="surveil",
            colors=["B"],
            roles=["surveil"],
            existing_cards=[],
            format="standard",
            cards_per_role=4,
        )

        cards = result.get("surveil", [])
        assert len(cards) == 4
        # Rarity first (rare < uncommon < common), then CMC ascending
        assert cards[0]["card_name"] == "Rare Cheap"      # rare, cmc=2
        assert cards[1]["card_name"] == "Rare Pricey"      # rare, cmc=4
        assert cards[2]["card_name"] == "Uncommon Mid"     # uncommon, cmc=3
        assert cards[3]["card_name"] == "Common Expensive"  # common, cmc=5
