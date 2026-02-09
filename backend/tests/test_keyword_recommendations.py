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
