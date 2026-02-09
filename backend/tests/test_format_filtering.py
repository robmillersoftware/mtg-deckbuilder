"""
Tests for format-based card filtering to prevent Commander cards
from appearing in Standard (or other 60-card format) suggestions.
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.card_service import (
    get_format_legality_condition,
    get_format_view,
    FORMAT_LEGALITY_MAP,
    FORMAT_VIEW_MAP,
)
from app.services.chat_service import ChatService


# ---------------------------------------------------------------------------
# get_format_legality_condition tests
# ---------------------------------------------------------------------------

class TestFormatLegalityCondition:
    """Tests for the format legality condition helper."""

    def test_standard_maps_to_standard(self):
        assert FORMAT_LEGALITY_MAP["standard"] == "standard"

    def test_cedh_maps_to_commander(self):
        assert FORMAT_LEGALITY_MAP["cedh"] == "commander"

    def test_unknown_format_defaults_to_standard(self):
        """An unknown format string should fall back to 'standard'."""
        cond = get_format_legality_condition("unknown_format")
        # The condition should still be usable (no exception)
        assert cond is not None


# ---------------------------------------------------------------------------
# Materialized view routing tests
# ---------------------------------------------------------------------------

class TestFormatViewRouting:
    """Tests that format names are correctly routed to materialized views."""

    def test_standard_routes_to_cards_standard(self):
        assert get_format_view("standard") == "cards_standard"

    def test_modern_routes_to_cards_modern(self):
        assert get_format_view("modern") == "cards_modern"

    def test_legacy_routes_to_cards_legacy(self):
        assert get_format_view("legacy") == "cards_legacy"

    def test_historic_routes_to_cards_historic(self):
        assert get_format_view("historic") == "cards_historic"

    def test_cedh_routes_to_cards_commander(self):
        assert get_format_view("cedh") == "cards_commander"

    def test_unknown_format_falls_back_to_cards(self):
        assert get_format_view("unknown_format") == "cards"

    def test_all_legality_formats_have_views(self):
        """Every format in FORMAT_LEGALITY_MAP must have a materialized view."""
        for fmt in FORMAT_LEGALITY_MAP:
            view = FORMAT_VIEW_MAP.get(fmt)
            assert view is not None, f"Format '{fmt}' has no materialized view"


# ---------------------------------------------------------------------------
# _validate_suggestions_legality tests
# ---------------------------------------------------------------------------

class TestValidateSuggestionsLegality:
    """Tests for the ChatService._validate_suggestions_legality safety net."""

    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        return db

    @pytest.fixture
    def chat_service(self, mock_db):
        service = ChatService.__new__(ChatService)
        service.db = mock_db
        return service

    @pytest.mark.asyncio
    async def test_empty_suggestions_returns_empty(self, chat_service):
        result = await chat_service._validate_suggestions_legality([], "standard")
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_out_illegal_cards(self, chat_service):
        """Cards not in the legal set should be removed from suggestion groups."""
        suggestions = [
            {
                "group_name": "Threats",
                "role": "threats",
                "is_batch": False,
                "cards": [
                    {"card_name": "Legal Card", "quantity": 4},
                    {"card_name": "Illegal Commander Card", "quantity": 4},
                ],
            }
        ]

        # Mock DB to return only "Legal Card" as legal
        mock_result = MagicMock()
        mock_result.all.return_value = [("Legal Card",)]
        chat_service.db.execute = AsyncMock(return_value=mock_result)

        result = await chat_service._validate_suggestions_legality(suggestions, "standard")

        assert len(result) == 1
        assert len(result[0]["cards"]) == 1
        assert result[0]["cards"][0]["card_name"] == "Legal Card"

    @pytest.mark.asyncio
    async def test_drops_group_when_all_cards_illegal(self, chat_service):
        """An entire group should be dropped if no cards are legal."""
        suggestions = [
            {
                "group_name": "Commanders",
                "role": "commanders",
                "is_batch": False,
                "cards": [
                    {"card_name": "Rise of the Dark Realms", "quantity": 1},
                    {"card_name": "Consuming Aberration", "quantity": 1},
                ],
            }
        ]

        # Mock DB to return no legal cards
        mock_result = MagicMock()
        mock_result.all.return_value = []
        chat_service.db.execute = AsyncMock(return_value=mock_result)

        result = await chat_service._validate_suggestions_legality(suggestions, "standard")

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_preserves_multiple_valid_groups(self, chat_service):
        """Multiple groups with legal cards should all be preserved."""
        suggestions = [
            {
                "group_name": "Threats",
                "role": "threats",
                "is_batch": False,
                "cards": [{"card_name": "Card A", "quantity": 4}],
            },
            {
                "group_name": "Removal",
                "role": "removal",
                "is_batch": False,
                "cards": [{"card_name": "Card B", "quantity": 4}],
            },
        ]

        # Mock DB to return both cards as legal (called twice, once per group)
        mock_result_a = MagicMock()
        mock_result_a.all.return_value = [("Card A",)]
        mock_result_b = MagicMock()
        mock_result_b.all.return_value = [("Card B",)]
        chat_service.db.execute = AsyncMock(side_effect=[mock_result_a, mock_result_b])

        result = await chat_service._validate_suggestions_legality(suggestions, "standard")

        assert len(result) == 2


# ---------------------------------------------------------------------------
# Format guidance prompt tests
# ---------------------------------------------------------------------------

class TestFormatGuidance:
    """Tests that the system prompt contains correct format-specific instructions."""

    @pytest.fixture
    def chat_service(self):
        service = ChatService.__new__(ChatService)
        return service

    def test_standard_guidance_forbids_commander_roles(self, chat_service):
        guidance = chat_service._get_format_guidance("standard", "Standard")
        assert "NEVER use" in guidance
        assert "commanders" in guidance.lower()
        assert "60-card" in guidance.lower() or "60 card" in guidance.lower()

    def test_standard_guidance_requires_format_legal_cards(self, chat_service):
        guidance = chat_service._get_format_guidance("standard", "Standard")
        assert "legal in Standard" in guidance

    def test_cedh_guidance_mentions_singleton(self, chat_service):
        guidance = chat_service._get_format_guidance("cedh", "cEDH")
        assert "singleton" in guidance.lower() or "1 copy" in guidance.lower()

    def test_modern_guidance_forbids_commander_roles(self, chat_service):
        guidance = chat_service._get_format_guidance("modern", "Modern")
        assert "NEVER use" in guidance
        assert "commanders" in guidance.lower()


# ---------------------------------------------------------------------------
# Semantic search table routing tests
# ---------------------------------------------------------------------------

class TestSemanticSearchRouting:
    """Tests that semantic_search uses the correct materialized view."""

    @pytest.fixture
    def card_service(self):
        from app.services.card_service import CardService
        service = CardService.__new__(CardService)
        service.db = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_standard_queries_standard_view(self, card_service):
        """When format='standard', semantic search should query cards_standard."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.get_query_embedding = AsyncMock(return_value=[0.1] * 1536)

        # Make the DB execute return empty results
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        card_service.db.execute = AsyncMock(return_value=mock_result)

        with patch("app.services.card_service.get_embedding_service", return_value=mock_embedding_service):
            await card_service.semantic_search(query="test", format="standard")

        # Inspect the SQL that was passed to db.execute
        call_args = card_service.db.execute.call_args
        sql_text = str(call_args[0][0])
        assert "cards_standard" in sql_text
        assert "FROM cards\n" not in sql_text  # Should NOT query raw cards table

    @pytest.mark.asyncio
    async def test_cedh_queries_commander_view(self, card_service):
        """When format='cedh', semantic search should query cards_commander."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.get_query_embedding = AsyncMock(return_value=[0.1] * 1536)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        card_service.db.execute = AsyncMock(return_value=mock_result)

        with patch("app.services.card_service.get_embedding_service", return_value=mock_embedding_service):
            await card_service.semantic_search(query="test", format="cedh")

        call_args = card_service.db.execute.call_args
        sql_text = str(call_args[0][0])
        assert "cards_commander" in sql_text

    @pytest.mark.asyncio
    async def test_no_format_queries_cards_table_with_legality(self, card_service):
        """Without a format, should query the main cards table with standard_only filter."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.get_query_embedding = AsyncMock(return_value=[0.1] * 1536)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        card_service.db.execute = AsyncMock(return_value=mock_result)

        with patch("app.services.card_service.get_embedding_service", return_value=mock_embedding_service):
            await card_service.semantic_search(query="test", standard_only=True)

        call_args = card_service.db.execute.call_args
        sql_text = str(call_args[0][0])
        # Should fall back to main cards table with is_standard_legal filter
        assert "is_standard_legal" in sql_text
