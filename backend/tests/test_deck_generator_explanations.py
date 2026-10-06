"""DeckGenerator.generate accepts include_explanations (the /decks/generate route passes it)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services import deck_generator as dg
from app.services.deck_generator import DeckGenerator


@pytest.fixture
def generator(monkeypatch):
    g = DeckGenerator.__new__(DeckGenerator)
    g.db = MagicMock(commit=AsyncMock(), refresh=AsyncMock())
    conversation = MagicMock(id="11111111-1111-1111-1111-111111111111")
    g._get_or_create_conversation = AsyncMock(return_value=conversation)
    g._get_meta_context = AsyncMock(return_value={})
    g._get_archetype_template = AsyncMock(return_value=None)
    g._validate_and_fix_cards = AsyncMock(side_effect=lambda cards, format="standard": cards)
    g._format_deck_response = MagicMock(return_value="ok")
    g.validator = SimpleNamespace(validate=AsyncMock(return_value=SimpleNamespace(is_valid=True, errors=[])))
    g.ai_service = SimpleNamespace(
        parse_deck_request=AsyncMock(return_value={"archetype": "aggro", "colors": ["R"], "strategy": "burn"}),
        generate_deck=AsyncMock(return_value={
            "name": "Burn", "strategy_summary": "Go face",
            "main_deck": [{"card_name": "Shock", "quantity": 4}], "sideboard": [],
        }),
        generate_card_explanations=AsyncMock(return_value={"Shock": "Cheap burn."}),
    )
    monkeypatch.setattr(dg.deck_fit, "review_deck", AsyncMock(return_value=(None, {})))
    monkeypatch.setattr(dg.deck_fill, "assemble", AsyncMock(side_effect=RuntimeError("no jev")))
    return g


async def test_explanations_generated_when_requested(generator):
    result = await generator.generate("Build me a mono-red aggro deck", include_explanations=True)
    generator.ai_service.generate_card_explanations.assert_awaited_once()
    assert result.deck.card_explanations == {"Shock": "Cheap burn."}


async def test_explanations_skipped_by_default(generator):
    result = await generator.generate("Build me a mono-red aggro deck")
    generator.ai_service.generate_card_explanations.assert_not_awaited()
    assert result.deck.card_explanations is None


async def test_tournament_synergy_cards_exists_and_handles_no_themes():
    # generate_deck calls this when the request names specific cards; it was deleted in 2bb5a0c
    from app.services.ai_service import AIService
    service = AIService.__new__(AIService)
    assert await service._get_tournament_synergy_cards([], format="standard") == []
