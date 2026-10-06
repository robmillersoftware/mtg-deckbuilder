"""DeckGenerator.generate builds with Jev assembly and falls back to the LLM path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services import deck_generator as dg
from app.services.deck_generator import DeckGenerator

ASSEMBLED = {
    "name": "Boros Aggro", "strategy_summary": "Built from recent lists.",
    "main_deck": [{"card_name": "Hired Claw", "quantity": 4}],
    "sideboard": [{"card_name": "Rest in Peace", "quantity": 2}],
}


@pytest.fixture
def generator(monkeypatch):
    g = DeckGenerator.__new__(DeckGenerator)
    g.db = MagicMock(commit=AsyncMock(), refresh=AsyncMock())
    g._get_or_create_conversation = AsyncMock(return_value=MagicMock(id="11111111-1111-1111-1111-111111111111"))
    g._get_meta_context = AsyncMock(return_value={})
    g._get_archetype_template = AsyncMock(return_value=None)
    g._validate_and_fix_cards = AsyncMock(side_effect=lambda cards, format="standard": cards)
    g._format_deck_response = MagicMock(return_value="ok")
    g.validator = SimpleNamespace(validate=AsyncMock(return_value=SimpleNamespace(is_valid=True, errors=[])))
    g.ai_service = SimpleNamespace(
        parse_deck_request=AsyncMock(return_value={
            "archetype": "aggro", "colors": ["R", "W"], "colors_specified": True,
            "strategy": "", "specific_cards": ["Hired Claw"]}),
        generate_deck=AsyncMock(return_value={
            "name": "LLM Deck", "strategy_summary": "llm",
            "main_deck": [{"card_name": "Shock", "quantity": 4}], "sideboard": []}),
        generate_card_explanations=AsyncMock(return_value={}),
    )
    monkeypatch.setattr(dg.deck_fit, "review_deck", AsyncMock(return_value=(None, {})))
    return g


async def test_uses_jev_assembly(generator, monkeypatch):
    assemble = AsyncMock(return_value=ASSEMBLED)
    monkeypatch.setattr(dg.deck_fill, "assemble", assemble)
    result = await generator.generate("Build me Boros aggro", include_sideboard=True)
    assemble.assert_awaited_once_with(
        generator.db, "Build me Boros aggro", ["R", "W"], ["Hired Claw"],
        format="standard", include_sideboard=True, archetype="aggro")
    generator.ai_service.generate_deck.assert_not_awaited()
    generator._get_meta_context.assert_not_awaited()
    assert result.deck.name == "Boros Aggro"
    assert result.deck.main_deck == ASSEMBLED["main_deck"] and result.deck.sideboard == ASSEMBLED["sideboard"]
    assert result.strategy_summary == "Built from recent lists."


async def test_falls_back_to_the_llm_path_when_assembly_raises(generator, monkeypatch):
    monkeypatch.setattr(dg.deck_fill, "assemble", AsyncMock(side_effect=RuntimeError("jev down")))
    result = await generator.generate("Build me Boros aggro")
    generator.ai_service.generate_deck.assert_awaited_once()
    assert result.deck.name == "LLM Deck"


async def test_cedh_keeps_the_llm_path(generator):
    # the real assemble rejects cEDH before touching Jev or the database
    generator.ai_service.get_commander_color_identity = AsyncMock(return_value=["B"])
    result = await generator.generate("cEDH Tymna", format="cedh", colors=["W", "B"])
    generator.ai_service.generate_deck.assert_awaited_once()
    assert result.deck.name == "LLM Deck"
