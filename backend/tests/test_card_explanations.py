"""Card explanations are grounded in each card's rules text, not just its name."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services import ai_service as ai
from app.services.ai_service import AIService


async def test_explanation_prompt_includes_rules_text(monkeypatch):
    service = AIService.__new__(AIService)
    service.card_service = SimpleNamespace(get_cards_by_names=AsyncMock(return_value={
        "smaug the magnificent": SimpleNamespace(
            name="Smaug the Magnificent", mana_cost="{3}{R}{R}", type_line="Legendary Creature — Dragon",
            oracle_text="Flying\nWhenever Smaug deals combat damage to a player, create Treasure tokens."),
    }))
    seen = {}

    def complete(system, user, max_tokens=4096):
        seen["system"] = system
        return '{"Smaug the Magnificent": "Top-end flier."}'
    monkeypatch.setattr(ai.llm, "is_configured", lambda: True)
    monkeypatch.setattr(ai.llm, "complete", complete)
    out = await service.generate_card_explanations(
        {"name": "Red", "main_deck": [{"card_name": "Smaug the Magnificent", "quantity": 4},
                                      {"card_name": "Mountain", "quantity": 20}], "sideboard": []},
        "aggro", "attack")
    assert out == {"Smaug the Magnificent": "Top-end flier."}
    assert "4x Smaug the Magnificent ({3}{R}{R} Legendary Creature — Dragon): Flying" in seen["system"]
    assert "20x Mountain" in seen["system"]
    assert "only on each card's rules text" in seen["system"]
