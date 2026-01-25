"""
AI-powered deck building service.

This is the main service class that combines all the deck building capabilities.
It inherits from various mixins to provide a clean, modular structure.
"""

from typing import Optional, List, Dict, Any
from collections import defaultdict
import logging
import json
import time

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.services.card_service import CardService
from app.services.deck_validator import BASIC_LANDS
from app.models.meta import Decklist, Event, CardCooccurrence

from app.services.ai.mana_base import ManaBaseMixin
from app.services.ai.deck_validation import DeckValidationMixin
from app.services.ai.json_helpers import repair_json, extract_deck_from_malformed_json
from app.services.ai.deck_parsing import (
    parse_deck_request,
    fallback_parse,
    extract_card_names_from_prompt,
    get_commander_color_identity,
)

logger = logging.getLogger(__name__)

MAX_DECKLIST_EXAMPLES = 3

# Cache for tournament cards
_tournament_cards_cache: Dict[str, Any] = {
    "data": None,
    "timestamp": 0,
    "ttl": 300,
}


class AIService(ManaBaseMixin, DeckValidationMixin):
    """
    Service for AI-powered deck building using Claude API.
    Implements the constrained card selection system to prevent hallucination.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.card_service = CardService(db)

    # Delegate to module functions
    def _repair_json(self, json_str: str) -> str:
        return repair_json(json_str)

    def _extract_deck_from_malformed_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        return extract_deck_from_malformed_json(json_str)

    async def parse_deck_request(self, prompt: str) -> Dict[str, Any]:
        return await parse_deck_request(prompt, self.db)

    async def _fallback_parse(self, prompt: str) -> Dict[str, Any]:
        return await fallback_parse(prompt, self.db)

    async def _extract_card_names_from_prompt(self, prompt: str, format: str = "standard") -> List[str]:
        return await extract_card_names_from_prompt(prompt, self.db, format)

    async def get_commander_color_identity(self, card_name: str) -> List[str]:
        return await get_commander_color_identity(card_name, self.db)

    # The remaining methods are inherited from mixins or will be migrated gradually
    # For now, the generate_deck and other complex methods remain in the main file
