from typing import Optional, List, Dict, Any
from uuid import UUID
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.deck import Deck
from app.services.card_service import CardService

logger = logging.getLogger(__name__)


class DeckService:
    """
    Service for deck-related operations.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.card_service = CardService(db)

    async def get_by_id(
        self,
        deck_id: UUID,
        owner_id: Optional[UUID] = None,
    ) -> Optional[Deck]:
        """Get a deck by ID, optionally filtering by owner."""
        query = select(Deck).where(Deck.id == deck_id)
        if owner_id:
            query = query.where(Deck.owner_id == owner_id)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_share_token(self, share_token: str) -> Optional[Deck]:
        """Get a public deck by share token."""
        result = await self.db.execute(
            select(Deck).where(
                and_(
                    Deck.share_token == share_token,
                    Deck.visibility == "public",
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_user_decks(
        self,
        owner_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Deck]:
        """List all decks for a user."""
        result = await self.db.execute(
            select(Deck)
            .where(Deck.owner_id == owner_id)
            .order_by(Deck.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def enrich_deck_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich deck entries with card data using batch fetching.
        Returns entries with card details populated.
        """
        if not entries:
            return []

        # Collect all unique card names
        card_names = list(set(e.get("card_name", "") for e in entries if e.get("card_name")))

        # Batch fetch all cards
        cards_map = await self.card_service.get_cards_by_names(card_names)

        # Enrich entries
        enriched = []
        for entry in entries:
            card_name = entry.get("card_name", "")
            card = cards_map.get(card_name.lower())
            enriched.append({
                **entry,
                "card": {
                    "id": str(card.id) if card else None,
                    "name": card.name if card else card_name,
                    "mana_cost": card.mana_cost if card else None,
                    "type_line": card.type_line if card else None,
                    "oracle_text": card.oracle_text if card else None,
                    "image_uri": card.image_uri if card else None,
                    "image_uri_small": card.image_uri_small if card else None,
                } if card else None,
            })

        return enriched

    async def enrich_deck(self, deck: Deck) -> Deck:
        """
        Enrich a deck's card entries with full card data.
        Modifies deck.main_deck and deck.sideboard in place.
        """
        deck.main_deck = await self.enrich_deck_entries(deck.main_deck or [])
        deck.sideboard = await self.enrich_deck_entries(deck.sideboard or [])
        return deck

    async def enrich_deck_cards(self, deck: Deck) -> Dict[str, Any]:
        """
        Enrich a deck with full card information.
        Returns deck data with card details populated.
        """
        enriched_main = await self.enrich_deck_entries(deck.main_deck or [])
        enriched_sideboard = await self.enrich_deck_entries(deck.sideboard or [])

        return {
            "id": str(deck.id),
            "name": deck.name,
            "description": deck.description,
            "format": deck.format,
            "archetype": deck.archetype,
            "main_deck": enriched_main,
            "sideboard": enriched_sideboard,
            "strategy_summary": deck.strategy_summary,
            "visibility": deck.visibility,
            "share_token": deck.share_token,
            "is_validated": deck.is_validated,
        }
