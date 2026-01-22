from typing import Optional, List, Dict, Any
from uuid import UUID
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

from app.models.card import Card
from app.schemas.card import CardResponse

logger = logging.getLogger(__name__)


class CardService:
    """Service for card-related operations including search and semantic lookup."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, card_id: UUID) -> Optional[Card]:
        """Get a card by its ID."""
        result = await self.db.execute(select(Card).where(Card.id == card_id))
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str, standard_only: bool = True) -> Optional[Card]:
        """Get a card by exact name (case-insensitive). Returns first match if multiple printings exist."""
        query = select(Card).where(func.lower(Card.name) == name.lower())
        if standard_only:
            query = query.where(Card.is_standard_legal == True)
        query = query.limit(1)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def fuzzy_search_by_name(
        self,
        name: str,
        limit: int = 5,
        standard_only: bool = True,
    ) -> List[Card]:
        """Find cards with names similar to the given name."""
        from rapidfuzz import fuzz

        # First try exact prefix match
        query = select(Card).where(
            func.lower(Card.name).like(f"{name.lower()}%")
        )
        if standard_only:
            query = query.where(Card.is_standard_legal == True)
        query = query.limit(limit * 2)

        result = await self.db.execute(query)
        candidates = result.scalars().all()

        if candidates:
            # Sort by fuzzy match score
            scored = [(card, fuzz.ratio(name.lower(), card.name.lower())) for card in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [card for card, _ in scored[:limit]]

        # Fallback to contains match
        query = select(Card).where(
            func.lower(Card.name).like(f"%{name.lower()}%")
        )
        if standard_only:
            query = query.where(Card.is_standard_legal == True)
        query = query.limit(limit * 2)

        result = await self.db.execute(query)
        candidates = result.scalars().all()

        if candidates:
            scored = [(card, fuzz.ratio(name.lower(), card.name.lower())) for card in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [card for card, _ in scored[:limit]]

        return []

    async def search(
        self,
        q: Optional[str] = None,
        colors: Optional[List[str]] = None,
        cmc_min: Optional[int] = None,
        cmc_max: Optional[int] = None,
        card_type: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        standard_only: bool = True,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Card]:
        """Search cards with various filters. Returns unique cards by name."""

        conditions = []

        if standard_only:
            conditions.append(Card.is_standard_legal == True)

        if q:
            search_term = f"%{q.lower()}%"
            conditions.append(
                or_(
                    func.lower(Card.name).like(search_term),
                    func.lower(Card.oracle_text).like(search_term),
                )
            )

        if colors:
            for color in colors:
                if color.upper() in ["W", "U", "B", "R", "G"]:
                    conditions.append(Card.colors.contains([color.upper()]))

        if cmc_min is not None:
            conditions.append(Card.cmc >= cmc_min)
        if cmc_max is not None:
            conditions.append(Card.cmc <= cmc_max)

        if card_type:
            conditions.append(func.lower(Card.type_line).like(f"%{card_type.lower()}%"))

        if keywords:
            for kw in keywords:
                conditions.append(Card.keywords.contains([kw]))

        # Build query with conditions
        query = select(Card)
        if conditions:
            query = query.where(and_(*conditions))

        # Fetch a large batch and dedupe in Python
        # We need enough to cover all unique cards across the alphabet
        fetch_limit = max(limit * 50, 5000)  # Fetch up to 5000 cards minimum
        query = query.order_by(Card.name, Card.id).limit(fetch_limit)

        result = await self.db.execute(query)
        all_cards = list(result.scalars().all())

        # Deduplicate by name
        seen_names = set()
        unique_cards = []
        for card in all_cards:
            if card.name not in seen_names:
                seen_names.add(card.name)
                unique_cards.append(card)

        # Apply offset and limit after deduplication
        return unique_cards[offset:offset + limit]

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        standard_only: bool = True,
    ) -> List[Card]:
        """
        Search cards using semantic similarity.
        Falls back to text search if embeddings are not available.
        """
        # For now, fallback to text search
        # TODO: Implement vector search with pgvector when embeddings are populated
        return await self.search(q=query, standard_only=standard_only, limit=limit)

    async def get_candidates(
        self,
        role: str,
        description: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        exclude_cards: Optional[List[str]] = None,
        min_results: int = 3,
        max_results: int = 10,
    ) -> List[Card]:
        """
        Get candidate cards for a specific role based on constraints.
        Used by the AI to select cards for deck building.
        """
        constraints = constraints or {}
        query = select(Card).where(Card.is_standard_legal == True)
        conditions = []

        # Apply color constraints
        if "colors" in constraints:
            colors = constraints["colors"]
            if colors:
                for color in colors:
                    conditions.append(Card.colors.contains([color.upper()]))

        # Apply CMC constraints
        if "cmc_max" in constraints:
            conditions.append(Card.cmc <= constraints["cmc_max"])
        if "cmc_min" in constraints:
            conditions.append(Card.cmc >= constraints["cmc_min"])
        if "cmc" in constraints:
            conditions.append(Card.cmc == constraints["cmc"])

        # Apply type constraints
        if "type" in constraints:
            conditions.append(
                func.lower(Card.type_line).like(f"%{constraints['type'].lower()}%")
            )

        # Apply keyword constraints
        if "keywords" in constraints:
            for kw in constraints["keywords"]:
                conditions.append(Card.keywords.contains([kw]))

        # Exclude specific cards
        if exclude_cards:
            conditions.append(~Card.name.in_(exclude_cards))

        if conditions:
            query = query.where(and_(*conditions))

        # If we have a description, try to match it
        if description:
            search_term = f"%{description.lower()}%"
            query = query.where(
                or_(
                    func.lower(Card.oracle_text).like(search_term),
                    func.lower(Card.type_line).like(search_term),
                )
            )

        query = query.order_by(Card.name).limit(max_results * 2)

        result = await self.db.execute(query)
        candidates = list(result.scalars().all())

        # Ensure we have at least min_results
        if len(candidates) < min_results:
            # Relax constraints and try again
            fallback_query = select(Card).where(Card.is_standard_legal == True)

            if "type" in constraints:
                fallback_query = fallback_query.where(
                    func.lower(Card.type_line).like(f"%{constraints['type'].lower()}%")
                )

            if exclude_cards:
                fallback_query = fallback_query.where(~Card.name.in_(exclude_cards))

            fallback_query = fallback_query.limit(max_results)
            fallback_result = await self.db.execute(fallback_query)
            candidates = list(fallback_result.scalars().all())

        return candidates[:max_results]
