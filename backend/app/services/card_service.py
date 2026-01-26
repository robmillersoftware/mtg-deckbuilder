from typing import Optional, List, Dict, Any
from uuid import UUID
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text

from app.models.card import Card
from app.schemas.card import CardResponse
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

# Map internal format names to Scryfall legality keys
FORMAT_LEGALITY_MAP = {
    "standard": "standard",
    "historic": "historic",
    "modern": "modern",
    "legacy": "legacy",
    "cedh": "commander",  # cEDH uses Commander legality
}


def get_format_legality_condition(format_name: str):
    """
    Get the SQLAlchemy condition for checking card legality in a format.
    Returns a condition that checks the legalities JSONB field.
    """
    legality_key = FORMAT_LEGALITY_MAP.get(format_name, "standard")
    # Check if legalities->>key = 'legal'
    return Card.legalities[legality_key].astext == "legal"


class CardService:
    """Service for card-related operations including search and semantic lookup."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, card_id: UUID) -> Optional[Card]:
        """Get a card by its ID."""
        result = await self.db.execute(select(Card).where(Card.id == card_id))
        return result.scalar_one_or_none()

    async def get_by_name(
        self,
        name: str,
        standard_only: bool = True,
        format: Optional[str] = None,
    ) -> Optional[Card]:
        """
        Get a card by exact name (case-insensitive). Returns first match if multiple printings exist.
        For double-faced cards (DFCs), matches either face name.
        E.g., searching "Sephiroth, Fabled SOLDIER" will match "Sephiroth, Fabled SOLDIER // Sephiroth, One-Winged Angel"
        """
        # Build base conditions for format/legality
        format_conditions = []
        if format and format in FORMAT_LEGALITY_MAP:
            format_conditions.append(get_format_legality_condition(format))
        elif standard_only:
            format_conditions.append(Card.is_standard_legal == True)

        # First try exact match
        query = select(Card).where(func.lower(Card.name) == name.lower())
        if format_conditions:
            query = query.where(and_(*format_conditions))
        query = query.limit(1)
        result = await self.db.execute(query)
        card = result.scalar_one_or_none()
        if card:
            return card

        # If no exact match and name doesn't contain "//", try matching DFC faces
        if "//" not in name:
            # Match front face: "Name // Back" or back face: "Front // Name"
            dfc_query = select(Card).where(
                or_(
                    func.lower(Card.name).like(f"{name.lower()} // %"),  # Front face
                    func.lower(Card.name).like(f"% // {name.lower()}"),  # Back face
                )
            )
            if format_conditions:
                dfc_query = dfc_query.where(and_(*format_conditions))
            dfc_query = dfc_query.limit(1)
            result = await self.db.execute(dfc_query)
            return result.scalar_one_or_none()

        return None

    async def fuzzy_search_by_name(
        self,
        name: str,
        limit: int = 5,
        standard_only: bool = True,
    ) -> List[Card]:
        """Find cards with names similar to the given name. Returns unique cards by name."""
        from rapidfuzz import fuzz

        def score_card(card: Card, search_name: str) -> int:
            """Score a card against search name, with DFC face matching."""
            search_lower = search_name.lower()
            card_name_lower = card.name.lower()

            # Exact match gets highest score
            if card_name_lower == search_lower:
                return 100

            # For DFCs, check if search matches either face exactly
            if " // " in card.name:
                faces = card.name.split(" // ")
                for face in faces:
                    if face.lower() == search_lower:
                        return 99  # Near-perfect match for exact face
                    face_score = fuzz.ratio(search_lower, face.lower())
                    if face_score > 90:
                        return face_score

            # Standard fuzzy match
            return fuzz.ratio(search_lower, card_name_lower)

        def dedupe_and_score(cards: List[Card], search_name: str, max_results: int) -> List[Card]:
            """Deduplicate by name and sort by fuzzy match score."""
            # Score all cards
            scored = [(card, score_card(card, search_name)) for card in cards]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Deduplicate by name, keeping highest scored version
            seen_names = set()
            unique = []
            for card, _ in scored:
                if card.name not in seen_names:
                    seen_names.add(card.name)
                    unique.append(card)
                    if len(unique) >= max_results:
                        break
            return unique

        # First try exact prefix match
        query = select(Card).where(
            func.lower(Card.name).like(f"{name.lower()}%")
        )
        if standard_only:
            query = query.where(Card.is_standard_legal == True)
        query = query.limit(limit * 10)  # Fetch more to account for duplicates

        result = await self.db.execute(query)
        candidates = list(result.scalars().all())

        if candidates:
            return dedupe_and_score(candidates, name, limit)

        # Fallback to contains match
        query = select(Card).where(
            func.lower(Card.name).like(f"%{name.lower()}%")
        )
        if standard_only:
            query = query.where(Card.is_standard_legal == True)
        query = query.limit(limit * 10)  # Fetch more to account for duplicates

        result = await self.db.execute(query)
        candidates = list(result.scalars().all())

        if candidates:
            return dedupe_and_score(candidates, name, limit)

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
        format: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Card]:
        """Search cards with various filters. Returns unique cards by name."""

        conditions = []

        if format and format in FORMAT_LEGALITY_MAP:
            conditions.append(get_format_legality_condition(format))
        elif standard_only:
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
        format: Optional[str] = None,
        colors: Optional[List[str]] = None,
    ) -> List[Card]:
        """
        Search cards using semantic similarity via embeddings.
        Falls back to text search if embeddings are not available.
        """
        embedding_service = get_embedding_service()
        query_embedding = await embedding_service.get_query_embedding(query)

        if query_embedding is None:
            logger.info("No embedding available, falling back to text search")
            return await self.search(q=query, standard_only=standard_only, format=format, limit=limit, colors=colors)

        try:
            # Build the vector search query using cosine distance (<=>)
            # pgvector uses <=> for cosine distance, <-> for L2 distance
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

            conditions = ["embedding IS NOT NULL"]
            if format and format in FORMAT_LEGALITY_MAP:
                legality_key = FORMAT_LEGALITY_MAP[format]
                conditions.append(f"legalities->>'{legality_key}' = 'legal'")
            elif standard_only:
                conditions.append("is_standard_legal = true")
            if colors:
                for color in colors:
                    if color.upper() in ["W", "U", "B", "R", "G"]:
                        conditions.append(f"'{color.upper()}' = ANY(colors)")

            where_clause = " AND ".join(conditions)

            # Use raw SQL for vector similarity search
            sql = text(f"""
                SELECT id, name, mana_cost, cmc, type_line, oracle_text, power, toughness,
                       colors, color_identity, keywords, legalities, set_code, collector_number,
                       rarity, image_uri, image_uri_small, image_uri_art_crop, price_usd,
                       price_usd_foil, scryfall_uri, oracle_id, set_name, scryfall_id,
                       is_standard_legal, created_at, updated_at,
                       embedding <=> :embedding AS distance
                FROM cards
                WHERE {where_clause}
                ORDER BY embedding <=> :embedding
                LIMIT :limit
            """)

            result = await self.db.execute(sql, {"embedding": embedding_str, "limit": limit * 3})
            rows = result.fetchall()

            # Deduplicate by name (keep highest similarity)
            seen_names = set()
            unique_cards = []
            for row in rows:
                if row.name not in seen_names:
                    seen_names.add(row.name)
                    # Fetch the full Card object
                    card = await self.get_by_name(row.name, standard_only=False)
                    if card:
                        unique_cards.append(card)
                    if len(unique_cards) >= limit:
                        break

            return unique_cards

        except Exception as e:
            logger.error(f"Vector search failed: {e}, falling back to text search")
            return await self.search(q=query, standard_only=standard_only, limit=limit, colors=colors)

    async def get_candidates(
        self,
        role: str,
        description: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        exclude_cards: Optional[List[str]] = None,
        min_results: int = 3,
        max_results: int = 10,
        use_semantic: bool = True,
        format: Optional[str] = None,
    ) -> List[Card]:
        """
        Get candidate cards for a specific role based on constraints.
        Uses semantic search when a description is provided and embeddings are available.
        Used by the AI to select cards for deck building.
        """
        constraints = constraints or {}
        format = format or constraints.get("format", "standard")

        # If we have a description, try semantic search first
        if description and use_semantic:
            colors = constraints.get("colors", [])
            semantic_results = await self.semantic_search(
                query=f"{role}: {description}",
                limit=max_results * 2,
                format=format,
                colors=colors if colors else None,
            )

            # Filter by other constraints
            if semantic_results:
                filtered = []
                for card in semantic_results:
                    if exclude_cards and card.name in exclude_cards:
                        continue
                    if "cmc_max" in constraints and card.cmc and card.cmc > constraints["cmc_max"]:
                        continue
                    if "cmc_min" in constraints and card.cmc and card.cmc < constraints["cmc_min"]:
                        continue
                    if "type" in constraints and constraints["type"].lower() not in (card.type_line or "").lower():
                        continue
                    filtered.append(card)

                if len(filtered) >= min_results:
                    return filtered[:max_results]

        # Fall back to database query
        if format and format in FORMAT_LEGALITY_MAP:
            query = select(Card).where(get_format_legality_condition(format))
        else:
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
            if format and format in FORMAT_LEGALITY_MAP:
                fallback_query = select(Card).where(get_format_legality_condition(format))
            else:
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
