from typing import Optional, List, Dict, Any
from uuid import UUID
import logging
import re

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

# Map internal format names to their materialized view table names.
# These views contain ONLY cards legal in the given format, making
# format enforcement structural rather than filter-based.
FORMAT_VIEW_MAP = {
    "standard": "cards_standard",
    "historic": "cards_historic",
    "modern": "cards_modern",
    "legacy": "cards_legacy",
    "cedh": "cards_commander",
}

# Full-text document for a card. Migration 016 indexes exactly this expression.
CARD_TSVECTOR = (
    "to_tsvector('english', name || ' ' || coalesce(type_line, '') || ' ' || coalesce(oracle_text, ''))"
)
SHORTLIST_SIZE = 60
VALID_COLORS = {"W", "U", "B", "R", "G"}


def get_format_view(format_name: str) -> str:
    """
    Return the materialized view name for a format.

    Falls back to the main ``cards`` table if the format has no dedicated view.
    """
    return FORMAT_VIEW_MAP.get(format_name, "cards")


def get_format_legality_condition(format_name: str):
    """
    Get the SQLAlchemy condition for checking card legality in a format.
    Returns a condition that checks the legalities JSONB field.

    Used as a backup filter for ORM queries that still hit the main cards table.
    The materialized views are the primary enforcement layer.
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

    async def get_cards_by_names(
        self,
        names: List[str],
        standard_only: bool = False,
    ) -> Dict[str, Card]:
        """
        Batch fetch cards by names (case-insensitive).
        Returns a dict mapping lowercase card name to Card object.
        Handles DFC face matching - both faces map to the same card.
        """
        if not names:
            return {}

        # Normalize names to lowercase for lookup
        names_lower = [n.lower() for n in names]

        # First, try exact matches
        result = await self.db.execute(
            select(Card).where(func.lower(Card.name).in_(names_lower))
        )
        exact_matches = result.scalars().all()

        # Build result dict
        cards_map: Dict[str, Card] = {}
        matched_names = set()

        for card in exact_matches:
            card_name_lower = card.name.lower()
            cards_map[card_name_lower] = card
            matched_names.add(card_name_lower)
            # Also map DFC face names to this card
            if " // " in card.name:
                for face in card.name.split(" // "):
                    face_lower = face.lower()
                    if face_lower not in cards_map:
                        cards_map[face_lower] = card

        # Find unmatched names that might be DFC faces
        unmatched = [n for n in names_lower if n not in cards_map]

        if unmatched:
            # Try matching as DFC faces
            dfc_conditions = []
            for name in unmatched:
                dfc_conditions.append(func.lower(Card.name).like(f"{name} // %"))
                dfc_conditions.append(func.lower(Card.name).like(f"% // {name}"))

            if dfc_conditions:
                dfc_result = await self.db.execute(
                    select(Card).where(or_(*dfc_conditions))
                )
                dfc_cards = dfc_result.scalars().all()

                for card in dfc_cards:
                    if " // " in card.name:
                        for face in card.name.split(" // "):
                            face_lower = face.lower()
                            if face_lower in unmatched and face_lower not in cards_map:
                                cards_map[face_lower] = card

        return cards_map

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
            rows = await self._vector_search(
                query_embedding, format=format, standard_only=standard_only,
                colors=colors, limit=limit,
            )

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
            # Rollback the aborted transaction so the text-search fallback can run
            await self.db.rollback()
            return await self.search(q=query, standard_only=standard_only, format=format, limit=limit, colors=colors)

    async def _vector_search(
        self,
        query_embedding: List[float],
        format: Optional[str] = None,
        standard_only: bool = True,
        colors: Optional[List[str]] = None,
        limit: int = 10,
    ):
        """
        Execute the raw vector similarity query.

        Tries the format-specific materialized view first; if it doesn't exist
        (migration hasn't run yet), falls back to the main ``cards`` table with
        a legality WHERE clause.
        """
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Try the format view first, fall back to cards table
        if format and format in FORMAT_VIEW_MAP:
            source_table = get_format_view(format)
        else:
            source_table = "cards"

        try:
            return await self._execute_vector_query(
                embedding_str, source_table, format, standard_only, colors, limit,
            )
        except Exception as e:
            if source_table != "cards":
                # View probably doesn't exist yet — fall back to main table
                logger.warning(
                    f"Format view '{source_table}' query failed ({e}), "
                    "falling back to cards table with legality filter"
                )
                await self.db.rollback()
                return await self._execute_vector_query(
                    embedding_str, "cards", format, standard_only, colors, limit,
                )
            raise

    async def _execute_vector_query(
        self,
        embedding_str: str,
        source_table: str,
        format: Optional[str],
        standard_only: bool,
        colors: Optional[List[str]],
        limit: int,
    ):
        """Build and execute a single vector similarity SQL query."""
        # The format views already hold only legal cards; the main table needs the filter.
        conditions = ["embedding IS NOT NULL", *self._filter_sql(
            format, standard_only, colors, legality=(source_table == "cards"))]
        where_clause = " AND ".join(conditions)

        sql = text(f"""
            SELECT id, name, mana_cost, cmc, type_line, oracle_text, power, toughness,
                   colors, color_identity, keywords, legalities, set_code, collector_number,
                   rarity, image_uri, image_uri_small, image_uri_art_crop, price_usd,
                   price_usd_foil, scryfall_uri, oracle_id, set_name, scryfall_id,
                   is_standard_legal, created_at, updated_at,
                   embedding <=> :embedding AS distance
            FROM {source_table}
            WHERE {where_clause}
            ORDER BY embedding <=> :embedding
            LIMIT :limit
        """)

        result = await self.db.execute(sql, {"embedding": embedding_str, "limit": limit * 3})
        return result.fetchall()

    @staticmethod
    def _filter_sql(
        format: Optional[str],
        standard_only: bool,
        colors: Optional[List[str]],
        legality: bool = True,
        alias: str = "",
    ) -> List[str]:
        """WHERE fragments for format legality and deck colors (a card's colors must be
        a subset of the deck's, or colorless)."""
        p = f"{alias}." if alias else ""
        conditions = []
        if legality:
            if format and format in FORMAT_LEGALITY_MAP:
                conditions.append(f"{p}legalities->>'{FORMAT_LEGALITY_MAP[format]}' = 'legal'")
            elif standard_only:
                conditions.append(f"{p}is_standard_legal = true")
        valid = [c.upper() for c in colors or [] if c.upper() in VALID_COLORS]
        if valid:
            arr = "ARRAY[" + ",".join(f"'{c}'" for c in valid) + "]::varchar[]"
            conditions.append(f"({p}colors <@ {arr} OR {p}colors = '{{}}' OR {p}colors IS NULL)")
        return conditions

    async def text_search_names(
        self,
        query: str,
        format: Optional[str] = None,
        standard_only: bool = True,
        colors: Optional[List[str]] = None,
        limit: int = SHORTLIST_SIZE,
    ) -> List[str]:
        """Card names matching any query word (Postgres full text with english
        stemming and stopwords), best match first. Runs on `cards` so the
        migration-016 index applies."""
        words = re.findall(r"[a-z0-9]+", query.lower())
        if not words:
            return []
        where = " AND ".join([f"{CARD_TSVECTOR} @@ to_tsquery('english', :q)",
                              *self._filter_sql(format, standard_only, colors)])
        sql = text(f"""
            SELECT name
            FROM cards
            WHERE {where}
            GROUP BY name
            ORDER BY MAX(ts_rank({CARD_TSVECTOR}, to_tsquery('english', :q))) DESC, name
            LIMIT :limit
        """)
        result = await self.db.execute(sql, {"q": " | ".join(words), "limit": limit})
        return [row[0] for row in result.all()]

    async def popular_card_names(
        self,
        format: Optional[str] = None,
        standard_only: bool = True,
        colors: Optional[List[str]] = None,
        limit: int = SHORTLIST_SIZE,
    ) -> List[str]:
        """Nonbasic cards legal in the format and colors, most tournament decklists first."""
        # ponytail: exact name join misses DFCs listed by front face only; add a face match if it matters
        where = " AND ".join(["coalesce(c.type_line, '') NOT LIKE 'Basic Land%'",
                              *self._filter_sql(format, standard_only, colors, alias="c")])
        sql = text(f"""
            SELECT c.name, COUNT(DISTINCT d.id) AS freq
            FROM decklists d
            JOIN events e ON d.event_id = e.id
            CROSS JOIN LATERAL jsonb_array_elements(d.main_deck) AS card_entry
            JOIN cards c ON LOWER(c.name) = LOWER(card_entry->>'card_name')
            WHERE e.format = :format AND {where}
            GROUP BY c.name
            ORDER BY freq DESC, c.name
            LIMIT :limit
        """)
        result = await self.db.execute(sql, {"format": format or "standard", "limit": limit})
        return [row[0] for row in result.all()]

    async def tournament_frequency(
        self,
        card_names: List[str],
        format: str = "standard",
    ) -> Dict[str, int]:
        """
        Look up tournament decklist frequency for a list of card names.

        Returns a dict mapping lowercase card name -> frequency count.
        Cards not found in tournament data are absent.
        """
        if not card_names:
            return {}

        name_params: Dict[str, Any] = {}
        name_placeholders = []
        for i, name in enumerate(card_names):
            name_params[f"n_{i}"] = name.lower()
            name_placeholders.append(f":n_{i}")

        freq_sql = f"""
            SELECT
                LOWER(card_entry->>'card_name') as card_name,
                COUNT(DISTINCT d.id) as freq
            FROM decklists d
            JOIN events e ON d.event_id = e.id,
                 jsonb_array_elements(d.main_deck) as card_entry
            WHERE e.format = :format
              AND LOWER(card_entry->>'card_name') IN ({', '.join(name_placeholders)})
            GROUP BY LOWER(card_entry->>'card_name')
        """
        name_params["format"] = format

        result = await self.db.execute(text(freq_sql), name_params)
        return {row[0]: row[1] for row in result.all()}

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
