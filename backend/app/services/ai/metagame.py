"""Metagame and tournament data utilities."""

import time
import logging
from typing import List, Dict, Any
from collections import defaultdict, Counter

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, union_all

from app.models.meta import Decklist, Event, CardCooccurrence
from app.services.card_service import get_format_legality_condition

logger = logging.getLogger(__name__)

MAX_DECKLIST_EXAMPLES = 3

# Cache for tournament cards
_tournament_cards_cache: Dict[str, Any] = {
    "data": None,
    "timestamp": 0,
    "ttl": 300,
}

# Dual land color mappings
DUAL_LAND_COLORS = {
    # Shock lands
    "hallowed fountain": ["W", "U"], "watery grave": ["U", "B"],
    "blood crypt": ["B", "R"], "stomping ground": ["R", "G"],
    "temple garden": ["G", "W"], "godless shrine": ["W", "B"],
    "steam vents": ["U", "R"], "overgrown tomb": ["B", "G"],
    "sacred foundry": ["R", "W"], "breeding pool": ["G", "U"],
    # Slow lands
    "deserted beach": ["W", "U"], "shipwreck marsh": ["U", "B"],
    "haunted ridge": ["B", "R"], "rockfall vale": ["R", "G"],
    "overgrown farmland": ["G", "W"], "shattered sanctum": ["W", "B"],
    "stormcarved coast": ["U", "R"], "deathcap glade": ["B", "G"],
    "sundown pass": ["R", "W"], "dreamroot cascade": ["G", "U"],
    # Pain lands
    "adarkar wastes": ["W", "U"], "underground river": ["U", "B"],
    "sulfurous springs": ["B", "R"], "karplusan forest": ["R", "G"],
    "brushland": ["G", "W"], "caves of koilos": ["W", "B"],
    "shivan reef": ["U", "R"], "llanowar wastes": ["B", "G"],
    "battlefield forge": ["R", "W"], "yavimaya coast": ["G", "U"],
    # Surveil lands
    "meticulous archive": ["W", "U"], "undercity sewers": ["U", "B"],
    "raucous theater": ["B", "R"], "commercial district": ["R", "G"],
    "lush portico": ["G", "W"], "shadowy backstreet": ["W", "B"],
    "thundering falls": ["U", "R"], "underground mortuary": ["B", "G"],
    "elegant parlor": ["R", "W"], "hedge maze": ["G", "U"],
    # Fast lands
    "seachrome coast": ["W", "U"], "darkslick shores": ["U", "B"],
    "blackcleave cliffs": ["B", "R"], "copperline gorge": ["R", "G"],
    "razorverge thicket": ["G", "W"], "concealed courtyard": ["W", "B"],
    "spirebluff canal": ["U", "R"], "blooming marsh": ["B", "G"],
    "inspiring vantage": ["R", "W"], "botanical sanctum": ["G", "U"],
    # Pathways
    "hengegate pathway": ["W", "U"], "clearwater pathway": ["U", "B"],
    "blightstep pathway": ["B", "R"], "cragcrown pathway": ["R", "G"],
    "branchloft pathway": ["G", "W"], "brightclimb pathway": ["W", "B"],
    "riverglide pathway": ["U", "R"], "darkbore pathway": ["B", "G"],
    "needleverge pathway": ["R", "W"], "barkchannel pathway": ["G", "U"],
    # Verge lands
    "sunlit verge": ["W", "U"], "gloomlake verge": ["U", "B"],
    "blazemire verge": ["B", "R"], "thornspire verge": ["R", "G"],
    "willowrush verge": ["G", "W"], "shadowed verge": ["W", "B"],
    "stormwild verge": ["U", "R"], "wastewood verge": ["B", "G"],
    "emberfall verge": ["R", "W"], "flooded verge": ["G", "U"],
}


class MetagameMixin:
    """Mixin providing metagame and tournament data methods."""

    db: AsyncSession

    async def _filter_tournament_cards_by_color(
        self,
        tournament_card_names: set,
        colors: List[str],
    ) -> List[str]:
        """Filter tournament cards to only those matching the deck's colors."""
        from app.models.card import Card

        colors_upper = [c.upper() for c in colors]
        is_mono_color = len(colors_upper) == 1
        matching_cards = []

        query = select(Card).where(
            func.lower(Card.name).in_([n.lower() for n in tournament_card_names])
        )
        result = await self.db.execute(query)
        cards = result.scalars().all()

        seen = set()
        for card in cards:
            if card.name in seen:
                continue
            seen.add(card.name)

            card_colors = card.colors or []
            is_land = "land" in (card.type_line or "").lower()
            is_colorless = len(card_colors) == 0
            matches_colors = all(c in colors_upper for c in card_colors)

            if is_land:
                oracle = (card.oracle_text or "").lower()
                name_lower = card.name.lower()

                if is_mono_color:
                    if "search" in oracle and "basic land" in oracle:
                        continue
                    if "any color" in oracle or "choose" in oracle:
                        continue
                    if any(x in name_lower for x in ["passage", "gate", "plaza", "junction"]):
                        continue

                matching_cards.append(card.name)
            elif is_colorless or matches_colors:
                matching_cards.append(card.name)

        return matching_cards

    async def _get_all_tournament_cards(self, format: str = "standard") -> List[str]:
        """Get all unique card names that appear in tournament decklists for the given format. Uses cache."""
        global _tournament_cards_cache

        # Use format-specific cache key
        cache_key = f"{format}_data"
        now = time.time()

        # Check format-specific cache
        if (cache_key in _tournament_cards_cache and
            _tournament_cards_cache[cache_key] is not None and
            now - _tournament_cards_cache.get(f"{format}_timestamp", 0) < _tournament_cards_cache["ttl"]):
            logger.debug(f"Using cached {format} tournament cards")
            return _tournament_cards_cache[cache_key]

        # Query for the specified format
        query = select(Decklist).join(Event).where(Event.format == format).limit(200)
        result = await self.db.execute(query)
        decklists = result.scalars().all()

        all_cards = set()
        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    all_cards.add(card_name)
            for entry in (decklist.sideboard or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    all_cards.add(card_name)

        cards_list = list(all_cards)
        _tournament_cards_cache[cache_key] = cards_list
        _tournament_cards_cache[f"{format}_timestamp"] = now
        logger.info(f"Cached {len(cards_list)} {format} tournament cards")

        return cards_list

    async def _find_decks_with_cards(self, card_names: List[str], format: str = "standard") -> List[Decklist]:
        """Find tournament decklists that contain the specified cards."""
        query = select(Decklist).join(Event).where(Event.format == format)
        result = await self.db.execute(query.limit(100))
        all_decklists = result.scalars().all()

        matching_decklists = []
        card_names_lower = [name.lower() for name in card_names]

        for decklist in all_decklists:
            deck_card_names = [
                entry.get("card_name", "").lower()
                for entry in (decklist.main_deck or [])
            ]
            if any(card in deck_card_names for card in card_names_lower):
                matching_decklists.append(decklist)

        logger.info(f"Found {len(matching_decklists)} decklists containing {card_names}")
        return matching_decklists

    def _extract_colors_from_decks(self, decklists: List[Decklist]) -> List[str]:
        """Extract colors by analyzing the actual land base in decklists."""
        color_counts = Counter()

        for decklist in decklists:
            deck_colors = set()
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "").lower()
                if card_name in DUAL_LAND_COLORS:
                    for color in DUAL_LAND_COLORS[card_name]:
                        deck_colors.add(color)
                elif card_name == "plains":
                    deck_colors.add("W")
                elif card_name == "island":
                    deck_colors.add("U")
                elif card_name == "swamp":
                    deck_colors.add("B")
                elif card_name == "mountain":
                    deck_colors.add("R")
                elif card_name == "forest":
                    deck_colors.add("G")

            if deck_colors:
                color_combos = tuple(sorted(deck_colors))
                color_counts[color_combos] += 1

        if color_counts:
            most_common = color_counts.most_common(1)[0][0]
            logger.debug(f"[AI-SERVICE] Extracted colors {list(most_common)} from {len(decklists)} decklists")
            return list(most_common)

        logger.debug(f"[AI-SERVICE] Could not extract colors from decklists")
        return []

    def _extract_cards_from_decks(self, decklists: List[Decklist]) -> List[Dict[str, Any]]:
        """Extract the most commonly played cards from decklists."""
        card_counts = defaultdict(lambda: {"count": 0, "total_quantity": 0})

        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    quantity = entry.get("quantity", 1)
                    card_counts[card_name]["count"] += 1
                    card_counts[card_name]["total_quantity"] += quantity

        common_cards = []
        for card_name, data in card_counts.items():
            avg_qty = round(data["total_quantity"] / data["count"])
            common_cards.append({
                "name": card_name,
                "recommended_quantity": avg_qty,
                "frequency": data["count"],
            })

        common_cards.sort(key=lambda x: x["frequency"], reverse=True)
        return common_cards

    def _infer_colors_from_archetype(self, archetype: str) -> set:
        """Infer likely colors based on archetype name."""
        archetype_lower = archetype.lower()
        colors = set()

        color_keywords = {
            "W": ["white", "azorius", "orzhov", "boros", "selesnya", "esper", "jeskai", "mardu", "abzan", "naya"],
            "U": ["blue", "azorius", "dimir", "izzet", "simic", "esper", "jeskai", "grixis", "sultai", "temur"],
            "B": ["black", "dimir", "rakdos", "orzhov", "golgari", "esper", "grixis", "mardu", "sultai", "abzan"],
            "R": ["red", "boros", "rakdos", "izzet", "gruul", "jeskai", "mardu", "grixis", "temur", "naya"],
            "G": ["green", "selesnya", "golgari", "simic", "gruul", "abzan", "sultai", "temur", "naya", "jund"],
        }

        for color, keywords in color_keywords.items():
            if any(kw in archetype_lower for kw in keywords):
                colors.add(color)

        if "mono" in archetype_lower:
            if "red" in archetype_lower or "rdw" in archetype_lower:
                colors = {"R"}
            elif "white" in archetype_lower:
                colors = {"W"}
            elif "blue" in archetype_lower:
                colors = {"U"}
            elif "black" in archetype_lower:
                colors = {"B"}
            elif "green" in archetype_lower:
                colors = {"G"}

        return colors

    def _format_decklists_as_examples(
        self,
        decklists: List[Decklist],
        max_examples: int = MAX_DECKLIST_EXAMPLES
    ) -> str:
        """Format decklists as example text for the AI prompt."""
        examples = []

        for i, decklist in enumerate(decklists[:max_examples]):
            deck_text = f"\n--- Example Deck {i+1}: {decklist.archetype or 'Unknown'} ---\n"

            main_deck = decklist.main_deck or []
            sideboard = decklist.sideboard or []

            deck_text += "Main Deck:\n"
            for entry in main_deck:
                card_name = entry.get("card_name", "Unknown")
                quantity = entry.get("quantity", 1)
                deck_text += f"  {quantity} {card_name}\n"

            if sideboard:
                deck_text += "\nSideboard:\n"
                for entry in sideboard:
                    card_name = entry.get("card_name", "Unknown")
                    quantity = entry.get("quantity", 1)
                    deck_text += f"  {quantity} {card_name}\n"

            examples.append(deck_text)

        return "\n".join(examples)

    async def _get_mana_base_from_meta(self, archetype: str, colors: List[str], format: str = "standard") -> Dict[str, Any]:
        """Get recommended mana base by analyzing similar tournament decks."""
        from app.models.card import Card

        colors_upper = [c.upper() for c in colors]
        is_mono_color = len(colors_upper) == 1

        # Find similar decklists from tournaments
        query = select(Decklist).join(Event).where(Event.format == format)

        # Try to match archetype name (fuzzy)
        if archetype:
            query = query.where(Decklist.archetype.ilike(f"%{archetype}%"))

        query = query.limit(50)  # Get more to filter by color
        result = await self.db.execute(query)
        decklists = result.scalars().all()

        if not decklists:
            # Fallback: get any recent decklists for this format
            query = select(Decklist).join(Event).where(Event.format == format).limit(50)
            result = await self.db.execute(query)
            decklists = result.scalars().all()

        # Collect all unique card names from decklists
        all_card_names = set()
        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    all_card_names.add(card_name)

        # Batch query to find lands and their color identity
        land_query = select(Card).where(
            func.lower(Card.name).in_([n.lower() for n in all_card_names]),
            Card.type_line.ilike("%land%")
        )
        land_result = await self.db.execute(land_query)
        all_lands = land_result.scalars().all()

        # Build a map of land name -> color identity, filtering for our colors
        valid_lands = {}
        for land in all_lands:
            land_colors = land.color_identity or []
            land_name_lower = land.name.lower()

            # Skip if we already processed this land
            if land_name_lower in valid_lands:
                continue

            # For mono-color decks, only allow:
            # - Basic lands of the color
            # - Colorless lands (empty color identity)
            # - Lands that only produce our color
            if is_mono_color:
                if len(land_colors) == 0:
                    # Colorless utility lands are fine
                    valid_lands[land_name_lower] = land.name
                elif land_colors == colors_upper:
                    # Exact color match (e.g., a red-only land for mono-red)
                    valid_lands[land_name_lower] = land.name
                # Skip multi-color lands like Restless Spire for mono-color decks
            else:
                # For multi-color decks, allow lands that fit within our colors
                if len(land_colors) == 0 or all(c in colors_upper for c in land_colors):
                    valid_lands[land_name_lower] = land.name

        # Count land occurrences across decklists (only valid lands)
        land_counts = defaultdict(lambda: {"count": 0, "total_quantity": 0})

        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name.lower() in valid_lands:
                    quantity = entry.get("quantity", 1)
                    # Use the canonical name from the database
                    canonical_name = valid_lands[card_name.lower()]
                    land_counts[canonical_name]["count"] += 1
                    land_counts[canonical_name]["total_quantity"] += quantity

        # Calculate average quantity for each land
        recommended_lands = []
        for land_name, data in land_counts.items():
            if data["count"] >= 2:  # Only include lands that appear in multiple decks
                avg_qty = round(data["total_quantity"] / data["count"])
                recommended_lands.append({
                    "name": land_name,
                    "recommended_quantity": avg_qty,
                    "frequency": data["count"],
                })

        # Sort by frequency (most common first)
        recommended_lands.sort(key=lambda x: x["frequency"], reverse=True)

        logger.info(f"Mana base for {colors}: {len(recommended_lands)} valid lands from {len(decklists)} decklists")

        return {
            "sample_size": len(decklists),
            "recommended_lands": recommended_lands[:15],  # Top 15 lands
        }

    async def _get_meta_cards(self, archetype: str, strategy: str = "", format: str = "standard") -> List[Dict[str, Any]]:
        """Get commonly played non-land cards from tournament decks for the given archetype."""
        from app.models.card import Card

        # Map strategy keywords to archetype search terms
        strategy_archetype_map = {
            "graveyard": ["reanimator", "graveyard", "dredge"],
            "reanimate": ["reanimator"],
            "reanimation": ["reanimator"],
            "mill": ["mill", "reanimator"],
            "self-mill": ["mill", "reanimator"],
            "discard": ["reanimator", "madness"],
            "sacrifice": ["sacrifice", "aristocrats"],
            "tokens": ["tokens", "go-wide"],
            "ramp": ["ramp", "big mana"],
            "control": ["control"],
            "aggro": ["aggro", "red deck"],
            "burn": ["burn", "red deck", "aggro"],
        }

        # Build archetype search terms from both archetype and strategy
        search_terms = []
        if archetype:
            search_terms.append(archetype)

        # Add strategy-based search terms
        strategy_lower = strategy.lower()
        for keyword, terms in strategy_archetype_map.items():
            if keyword in strategy_lower:
                search_terms.extend(terms)

        # Find similar decklists from tournaments
        query = select(Decklist).join(Event).where(Event.format == format)

        if search_terms:
            # Search for any matching archetype term
            conditions = [Decklist.archetype.ilike(f"%{term}%") for term in search_terms]
            query = query.where(or_(*conditions))

        query = query.limit(20)
        result = await self.db.execute(query)
        decklists = result.scalars().all()

        if not decklists:
            return []

        # Collect all card names and quantities
        card_counts = defaultdict(lambda: {"count": 0, "total_quantity": 0})
        all_card_names = set()

        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    all_card_names.add(card_name)
                    quantity = entry.get("quantity", 1)
                    card_counts[card_name]["count"] += 1
                    card_counts[card_name]["total_quantity"] += quantity

        # Batch query to find which cards are NOT lands
        land_query = select(Card.name).where(
            func.lower(Card.name).in_([n.lower() for n in all_card_names]),
            Card.type_line.ilike("%land%")
        ).distinct(Card.name)
        land_result = await self.db.execute(land_query)
        land_names = {row[0].lower() for row in land_result.all()}

        # Filter to non-land cards that appear in multiple decks
        meta_cards = []
        for card_name, data in card_counts.items():
            if card_name.lower() not in land_names and data["count"] >= 2:
                avg_qty = round(data["total_quantity"] / data["count"])
                meta_cards.append({
                    "name": card_name,
                    "recommended_quantity": avg_qty,
                    "frequency": data["count"],
                })

        # Sort by frequency (most common first)
        meta_cards.sort(key=lambda x: x["frequency"], reverse=True)

        return meta_cards

    async def _get_archetype_decklists(
        self,
        archetype: str,
        colors: List[str],
        strategy: str = "",
        limit: int = MAX_DECKLIST_EXAMPLES,
        format: str = "standard"
    ) -> List[Decklist]:
        """Get tournament decklists for a given archetype."""
        # Map strategy keywords to archetype search terms
        strategy_archetype_map = {
            "graveyard": ["reanimator", "graveyard", "dredge"],
            "reanimate": ["reanimator"],
            "reanimation": ["reanimator"],
            "mill": ["mill", "reanimator"],
            "self-mill": ["mill", "reanimator"],
            "discard": ["reanimator", "madness"],
            "sacrifice": ["sacrifice", "aristocrats"],
            "tokens": ["tokens", "go-wide"],
            "ramp": ["ramp", "big mana"],
            "control": ["control"],
            "aggro": ["aggro", "red deck"],
            "burn": ["burn", "red deck", "aggro"],
        }

        # Build archetype search terms from both archetype and strategy
        search_terms = []
        if archetype:
            search_terms.append(archetype)

        # Add strategy-based search terms
        strategy_lower = strategy.lower()
        for keyword, terms in strategy_archetype_map.items():
            if keyword in strategy_lower:
                search_terms.extend(terms)

        query = select(Decklist).join(Event).where(
            Event.format == format
        )

        # Match archetype name (fuzzy) with any search term
        if search_terms:
            conditions = [Decklist.archetype.ilike(f"%{term}%") for term in search_terms]
            query = query.where(or_(*conditions))

        # Order by placement (best finishes first)
        query = query.order_by(Decklist.placement.asc().nullslast()).limit(limit * 3)

        result = await self.db.execute(query)
        decklists = result.scalars().all()

        if not decklists:
            # Fallback: get any recent top-placing decklists for this format
            query = select(Decklist).join(Event).where(
                Event.format == format
            ).order_by(Decklist.placement.asc().nullslast()).limit(limit * 3)
            result = await self.db.execute(query)
            decklists = result.scalars().all()

        # Filter to decklists that roughly match the requested colors
        if colors and decklists:
            colors_upper = set(c.upper() for c in colors)
            matching_decklists = []
            for decklist in decklists:
                # Infer deck colors from archetype name
                deck_colors = self._infer_colors_from_archetype(decklist.archetype or "")
                if not deck_colors or deck_colors == colors_upper or deck_colors.issubset(colors_upper):
                    matching_decklists.append(decklist)
                    if len(matching_decklists) >= limit:
                        break
            return matching_decklists

        return decklists[:limit]

    async def _get_cooccurrence_cards(
        self,
        card_names: List[str],
        colors: List[str],
        limit: int = 30,
        format: str = "standard"
    ) -> List[Dict[str, Any]]:
        """Get cards that frequently co-occur with the given cards in tournament decks."""
        from app.models.card import Card

        if not card_names:
            return []

        card_names_lower = [n.lower() for n in card_names]

        # Co-occurrence stores pairs in sorted order, so we need to check both directions
        # Query 1: where card_a matches, return card_b
        query_a = select(
            CardCooccurrence.card_b.label("partner"),
            CardCooccurrence.cooccurrence_count.label("count")
        ).where(
            CardCooccurrence.format == format,
            func.lower(CardCooccurrence.card_a).in_(card_names_lower)
        )

        # Query 2: where card_b matches, return card_a
        query_b = select(
            CardCooccurrence.card_a.label("partner"),
            CardCooccurrence.cooccurrence_count.label("count")
        ).where(
            CardCooccurrence.format == format,
            func.lower(CardCooccurrence.card_b).in_(card_names_lower)
        )

        # Union and aggregate
        combined = union_all(query_a, query_b).subquery()
        final_query = select(
            combined.c.partner,
            func.sum(combined.c.count).label("total_count")
        ).group_by(
            combined.c.partner
        ).order_by(
            func.sum(combined.c.count).desc()
        ).limit(limit * 2)

        result = await self.db.execute(final_query)
        cooccurrence_results = result.all()

        if not cooccurrence_results:
            logger.info(f"No co-occurrence data found for {card_names}")
            return []

        # Get card info to filter by color
        card_names_to_check = [row[0] for row in cooccurrence_results]
        colors_upper = [c.upper() for c in colors]

        card_query = select(Card).where(
            func.lower(Card.name).in_([n.lower() for n in card_names_to_check]),
            get_format_legality_condition(format)
        )
        card_result = await self.db.execute(card_query)
        cards = card_result.scalars().all()

        # Build map of card name to colors
        card_color_map = {}
        for card in cards:
            card_colors = card.colors or []
            is_colorless = len(card_colors) == 0
            is_land = "land" in (card.type_line or "").lower()
            matches_colors = all(c in colors_upper for c in card_colors)
            if is_colorless or is_land or matches_colors:
                card_color_map[card.name.lower()] = card

        # Filter results by color and build output
        synergy_cards = []
        for card_b, count in cooccurrence_results:
            card_b_lower = card_b.lower()
            if card_b_lower in card_color_map and card_b_lower not in [n.lower() for n in card_names]:
                synergy_cards.append({
                    "name": card_color_map[card_b_lower].name,
                    "cooccurrence_count": count,
                    "recommended_quantity": 4,  # Will be refined later
                })
                if len(synergy_cards) >= limit:
                    break

        logger.info(f"Found {len(synergy_cards)} co-occurring cards for {card_names}")
        return synergy_cards

    async def _get_sideboard_patterns(
        self,
        archetype: str,
        colors: List[str],
        format: str = "standard"
    ) -> Dict[str, Any]:
        """Analyze sideboard patterns from tournament decks to identify hate cards."""
        from app.models.card import Card

        colors_upper = [c.upper() for c in colors]

        # Get decklists with sideboards
        query = select(Decklist).join(Event).where(
            Event.format == format
        )
        if archetype:
            query = query.where(Decklist.archetype.ilike(f"%{archetype}%"))
        query = query.limit(100)

        result = await self.db.execute(query)
        decklists = result.scalars().all()

        if not decklists:
            return {"sideboard_staples": [], "matchup_cards": {}}

        # Count sideboard card frequencies
        sideboard_counts = defaultdict(lambda: {"count": 0, "total_quantity": 0})
        all_sideboard_names = set()

        for decklist in decklists:
            for entry in (decklist.sideboard or []):
                card_name = entry.get("card_name", "")
                quantity = entry.get("quantity", 0)
                if card_name:
                    all_sideboard_names.add(card_name)
                    sideboard_counts[card_name]["count"] += 1
                    sideboard_counts[card_name]["total_quantity"] += quantity

        # Filter to cards matching the deck's colors
        valid_sideboard_cards = set()
        if all_sideboard_names:
            card_query = select(Card).where(
                func.lower(Card.name).in_([n.lower() for n in all_sideboard_names]),
                get_format_legality_condition(format)
            )
            card_result = await self.db.execute(card_query)
            cards = card_result.scalars().all()

            for card in cards:
                card_colors = card.colors or []
                is_colorless = len(card_colors) == 0
                matches_colors = all(c in colors_upper for c in card_colors)
                if is_colorless or matches_colors:
                    valid_sideboard_cards.add(card.name.lower())

        # Build sideboard staples list
        sideboard_staples = []
        for card_name, data in sideboard_counts.items():
            if card_name.lower() in valid_sideboard_cards and data["count"] >= 3:
                avg_qty = round(data["total_quantity"] / data["count"])
                sideboard_staples.append({
                    "name": card_name,
                    "frequency": data["count"],
                    "avg_quantity": avg_qty,
                    "total_decks": len(decklists),
                })

        # Sort by frequency
        sideboard_staples.sort(key=lambda x: x["frequency"], reverse=True)

        return {
            "sideboard_staples": sideboard_staples[:20],
            "sample_size": len(decklists),
        }

    async def _get_deck_composition_from_meta(
        self,
        archetype: str,
        colors: List[str],
        format: str = "standard"
    ) -> Dict[str, Any]:
        """Analyze actual tournament decks to derive typical composition ratios."""
        from app.models.card import Card

        query = select(Decklist).join(Event).where(Event.format == format)
        if archetype:
            query = query.where(Decklist.archetype.ilike(f"%{archetype}%"))
        query = query.limit(50)

        result = await self.db.execute(query)
        decklists = result.scalars().all()

        if not decklists:
            return {
                "avg_creatures": 20,
                "avg_spells": 16,
                "avg_lands": 24,
                "sample_size": 0
            }

        # Collect all card names to query types
        all_card_names = set()
        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                all_card_names.add(entry.get("card_name", ""))

        # Query card types
        card_query = select(Card.name, Card.type_line).where(
            func.lower(Card.name).in_([n.lower() for n in all_card_names])
        )
        card_result = await self.db.execute(card_query)
        card_types = {row[0].lower(): row[1] for row in card_result.all()}

        # Analyze composition
        compositions = []
        for decklist in decklists:
            creatures = 0
            lands = 0
            other_spells = 0

            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                quantity = entry.get("quantity", 0)
                type_line = card_types.get(card_name.lower(), "").lower()

                if "land" in type_line:
                    lands += quantity
                elif "creature" in type_line:
                    creatures += quantity
                else:
                    other_spells += quantity

            compositions.append({
                "creatures": creatures,
                "lands": lands,
                "spells": other_spells,
            })

        # Calculate averages
        avg_creatures = round(sum(c["creatures"] for c in compositions) / len(compositions))
        avg_lands = round(sum(c["lands"] for c in compositions) / len(compositions))
        avg_spells = round(sum(c["spells"] for c in compositions) / len(compositions))

        logger.info(f"Deck composition for {archetype}: {avg_creatures} creatures, {avg_spells} spells, {avg_lands} lands (from {len(decklists)} decks)")

        return {
            "avg_creatures": avg_creatures,
            "avg_spells": avg_spells,
            "avg_lands": avg_lands,
            "sample_size": len(decklists),
        }
