"""
Deck analysis service for the guided builder.

Provides real-time analysis of an in-progress deck: mana curve,
color distribution, role coverage gaps, and card suggestions.
"""

from typing import Optional, List, Dict, Any
from collections import defaultdict
import logging
import math

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_

from app.models.meta import Decklist, Event
from app.services.card_service import CardService, get_format_legality_condition

logger = logging.getLogger(__name__)

# Role targets by archetype - rough guide for what a deck "wants"
ARCHETYPE_ROLE_TARGETS = {
    "aggro": {
        "creatures": (20, 28),
        "removal": (4, 8),
        "card_advantage": (0, 4),
        "lands": (20, 24),
    },
    "midrange": {
        "creatures": (14, 22),
        "removal": (6, 10),
        "card_advantage": (4, 8),
        "lands": (24, 26),
    },
    "control": {
        "creatures": (2, 8),
        "removal": (8, 14),
        "card_advantage": (6, 10),
        "counterspells": (4, 8),
        "lands": (25, 27),
    },
    "combo": {
        "combo_pieces": (8, 16),
        "card_advantage": (6, 10),
        "protection": (4, 8),
        "lands": (22, 26),
    },
}


class DeckAnalyzer:
    """Analyzes an in-progress deck and surfaces insights."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.card_service = CardService(db)

    async def analyze(
        self,
        main_deck: List[Dict[str, Any]],
        sideboard: List[Dict[str, Any]],
        format: str = "standard",
    ) -> Dict[str, Any]:
        """
        Analyze a deck and return stats + actionable insights.
        """
        # Basic counts
        main_count = sum(e.get("quantity", 0) for e in main_deck)
        sb_count = sum(e.get("quantity", 0) for e in sideboard)

        # Categorize cards
        creatures = []
        noncreature_spells = []
        lands = []
        for entry in main_deck:
            card_data = entry.get("card", {}) or {}
            type_line = (card_data.get("type_line") or "").lower()
            name = entry.get("card_name", "").lower()
            qty = entry.get("quantity", 0)

            if "land" in type_line or name in ["plains", "island", "swamp", "mountain", "forest"]:
                lands.append((entry, qty))
            elif "creature" in type_line:
                creatures.append((entry, qty))
            else:
                noncreature_spells.append((entry, qty))

        creature_count = sum(q for _, q in creatures)
        spell_count = sum(q for _, q in noncreature_spells)
        land_count = sum(q for _, q in lands)

        # Mana curve (nonland cards only)
        curve = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # 6 = 6+
        for entry in main_deck:
            card_data = entry.get("card", {}) or {}
            type_line = (card_data.get("type_line") or "").lower()
            if "land" in type_line:
                continue
            mana_cost = card_data.get("mana_cost", "")
            cmc = self._estimate_cmc(mana_cost)
            bucket = min(cmc, 6)
            curve[bucket] = curve.get(bucket, 0) + entry.get("quantity", 0)

        # Color distribution (from mana costs)
        colors_used: Dict[str, int] = {}
        for entry in main_deck:
            card_data = entry.get("card", {}) or {}
            mana_cost = card_data.get("mana_cost", "") or ""
            qty = entry.get("quantity", 0)
            for color in ["W", "U", "B", "R", "G"]:
                pips = mana_cost.count(f"{{{color}}}")
                if pips > 0:
                    colors_used[color] = colors_used.get(color, 0) + (pips * qty)

        # Identify issues
        issues = []
        suggestions = []

        target_main = 60 if format != "cedh" else 100
        target_sb = 15 if format != "cedh" else 0

        if main_count < target_main:
            issues.append(f"Need {target_main - main_count} more cards in main deck ({main_count}/{target_main})")
        elif main_count > target_main:
            issues.append(f"{main_count - target_main} cards over the {target_main}-card limit")

        if format != "cedh" and sb_count < target_sb and sb_count > 0:
            issues.append(f"Sideboard is {target_sb - sb_count} cards short ({sb_count}/{target_sb})")

        if land_count == 0 and main_count > 0:
            issues.append("No lands in deck")
            suggestions.append("Add lands to your mana base")
        elif main_count >= 30:
            land_ratio = land_count / main_count
            if land_ratio < 0.33:
                suggestions.append("Consider adding more lands - you may struggle to cast spells on curve")
            elif land_ratio > 0.47:
                suggestions.append("Land count is high - you might flood out. Consider cutting a few")

        # Curve analysis
        nonland_total = creature_count + spell_count
        if nonland_total > 20:
            low_end = curve.get(1, 0) + curve.get(2, 0)
            high_end = curve.get(5, 0) + curve.get(6, 0)
            if low_end < nonland_total * 0.3:
                suggestions.append("Your curve is top-heavy. More cheap spells would help you survive early turns")
            if high_end > nonland_total * 0.3:
                suggestions.append("Lots of expensive spells. Make sure you have enough ramp or early interaction to get there")

        return {
            "main_deck_count": main_count,
            "sideboard_count": sb_count,
            "target_main": target_main,
            "target_sideboard": target_sb,
            "creature_count": creature_count,
            "spell_count": spell_count,
            "land_count": land_count,
            "mana_curve": curve,
            "colors": colors_used,
            "issues": issues,
            "suggestions": suggestions,
        }

    async def suggest_cards(
        self,
        main_deck: List[Dict[str, Any]],
        colors: List[str],
        role: str,
        format: str = "standard",
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        """Suggest cards that fill a specific role for the deck."""
        existing_names = {e.get("card_name", "").lower() for e in main_deck}

        query = f"{role} cards for {''.join(colors)} deck"
        try:
            cards = await self.card_service.semantic_search(
                query=query,
                colors=colors if colors else None,
                format=format,
                limit=limit * 3,
            )
        except Exception:
            cards = await self.card_service.search(
                colors=colors if colors else None,
                standard_only=(format == "standard"),
                format=format,
                limit=limit * 3,
            )

        results = []
        for card in cards:
            if card.name.lower() in existing_names:
                continue
            if len(results) >= limit:
                break
            results.append({
                "card_name": card.name,
                "card_id": str(card.id),
                "mana_cost": card.mana_cost,
                "type_line": card.type_line,
                "image_uri": card.image_uri,
            })

        return results

    async def _get_tournament_card_names(self, strategy: str, format: str = "standard") -> set:
        """Get card names that appear in tournament decklists matching the strategy."""
        query = select(Decklist).join(Event).where(Event.format == format)

        # Try to match strategy to archetype
        if strategy:
            search_terms = [strategy]
            # Map common strategy terms to archetype names
            strategy_map = {
                "graveyard": ["reanimator", "graveyard", "dredge"],
                "reanimate": ["reanimator"],
                "sacrifice": ["sacrifice", "aristocrats"],
                "tokens": ["tokens", "go-wide"],
                "ramp": ["ramp", "big mana"],
                "control": ["control"],
                "aggro": ["aggro", "red deck"],
                "burn": ["burn", "red deck", "aggro"],
                "tempo": ["tempo"],
                "midrange": ["midrange"],
                "combo": ["combo"],
            }
            for keyword, terms in strategy_map.items():
                if keyword in strategy.lower():
                    search_terms.extend(terms)

            conditions = [Decklist.archetype.ilike(f"%{term}%") for term in search_terms]
            query = query.where(or_(*conditions))

        query = query.limit(50)
        result = await self.db.execute(query)
        decklists = result.scalars().all()

        # If no matching archetypes, fall back to all tournament cards for format
        if not decklists:
            query = select(Decklist).join(Event).where(Event.format == format).limit(100)
            result = await self.db.execute(query)
            decklists = result.scalars().all()

        tournament_cards = set()
        card_frequency = defaultdict(int)
        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    tournament_cards.add(card_name.lower())
                    card_frequency[card_name.lower()] += 1

        return tournament_cards

    async def suggest_cards_for_strategy(
        self,
        strategy: str,
        colors: List[str],
        roles: List[str],
        existing_cards: List[str],
        format: str = "standard",
        cards_per_role: int = 8,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Suggest cards grouped by role for a given strategy.
        Returns {role: [card_data, ...]} with rich metadata.
        Blends semantic search with tournament data to prioritize competitive cards.
        """
        existing_lower = {n.lower() for n in existing_cards}
        results: Dict[str, List[Dict[str, Any]]] = {}

        # Get tournament-played cards for this strategy
        tournament_cards = await self._get_tournament_card_names(strategy, format)
        logger.info(f"Found {len(tournament_cards)} tournament cards for strategy '{strategy}' in {format}")

        for role in roles:
            query = f"{role} {strategy} cards for {''.join(colors)} deck"
            try:
                cards = await self.card_service.semantic_search(
                    query=query,
                    colors=colors if colors else None,
                    format=format,
                    limit=cards_per_role * 4,
                )
            except Exception:
                cards = await self.card_service.search(
                    colors=colors if colors else None,
                    standard_only=(format == "standard"),
                    format=format,
                    limit=cards_per_role * 4,
                )

            # Split into tournament-played and non-tournament cards
            tournament_matches = []
            semantic_only = []
            for card in cards:
                if card.name.lower() in existing_lower:
                    continue
                card_data = {
                    "card_name": card.name,
                    "card_id": str(card.id),
                    "mana_cost": card.mana_cost,
                    "type_line": card.type_line,
                    "oracle_text": card.oracle_text,
                    "image_uri": card.image_uri,
                    "image_uri_small": card.image_uri_small,
                }
                if card.name.lower() in tournament_cards:
                    tournament_matches.append(card_data)
                else:
                    semantic_only.append(card_data)

            # Prioritize tournament-played cards, fill remainder with semantic results
            role_cards = tournament_matches[:cards_per_role]
            remaining = cards_per_role - len(role_cards)
            if remaining > 0:
                role_cards.extend(semantic_only[:remaining])

            if role_cards:
                results[role] = role_cards

        return results

    async def compute_mana_base(
        self,
        main_deck: List[Dict[str, Any]],
        colors: List[str],
        format: str = "standard",
        target_total: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute a mana base for the current deck.
        Returns a list of land entries with quantities.
        """
        from app.services.card_service import get_format_legality_condition
        from app.models.card import Card

        nonland_count = 0
        color_pips: Dict[str, int] = {}
        for entry in main_deck:
            card_data = entry.get("card", {}) or {}
            type_line = (card_data.get("type_line") or "").lower()
            if "land" in type_line:
                continue
            qty = entry.get("quantity", 0)
            nonland_count += qty
            mana_cost = card_data.get("mana_cost", "") or ""
            for color in ["W", "U", "B", "R", "G"]:
                pips = mana_cost.count(f"{{{color}}}")
                if pips > 0:
                    color_pips[color] = color_pips.get(color, 0) + (pips * qty)

        if format == "cedh":
            target = target_total or max(34, 99 - nonland_count)
        else:
            target = target_total or max(20, 60 - nonland_count)

        total_pips = sum(color_pips.values()) or 1
        lands: List[Dict[str, Any]] = []

        # Allocate dual/fetch lands first, then basics
        # For now, use basics proportional to color pips
        basic_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
        allocated = 0

        # Search for dual lands in the colors
        if len(colors) >= 2:
            dual_query = f"dual land {''.join(colors)}"
            try:
                dual_cards = await self.card_service.semantic_search(
                    query=dual_query,
                    format=format,
                    limit=20,
                )
            except Exception:
                dual_cards = []

            dual_count = min(len(dual_cards), target // 3)
            for card in dual_cards[:dual_count]:
                type_line = (card.type_line or "").lower()
                if "land" not in type_line:
                    continue
                lands.append({
                    "card_name": card.name,
                    "quantity": 1 if format == "cedh" else min(4, 2),
                    "mana_cost": card.mana_cost,
                    "type_line": card.type_line,
                    "image_uri": card.image_uri,
                })
                allocated += lands[-1]["quantity"]
                if allocated >= target // 3:
                    break

        # Fill remaining with basics
        remaining = target - allocated
        if remaining > 0 and colors:
            for color in colors:
                basic_name = basic_map.get(color)
                if not basic_name:
                    continue
                ratio = color_pips.get(color, 1) / total_pips
                qty = max(1, round(remaining * ratio))
                card = await self.card_service.get_by_name(basic_name, format=format)
                if card:
                    lands.append({
                        "card_name": card.name,
                        "quantity": qty,
                        "mana_cost": card.mana_cost,
                        "type_line": card.type_line,
                        "image_uri": card.image_uri,
                    })

        return lands

    def _estimate_cmc(self, mana_cost: str) -> int:
        """Estimate CMC from mana cost string like {2}{U}{U}."""
        if not mana_cost:
            return 0
        cmc = 0
        i = 0
        while i < len(mana_cost):
            if mana_cost[i] == '{':
                end = mana_cost.index('}', i)
                symbol = mana_cost[i + 1:end]
                if symbol.isdigit():
                    cmc += int(symbol)
                elif symbol in ('W', 'U', 'B', 'R', 'G'):
                    cmc += 1
                elif symbol == 'X':
                    pass  # X = 0 for curve purposes
                else:
                    cmc += 1  # hybrid, phyrexian, etc.
                i = end + 1
            else:
                i += 1
        return cmc
