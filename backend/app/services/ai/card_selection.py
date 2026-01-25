"""Card selection and synergy utilities."""

import logging
from typing import List, Dict, Any
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text

logger = logging.getLogger(__name__)


class CardSelectionMixin:
    """Mixin providing card selection and synergy methods."""

    db: AsyncSession

    async def _semantic_card_search(
        self,
        strategy: str,
        colors: List[str],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for cards that match the strategy semantically."""
        from app.models.card import Card
        from app.services.card_service import get_format_legality_condition

        keywords = strategy.lower().split()
        results = []
        seen = set()

        for keyword in keywords:
            if len(keyword) < 3:
                continue

            query = select(Card).where(
                Card.oracle_text.ilike(f"%{keyword}%") |
                Card.type_line.ilike(f"%{keyword}%") |
                Card.name.ilike(f"%{keyword}%"),
                Card.is_standard_legal == True
            ).limit(10)

            result = await self.db.execute(query)
            cards = result.scalars().all()

            for card in cards:
                if card.name not in seen:
                    seen.add(card.name)
                    results.append({
                        "name": card.name,
                        "mana_cost": card.mana_cost,
                        "type_line": card.type_line,
                        "oracle_text": card.oracle_text,
                    })

            if len(results) >= limit:
                break

        return results[:limit]

    async def _detect_card_themes(self, card_names: List[str]) -> List[str]:
        """Detect themes from a list of card names."""
        from app.models.card import Card

        if not card_names:
            return []

        themes = set()

        query = select(Card).where(
            func.lower(Card.name).in_([n.lower() for n in card_names])
        )
        result = await self.db.execute(query)
        cards = result.scalars().all()

        for card in cards:
            oracle = (card.oracle_text or "").lower()
            type_line = (card.type_line or "").lower()

            # Creature types
            for creature_type in ["angel", "demon", "dragon", "elf", "goblin", "human",
                                 "merfolk", "vampire", "zombie", "warrior", "wizard", "knight"]:
                if creature_type in type_line:
                    themes.add(creature_type)

            # Mechanics
            if "graveyard" in oracle or "mill" in oracle:
                themes.add("graveyard")
            if "token" in oracle:
                themes.add("tokens")
            if "counter" in oracle and "creature" in type_line:
                themes.add("+1/+1 counters")
            if "artifact" in type_line:
                themes.add("artifact")
            if "enchantment" in type_line:
                themes.add("enchantment")
            if "instant" in type_line or "sorcery" in type_line:
                themes.add("spellslinger")

        return list(themes)[:5]

    async def _get_tournament_synergy_cards(
        self,
        themes: List[str],
        limit: int = 40
    ) -> List[Dict[str, Any]]:
        """Get tournament-played cards that match the given themes."""
        from app.models.card import Card
        from app.models.meta import Decklist, Event

        if not themes:
            return []

        # Get recent tournament decklists
        query = select(Decklist).join(Event).where(
            Event.format == "standard"
        ).limit(100)
        result = await self.db.execute(query)
        decklists = result.scalars().all()

        # Collect all cards from decklists
        card_counts = defaultdict(int)
        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    card_counts[card_name] += 1

        # Get card data for the most played cards
        top_cards = sorted(card_counts.items(), key=lambda x: x[1], reverse=True)[:200]
        card_names = [c[0] for c in top_cards]

        if not card_names:
            return []

        query = select(Card).where(
            func.lower(Card.name).in_([n.lower() for n in card_names])
        )
        result = await self.db.execute(query)
        cards = result.scalars().all()

        # Filter to cards matching themes
        matching_cards = []
        for card in cards:
            oracle = (card.oracle_text or "").lower()
            type_line = (card.type_line or "").lower()
            name_lower = card.name.lower()

            for theme in themes:
                theme_lower = theme.lower()
                if (theme_lower in oracle or theme_lower in type_line or
                    theme_lower in name_lower):
                    matching_cards.append({
                        "name": card.name,
                        "mana_cost": card.mana_cost,
                        "type_line": card.type_line,
                        "recommended_quantity": 4,
                        "frequency": card_counts.get(card.name, 1),
                    })
                    break

        matching_cards.sort(key=lambda x: x["frequency"], reverse=True)
        return matching_cards[:limit]

    async def _get_synergy_cards(
        self,
        themes: List[str],
        colors: List[str],
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Get cards that synergize with the given themes."""
        from app.models.card import Card

        if not themes:
            return []

        colors_upper = [c.upper() for c in colors]
        synergy_cards = []
        seen = set()

        for theme in themes:
            query = select(Card).where(
                Card.oracle_text.ilike(f"%{theme}%") |
                Card.type_line.ilike(f"%{theme}%"),
                Card.is_standard_legal == True
            ).limit(20)

            result = await self.db.execute(query)
            cards = result.scalars().all()

            for card in cards:
                if card.name in seen:
                    continue

                card_colors = card.colors or []
                is_colorless = len(card_colors) == 0
                matches_colors = all(c in colors_upper for c in card_colors)

                if is_colorless or matches_colors:
                    seen.add(card.name)
                    synergy_cards.append({
                        "name": card.name,
                        "mana_cost": card.mana_cost,
                        "type_line": card.type_line,
                        "oracle_text": card.oracle_text,
                    })

        logger.info(f"Found {len(synergy_cards)} synergy cards for themes {themes}")
        return synergy_cards[:limit]
