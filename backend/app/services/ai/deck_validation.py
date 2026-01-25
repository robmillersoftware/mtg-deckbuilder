"""Deck validation and fixing utilities."""

import logging
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.services.deck_validator import BASIC_LANDS

logger = logging.getLogger(__name__)


class DeckValidationMixin:
    """Mixin providing deck validation and fixing methods."""

    db: AsyncSession

    async def _filter_by_color(
        self,
        deck_data: Dict[str, Any],
        colors: List[str],
    ) -> Dict[str, Any]:
        """Remove cards that don't match the deck's color identity."""
        from app.models.card import Card

        colors_upper = [c.upper() for c in colors]

        async def filter_list(card_list: List[Dict]) -> List[Dict]:
            filtered = []
            for entry in card_list:
                card_name = entry.get("card_name", "")
                query = select(Card).where(func.lower(Card.name) == card_name.lower()).limit(1)
                result = await self.db.execute(query)
                card = result.scalar_one_or_none()

                if card:
                    card_colors = card.colors or []
                    card_color_identity = card.color_identity or []
                    is_land = "land" in (card.type_line or "").lower()
                    is_colorless = len(card_colors) == 0 and len(card_color_identity) == 0

                    if is_land:
                        land_fits = len(card_color_identity) == 0 or all(c in colors_upper for c in card_color_identity)
                        if land_fits:
                            filtered.append(entry)
                        else:
                            logger.debug(f" Removed off-color land: {card_name} (color_identity: {card_color_identity})")
                    elif is_colorless:
                        filtered.append(entry)
                    elif all(c in colors_upper for c in card_color_identity):
                        filtered.append(entry)
                    else:
                        logger.debug(f" Removed off-color card: {card_name} (color_identity: {card_color_identity})")
                else:
                    filtered.append(entry)

            return filtered

        deck_data["main_deck"] = await filter_list(deck_data.get("main_deck", []))
        deck_data["sideboard"] = await filter_list(deck_data.get("sideboard", []))

        return deck_data

    async def _enrich_deck_with_card_data(
        self,
        deck_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enrich deck entries with card data (type_line) for frontend categorization."""
        from app.models.card import Card

        all_card_names = set()
        for entry in deck_data.get("main_deck", []):
            all_card_names.add(entry.get("card_name", ""))
        for entry in deck_data.get("sideboard", []):
            all_card_names.add(entry.get("card_name", ""))

        # Include commander name if it exists (may be a string at this point)
        commander_name = deck_data.get("commander")
        if commander_name and isinstance(commander_name, str):
            all_card_names.add(commander_name)

        if not all_card_names:
            return deck_data

        query = select(Card).where(
            func.lower(Card.name).in_([n.lower() for n in all_card_names])
        )
        result = await self.db.execute(query)
        cards = result.scalars().all()

        card_map = {}
        for card in cards:
            card_map[card.name.lower()] = {
                "type_line": card.type_line,
                "mana_cost": card.mana_cost,
                "oracle_text": card.oracle_text,
                "colors": card.colors,
                "image_uri": card.image_uri,
            }

        for entry in deck_data.get("main_deck", []):
            card_name = entry.get("card_name", "")
            card_data = card_map.get(card_name.lower(), {})
            if card_data:
                entry["card"] = card_data

        for entry in deck_data.get("sideboard", []):
            card_name = entry.get("card_name", "")
            card_data = card_map.get(card_name.lower(), {})
            if card_data:
                entry["card"] = card_data

        # Convert commander from string to DeckEntry format and enrich it
        if commander_name and isinstance(commander_name, str):
            commander_card_data = card_map.get(commander_name.lower(), {})
            deck_data["commander"] = {
                "card_name": commander_name,
                "quantity": 1,
                "card": commander_card_data if commander_card_data else None,
            }

        return deck_data

    async def _generate_fallback_deck(
        self,
        archetype: str,
        colors: List[str],
        strategy: str,
        format: str = "standard",
    ) -> Dict[str, Any]:
        """Generate a basic deck structure as fallback."""
        available = await self.card_service.search(
            colors=colors,
            format=format,
            limit=100,
        )

        main_deck = []
        sideboard = []

        creatures = [c for c in available if "creature" in (c.type_line or "").lower()]
        instants = [c for c in available if "instant" in (c.type_line or "").lower()]
        sorceries = [c for c in available if "sorcery" in (c.type_line or "").lower()]

        for i, card in enumerate(creatures[:6]):
            main_deck.append({"card_name": card.name, "quantity": 4})

        for card in instants[:2]:
            main_deck.append({"card_name": card.name, "quantity": 4})
        for card in sorceries[:1]:
            main_deck.append({"card_name": card.name, "quantity": 4})

        basic_land_map = {"R": "Mountain", "U": "Island", "B": "Swamp", "W": "Plains", "G": "Forest"}
        if colors:
            primary_color = colors[0]
            land_name = basic_land_map.get(primary_color, "Mountain")
            main_deck.append({"card_name": land_name, "quantity": 20})

        for card in instants[2:5]:
            sideboard.append({"card_name": card.name, "quantity": 3})

        deck_data = {
            "name": f"{archetype.title()} {'/'.join(colors)}",
            "strategy_summary": f"A {archetype} deck focusing on {strategy}",
            "main_deck": main_deck,
            "sideboard": sideboard,
            "slot_recommendations": [],
            "sideboard_guide": [],
        }

        deck_data = await self._enrich_deck_with_card_data(deck_data)
        return deck_data
