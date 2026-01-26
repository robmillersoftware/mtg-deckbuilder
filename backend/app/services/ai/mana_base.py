"""Mana base management for deck building."""

import logging
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

logger = logging.getLogger(__name__)


class ManaBaseMixin:
    """Mixin providing mana base management methods."""

    db: AsyncSession  # Type hint for mixin

    async def _fix_land_count(
        self,
        deck_data: Dict[str, Any],
        target_lands: int,
        colors: List[str],
    ) -> None:
        """
        Validate and fix the land count in a generated deck.
        If the deck has too few lands, add basics. If too many, remove some.
        Modifies deck_data in place.
        """
        from app.models.card import Card

        main_deck = deck_data.get("main_deck", [])

        land_count = 0
        land_entries = []
        nonland_entries = []

        for entry in main_deck:
            card_name = entry.get("card_name", "").lower()
            is_land = any(basic in card_name for basic in
                        ["plains", "island", "swamp", "mountain", "forest"]) or \
                      "land" in card_name.lower()

            if not is_land:
                query = select(Card.type_line).where(
                    func.lower(Card.name) == card_name.lower()
                ).limit(1)
                result = await self.db.execute(query)
                type_line = result.scalar_one_or_none()
                if type_line and "land" in type_line.lower():
                    is_land = True

            if is_land:
                land_count += entry.get("quantity", 1)
                land_entries.append(entry)
            else:
                nonland_entries.append(entry)

        logger.info(f"Land count validation: {land_count} lands (target: {target_lands})")

        if land_count < target_lands:
            lands_needed = target_lands - land_count
            basic_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
            basics_to_add = [basic_map[c] for c in colors if c in basic_map]

            if basics_to_add:
                per_basic = lands_needed // len(basics_to_add)
                remainder = lands_needed % len(basics_to_add)

                for i, basic in enumerate(basics_to_add):
                    qty = per_basic + (1 if i < remainder else 0)
                    if qty > 0:
                        existing = next((e for e in land_entries
                                       if e.get("card_name", "").lower() == basic.lower()), None)
                        if existing:
                            existing["quantity"] = existing.get("quantity", 0) + qty
                        else:
                            land_entries.append({"card_name": basic, "quantity": qty})

                logger.info(f"Added {lands_needed} basic lands to reach target of {target_lands}")

            total_nonlands = sum(e.get("quantity", 1) for e in nonland_entries)
            target_nonlands = 60 - target_lands

            if total_nonlands > target_nonlands:
                cards_to_remove = total_nonlands - target_nonlands
                while cards_to_remove > 0 and nonland_entries:
                    last_entry = nonland_entries[-1]
                    qty = last_entry.get("quantity", 1)
                    if qty <= cards_to_remove:
                        nonland_entries.pop()
                        cards_to_remove -= qty
                    else:
                        last_entry["quantity"] = qty - cards_to_remove
                        cards_to_remove = 0

        elif land_count > target_lands + 2:
            lands_to_remove = land_count - target_lands
            for entry in land_entries:
                if lands_to_remove <= 0:
                    break
                card_name = entry.get("card_name", "").lower()
                if card_name in ["plains", "island", "swamp", "mountain", "forest"]:
                    qty = entry.get("quantity", 1)
                    remove_qty = min(qty, lands_to_remove)
                    entry["quantity"] = qty - remove_qty
                    lands_to_remove -= remove_qty

            land_entries = [e for e in land_entries if e.get("quantity", 0) > 0]

        deck_data["main_deck"] = nonland_entries + land_entries

    async def _ensure_cedh_mana_base(
        self,
        deck_data: Dict[str, Any],
        colors: List[str],
    ) -> None:
        """
        Ensure cEDH deck has proper mana base with fetches, duals, shocks.
        Removes bad lands and ensures all on-color premium lands are included.
        Modifies deck_data in place.
        """
        from app.services.cedh_knowledge import get_cedh_lands_for_colors
        from app.models.card import Card

        main_deck = deck_data.get("main_deck", [])
        optimal_lands = get_cedh_lands_for_colors(colors)

        bad_lands = {
            "evolving wilds", "terramorphic expanse", "fabled passage",
            "gateway plaza", "rupture spire", "transguild promenade",
            "opulent palace", "arcane sanctum", "crumbling necropolis",
            "jungle shrine", "savage lands", "seaside citadel",
            "sandsteppe citadel", "frontier bivouac", "mystic monastery",
            "nomad outpost", "hedge maze", "soulstone sanctuary",
            "demolition field", "fountainport", "meticulous archive",
        }

        required_lands = set()
        for category in ["fetch_lands", "original_duals", "shock_lands", "rainbow_lands", "utility_lands"]:
            for land in optimal_lands.get(category, []):
                required_lands.add(land.lower())

        land_entries = []
        nonland_entries = []
        existing_lands = set()

        for entry in main_deck:
            card_name = entry.get("card_name", "")
            card_name_lower = card_name.lower()

            query = select(Card.type_line).where(
                func.lower(Card.name) == card_name_lower
            ).limit(1)
            result = await self.db.execute(query)
            type_line = result.scalar_one_or_none()

            is_land = type_line and "land" in type_line.lower()
            is_basic = card_name_lower in [
                "plains", "island", "swamp", "mountain", "forest",
                "snow-covered plains", "snow-covered island",
                "snow-covered swamp", "snow-covered mountain",
                "snow-covered forest"
            ]

            if is_land:
                if card_name_lower in bad_lands:
                    logger.debug(f"[AI-SERVICE] Removing bad land: {card_name}")
                    continue

                if is_basic:
                    if card_name_lower in existing_lands:
                        logger.debug(f"[AI-SERVICE] Removing duplicate basic: {card_name}")
                        continue
                    entry["quantity"] = 1

                existing_lands.add(card_name_lower)
                land_entries.append(entry)
            else:
                nonland_entries.append(entry)

        # Add missing required lands (priority: fetches > duals > rainbow > utility > shocks)
        for category in ["fetch_lands", "original_duals", "rainbow_lands", "utility_lands", "shock_lands"]:
            for land in optimal_lands.get(category, []):
                if land.lower() not in existing_lands:
                    query = select(Card.name).where(
                        func.lower(Card.name) == land.lower()
                    ).limit(1)
                    result = await self.db.execute(query)
                    valid_name = result.scalar_one_or_none()

                    if valid_name:
                        land_entries.append({"card_name": valid_name, "quantity": 1})
                        existing_lands.add(land.lower())
                        logger.debug(f"[AI-SERVICE] Added cEDH land: {valid_name}")

        target_lands = 29
        current_land_count = len(land_entries)

        if current_land_count < target_lands - 2:
            basic_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
            for color in colors:
                if color.upper() in basic_map:
                    basic = basic_map[color.upper()]
                    if basic.lower() not in existing_lands:
                        land_entries.append({"card_name": basic, "quantity": 1})
                        existing_lands.add(basic.lower())
                        logger.debug(f"[AI-SERVICE] Added basic land: {basic}")

        current_land_count = len(land_entries)
        if current_land_count > target_lands:
            excess = current_land_count - target_lands
            filtered_lands = []
            for entry in land_entries:
                card_name_lower = entry.get("card_name", "").lower()
                is_basic = card_name_lower in ["plains", "island", "swamp", "mountain", "forest"]
                if is_basic and excess > 0:
                    excess -= 1
                    logger.debug(f"[AI-SERVICE] Removed excess basic: {entry.get('card_name')}")
                    continue
                filtered_lands.append(entry)
            land_entries = filtered_lands

        deck_data["main_deck"] = nonland_entries + land_entries
        logger.debug(f"[AI-SERVICE] cEDH mana base: {len(land_entries)} lands")
