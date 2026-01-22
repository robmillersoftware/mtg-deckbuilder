from typing import List, Dict, Any, Optional
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.card import Card

logger = logging.getLogger(__name__)


class DeckExporter:
    """
    Exports decks to various text formats.
    Supports: Arena, MTGO, and plain text.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def export_deck(
        self,
        main_deck: List[Dict[str, Any]],
        sideboard: List[Dict[str, Any]],
        format: str = "arena",
        deck_name: Optional[str] = None,
    ) -> str:
        """
        Export a deck to the specified format.

        Args:
            main_deck: List of {card_name, quantity, ...} entries
            sideboard: List of {card_name, quantity, ...} entries
            format: "arena", "mtgo", or "plain"
            deck_name: Optional deck name for comments

        Returns:
            Formatted deck string
        """
        if format == "arena":
            return await self._export_arena(main_deck, sideboard, deck_name)
        elif format == "mtgo":
            return await self._export_mtgo(main_deck, sideboard, deck_name)
        else:
            return await self._export_plain(main_deck, sideboard, deck_name)

    async def _export_arena(
        self,
        main_deck: List[Dict[str, Any]],
        sideboard: List[Dict[str, Any]],
        deck_name: Optional[str] = None,
    ) -> str:
        """Export in Arena format: "4 Lightning Strike (DMU) 137" """
        lines = []

        if deck_name:
            lines.append(f"// {deck_name}")
            lines.append("")

        # Get card info for set codes and collector numbers
        card_info = await self._get_card_info(main_deck + sideboard)

        # Main deck
        for entry in main_deck:
            card_name = entry.get("card_name", "")
            quantity = entry.get("quantity", 1)
            info = card_info.get(card_name.lower(), {})
            set_code = entry.get("set_code") or info.get("set_code", "")
            collector_number = entry.get("collector_number") or info.get("collector_number", "")

            if set_code and collector_number:
                lines.append(f"{quantity} {card_name} ({set_code.upper()}) {collector_number}")
            else:
                lines.append(f"{quantity} {card_name}")

        # Sideboard
        if sideboard:
            lines.append("")
            for entry in sideboard:
                card_name = entry.get("card_name", "")
                quantity = entry.get("quantity", 1)
                info = card_info.get(card_name.lower(), {})
                set_code = entry.get("set_code") or info.get("set_code", "")
                collector_number = entry.get("collector_number") or info.get("collector_number", "")

                if set_code and collector_number:
                    lines.append(f"{quantity} {card_name} ({set_code.upper()}) {collector_number}")
                else:
                    lines.append(f"{quantity} {card_name}")

        return "\n".join(lines)

    async def _export_mtgo(
        self,
        main_deck: List[Dict[str, Any]],
        sideboard: List[Dict[str, Any]],
        deck_name: Optional[str] = None,
    ) -> str:
        """Export in MTGO format with "SB:" prefix for sideboard."""
        lines = []

        if deck_name:
            lines.append(f"// {deck_name}")
            lines.append("")

        # Main deck
        for entry in main_deck:
            card_name = entry.get("card_name", "")
            quantity = entry.get("quantity", 1)
            lines.append(f"{quantity} {card_name}")

        # Sideboard
        if sideboard:
            lines.append("")
            for entry in sideboard:
                card_name = entry.get("card_name", "")
                quantity = entry.get("quantity", 1)
                lines.append(f"SB: {quantity} {card_name}")

        return "\n".join(lines)

    async def _export_plain(
        self,
        main_deck: List[Dict[str, Any]],
        sideboard: List[Dict[str, Any]],
        deck_name: Optional[str] = None,
    ) -> str:
        """Export in plain text format."""
        lines = []

        if deck_name:
            lines.append(deck_name)
            lines.append("=" * len(deck_name))
            lines.append("")

        lines.append("Main Deck:")
        for entry in main_deck:
            card_name = entry.get("card_name", "")
            quantity = entry.get("quantity", 1)
            lines.append(f"{quantity} {card_name}")

        if sideboard:
            lines.append("")
            lines.append("Sideboard:")
            for entry in sideboard:
                card_name = entry.get("card_name", "")
                quantity = entry.get("quantity", 1)
                lines.append(f"{quantity} {card_name}")

        return "\n".join(lines)

    async def _get_card_info(
        self,
        entries: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        """Get set codes and collector numbers for cards."""
        card_names = [e.get("card_name", "").lower() for e in entries if e.get("card_name")]

        if not card_names:
            return {}

        result = await self.db.execute(
            select(Card.name, Card.set_code, Card.collector_number)
            .where(func.lower(Card.name).in_(card_names))
        )

        return {
            row[0].lower(): {
                "set_code": row[1] or "",
                "collector_number": row[2] or "",
            }
            for row in result.all()
        }
