from typing import List, Dict, Any, Optional, Tuple
import re
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.deck import (
    DeckImportResponse,
    DeckCardEntry,
    CardSuggestion,
)
from app.services.card_service import CardService
from app.services.deck_validator import DeckValidator

logger = logging.getLogger(__name__)


class DeckImporter:
    """
    Imports decks from various text formats.
    Supports: Arena, MTGO, and simple "N Card Name" format.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.card_service = CardService(db)
        self.validator = DeckValidator(db)

    async def import_deck(
        self,
        decklist_text: str,
        format_hint: str = "auto",
    ) -> DeckImportResponse:
        """
        Import a deck from text.

        Args:
            decklist_text: Raw decklist text
            format_hint: "arena", "mtgo", "simple", or "auto"

        Returns:
            DeckImportResponse with parsed deck and any errors
        """
        # Detect format if auto
        if format_hint == "auto":
            format_hint = self._detect_format(decklist_text)

        # Parse based on format
        if format_hint == "arena":
            main_deck, sideboard, errors = await self._parse_arena(decklist_text)
        elif format_hint == "mtgo":
            main_deck, sideboard, errors = await self._parse_mtgo(decklist_text)
        else:
            main_deck, sideboard, errors = await self._parse_simple(decklist_text)

        # Validate and get suggestions for invalid cards
        warnings = []
        card_suggestions = []

        all_entries = main_deck + sideboard
        for entry in all_entries:
            card = await self.card_service.get_by_name(entry.card_name)
            if card:
                entry.card_id = card.id
            else:
                # Try fuzzy search for suggestions
                similar = await self.card_service.fuzzy_search_by_name(entry.card_name, limit=3)
                if similar:
                    card_suggestions.append(
                        CardSuggestion(
                            original=entry.card_name,
                            suggestions=[c.name for c in similar],
                            reason="Card not found - did you mean one of these?",
                        )
                    )
                    errors.append(f"Card not found: '{entry.card_name}'")

        # Validate deck structure
        validation = await self.validator.validate(
            [e.model_dump() for e in main_deck],
            [e.model_dump() for e in sideboard],
        )

        for error in validation.errors:
            if error.type not in ["main_deck_size", "sideboard_size", "card_not_found"]:
                errors.append(error.message)

        # Determine archetype (simple heuristic)
        archetype = self._classify_archetype(main_deck)

        return DeckImportResponse(
            valid=len(errors) == 0,
            main_deck=main_deck,
            sideboard=sideboard,
            errors=errors,
            warnings=warnings,
            card_suggestions=card_suggestions,
            archetype=archetype,
        )

    def _detect_format(self, text: str) -> str:
        """Detect the deck format from text."""
        lines = text.strip().split("\n")

        # Arena format has set code and collector number: "4 Lightning Strike (DMU) 137"
        arena_pattern = r"^\d+\s+.+\s+\([A-Z0-9]{3,4}\)\s+\d+"
        if any(re.match(arena_pattern, line.strip()) for line in lines):
            return "arena"

        # MTGO format has "SB:" prefix for sideboard
        if any(line.strip().upper().startswith("SB:") for line in lines):
            return "mtgo"

        return "simple"

    async def _parse_arena(
        self,
        text: str,
    ) -> Tuple[List[DeckCardEntry], List[DeckCardEntry], List[str]]:
        """Parse Arena format: "4 Lightning Strike (DMU) 137" """
        main_deck = []
        sideboard = []
        errors = []
        in_sideboard = False

        # Arena pattern: quantity, name, (set), collector number
        pattern = r"^(\d+)\s+(.+?)\s+\(([A-Z0-9]{3,4})\)\s+(\d+)"
        simple_pattern = r"^(\d+)\s+(.+)$"

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                # Empty line often separates main deck from sideboard
                in_sideboard = True
                continue

            if line.lower().startswith("sideboard") or line.lower() == "sb:":
                in_sideboard = True
                continue

            match = re.match(pattern, line)
            if match:
                quantity = int(match.group(1))
                card_name = match.group(2).strip()
                set_code = match.group(3)
                collector_number = match.group(4)

                entry = DeckCardEntry(
                    card_name=card_name,
                    quantity=quantity,
                    set_code=set_code,
                    collector_number=collector_number,
                )

                if in_sideboard:
                    sideboard.append(entry)
                else:
                    main_deck.append(entry)
            else:
                # Try simple pattern as fallback
                simple_match = re.match(simple_pattern, line)
                if simple_match:
                    quantity = int(simple_match.group(1))
                    card_name = simple_match.group(2).strip()
                    # Remove set info if present
                    card_name = re.sub(r"\s+\([A-Z0-9]{3,4}\).*$", "", card_name)

                    entry = DeckCardEntry(card_name=card_name, quantity=quantity)
                    if in_sideboard:
                        sideboard.append(entry)
                    else:
                        main_deck.append(entry)
                elif line and not line.startswith("#"):
                    errors.append(f"Could not parse line: '{line}'")

        return main_deck, sideboard, errors

    async def _parse_mtgo(
        self,
        text: str,
    ) -> Tuple[List[DeckCardEntry], List[DeckCardEntry], List[str]]:
        """Parse MTGO format with optional "SB:" prefix."""
        main_deck = []
        sideboard = []
        errors = []

        pattern = r"^(?:SB:\s*)?(\d+)\s+(.+)$"

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            is_sideboard = line.upper().startswith("SB:")

            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                quantity = int(match.group(1))
                card_name = match.group(2).strip()

                entry = DeckCardEntry(card_name=card_name, quantity=quantity)

                if is_sideboard:
                    sideboard.append(entry)
                else:
                    main_deck.append(entry)
            elif line and not line.upper().startswith("SB:"):
                errors.append(f"Could not parse line: '{line}'")

        return main_deck, sideboard, errors

    async def _parse_simple(
        self,
        text: str,
    ) -> Tuple[List[DeckCardEntry], List[DeckCardEntry], List[str]]:
        """Parse simple "N Card Name" format."""
        main_deck = []
        sideboard = []
        errors = []
        in_sideboard = False

        pattern = r"^(\d+)\s+(.+)$"

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                in_sideboard = True
                continue

            if line.lower() in ["sideboard", "sideboard:", "sb:"]:
                in_sideboard = True
                continue

            if line.startswith("#"):
                continue

            match = re.match(pattern, line)
            if match:
                quantity = int(match.group(1))
                card_name = match.group(2).strip()

                entry = DeckCardEntry(card_name=card_name, quantity=quantity)

                if in_sideboard:
                    sideboard.append(entry)
                else:
                    main_deck.append(entry)
            else:
                errors.append(f"Could not parse line: '{line}'")

        return main_deck, sideboard, errors

    def _classify_archetype(self, main_deck: List[DeckCardEntry]) -> Optional[str]:
        """Simple archetype classification based on card presence."""
        card_names = {e.card_name.lower() for e in main_deck}

        # Simple heuristics based on common cards
        if any("monastery swiftspear" in n for n in card_names):
            return "Red Aggro"
        if any("counterspell" in n or "negate" in n for n in card_names):
            if any("teferi" in n for n in card_names):
                return "Azorius Control"
            return "Control"
        if any("sheoldred" in n for n in card_names):
            return "Midrange"

        return None
