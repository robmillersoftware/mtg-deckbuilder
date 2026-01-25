from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.card import Card
from app.schemas.deck import DeckValidationReport, DeckValidationError
from app.services.card_service import FORMAT_LEGALITY_MAP

logger = logging.getLogger(__name__)


# Basic land names that can have any number of copies
BASIC_LANDS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}

# Format-specific deck construction rules
FORMAT_RULES = {
    "standard": {"main_deck_size": 60, "sideboard_size": 15, "max_copies": 4},
    "historic": {"main_deck_size": 60, "sideboard_size": 15, "max_copies": 4},
    "modern": {"main_deck_size": 60, "sideboard_size": 15, "max_copies": 4},
    "legacy": {"main_deck_size": 60, "sideboard_size": 15, "max_copies": 4},
    "cedh": {"main_deck_size": 99, "sideboard_size": 0, "max_copies": 1},
}

# Backwards-compatible defaults
MAIN_DECK_SIZE = 60
SIDEBOARD_SIZE = 15
MAX_COPIES = 4


class DeckValidator:
    """
    Validates MTG decks against format-specific rules.
    Checks: deck size, copy limits, legality, banned list.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def validate(
        self,
        main_deck: List[Dict[str, Any]],
        sideboard: List[Dict[str, Any]],
        format: str = "standard",
    ) -> DeckValidationReport:
        """
        Validate a deck against format-specific rules.

        Args:
            main_deck: List of {card_name, quantity, ...} entries
            sideboard: List of {card_name, quantity, ...} entries
            format: Game format (standard, historic, modern, legacy, cedh)

        Returns:
            DeckValidationReport with is_valid flag and any errors
        """
        rules = FORMAT_RULES.get(format, FORMAT_RULES["standard"])
        main_deck_size = rules["main_deck_size"]
        sideboard_size = rules["sideboard_size"]
        max_copies = rules["max_copies"]

        errors: List[DeckValidationError] = []
        warnings: List[str] = []

        # Calculate total cards
        main_deck_count = sum(entry.get("quantity", 0) for entry in main_deck)
        sideboard_count = sum(entry.get("quantity", 0) for entry in sideboard)

        # Check deck sizes
        if main_deck_count != main_deck_size:
            errors.append(
                DeckValidationError(
                    type="main_deck_size",
                    message=f"Main deck must be exactly {main_deck_size} cards",
                    expected=main_deck_size,
                    actual=main_deck_count,
                )
            )

        if sideboard_count != sideboard_size:
            if sideboard_size == 0 and sideboard_count > 0:
                errors.append(
                    DeckValidationError(
                        type="sideboard_size",
                        message=f"{format.upper() if format != 'cedh' else 'cEDH'} does not use a sideboard",
                        expected=0,
                        actual=sideboard_count,
                    )
                )
            elif sideboard_size > 0:
                errors.append(
                    DeckValidationError(
                        type="sideboard_size",
                        message=f"Sideboard must be exactly {sideboard_size} cards",
                        expected=sideboard_size,
                        actual=sideboard_count,
                    )
                )

        # Count copies of each card across main and sideboard
        card_counts: Dict[str, int] = defaultdict(int)
        all_card_names: set = set()

        for entry in main_deck:
            card_name = entry.get("card_name", "")
            quantity = entry.get("quantity", 0)
            card_counts[card_name] += quantity
            all_card_names.add(card_name)

        for entry in sideboard:
            card_name = entry.get("card_name", "")
            quantity = entry.get("quantity", 0)
            card_counts[card_name] += quantity
            all_card_names.add(card_name)

        # Check copy limit (except basic lands)
        for card_name, count in card_counts.items():
            if card_name not in BASIC_LANDS and count > max_copies:
                if format == "cedh":
                    errors.append(
                        DeckValidationError(
                            type="card_limit_exceeded",
                            message=f"cEDH is singleton - only 1 copy of {card_name} allowed",
                            card_name=card_name,
                            expected=1,
                            actual=count,
                        )
                    )
                else:
                    errors.append(
                        DeckValidationError(
                            type="card_limit_exceeded",
                            message=f"Maximum {max_copies} copies of {card_name} allowed",
                            card_name=card_name,
                            expected=max_copies,
                            actual=count,
                        )
                    )

        # Check card legality
        if all_card_names:
            legality_key = FORMAT_LEGALITY_MAP.get(format, "standard")

            result = await self.db.execute(
                select(Card.name, Card.is_standard_legal, Card.legalities)
                .where(func.lower(Card.name).in_([n.lower() for n in all_card_names]))
            )
            rows = result.all()
            card_data = {row[0].lower(): {"is_standard_legal": row[1], "legalities": row[2]} for row in rows}

            for card_name in all_card_names:
                card_name_lower = card_name.lower()
                if card_name_lower not in card_data:
                    errors.append(
                        DeckValidationError(
                            type="card_not_found",
                            message=f"Card '{card_name}' not found in database",
                            card_name=card_name,
                        )
                    )
                else:
                    data = card_data[card_name_lower]
                    legalities = data.get("legalities") or {}
                    is_legal = legalities.get(legality_key) == "legal"

                    # Fallback to is_standard_legal for standard format
                    if format == "standard":
                        is_legal = data["is_standard_legal"]

                    if not is_legal:
                        format_display = "cEDH" if format == "cedh" else format.capitalize()
                        errors.append(
                            DeckValidationError(
                                type="not_legal",
                                message=f"Card '{card_name}' is not legal in {format_display}",
                                card_name=card_name,
                            )
                        )

        # Note: Banned cards are already handled - Scryfall marks them as not legal

        return DeckValidationReport(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            main_deck_count=main_deck_count,
            sideboard_count=sideboard_count,
        )

    async def validate_single_card(
        self,
        card_name: str,
        quantity: int,
        current_copies: int = 0,
        format: str = "standard",
    ) -> Optional[DeckValidationError]:
        """
        Validate a single card addition.

        Args:
            card_name: Name of the card
            quantity: Quantity to add
            current_copies: Current copies in deck
            format: Game format

        Returns:
            DeckValidationError if invalid, None if valid
        """
        rules = FORMAT_RULES.get(format, FORMAT_RULES["standard"])
        max_copies = rules["max_copies"]

        # Check copy limit
        if card_name not in BASIC_LANDS:
            total_copies = current_copies + quantity
            if total_copies > max_copies:
                if format == "cedh":
                    return DeckValidationError(
                        type="card_limit_exceeded",
                        message=f"cEDH is singleton - only 1 copy of {card_name} allowed",
                        card_name=card_name,
                        expected=1,
                        actual=total_copies,
                    )
                return DeckValidationError(
                    type="card_limit_exceeded",
                    message=f"Maximum {max_copies} copies of {card_name} allowed",
                    card_name=card_name,
                    expected=max_copies,
                    actual=total_copies,
                )

        # Check legality
        legality_key = FORMAT_LEGALITY_MAP.get(format, "standard")
        result = await self.db.execute(
            select(Card.is_standard_legal, Card.legalities)
            .where(func.lower(Card.name) == card_name.lower())
        )
        row = result.first()

        if row is None:
            return DeckValidationError(
                type="card_not_found",
                message=f"Card '{card_name}' not found in database",
                card_name=card_name,
            )

        legalities = row[1] or {}
        is_legal = legalities.get(legality_key) == "legal"
        if format == "standard":
            is_legal = row[0]

        if not is_legal:
            format_display = "cEDH" if format == "cedh" else format.capitalize()
            return DeckValidationError(
                type="not_legal",
                message=f"Card '{card_name}' is not legal in {format_display}",
                card_name=card_name,
            )

        return None
