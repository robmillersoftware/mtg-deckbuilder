from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.card import Card
from app.schemas.deck import DeckValidationReport, DeckValidationError

logger = logging.getLogger(__name__)


# Basic land names that can have any number of copies
BASIC_LANDS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}

# Standard deck requirements
MAIN_DECK_SIZE = 60
SIDEBOARD_SIZE = 15
MAX_COPIES = 4


class DeckValidator:
    """
    Validates MTG decks against Standard format rules.
    Checks: deck size, copy limits, legality, banned list.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def validate(
        self,
        main_deck: List[Dict[str, Any]],
        sideboard: List[Dict[str, Any]],
    ) -> DeckValidationReport:
        """
        Validate a deck against Standard rules.

        Args:
            main_deck: List of {card_name, quantity, ...} entries
            sideboard: List of {card_name, quantity, ...} entries

        Returns:
            DeckValidationReport with is_valid flag and any errors
        """
        errors: List[DeckValidationError] = []
        warnings: List[str] = []

        # Calculate total cards
        main_deck_count = sum(entry.get("quantity", 0) for entry in main_deck)
        sideboard_count = sum(entry.get("quantity", 0) for entry in sideboard)

        # Check deck sizes
        if main_deck_count != MAIN_DECK_SIZE:
            errors.append(
                DeckValidationError(
                    type="main_deck_size",
                    message=f"Main deck must be exactly {MAIN_DECK_SIZE} cards",
                    expected=MAIN_DECK_SIZE,
                    actual=main_deck_count,
                )
            )

        if sideboard_count != SIDEBOARD_SIZE:
            errors.append(
                DeckValidationError(
                    type="sideboard_size",
                    message=f"Sideboard must be exactly {SIDEBOARD_SIZE} cards",
                    expected=SIDEBOARD_SIZE,
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

        # Check 4-copy limit (except basic lands)
        for card_name, count in card_counts.items():
            if card_name not in BASIC_LANDS and count > MAX_COPIES:
                errors.append(
                    DeckValidationError(
                        type="card_limit_exceeded",
                        message=f"Maximum {MAX_COPIES} copies of {card_name} allowed",
                        card_name=card_name,
                        expected=MAX_COPIES,
                        actual=count,
                    )
                )

        # Check card legality
        if all_card_names:
            result = await self.db.execute(
                select(Card.name, Card.is_standard_legal)
                .where(func.lower(Card.name).in_([n.lower() for n in all_card_names]))
            )
            card_legality = {row[0].lower(): row[1] for row in result.all()}

            for card_name in all_card_names:
                card_name_lower = card_name.lower()
                if card_name_lower not in card_legality:
                    errors.append(
                        DeckValidationError(
                            type="card_not_found",
                            message=f"Card '{card_name}' not found in database",
                            card_name=card_name,
                        )
                    )
                elif not card_legality[card_name_lower]:
                    errors.append(
                        DeckValidationError(
                            type="not_standard_legal",
                            message=f"Card '{card_name}' is not Standard-legal",
                            card_name=card_name,
                        )
                    )

        # TODO: Check banned list when implemented

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
    ) -> Optional[DeckValidationError]:
        """
        Validate a single card addition.

        Args:
            card_name: Name of the card
            quantity: Quantity to add
            current_copies: Current copies in deck

        Returns:
            DeckValidationError if invalid, None if valid
        """
        # Check copy limit
        if card_name not in BASIC_LANDS:
            total_copies = current_copies + quantity
            if total_copies > MAX_COPIES:
                return DeckValidationError(
                    type="card_limit_exceeded",
                    message=f"Maximum {MAX_COPIES} copies of {card_name} allowed",
                    card_name=card_name,
                    expected=MAX_COPIES,
                    actual=total_copies,
                )

        # Check legality
        result = await self.db.execute(
            select(Card.is_standard_legal)
            .where(func.lower(Card.name) == card_name.lower())
        )
        row = result.first()

        if row is None:
            return DeckValidationError(
                type="card_not_found",
                message=f"Card '{card_name}' not found in database",
                card_name=card_name,
            )

        if not row[0]:
            return DeckValidationError(
                type="not_standard_legal",
                message=f"Card '{card_name}' is not Standard-legal",
                card_name=card_name,
            )

        return None
