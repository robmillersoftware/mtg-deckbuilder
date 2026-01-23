"""
Card Role Service - Query helpers for role-based card lookup.

Used by deck generation to find "best in slot" cards for specific roles.
"""

import logging
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import joinedload

from app.models.card import Card, CardRole, CARD_ROLES

logger = logging.getLogger(__name__)


class CardRoleService:
    """Service for querying cards by their functional roles."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_best_in_slot(
        self,
        role: str,
        colors: Optional[List[str]] = None,
        max_cmc: Optional[float] = None,
        min_efficiency: int = 3,
        limit: int = 4,
        exclude_cards: Optional[List[str]] = None,
    ) -> List[Card]:
        """
        Get the most efficient cards for a given role.

        Args:
            role: The role to search for (e.g., "removal_targeted")
            colors: Color identity filter (e.g., ["W", "B"])
            max_cmc: Maximum converted mana cost
            min_efficiency: Minimum efficiency rating (1-5)
            limit: Number of cards to return
            exclude_cards: Card names to exclude

        Returns:
            List of Card objects sorted by efficiency (desc) then CMC (asc)
        """
        if role not in CARD_ROLES:
            logger.warning(f"Invalid role: {role}")
            return []

        query = (
            select(Card)
            .join(CardRole, Card.id == CardRole.card_id)
            .where(
                and_(
                    CardRole.role == role,
                    CardRole.efficiency >= min_efficiency,
                    Card.is_standard_legal == True,
                )
            )
            .order_by(CardRole.efficiency.desc(), Card.cmc.asc())
        )

        # Color filter - card's color identity must be subset of deck colors
        if colors:
            # For colorless cards (empty color_identity), always include
            # For colored cards, all colors must be in the allowed colors
            color_conditions = [
                Card.color_identity == [],  # Colorless cards
                Card.color_identity == None,  # Null color identity
            ]
            # Add condition for each color - all card colors must be in allowed
            if colors:
                # Card's color identity must only contain allowed colors
                for color in ["W", "U", "B", "R", "G"]:
                    if color not in colors:
                        # Exclude cards with this color
                        color_conditions = []
                        break

                # Simpler approach: use array containment
                # Card color_identity must be contained in the allowed colors
                query = query.where(
                    or_(
                        Card.color_identity == [],
                        Card.color_identity == None,
                        Card.color_identity.contained_by(colors),
                    )
                )

        # CMC filter
        if max_cmc is not None:
            query = query.where(Card.cmc <= max_cmc)

        # Exclude specific cards
        if exclude_cards:
            query = query.where(Card.name.notin_(exclude_cards))

        query = query.limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_cards_by_roles(
        self,
        roles: List[str],
        colors: Optional[List[str]] = None,
        min_efficiency: int = 2,
        limit_per_role: int = 10,
    ) -> Dict[str, List[Card]]:
        """
        Get cards grouped by multiple roles.

        Args:
            roles: List of roles to fetch
            colors: Color identity filter
            min_efficiency: Minimum efficiency rating
            limit_per_role: Max cards per role

        Returns:
            Dict mapping role -> list of cards
        """
        result = {}
        for role in roles:
            cards = await self.get_best_in_slot(
                role=role,
                colors=colors,
                min_efficiency=min_efficiency,
                limit=limit_per_role,
            )
            if cards:
                result[role] = cards
        return result

    async def get_removal_package(
        self,
        colors: List[str],
        count: int = 6,
        include_mass: bool = True,
    ) -> List[Card]:
        """
        Get a removal package for a deck.

        Args:
            colors: Deck's color identity
            count: Total removal cards wanted
            include_mass: Whether to include board wipes

        Returns:
            List of removal cards
        """
        cards = []
        exclude = []

        # Get targeted removal first (most important)
        targeted = await self.get_best_in_slot(
            role="removal_targeted",
            colors=colors,
            limit=count - 1 if include_mass else count,
            exclude_cards=exclude,
        )
        cards.extend(targeted)
        exclude.extend([c.name for c in targeted])

        # Add a board wipe if requested
        if include_mass and len(cards) < count:
            mass = await self.get_best_in_slot(
                role="removal_mass",
                colors=colors,
                limit=1,
                exclude_cards=exclude,
            )
            cards.extend(mass)

        return cards[:count]

    async def get_lands_for_deck(
        self,
        colors: List[str],
        nonbasic_count: int = 16,
        prefer_untapped: bool = True,
    ) -> Dict[str, List[Card]]:
        """
        Get a land package for a deck following the priority system:
        1. Utility lands (synergistic)
        2. Untapped fixing
        3. Tapped fixing with upside

        Args:
            colors: Deck's color identity
            nonbasic_count: Target number of nonbasic lands
            prefer_untapped: Prioritize untapped lands

        Returns:
            Dict with 'utility', 'fixing_untapped', 'fixing_tapped' keys
        """
        result = {
            "utility": [],
            "fixing_untapped": [],
            "fixing_tapped": [],
        }
        exclude = []

        # 1. Utility lands (limit to ~4)
        utility = await self.get_best_in_slot(
            role="land_utility",
            colors=colors,
            min_efficiency=3,
            limit=4,
        )
        result["utility"] = utility
        exclude.extend([c.name for c in utility])

        # 2. Creature lands (add to utility, limit to ~2)
        creature_lands = await self.get_best_in_slot(
            role="land_creature",
            colors=colors,
            min_efficiency=3,
            limit=2,
            exclude_cards=exclude,
        )
        result["utility"].extend(creature_lands)
        exclude.extend([c.name for c in creature_lands])

        remaining = nonbasic_count - len(result["utility"])

        # 3. Untapped fixing
        if remaining > 0:
            untapped = await self.get_best_in_slot(
                role="land_fixing_untapped",
                colors=colors,
                min_efficiency=2,
                limit=remaining,
                exclude_cards=exclude,
            )
            result["fixing_untapped"] = untapped
            exclude.extend([c.name for c in untapped])
            remaining -= len(untapped)

        # 4. Tapped fixing (fill remaining)
        if remaining > 0:
            tapped = await self.get_best_in_slot(
                role="land_fixing_tapped",
                colors=colors,
                min_efficiency=2,
                limit=remaining,
                exclude_cards=exclude,
            )
            result["fixing_tapped"] = tapped

        return result

    async def get_card_roles(self, card_name: str) -> List[Dict[str, Any]]:
        """Get all roles for a specific card."""
        result = await self.db.execute(
            select(CardRole)
            .join(Card, Card.id == CardRole.card_id)
            .where(Card.name == card_name)
            .order_by(CardRole.efficiency.desc())
        )
        roles = result.scalars().all()

        return [
            {
                "role": r.role,
                "efficiency": r.efficiency,
                "reasoning": r.reasoning,
            }
            for r in roles
        ]

    async def find_similar_cards(
        self,
        card_name: str,
        colors: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Card]:
        """
        Find cards with similar roles to a given card.
        Useful for finding alternatives or budget replacements.
        """
        # First get the roles of the target card
        roles = await self.get_card_roles(card_name)
        if not roles:
            return []

        # Get the primary role (highest efficiency)
        primary_role = roles[0]["role"]

        # Find other cards with the same primary role
        similar = await self.get_best_in_slot(
            role=primary_role,
            colors=colors,
            min_efficiency=2,
            limit=limit + 1,  # +1 to account for the original card
            exclude_cards=[card_name],
        )

        return similar[:limit]
