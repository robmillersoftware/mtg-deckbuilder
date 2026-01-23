"""
Card Role Classification Job
Schedule: Run manually or after Scryfall sync

Uses AI to classify all Standard-legal cards into functional roles
for deck building (removal, threats, ramp, etc.)
"""

import asyncio
import logging
import json
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct
from sqlalchemy.dialects.postgresql import insert

from app.db.session import async_session_factory
from app.models.card import Card, CardRole, CARD_ROLES
from app.core.config import settings

logger = logging.getLogger(__name__)

# Batch size for classification (50 cards per API call)
BATCH_SIZE = 50

CLASSIFICATION_PROMPT = """Classify these Magic: The Gathering cards into functional deck-building roles.

CARDS:
{cards_json}

AVAILABLE ROLES (only use these exact values):
- removal_targeted: Destroys/exiles single creature or planeswalker
- removal_mass: Board wipes, destroys/exiles multiple creatures
- removal_artifact_enchantment: Removes artifacts or enchantments
- card_draw: Draws one or more cards
- card_selection: Scry, surveil, look at top cards, filters draws
- ramp: Adds mana or fetches lands
- counterspell: Counters spells
- discard: Forces opponent to discard
- threat_cheap: Efficient creature/threat CMC 2 or less
- threat_midrange: Value creature/planeswalker CMC 3-4
- threat_finisher: Game-ending threat CMC 5+
- protection: Gives hexproof, indestructible, prevents damage
- burn: Deals damage to players/any target
- lifegain: Gains life (as main purpose, not incidental)
- recursion: Returns cards from graveyard
- graveyard_hate: Exiles cards from graveyards
- tutor: Searches library for non-land cards
- land_fixing_untapped: Multi-color land that can enter untapped
- land_fixing_tapped: Multi-color land that enters tapped
- land_utility: Land with useful activated abilities
- land_creature: Land that becomes a creature
- land_basic: Basic land (Plains, Island, Swamp, Mountain, Forest)

EFFICIENCY RATING (1-5):
5 = Best in Standard for this role (Murder for removal, Consider for card selection)
4 = Very good, commonly played
3 = Solid, sees play
2 = Playable but situational
1 = Below rate but has the effect

Return a JSON array. Cards can have MULTIPLE roles (e.g., a creature with ETB removal).
Only include roles that genuinely apply. Omit cards that don't fit any role.

Example output:
[
  {{
    "name": "Murder",
    "roles": [
      {{"role": "removal_targeted", "efficiency": 5, "reasoning": "3 mana instant unconditional removal"}}
    ]
  }},
  {{
    "name": "Bloodtithe Harvester",
    "roles": [
      {{"role": "threat_cheap", "efficiency": 4, "reasoning": "2 mana 3/2 with upside"}},
      {{"role": "removal_targeted", "efficiency": 3, "reasoning": "Can sacrifice to kill small creature"}}
    ]
  }}
]

Only return the JSON array, no other text."""


async def get_unclassified_cards(db: AsyncSession, limit: int = 500) -> List[Card]:
    """Get Standard-legal cards that haven't been classified yet (unique names only).

    Returns one representative Card object per unique card name.
    This avoids processing the same card multiple times across different printings.
    """
    # Subquery to find card names that already have roles
    # Join CardRole -> Card to get names of classified cards
    classified_names_subquery = (
        select(Card.name)
        .join(CardRole, Card.id == CardRole.card_id)
        .distinct()
        .scalar_subquery()
    )

    # First get distinct unclassified card names
    names_result = await db.execute(
        select(distinct(Card.name))
        .where(Card.is_standard_legal == True)
        .where(Card.name.notin_(classified_names_subquery))
        .order_by(Card.name)
        .limit(limit)
    )
    unique_names = [row[0] for row in names_result.all()]

    if not unique_names:
        return []

    # Now fetch one card per name (any printing will do, they have same oracle text)
    result = await db.execute(
        select(Card)
        .where(Card.name.in_(unique_names))
        .distinct(Card.name)
        .order_by(Card.name, Card.id)  # Consistent ordering
    )
    return list(result.scalars().all())


async def classify_cards_batch(cards: List[Card]) -> List[Dict[str, Any]]:
    """Classify a batch of cards using Claude."""
    if not settings.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not configured")
        return []

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        # Build card data for prompt
        cards_data = []
        for card in cards:
            card_info = {
                "name": card.name,
                "mana_cost": card.mana_cost or "",
                "cmc": card.cmc,
                "type": card.type_line or "",
                "text": card.oracle_text or "",
            }
            if card.power and card.toughness:
                card_info["pt"] = f"{card.power}/{card.toughness}"
            cards_data.append(card_info)

        cards_json = json.dumps(cards_data, indent=2)
        prompt = CLASSIFICATION_PROMPT.format(cards_json=cards_json)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        if not response.content:
            logger.error("Empty response from Claude")
            return []

        content = response.content[0].text

        # Parse JSON from response
        if "[" in content:
            json_start = content.index("[")
            json_end = content.rindex("]") + 1
            result = json.loads(content[json_start:json_end])
            return result

        return []

    except Exception as e:
        logger.error(f"Classification API call failed: {e}")
        return []


async def save_card_roles(
    db: AsyncSession,
    cards: List[Card],
    classifications: List[Dict[str, Any]]
) -> int:
    """Save classification results to database for all printings of each card."""
    # Get all card names from this batch
    card_names = [card.name for card in cards]

    # Fetch ALL card IDs for these names (all printings)
    all_cards_result = await db.execute(
        select(Card.id, Card.name)
        .where(Card.name.in_(card_names))
    )
    # Build lookup: card_name -> list of all card IDs with that name
    name_to_ids: Dict[str, List] = defaultdict(list)
    for card_id, card_name in all_cards_result.all():
        name_to_ids[card_name].append(card_id)

    saved = 0
    for classification in classifications:
        card_name = classification.get("name")
        card_ids = name_to_ids.get(card_name, [])

        if not card_ids:
            logger.warning(f"Card not found in batch: {card_name}")
            continue

        roles = classification.get("roles", [])
        for role_data in roles:
            role = role_data.get("role")

            # Validate role
            if role not in CARD_ROLES:
                logger.warning(f"Invalid role '{role}' for card {card_name}")
                continue

            efficiency = role_data.get("efficiency")
            reasoning = role_data.get("reasoning", "")

            # Save role for ALL printings of this card
            for card_id in card_ids:
                stmt = insert(CardRole).values(
                    card_id=card_id,
                    role=role,
                    efficiency=efficiency,
                    confidence=0.9,  # High confidence for Claude classifications
                    reasoning=reasoning,
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_card_role",
                    set_={
                        "efficiency": stmt.excluded.efficiency,
                        "reasoning": stmt.excluded.reasoning,
                        "created_at": datetime.utcnow(),
                    },
                )
                await db.execute(stmt)
                saved += 1

    await db.commit()
    return saved


async def classify_all_cards() -> Dict[str, Any]:
    """
    Main classification function - classifies all unclassified Standard cards.

    Returns:
        Dict with classification statistics
    """
    start_time = datetime.utcnow()
    stats = {
        "started_at": start_time.isoformat(),
        "cards_processed": 0,
        "roles_saved": 0,
        "batches": 0,
        "errors": [],
    }

    async with async_session_factory() as db:
        try:
            # Get count of unique unclassified card NAMES (not printings)
            classified_names_subquery = (
                select(Card.name)
                .join(CardRole, Card.id == CardRole.card_id)
                .distinct()
                .scalar_subquery()
            )
            count_result = await db.execute(
                select(func.count(distinct(Card.name)))
                .where(Card.is_standard_legal == True)
                .where(Card.name.notin_(classified_names_subquery))
            )
            total_unclassified = count_result.scalar()
            logger.info(f"Found {total_unclassified} unique unclassified Standard cards")

            if total_unclassified == 0:
                logger.info("All cards already classified")
                stats["completed_at"] = datetime.utcnow().isoformat()
                return stats

            # Process in batches
            while True:
                cards = await get_unclassified_cards(db, limit=BATCH_SIZE)

                if not cards:
                    break

                stats["batches"] += 1
                logger.info(f"Batch {stats['batches']}: Classifying {len(cards)} cards")

                # Classify batch
                classifications = await classify_cards_batch(cards)

                if classifications:
                    saved = await save_card_roles(db, cards, classifications)
                    stats["roles_saved"] += saved
                    logger.info(f"Batch {stats['batches']}: Saved {saved} roles")
                else:
                    logger.warning(f"Batch {stats['batches']}: No classifications returned")

                stats["cards_processed"] += len(cards)

                # Small delay to avoid rate limits
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Classification job failed: {e}")
            stats["errors"].append(str(e))
            raise

    stats["completed_at"] = datetime.utcnow().isoformat()
    logger.info(f"Classification complete: {stats}")
    return stats


async def get_classification_stats() -> Dict[str, Any]:
    """Get statistics about current card classifications."""
    async with async_session_factory() as db:
        # Total unique Standard card names
        total_result = await db.execute(
            select(func.count(distinct(Card.name))).where(Card.is_standard_legal == True)
        )
        total_cards = total_result.scalar()

        # Unique card names with roles
        classified_result = await db.execute(
            select(func.count(distinct(Card.name)))
            .select_from(Card)
            .join(CardRole, Card.id == CardRole.card_id)
        )
        classified_cards = classified_result.scalar()

        # Roles by type (count unique card names per role)
        role_counts_result = await db.execute(
            select(CardRole.role, func.count(distinct(Card.name)))
            .select_from(CardRole)
            .join(Card, Card.id == CardRole.card_id)
            .group_by(CardRole.role)
            .order_by(func.count(distinct(Card.name)).desc())
        )
        role_counts = {row[0]: row[1] for row in role_counts_result.all()}

        return {
            "total_unique_standard_cards": total_cards,
            "classified_unique_cards": classified_cards,
            "unclassified_unique_cards": total_cards - classified_cards,
            "total_role_assignments": sum(role_counts.values()),
            "unique_cards_by_role": role_counts,
        }


if __name__ == "__main__":
    # Allow running directly for testing
    asyncio.run(classify_all_cards())
