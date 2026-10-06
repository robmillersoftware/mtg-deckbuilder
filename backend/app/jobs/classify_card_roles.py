"""
Card Role Classification Job
Schedule: Run manually or after Scryfall sync

Uses Jev (TypeSafe System One) to tag Standard-legal cards with functional
deck-building roles (removal, threats, ramp, etc.): one request per card,
with a Noul per role and an efficiency Score per role.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct
from sqlalchemy.dialects.postgresql import insert
from typesafe_sdk import Noul, Score

from app.db.session import async_session_factory
from app.models.card import Card, CardRole, CARD_ROLES, ROLE_DEFINITIONS
from app.services import jev

logger = logging.getLogger(__name__)

# Cards per batch; each card is one Jev request, all sent concurrently through the shared cap
BATCH_SIZE = 50

ROLE_THRESHOLD = 0.5
# ponytail: 44 questions per card can outlast the 2 s chat timeout; offline job, so allow longer
TAG_REQUEST_TIMEOUT = 15.0

EFFICIENCY_LEVELS = [
    "Pays far more mana or conditions than usual for this effect",
    "Below rate; the effect is narrow or overcosted",
    "Fair rate; a typical cost for this effect",
    "Strong rate; cheap or flexible for what it does",
    "Exceptional rate; the effect is far cheaper or stronger than typical",
]


def _card_state(card: Card) -> Dict[str, Any]:
    state = {
        "name": card.name,
        "mana_cost": card.mana_cost or "",
        "mana_value": card.cmc,
        "type_line": card.type_line or "",
        "oracle_text": card.oracle_text or "",
    }
    if card.power and card.toughness:
        state["power_toughness"] = f"{card.power}/{card.toughness}"
    return {"card": state}


def _role_questions() -> Dict[str, Any]:
    """One Noul per role plus a speculative efficiency Score per role, all in one request."""
    questions: Dict[str, Any] = {}
    for role, definition in ROLE_DEFINITIONS.items():
        questions[f"is:{role}"] = Noul(instructions={
            "question": "Does `card` fill this deck-building role, judged from its oracle text and stats?",
            "role": definition,
        })
        questions[f"eff:{role}"] = Score(
            instructions={
                "question": "Assuming `card` fills this role, how efficient is it at it relative to its mana cost?",
                "role": definition,
            },
            criteria=EFFICIENCY_LEVELS,
        )
    return questions


def _roles_from(resp: Any) -> List[Dict[str, Any]]:
    roles = []
    for role in ROLE_DEFINITIONS:
        p = resp.nouls[f"is:{role}"].noul
        if p >= ROLE_THRESHOLD:
            roles.append({
                "role": role,
                "efficiency": round(resp.scores[f"eff:{role}"].score) + 1,
                "confidence": round(p, 2),
            })
    return roles


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


async def classify_cards_batch(cards: List[Card], client=None) -> List[Dict[str, Any]]:
    """Tag a batch with Jev: one request per card, concurrent through the shared cap.

    Returns [] (the batch is skipped and its cards stay unclassified for the next
    run) when Jev is not configured, any request fails, or an answer is incomplete.
    """
    if not cards:
        return []
    questions = _role_questions()
    async with jev.session(client) as client:
        if client is None:
            logger.error("TYPESAFE_API_KEY not configured")
            return []
        try:
            responses = await jev.ask_many(
                client, [(_card_state(c), questions) for c in cards], timeout=TAG_REQUEST_TIMEOUT)
            return [{"name": c.name, "roles": _roles_from(r)} for c, r in zip(cards, responses)]
        except Exception as e:
            logger.error(f"Role tagging failed, skipping batch of {len(cards)} cards: {e}")
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
            confidence = role_data.get("confidence")
            reasoning = role_data.get("reasoning")

            # Save role for ALL printings of this card
            for card_id in card_ids:
                stmt = insert(CardRole).values(
                    card_id=card_id,
                    role=role,
                    efficiency=efficiency,
                    confidence=confidence,
                    reasoning=reasoning,
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_card_role",
                    set_={
                        "efficiency": stmt.excluded.efficiency,
                        "confidence": stmt.excluded.confidence,
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

            # Snapshot once and make a single pass: cards that fit no role are
            # never saved, so re-querying "unclassified" each batch would loop forever
            pending = await get_unclassified_cards(db, limit=total_unclassified)
            for i in range(0, len(pending), BATCH_SIZE):
                cards = pending[i:i + BATCH_SIZE]

                stats["batches"] += 1
                logger.info(f"Batch {stats['batches']}: Classifying {len(cards)} cards")

                classifications = await classify_cards_batch(cards)

                if classifications:
                    saved = await save_card_roles(db, cards, classifications)
                    stats["roles_saved"] += saved
                    logger.info(f"Batch {stats['batches']}: Saved {saved} roles")
                else:
                    logger.warning(f"Batch {stats['batches']}: No classifications returned")

                stats["cards_processed"] += len(cards)

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
