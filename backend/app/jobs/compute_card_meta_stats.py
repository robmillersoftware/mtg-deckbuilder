"""
Card Meta Stats Computation Job
Schedule: Run after MTGTop8 scraper completes

Analyzes tournament decklists to compute per-card metagame representation:
how frequently each card appears, average copies, main vs sideboard split,
and which archetypes use it.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func

from app.db.session import async_session_factory
from app.models.meta import Event, Decklist, CardMetaStats
from app.models.card import Card

logger = logging.getLogger(__name__)


async def calculate_card_meta_stats(
    db: AsyncSession,
    format: str = "standard",
    days: int = 14,
) -> List[Dict[str, Any]]:
    """
    Calculate per-card meta representation from recent tournament decklists.

    For each card, computes:
    - How many decks include it (main, sideboard, or either)
    - Average number of copies when present
    - Which archetypes run it
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    result = await db.execute(
        select(Decklist.main_deck, Decklist.sideboard, Decklist.archetype)
        .join(Event, Decklist.event_id == Event.id)
        .where(Event.format == format)
        .where(Event.date >= cutoff.date())
    )
    rows = result.all()

    total_decks = len(rows)
    if total_decks == 0:
        return []

    # Per-card accumulators
    card_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "main_deck_count": 0,
        "sideboard_count": 0,
        "total_copies": 0,  # sum of copies across all decks that include it
        "deck_set": set(),   # set of deck indices that include it (for dedup)
        "archetypes": defaultdict(int),
    })

    for idx, (main_deck, sideboard, archetype) in enumerate(rows):
        arch_name = archetype or "Unknown"

        for entry in (main_deck or []):
            card_name = entry.get("card_name")
            qty = entry.get("quantity", 1)
            if not card_name:
                continue
            data = card_data[card_name]
            data["main_deck_count"] += 1
            data["total_copies"] += qty
            data["deck_set"].add(idx)
            data["archetypes"][arch_name] += 1

        for entry in (sideboard or []):
            card_name = entry.get("card_name")
            qty = entry.get("quantity", 1)
            if not card_name:
                continue
            data = card_data[card_name]
            data["sideboard_count"] += 1
            data["total_copies"] += qty
            data["deck_set"].add(idx)
            data["archetypes"][arch_name] += 1

    # Build result list
    stats = []
    for card_name, data in card_data.items():
        deck_count = len(data["deck_set"])
        meta_pct = (deck_count / total_decks) * 100
        avg_copies = data["total_copies"] / deck_count if deck_count else 0

        # Top archetypes (sorted by count descending, limit to 5)
        sorted_archs = sorted(data["archetypes"].items(), key=lambda x: -x[1])[:5]
        archetypes_list = [
            {"name": name, "count": count, "percentage": round((count / deck_count) * 100, 1)}
            for name, count in sorted_archs
        ]

        stats.append({
            "card_name": card_name,
            "deck_count": deck_count,
            "total_decks": total_decks,
            "meta_percentage": round(meta_pct, 2),
            "main_deck_count": data["main_deck_count"],
            "sideboard_count": data["sideboard_count"],
            "avg_copies": round(avg_copies, 1),
            "archetypes": archetypes_list,
        })

    return stats


async def save_card_meta_stats(
    db: AsyncSession,
    stats: List[Dict[str, Any]],
    format: str = "standard",
) -> int:
    """Persist computed card meta stats to the database."""
    today = datetime.utcnow().date()

    # Delete today's snapshot so re-runs are idempotent
    await db.execute(
        delete(CardMetaStats)
        .where(CardMetaStats.format == format)
        .where(CardMetaStats.snapshot_date == today)
    )

    # Resolve card IDs in bulk
    card_names = [s["card_name"] for s in stats]
    result = await db.execute(
        select(Card.id, Card.name).where(Card.name.in_(card_names))
    )
    card_id_map = {row.name: row.id for row in result.all()}

    for s in stats:
        record = CardMetaStats(
            card_name=s["card_name"],
            card_id=card_id_map.get(s["card_name"]),
            format=format,
            snapshot_date=today,
            deck_count=s["deck_count"],
            total_decks=s["total_decks"],
            meta_percentage=Decimal(str(s["meta_percentage"])),
            main_deck_count=s["main_deck_count"],
            sideboard_count=s["sideboard_count"],
            avg_copies=Decimal(str(s["avg_copies"])),
            archetypes=s["archetypes"],
        )
        db.add(record)

    await db.commit()
    return len(stats)


async def compute_card_meta_stats(
    formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main entry point. Computes and saves card meta stats for the given formats.

    Returns:
        Dict with computation statistics.
    """
    from app.jobs.mtgtop8_scrape import FORMAT_CONFIG

    if formats is None:
        formats = list(FORMAT_CONFIG.keys())

    start_time = datetime.utcnow()
    result: Dict[str, Any] = {
        "started_at": start_time.isoformat(),
        "formats": {},
    }

    async with async_session_factory() as db:
        for format_name in formats:
            days = FORMAT_CONFIG.get(format_name, {}).get("days", 14)
            try:
                stats = await calculate_card_meta_stats(db, format=format_name, days=days)
                saved = await save_card_meta_stats(db, stats, format=format_name)
                result["formats"][format_name] = {"cards_tracked": saved}
                logger.info(f"Saved card meta stats for {format_name}: {saved} cards")
            except Exception as e:
                logger.error(f"Card meta stats failed for {format_name}: {e}")
                result["formats"][format_name] = {"error": str(e)}
                await db.rollback()

    result["completed_at"] = datetime.utcnow().isoformat()
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(compute_card_meta_stats())
