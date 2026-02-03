"""
Archetype deduplication by card similarity.

Merges archetype names that represent functionally the same deck
(e.g. "Simic Nature's Rhythm", "Simic Aggro", "Simic Cub") by comparing
their core card signatures using Jaccard similarity.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import delete, distinct, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.card import Card
from app.models.meta import Decklist, Event, MetaSnapshot

logger = logging.getLogger(__name__)


class UnionFind:
    """Simple Union-Find with path compression for transitive merging."""

    def __init__(self, items: List[str]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


async def _get_land_names(db: AsyncSession) -> Set[str]:
    """Get all land card names from the Card table."""
    result = await db.execute(
        select(Card.name).where(Card.type_line.ilike("%Land%"))
    )
    return set(result.scalars().all())


async def _build_archetype_signatures(
    db: AsyncSession,
    archetypes: Dict[str, Dict[str, Any]],
    land_names: Set[str],
    format: str,
    days: int,
    min_decklists: int = 3,
    core_threshold: float = 0.5,
    reference_date: Optional[date] = None,
) -> Dict[str, Set[str]]:
    """Build core card signatures per archetype.

    For each archetype, finds cards appearing in >= core_threshold fraction
    of its decklists (by presence, not quantity). Lands are excluded.
    Archetypes with fewer than min_decklists are skipped.

    reference_date: if provided, the date window is computed relative to this
    date instead of today. Used for backfilling historical snapshots.
    """
    ref = reference_date or datetime.utcnow().date()
    cutoff = ref - timedelta(days=days)

    # Single query: all decklists for this format/date range
    result = await db.execute(
        select(Decklist.archetype, Decklist.main_deck)
        .join(Event, Decklist.event_id == Event.id)
        .where(Event.format == format)
        .where(Event.date >= cutoff)
    )
    rows = result.all()

    # Group decklists by archetype
    decks_by_archetype: Dict[str, List[List[dict]]] = defaultdict(list)
    for archetype_name, main_deck in rows:
        name = archetype_name or "Unknown"
        if name in archetypes:
            decks_by_archetype[name].append(main_deck or [])

    signatures: Dict[str, Set[str]] = {}
    for name, decks in decks_by_archetype.items():
        if len(decks) < min_decklists:
            continue

        # Count which non-land cards appear in each decklist (presence only)
        card_presence: Dict[str, int] = defaultdict(int)
        for deck in decks:
            seen = set()
            for entry in deck:
                card_name = entry.get("card_name")
                if card_name and card_name not in land_names and card_name not in seen:
                    seen.add(card_name)
                    card_presence[card_name] += 1

        # Core cards = those in >= threshold fraction of decklists
        threshold_count = len(decks) * core_threshold
        core = {card for card, count in card_presence.items() if count >= threshold_count}
        if core:
            signatures[name] = core

    return signatures


def _compute_merge_groups(
    signatures: Dict[str, Set[str]],
    archetypes: Dict[str, Dict[str, Any]],
    threshold: float = 0.7,
) -> Dict[str, List[str]]:
    """Compute merge groups using pairwise Jaccard similarity + Union-Find.

    Returns {primary_name: [all_names_in_group]} where primary is the
    archetype with the highest decklist count.
    """
    names = list(signatures.keys())
    if len(names) < 2:
        return {}

    uf = UnionFind(names)

    # Pairwise Jaccard
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            intersection = len(signatures[a] & signatures[b])
            union = len(signatures[a] | signatures[b])
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard >= threshold:
                logger.info(
                    f"Merging '{a}' ({archetypes[a]['count']} decks) and "
                    f"'{b}' ({archetypes[b]['count']} decks), Jaccard={jaccard:.2f}"
                )
                uf.union(a, b)

    # Group by root
    groups: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        root = uf.find(name)
        groups[root].append(name)

    # Only return groups with actual merges (>1 member)
    merge_groups: Dict[str, List[str]] = {}
    for members in groups.values():
        if len(members) > 1:
            # Primary = highest count
            primary = max(members, key=lambda n: archetypes[n]["count"])
            merge_groups[primary] = members
            if len(members) > 5:
                logger.warning(
                    f"Large merge group ({len(members)} archetypes) under '{primary}' "
                    f"- possible over-merging: {members}"
                )

    return merge_groups


async def merge_similar_archetypes(
    db: AsyncSession,
    archetypes: Dict[str, Dict[str, Any]],
    format: str,
    days: int,
    threshold: float = 0.7,
    min_decklists: int = 3,
    core_threshold: float = 0.5,
    reference_date: Optional[date] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """Main entry point: merge archetypes with similar card signatures.

    Args:
        reference_date: if provided, the date window is computed relative to
            this date instead of today. Used for backfilling historical snapshots.

    Returns:
        (merged_archetypes, aliases) where:
        - merged_archetypes: updated dict with merged counts/percentages
        - aliases: {primary_name: [all_names_in_group]} for groups that were merged
    """
    if len(archetypes) < 2:
        return archetypes, {}

    land_names = await _get_land_names(db)
    signatures = await _build_archetype_signatures(
        db, archetypes, land_names, format, days,
        min_decklists=min_decklists, core_threshold=core_threshold,
        reference_date=reference_date,
    )

    merge_groups = _compute_merge_groups(signatures, archetypes, threshold)
    if not merge_groups:
        logger.info(f"No archetype merges needed for {format}")
        return archetypes, {}

    # Build merged result
    merged = dict(archetypes)
    aliases: Dict[str, List[str]] = {}
    total_count = sum(d["count"] for d in archetypes.values())

    for primary, members in merge_groups.items():
        combined_count = sum(archetypes[name]["count"] for name in members)
        merged[primary] = {
            "count": combined_count,
            "percentage": (combined_count / total_count) * 100 if total_count > 0 else 0,
        }
        aliases[primary] = members

        # Remove non-primary members
        for name in members:
            if name != primary:
                merged.pop(name, None)

    original_count = len(archetypes)
    merged_count = len(merged)
    logger.info(f"Dedup reduced {original_count} archetypes to {merged_count} for {format}")

    return merged, aliases


FORMAT_DAYS = {
    "standard": 14,
    "cedh": 30,
    "duel_commander": 30,
    "modern": 14,
    "pioneer": 14,
    "legacy": 30,
    "vintage": 30,
    "pauper": 14,
}


async def backfill_dedup_snapshots(
    formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Re-run archetype dedup on all historical MetaSnapshot rows.

    For each (format, snapshot_date) pair, reconstructs archetypes from the
    stored snapshot rows, runs dedup using the correct historical date window,
    and rewrites the snapshot rows with merged data.

    Returns stats dict with counts of dates processed and snapshots rewritten.
    """
    from app.db.session import async_session_factory

    stats: Dict[str, Any] = {"dates_processed": 0, "snapshots_before": 0, "snapshots_after": 0}

    async with async_session_factory() as db:
        # Get all distinct (format, snapshot_date) pairs
        query = select(
            MetaSnapshot.format,
            MetaSnapshot.snapshot_date,
        ).distinct()
        if formats:
            query = query.where(MetaSnapshot.format.in_(formats))
        query = query.order_by(MetaSnapshot.format, MetaSnapshot.snapshot_date)

        result = await db.execute(query)
        date_pairs = result.all()

        if not date_pairs:
            logger.info("No historical snapshots found to backfill")
            return stats

        logger.info(f"Backfilling dedup for {len(date_pairs)} (format, date) pairs")
        land_names = await _get_land_names(db)

        for fmt, snap_date in date_pairs:
            # Reconstruct archetypes dict from existing snapshot rows
            result = await db.execute(
                select(MetaSnapshot)
                .where(MetaSnapshot.format == fmt)
                .where(MetaSnapshot.snapshot_date == snap_date)
            )
            snapshots = result.scalars().all()
            if not snapshots:
                continue

            archetypes = {}
            for s in snapshots:
                archetypes[s.archetype] = {
                    "count": s.sample_size or 0,
                    "percentage": float(s.meta_percentage or 0),
                }

            stats["snapshots_before"] += len(archetypes)

            if len(archetypes) < 2:
                stats["snapshots_after"] += len(archetypes)
                stats["dates_processed"] += 1
                continue

            days = FORMAT_DAYS.get(fmt, 14)

            # Build signatures using the historical date window
            signatures = await _build_archetype_signatures(
                db, archetypes, land_names, fmt, days,
                reference_date=snap_date,
            )
            merge_groups = _compute_merge_groups(signatures, archetypes)

            if not merge_groups:
                stats["snapshots_after"] += len(archetypes)
                stats["dates_processed"] += 1
                continue

            # Apply merges
            total_count = sum(d["count"] for d in archetypes.values())
            merged = dict(archetypes)
            aliases: Dict[str, List[str]] = {}

            for primary, members in merge_groups.items():
                combined_count = sum(archetypes[name]["count"] for name in members)
                merged[primary] = {
                    "count": combined_count,
                    "percentage": (combined_count / total_count) * 100 if total_count > 0 else 0,
                }
                aliases[primary] = members
                for name in members:
                    if name != primary:
                        merged.pop(name, None)

            # Delete old snapshots for this date/format and write new ones
            await db.execute(
                delete(MetaSnapshot)
                .where(MetaSnapshot.format == fmt)
                .where(MetaSnapshot.snapshot_date == snap_date)
            )

            for archetype, data in merged.items():
                # Compute key cards from all aliased names
                names_to_query = aliases.get(archetype, [archetype])
                cutoff = snap_date - timedelta(days=days)

                kc_result = await db.execute(
                    select(Decklist.main_deck)
                    .join(Event, Decklist.event_id == Event.id)
                    .where(Decklist.archetype.in_(names_to_query))
                    .where(Event.format == fmt)
                    .where(Event.date >= cutoff)
                    .limit(10)
                )
                main_decks = kc_result.scalars().all()

                card_counts: Dict[str, int] = defaultdict(int)
                for main_deck in main_decks:
                    for entry in main_deck or []:
                        card_name = entry.get("card_name")
                        if card_name:
                            card_counts[card_name] += entry.get("quantity", 1)

                key_cards = [c for c, _ in sorted(card_counts.items(), key=lambda x: -x[1])[:10]]

                snapshot = MetaSnapshot(
                    format=fmt,
                    archetype=archetype,
                    meta_percentage=data["percentage"],
                    sample_size=data["count"],
                    key_cards=key_cards,
                    snapshot_date=snap_date,
                )
                db.add(snapshot)

            await db.commit()
            stats["snapshots_after"] += len(merged)
            stats["dates_processed"] += 1

            logger.info(
                f"Backfill {fmt} {snap_date}: {len(archetypes)} -> {len(merged)} archetypes"
            )

    logger.info(
        f"Backfill complete: {stats['dates_processed']} dates, "
        f"{stats['snapshots_before']} -> {stats['snapshots_after']} total snapshots"
    )
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import sys
    formats = sys.argv[1:] or None
    result = asyncio.run(backfill_dedup_snapshots(formats=formats))
    print(f"Done: {result}")
