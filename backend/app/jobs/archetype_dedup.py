"""
Archetype deduplication by card similarity.

Merges archetype names that represent functionally the same deck
(e.g. "Simic Nature's Rhythm", "Simic Aggro", "Simic Cub") by comparing
their core card signatures using Jaccard similarity.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.card import Card
from app.models.meta import Decklist, Event

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
) -> Dict[str, Set[str]]:
    """Build core card signatures per archetype.

    For each archetype, finds cards appearing in >= core_threshold fraction
    of its decklists (by presence, not quantity). Lands are excluded.
    Archetypes with fewer than min_decklists are skipped.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    # Single query: all decklists for this format/date range
    result = await db.execute(
        select(Decklist.archetype, Decklist.main_deck)
        .join(Event, Decklist.event_id == Event.id)
        .where(Event.format == format)
        .where(Event.date >= cutoff.date())
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
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """Main entry point: merge archetypes with similar card signatures.

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
