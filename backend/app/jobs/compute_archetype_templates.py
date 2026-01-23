"""
Archetype Template Computation Job
Schedule: Run after MTGTop8 scraper completes

Analyzes tournament decklists to compute average role distributions
for each archetype category (aggro, midrange, control, combo).
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from app.db.session import async_session_factory
from app.models.meta import ArchetypeTemplate, Decklist

logger = logging.getLogger(__name__)

# Archetype categorization rules
ARCHETYPE_CATEGORIES = {
    "aggro": ["aggro", "weenie", "red deck wins", "rdw", "burn", "sligh"],
    "control": ["control"],
    "midrange": ["midrange", "landfall", "rhythm", "ramp", "value"],
    "combo": ["combo", "reanimate", "reanimator", "storm"],
}


def categorize_archetype(archetype: str) -> str:
    """Categorize a specific archetype into a broad category."""
    if not archetype:
        return "other"

    arch_lower = archetype.lower()

    for category, keywords in ARCHETYPE_CATEGORIES.items():
        for keyword in keywords:
            if keyword in arch_lower:
                return category

    return "other"


async def get_card_roles(db: AsyncSession) -> Dict[str, List[str]]:
    """Get distinct card name -> roles mapping."""
    result = await db.execute(text("""
        SELECT DISTINCT c.name, cr.role
        FROM cards c
        JOIN card_roles cr ON c.id = cr.card_id
    """))

    card_roles = defaultdict(set)
    for name, role in result.all():
        card_roles[name].add(role)

    return {k: list(v) for k, v in card_roles.items()}


async def analyze_decklists(
    db: AsyncSession,
    format: str = "standard"
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all decklists and compute role distributions by archetype category.

    Returns:
        Dict mapping archetype_category -> {
            'count': int,
            'land_counts': list,
            'nonland_counts': list,
            'role_counts': {role: [counts per deck]},
            'archetype_breakdown': {specific_archetype: count}
        }
    """
    # Get all decklists
    result = await db.execute(
        select(Decklist.archetype, Decklist.main_deck)
        .where(Decklist.main_deck.isnot(None))
    )
    decklists = result.all()

    if not decklists:
        logger.warning("No decklists found to analyze")
        return {}

    # Get card roles
    card_roles = await get_card_roles(db)

    # Initialize stats per category
    category_stats = defaultdict(lambda: {
        'count': 0,
        'land_counts': [],
        'nonland_counts': [],
        'role_counts': defaultdict(list),
        'archetype_breakdown': defaultdict(int),
    })

    for archetype, main_deck in decklists:
        category = categorize_archetype(archetype)
        stats = category_stats[category]
        stats['count'] += 1
        stats['archetype_breakdown'][archetype or "Unknown"] += 1

        role_counts = defaultdict(int)
        land_count = 0
        nonland_count = 0

        for entry in main_deck:
            card_name = entry.get('card_name') or entry.get('name')
            qty = entry.get('quantity', 1)

            roles = card_roles.get(card_name, [])
            is_land = any(r.startswith('land_') for r in roles)

            if is_land:
                land_count += qty
            else:
                nonland_count += qty

            for role in roles:
                role_counts[role] += qty

        stats['land_counts'].append(land_count)
        stats['nonland_counts'].append(nonland_count)
        for role, count in role_counts.items():
            stats['role_counts'][role].append(count)

    return dict(category_stats)


async def save_archetype_templates(
    db: AsyncSession,
    category_stats: Dict[str, Dict[str, Any]],
    format: str = "standard"
) -> int:
    """
    Save computed archetype templates to database.

    Returns:
        Number of templates saved
    """
    saved = 0
    now = datetime.utcnow()

    for category, stats in category_stats.items():
        if stats['count'] == 0:
            continue

        # Compute averages
        avg_lands = sum(stats['land_counts']) / len(stats['land_counts'])
        avg_nonlands = sum(stats['nonland_counts']) / len(stats['nonland_counts'])

        # Compute role distribution (only roles with avg >= 0.5)
        role_distribution = {}
        for role, counts in stats['role_counts'].items():
            avg = sum(counts) / stats['count']
            if avg >= 0.5:
                role_distribution[role] = round(avg, 1)

        # Sort role distribution by count (descending)
        role_distribution = dict(
            sorted(role_distribution.items(), key=lambda x: -x[1])
        )

        # Upsert template
        stmt = insert(ArchetypeTemplate).values(
            archetype_category=category,
            format=format,
            sample_size=stats['count'],
            computed_at=now,
            avg_lands=Decimal(str(round(avg_lands, 1))),
            avg_nonlands=Decimal(str(round(avg_nonlands, 1))),
            role_distribution=role_distribution,
            archetype_breakdown=dict(stats['archetype_breakdown']),
        )
        stmt = stmt.on_conflict_do_update(
            constraint='uq_archetype_template',
            set_={
                'sample_size': stmt.excluded.sample_size,
                'computed_at': stmt.excluded.computed_at,
                'avg_lands': stmt.excluded.avg_lands,
                'avg_nonlands': stmt.excluded.avg_nonlands,
                'role_distribution': stmt.excluded.role_distribution,
                'archetype_breakdown': stmt.excluded.archetype_breakdown,
            }
        )
        await db.execute(stmt)
        saved += 1

        logger.info(
            f"Saved template for {category}: {stats['count']} decks, "
            f"avg {avg_lands:.1f} lands, {len(role_distribution)} roles"
        )

    await db.commit()
    return saved


async def compute_archetype_templates(format: str = "standard") -> Dict[str, Any]:
    """
    Main function to compute and save archetype templates.

    Returns:
        Dict with computation statistics
    """
    start_time = datetime.utcnow()
    stats = {
        "started_at": start_time.isoformat(),
        "format": format,
        "templates_saved": 0,
        "categories": {},
    }

    async with async_session_factory() as db:
        try:
            # Analyze decklists
            category_stats = await analyze_decklists(db, format)

            if not category_stats:
                stats["error"] = "No decklists to analyze"
                return stats

            # Save templates
            saved = await save_archetype_templates(db, category_stats, format)
            stats["templates_saved"] = saved

            # Add category summaries
            for category, data in category_stats.items():
                if data['count'] > 0:
                    avg_lands = sum(data['land_counts']) / len(data['land_counts'])
                    stats["categories"][category] = {
                        "deck_count": data['count'],
                        "avg_lands": round(avg_lands, 1),
                    }

        except Exception as e:
            logger.error(f"Archetype template computation failed: {e}")
            stats["error"] = str(e)
            raise

    stats["completed_at"] = datetime.utcnow().isoformat()
    logger.info(f"Archetype template computation complete: {stats}")
    return stats


async def get_archetype_template(
    db: AsyncSession,
    archetype_category: str,
    format: str = "standard"
) -> Dict[str, Any] | None:
    """
    Get the archetype template for a given category.

    Returns:
        Dict with template data or None if not found
    """
    result = await db.execute(
        select(ArchetypeTemplate)
        .where(ArchetypeTemplate.archetype_category == archetype_category)
        .where(ArchetypeTemplate.format == format)
    )
    template = result.scalar_one_or_none()

    if not template:
        return None

    return {
        "archetype_category": template.archetype_category,
        "format": template.format,
        "sample_size": template.sample_size,
        "avg_lands": float(template.avg_lands),
        "avg_nonlands": float(template.avg_nonlands),
        "role_distribution": template.role_distribution,
        "archetype_breakdown": template.archetype_breakdown,
    }


async def get_all_archetype_templates(
    db: AsyncSession,
    format: str = "standard"
) -> List[Dict[str, Any]]:
    """Get all archetype templates for a format."""
    result = await db.execute(
        select(ArchetypeTemplate)
        .where(ArchetypeTemplate.format == format)
        .order_by(ArchetypeTemplate.sample_size.desc())
    )
    templates = result.scalars().all()

    return [
        {
            "archetype_category": t.archetype_category,
            "format": t.format,
            "sample_size": t.sample_size,
            "avg_lands": float(t.avg_lands),
            "avg_nonlands": float(t.avg_nonlands),
            "role_distribution": t.role_distribution,
        }
        for t in templates
    ]


if __name__ == "__main__":
    # Allow running directly for testing
    logging.basicConfig(level=logging.INFO)
    asyncio.run(compute_archetype_templates())
