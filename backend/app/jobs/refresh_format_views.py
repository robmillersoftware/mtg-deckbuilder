"""
Refresh format-specific materialized views.

Called after Scryfall sync and mtgtop8 scrape to keep the per-format
card views in sync with the main cards table.

Using CONCURRENTLY so reads are not blocked during refresh.
"""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Must match the views created in migration 014
FORMAT_VIEWS = [
    "cards_standard",
    "cards_modern",
    "cards_legacy",
    "cards_historic",
    "cards_commander",
]


async def refresh_format_views(db: AsyncSession) -> int:
    """
    Refresh all format materialized views.

    Uses CONCURRENTLY where possible so existing queries aren't blocked.
    Returns the number of views refreshed.
    """
    refreshed = 0
    for view in FORMAT_VIEWS:
        try:
            # CONCURRENTLY requires the unique index we added in the migration.
            # Need to run outside a transaction for CONCURRENTLY.
            # If that fails, fall back to a normal (blocking) refresh.
            await db.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}"))
            await db.commit()
            refreshed += 1
            logger.info(f"Refreshed materialized view: {view}")
        except Exception as e:
            await db.rollback()
            logger.warning(
                f"CONCURRENTLY refresh failed for {view} ({e}), "
                "trying blocking refresh"
            )
            try:
                await db.execute(text(f"REFRESH MATERIALIZED VIEW {view}"))
                await db.commit()
                refreshed += 1
                logger.info(f"Refreshed materialized view (blocking): {view}")
            except Exception as e2:
                await db.rollback()
                logger.error(f"Failed to refresh {view}: {e2}")

    return refreshed
