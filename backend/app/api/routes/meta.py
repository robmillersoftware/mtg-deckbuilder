from typing import Optional, List
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.db.session import get_db
from app.models.meta import MetaSnapshot, CardCooccurrence
from app.schemas.meta import (
    MetaSnapshotResponse,
    MetaDashboardResponse,
    ArchetypeEntry,
    MatchupAnalysis,
    CooccurrenceResult,
)

router = APIRouter()


@router.get("", response_model=MetaDashboardResponse)
async def get_meta_dashboard(
    format: str = "standard",
    db: AsyncSession = Depends(get_db),
):
    """
    Get the current meta dashboard showing archetype percentages and statistics.
    Data reflects the latest meta_snapshot (no older than 7 days).
    """
    # Get the most recent snapshot date for this format
    result = await db.execute(
        select(MetaSnapshot.snapshot_date)
        .where(MetaSnapshot.format == format)
        .order_by(desc(MetaSnapshot.snapshot_date))
        .limit(1)
    )
    latest_date = result.scalar_one_or_none()

    if latest_date is None:
        return MetaDashboardResponse(
            format=format,
            last_updated=date.today(),
            archetypes=[],
        )

    # Get all archetypes for the latest snapshot
    result = await db.execute(
        select(MetaSnapshot)
        .where(
            MetaSnapshot.format == format,
            MetaSnapshot.snapshot_date == latest_date,
        )
        .order_by(desc(MetaSnapshot.meta_percentage))
    )
    snapshots = result.scalars().all()

    archetypes = [
        ArchetypeEntry(
            name=s.archetype,
            meta_percentage=float(s.meta_percentage) if s.meta_percentage else 0.0,
            sample_size=s.sample_size or 0,
            avg_finish=float(s.avg_finish) if s.avg_finish else 0.0,
            key_cards=s.key_cards or [],
        )
        for s in snapshots
    ]

    return MetaDashboardResponse(
        format=format,
        last_updated=latest_date,
        archetypes=archetypes,
    )


@router.get("/archetypes/{archetype}", response_model=MetaSnapshotResponse)
async def get_archetype_details(
    archetype: str,
    format: str = "standard",
    db: AsyncSession = Depends(get_db),
):
    """Get detailed information about a specific archetype."""
    result = await db.execute(
        select(MetaSnapshot)
        .where(
            MetaSnapshot.format == format,
            MetaSnapshot.archetype == archetype,
        )
        .order_by(desc(MetaSnapshot.snapshot_date))
        .limit(1)
    )
    snapshot = result.scalar_one_or_none()

    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Archetype not found",
        )

    return snapshot


@router.get("/cooccurrence/{card_name}", response_model=List[CooccurrenceResult])
async def get_card_cooccurrence(
    card_name: str,
    format: str = "standard",
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Get cards that frequently appear together with the specified card.
    Useful for synergy recommendations.
    """
    result = await db.execute(
        select(CardCooccurrence)
        .where(
            CardCooccurrence.format == format,
            CardCooccurrence.card_a == card_name,
        )
        .order_by(desc(CardCooccurrence.cooccurrence_count))
        .limit(limit)
    )
    cooccurrences = result.scalars().all()

    return [
        CooccurrenceResult(
            card_a=c.card_a,
            card_b=c.card_b,
            count=c.cooccurrence_count,
        )
        for c in cooccurrences
    ]


@router.get("/history", response_model=List[MetaSnapshotResponse])
async def get_meta_history(
    archetype: str,
    format: str = "standard",
    limit: int = Query(10, ge=1, le=52),
    db: AsyncSession = Depends(get_db),
):
    """Get historical meta data for a specific archetype."""
    result = await db.execute(
        select(MetaSnapshot)
        .where(
            MetaSnapshot.format == format,
            MetaSnapshot.archetype == archetype,
        )
        .order_by(desc(MetaSnapshot.snapshot_date))
        .limit(limit)
    )
    snapshots = result.scalars().all()

    return snapshots
