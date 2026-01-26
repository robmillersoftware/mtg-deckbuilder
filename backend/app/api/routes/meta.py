from typing import Optional, List
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, distinct

from app.db.session import get_db
from app.models.meta import MetaSnapshot, CardCooccurrence
from app.schemas.meta import (
    MetaSnapshotResponse,
    MetaDashboardResponse,
    ArchetypeEntry,
    MatchupAnalysis,
    CooccurrenceResult,
    ArchetypeTrend,
    MetaTrendsResponse,
    MetaHealthResponse,
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


@router.get("/trends", response_model=MetaTrendsResponse)
async def get_meta_trends(
    format: str = "standard",
    days_back: int = Query(7, ge=1, le=30, description="Days to look back for comparison"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get emerging and declining archetypes by comparing current meta to historical data.

    Returns archetypes that have risen or fallen significantly in meta share,
    as well as archetypes that are new or have disappeared.
    """
    # Get the two most recent distinct snapshot dates
    result = await db.execute(
        select(distinct(MetaSnapshot.snapshot_date))
        .where(MetaSnapshot.format == format)
        .order_by(desc(MetaSnapshot.snapshot_date))
        .limit(10)  # Get a few dates to find one that's far enough back
    )
    dates = [row[0] for row in result.fetchall()]

    if len(dates) < 2:
        # Not enough historical data
        return MetaTrendsResponse(
            format=format,
            current_date=dates[0] if dates else date.today(),
            comparison_date=dates[0] if dates else date.today(),
            rising=[],
            falling=[],
            new_archetypes=[],
            disappeared=[],
        )

    current_date = dates[0]

    # Find a comparison date approximately days_back ago
    target_date = current_date - timedelta(days=days_back)
    comparison_date = min(dates[1:], key=lambda d: abs((d - target_date).days))

    # Get current snapshot
    result = await db.execute(
        select(MetaSnapshot)
        .where(
            MetaSnapshot.format == format,
            MetaSnapshot.snapshot_date == current_date,
        )
    )
    current_snapshots = {s.archetype: s for s in result.scalars().all()}

    # Get comparison snapshot
    result = await db.execute(
        select(MetaSnapshot)
        .where(
            MetaSnapshot.format == format,
            MetaSnapshot.snapshot_date == comparison_date,
        )
    )
    old_snapshots = {s.archetype: s for s in result.scalars().all()}

    # Calculate trends
    rising = []
    falling = []
    new_archetypes = []

    for archetype, current in current_snapshots.items():
        current_pct = float(current.meta_percentage) if current.meta_percentage else 0.0

        if archetype in old_snapshots:
            old_pct = float(old_snapshots[archetype].meta_percentage) if old_snapshots[archetype].meta_percentage else 0.0
            change = current_pct - old_pct

            # Calculate relative change (avoid division by zero)
            if old_pct > 0:
                change_percent = (change / old_pct) * 100
            else:
                change_percent = 100.0 if current_pct > 0 else 0.0

            # Only include significant changes (> 0.5 percentage points)
            if abs(change) >= 0.5:
                trend = ArchetypeTrend(
                    name=archetype,
                    current_percentage=current_pct,
                    previous_percentage=old_pct,
                    change=change,
                    change_percent=change_percent,
                    sample_size=current.sample_size or 0,
                    key_cards=current.key_cards or [],
                )
                if change > 0:
                    rising.append(trend)
                else:
                    falling.append(trend)
        else:
            # New archetype (wasn't in the old snapshot)
            if current_pct >= 1.0:  # Only include archetypes with at least 1% share
                new_archetypes.append(ArchetypeEntry(
                    name=archetype,
                    meta_percentage=current_pct,
                    sample_size=current.sample_size or 0,
                    avg_finish=float(current.avg_finish) if current.avg_finish else 0.0,
                    key_cards=current.key_cards or [],
                ))

    # Find disappeared archetypes (were in old but not in current)
    disappeared = [
        arch for arch in old_snapshots.keys()
        if arch not in current_snapshots and (float(old_snapshots[arch].meta_percentage) if old_snapshots[arch].meta_percentage else 0.0) >= 1.0
    ]

    # Sort by absolute change
    rising.sort(key=lambda x: x.change, reverse=True)
    falling.sort(key=lambda x: x.change)

    return MetaTrendsResponse(
        format=format,
        current_date=current_date,
        comparison_date=comparison_date,
        rising=rising[:5],  # Top 5 rising
        falling=falling[:5],  # Top 5 falling
        new_archetypes=new_archetypes,
        disappeared=disappeared,
    )


@router.get("/health", response_model=MetaHealthResponse)
async def get_meta_health(
    format: str = "standard",
    db: AsyncSession = Depends(get_db),
):
    """
    Get meta health metrics showing format diversity and concentration.

    Uses a diversity score based on the Herfindahl-Hirschman Index (HHI),
    where higher scores indicate a healthier, more diverse meta.
    """
    # Get the most recent snapshot date
    result = await db.execute(
        select(MetaSnapshot.snapshot_date)
        .where(MetaSnapshot.format == format)
        .order_by(desc(MetaSnapshot.snapshot_date))
        .limit(1)
    )
    latest_date = result.scalar_one_or_none()

    if latest_date is None:
        return MetaHealthResponse(
            format=format,
            snapshot_date=date.today(),
            diversity_score=0.0,
            top_deck_share=0.0,
            top_3_share=0.0,
            total_archetypes=0,
            health_rating="Unknown",
            assessment="No meta data available for this format.",
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

    if not snapshots:
        return MetaHealthResponse(
            format=format,
            snapshot_date=latest_date,
            diversity_score=0.0,
            top_deck_share=0.0,
            top_3_share=0.0,
            total_archetypes=0,
            health_rating="Unknown",
            assessment="No archetype data available.",
        )

    # Extract percentages
    percentages = [float(s.meta_percentage) if s.meta_percentage else 0.0 for s in snapshots]
    total_archetypes = len(percentages)

    # Calculate HHI (sum of squared market shares)
    # HHI ranges from 0 (perfect competition) to 10000 (monopoly)
    hhi = sum((p ** 2) for p in percentages)

    # Convert HHI to a 0-100 diversity score (inverted and scaled)
    # Perfect diversity (many equal decks) -> score of 100
    # Single dominant deck (100% share) -> score of 0
    # We normalize by the maximum possible HHI (10000) and minimum for N decks (10000/N)
    if total_archetypes > 1:
        min_hhi = 10000 / total_archetypes  # Best possible HHI for N decks
        max_hhi = 10000  # Worst case (monopoly)
        # Normalize: 100 when HHI = min_hhi, 0 when HHI = max_hhi
        diversity_score = max(0, min(100, 100 * (max_hhi - hhi) / (max_hhi - min_hhi)))
    else:
        diversity_score = 0.0

    # Get top deck shares
    top_deck_share = percentages[0] if percentages else 0.0
    top_3_share = sum(percentages[:3]) if len(percentages) >= 3 else sum(percentages)

    # Determine health rating
    if diversity_score >= 70 and top_deck_share <= 15:
        health_rating = "Healthy"
        assessment = f"The meta is well-balanced with {total_archetypes} viable archetypes. No single deck dominates."
    elif diversity_score >= 50 and top_deck_share <= 25:
        health_rating = "Moderate"
        assessment = f"The meta is reasonably diverse with {total_archetypes} archetypes, though a few decks are more prominent."
    elif diversity_score >= 30 and top_deck_share <= 35:
        health_rating = "Concentrated"
        assessment = f"The meta shows some concentration with the top deck at {top_deck_share:.1f}% share. Consider diversity options."
    else:
        health_rating = "Unhealthy"
        assessment = f"The meta is heavily concentrated. Top deck holds {top_deck_share:.1f}% of the meta. Format may need intervention."

    return MetaHealthResponse(
        format=format,
        snapshot_date=latest_date,
        diversity_score=round(diversity_score, 1),
        top_deck_share=round(top_deck_share, 1),
        top_3_share=round(top_3_share, 1),
        total_archetypes=total_archetypes,
        health_rating=health_rating,
        assessment=assessment,
    )
