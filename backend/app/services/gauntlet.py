"""The opponents a deck is tested against: the top meta archetypes (or chosen ones),
each as its most recent best-placing list from the recent window."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meta import MetaSnapshot
from app.services.deck_plan import RECENT

GAUNTLET_SIZE = 5

LIST_SQL = text(f"""
    SELECT d.main_deck
    FROM decklists d JOIN events e ON e.id = d.event_id
    WHERE {RECENT} AND lower(trim(d.archetype)) = lower(trim(:archetype))
    ORDER BY e.date DESC, d.placement NULLS LAST
    LIMIT 1
""")


@dataclass
class Opponent:
    archetype: str
    share: float  # weight in the overall win rate
    main: Dict[str, int]


async def gauntlet(db: AsyncSession, format: str, archetypes: Optional[Sequence[str]] = None) -> List[Opponent]:
    """With `archetypes`, those (equal weight); otherwise the top GAUNTLET_SIZE by meta
    share (weighted by share). Archetypes without a recent list are skipped."""
    snapshots = (await db.execute(
        select(MetaSnapshot).where(
            MetaSnapshot.format == format,
            MetaSnapshot.snapshot_date == select(func.max(MetaSnapshot.snapshot_date))
            .where(MetaSnapshot.format == format).scalar_subquery())
        .order_by(MetaSnapshot.meta_percentage.desc()))).scalars().all()
    shares = {s.archetype.strip().lower(): float(s.meta_percentage or 0) for s in snapshots}
    names = [a.strip() for a in archetypes] if archetypes else [s.archetype.strip() for s in snapshots]
    seen: set = set()
    names = [n for n in names if not (n.lower() in seen or seen.add(n.lower()))]  # case-insensitive dedupe
    size = len(names) if archetypes else GAUNTLET_SIZE
    out: List[Opponent] = []
    for name in names:
        if len(out) >= size:
            break
        row = (await db.execute(LIST_SQL, {"format": format, "archetype": name})).first()
        if row is None or not row.main_deck:
            continue
        main: Dict[str, int] = {}
        for e in row.main_deck:  # a list can name a card on more than one line
            main[e["card_name"]] = main.get(e["card_name"], 0) + int(e["quantity"])
        out.append(Opponent(name, 1.0 if archetypes else shares.get(name.lower(), 0.0), main))
    return out
