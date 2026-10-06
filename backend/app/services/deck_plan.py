"""
Slot plans for Jev deck assembly: which kinds of cards a deck needs, and how many.

A plan comes from a reference archetype's recent decklists, or, for a brew,
from one LLM call (with fixed defaults when that fails).
Design: docs/superpowers/specs/2026-10-06-jev-deck-assembly-design.md
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Choice

from app.models.card import ROLE_DEFINITIONS
from app.services import jev

logger = logging.getLogger(__name__)

WINDOW_DAYS = 14  # "tournament-playable" = in a decklist from the last 14 days
REFERENCE_CONFIDENCE = 0.5
MAX_OPTIONS = 255  # Jev Choice maximum
MIN_LISTS = 2  # an archetype needs this many recent lists to be a reference
CHOICE_TIMEOUT = 10.0  # one Choice over 255 options can take a few seconds
CHOICE_DEADLINE = 15.0
NONE_OPTION = "none"
WUBRG = "WUBRG"
BANDS: List[Tuple[int, int]] = [(0, 1), (2, 2), (3, 3), (4, 4), (5, 99)]
# Decklists in the format from the last WINDOW_DAYS days (alias e = events). The
# day count is inlined: asyncpg cannot bind an integer into date arithmetic.
RECENT = f"e.format = :format AND e.date >= CURRENT_DATE - {WINDOW_DAYS}"


@dataclass
class Slot:
    role: str  # a non-land CARD_ROLES role, "creature" or "noncreature" (deck_fill adds "any", "land")
    cmc_min: int
    cmc_max: int
    copies: int
    description: str = ""
    type_contains: Optional[str] = None


@dataclass
class Plan:
    slots: List[Slot]
    lands: int
    nonbasic_lands: Optional[int] = None  # None: brew rule by color count
    copies: Dict[str, int] = field(default_factory=dict)  # reference main-deck copies by card name
    colors: List[str] = field(default_factory=list)  # reference lists' colors, WUBRG order
    reference: Optional[str] = None


def band_of(cmc: float) -> Tuple[int, int]:
    return next(b for b in BANDS if cmc <= b[1])


def describe(role: str, cmc_min: int, cmc_max: int) -> str:
    what = {"creature": "A creature", "noncreature": "A noncreature spell"}.get(role) or ROLE_DEFINITIONS[role]
    mv = f"mana value {cmc_min}" if cmc_min == cmc_max else (
        f"mana value {cmc_min} or more" if cmc_max >= 99 else f"mana value {cmc_min}-{cmc_max}")
    return f"{what} ({mv})"


def largest_remainder(sizes: Sequence[float], total: int) -> List[int]:
    """Integers proportional to `sizes` that sum to `total`."""
    scale = total / sum(sizes) if sum(sizes) else 0
    raw = [s * scale for s in sizes]
    out = [int(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return out


async def choose(client, state: Dict[str, Any], question: Choice):
    """One Jev Choice through the shared cap and a deadline; returns the answer
    (`.choice`, `.confidence`, `.probabilities`). Raises when Jev fails."""
    (resp,) = await jev.ask_many(client, [(state, {"pick": question})],
                                 deadline=CHOICE_DEADLINE, timeout=CHOICE_TIMEOUT)
    return resp.choices["pick"]


async def recent_archetypes(db: AsyncSession, format: str) -> List[Tuple[str, int]]:
    """(archetype, decklist count) for the window, most lists first. Names are
    grouped case- and whitespace-insensitively."""
    sql = text(f"""
        SELECT MIN(trim(d.archetype)) AS name, COUNT(*) AS n
        FROM decklists d JOIN events e ON e.id = d.event_id
        WHERE {RECENT}
          AND coalesce(trim(d.archetype), '') <> ''
        GROUP BY lower(trim(d.archetype))
        ORDER BY n DESC, name
    """)
    result = await db.execute(sql, {"format": format})
    return [(row[0], row[1]) for row in result.all()]


async def choose_reference(client, request_text: str, colors: List[str],
                           archetypes: List[Tuple[str, int]], format: str) -> Optional[str]:
    """The current archetype the user is asking to build, or None for a brew."""
    names = [(a, n) for a, n in archetypes if n >= MIN_LISTS and a.lower() != NONE_OPTION]
    names = names[: MAX_OPTIONS - 1]
    if not names:
        return None
    criteria = {a: f"{a}: {n} recent {format} tournament decklists" for a, n in names}
    criteria[NONE_OPTION] = "None of these: the user wants a different or new deck of their own"
    answer = await choose(client, {"request": request_text, "colors": colors}, Choice(
        instructions="Which current archetype is the user asking to build?", criteria=criteria))
    if answer.choice == NONE_OPTION or answer.confidence < REFERENCE_CONFIDENCE:
        return None
    return answer.choice
