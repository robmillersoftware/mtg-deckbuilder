"""
Slot plans for Jev deck assembly: which kinds of cards a deck needs, and how many.

A plan comes from a reference archetype's recent decklists, or, for a brew,
from one LLM call (with fixed defaults when that fails).
Design: docs/superpowers/specs/2026-10-06-jev-deck-assembly-design.md
"""

import logging
from collections import defaultdict
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


MIN_SLOT = 2  # slots under this many copies merge into a neighbor
MAIN_SIZE = 60


# One row per main-deck entry in the archetype's lists, resolved to a card name
# (decklists list DFCs by front face; cards store "Front // Back").
REFERENCE_SQL = text(f"""
    WITH lists AS (
        SELECT d.id, d.main_deck
        FROM decklists d JOIN events e ON e.id = d.event_id
        WHERE {RECENT}
          AND lower(trim(d.archetype)) = lower(trim(:archetype))
    ),
    entries AS (
        SELECT l.id AS deck_id, x->>'card_name' AS entry, (x->>'quantity')::int AS qty
        FROM lists l CROSS JOIN LATERAL jsonb_array_elements(l.main_deck) x
    ),
    card AS (
        SELECT DISTINCT ON (k) k, c.name, c.type_line, c.cmc,
               coalesce(nullif(c.colors, '{{}}'), c.color_identity, '{{}}') AS colors
        FROM (SELECT DISTINCT lower(split_part(entry, ' // ', 1)) AS k FROM entries) n
        JOIN cards c ON lower(split_part(c.name, ' // ', 1)) = n.k
        ORDER BY k, c.name
    ),
    top_role AS (
        SELECT DISTINCT ON (c.name) c.name, r.role
        FROM card_roles r JOIN cards c ON c.id = r.card_id
        WHERE c.name IN (SELECT name FROM card) AND r.role NOT LIKE 'land%'
        ORDER BY c.name, r.confidence DESC NULLS LAST, r.efficiency DESC NULLS LAST
    )
    SELECT en.deck_id, coalesce(card.name, en.entry) AS name, en.qty,
           card.type_line, card.cmc, card.colors, top_role.role
    FROM entries en
    LEFT JOIN card ON card.k = lower(split_part(en.entry, ' // ', 1))
    LEFT JOIN top_role ON top_role.name = card.name
""")


def is_land(type_line: Optional[str]) -> bool:
    return "Land" in (type_line or "").split(" // ")[0]


def _slot_sizes(rows: Sequence[Any], n_lists: int) -> Dict[Tuple[str, Tuple[int, int]], float]:
    sums: Dict[Tuple[str, Tuple[int, int]], float] = defaultdict(float)
    for r in rows:
        role = r.role or ("creature" if "Creature" in (r.type_line or "") else "noncreature")
        sums[(role, band_of(r.cmc or 0))] += r.qty / n_lists
    return sums


def _merge_small(sizes: Dict[Tuple[str, Tuple[int, int]], float]) -> List[List[Any]]:
    """[role, cmc_min, cmc_max, avg copies] with every slot >= MIN_SLOT: a small
    slot merges into the same role's nearest band (widening its mana range),
    else into the same band's largest slot, else into the largest slot."""
    slots = [[role, lo, hi, avg] for (role, (lo, hi)), avg in sizes.items()]
    while len(slots) > 1:
        small = min(slots, key=lambda s: (s[3], s[0], s[1]))
        if small[3] >= MIN_SLOT:
            break
        others = [s for s in slots if s is not small]
        same_role = [s for s in others if s[0] == small[0]]
        if same_role:
            target = min(same_role, key=lambda s: (abs(s[1] - small[1]), s[1]))
            target[1], target[2] = min(target[1], small[1]), max(target[2], small[2])
        else:
            same_band = [s for s in others if (s[1], s[2]) == (small[1], small[2])]
            target = max(same_band or others, key=lambda s: s[3])
        target[3] += small[3]
        slots.remove(small)
    return slots


def _to_slots(sizes: Dict[Tuple[str, Tuple[int, int]], float], total: int) -> List[Slot]:
    merged = _merge_small(sizes)
    if not merged or total <= 0:
        return []
    counts = largest_remainder([s[3] for s in merged], total)
    return [Slot(role, lo, hi, n, describe(role, lo, hi))
            for (role, lo, hi, _), n in zip(merged, counts) if n > 0]


def _avg_copies(rows: Sequence[Any]) -> Dict[str, int]:
    """Average copies per card across the lists that play it, rounded, at least 1."""
    per: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        per[r.name].append(r.qty)
    return {name: max(1, round(sum(q) / len(q))) for name, q in per.items()}


async def plan_from_decklists(db: AsyncSession, archetype: str, format: str) -> Optional[Plan]:
    """Main-deck slot plan from the archetype's decklists in the window; None
    without lists. Sideboards are not slot-planned (deck_fill.fill_sideboard)."""
    rows = (await db.execute(REFERENCE_SQL, {
        "format": format, "archetype": archetype})).all()
    n_lists = len({r.deck_id for r in rows})
    if not n_lists:
        return None
    lands = [r for r in rows if is_land(r.type_line)]
    spells = [r for r in rows if not is_land(r.type_line)]
    land_count = round(sum(r.qty for r in lands) / n_lists)
    nonbasic = round(sum(r.qty for r in lands if not (r.type_line or "").startswith("Basic")) / n_lists)
    # A color counts when at least half the lists play it (splashes in one list don't).
    lists_with = {c: len({r.deck_id for r in spells if c in (r.colors or [])}) for c in WUBRG}
    return Plan(
        slots=_to_slots(_slot_sizes(spells, n_lists), MAIN_SIZE - land_count),
        lands=land_count,
        nonbasic_lands=min(nonbasic, land_count),
        copies=_avg_copies(rows),
        colors=[c for c in WUBRG if lists_with[c] * 2 >= n_lists],
        reference=archetype,
    )
