"""
Jev deck assembly: fill a slot plan with cards played in recent tournaments.

Each slot is one Jev Choice over its pool of played cards; code turns the
ranking into copies. Requested cards always go in.
Design: docs/superpowers/specs/2026-10-06-jev-deck-assembly-design.md
"""

import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Choice

from app.services.card_service import FORMAT_LEGALITY_MAP
from app.services.deck_plan import MAX_OPTIONS, RECENT, Slot, choose

logger = logging.getLogger(__name__)

MAX_COPIES = 4
BREW_COPIES = [4, 4, 3, 2, 1]  # brew copies by Jev rank, then 1
ANY_SLOT = "Any card that makes this deck stronger"
SLOT_QUESTION = ("Which card best fills this slot in the deck described by `deck`? Prefer cards proven "
                 "in recent tournament play (see each option's play count) when they fit the slot.")

# A card's colors; DFCs store colors per face (top-level colors are empty), so
# fall back to color identity.
CARD_COLORS = "coalesce(nullif(c.colors, '{}'), c.color_identity, '{}')"


def _plays_cte() -> str:
    """Main-deck plays per card (front face, lower case) in the window."""
    return f"""
        played AS (
            SELECT lower(split_part(x->>'card_name', ' // ', 1)) AS k, COUNT(DISTINCT d.id) AS plays
            FROM decklists d JOIN events e ON e.id = d.event_id
            CROSS JOIN LATERAL jsonb_array_elements(d.main_deck) AS x
            WHERE {RECENT}
            GROUP BY 1
        )"""


def _role_sql(slot: Slot) -> str:
    if slot.role == "any":
        cond = "true"
    elif slot.role == "creature":
        cond = "c.type_line LIKE '%Creature%'"
    elif slot.role == "noncreature":
        cond = "c.type_line NOT LIKE '%Creature%'"
    else:
        cond = "EXISTS (SELECT 1 FROM card_roles r WHERE r.card_id = c.id AND r.role = :role)"
    if slot.type_contains:
        cond = f"({cond} OR c.type_line ILIKE '%' || :type_contains || '%')"
    return cond


async def slot_pool(db: AsyncSession, slot: Slot, colors: List[str], format: str,
                    chosen: Sequence[str]) -> List[Any]:
    """Played, legal, on-color nonland cards that fit the slot, most played first
    (at most MAX_OPTIONS). Rows: name, mana_cost, type_line, oracle_text, cmc, plays."""
    sql = text(f"""
        WITH {_plays_cte()}
        SELECT c.name, MAX(c.mana_cost) AS mana_cost, MAX(c.type_line) AS type_line,
               MAX(c.oracle_text) AS oracle_text, MAX(c.cmc) AS cmc, MAX(p.plays) AS plays,
               MAX(c.color_identity) AS color_identity
        FROM cards c JOIN played p ON p.k = lower(split_part(c.name, ' // ', 1))
        WHERE c.legalities->>:legality = 'legal'
          AND split_part(coalesce(c.type_line, ''), ' // ', 1) NOT LIKE '%Land%'
          AND {CARD_COLORS} <@ CAST(:colors AS varchar[])
          AND c.cmc BETWEEN :cmc_min AND :cmc_max
          AND {_role_sql(slot)}
          AND NOT (c.name = ANY(CAST(:chosen AS varchar[])))
        GROUP BY c.name
        ORDER BY plays DESC, c.name
        LIMIT {MAX_OPTIONS}
    """)
    params = {"format": format, "legality": FORMAT_LEGALITY_MAP[format], "colors": list(colors),
              "cmc_min": slot.cmc_min, "cmc_max": slot.cmc_max, "chosen": list(chosen),
              "role": slot.role, "type_contains": slot.type_contains}
    return list((await db.execute(sql, params)).all())


def option_text(row: Any, format: str) -> str:
    oracle = (row.oracle_text or "").replace("\n", " ")[:200]
    return (f"{row.mana_cost or ''} {row.type_line or ''}. {oracle} "
            f"[Played in {row.plays} recent {format.title()} tournament decklists]").strip()


def brew_copies(name: str, rank: int) -> int:
    return BREW_COPIES[rank] if rank < len(BREW_COPIES) else 1


async def fill_slot(client, slot: Slot, pool: Sequence[Any], state: Dict[str, Any], format: str,
                    copies_for: Callable[[str, int], int], question: str = SLOT_QUESTION) -> List[Tuple[str, int]]:
    """Jev ranks the pool for the slot; code takes copies down that ranking until
    the slot is full. Fewer than slot.copies when the pool runs out."""
    if not pool or slot.copies <= 0:
        return []
    answer = await choose(client, state, Choice(
        instructions={"question": question, "slot": slot.description},
        criteria={r.name: option_text(r, format) for r in pool}))
    probs = answer.probabilities or {}
    ranked = sorted((r.name for r in pool), key=lambda n: -probs.get(n, 0.0))  # stable: ties keep play order
    picks, need = [], slot.copies
    for rank, name in enumerate(ranked):
        if need <= 0:
            break
        q = min(max(1, copies_for(name, rank)), MAX_COPIES, need)
        picks.append((name, q))
        need -= q
    return picks


class Build:
    """A deck in progress: copies by card name plus each card's row."""

    def __init__(self) -> None:
        self.copies: Dict[str, int] = {}
        self.rows: Dict[str, Any] = {}

    def add(self, name: str, qty: int, row: Any = None) -> None:
        self.copies[name] = self.copies.get(name, 0) + qty
        if row is not None:
            self.rows.setdefault(name, row)

    def total(self) -> int:
        return sum(self.copies.values())

    def entries(self) -> List[Dict[str, Any]]:
        return [{"card_name": n, "quantity": q} for n, q in self.copies.items() if q > 0]


async def fill_slots(db: AsyncSession, client, slots: List[Slot], build: Build, colors: List[str],
                     format: str, state: Callable[[], Dict[str, Any]],
                     copies_for: Callable[[str, int], int]) -> int:
    """Fill `slots` into `build`, largest remaining slot first. A slot's shortfall
    moves to a remaining slot with the same role, else the largest remaining slot;
    a final shortfall gets one catch-all slot. Returns copies still unfilled."""
    todo = [Slot(**vars(s)) for s in slots if s.copies > 0]  # copies: the plan stays intact
    short = 0
    while todo:
        slot = max(todo, key=lambda s: s.copies)
        todo.remove(slot)
        pool = await slot_pool(db, slot, colors, format, list(build.copies))
        rows = {r.name: r for r in pool}
        picks = await fill_slot(client, slot, pool, state(), format, copies_for)
        for name, q in picks:
            build.add(name, q, rows[name])
        short = slot.copies - sum(q for _, q in picks)
        if short and todo:
            target = next((s for s in todo if s.role == slot.role), None) or max(todo, key=lambda s: s.copies)
            target.copies += short
            short = 0
    if short:
        catch_all = Slot("any", 0, 99, short, ANY_SLOT)
        pool = await slot_pool(db, catch_all, colors, format, list(build.copies))
        rows = {r.name: r for r in pool}
        picks = await fill_slot(client, catch_all, pool, state(), format, copies_for)
        for name, q in picks:
            build.add(name, q, rows[name])
        short -= sum(q for _, q in picks)
    return short
