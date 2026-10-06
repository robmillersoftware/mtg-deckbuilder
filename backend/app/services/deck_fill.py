"""
Jev deck assembly: fill a slot plan with cards played in recent tournaments.

Each slot is one Jev Choice over its pool of played cards; code turns the
ranking into copies. Requested cards always go in.
Design: docs/superpowers/specs/2026-10-06-jev-deck-assembly-design.md
"""

import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Choice

from app.services import jev, llm
from app.services.card_service import FORMAT_LEGALITY_MAP
from app.services.deck_plan import (
    MAX_OPTIONS, RECENT, WUBRG, Plan, Slot, choose, choose_reference, is_land, largest_remainder,
    plan_from_decklists, plan_with_llm, recent_archetypes,
)
from app.services.guided_builder import front_cost

logger = logging.getLogger(__name__)

MAX_COPIES = 4
BREW_COPIES = [4, 4, 3, 2, 1]  # brew copies by Jev rank, then 1
ANY_SLOT = "Any card that makes this deck stronger"
SLOT_QUESTION = ("Which card best fills this slot in the deck described by `deck`? Prefer cards proven "
                 "in recent tournament play (see each option's play count) when they fit the slot. "
                 "The card must support the deck's plan; never pick one that works against it (e.g. a "
                 "sweeper that kills its own creatures in a creature deck).")

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
    probs = answer.probabilities or {answer.choice: 1.0}
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


BASICS = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
BREW_NONBASICS = {1: 0, 2: 4}  # by deck color count; 3+ colors: 6
LAND_SLOT = "a land for this deck's mana: fixing for its colors, or utility"
LAND_SEARCH = re.compile(r"search your library for ([^.]*)", re.IGNORECASE)


def fetches_for_colors(oracle_text: Optional[str], colors: Sequence[str]) -> bool:
    """False for a fetchland whose searches name basic land types (Plains, Island,
    Swamp, Mountain, Forest) but none of the deck colors' types. A generic
    "basic land card" search, or no search, is fine."""
    wanted = {BASICS[c] for c in colors}
    for match in LAND_SEARCH.finditer(oracle_text or ""):
        types = {t for t in BASICS.values() if t in match.group(1)}
        if types and not types & wanted:
            return False
    return True


async def land_pool(db: AsyncSession, colors: List[str], format: str, chosen: Sequence[str]) -> List[Any]:
    """Played, legal nonbasic lands whose color identity fits the deck (colorless
    included), most played first. Fetchlands have identity {}, so those that only
    find other colors' basic types are dropped here."""
    sql = text(f"""
        WITH {_plays_cte()}
        SELECT c.name, MAX(c.mana_cost) AS mana_cost, MAX(c.type_line) AS type_line,
               MAX(c.oracle_text) AS oracle_text, MAX(c.cmc) AS cmc, MAX(p.plays) AS plays,
               MAX(c.color_identity) AS color_identity
        FROM cards c JOIN played p ON p.k = lower(split_part(c.name, ' // ', 1))
        WHERE c.legalities->>:legality = 'legal'
          AND split_part(coalesce(c.type_line, ''), ' // ', 1) LIKE '%Land%'
          AND c.type_line NOT LIKE 'Basic%'
          AND coalesce(c.color_identity, '{{}}') <@ CAST(:colors AS varchar[])
          AND NOT (c.name = ANY(CAST(:chosen AS varchar[])))
        GROUP BY c.name
        ORDER BY plays DESC, c.name
        LIMIT {MAX_OPTIONS}
    """)
    params = {"format": format, "legality": FORMAT_LEGALITY_MAP[format], "colors": list(colors),
              "chosen": list(chosen)}
    rows = (await db.execute(sql, params)).all()
    return [r for r in rows if fetches_for_colors(r.oracle_text, colors)]


REQUESTED_SQL = text(f"""
    SELECT c.name, c.mana_cost, c.type_line, c.oracle_text, c.cmc, c.color_identity,
           {CARD_COLORS} AS colors,
           ARRAY(SELECT DISTINCT r.role FROM card_roles r JOIN cards c2 ON c2.id = r.card_id
                 WHERE c2.name = c.name) AS roles
    FROM cards c
    WHERE (lower(c.name) = lower(:name) OR lower(split_part(c.name, ' // ', 1)) = lower(:name))
      AND c.legalities->>:legality = 'legal'
    ORDER BY c.name
    LIMIT 1
""")


def _fits(slot: Slot, row: Any) -> bool:
    role_ok = (slot.role in ("any", *(row.roles or []))
               or (slot.type_contains and slot.type_contains.lower() in (row.type_line or "").lower())
               or (slot.role == "creature" and "Creature" in (row.type_line or ""))
               or (slot.role == "noncreature" and "Creature" not in (row.type_line or "")))
    return role_ok and slot.cmc_min <= (row.cmc or 0) <= slot.cmc_max


def _shrink_largest(slots: List[Slot], n: int) -> None:
    while n > 0 and any(s.copies for s in slots):
        largest = max(slots, key=lambda s: s.copies)
        cut = min(n, largest.copies)
        largest.copies -= cut
        n -= cut


async def reserve_requested(db: AsyncSession, names: Sequence[str], plan: Plan, format: str,
                            build: Build) -> List[str]:
    """Put each requested card in `build` (4 copies, 1 if Legendary) and take its
    space out of the plan: a land from the land counts, a spell from the first
    slot it fits (any excess from the largest slot). Returns the requested
    cards' colors. Unknown or format-illegal names are skipped."""
    colors: set = set()
    requested: List[str] = []
    for name in names:
        row = (await db.execute(REQUESTED_SQL, {"name": name, "legality": FORMAT_LEGALITY_MAP[format]})).first()
        if row is None:
            logger.warning(f"[ASSEMBLY] Requested card not found or not legal in {format}: {name}")
            continue
        if row.name in build.copies:
            continue
        qty = 1 if "Legendary" in (row.type_line or "") else MAX_COPIES
        build.add(row.name, qty, row)
        requested.append(row.name)
        colors.update(row.colors or [])
        if is_land(row.type_line):
            left = max(0, qty - plan.lands)
            plan.lands = max(0, plan.lands - qty)
            _shrink_largest(plan.slots, left)  # more requested lands than land slots: spells give way
            continue
        slot = next((s for s in plan.slots if s.copies and _fits(s, row)), None)
        excess = qty
        if slot is not None:
            excess = max(0, qty - slot.copies)
            slot.copies = max(0, slot.copies - qty)
        before = sum(s.copies for s in plan.slots)
        _shrink_largest(plan.slots, excess)
        left = excess - (before - sum(s.copies for s in plan.slots))
        plan.lands = max(0, plan.lands - left)  # no spell slots left: lands give way
    # Requests alone over 60: cut copies toward 1, last requested first.
    for name in reversed(requested):
        cut = min(max(0, build.total() - 60), build.copies[name] - 1)
        build.copies[name] -= cut
    return [c for c in WUBRG if c in colors]


def pips(rows: Sequence[Tuple[Any, int]]) -> Dict[str, int]:
    """Colored mana symbols across (row, copies) pairs' mana costs (hybrid counts both
    colors). A card without a mana cost (DFCs store it per face) counts each
    color of its identity once."""
    out = {c: 0 for c in WUBRG}
    for r, qty in rows:
        symbols = re.findall(r"\{([^}]*)\}", front_cost(r.mana_cost))
        letters = [ch for s in symbols for ch in s if ch in WUBRG] if symbols else [
            ch for ch in (r.color_identity or "") if ch in WUBRG]
        for ch in letters:
            out[ch] += qty
    return out


def split_basics(counts: Dict[str, int], colors: Sequence[str], n: int) -> Dict[str, int]:
    """n basics split by pip share, at least 1 per deck color when n allows."""
    colors = [c for c in WUBRG if c in colors]
    if n <= 0 or not colors:
        return {}
    weights = [counts.get(c, 0) or 0 for c in colors]
    if not any(weights):
        weights = [1] * len(colors)
    floor = 1 if n >= len(colors) else 0
    extra = largest_remainder(weights, n - floor * len(colors))
    return {BASICS[c]: floor + e for c, e in zip(colors, extra) if floor + e > 0}


def brew_nonbasics(colors: Sequence[str]) -> int:
    return BREW_NONBASICS.get(len(colors), 6)


def _nonbasic_lands(build: Build) -> int:
    return sum(q for n, q in build.copies.items() if n in build.rows
               and is_land(build.rows[n].type_line) and not (build.rows[n].type_line or "").startswith("Basic"))


async def fill_lands(db: AsyncSession, client, build: Build, plan: Plan, colors: List[str], format: str,
                     state: Callable[[], Dict[str, Any]], copies_for: Callable[[str, int], int]) -> None:
    """Add plan.lands lands (requested lands already took their share): nonbasics
    by Jev pick up to the nonbasic count, then basics split by the spells' pips."""
    target = plan.nonbasic_lands if plan.nonbasic_lands is not None else brew_nonbasics(colors)
    need = max(0, min(target - _nonbasic_lands(build), plan.lands))
    picked = 0
    if need:
        pool = await land_pool(db, colors, format, list(build.copies))
        rows = {r.name: r for r in pool}
        for name, q in await fill_slot(client, Slot("land", 0, 99, need, LAND_SLOT), pool, state(),
                                       format, copies_for):
            build.add(name, q, rows[name])
            picked += q
    spells = [(r, build.copies[n]) for n, r in build.rows.items() if not is_land(r.type_line)]
    for name, q in split_basics(pips(spells), colors, plan.lands - picked).items():
        build.add(name, q)


SIXTY_CARD_FORMATS = {f for f in FORMAT_LEGALITY_MAP if f != "cedh"}
COLOR_WORDS = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
SUMMARY_SYSTEM = """You name and describe a finished Magic: The Gathering deck.
Reply with only JSON: {"name": "a short deck name", "strategy_summary": "2-4 sentences on how the deck plays and wins"}."""
SIDEBOARD_SIZE = 15
SIDEBOARD_QUESTION = "Which card best belongs in this deck's sideboard?"
SIDEBOARD_SLOT = "A sideboard card: an answer or swap for this deck's hard matchups after game 1"


async def sideboard_pool(db: AsyncSession, colors: List[str], format: str, chosen: Sequence[str],
                         archetype: Optional[str] = None) -> List[Any]:
    """Legal, on-color, nonbasic cards from sideboards in the window: the
    archetype's lists, or every list in the format when archetype is None. Most
    played first, at most MAX_OPTIONS. Rows add `copies`: the card's average
    sideboard copies there, rounded, at least 1."""
    scope = "AND lower(trim(d.archetype)) = lower(trim(:archetype))" if archetype else ""
    sql = text(f"""
        WITH side AS (
            SELECT lower(split_part(x->>'card_name', ' // ', 1)) AS k, COUNT(DISTINCT d.id) AS plays,
                   AVG((x->>'quantity')::int) AS avg_copies
            FROM decklists d JOIN events e ON e.id = d.event_id
            CROSS JOIN LATERAL jsonb_array_elements(d.sideboard) AS x
            WHERE {RECENT} {scope}
            GROUP BY 1
        )
        SELECT c.name, MAX(c.mana_cost) AS mana_cost, MAX(c.type_line) AS type_line,
               MAX(c.oracle_text) AS oracle_text, MAX(c.cmc) AS cmc, MAX(s.plays) AS plays,
               GREATEST(1, ROUND(MAX(s.avg_copies)))::int AS copies,
               MAX(c.color_identity) AS color_identity
        FROM cards c JOIN side s ON s.k = lower(split_part(c.name, ' // ', 1))
        WHERE c.legalities->>:legality = 'legal'
          AND {CARD_COLORS} <@ CAST(:colors AS varchar[])
          AND coalesce(c.type_line, '') NOT LIKE 'Basic%'
          AND NOT (c.name = ANY(CAST(:chosen AS varchar[])))
        GROUP BY c.name
        ORDER BY plays DESC, c.name
        LIMIT {MAX_OPTIONS}
    """)
    params = {"format": format, "legality": FORMAT_LEGALITY_MAP[format], "colors": list(colors),
              "chosen": list(chosen)}
    if archetype:
        params["archetype"] = archetype
    return list((await db.execute(sql, params)).all())


async def fill_sideboard(db: AsyncSession, client, main: Build, side: Build, colors: List[str], format: str,
                         state: Callable[[], Dict[str, Any]], reference: Optional[str]) -> None:
    """Fill `side` to SIDEBOARD_SIZE: one Jev ranking over the reference lists'
    sideboard cards, then (to top up a short pool, or alone for a brew) one over
    the format's sideboard cards. Copies are each card's average sideboard copies,
    capped at 4; main-deck cards are excluded, so no card passes 4 in total."""
    for archetype in ([reference] if reference else []) + [None]:
        need = SIDEBOARD_SIZE - side.total()
        if need <= 0:
            return
        pool = await sideboard_pool(db, colors, format, [*main.copies, *side.copies], archetype)
        rows = {r.name: r for r in pool}
        picks = await fill_slot(client, Slot("sideboard", 0, 99, need, SIDEBOARD_SLOT), pool, state(), format,
                                lambda name, rank: rows[name].copies, SIDEBOARD_QUESTION)
        for name, q in picks:
            side.add(name, q, rows[name])
    if side.total() < SIDEBOARD_SIZE:
        raise ValueError(f"Only {side.total()} sideboard cards played recently in {format}")


async def summarize(main: Build, side: Build, request_text: str, reference: Optional[str],
                    colors: List[str], format: str) -> Tuple[str, str]:
    """(deck name, strategy summary) from one LLM call; a template without the LLM."""
    name = reference or f"{'-'.join(COLOR_WORDS[c] for c in colors)} Deck"
    summary = (f"Built from recent {format.title()} tournament lists of {reference}." if reference
               else f"A {format.title()} brew for: {request_text}")
    if not llm.is_configured():
        return name, summary
    listing = "\n".join(f"{q} {n}" for n, q in main.copies.items())
    if side.copies:
        listing += "\nSideboard:\n" + "\n".join(f"{q} {n}" for n, q in side.copies.items())
    try:
        content = await asyncio.to_thread(llm.complete, SUMMARY_SYSTEM,
                                          f"Request: {request_text}\n\n{listing}", 600)
        data = json.loads(content[content.index("{"): content.rindex("}") + 1])
        if isinstance(data.get("name"), str) and isinstance(data.get("strategy_summary"), str) \
                and data["name"].strip() and data["strategy_summary"].strip():
            return data["name"].strip(), data["strategy_summary"].strip()
    except Exception as e:
        logger.warning(f"[ASSEMBLY] Summary LLM call failed, using the template: {e}")
    return name, summary


async def assemble(db: AsyncSession, request_text: str, colors: Optional[List[str]],
                   specific_cards: Optional[List[str]], format: str = "standard",
                   include_sideboard: bool = True, archetype: str = "", client=None) -> Dict[str, Any]:
    """Build a deck: reference or brew plan, requested cards, Jev-filled slots,
    lands, sideboard, summary. Returns {name, strategy_summary,
    main_deck, sideboard} like ai_service.generate_deck, plus the reference
    archetype (None for a brew) and the deck colors. Raises when Jev is not
    configured or fails, the format is not 60-card, the format has no recent
    decklists, or a brew has no colors; the caller falls back to the LLM path."""
    if format not in SIXTY_CARD_FORMATS:
        raise ValueError(f"Jev assembly builds 60-card formats only, not {format}")
    async with jev.session(client) as client:
        if client is None:
            raise RuntimeError("Jev is not configured")
        archetypes = await recent_archetypes(db, format)
        if not archetypes:
            raise ValueError(f"No recent {format} decklists")
        reference = await choose_reference(client, request_text, colors or [], archetypes, format)
        plan = await plan_from_decklists(db, reference, format) if reference else None
        if plan is None:
            reference = None
            plan = await plan_with_llm(request_text, colors or [], archetype)

        main, side = Build(), Build()
        requested_colors = await reserve_requested(db, specific_cards or [], plan, format, main)
        deck_colors = [c for c in WUBRG if c in {*(colors or plan.colors), *requested_colors}]
        if not deck_colors:
            raise ValueError("A brew needs colors")

        def state() -> Dict[str, Any]:
            return {"deck": {
                "plan": f"{reference}, a current {format.title()} archetype" if reference else request_text,
                "request": request_text, "colors": deck_colors,
                "chosen": [f"{q}x {n}" for n, q in main.copies.items()],
                "sideboard": [f"{q}x {n}" for n, q in side.copies.items()],
            }}

        def main_copies(name: str, rank: int) -> int:
            return plan.copies.get(name) or brew_copies(name, rank)

        short = await fill_slots(db, client, plan.slots, main, deck_colors, format, state, main_copies)
        plan.lands += short  # a pool too thin to fill the spells: basics keep the deck at 60
        await fill_lands(db, client, main, plan, deck_colors, format, state, main_copies)
        if include_sideboard:
            await fill_sideboard(db, client, main, side, deck_colors, format, state, reference)

    name, summary = await summarize(main, side, request_text, reference, deck_colors, format)
    logger.info(f"[ASSEMBLY] {name}: reference={reference} colors={deck_colors} "
                f"main={main.total()} sideboard={side.total()}")
    return {"name": name, "strategy_summary": summary, "main_deck": main.entries(),
            "sideboard": side.entries(), "reference": reference, "colors": deck_colors}
