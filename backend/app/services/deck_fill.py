"""
Jev deck assembly: fill a slot plan with cards played in recent tournaments.

Each slot is one Jev Choice over its pool of played cards; code turns the
ranking into copies. Requested cards always go in.
Design: docs/superpowers/specs/2026-10-06-jev-deck-assembly-design.md
"""

import asyncio
import itertools
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Choice, Noul

from app.services import jev, llm
from app.services.card_service import FORMAT_LEGALITY_MAP
from app.services.deck_plan import (
    CARD_COLORS, CHOICE_DEADLINE, CHOICE_TIMEOUT, MAX_OPTIONS, RECENT, WUBRG, Plan, Slot, archetype_keys, choose,
    choose_reference, choose_relatives, is_land, largest_remainder, plan_from_decklists, plan_with_llm,
    SPELL_ROLES, TYPE_ROLES, recent_archetypes, relative_candidates,
)
from app.services.guided_builder import front_cost

logger = logging.getLogger(__name__)

MAX_COPIES = 4
BREW_COPIES = [4, 4, 3, 2, 1]  # brew copies by Jev rank, then 1
ANY_SLOT = "Any card that makes this deck stronger"
# A slot Choice can answer "none of these fit": cards ranked below it are not taken,
# and the slot's count moves to other slots (thin pools otherwise force off-plan picks).
NO_FIT = "(none of these fit)"
NO_FIT_TEXT = "No card here fits this deck's plan; leave these copies to other slots"
SLOT_QUESTION = ("Which card best fills this slot in the deck described by `deck`? Prefer cards proven "
                 "in recent tournament play (see each option's play count) when they fit the slot. "
                 "The card must support the deck's plan; never pick one that works against it (e.g. a "
                 "sweeper that kills its own creatures in a creature deck).")


# Decklists of a set of archetypes (bind :archetypes to archetype_keys(...)).
ARCHETYPE_SCOPE = "AND lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))"


def _plays_cte(scoped: bool = False) -> str:
    """Main-deck plays per card (front face, lower case) in the window, within
    the :archetypes lists when `scoped`."""
    scope = ARCHETYPE_SCOPE if scoped else ""
    return f"""
        played AS (
            SELECT lower(split_part(x->>'card_name', ' // ', 1)) AS k, COUNT(DISTINCT d.id) AS plays
            FROM decklists d JOIN events e ON e.id = d.event_id
            CROSS JOIN LATERAL jsonb_array_elements(d.main_deck) AS x
            WHERE {RECENT} {scope}
            GROUP BY 1
        )"""


def _role_sql(slot: Slot) -> str:
    if slot.role == "any":
        cond = "true"
    elif slot.role == "type":  # type_contains alone decides
        cond = "false"
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
                    chosen: Sequence[str], archetypes: Sequence[str] = ()) -> List[Any]:
    """Played, legal, on-color nonland cards that fit the slot, most played first
    (at most MAX_OPTIONS); plays are counted within the `archetypes`' lists when
    given. Rows: name, mana_cost, type_line, oracle_text, cmc, plays."""
    sql = text(f"""
        WITH {_plays_cte(bool(archetypes))}
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
    if archetypes:
        params["archetypes"] = archetype_keys(archetypes)
    return list((await db.execute(sql, params)).all())


def option_text(row: Any, format: str) -> str:
    oracle = (row.oracle_text or "").replace("\n", " ")[:200]
    return (f"{row.mana_cost or ''} {row.type_line or ''}. {oracle} "
            f"[Played in {row.plays} recent {format.title()} tournament decklists]").strip()


def brew_copies(name: str, rank: int) -> int:
    return BREW_COPIES[rank] if rank < len(BREW_COPIES) else 1


async def fill_slot(client, slot: Slot, pool: Sequence[Any], state: Dict[str, Any], format: str,
                    copies_for: Callable[[str, int], int], question: str = SLOT_QUESTION,
                    allow_none: bool = True) -> List[Tuple[str, int]]:
    """Jev ranks the pool for the slot; code takes copies down that ranking until
    the slot is full. Fewer than slot.copies when the pool runs out or, with
    `allow_none`, when Jev ranks the rest below "none of these fit"."""
    if not pool or slot.copies <= 0:
        return []
    criteria = {r.name: option_text(r, format) for r in pool}
    if allow_none:
        criteria[NO_FIT] = NO_FIT_TEXT
    answer = await choose(client, state, Choice(
        instructions={"question": question, "slot": slot.description}, criteria=criteria))
    probs = answer.probabilities or {answer.choice: 1.0}
    floor = probs.get(NO_FIT, 0.0) if allow_none else 0.0
    ranked = sorted((r.name for r in pool if probs.get(r.name, 0.0) >= floor),
                    key=lambda n: -probs.get(n, 0.0))  # stable: ties keep play order
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


async def _fill_one(db: AsyncSession, client, slot: Slot, build: Build, colors: List[str], format: str,
                    state: Callable[[], Dict[str, Any]], copies_for: Callable[[str, int], int],
                    scope: Sequence[str], allow_none: bool = True) -> int:
    """Fill one slot: from the `scope` archetypes' cards first, then (for the
    shortfall) from the whole format. Returns copies still unfilled."""
    short = slot.copies
    for archetypes in ([scope] if scope else []) + [()]:
        if short <= 0:
            break
        part = Slot(**{**vars(slot), "copies": short})
        pool = await slot_pool(db, part, colors, format, list(build.copies), archetypes)
        rows = {r.name: r for r in pool}
        for name, q in await fill_slot(client, part, pool, state(), format, copies_for,
                                       allow_none=allow_none):
            build.add(name, q, rows[name])
            short -= q
    return short


async def fill_slots(db: AsyncSession, client, slots: List[Slot], build: Build, colors: List[str],
                     format: str, state: Callable[[], Dict[str, Any]],
                     copies_for: Callable[[str, int], int], scope: Sequence[str] = ()) -> int:
    """Fill `slots` into `build`, largest remaining slot first, each from the
    `scope` archetypes' played cards first (the reference, or a brew's
    relatives), then the format's. A slot's shortfall moves to a remaining slot with the same role,
    else the largest remaining slot; a final shortfall gets one catch-all slot.
    Returns copies still unfilled."""
    todo = [Slot(**vars(s)) for s in slots if s.copies > 0]  # copies: the plan stays intact
    short = 0
    while todo:
        slot = max(todo, key=lambda s: s.copies)
        todo.remove(slot)
        short = await _fill_one(db, client, slot, build, colors, format, state, copies_for, scope)
        if short and todo:
            target = next((s for s in todo if s.role == slot.role), None) or max(todo, key=lambda s: s.copies)
            target.copies += short
            short = 0
    if short:
        # last resort: the deck needs its spell count, so this slot may not answer "none"
        short = await _fill_one(db, client, Slot("any", 0, 99, short, ANY_SLOT), build, colors, format, state,
                                copies_for, scope, allow_none=False)
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


async def requested_rows(db: AsyncSession, names: Sequence[str], format: str) -> List[Any]:
    """The requested cards that exist and are legal in `format`."""
    rows = []
    for name in names:
        row = (await db.execute(REQUESTED_SQL, {"name": name, "legality": FORMAT_LEGALITY_MAP[format]})).first()
        if row is not None:
            rows.append(row)
    return rows


def colors_of(rows: Sequence[Any]) -> List[str]:
    return [c for c in WUBRG if any(c in (r.colors or []) for r in rows)]


def card_text(row: Any) -> str:
    return f"{row.name} ({row.mana_cost or ''} {row.type_line}): {(row.oracle_text or '').replace(chr(10), ' ')}"


# A deck built around a card gets one slot of the cards that card needs most.
SYNERGY_COPIES = (8, 16)
SYNERGY_SYSTEM = (
    "A player is building a 60-card Magic: The Gathering deck around the cards below. Judging "
    "only from their rules text, describe the one group of supporting cards they need most. "
    "Reply with only a JSON object: {\"role\": one of ROLES, \"type_contains\": one type-line "
    "word every supporting card has (such as Artifact, Equipment, Creature, Enchantment) or "
    "null, \"cmc_min\": int, \"cmc_max\": int, \"copies\": int from 8 to 16, "
    "\"description\": one sentence on what the supporting cards do for them}. When the cards "
    "reward doing something often, keep the supporting cards cheap. ROLES: "
    + ", ".join(SPELL_ROLES + list(TYPE_ROLES))
)


def parse_synergy_slot(content: str) -> Optional[Slot]:
    """The synergy slot from the LLM's JSON, or None if anything is off. With a type
    filter the slot takes cards of that type only (role "type")."""
    try:
        it = json.loads(content[content.index("{"): content.rindex("}") + 1])
        tc = (it.get("type_contains") or "").strip() or None
        role = "type" if tc else it["role"]
        lo, hi, copies = it.get("cmc_min", 0), it.get("cmc_max", 99), it["copies"]
        if role != "type" and role not in SPELL_ROLES and role not in TYPE_ROLES:
            return None
        if any(type(v) is not int or v < 0 for v in (lo, hi, copies)) or lo > hi:
            return None
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    copies = min(max(copies, SYNERGY_COPIES[0]), SYNERGY_COPIES[1])
    return Slot(role, lo, hi, copies, str(it.get("description") or "Cards that support the build-around")[:200], tc)


async def synergy_slot(rows: Sequence[Any]) -> Optional[Slot]:
    """One LLM call: the slot of supporting cards the build-around cards need, or None."""
    if not rows or not llm.is_configured():
        return None
    try:
        content = await asyncio.to_thread(
            llm.complete, SYNERGY_SYSTEM, "Build-around cards:\n" + "\n".join(map(card_text, rows)), 400)
    except Exception as e:
        logger.warning(f"[ASSEMBLY] Synergy slot LLM call failed, planning without it: {e}")
        return None
    return parse_synergy_slot(content)


MAX_ADDED_COLORS = 2
SHOWN_SUPPORT = 8  # supporting cards listed per color option
COLOR_QUESTION = (
    "Which colors give the build-around cards in `deck` the strongest deck? Judge how well each "
    "option's added supporting cards work with the build-around cards' rules text, and where the "
    "build-around cards are already played. A second or third color is worth it when its cards "
    "make the build-around much stronger.")

PLAYED_IN_SQL = text(f"""
    SELECT MIN(trim(d.archetype)) AS archetype, COUNT(DISTINCT d.id) AS lists
    FROM decklists d JOIN events e ON e.id = d.event_id
    CROSS JOIN LATERAL jsonb_array_elements(d.main_deck) AS x
    WHERE {RECENT} AND coalesce(trim(d.archetype), '') <> ''
      AND lower(split_part(x->>'card_name', ' // ', 1)) = ANY(CAST(:names AS varchar[]))
    GROUP BY lower(trim(d.archetype))
    ORDER BY lists DESC, archetype
""")


async def played_in(db: AsyncSession, rows: Sequence[Any], format: str) -> List[str]:
    """The recent archetypes that play the build-around cards, with list counts."""
    names = [r.name.split(" // ")[0].lower() for r in rows]
    result = await db.execute(PLAYED_IN_SQL, {"format": format, "names": names})
    return [f"{a}: {n} recent {format} lists" for a, n in result.all()]


async def choose_colors(db: AsyncSession, client, base: List[str], rows: Sequence[Any], slot: Slot,
                        format: str) -> List[str]:
    """The build-around cards' colors plus up to MAX_ADDED_COLORS more, by one Jev Choice
    over what each option adds to the synergy slot's pool of played cards."""
    others = [c for c in WUBRG if c not in base]
    options = [[c for c in WUBRG if c in {*base, *extra}]
               for k in range(MAX_ADDED_COLORS + 1) for extra in itertools.combinations(others, k)]
    base_names = {r.name for r in await slot_pool(db, slot, base, format, [])}
    criteria = {}
    for option in options:
        pool = await slot_pool(db, slot, option, format, [])
        added = [r for r in pool if r.name not in base_names][:SHOWN_SUPPORT]
        text_ = f"{len(pool)} played supporting cards"
        if added:
            text_ += "; adds " + ", ".join(f"{r.name} ({r.mana_cost or ''} {r.type_line}, {r.plays} lists)"
                                           for r in added)
        criteria["".join(option) or "colorless"] = text_
    state = {"deck": {"build_around": [card_text(r) for r in rows], "supporting_cards": slot.description,
                      "build_around_played_in": await played_in(db, rows, format)}}
    async with asyncio.timeout(CHOICE_DEADLINE):
        resp = await jev.ask(client, state, {"colors": Choice(instructions=COLOR_QUESTION, criteria=criteria)},
                             timeout=CHOICE_TIMEOUT)
    pick = resp.choices["colors"].choice
    return [] if pick == "colorless" else [c for c in WUBRG if c in pick]


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
    """Fallback when there's no data: by deck color count."""
    return BREW_NONBASICS.get(len(colors), 6)


# Nonbasic land count of each recent decklist whose spells use exactly :n_colors colors.
BREW_NONBASICS_SQL = text(f"""
    WITH lists AS (
        SELECT d.id, d.main_deck FROM decklists d JOIN events e ON e.id = d.event_id
        WHERE {RECENT}
    ), entries AS (
        SELECT l.id, x->>'card_name' AS n, (x->>'quantity')::int AS q
        FROM lists l, jsonb_array_elements(l.main_deck) x
    ), joined AS (
        SELECT en.id, en.q, c.type_line, {CARD_COLORS} AS cols
        FROM entries en
        LEFT JOIN LATERAL (
            SELECT * FROM cards c
            WHERE lower(split_part(c.name, ' // ', 1)) = lower(split_part(en.n, ' // ', 1)) LIMIT 1
        ) c ON true
    ), deck_colors AS (
        SELECT id, count(DISTINCT col) AS n FROM joined, unnest(cols) col
        WHERE split_part(coalesce(type_line, ''), ' // ', 1) NOT LIKE '%Land%'
        GROUP BY id
    )
    SELECT coalesce(sum(j.q) FILTER (WHERE split_part(coalesce(j.type_line, ''), ' // ', 1) LIKE '%Land%'
                                      AND j.type_line NOT LIKE 'Basic%'), 0) AS nonbasic
    FROM joined j JOIN deck_colors dc ON dc.id = j.id
    WHERE dc.n = :n_colors
    GROUP BY j.id
""")


async def brew_nonbasic_target(db: AsyncSession, colors: Sequence[str], format: str) -> int:
    """Lower quartile of the nonbasic land counts of recent decklists with as many
    colors as the deck; the fixed rule without data. Mono-colored lists run 3-12
    nonbasics, but the high end is land-heavy strategies (green landfall), which a
    median would hand to every mono brew."""
    rows = (await db.execute(BREW_NONBASICS_SQL, {"format": format, "n_colors": len(colors)})).all()
    counts = sorted(r[0] for r in rows)
    return counts[len(counts) // 4] if counts else brew_nonbasics(colors)


def _nonbasic_lands(build: Build) -> int:
    return sum(q for n, q in build.copies.items() if n in build.rows
               and is_land(build.rows[n].type_line) and not (build.rows[n].type_line or "").startswith("Basic"))


UTILITY_QUESTION = ("Does `land` do anything useful besides producing mana or fixing colors "
                    "(for example becoming a creature, or a sacrifice or activated effect)?")
UTILITY_THRESHOLD = 0.5


async def utility_lands(client, pool: Sequence[Any]) -> List[Any]:
    """The lands Jev judges to do something beyond making or fixing mana: a one-color
    deck gains nothing from fixing, so those are no better than a basic."""
    if not pool:
        return []
    requests = [({"land": {"name": r.name, "type_line": r.type_line, "oracle_text": r.oracle_text}},
                 {"utility": Noul(instructions=UTILITY_QUESTION)}) for r in pool]
    answers = await jev.ask_many(client, requests, deadline=CHOICE_DEADLINE, timeout=CHOICE_TIMEOUT)
    return [r for r, a in zip(pool, answers) if a.nouls["utility"].noul >= UTILITY_THRESHOLD]


async def fill_lands(db: AsyncSession, client, build: Build, plan: Plan, colors: List[str], format: str,
                     state: Callable[[], Dict[str, Any]], copies_for: Callable[[str, int], int]) -> None:
    """Add plan.lands lands (requested lands already took their share): nonbasics
    by Jev pick up to the nonbasic count, then basics split by the spells' pips."""
    target = (plan.nonbasic_lands if plan.nonbasic_lands is not None
              else await brew_nonbasic_target(db, colors, format))
    need = max(0, min(target - _nonbasic_lands(build), plan.lands))
    picked = 0
    if need:
        pool = await land_pool(db, colors, format, list(build.copies))
        if len(colors) == 1:
            pool = await utility_lands(client, pool)
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
Base every statement only on the cards' rules text listed below. Use standard Magic terms (e.g. haste, flying, burn, removal, card advantage). Do not invent mechanics, creature types, themes or flavor that the rules text doesn't state, and mention only cards in the list.
Reply with only JSON: {"name": "a short deck name", "strategy_summary": "2-4 sentences on how the deck plays and wins"}."""
RULES_TEXT_CHARS = 300
SIDEBOARD_SIZE = 15
SIDEBOARD_QUESTION = "Which card best belongs in this deck's sideboard?"
SIDEBOARD_SLOT = "A sideboard card: an answer or swap for this deck's hard matchups after game 1"


async def sideboard_pool(db: AsyncSession, colors: List[str], format: str, chosen: Sequence[str],
                         archetypes: Sequence[str] = ()) -> List[Any]:
    """Legal, on-color, nonbasic cards from sideboards in the window: the
    `archetypes`' lists, or every list in the format when none are given. Most
    played first, at most MAX_OPTIONS. Rows add `copies`: the card's average
    sideboard copies there, rounded, at least 1."""
    scope = ARCHETYPE_SCOPE if archetypes else ""
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
    if archetypes:
        params["archetypes"] = archetype_keys(archetypes)
    return list((await db.execute(sql, params)).all())


async def fill_sideboard(db: AsyncSession, client, main: Build, side: Build, colors: List[str], format: str,
                         state: Callable[[], Dict[str, Any]], scope: Sequence[str]) -> None:
    """Fill `side` to SIDEBOARD_SIZE: one Jev ranking over the `scope` archetypes'
    sideboard cards, then (to top up a short pool, or alone when there is no
    scope) one over the format's sideboard cards. Copies are each card's average sideboard copies,
    capped at 4; main-deck cards are excluded, so no card passes 4 in total."""
    for archetypes in ([scope] if scope else []) + [()]:
        need = SIDEBOARD_SIZE - side.total()
        if need <= 0:
            return
        pool = await sideboard_pool(db, colors, format, [*main.copies, *side.copies], archetypes)
        rows = {r.name: r for r in pool}
        picks = await fill_slot(client, Slot("sideboard", 0, 99, need, SIDEBOARD_SLOT), pool, state(), format,
                                lambda name, rank: rows[name].copies, SIDEBOARD_QUESTION,
                                allow_none=False)
        for name, q in picks:
            side.add(name, q, rows[name])
    if side.total() < SIDEBOARD_SIZE:
        raise ValueError(f"Only {side.total()} sideboard cards played recently in {format}")


def card_listing(build: Build) -> str:
    """One line per card: copies, name, cost and type, and its rules text, so text the
    LLM writes about the deck rests on what the cards do, not on their names."""
    lines = []
    for name, qty in build.copies.items():
        row = build.rows.get(name)
        if row is None:  # basic lands
            lines.append(f"{qty} {name}")
            continue
        head = " ".join(x for x in (row.mana_cost, row.type_line) if x)
        rules = " ".join((row.oracle_text or "").split())[:RULES_TEXT_CHARS]
        lines.append(f"{qty} {name} ({head}): {rules}")
    return "\n".join(lines)


async def summarize(main: Build, side: Build, request_text: str, reference: Optional[str],
                    colors: List[str], format: str) -> Tuple[str, str]:
    """(deck name, strategy summary) from one LLM call; a template without the LLM."""
    name = reference or f"{'-'.join(COLOR_WORDS[c] for c in colors)} Deck"
    summary = (f"Built from recent {format.title()} tournament lists of {reference}." if reference
               else f"A {format.title()} brew for: {request_text}")
    if not llm.is_configured():
        return name, summary
    listing = card_listing(main)
    if side.copies:
        listing += "\nSideboard:\n" + card_listing(side)
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


ASSEMBLY_DEADLINE = 60  # seconds for all Jev work; a slow Jev falls back to the LLM path


async def assemble(db: AsyncSession, request_text: str, colors: Optional[List[str]],
                   specific_cards: Optional[List[str]], format: str = "standard",
                   include_sideboard: bool = True, archetype: str = "", client=None) -> Dict[str, Any]:
    """Build a deck: reference or brew plan, requested cards, Jev-filled slots,
    lands, sideboard, summary. A brew is planned and filled from its relatives
    (the current archetypes Jev judges closest to the request), or from the LLM
    plan when it has none. Returns {name, strategy_summary, main_deck,
    sideboard} like ai_service.generate_deck, plus the reference archetype (None
    for a brew), the brew's relatives and the deck colors. Raises when Jev is not
    configured or fails, the format is not 60-card, the format has no recent
    decklists, or a brew has no colors; the caller falls back to the LLM path."""
    if format not in SIXTY_CARD_FORMATS:
        raise ValueError(f"Jev assembly builds 60-card formats only, not {format}")
    async with asyncio.timeout(ASSEMBLY_DEADLINE), jev.session(client) as client:
        if client is None:
            raise RuntimeError("Jev is not configured")
        archetypes = await recent_archetypes(db, format)
        if not archetypes:
            raise ValueError(f"No recent {format} decklists")
        build_around = [r for r in await requested_rows(db, specific_cards or [], format)
                        if not is_land(r.type_line)]
        synergy = await synergy_slot(build_around)
        if not colors and build_around:  # no colors asked for: the build-around's, plus its support's
            base = colors_of(build_around)
            colors = (await choose_colors(db, client, base, build_around, synergy, format) if synergy
                      else base) or None
        reference = await choose_reference(client, request_text, colors or [], archetypes, format)
        plan = await plan_from_decklists(db, [reference], format) if reference else None
        scope = [reference] if plan else []  # archetypes whose cards fill the deck first
        relatives: List[str] = []
        if plan is None:
            reference = None
            candidates = await relative_candidates(db, colors or [], format)
            relatives = await choose_relatives(client, request_text, colors or [], candidates)
            plan = await plan_from_decklists(db, relatives, format, colors) if relatives else None
            relatives = relatives if plan else []
            scope = relatives
        if plan is None:
            plan = await plan_with_llm(request_text, colors or [], archetype)

        if synergy and reference is None:
            _shrink_largest(plan.slots, synergy.copies)
            plan.slots.append(synergy)

        main, side = Build(), Build()
        requested_colors = await reserve_requested(db, specific_cards or [], plan, format, main)
        deck_colors = [c for c in WUBRG if c in {*(colors or plan.colors), *requested_colors}]
        if not deck_colors:
            raise ValueError("A brew needs colors")

        def state() -> Dict[str, Any]:
            return {"deck": {
                "plan": (f"{reference}, a current {format.title()} archetype" if reference
                         else f"{request_text}, built from the current archetypes closest to it" if relatives
                         else request_text),
                "request": request_text, "colors": deck_colors,
                **({"build_around": [card_text(r) for r in build_around]} if build_around else {}),
                "chosen": [f"{q}x {n}" for n, q in main.copies.items()],
                "sideboard": [f"{q}x {n}" for n, q in side.copies.items()],
            }}

        def main_copies(name: str, rank: int) -> int:
            return plan.copies.get(name) or brew_copies(name, rank)

        short = await fill_slots(db, client, plan.slots, main, deck_colors, format, state, main_copies, scope)
        plan.lands += short  # a pool too thin to fill the spells: basics keep the deck at 60
        await fill_lands(db, client, main, plan, deck_colors, format, state, main_copies)
        if include_sideboard:
            await fill_sideboard(db, client, main, side, deck_colors, format, state, scope)

    name, summary = await summarize(main, side, request_text, reference, deck_colors, format)
    logger.info(f"[ASSEMBLY] {name}: reference={reference} relatives={relatives} colors={deck_colors} "
                f"main={main.total()} sideboard={side.total()}")
    return {"name": name, "strategy_summary": summary, "main_deck": main.entries(),
            "sideboard": side.entries(), "reference": reference, "relatives": relatives, "colors": deck_colors,
            "synergy": ({"type_contains": synergy.type_contains, "min": SYNERGY_COPIES[0]}
                        if synergy and synergy.type_contains else None),
            "requested": list(specific_cards or [])}
