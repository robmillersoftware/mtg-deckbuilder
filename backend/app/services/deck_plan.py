"""
Slot plans for Jev deck assembly: which kinds of cards a deck needs, and how many.

A plan comes from a reference archetype's recent decklists, or, for a brew,
from one LLM call (with fixed defaults when that fails).
Design: docs/superpowers/specs/2026-10-06-jev-deck-assembly-design.md
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Choice, Noul

from app.models.card import CARD_ROLES, ROLE_DEFINITIONS
from app.services import jev, llm

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
# A card's colors; DFCs store colors per face (top-level colors are empty), so
# fall back to color identity.
CARD_COLORS = "coalesce(nullif(c.colors, '{}'), c.color_identity, '{}')"


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


def archetype_keys(names: Sequence[str]) -> List[str]:
    """Archetype names as the SQL compares them (lower case, trimmed), deduplicated:
    bind as :archetypes against lower(trim(d.archetype))."""
    return sorted({n.strip().lower() for n in names if n.strip()})


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


# One row per main-deck entry in the :archetypes lists (archetype_keys), resolved
# to a card name (decklists list DFCs by front face; cards store "Front // Back").
REFERENCE_SQL = text(f"""
    WITH lists AS (
        SELECT d.id, d.main_deck
        FROM decklists d JOIN events e ON e.id = d.event_id
        WHERE {RECENT}
          AND lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))
    ),
    entries AS (
        SELECT l.id AS deck_id, x->>'card_name' AS entry, (x->>'quantity')::int AS qty
        FROM lists l CROSS JOIN LATERAL jsonb_array_elements(l.main_deck) x
    ),
    card AS (
        SELECT DISTINCT ON (k) k, c.name, c.type_line, c.cmc,
               {CARD_COLORS} AS colors
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


async def plan_from_decklists(db: AsyncSession, archetypes: Sequence[str], format: str,
                              colors: Optional[Sequence[str]] = None) -> Optional[Plan]:
    """Main-deck slot plan from the `archetypes`' decklists in the window; None
    without lists. Sideboards are not slot-planned (deck_fill.fill_sideboard).

    A reference passes [archetype] and no colors. A brew passes its relatives and
    the deck colors: slots then count only spells whose colors fit the deck
    (colorless included), the nonbasic count is left to the brew rule, and the
    plan's colors are the deck colors."""
    rows = (await db.execute(REFERENCE_SQL, {
        "format": format, "archetypes": archetype_keys(archetypes)})).all()
    n_lists = len({r.deck_id for r in rows})
    if not n_lists:
        return None
    lands = [r for r in rows if is_land(r.type_line)]
    spells = [r for r in rows if not is_land(r.type_line)]
    if colors is not None:  # unknown names (colors None) can't be filled, so they don't count
        spells = [r for r in spells if r.colors is not None and set(r.colors) <= set(colors)]
    land_count = round(sum(r.qty for r in lands) / n_lists)
    nonbasic = round(sum(r.qty for r in lands if not (r.type_line or "").startswith("Basic")) / n_lists)
    # A color counts when at least half the lists play it (splashes in one list don't).
    lists_with = {c: len({r.deck_id for r in spells if c in (r.colors or [])}) for c in WUBRG}
    return Plan(
        slots=_to_slots(_slot_sizes(spells, n_lists), MAIN_SIZE - land_count),
        lands=land_count,
        nonbasic_lands=None if colors is not None else min(nonbasic, land_count),
        copies=_avg_copies(rows),
        colors=([c for c in WUBRG if c in colors] if colors is not None
                else [c for c in WUBRG if lists_with[c] * 2 >= n_lists]),
        reference=" + ".join(archetypes),
    )


MIN_ON_COLOR_SPELLS = 8  # a brew relative's lists average this many on-color main-deck spells
RELATIVE_THRESHOLD = 0.5
MAX_RELATIVES = 5
TOP_CARDS = 25  # cards per candidate shown to Jev
RELATIVE_QUESTION = ("Does `archetype` play the same kind of game as the deck the user asked for "
                     "(for example fast aggro, midrange, control or ramp)? Judge its game plan from its "
                     "cards; ignore its other colors.")

# Per (archetype, main-deck spell) in the window: the archetype's list count, the
# spell's average copies per list, and whether its colors fit :colors (colorless
# included). Archetypes with fewer than :min_lists lists and unknown names are left out.
CANDIDATES_SQL = text(f"""
    WITH lists AS (
        SELECT d.id, lower(trim(d.archetype)) AS a, trim(d.archetype) AS label, d.main_deck
        FROM decklists d JOIN events e ON e.id = d.event_id
        WHERE {RECENT}
          AND coalesce(trim(d.archetype), '') <> ''
    ),
    sizes AS (
        SELECT a, MIN(label) AS archetype, COUNT(*) AS lists FROM lists GROUP BY a HAVING COUNT(*) >= :min_lists
    ),
    entries AS (
        SELECT l.a, x->>'card_name' AS entry, (x->>'quantity')::int AS qty
        FROM lists l JOIN sizes s ON s.a = l.a
        CROSS JOIN LATERAL jsonb_array_elements(l.main_deck) x
    ),
    card AS (
        SELECT DISTINCT ON (k) k, c.name, c.type_line, {CARD_COLORS} AS colors
        FROM (SELECT DISTINCT lower(split_part(entry, ' // ', 1)) AS k FROM entries) n
        JOIN cards c ON lower(split_part(c.name, ' // ', 1)) = n.k
        ORDER BY k, c.name
    )
    SELECT s.archetype, s.lists, card.name, SUM(en.qty)::float / s.lists AS avg_copies,
           bool_and(card.colors <@ CAST(:colors AS varchar[])) AS on_color
    FROM entries en
    JOIN card ON card.k = lower(split_part(en.entry, ' // ', 1))
    JOIN sizes s ON s.a = en.a
    WHERE split_part(coalesce(card.type_line, ''), ' // ', 1) NOT LIKE '%Land%'
    GROUP BY s.archetype, s.lists, card.name
    ORDER BY s.lists DESC, s.archetype, avg_copies DESC, card.name
""")


async def relative_candidates(db: AsyncSession, colors: Sequence[str],
                              format: str) -> List[Tuple[str, int, List[Tuple[str, float]]]]:
    """(archetype, list count, top cards) for each current archetype with at least
    MIN_LISTS lists whose lists average at least MIN_ON_COLOR_SPELLS main-deck
    spell copies with colors within `colors`, most lists first. Top cards are its
    TOP_CARDS most-played main-deck spells, any color, with average copies per
    list. No colors: no candidates."""
    if not colors:
        return []
    rows = (await db.execute(CANDIDATES_SQL, {
        "format": format, "colors": list(colors), "min_lists": MIN_LISTS})).all()
    found: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        a = found.setdefault(r.archetype, {"lists": r.lists, "on_color": 0.0, "cards": []})
        a["on_color"] += r.avg_copies if r.on_color else 0.0
        a["cards"].append((r.name, round(r.avg_copies, 1)))
    return [(name, a["lists"], a["cards"][:TOP_CARDS]) for name, a in found.items()
            if a["on_color"] >= MIN_ON_COLOR_SPELLS]


async def choose_relatives(client, request_text: str, colors: Sequence[str],
                           candidates: Sequence[Tuple[str, int, List[Tuple[str, float]]]]) -> List[str]:
    """The candidates Jev judges close relatives of the requested deck (one Noul
    each, judged from the archetype's cards, not only its name): at or above
    RELATIVE_THRESHOLD, most likely first, at most MAX_RELATIVES. Raises when Jev
    fails."""
    if not candidates:
        return []
    requests = [({"request": request_text, "colors": list(colors),
                  "archetype": {"name": name, "lists": lists, "cards": dict(cards)}},
                 {"relative": Noul(instructions=RELATIVE_QUESTION)}) for name, lists, cards in candidates]
    answers = await jev.ask_many(client, requests, deadline=CHOICE_DEADLINE, timeout=CHOICE_TIMEOUT)
    scored = [(a.nouls["relative"].noul, name) for a, (name, _, _) in zip(answers, candidates)]
    ranked = sorted((s for s in scored if s[0] >= RELATIVE_THRESHOLD), key=lambda s: -s[0])  # stable
    return [name for _, name in ranked[:MAX_RELATIVES]]


TYPE_ROLES = ("creature", "noncreature")  # fallback roles for untagged cards
SPELL_ROLES = [r for r in CARD_ROLES if not r.startswith("land_")]
LAND_COUNTS = {"control": 24, "midrange": 23}  # brews; everything else 22

# Brew defaults when the LLM plan is missing or invalid: (role, cmc_min, cmc_max, copies).
DEFAULT_PLANS: Dict[str, List[Tuple[str, int, int, int]]] = {
    "aggro": [("threat_cheap", 0, 1, 8), ("threat_cheap", 2, 2, 12), ("threat_midrange", 3, 4, 8),
              ("removal_targeted", 0, 2, 6), ("burn", 0, 3, 4)],
    "midrange": [("threat_cheap", 0, 2, 8), ("threat_midrange", 3, 4, 10), ("threat_finisher", 5, 99, 4),
                 ("removal_targeted", 0, 3, 8), ("card_draw", 2, 4, 4), ("removal_mass", 3, 5, 3)],
    "control": [("removal_targeted", 0, 3, 8), ("counterspell", 1, 3, 6), ("removal_mass", 3, 5, 4),
                ("card_draw", 1, 4, 8), ("threat_finisher", 4, 99, 5), ("threat_midrange", 2, 4, 5)],
}

PLAN_SYSTEM = """You plan the nonland card slots of a 60-card Magic: The Gathering deck.
Reply with only a JSON list. Each item: {"role": one of ROLES, "cmc_min": int,
"cmc_max": int, "type_contains": a type-line word such as "Creature" or "Equipment" or null,
"copies": positive int, "description": one sentence on what the slot does}.
Use 6 to 10 slots. Copies across slots should total about the deck's nonland count.
ROLES: """ + ", ".join(SPELL_ROLES + list(TYPE_ROLES))


def lands_for(archetype_hint: str) -> int:
    return LAND_COUNTS.get((archetype_hint or "").strip().lower(), 22)


def parse_llm_plan(content: str, lands: int) -> Optional[List[Slot]]:
    """Validated slots scaled to MAIN_SIZE - lands, or None if anything is off."""
    try:
        items = json.loads(content[content.index("["): content.rindex("]") + 1])
        slots = []
        for it in items:
            role, copies = it["role"], it["copies"]
            lo, hi, tc = it.get("cmc_min", 0), it.get("cmc_max", 99), it.get("type_contains")
            if any(type(v) is not int or v < 0 for v in (lo, hi)) or not (tc is None or isinstance(tc, str)):
                return None
            if role not in SPELL_ROLES and role not in TYPE_ROLES:
                return None
            if not isinstance(copies, int) or isinstance(copies, bool) or copies <= 0 or lo > hi:
                return None
            slots.append(Slot(role, lo, hi, copies, str(it.get("description") or describe(role, lo, hi))[:200],
                              (tc or "").strip() or None))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if not slots:
        return None
    for s, n in zip(slots, largest_remainder([s.copies for s in slots], MAIN_SIZE - lands)):
        s.copies = n
    return [s for s in slots if s.copies > 0]


def default_plan(archetype_hint: str) -> List[Slot]:
    spec = DEFAULT_PLANS.get((archetype_hint or "").strip().lower(), DEFAULT_PLANS["midrange"])
    lands = lands_for(archetype_hint)
    counts = largest_remainder([c for *_, c in spec], MAIN_SIZE - lands)
    return [Slot(role, lo, hi, n, describe(role, lo, hi)) for (role, lo, hi, _), n in zip(spec, counts)]


async def plan_with_llm(request_text: str, colors: List[str], archetype_hint: str) -> Plan:
    """Brew plan from one LLM call; the archetype default when that fails."""
    lands = lands_for(archetype_hint)
    slots = None
    if llm.is_configured():
        try:
            content = await asyncio.to_thread(
                llm.complete, PLAN_SYSTEM,
                f"Deck request: {request_text}\nColors: {', '.join(colors) or 'any'}\n"
                f"Archetype: {archetype_hint or 'unspecified'}\nNonland cards: {MAIN_SIZE - lands}",
                1500)
            slots = parse_llm_plan(content, lands)
        except Exception as e:
            logger.warning(f"LLM slot plan failed, using the default: {e}")
    return Plan(slots=slots or default_plan(archetype_hint), lands=lands)
