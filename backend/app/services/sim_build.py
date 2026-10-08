"""Playtesting a deck built from chat: queue the run, then (on the sim worker) improve
the deck by playing it with deck_search.playtest."""

import logging
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.queue import sim_worker_running
from app.models.conversation import Conversation
from app.models.simulation import SimulationRun
from app.services import forge, jev
from app.services.deck_fill import (
    MAX_COPIES, SIXTY_CARD_FORMATS, card_text, colors_of, fill_slot, requested_rows, slot_pool,
)
from app.services.deck_plan import RECENT, Slot, archetype_keys, is_land
from app.services.deck_search import BASIC_NAMES, Protect, SearchConfig, playtest
from app.services.gauntlet import GAUNTLET_SIZE, gauntlet
from app.services.sim_report import build_report
from app.services.sim_runs import Progress, SimError, enqueue, entries_of, main_of

logger = logging.getLogger(__name__)

BUILD_MINUTES = 10
NOT_RUNNING_NOTE = "Couldn't playtest this deck (the simulator isn't running), so this is the untested build."
REPLACE_QUESTION = ("Which card should replace the card in `deck.cut`? It underperformed when this deck was "
                    "playtested; prefer a card that helps against the decks in `deck.losing_to`.")
_FACES = re.compile(r"\s+//?\s+")


async def start_build(db: AsyncSession, deck: Dict[str, Any], assembly: Optional[Dict[str, Any]],
                      conversation_id: Optional[UUID], user_id: Optional[UUID],
                      format: str) -> Tuple[Optional[SimulationRun], Optional[str]]:
    """Queue a playtest of a deck Jev assembly built. Returns the run, or a note for
    the user when the simulator can't take it."""
    if not settings.FORGE_ENABLED or format not in SIXTY_CARD_FORMATS or not assembly:
        return None, None
    if not sim_worker_running():
        return None, NOT_RUNNING_NOTE
    archetypes = [assembly["reference"]] if assembly.get("reference") else list(assembly.get("relatives") or [])
    run = SimulationRun(
        id=uuid4(), user_id=user_id, conversation_id=conversation_id, kind="build", status="queued", format=format,
        deck=deck, games_per_matchup=SearchConfig().screen_games, stop_requested=False,
        options={"requested": list(assembly.get("requested") or []), "archetypes": archetypes,
                 "synergy": assembly.get("synergy"), "colors": list(assembly.get("colors") or [])},
        created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.add(run)
    await db.commit()
    try:
        enqueue(run.id)
    except Exception:
        run.status, run.error = "failed", "Couldn't start the playtest."
        await db.commit()
        raise
    return run, None


STAPLES_SQL = text(f"""
    WITH lists AS (
      SELECT d.id, d.main_deck
      FROM decklists d JOIN events e ON e.id = d.event_id
      WHERE {RECENT} AND lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))
    )
    SELECT lower(split_part(x->>'card_name', ' // ', 1)) AS k
    FROM lists CROSS JOIN LATERAL jsonb_array_elements(lists.main_deck) AS x
    GROUP BY 1
    HAVING COUNT(DISTINCT lists.id) * 2 > (SELECT COUNT(*) FROM lists)
""")


async def staples(db: AsyncSession, archetypes: Sequence[str], format: str, names: Sequence[str]) -> Set[str]:
    """The deck's cards that more than half of the archetypes' recent lists play."""
    if not archetypes:
        return set()
    rows = (await db.execute(STAPLES_SQL, {"format": format, "archetypes": archetype_keys(archetypes)})).all()
    keys = {r.k for r in rows}
    return {n for n in names if _FACES.split(n)[0].lower() in keys}


def make_protect(protected: Set[str], synergy: Optional[Dict[str, Any]], rows: Dict[str, Any]) -> Protect:
    """Never cut a protected card, or a synergy card when the cut leaves fewer than
    synergy["min"] copies of that type."""
    word = synergy["type_contains"].lower() if synergy else None

    def of_type(name: str) -> bool:
        return word is not None and name in rows and word in (rows[name].type_line or "").lower()

    def protect(main: Dict[str, int], cut: str) -> bool:
        if cut in protected:
            return True
        if of_type(cut):
            left = sum(q for n, q in main.items() if of_type(n)) - main[cut]
            return left < synergy["min"]
        return False
    return protect


async def replacement(db: AsyncSession, client, main: Dict[str, int], cut: str, losing: List[str],
                      colors: List[str], format: str, known: Set[str], rows: Dict[str, Any]) -> Optional[str]:
    """Jev's pick, from played cards like `cut` (role, mana value within one), for its place."""
    if client is None:
        return None
    if cut not in rows:
        rows.update({r.name: r for r in await requested_rows(db, [cut], format)})
    row = rows.get(cut)
    if row is None:
        return None
    role = next((r for r in (row.roles or []) if not r.startswith("land_")), None) or (
        "creature" if "Creature" in (row.type_line or "") else "noncreature")
    cmc = int(row.cmc or 0)
    slot = Slot(role, max(0, cmc - 1), cmc + 1, main[cut], f"a replacement for {row.name}: {row.type_line}")
    pool = [r for r in await slot_pool(db, slot, colors, format, list(main)) if forge.forge_name(r.name, known)]
    if not pool:
        return None
    state = {"deck": {"cards": [f"{q}x {n}" for n, q in main.items()], "cut": card_text(row),
                      "losing_to": list(losing)}}
    picks = await fill_slot(client, slot, pool, state, format, lambda name, rank: MAX_COPIES,
                            question=REPLACE_QUESTION)
    return picks[0][0] if picks else None


def merge_entries(entries: Sequence[Dict[str, Any]], main: Dict[str, int]) -> List[Dict[str, Any]]:
    """`main` as deck entries, keeping each existing entry (and its card object) with its
    quantity updated; new cards get bare {card_name, quantity} entries."""
    by_name = {e["card_name"]: e for e in entries}
    return [{**by_name.get(n, {}), "card_name": n, "quantity": q} for n, q in main.items() if q > 0]


async def write_back(db: AsyncSession, run: SimulationRun, main: Dict[str, int]) -> None:
    """Put the playtested main deck into the run's conversation, unless the user has
    changed the deck since the run started."""
    if not run.conversation_id or main == main_of(run.deck["main_deck"]):
        return
    conversation = await db.get(Conversation, run.conversation_id)
    deck = conversation.current_deck if conversation else None
    # ponytail: read-then-write without a row lock; a chat edit landing in between is lost
    if not deck or main_of(deck.get("main_deck") or []) != main_of(run.deck["main_deck"]):
        return
    conversation.current_deck = {**deck, "main_deck": merge_entries(deck["main_deck"], main)}


async def run_build(db: AsyncSession, run: SimulationRun) -> None:
    opts = run.options or {}
    opponents = await gauntlet(db, run.format)
    if not opponents:
        raise SimError("There are no recent decklists to playtest against.")
    seed_main = main_of(run.deck["main_deck"])
    known = forge.card_names()
    missing = forge.missing_cards(seed_main, known)
    rows = {r.name: r for r in await requested_rows(db, list(seed_main), run.format)}
    lands = {n for n, r in rows.items() if is_land(r.type_line)} | BASIC_NAMES
    protected = (set(opts.get("requested") or []) | set(missing)
                 | await staples(db, opts.get("archetypes") or [], run.format, list(seed_main)))
    protect = make_protect(protected, opts.get("synergy"), rows)
    colors = list(opts.get("colors") or colors_of(list(rows.values())))
    cfg = SearchConfig()
    # The baseline, one round and the confirmation; the search re-plans after each round.
    planned = len(opponents) * (cfg.screen_games * (1 + cfg.candidates) + 2 * cfg.confirm_games)
    progress = Progress(db, run, opponents, planned)
    await progress.deck(seed_main)
    for name in missing:
        await progress.event(f"Forge can't play [[{name}]] yet, so it was left out of testing and kept in your list.")
    matchups = [forge.Matchup(o.archetype, o.main) for o in opponents]

    baseline = [True]  # the first evaluation is the first draft's: tally it in the live table

    async def evaluate(main: Dict[str, int], games: int, seed: int):
        tally, baseline[0] = baseline[0], False

        async def counted(key, records):
            await progress.games(key, records, tally=tally)
        return await forge.play(main, matchups, games, seed, counted)

    async with jev.session(None) as client:
        async def find_add(main: Dict[str, int], cut: str, losing: List[str]) -> Optional[str]:
            try:
                async with db.begin_nested():  # a failed pick rolls back only this savepoint
                    added = await replacement(db, client, main, cut, losing, colors, run.format, known, rows)
                    if added and added not in rows:  # so the synergy floor counts it
                        rows.update({r.name: r for r in await requested_rows(db, [added], run.format)})
                    return added
            except Exception as e:  # a failed pick skips this swap, not the run
                logger.warning(f"[SIM] replacement for {cut} failed: {e}")
                return None
        result = await playtest(seed_main, opponents, lands, evaluate, find_add, protect, progress, cfg,
                                rng_seed=random.randrange(1 << 30))

    await db.refresh(run, ["status"])
    if run.status == "failed":  # the reaper gave up on this run while it played
        return
    run.error = None
    await write_back(db, run, result.main)
    run.final_deck = {**run.deck, "main_deck": entries_of(result.main)}
    run.report = build_report(result.final, [r for rs in result.records.values() for r in rs], result.main, lands,
                              missing, baseline=result.baseline, changes=result.changes, stopped=result.stopped)
    run.status = "stopped" if result.stopped == "user" else "completed"
    progress.data["stage"] = "Stopped" if result.stopped == "user" else "Done"
    progress.data["deck"] = entries_of(result.main)
    await progress.save()
