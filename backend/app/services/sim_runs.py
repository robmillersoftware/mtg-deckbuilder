"""Simulation runs: queueing, live progress, and the 'test a deck' job. Builds
(sim_build.run_build) share Progress and execute()."""

import copy
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.queue import sim_queue
from app.db.session import async_session_factory
from app.models.simulation import SimulationRun
from app.services import forge
from app.services.deck_search import Stopped
from app.services.forge import GameRecord
from app.services.gauntlet import Opponent, gauntlet
from app.services.sim_report import build_report
from app.services.sim_stats import MatchupStats, matchup_stats

logger = logging.getLogger(__name__)

MIN_GAMES, MAX_GAMES, TEST_GAMES = 10, 100, 50
# At about 0.6 games/s, 1000 games is about 28 minutes, inside the job timeout.
MAX_OPPONENTS, MAX_TOTAL_GAMES = 10, 1000
MAX_EVENTS = 60
JOB_TIMEOUT_S = 3600


class SimError(ValueError):
    """A reason the run can't go on, worded for the user to follow "Couldn't finish
    playtesting: ", so it starts lowercase."""


def main_of(entries: Sequence[Dict]) -> Dict[str, int]:
    main: Dict[str, int] = {}
    for e in entries or []:
        main[e["card_name"]] = main.get(e["card_name"], 0) + int(e["quantity"])
    return main


def entries_of(main: Dict[str, int]) -> List[Dict]:
    return [{"card_name": n, "quantity": q} for n, q in main.items() if q > 0]


LANDS_SQL = text("""
    SELECT DISTINCT name FROM cards
    WHERE name = ANY(CAST(:names AS varchar[]))
      AND split_part(coalesce(type_line, ''), ' // ', 1) LIKE '%Land%'
""")


async def land_names(db: AsyncSession, names: Sequence[str]) -> Set[str]:
    return {r.name for r in (await db.execute(LANDS_SQL, {"names": list(names)})).all()}


def enqueue(run_id: UUID) -> None:
    sim_queue.enqueue("app.jobs.sim_tasks.run_simulation", str(run_id), job_id=str(run_id),
                      job_timeout=JOB_TIMEOUT_S)


def queue_position(run_id: UUID) -> Optional[int]:
    ids = sim_queue.job_ids
    return ids.index(str(run_id)) + 1 if str(run_id) in ids else None


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


class Progress:
    """The live view of a run, written to run.progress as it changes."""

    def __init__(self, db: AsyncSession, run: SimulationRun, opponents: Sequence[Opponent], games_planned: int):
        self.db, self.run = db, run
        self.data = {
            "stage": "Starting", "games_done": 0, "games_planned": games_planned, "started_at": _now(),
            "matchups": [{"opponent": o.archetype, "share": o.share, "wins": 0, "losses": 0, "draws": 0}
                         for o in opponents],
            "events": [], "deck": None,
        }

    async def save(self) -> None:
        self.run.progress = copy.deepcopy(self.data)  # a new object, so SQLAlchemy sees the change
        self.run.updated_at = datetime.utcnow()
        await self.db.commit()

    async def stage(self, text_: str) -> None:
        self.data["stage"] = text_
        await self.save()

    async def event(self, text_: str, kind: str = "info") -> None:
        self.data["events"] = (self.data["events"] + [{"at": _now(), "text": text_, "kind": kind}])[-MAX_EVENTS:]
        await self.save()

    async def games(self, key: str, records: Sequence[GameRecord], tally: bool = True) -> None:
        """Count finished games (and, for the deck being reported, tally them); raise
        Stopped when the user asked to stop."""
        self.data["games_done"] += len(records)
        if tally:
            row = next(m for m in self.data["matchups"] if m["opponent"] == key)
            for r in records:
                field = "wins" if r.winner == forge.TESTED else "losses" if r.winner == forge.OPPONENT else "draws"
                row[field] += 1
        await self.save()
        if await self.stop_requested():
            raise Stopped()

    def plan(self, games_left: int) -> None:
        """Re-estimate the games still to play (shown on the next save)."""
        self.data["games_planned"] = self.data["games_done"] + games_left

    async def set_matchups(self, stats: Sequence[MatchupStats]) -> None:
        self.data["matchups"] = [{"opponent": m.opponent, "share": m.share, "wins": m.wins, "losses": m.losses,
                                  "draws": m.draws} for m in stats]
        await self.save()

    async def deck(self, main: Dict[str, int]) -> None:
        self.data["deck"] = entries_of(main)
        await self.save()

    async def stop_requested(self) -> bool:
        await self.db.refresh(self.run, ["stop_requested"])
        return bool(self.run.stop_requested)


async def run_test(db: AsyncSession, run: SimulationRun) -> None:
    opponents = await gauntlet(db, run.format, run.opponents)
    if not opponents:
        raise SimError("there are no recent decklists for these opponents to play against.")
    main = main_of(run.deck["main_deck"])
    known = forge.card_names()
    missing = forge.missing_cards(main, known)
    lands = await land_names(db, list(main))
    progress = Progress(db, run, opponents, run.games_per_matchup * len(opponents))
    await progress.stage(f"Playing {run.games_per_matchup} games against each of {len(opponents)} decks")
    for name in missing:
        await progress.event(f"Forge can't play [[{name}]] yet, so it was left out of testing.")

    played: Dict[str, List[GameRecord]] = {o.archetype: [] for o in opponents}

    async def on_games(key: str, records: List[GameRecord]) -> None:
        played[key].extend(records)
        await progress.games(key, records)

    stopped = None
    try:
        await forge.play(main, [forge.Matchup(o.archetype, o.main) for o in opponents], run.games_per_matchup,
                         random.randrange(1 << 30), on_games)
    except Stopped:
        stopped = "user"
    stats = [matchup_stats(o.archetype, o.share, played[o.archetype]) for o in opponents]
    await db.refresh(run, ["status"])
    if run.status == "failed":  # the reaper gave up on this run while it played
        return
    run.error = None
    run.report = build_report(stats, [r for rs in played.values() for r in rs], main, lands, missing,
                              stopped=stopped)
    run.status = "stopped" if stopped else "completed"
    progress.data["stage"] = "Stopped" if stopped else "Done"
    await progress.save()


async def execute(run_id: UUID) -> None:
    """Run a queued simulation (the sim worker's job)."""
    async with async_session_factory() as db:
        run = await db.get(SimulationRun, run_id)
        if run is None:
            return
        if run.stop_requested:
            run.status = "stopped"
            await db.commit()
            return
        run.status = "running"
        await db.commit()
        try:
            if not forge.available():
                raise SimError("the simulator isn't set up yet.")
            if run.kind == "build":
                from app.services.sim_build import run_build
                await run_build(db, run)
            else:
                await run_test(db, run)
        except Exception as e:
            logger.exception(f"[SIM] run {run_id} failed")
            await db.rollback()
            run = await db.get(SimulationRun, run_id)
            run.status = "failed"
            run.error = ("Couldn't finish playtesting: the simulator failed while playing games."
                         if isinstance(e, forge.ForgeError)
                         else f"Couldn't finish playtesting: {e}" if isinstance(e, SimError)
                         else "Something went wrong while playtesting.")
            await db.commit()
