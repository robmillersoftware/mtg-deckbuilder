"""Forge simulations: test a deck against the meta, read live progress and the report, stop a run."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.auth import get_current_user
from app.core.queue import sim_queue, sim_worker_running
from app.db.session import get_db
from app.models.deck import Deck
from app.models.meta import MetaSnapshot
from app.models.simulation import SimulationRun
from app.models.user import User
from app.services.deck_fill import SIXTY_CARD_FORMATS
from app.services.gauntlet import GAUNTLET_SIZE
from app.services.sim_runs import (MAX_GAMES, MAX_OPPONENTS, MAX_TOTAL_GAMES, MIN_GAMES, TEST_GAMES,
                                  enqueue, queue_position)

router = APIRouter()

STALE_AFTER = timedelta(minutes=15)
STALLED = "The playtest stopped responding before it finished."


def reap_if_stale(run: SimulationRun) -> bool:
    """Fail a running run that has not reported progress for STALE_AFTER (its job died)."""
    if run.status == "running" and run.updated_at and datetime.utcnow() - run.updated_at > STALE_AFTER:
        run.status, run.error = "failed", STALLED
        return True
    return False


NOT_RUNNING = "The simulator isn't running right now, so this deck can't be tested. Try again in a few minutes."
TOO_MANY_OPPONENTS = f"Choose at most {MAX_OPPONENTS} opponents."
TOO_MANY_GAMES = "That's too many games for one test: choose fewer opponents or games."


class SimulationCreate(BaseModel):
    deck_id: Optional[UUID] = None
    deck: Optional[Dict[str, Any]] = None  # {name, main_deck: [{card_name, quantity}], sideboard}
    format: str = "standard"
    opponents: Optional[List[str]] = None  # archetype names; none = the top meta decks
    games: int = Field(TEST_GAMES, ge=MIN_GAMES, le=MAX_GAMES)


class SimulationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    kind: str
    status: str
    format: str
    deck: Dict[str, Any]
    opponents: Optional[List[str]] = None
    games_per_matchup: int
    progress: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    final_deck: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stop_requested: bool = False
    queue_position: Optional[int] = None
    created_at: datetime
    updated_at: datetime


def to_response(run: SimulationRun) -> SimulationResponse:
    resp = SimulationResponse.model_validate(run)
    resp.queue_position = queue_position(run.id) if run.status == "queued" else None
    return resp


def visible(run: Optional[SimulationRun], user: Optional[User]) -> bool:
    return run is not None and (run.user_id is None or (user is not None and run.user_id == user.id))


async def _deck_for(body: SimulationCreate, db: AsyncSession, user: Optional[User]) -> Dict[str, Any]:
    if body.deck_id:
        deck = await db.get(Deck, body.deck_id)
        shared = deck is not None and deck.visibility in ("public", "unlisted")  # NULL counts as private
        if deck is None or (not shared and (user is None or deck.owner_id != user.id)):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "That deck wasn't found.")
        return {"name": deck.name, "main_deck": deck.main_deck or [], "sideboard": deck.sideboard or []}
    if body.deck and body.deck.get("main_deck"):
        for e in body.deck["main_deck"]:
            try:
                ok = isinstance(e["card_name"], str) and e["card_name"].strip() and int(e["quantity"]) >= 1
            except (KeyError, TypeError, ValueError):
                ok = False
            if not ok:
                raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                    "Each card needs a name and a quantity of at least 1.")
        return {"name": body.deck.get("name") or "Untitled deck", "main_deck": body.deck["main_deck"],
                "sideboard": body.deck.get("sideboard") or []}
    raise HTTPException(status.HTTP_400_BAD_REQUEST, "Choose a deck with a main deck to test.")


@router.post("", response_model=SimulationResponse)
async def create_simulation(body: SimulationCreate, db: AsyncSession = Depends(get_db),
                            current_user: Optional[User] = Depends(get_current_user)):
    if body.format not in SIXTY_CARD_FORMATS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Simulation supports 60-card formats only.")
    if len(body.opponents or []) > MAX_OPPONENTS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, TOO_MANY_OPPONENTS)
    if body.games * len(body.opponents or [None] * GAUNTLET_SIZE) > MAX_TOTAL_GAMES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, TOO_MANY_GAMES)
    deck = await _deck_for(body, db, current_user)
    if not sim_worker_running():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, NOT_RUNNING)
    run = SimulationRun(id=uuid4(), user_id=current_user.id if current_user else None, kind="test", status="queued",
                        format=body.format, deck=deck, opponents=body.opponents or None,
                        games_per_matchup=body.games, stop_requested=False,
                        created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.add(run)
    await db.commit()
    enqueue(run.id)
    return to_response(run)


@router.get("", response_model=List[SimulationResponse])
async def list_simulations(db: AsyncSession = Depends(get_db),
                           current_user: Optional[User] = Depends(get_current_user)):
    if current_user is None:
        return []
    runs = (await db.execute(select(SimulationRun).where(SimulationRun.user_id == current_user.id)
                             .order_by(SimulationRun.created_at.desc()).limit(20))).scalars().all()
    if any([reap_if_stale(r) for r in runs]):
        await db.commit()
    return [to_response(r) for r in runs]


@router.get("/archetypes", response_model=List[str])
async def list_archetypes(format: str = "standard", db: AsyncSession = Depends(get_db)):
    """Meta archetypes to choose as opponents, by share."""
    latest = select(func.max(MetaSnapshot.snapshot_date)).where(MetaSnapshot.format == format).scalar_subquery()
    snaps = (await db.execute(select(MetaSnapshot).where(MetaSnapshot.format == format,
                                                         MetaSnapshot.snapshot_date == latest)
                              .order_by(MetaSnapshot.meta_percentage.desc()))).scalars().all()
    seen, names = set(), []
    for s in snaps:
        name = s.archetype.strip()
        if name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names


@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(simulation_id: UUID, db: AsyncSession = Depends(get_db),
                         current_user: Optional[User] = Depends(get_current_user)):
    run = await db.get(SimulationRun, simulation_id)
    if not visible(run, current_user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Simulation not found")
    if reap_if_stale(run):
        await db.commit()
    return to_response(run)


@router.post("/{simulation_id}/stop", response_model=SimulationResponse)
async def stop_simulation(simulation_id: UUID, db: AsyncSession = Depends(get_db),
                          current_user: Optional[User] = Depends(get_current_user)):
    """Stop after the current games; a queued run stops right away."""
    run = await db.get(SimulationRun, simulation_id)
    if not visible(run, current_user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Simulation not found")
    if reap_if_stale(run):
        await db.commit()
    if run.status in ("completed", "failed", "stopped"):
        return to_response(run)
    run.stop_requested = True
    if run.status == "queued":
        sim_queue.remove(str(run.id))
        run.status = "stopped"
    await db.commit()
    return to_response(run)
