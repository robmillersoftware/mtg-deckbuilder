"""
Simulation API routes for running game simulations between decks.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.api.deps.auth import get_current_user
from app.models.user import User
from app.services.game_simulator import GameSimulator
from app.schemas.simulation import (
    SimulationRequest,
    MatchupAnalysisResult,
    QuickSimRequest,
    SimulationRunResponse,
    SimulationRunListResponse,
    GameResult,
    DeckRecommendation,
)

router = APIRouter()


# =============================================================================
# Persistent Simulation Runs (Background Execution)
# =============================================================================


@router.get("/runs", response_model=SimulationRunListResponse)
async def list_simulation_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List simulation runs for the current user.
    """
    simulator = GameSimulator(db)
    runs, total = await simulator.list_simulation_runs(
        user_id=current_user.id,
        limit=limit,
        offset=offset,
        status=status,
    )

    return SimulationRunListResponse(
        items=[_run_to_response(run) for run in runs],
        total=total,
    )


@router.post("/runs", response_model=SimulationRunResponse)
async def create_simulation_run(
    request: QuickSimRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new simulation run that executes in the background.

    For 2-player games, provide opponent_archetype.
    For multiplayer (3-4 players), provide opponent_archetypes list and set num_players.

    Returns immediately with the simulation run ID. Poll GET /runs/{id}
    to check status and get results when complete.
    """
    simulator = GameSimulator(db)

    try:
        # Create the simulation run record
        sim_run = await simulator.create_simulation_run(
            user_id=current_user.id,
            deck_id=request.deck_id,
            opponent_archetype=request.opponent_archetype,
            num_games=request.num_games,
            opponent_archetypes=request.opponent_archetypes,
            num_players=request.num_players,
        )

        # Queue background execution
        background_tasks.add_task(
            _execute_simulation_background,
            simulation_id=sim_run.id,
        )

        return _run_to_response(sim_run)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/runs/{simulation_id}", response_model=SimulationRunResponse)
async def get_simulation_run(
    simulation_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get a simulation run by ID.
    """
    simulator = GameSimulator(db)
    sim_run = await simulator.get_simulation_run(simulation_id, current_user.id)

    if not sim_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation run not found",
        )

    return _run_to_response(sim_run)


@router.delete("/runs/{simulation_id}")
async def delete_simulation_run(
    simulation_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a simulation run.
    """
    simulator = GameSimulator(db)
    deleted = await simulator.delete_simulation_run(simulation_id, current_user.id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation run not found",
        )

    return {"message": "Simulation run deleted"}


@router.post("/runs/{simulation_id}/retry", response_model=SimulationRunResponse)
async def retry_simulation_run(
    simulation_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retry a failed simulation run.

    Only works for simulations with status 'failed'.
    Resets the simulation and queues it for background execution.
    """
    simulator = GameSimulator(db)
    sim_run = await simulator.retry_simulation_run(simulation_id, current_user.id)

    if not sim_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation run not found or not in failed status",
        )

    # Queue background execution
    background_tasks.add_task(
        _execute_simulation_background,
        simulation_id=sim_run.id,
    )

    return _run_to_response(sim_run)


# =============================================================================
# Synchronous Simulation (Legacy - waits for completion)
# =============================================================================


@router.post("", response_model=MatchupAnalysisResult)
async def run_simulation(
    request: SimulationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Run a game simulation between two decks (synchronous).

    This endpoint waits for the simulation to complete before returning.
    For background execution, use POST /simulation/runs instead.
    """
    simulator = GameSimulator(db)

    try:
        result = await simulator.simulate_match(
            your_deck=request.your_deck,
            opponent_deck=request.opponent_deck,
            num_games=request.num_games,
            include_sideboard_games=request.include_sideboard_games,
            format=request.format,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}",
        )


@router.post("/vs-archetype", response_model=MatchupAnalysisResult)
async def simulate_vs_archetype(
    request: QuickSimRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Quick simulation against a meta archetype (synchronous).

    This endpoint waits for the simulation to complete before returning.
    For background execution, use POST /simulation/runs instead.
    """
    simulator = GameSimulator(db)

    try:
        result = await simulator.simulate_vs_archetype(
            deck_id=request.deck_id,
            opponent_archetype=request.opponent_archetype,
            num_games=request.num_games,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}",
        )


@router.get("/archetypes", response_model=List[str])
async def get_available_archetypes(
    format: str = Query("standard", description="Format to filter archetypes by"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get list of meta archetypes available for simulation.

    Returns archetype names that have tournament decklists in the database
    for the specified format.
    """
    simulator = GameSimulator(db)
    return await simulator.get_meta_archetypes(format=format)


# =============================================================================
# Helper Functions
# =============================================================================


def _run_to_response(sim_run) -> SimulationRunResponse:
    """Convert a SimulationRun model to a response schema."""
    games = None
    if sim_run.games:
        games = [GameResult(**g) for g in sim_run.games]

    return SimulationRunResponse(
        id=sim_run.id,
        status=sim_run.status,
        your_deck_id=sim_run.your_deck_id,
        your_deck_name=sim_run.your_deck_name,
        opponent_deck_name=sim_run.opponent_deck_name,
        opponent_archetype=sim_run.opponent_archetype,
        # Multiplayer fields
        num_players=sim_run.num_players,
        opponent_deck_names=sim_run.opponent_deck_names,
        opponent_archetypes=sim_run.opponent_archetypes,
        format=sim_run.format,
        num_games=sim_run.num_games,
        include_sideboard_games=bool(sim_run.include_sideboard_games),
        games_completed=sim_run.games_completed,
        current_game_turn=sim_run.current_game_turn,
        your_wins=sim_run.your_wins,
        opponent_wins=sim_run.opponent_wins,
        # Multiplayer results
        your_placement_avg=float(sim_run.your_placement_avg) if sim_run.your_placement_avg else None,
        first_place_count=sim_run.first_place_count,
        win_rate=float(sim_run.win_rate) if sim_run.win_rate else None,
        average_game_length=float(sim_run.average_game_length) if sim_run.average_game_length else None,
        matchup_assessment=sim_run.matchup_assessment,
        games=games,
        key_cards_for_you=sim_run.key_cards_for_you,
        key_cards_against_you=sim_run.key_cards_against_you,
        sideboard_guide=sim_run.sideboard_guide,
        strategic_advice=sim_run.strategic_advice,
        mulligan_advice=sim_run.mulligan_advice,
        deck_recommendations=[DeckRecommendation(**r) for r in sim_run.deck_recommendations] if sim_run.deck_recommendations else None,
        error_message=sim_run.error_message,
        created_at=sim_run.created_at,
        started_at=sim_run.started_at,
        completed_at=sim_run.completed_at,
    )


async def _execute_simulation_background(simulation_id: UUID):
    """
    Execute a simulation in the background.
    This runs in a separate task after the request returns.
    """
    from app.db.session import async_session_factory

    async with async_session_factory() as db:
        simulator = GameSimulator(db)
        try:
            await simulator.execute_simulation_run(simulation_id)
        except Exception as e:
            # Error is already logged and saved to the run record
            pass
