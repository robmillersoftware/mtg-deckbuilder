"""
Simulation API routes for running game simulations between decks.
"""

from typing import List
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
)

router = APIRouter()


@router.post("", response_model=MatchupAnalysisResult)
async def run_simulation(
    request: SimulationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Run a game simulation between two decks.

    Simulates multiple games and returns:
    - Win rate statistics
    - Key cards analysis
    - Sideboard recommendations
    - Strategic advice
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
    Quick simulation against a meta archetype.

    Automatically fetches a representative tournament decklist
    for the specified archetype.
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
