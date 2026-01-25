from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class DeckInput(BaseModel):
    """Deck input for simulation - either a deck ID or raw decklist."""
    deck_id: Optional[UUID] = None
    main_deck: Optional[List[Dict[str, Any]]] = None  # [{card_name, quantity}]
    sideboard: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = "Unknown Deck"


class SimulationRequest(BaseModel):
    """Request to simulate games between two decks."""
    your_deck: DeckInput
    opponent_deck: DeckInput
    num_games: int = Field(default=5, ge=1, le=20)
    include_sideboard_games: bool = True  # Play games 2-3 with sideboarding
    format: str = "standard"


class GameAction(BaseModel):
    """A single action taken during a game."""
    turn: int
    player: str  # "you" or "opponent"
    phase: str  # "main1", "combat", "main2", etc.
    action: str  # Description of what happened
    cards_involved: List[str] = []


class GameResult(BaseModel):
    """Result of a single simulated game."""
    game_number: int
    winner: str  # "you" or "opponent"
    turns: int
    your_life: int
    opponent_life: int
    win_condition: str  # "damage", "decked", "concede", etc.
    key_moments: List[str]  # Important plays that swung the game
    your_key_cards: List[str]  # Cards that performed well
    opponent_key_cards: List[str]
    sideboard_in: Optional[List[str]] = None  # Cards you sided in (games 2-3)
    sideboard_out: Optional[List[str]] = None


class MatchupAnalysisResult(BaseModel):
    """Aggregated analysis from multiple game simulations."""
    your_deck_name: str
    opponent_deck_name: str
    games_played: int
    your_wins: int
    opponent_wins: int
    win_rate: float
    average_game_length: float

    # Strategic insights
    matchup_assessment: str  # "favored", "even", "unfavored"
    key_cards_for_you: List[Dict[str, Any]]  # [{card, importance, reason}]
    key_cards_against_you: List[Dict[str, Any]]

    # Sideboard recommendations
    sideboard_guide: Dict[str, List[str]]  # {"in": [...], "out": [...]}

    # Play pattern advice
    strategic_advice: List[str]
    mulligan_advice: str

    # Individual game summaries
    games: List[GameResult]


class SimulationStatus(BaseModel):
    """Status of a running simulation."""
    simulation_id: UUID
    status: str  # "pending", "running", "completed", "failed"
    games_completed: int
    total_games: int
    current_game_turn: Optional[int] = None
    result: Optional[MatchupAnalysisResult] = None
    error: Optional[str] = None


class QuickSimRequest(BaseModel):
    """Quick simulation against a meta archetype."""
    deck_id: UUID
    opponent_archetype: str  # e.g., "Mono-Red Aggro", "Azorius Control"
    num_games: int = Field(default=5, ge=1, le=10)
