from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class DeckRecommendation(BaseModel):
    """A specific recommendation for improving the deck."""
    category: str  # "add_cards", "remove_cards", "adjust_quantities", "sideboard", "strategy"
    priority: str  # "high", "medium", "low"
    suggestion: str  # The actual recommendation text
    cards_mentioned: List[str] = []  # Specific cards referenced
    reasoning: str  # Why this change would help


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
    winner: str  # "you" or "opponent" (or "opponent_1", "opponent_2", "opponent_3" for multiplayer)
    turns: int
    your_life: int
    opponent_life: int  # For 2-player games
    # For multiplayer games (Commander), track all life totals
    life_totals: Optional[Dict[str, int]] = None  # {"you": 40, "opponent_1": 35, ...}
    elimination_order: Optional[List[str]] = None  # Order players were eliminated
    win_condition: str  # "damage", "decked", "concede", "commander_damage", etc.
    key_moments: List[str]  # Important plays that swung the game
    your_key_cards: List[str]  # Cards that performed well
    opponent_key_cards: List[str]  # For 2-player or combined opponents
    opponent_key_cards_by_player: Optional[Dict[str, List[str]]] = None  # Per-opponent in multiplayer
    sideboard_in: Optional[List[str]] = None  # Cards you sided in (games 2-3)
    sideboard_out: Optional[List[str]] = None


class MatchupAnalysisResult(BaseModel):
    """Aggregated analysis from multiple game simulations."""
    your_deck_name: str
    opponent_deck_name: str  # For 2-player or combined description
    opponent_deck_names: Optional[List[str]] = None  # For multiplayer
    num_players: int = 2
    games_played: int
    your_wins: int
    opponent_wins: int  # For 2-player
    # Multiplayer stats
    first_place_count: Optional[int] = None  # Games where you finished 1st
    your_placement_avg: Optional[float] = None  # Average finishing position
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

    # Deck improvement recommendations
    deck_recommendations: Optional[List[DeckRecommendation]] = None

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
    """Quick simulation against meta archetype(s)."""
    deck_id: UUID
    opponent_archetype: Optional[str] = None  # e.g., "Mono-Red Aggro" (for 2-player)
    opponent_archetypes: Optional[List[str]] = None  # For multiplayer (up to 3 opponents)
    num_games: int = Field(default=5, ge=1, le=10)
    num_players: int = Field(default=2, ge=2, le=4)  # 2-4 players (4 for Commander)


class SimulationRunResponse(BaseModel):
    """Response for a simulation run."""
    id: UUID
    status: str
    your_deck_id: Optional[UUID] = None
    your_deck_name: str
    opponent_deck_name: str  # For 2-player or combined name
    opponent_archetype: Optional[str] = None  # For 2-player
    # Multiplayer support
    num_players: int = 2
    opponent_deck_names: Optional[List[str]] = None  # For multiplayer
    opponent_archetypes: Optional[List[str]] = None  # For multiplayer
    format: str
    num_games: int
    include_sideboard_games: bool
    games_completed: int
    current_game_turn: Optional[int] = None

    # Results (when completed)
    your_wins: Optional[int] = None
    opponent_wins: Optional[int] = None  # For 2-player
    # Multiplayer results
    your_placement_avg: Optional[float] = None  # Average finishing position (1st = best)
    first_place_count: Optional[int] = None  # How many games you won outright
    win_rate: Optional[float] = None
    average_game_length: Optional[float] = None
    matchup_assessment: Optional[str] = None
    games: Optional[List[GameResult]] = None
    key_cards_for_you: Optional[List[Dict[str, Any]]] = None
    key_cards_against_you: Optional[List[Dict[str, Any]]] = None
    sideboard_guide: Optional[Dict[str, List[str]]] = None
    strategic_advice: Optional[List[str]] = None
    mulligan_advice: Optional[str] = None
    deck_recommendations: Optional[List[DeckRecommendation]] = None

    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SimulationRunListResponse(BaseModel):
    """List of simulation runs."""
    items: List[SimulationRunResponse]
    total: int
