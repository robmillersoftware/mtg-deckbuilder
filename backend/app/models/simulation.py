"""
Simulation run model for persisting game simulations.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.session import Base


class SimulationRun(Base):
    """
    Persisted simulation run that tracks game simulations between decks.
    """

    __tablename__ = "simulation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Status: pending, running, completed, failed
    status = Column(String(20), nullable=False, default="pending")

    # Configuration
    your_deck_id = Column(UUID(as_uuid=True), ForeignKey("decks.id", ondelete="SET NULL"), nullable=True)
    your_deck_name = Column(String(255), nullable=False)
    your_deck_snapshot = Column(JSONB, nullable=True)  # Snapshot of deck at simulation time

    # Single opponent (2-player) - kept for backward compatibility
    opponent_deck_name = Column(String(255), nullable=False)
    opponent_archetype = Column(String(255), nullable=True)
    opponent_deck_snapshot = Column(JSONB, nullable=True)  # Snapshot of opponent deck

    # Multiplayer support (3-4 players)
    num_players = Column(Integer, nullable=False, default=2)  # 2-4 players
    opponent_deck_names = Column(JSONB, nullable=True)  # ["Deck 1", "Deck 2", "Deck 3"]
    opponent_archetypes = Column(JSONB, nullable=True)  # ["Archetype 1", "Archetype 2", ...]
    opponent_deck_snapshots = Column(JSONB, nullable=True)  # [snapshot1, snapshot2, ...]

    format = Column(String(50), nullable=False, default="standard")
    num_games = Column(Integer, nullable=False, default=5)
    include_sideboard_games = Column(Integer, nullable=False, default=1)  # Boolean as int

    # Progress tracking
    games_completed = Column(Integer, nullable=False, default=0)
    current_game_turn = Column(Integer, nullable=True)

    # Results (populated when completed)
    your_wins = Column(Integer, nullable=True)  # 2-player wins or 1st place finishes
    opponent_wins = Column(Integer, nullable=True)  # 2-player only
    # Multiplayer results
    first_place_count = Column(Integer, nullable=True)  # How many times you finished 1st
    your_placement_avg = Column(Numeric(3, 2), nullable=True)  # Average finishing position
    win_rate = Column(Numeric(5, 4), nullable=True)
    average_game_length = Column(Numeric(5, 2), nullable=True)
    matchup_assessment = Column(String(20), nullable=True)  # favored, even, unfavored

    # Detailed results stored as JSON
    games = Column(JSONB, nullable=True)  # List of game results
    key_cards_for_you = Column(JSONB, nullable=True)
    key_cards_against_you = Column(JSONB, nullable=True)
    sideboard_guide = Column(JSONB, nullable=True)
    strategic_advice = Column(JSONB, nullable=True)
    mulligan_advice = Column(String(1000), nullable=True)
    deck_recommendations = Column(JSONB, nullable=True)  # List of deck improvement suggestions

    # Error tracking
    error_message = Column(String(1000), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", backref="simulation_runs")
    your_deck = relationship("Deck", foreign_keys=[your_deck_id])

    __table_args__ = (
        Index("idx_simulation_runs_user_status", "user_id", "status"),
        Index("idx_simulation_runs_user_created", "user_id", "created_at"),
    )

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "id": str(self.id),
            "status": self.status,
            "your_deck_id": str(self.your_deck_id) if self.your_deck_id else None,
            "your_deck_name": self.your_deck_name,
            "opponent_deck_name": self.opponent_deck_name,
            "opponent_archetype": self.opponent_archetype,
            # Multiplayer fields
            "num_players": self.num_players,
            "opponent_deck_names": self.opponent_deck_names,
            "opponent_archetypes": self.opponent_archetypes,
            "format": self.format,
            "num_games": self.num_games,
            "include_sideboard_games": bool(self.include_sideboard_games),
            "games_completed": self.games_completed,
            "current_game_turn": self.current_game_turn,
            "your_wins": self.your_wins,
            "opponent_wins": self.opponent_wins,
            # Multiplayer results
            "first_place_count": self.first_place_count,
            "your_placement_avg": float(self.your_placement_avg) if self.your_placement_avg else None,
            "win_rate": float(self.win_rate) if self.win_rate else None,
            "average_game_length": float(self.average_game_length) if self.average_game_length else None,
            "matchup_assessment": self.matchup_assessment,
            "games": self.games,
            "key_cards_for_you": self.key_cards_for_you,
            "key_cards_against_you": self.key_cards_against_you,
            "sideboard_guide": self.sideboard_guide,
            "strategic_advice": self.strategic_advice,
            "mulligan_advice": self.mulligan_advice,
            "deck_recommendations": self.deck_recommendations,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
