"""A Forge simulation run: testing a deck against the meta, or playtesting a build."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.db.session import Base


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)  # anonymous allowed
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True)
    kind = Column(String(10), nullable=False)  # test | build
    status = Column(String(20), nullable=False, default="queued")  # queued, running, completed, failed, stopped
    format = Column(String(50), nullable=False, default="standard")
    deck = Column(JSONB, nullable=False)  # {name, main_deck: [{card_name, quantity}], sideboard}
    opponents = Column(JSONB, nullable=True)  # chosen archetype names; null = the meta gauntlet
    games_per_matchup = Column(Integer, nullable=False)
    options = Column(JSONB, nullable=True)  # build: requested, archetypes, synergy, colors
    progress = Column(JSONB, nullable=True)  # stage, games, live matchups, events, current deck
    report = Column(JSONB, nullable=True)
    final_deck = Column(JSONB, nullable=True)  # build: the playtested deck
    stop_requested = Column(Boolean, nullable=False, default=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (Index("idx_simulation_runs_user", "user_id", "created_at"),)
