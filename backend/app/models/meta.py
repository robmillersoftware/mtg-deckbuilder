import uuid
from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Date,
    DateTime,
    Numeric,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from app.db.session import Base


class Event(Base):
    """
    Tournament event from mtgtop8.
    """

    __tablename__ = "events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(50), nullable=False, default="mtgtop8")
    source_id = Column(String(100), nullable=True)
    mtgtop8_id = Column(String(100), nullable=True, index=True)  # Alias for source_id
    name = Column(String(255), nullable=False)
    event_name = Column(String(255), nullable=True)  # Deprecated, use name
    date = Column(Date, nullable=False)
    event_date = Column(Date, nullable=True)  # Deprecated, use date
    format = Column(String(50), nullable=False, default="standard")
    num_players = Column(Integer, nullable=True)
    source_url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    decklists = relationship("Decklist", back_populates="event", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_event_source"),
        Index("idx_events_format_date", "format", "date"),
    )


class Decklist(Base):
    """
    Tournament decklist from mtgtop8.
    """

    __tablename__ = "decklists"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(
        UUID(as_uuid=True),
        ForeignKey("events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    mtgtop8_deck_id = Column(String(100), nullable=True, index=True)
    player_name = Column(String(255), nullable=True)
    finish_position = Column(String(50), nullable=True)  # "1st", "2nd", "Top 4", etc.
    placement = Column(Integer, nullable=True)  # Numeric placement
    archetype = Column(String(100), nullable=True, index=True)
    archetype_tags = Column(ARRAY(String(50)), nullable=True)

    # Deck contents as JSON arrays of {card_name, quantity}
    main_deck = Column(JSONB, nullable=False, default=list)
    sideboard = Column(JSONB, nullable=False, default=list)

    source_url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    event = relationship("Event", back_populates="decklists")


class MetaSnapshot(Base):
    """
    Meta statistics snapshot computed periodically.
    """

    __tablename__ = "meta_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    format = Column(String(50), nullable=False, default="standard")
    snapshot_date = Column(Date, nullable=False)
    archetype = Column(String(100), nullable=False)
    meta_percentage = Column(Numeric(5, 2), nullable=True)
    sample_size = Column(Integer, nullable=True)
    avg_finish = Column(Numeric(4, 2), nullable=True)
    key_cards = Column(ARRAY(String(255)), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("format", "snapshot_date", "archetype", name="uq_meta_snapshot"),
        Index("idx_meta_format_date", "format", "snapshot_date"),
    )


class ArchetypeTemplate(Base):
    """
    Aggregated role distributions for deck archetypes.
    Computed from tournament decklists to guide deck generation.
    """

    __tablename__ = "archetype_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    archetype_category = Column(String(50), nullable=False)  # aggro, midrange, control, combo
    format = Column(String(50), nullable=False, default="standard")

    # Sample info
    sample_size = Column(Integer, nullable=False)  # Number of decklists analyzed
    computed_at = Column(DateTime, default=datetime.utcnow)

    # Land counts
    avg_lands = Column(Numeric(4, 1), nullable=False)
    avg_nonlands = Column(Numeric(4, 1), nullable=False)

    # Role distributions as JSONB: {role_name: avg_count, ...}
    role_distribution = Column(JSONB, nullable=False, default=dict)

    # Detailed archetype breakdown: {specific_archetype: count, ...}
    archetype_breakdown = Column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint("archetype_category", "format", name="uq_archetype_template"),
        Index("idx_archetype_template_format", "format"),
    )


class CardCooccurrence(Base):
    """
    Card co-occurrence matrix for synergy recommendations.
    """

    __tablename__ = "card_cooccurrence"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    card_a = Column(String(255), nullable=False)
    card_b = Column(String(255), nullable=False)
    card1_id = Column(UUID(as_uuid=True), nullable=True)  # Optional FK to cards
    card2_id = Column(UUID(as_uuid=True), nullable=True)  # Optional FK to cards
    format = Column(String(50), nullable=False, default="standard")
    cooccurrence_count = Column(Integer, nullable=False, default=0)
    last_updated = Column(Date, nullable=True)

    __table_args__ = (
        Index("idx_cooccurrence_card", "card_a", "format"),
        Index("idx_cooccurrence_pair", "card1_id", "card2_id", "format"),
        UniqueConstraint("card1_id", "card2_id", "format", name="uq_cooccurrence_pair"),
    )
