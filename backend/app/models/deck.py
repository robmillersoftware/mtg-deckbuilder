import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.session import Base


class DeckVisibility(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"


class Deck(Base):
    """
    Deck model storing user-created or generated MTG decks.
    """

    __tablename__ = "decks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    format = Column(String(50), default="standard")
    archetype = Column(String(100), nullable=True)
    commander = Column(String(255), nullable=True)  # Commander card name for cEDH/Commander decks

    # Deck contents stored as JSON arrays of {card_id, card_name, quantity}
    main_deck = Column(JSONB, nullable=False, default=list)
    sideboard = Column(JSONB, nullable=False, default=list)

    # Strategy and explanations
    strategy_summary = Column(Text, nullable=True)
    card_explanations = Column(JSONB, nullable=True)  # {card_id: explanation}
    matchup_notes = Column(JSONB, nullable=True)  # {archetype: notes}

    # Visibility and sharing
    visibility = Column(String(20), default=DeckVisibility.PRIVATE.value)
    share_token = Column(String(32), unique=True, nullable=True, index=True)

    # Metadata
    is_validated = Column(Boolean, default=False)
    validation_errors = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    owner = relationship("User", back_populates="decks")

    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uq_deck_owner_name"),
        Index("idx_deck_visibility", "visibility"),
    )

    def get_main_deck_count(self) -> int:
        """Calculate total cards in main deck."""
        return sum(card.get("quantity", 0) for card in (self.main_deck or []))

    def get_sideboard_count(self) -> int:
        """Calculate total cards in sideboard."""
        return sum(card.get("quantity", 0) for card in (self.sideboard or []))
