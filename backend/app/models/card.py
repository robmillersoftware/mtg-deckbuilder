import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    DateTime,
    Float,
    Numeric,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from pgvector.sqlalchemy import Vector

from app.db.session import Base


class Card(Base):
    """
    Card model representing MTG cards from Scryfall.
    Stores Standard-legal cards with embeddings for semantic search.
    """

    __tablename__ = "cards"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scryfall_id = Column(String(36), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    mana_cost = Column(String(50), nullable=True)
    cmc = Column(Float, nullable=True)
    type_line = Column(String(255), nullable=True)
    oracle_text = Column(Text, nullable=True)
    power = Column(String(10), nullable=True)
    toughness = Column(String(10), nullable=True)
    colors = Column(ARRAY(String(1)), nullable=True)  # W, U, B, R, G
    color_identity = Column(ARRAY(String(1)), nullable=True)
    keywords = Column(ARRAY(String(100)), nullable=True)
    legalities = Column(JSONB, nullable=True)
    set_code = Column(String(10), nullable=True)
    collector_number = Column(String(20), nullable=True)
    rarity = Column(String(20), nullable=True)
    image_uri = Column(Text, nullable=True)
    image_uri_small = Column(Text, nullable=True)
    image_uri_art_crop = Column(Text, nullable=True)
    price_usd = Column(Numeric(10, 2), nullable=True)
    price_usd_foil = Column(Numeric(10, 2), nullable=True)
    scryfall_uri = Column(Text, nullable=True)
    oracle_id = Column(String(36), nullable=True)
    set_name = Column(String(255), nullable=True)
    embedding = Column(Vector(1536), nullable=True)  # For semantic search
    is_standard_legal = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_cards_colors", "colors", postgresql_using="gin"),
        Index("idx_cards_type_line", "type_line"),
        Index("idx_cards_cmc", "cmc"),
        Index("idx_cards_standard_legal", "is_standard_legal"),
    )

    def is_basic_land(self) -> bool:
        """Check if this card is a basic land."""
        basic_lands = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
        return self.name in basic_lands
