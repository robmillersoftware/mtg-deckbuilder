import uuid
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    DateTime,
    Float,
    Numeric,
    Integer,
    Index,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
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

    # Relationships
    roles = relationship("CardRole", back_populates="card", cascade="all, delete-orphan")

    def is_basic_land(self) -> bool:
        """Check if this card is a basic land."""
        basic_lands = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
        return self.name in basic_lands


# Valid role values for classification
CARD_ROLES = [
    # Removal
    "removal_targeted",
    "removal_mass",
    "removal_artifact_enchantment",
    # Card advantage
    "card_draw",
    "card_selection",
    # Mana
    "ramp",
    # Interaction
    "counterspell",
    "discard",
    # Threats
    "threat_cheap",      # CMC ≤ 2
    "threat_midrange",   # CMC 3-4
    "threat_finisher",   # CMC 5+
    # Utility
    "protection",
    "burn",
    "lifegain",
    "recursion",
    "graveyard_hate",
    "tutor",
    # Lands
    "land_fixing_untapped",
    "land_fixing_tapped",
    "land_utility",
    "land_creature",
    "land_basic",
]

# What each system role means. Used as Jev question text for role tagging and role checks.
ROLE_DEFINITIONS: Dict[str, str] = {
    "removal_targeted": "Destroys, exiles, deals damage to, or otherwise removes a single opposing creature or planeswalker",
    "removal_mass": "Board wipe: destroys, exiles, or kills multiple creatures at once",
    "removal_artifact_enchantment": "Destroys or exiles artifacts or enchantments",
    "card_draw": "Draws one or more cards (searching the library for a specific card or land is not card draw)",
    "card_selection": "Scry, surveil, looks at top cards of the library, or otherwise filters draws",
    "ramp": "Accelerates mana beyond one land per turn: a mana creature or rock, or putting extra lands onto the battlefield. A land tapping for mana is not ramp",
    "counterspell": "Counters spells",
    "discard": "Makes an opponent discard cards",
    "threat_cheap": "An efficient creature or threat with mana value 2 or less that pressures the opponent in combat (mana and utility creatures do not count)",
    "threat_midrange": "A value creature or planeswalker with mana value 3 or 4",
    "threat_finisher": "A game-ending threat with mana value 5 or more",
    "protection": "Gives hexproof, indestructible, or ward, or prevents damage to your permanents",
    "burn": "Deals direct damage to players or any target",
    "lifegain": "Gaining life is a main purpose of the card, not a side effect of another ability or of a land entering",
    "recursion": "Returns cards from a graveyard to hand or battlefield",
    "graveyard_hate": "Exiles cards from graveyards",
    "tutor": "Searches the library for a card of the player's choice (not only a basic land) and puts it into hand or onto the battlefield",
    "land_fixing_untapped": "A land producing two or more colors that can enter untapped",
    "land_fixing_tapped": "A land producing two or more colors that always enters tapped",
    "land_utility": "A land with a useful ability beyond making mana, such as destroying a land or drawing cards. Entering tapped, gaining 1 life on entry, or becoming a creature do not count",
    "land_creature": "A land that can become a creature",
    "land_basic": "A basic land (Plains, Island, Swamp, Mountain, Forest)",
}

# Map user-facing role names (chat tools, guided builder) to system role names in card_roles
ROLE_MAP: Dict[str, List[str]] = {
    "threats": ["threat_cheap", "threat_midrange", "threat_finisher"],
    "creatures": ["threat_cheap", "threat_midrange", "threat_finisher"],
    "removal": ["removal_targeted", "removal_mass", "removal_artifact_enchantment"],
    "card advantage": ["card_draw", "card_selection"],
    "card draw": ["card_draw", "card_selection"],
    "counterspells": ["counterspell"],
    "protection": ["protection"],
    "ramp": ["ramp"],
    "burn": ["burn"],
    "recursion": ["recursion"],
    "finishers": ["threat_finisher"],
    "interaction": ["removal_targeted", "counterspell"],
    "discard": ["discard"],
    "lifegain": ["lifegain"],
    "graveyard hate": ["graveyard_hate"],
    "tutors": ["tutor"],
    "sacrifice outlets": ["recursion"],
    "board wipes": ["removal_mass"],
    "spot removal": ["removal_targeted"],
    "cheap threats": ["threat_cheap"],
    "big threats": ["threat_finisher"],
    "top end": ["threat_finisher"],
    "early threats": ["threat_cheap"],
}


class CardRole(Base):
    """
    Card role classification for deck building.
    A card can have multiple roles (e.g., a creature that also removes things).
    """

    __tablename__ = "card_roles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    card_id = Column(
        UUID(as_uuid=True),
        ForeignKey("cards.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(50), nullable=False, index=True)
    efficiency = Column(Integer, nullable=True)  # 1-5 rating within role
    confidence = Column(Numeric(3, 2), nullable=True)  # AI confidence 0.00-1.00
    reasoning = Column(Text, nullable=True)  # AI explanation
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    card = relationship("Card", back_populates="roles")

    __table_args__ = (
        UniqueConstraint("card_id", "role", name="uq_card_role"),
        Index("idx_card_role_lookup", "role", "efficiency"),
    )
