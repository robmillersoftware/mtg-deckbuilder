"""
SQLAlchemy models for Spellbook.
"""

# Import Base from db.session to ensure all models use the same Base
from app.db.session import Base

# Import all models to register them with Base
from app.models.user import User, VerificationToken, ResetToken, Preferences
from app.models.card import Card, CardRole, CARD_ROLES
from app.models.deck import Deck
from app.models.conversation import Conversation
from app.models.meta import Event, Decklist, MetaSnapshot, CardCooccurrence, ArchetypeTemplate
from app.models.job import JobRun

__all__ = [
    "Base",
    "User",
    "VerificationToken",
    "ResetToken",
    "Preferences",
    "Card",
    "CardRole",
    "CARD_ROLES",
    "Deck",
    "Conversation",
    "Event",
    "Decklist",
    "MetaSnapshot",
    "CardCooccurrence",
    "ArchetypeTemplate",
    "JobRun",
]
