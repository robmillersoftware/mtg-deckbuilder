import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.session import Base


class Conversation(Base):
    """
    Conversation model storing chat history with the AI assistant.
    Includes the current deck context for follow-up questions.
    """

    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,  # Allow anonymous conversations
        index=True,
    )

    # Summary for list view
    summary = Column(String(255), nullable=True)

    # Messages stored as JSON array of {role, content, timestamp}
    messages = Column(JSONB, nullable=False, default=list)

    # Current deck being discussed (JSON representation)
    current_deck = Column(JSONB, nullable=True)

    # Conversation context tracking - persists strategy, phase, and key details
    # across turns so the AI always knows what the user is building.
    # Schema: {strategy, colors, phase, build_around_cards, archetype, summary}
    context = Column(JSONB, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="conversations")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        if self.messages is None:
            self.messages = []
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_message_count(self) -> int:
        """Get the number of messages in the conversation."""
        return len(self.messages or [])

    def get_context(self) -> dict:
        """Get the conversation context, initializing if needed."""
        if self.context is None:
            self.context = {}
        return self.context

    def update_context(self, **kwargs) -> None:
        """Update specific fields in the conversation context."""
        if self.context is None:
            self.context = {}
        ctx = dict(self.context)
        ctx.update({k: v for k, v in kwargs.items() if v is not None})
        self.context = ctx
