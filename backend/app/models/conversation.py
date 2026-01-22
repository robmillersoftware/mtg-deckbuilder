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
