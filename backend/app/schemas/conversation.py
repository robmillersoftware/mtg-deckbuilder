from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field


class MessageEntry(BaseModel):
    role: str  # "user", "assistant"
    content: str
    timestamp: datetime


class ConversationResponse(BaseModel):
    id: UUID
    user_id: Optional[UUID]
    summary: Optional[str]
    messages: List[Dict[str, Any]]
    current_deck: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    id: UUID
    summary: Optional[str]
    message_count: int
    has_deck: bool
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[UUID] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: UUID
    deck: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None


class CardExplanationRequest(BaseModel):
    card_name: str
    conversation_id: UUID


class CardExplanationResponse(BaseModel):
    card_name: str
    role: str
    explanation: str
    synergies: List[str]
    alternatives: List[str]
