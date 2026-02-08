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
    context: Optional[Dict[str, Any]] = None
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
    format: Optional[str] = Field(default="standard", description="Game format (standard, historic, modern, legacy, cedh)")
    current_deck: Optional[Dict[str, Any]] = Field(default=None, description="Current local deck state synced from frontend")


class CardSuggestionItem(BaseModel):
    card_name: str
    quantity: int = 1
    mana_cost: Optional[str] = None
    type_line: Optional[str] = None
    image_uri: Optional[str] = None
    reasoning: Optional[str] = None


class CardSuggestionGroup(BaseModel):
    group_name: str
    role: str
    cards: List[CardSuggestionItem]
    is_batch: bool = False


class ChatResponse(BaseModel):
    response: str
    conversation_id: UUID
    deck: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    card_suggestions: Optional[List[Dict[str, Any]]] = None


class CardExplanationRequest(BaseModel):
    card_name: str
    conversation_id: UUID


class CardExplanationResponse(BaseModel):
    card_name: str
    role: str
    explanation: str
    synergies: List[str]
    alternatives: List[str]
