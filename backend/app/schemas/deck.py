from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field


class DeckVisibility(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"


class DeckCardEntry(BaseModel):
    """A card entry in a deck."""

    card_id: Optional[UUID] = None
    card_name: str
    quantity: int = Field(..., ge=1)  # No max - basic lands can have unlimited copies
    set_code: Optional[str] = None
    collector_number: Optional[str] = None


class DeckCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    format: str = "standard"
    main_deck: List[DeckCardEntry] = Field(default_factory=list)
    sideboard: List[DeckCardEntry] = Field(default_factory=list)
    visibility: DeckVisibility = DeckVisibility.PRIVATE


class DeckUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    main_deck: Optional[List[DeckCardEntry]] = None
    sideboard: Optional[List[DeckCardEntry]] = None
    visibility: Optional[DeckVisibility] = None


class DeckResponse(BaseModel):
    id: UUID
    owner_id: UUID
    name: str
    description: Optional[str]
    format: str
    archetype: Optional[str]
    main_deck: List[Dict[str, Any]]
    sideboard: List[Dict[str, Any]]
    strategy_summary: Optional[str]
    card_explanations: Optional[Dict[str, str]]
    matchup_notes: Optional[Dict[str, str]]
    visibility: str
    share_token: Optional[str]
    is_validated: bool
    validation_errors: Optional[List[Dict[str, Any]]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DeckListResponse(BaseModel):
    id: UUID
    name: str
    format: str
    archetype: Optional[str]
    visibility: str
    main_deck_count: int
    sideboard_count: int
    created_at: datetime
    updated_at: datetime


class PaginatedDeckResponse(BaseModel):
    """Paginated list of decks."""
    items: List[DeckResponse]
    total: int
    limit: int
    offset: int


class DeckValidationError(BaseModel):
    type: str
    message: str
    card_name: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None


class DeckValidationReport(BaseModel):
    is_valid: bool
    errors: List[DeckValidationError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    main_deck_count: int
    sideboard_count: int


class DeckImportFormat(str, Enum):
    ARENA = "arena"
    MTGO = "mtgo"
    SIMPLE = "simple"
    AUTO = "auto"


class DeckImportRequest(BaseModel):
    decklist_text: str
    format: DeckImportFormat = DeckImportFormat.AUTO


class CardSuggestion(BaseModel):
    original: str
    suggestions: List[str]
    reason: str


class DeckImportResponse(BaseModel):
    valid: bool
    deck: Optional[DeckResponse] = None
    main_deck: Optional[List[DeckCardEntry]] = None
    sideboard: Optional[List[DeckCardEntry]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    card_suggestions: List[CardSuggestion] = Field(default_factory=list)
    archetype: Optional[str] = None


class DeckExportFormat(str, Enum):
    ARENA = "arena"
    MTGO = "mtgo"
    PLAIN = "plain"


class DeckExportRequest(BaseModel):
    deck_id: UUID
    format: DeckExportFormat = DeckExportFormat.ARENA


class DeckExportResponse(BaseModel):
    content: str
    filename: str
    format: str


class DeckGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=10, description="Natural language deck request")
    conversation_id: Optional[UUID] = None
    include_sideboard: bool = True
    include_explanations: bool = True


class SlotRecommendation(BaseModel):
    slot_type: str
    role_description: str
    card_name: str
    quantity: int
    reasoning: str


class SideboardEntry(BaseModel):
    card_name: str
    quantity: int
    matchups: List[str]
    reasoning: str


class DeckGenerateResponse(BaseModel):
    deck: DeckResponse
    conversation_id: UUID
    strategy_summary: str
    slot_recommendations: List[SlotRecommendation]
    sideboard_guide: List[SideboardEntry]


class DeckIterateRequest(BaseModel):
    modification: str = Field(
        ..., min_length=5, description="Description of desired changes"
    )
    conversation_id: Optional[UUID] = None
    deck_id: Optional[UUID] = None


class ChangeLogEntry(BaseModel):
    action: str  # "added", "removed", "changed"
    card_name: str
    old_quantity: Optional[int] = None
    new_quantity: Optional[int] = None
    reasoning: str


class DeckIterateResponse(BaseModel):
    deck: DeckResponse
    changes: List[ChangeLogEntry]
    summary: str
