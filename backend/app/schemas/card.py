from typing import Optional, List
from uuid import UUID
from decimal import Decimal

from pydantic import BaseModel, Field


class CardResponse(BaseModel):
    id: UUID
    scryfall_id: str
    name: str
    mana_cost: Optional[str]
    cmc: Optional[Decimal]
    type_line: Optional[str]
    oracle_text: Optional[str]
    power: Optional[str]
    toughness: Optional[str]
    colors: Optional[List[str]]
    color_identity: Optional[List[str]]
    keywords: Optional[List[str]]
    rarity: Optional[str]
    set_code: Optional[str]
    collector_number: Optional[str]
    image_uri: Optional[str]
    image_uri_small: Optional[str]
    is_standard_legal: bool

    class Config:
        from_attributes = True


class CardSearchParams(BaseModel):
    q: Optional[str] = None
    colors: Optional[List[str]] = Field(None, description="Filter by colors (W, U, B, R, G)")
    cmc_min: Optional[int] = None
    cmc_max: Optional[int] = None
    type: Optional[str] = None
    rarity: Optional[str] = None
    keywords: Optional[List[str]] = None
    standard_only: bool = True
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class CardCandidateRequest(BaseModel):
    """Request for candidate cards based on role and constraints."""

    role: str = Field(..., description="Description of the card's role (e.g., 'removal', 'creature')")
    description: Optional[str] = Field(None, description="Additional description of what's needed")
    constraints: Optional[dict] = Field(
        None,
        description="Constraints like {colors: ['R'], cmc_max: 3, type: 'instant'}",
    )
    exclude_cards: Optional[List[str]] = Field(None, description="Card names to exclude")
    min_results: int = Field(3, ge=1, le=10)
    max_results: int = Field(10, ge=3, le=20)


class CardCandidateResponse(BaseModel):
    """Response with candidate cards for LLM selection."""

    candidates: List[CardResponse]
    role: str
    constraints_applied: dict


class CardSelectionRequest(BaseModel):
    """LLM's selection from candidate cards."""

    selection: UUID = Field(..., description="Card ID from the candidate list")
    quantity: int = Field(..., ge=1, le=4)
    reasoning: str = Field(..., description="Explanation for the selection")
    format: Optional[str] = Field("standard", description="Format for legality check")


class CardSelectionResponse(BaseModel):
    """Confirmation of card selection."""

    success: bool
    card: CardResponse
    quantity: int
    message: str


class SemanticSearchRequest(BaseModel):
    """Request for semantic card search."""

    query: str = Field(..., min_length=3, description="Natural language description")
    limit: int = Field(10, ge=5, le=20)
    format: Optional[str] = Field("standard", description="Format for legality filtering")
