from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class DeckAnalysisRequest(BaseModel):
    main_deck: List[Dict[str, Any]] = Field(default_factory=list)
    sideboard: List[Dict[str, Any]] = Field(default_factory=list)
    format: str = "standard"


class DeckAnalysisResponse(BaseModel):
    main_deck_count: int
    sideboard_count: int
    target_main: int
    target_sideboard: int
    creature_count: int
    spell_count: int
    land_count: int
    mana_curve: Dict[str, int]  # {"0": 2, "1": 8, ...}
    colors: Dict[str, int]  # {"W": 12, "U": 8, ...}
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class CardSuggestRequest(BaseModel):
    main_deck: List[Dict[str, Any]] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    role: str = "removal"
    format: str = "standard"
    limit: int = 6


class CardSuggestionEntry(BaseModel):
    card_name: str
    card_id: Optional[str] = None
    mana_cost: Optional[str] = None
    type_line: Optional[str] = None
    image_uri: Optional[str] = None


class CardSuggestResponse(BaseModel):
    role: str
    suggestions: List[CardSuggestionEntry] = Field(default_factory=list)
