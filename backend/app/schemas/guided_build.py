from typing import Optional, List, Dict, Any
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field


class GuidedBuildStep(str, Enum):
    STRATEGY = "strategy"
    COLORS = "colors"
    CORE = "core"
    SUPPORT = "support"
    MANA_BASE = "mana_base"
    SIDEBOARD = "sideboard"
    REVIEW = "review"


STEP_ORDER = [
    GuidedBuildStep.STRATEGY,
    GuidedBuildStep.COLORS,
    GuidedBuildStep.CORE,
    GuidedBuildStep.SUPPORT,
    GuidedBuildStep.MANA_BASE,
    GuidedBuildStep.SIDEBOARD,
    GuidedBuildStep.REVIEW,
]


class ArchetypeOption(BaseModel):
    name: str
    description: str
    playstyle: str
    meta_percentage: Optional[float] = None
    example_cards: List[str] = Field(default_factory=list)


class ColorOption(BaseModel):
    colors: List[str]
    name: str
    description: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class CardRecommendation(BaseModel):
    card_name: str
    card_id: Optional[str] = None
    quantity: int = 1
    role: str
    reasoning: str
    image_uri: Optional[str] = None
    mana_cost: Optional[str] = None
    type_line: Optional[str] = None


class CardSlotGroup(BaseModel):
    slot_name: str
    description: str
    target_count: int
    recommendations: List[CardRecommendation] = Field(default_factory=list)


class LandRecommendation(BaseModel):
    card_name: str
    card_id: Optional[str] = None
    quantity: int
    category: str  # "dual", "utility", "basic", "fetch", "other"
    reasoning: str
    image_uri: Optional[str] = None


class SideboardRecommendation(BaseModel):
    card_name: str
    card_id: Optional[str] = None
    quantity: int
    target_matchups: List[str]
    reasoning: str
    image_uri: Optional[str] = None


# --- Request schemas ---

class StartGuidedBuildRequest(BaseModel):
    format: str = "standard"


class AdvanceStepRequest(BaseModel):
    session_id: UUID
    selections: Dict[str, Any] = Field(default_factory=dict)


# --- Response schemas ---

class StrategyStepData(BaseModel):
    archetypes: List[ArchetypeOption]
    meta_summary: str
    format: str


class ColorStepData(BaseModel):
    options: List[ColorOption]
    recommendation: str
    archetype: str


class CoreStepData(BaseModel):
    slots: List[CardSlotGroup]
    strategy_note: str
    cards_needed: int


class SupportStepData(BaseModel):
    slots: List[CardSlotGroup]
    current_deck_size: int
    remaining_nonland_slots: int


class ManaBaseStepData(BaseModel):
    lands: List[LandRecommendation]
    total_lands: int
    color_requirements: Dict[str, int]
    mana_curve_note: str


class SideboardStepData(BaseModel):
    recommendations: List[SideboardRecommendation]
    meta_matchups: List[Dict[str, Any]]
    sideboard_strategy: str


class ReviewStepData(BaseModel):
    deck_name: str
    archetype: str
    colors: List[str]
    strategy_summary: str
    main_deck: List[Dict[str, Any]]
    sideboard: List[Dict[str, Any]]
    main_deck_count: int
    sideboard_count: int
    validation_errors: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class GuidedBuildStepResponse(BaseModel):
    session_id: UUID
    current_step: GuidedBuildStep
    step_index: int
    total_steps: int
    step_title: str
    step_description: str
    data: Dict[str, Any]
    ai_message: str


class GuidedBuildCompleteResponse(BaseModel):
    session_id: UUID
    deck_id: Optional[UUID] = None
    deck_name: str
    main_deck: List[Dict[str, Any]]
    sideboard: List[Dict[str, Any]]
    strategy_summary: str
    archetype: str
    colors: List[str]
    format: str
    is_valid: bool
    validation_errors: List[str] = Field(default_factory=list)
