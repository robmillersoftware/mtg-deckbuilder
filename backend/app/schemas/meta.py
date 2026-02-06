from datetime import date, datetime
from typing import Optional, List
from uuid import UUID
from decimal import Decimal

from pydantic import BaseModel, Field


class MetaSnapshotResponse(BaseModel):
    id: UUID
    format: str
    snapshot_date: date
    archetype: str
    meta_percentage: Optional[Decimal]
    sample_size: Optional[int]
    avg_finish: Optional[Decimal]
    key_cards: Optional[List[str]]

    class Config:
        from_attributes = True


class ArchetypeEntry(BaseModel):
    name: str
    meta_percentage: float
    sample_size: int
    avg_finish: float
    key_cards: List[str]


class MetaDashboardResponse(BaseModel):
    format: str
    last_updated: date
    archetypes: List[ArchetypeEntry]


class MatchupRating(BaseModel):
    archetype: str
    meta_percentage: float
    rating: str  # "Favored", "Even", "Unfavored"
    explanation: str


class SideboardSwap(BaseModel):
    cards_out: List[str]
    cards_in: List[str]
    reason: str


class MatchupAnalysis(BaseModel):
    deck_archetype: str
    matchups: List[MatchupRating]
    sideboard_guides: dict  # archetype -> SideboardSwap


class CooccurrenceResult(BaseModel):
    card_a: str
    card_b: str
    count: int


class ArchetypeTrend(BaseModel):
    """An archetype with its trend data (rising/falling)."""
    name: str
    current_percentage: float
    previous_percentage: float
    change: float  # Percentage point change (positive = rising, negative = falling)
    change_percent: float  # Relative change percentage
    sample_size: int
    key_cards: List[str]


class MetaTrendsResponse(BaseModel):
    """Response containing emerging and declining archetypes."""
    format: str
    current_date: date
    comparison_date: date
    rising: List[ArchetypeTrend]
    falling: List[ArchetypeTrend]
    new_archetypes: List[ArchetypeEntry]  # Archetypes that didn't exist before
    disappeared: List[str]  # Archetypes that disappeared from meta


class MetaHealthResponse(BaseModel):
    """Meta health and diversity metrics."""
    format: str
    snapshot_date: date
    diversity_score: float  # 0-100, higher = more diverse
    top_deck_share: float  # % of meta held by top deck
    top_3_share: float  # % of meta held by top 3 decks
    total_archetypes: int
    health_rating: str  # "Healthy", "Moderate", "Unhealthy"
    assessment: str  # Brief text assessment


class CardArchetypeBreakdown(BaseModel):
    """Which archetype uses a card and how often."""
    name: str
    count: int
    percentage: float


class CardMetaStatsEntry(BaseModel):
    """Per-card metagame representation stats."""
    card_name: str
    deck_count: int
    total_decks: int
    meta_percentage: float
    main_deck_count: int
    sideboard_count: int
    avg_copies: float
    archetypes: List[CardArchetypeBreakdown]

    class Config:
        from_attributes = True


class CardMetaStatsResponse(BaseModel):
    """Response for the card meta stats endpoint."""
    format: str
    snapshot_date: date
    total_cards: int
    cards: List[CardMetaStatsEntry]


class CardTrend(BaseModel):
    """A card with its trend data (rising/falling)."""
    card_name: str
    current_percentage: float
    previous_percentage: float
    change: float  # Percentage point change
    change_percent: float  # Relative change percentage
    current_deck_count: int
    avg_copies: float


class CardTrendsResponse(BaseModel):
    """Response containing rising and falling cards."""
    format: str
    current_date: date
    comparison_date: date
    rising: List[CardTrend]
    falling: List[CardTrend]
    new_cards: List[CardMetaStatsEntry]
    disappeared: List[str]
