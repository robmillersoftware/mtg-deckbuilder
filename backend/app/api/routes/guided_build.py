from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.guided_builder import DeckAnalyzer
from app.schemas.guided_build import (
    DeckAnalysisRequest,
    DeckAnalysisResponse,
    CardSuggestRequest,
    CardSuggestResponse,
    CardSuggestionEntry,
)

router = APIRouter()


@router.post("/analyze", response_model=DeckAnalysisResponse)
async def analyze_deck(
    request: DeckAnalysisRequest,
    db: AsyncSession = Depends(get_db),
):
    """Analyze a deck-in-progress and return stats, curve, and suggestions."""
    analyzer = DeckAnalyzer(db)
    result = await analyzer.analyze(
        main_deck=request.main_deck,
        sideboard=request.sideboard,
        format=request.format,
    )
    # Convert int keys to string keys for JSON
    result["mana_curve"] = {str(k): v for k, v in result["mana_curve"].items()}
    return result


@router.post("/suggest", response_model=CardSuggestResponse)
async def suggest_cards(
    request: CardSuggestRequest,
    db: AsyncSession = Depends(get_db),
):
    """Suggest cards for a specific role based on the current deck."""
    analyzer = DeckAnalyzer(db)
    cards = await analyzer.suggest_cards(
        main_deck=request.main_deck,
        colors=request.colors,
        role=request.role,
        format=request.format,
        limit=request.limit,
    )
    return CardSuggestResponse(
        role=request.role,
        suggestions=[CardSuggestionEntry(**c) for c in cards],
    )
