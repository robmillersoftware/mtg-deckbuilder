from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.dialects.postgresql import ARRAY

from app.db.session import get_db
from app.models.card import Card
from app.schemas.card import (
    CardResponse,
    CardSearchParams,
    CardCandidateRequest,
    CardCandidateResponse,
    CardSelectionRequest,
    CardSelectionResponse,
    SemanticSearchRequest,
)
from app.services.card_service import CardService

router = APIRouter()


@router.get("/search", response_model=List[CardResponse])
async def search_cards(
    q: Optional[str] = None,
    colors: Optional[str] = Query(None, description="Comma-separated colors (W,U,B,R,G)"),
    cmc_min: Optional[int] = None,
    cmc_max: Optional[int] = None,
    type: Optional[str] = None,
    rarity: Optional[str] = None,
    standard_only: bool = True,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Search cards with various filters.
    Response time target: < 500ms
    """
    query = select(Card)
    conditions = []

    # Standard-legal filter
    if standard_only:
        conditions.append(Card.is_standard_legal == True)

    # Text search (name or oracle text)
    if q:
        search_term = f"%{q.lower()}%"
        conditions.append(
            or_(
                func.lower(Card.name).like(search_term),
                func.lower(Card.oracle_text).like(search_term),
            )
        )

    # Color filter
    if colors:
        color_list = [c.strip().upper() for c in colors.split(",")]
        for color in color_list:
            if color in ["W", "U", "B", "R", "G"]:
                conditions.append(Card.colors.contains([color]))

    # CMC range
    if cmc_min is not None:
        conditions.append(Card.cmc >= cmc_min)
    if cmc_max is not None:
        conditions.append(Card.cmc <= cmc_max)

    # Type filter
    if type:
        conditions.append(func.lower(Card.type_line).like(f"%{type.lower()}%"))

    # Rarity filter
    if rarity:
        conditions.append(Card.rarity == rarity.lower())

    if conditions:
        query = query.where(and_(*conditions))

    # Fetch a large batch and dedupe in Python
    fetch_limit = max(limit * 50, 5000)
    query = query.order_by(Card.name).limit(fetch_limit)

    result = await db.execute(query)
    all_cards = result.scalars().all()

    # Deduplicate by name
    seen_names = set()
    unique_cards = []
    for card in all_cards:
        if card.name not in seen_names:
            seen_names.add(card.name)
            unique_cards.append(card)

    # Apply offset and limit after deduplication
    return unique_cards[offset:offset + limit]


@router.get("/{card_id}", response_model=CardResponse)
async def get_card(
    card_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific card by ID."""
    result = await db.execute(select(Card).where(Card.id == card_id))
    card = result.scalar_one_or_none()

    if card is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Card not found",
        )

    return card


@router.get("/by-name/{name}", response_model=CardResponse)
async def get_card_by_name(
    name: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific card by exact name."""
    result = await db.execute(
        select(Card).where(func.lower(Card.name) == name.lower()).limit(1)
    )
    card = result.scalar_one_or_none()

    if card is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Card not found",
        )

    return card


@router.post("/semantic-search", response_model=List[CardResponse])
async def semantic_search_cards(
    request: SemanticSearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Search cards using natural language description.
    Returns cards ranked by semantic similarity.
    Response time target: < 1 second
    """
    card_service = CardService(db)
    cards = await card_service.semantic_search(
        query=request.query,
        limit=request.limit,
        standard_only=True,
    )
    return cards


@router.post("/ai/candidate-cards", response_model=CardCandidateResponse)
async def get_candidate_cards(
    request: CardCandidateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Get candidate cards for LLM selection based on role and constraints.
    Returns 3-10 cards matching the criteria.
    """
    card_service = CardService(db)
    candidates = await card_service.get_candidates(
        role=request.role,
        description=request.description,
        constraints=request.constraints,
        exclude_cards=request.exclude_cards,
        min_results=request.min_results,
        max_results=request.max_results,
    )

    return CardCandidateResponse(
        candidates=candidates,
        role=request.role,
        constraints_applied=request.constraints or {},
    )


@router.post("/ai/select-card", response_model=CardSelectionResponse)
async def select_card(
    request: CardSelectionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Validate and confirm card selection from candidates.
    Verifies the card exists, is standard-legal, and quantity is valid.
    """
    result = await db.execute(select(Card).where(Card.id == request.selection))
    card = result.scalar_one_or_none()

    if card is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Selected card not found in database",
        )

    if not card.is_standard_legal:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Card '{card.name}' is not Standard-legal",
        )

    # Check quantity (max 4, except basic lands)
    if request.quantity > 4 and not card.is_basic_land():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum 4 copies allowed for non-basic-land cards",
        )

    return CardSelectionResponse(
        success=True,
        card=card,
        quantity=request.quantity,
        message=f"Selected {request.quantity}x {card.name}",
    )


@router.get("/suggestions/{partial_name}", response_model=List[CardResponse])
async def get_card_suggestions(
    partial_name: str,
    limit: int = Query(5, ge=1, le=10),
    db: AsyncSession = Depends(get_db),
):
    """Get card suggestions for autocomplete based on partial name."""
    result = await db.execute(
        select(Card)
        .where(
            and_(
                Card.is_standard_legal == True,
                func.lower(Card.name).like(f"%{partial_name.lower()}%"),
            )
        )
        .order_by(Card.name)
        .limit(limit)
    )
    cards = result.scalars().all()
    return cards
