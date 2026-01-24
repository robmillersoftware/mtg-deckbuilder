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
from app.services.card_service import CardService, FORMAT_LEGALITY_MAP

router = APIRouter()


@router.get("/search", response_model=List[CardResponse])
async def search_cards(
    q: Optional[str] = None,
    colors: Optional[str] = Query(None, description="Comma-separated colors (W,U,B,R,G)"),
    cmc_min: Optional[int] = None,
    cmc_max: Optional[int] = None,
    type: Optional[str] = None,
    rarity: Optional[str] = None,
    format: Optional[str] = Query("standard", description="Format to filter by (standard, historic, modern, legacy, cedh)"),
    standard_only: bool = Query(True, description="Deprecated: use format parameter instead"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Search cards with various filters.
    Response time target: < 500ms
    """
    from app.services.card_service import get_format_legality_condition

    query = select(Card)
    conditions = []

    # Format-based legality filter
    if format and format in FORMAT_LEGALITY_MAP:
        conditions.append(get_format_legality_condition(format))
    elif standard_only:
        # Backwards compatibility
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
    """Get a specific card by exact name. Handles DFC names automatically."""
    # Try exact match first
    result = await db.execute(
        select(Card).where(func.lower(Card.name) == name.lower()).limit(1)
    )
    card = result.scalar_one_or_none()

    # If not found, try matching as front face of a DFC (name starts with search term + " // ")
    if card is None:
        result = await db.execute(
            select(Card).where(
                func.lower(Card.name).like(f"{name.lower()} // %")
            ).limit(1)
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
        format=request.format or 'standard',
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
    # Get format from constraints if provided
    format = (request.constraints or {}).get("format", "standard")
    candidates = await card_service.get_candidates(
        role=request.role,
        description=request.description,
        constraints=request.constraints,
        exclude_cards=request.exclude_cards,
        min_results=request.min_results,
        max_results=request.max_results,
        format=format,
    )

    return CardCandidateResponse(
        candidates=candidates,
        role=request.role,
        constraints_applied=request.constraints or {},
    )


def is_card_legal_in_format(card: Card, format: str) -> bool:
    """Check if a card is legal in the specified format."""
    if format == "standard":
        return card.is_standard_legal
    if card.legalities is None:
        return False
    legality_key = FORMAT_LEGALITY_MAP.get(format, "standard")
    return card.legalities.get(legality_key) == "legal"


@router.post("/ai/select-card", response_model=CardSelectionResponse)
async def select_card(
    request: CardSelectionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Validate and confirm card selection from candidates.
    Verifies the card exists, is legal in the specified format, and quantity is valid.
    """
    result = await db.execute(select(Card).where(Card.id == request.selection))
    card = result.scalar_one_or_none()

    if card is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Selected card not found in database",
        )

    # Check format legality
    format = getattr(request, 'format', 'standard') or 'standard'
    if not is_card_legal_in_format(card, format):
        format_name = format.upper() if format != "cedh" else "cEDH"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Card '{card.name}' is not legal in {format_name}",
        )

    # Check quantity (max 4 for most formats, max 1 for singleton formats like cEDH)
    max_copies = 1 if format == "cedh" else 4
    if request.quantity > max_copies and not card.is_basic_land():
        if format == "cedh":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"cEDH is a singleton format - maximum 1 copy allowed for non-basic-land cards",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {max_copies} copies allowed for non-basic-land cards",
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
    format: Optional[str] = Query("standard", description="Format to filter by"),
    limit: int = Query(5, ge=1, le=10),
    db: AsyncSession = Depends(get_db),
):
    """Get card suggestions for autocomplete based on partial name."""
    from app.services.card_service import get_format_legality_condition

    conditions = [func.lower(Card.name).like(f"%{partial_name.lower()}%")]

    if format and format in FORMAT_LEGALITY_MAP:
        conditions.append(get_format_legality_condition(format))
    else:
        conditions.append(Card.is_standard_legal == True)

    result = await db.execute(
        select(Card)
        .where(and_(*conditions))
        .order_by(Card.name)
        .limit(limit)
    )
    cards = result.scalars().all()
    return cards
