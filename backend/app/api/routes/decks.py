from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.db.session import get_db
from app.models.deck import Deck, DeckVisibility
from app.models.user import User
from app.schemas.deck import (
    DeckCreate,
    DeckUpdate,
    DeckResponse,
    DeckListResponse,
    PaginatedDeckResponse,
    DeckImportRequest,
    DeckImportResponse,
    DeckExportRequest,
    DeckExportResponse,
    DeckGenerateRequest,
    DeckGenerateResponse,
    DeckIterateRequest,
    DeckIterateResponse,
    DeckValidationReport,
    DeckExportFormat,
    SideboardMatrixResponse,
)
from app.api.deps.auth import get_current_user_required, get_current_user
from app.services.deck_service import DeckService
from app.services.deck_validator import DeckValidator
from app.services.deck_exporter import DeckExporter
from app.services.deck_importer import DeckImporter
from app.services.deck_generator import DeckGenerator
from app.services.ai_service import AIService
from app.core.security import generate_share_token
from app.models.meta import MetaSnapshot

router = APIRouter()


@router.get("", response_model=PaginatedDeckResponse)
async def list_user_decks(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """List all decks owned by the current user."""
    from sqlalchemy import func

    # Get total count
    count_result = await db.execute(
        select(func.count()).select_from(Deck).where(Deck.owner_id == current_user.id)
    )
    total = count_result.scalar()

    # Get decks with pagination
    result = await db.execute(
        select(Deck)
        .where(Deck.owner_id == current_user.id)
        .order_by(Deck.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    decks = result.scalars().all()

    return PaginatedDeckResponse(
        items=decks,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=DeckResponse, status_code=status.HTTP_201_CREATED)
async def create_deck(
    deck_data: DeckCreate,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Create a new deck."""
    # Check for duplicate name
    result = await db.execute(
        select(Deck).where(
            and_(
                Deck.owner_id == current_user.id,
                Deck.name == deck_data.name,
            )
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A deck with this name already exists",
        )

    # Validate deck - use mode='json' to ensure UUIDs are serialized as strings
    validator = DeckValidator(db)
    main_deck_entries = [entry.model_dump(mode='json') for entry in deck_data.main_deck]
    sideboard_entries = [entry.model_dump(mode='json') for entry in deck_data.sideboard]
    validation = await validator.validate(main_deck_entries, sideboard_entries)

    # Create deck
    deck = Deck(
        owner_id=current_user.id,
        name=deck_data.name,
        description=deck_data.description,
        format=deck_data.format,
        main_deck=main_deck_entries,
        sideboard=sideboard_entries,
        visibility=deck_data.visibility.value,
        is_validated=validation.is_valid,
        validation_errors=[e.model_dump(mode='json') for e in validation.errors] if validation.errors else None,
    )

    # Generate share token for public decks
    if deck_data.visibility == DeckVisibility.PUBLIC:
        deck.share_token = generate_share_token()

    db.add(deck)
    await db.commit()
    await db.refresh(deck)

    return deck


@router.get("/{deck_id}", response_model=DeckResponse)
async def get_deck(
    deck_id: UUID,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific deck by ID."""
    result = await db.execute(
        select(Deck).where(
            and_(
                Deck.id == deck_id,
                Deck.owner_id == current_user.id,
            )
        )
    )
    deck = result.scalar_one_or_none()

    if deck is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deck not found",
        )

    return deck


@router.patch("/{deck_id}", response_model=DeckResponse)
async def update_deck(
    deck_id: UUID,
    deck_data: DeckUpdate,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Update an existing deck."""
    result = await db.execute(
        select(Deck).where(
            and_(
                Deck.id == deck_id,
                Deck.owner_id == current_user.id,
            )
        )
    )
    deck = result.scalar_one_or_none()

    if deck is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deck not found",
        )

    # Check for duplicate name if changing
    if deck_data.name and deck_data.name != deck.name:
        name_check = await db.execute(
            select(Deck).where(
                and_(
                    Deck.owner_id == current_user.id,
                    Deck.name == deck_data.name,
                    Deck.id != deck_id,
                )
            )
        )
        if name_check.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A deck with this name already exists",
            )
        deck.name = deck_data.name

    if deck_data.description is not None:
        deck.description = deck_data.description

    if deck_data.main_deck is not None:
        deck.main_deck = [entry.model_dump(mode='json') for entry in deck_data.main_deck]

    if deck_data.sideboard is not None:
        deck.sideboard = [entry.model_dump(mode='json') for entry in deck_data.sideboard]

    if deck_data.visibility is not None:
        deck.visibility = deck_data.visibility.value
        # Generate share token for public decks
        if deck_data.visibility == DeckVisibility.PUBLIC and not deck.share_token:
            deck.share_token = generate_share_token()

    # Re-validate if deck contents changed
    if deck_data.main_deck is not None or deck_data.sideboard is not None:
        validator = DeckValidator(db)
        validation = await validator.validate(deck.main_deck, deck.sideboard)
        deck.is_validated = validation.is_valid
        deck.validation_errors = [e.model_dump(mode='json') for e in validation.errors] if validation.errors else None

    await db.commit()
    await db.refresh(deck)

    return deck


@router.delete("/{deck_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_deck(
    deck_id: UUID,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Delete a deck."""
    result = await db.execute(
        select(Deck).where(
            and_(
                Deck.id == deck_id,
                Deck.owner_id == current_user.id,
            )
        )
    )
    deck = result.scalar_one_or_none()

    if deck is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deck not found",
        )

    await db.delete(deck)
    await db.commit()


@router.get("/public/{share_token}", response_model=DeckResponse)
async def get_public_deck(
    share_token: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a public deck by share token (no authentication required)."""
    result = await db.execute(
        select(Deck).where(
            and_(
                Deck.share_token == share_token,
                Deck.visibility == DeckVisibility.PUBLIC.value,
            )
        )
    )
    deck = result.scalar_one_or_none()

    if deck is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deck not found or not public",
        )

    return deck


@router.post("/import", response_model=DeckImportResponse)
async def import_deck(
    request: DeckImportRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Import a deck from text (Arena, MTGO, or simple format)."""
    importer = DeckImporter(db)
    result = await importer.import_deck(
        decklist_text=request.decklist_text,
        format_hint=request.format.value if request.format else "auto",
    )
    return result


@router.post("/export", response_model=DeckExportResponse)
async def export_deck(
    request: DeckExportRequest,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Export a deck to specified format."""
    result = await db.execute(
        select(Deck).where(
            and_(
                Deck.id == request.deck_id,
                Deck.owner_id == current_user.id,
            )
        )
    )
    deck = result.scalar_one_or_none()

    if deck is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deck not found",
        )

    exporter = DeckExporter(db)
    content = await exporter.export_deck(
        main_deck=deck.main_deck,
        sideboard=deck.sideboard,
        format=request.format.value,
        deck_name=deck.name,
    )

    extension = ".txt" if request.format in [DeckExportFormat.MTGO, DeckExportFormat.PLAIN] else ".txt"
    filename = f"{deck.name.replace(' ', '_')}{extension}"

    return DeckExportResponse(
        content=content,
        filename=filename,
        format=request.format.value,
    )


@router.post("/{deck_id}/validate", response_model=DeckValidationReport)
async def validate_deck(
    deck_id: UUID,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Validate a deck against Standard rules."""
    result = await db.execute(
        select(Deck).where(
            and_(
                Deck.id == deck_id,
                Deck.owner_id == current_user.id,
            )
        )
    )
    deck = result.scalar_one_or_none()

    if deck is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deck not found",
        )

    validator = DeckValidator(db)
    report = await validator.validate(deck.main_deck, deck.sideboard)

    # Update deck validation status
    deck.is_validated = report.is_valid
    deck.validation_errors = [e.model_dump() for e in report.errors] if report.errors else None
    await db.commit()

    return report


@router.post("/generate", response_model=DeckGenerateResponse)
async def generate_deck(
    request: DeckGenerateRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a deck using AI based on natural language prompt."""
    # Get user's preferred format
    format = "standard"
    if current_user and current_user.preferences:
        format = current_user.preferences.default_format or "standard"

    generator = DeckGenerator(db)
    result = await generator.generate(
        prompt=request.prompt,
        user_id=current_user.id if current_user else None,
        conversation_id=request.conversation_id,
        include_sideboard=request.include_sideboard,
        include_explanations=request.include_explanations,
        format=format,
    )
    return result


@router.post("/iterate", response_model=DeckIterateResponse)
async def iterate_deck(
    request: DeckIterateRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Modify an existing deck based on natural language instructions."""
    # Get user's preferred format
    format = "standard"
    if current_user and current_user.preferences:
        format = current_user.preferences.default_format or "standard"

    generator = DeckGenerator(db)
    result = await generator.iterate(
        modification=request.modification,
        conversation_id=request.conversation_id,
        deck_id=request.deck_id,
        user_id=current_user.id if current_user else None,
        format=format,
    )
    return result


@router.post("/{deck_id}/sideboard-matrix", response_model=SideboardMatrixResponse)
async def generate_sideboard_matrix(
    deck_id: UUID,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Generate a sideboard guide matrix for all current meta matchups."""
    from datetime import datetime

    # Fetch the deck
    result = await db.execute(
        select(Deck).where(
            and_(
                Deck.id == deck_id,
                Deck.owner_id == current_user.id,
            )
        )
    )
    deck = result.scalar_one_or_none()

    if deck is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Deck not found",
        )

    # Fetch current meta archetypes (top 10 by meta percentage)
    meta_result = await db.execute(
        select(MetaSnapshot)
        .where(MetaSnapshot.format == "standard")
        .order_by(MetaSnapshot.meta_percentage.desc())
        .limit(10)
    )
    meta_archetypes = meta_result.scalars().all()

    if not meta_archetypes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No meta data available. Please wait for meta scrape to complete.",
        )

    # Prepare meta archetype data for AI
    meta_archetype_data = [
        {
            "archetype": m.archetype,
            "meta_percentage": float(m.meta_percentage) if m.meta_percentage else 0,
            "key_cards": m.key_cards or [],
        }
        for m in meta_archetypes
    ]

    # Generate sideboard matrix using AI
    ai_service = AIService(db)
    matrix_data = await ai_service.generate_sideboard_matrix(
        deck_data={
            "name": deck.name,
            "archetype": deck.archetype,
            "main_deck": deck.main_deck,
            "sideboard": deck.sideboard,
            "strategy_summary": deck.strategy_summary,
        },
        archetype=deck.archetype or "Unknown",
        meta_archetypes=meta_archetype_data,
    )

    return SideboardMatrixResponse(
        deck_name=deck.name,
        deck_archetype=deck.archetype,
        generated_at=datetime.utcnow(),
        matchups=matrix_data.get("matchups", []),
        general_sideboard_notes=matrix_data.get("general_sideboard_notes", ""),
    )
