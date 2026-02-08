from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.user import User
from app.api.deps.auth import get_current_user
from app.services.guided_builder import GuidedBuilder
from app.schemas.guided_build import (
    StartGuidedBuildRequest,
    AdvanceStepRequest,
    GuidedBuildStepResponse,
    GuidedBuildCompleteResponse,
)

router = APIRouter()


@router.post("/start", response_model=GuidedBuildStepResponse)
async def start_guided_build(
    request: StartGuidedBuildRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a new guided deck building session."""
    builder = GuidedBuilder(db)
    result = await builder.start_session(
        user_id=current_user.id if current_user else None,
        format=request.format,
    )
    return result


@router.post("/advance", response_model=GuidedBuildStepResponse)
async def advance_guided_build(
    request: AdvanceStepRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit selections for the current step and advance to the next."""
    builder = GuidedBuilder(db)
    try:
        result = await builder.advance_step(
            session_id=request.session_id,
            selections=request.selections,
            user_id=current_user.id if current_user else None,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{session_id}/back", response_model=GuidedBuildStepResponse)
async def go_back_guided_build(
    session_id: UUID,
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Go back to the previous step."""
    builder = GuidedBuilder(db)
    try:
        result = await builder.go_back(
            session_id=session_id,
            user_id=current_user.id if current_user else None,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{session_id}/complete", response_model=GuidedBuildCompleteResponse)
async def complete_guided_build(
    session_id: UUID,
    deck_name: Optional[str] = None,
    save: bool = False,
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Complete the guided build and optionally save the deck."""
    builder = GuidedBuilder(db)
    try:
        result = await builder.complete_build(
            session_id=session_id,
            deck_name=deck_name,
            user_id=current_user.id if current_user else None,
            save=save,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
