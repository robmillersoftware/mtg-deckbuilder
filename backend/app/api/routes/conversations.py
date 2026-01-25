from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.db.session import get_db
from app.models.conversation import Conversation
from app.models.user import User
from app.schemas.conversation import (
    ConversationResponse,
    ConversationListResponse,
    ChatRequest,
    ChatResponse,
    CardExplanationRequest,
    CardExplanationResponse,
)
from app.api.deps.auth import get_current_user_required, get_current_user
from app.services.chat_service import ChatService

router = APIRouter()


@router.get("", response_model=List[ConversationListResponse])
async def list_conversations(
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """List all conversations for the current user."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    conversations = result.scalars().all()

    return [
        ConversationListResponse(
            id=conv.id,
            summary=conv.summary,
            message_count=conv.get_message_count(),
            has_deck=conv.current_deck is not None,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        )
        for conv in conversations
    ]


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific conversation by ID."""
    result = await db.execute(
        select(Conversation).where(
            and_(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id,
            )
        )
    )
    conversation = result.scalar_one_or_none()

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation."""
    result = await db.execute(
        select(Conversation).where(
            and_(
                Conversation.id == conversation_id,
                Conversation.user_id == current_user.id,
            )
        )
    )
    conversation = result.scalar_one_or_none()

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    await db.delete(conversation)
    await db.commit()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a chat message and receive a response.
    This is the main interface for deck building interactions.
    """
    chat_service = ChatService(db)
    response = await chat_service.process_message(
        message=request.message,
        conversation_id=request.conversation_id,
        user_id=current_user.id if current_user else None,
        format=request.format or "standard",
    )
    return response


@router.post("/explain-card", response_model=CardExplanationResponse)
async def explain_card(
    request: CardExplanationRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get an explanation for a specific card in the context of the current deck.
    """
    # Get conversation with deck context
    query = select(Conversation).where(Conversation.id == request.conversation_id)
    if current_user:
        query = query.where(Conversation.user_id == current_user.id)

    result = await db.execute(query)
    conversation = result.scalar_one_or_none()

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    if conversation.current_deck is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No deck in current conversation context",
        )

    chat_service = ChatService(db)
    explanation = await chat_service.explain_card(
        card_name=request.card_name,
        deck=conversation.current_deck,
    )

    return explanation
