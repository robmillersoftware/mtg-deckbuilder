import os
import uuid
from typing import Optional
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.models.user import User, Preferences
from app.schemas.user import (
    UserResponse,
    UserUpdate,
    PreferencesUpdate,
    PreferencesResponse,
    EmailChangeRequest,
    EmailChangeConfirm,
    PasswordChangeRequest,
)
from app.api.deps.auth import get_current_user_required
from app.core.config import settings
from app.core.security import generate_verification_token, verify_password, get_password_hash
from app.services.email import send_verification_email

# Directory for uploaded avatars
AVATAR_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "uploads", "avatars")
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_AVATAR_SIZE = 5 * 1024 * 1024  # 5MB

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user_required),
):
    """Get current user's profile."""
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_current_user_profile(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Update current user's profile."""
    if user_data.display_name is not None:
        if len(user_data.display_name.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Display name cannot be empty",
            )
        current_user.display_name = user_data.display_name

    if user_data.avatar_url is not None:
        current_user.avatar_url = user_data.avatar_url

    await db.commit()
    await db.refresh(current_user)

    return current_user


@router.get("/me/preferences", response_model=PreferencesResponse)
async def get_user_preferences(
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Get current user's preferences."""
    result = await db.execute(
        select(Preferences).where(Preferences.user_id == current_user.id)
    )
    preferences = result.scalar_one_or_none()

    if preferences is None:
        # Create default preferences
        preferences = Preferences(user_id=current_user.id)
        db.add(preferences)
        await db.commit()
        await db.refresh(preferences)

    return preferences


@router.patch("/me/preferences", response_model=PreferencesResponse)
async def update_user_preferences(
    prefs_data: PreferencesUpdate,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Update current user's preferences."""
    result = await db.execute(
        select(Preferences).where(Preferences.user_id == current_user.id)
    )
    preferences = result.scalar_one_or_none()

    if preferences is None:
        preferences = Preferences(user_id=current_user.id)
        db.add(preferences)

    if prefs_data.language is not None:
        preferences.language = prefs_data.language
    if prefs_data.theme is not None:
        preferences.theme = prefs_data.theme
    if prefs_data.default_format is not None:
        preferences.default_format = prefs_data.default_format

    await db.commit()
    await db.refresh(preferences)

    return preferences


@router.post("/me/avatar", response_model=UserResponse)
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Upload a new avatar image for the current user."""
    # Validate file type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > MAX_AVATAR_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size is {MAX_AVATAR_SIZE // (1024 * 1024)}MB",
        )

    # Ensure upload directory exists
    os.makedirs(AVATAR_UPLOAD_DIR, exist_ok=True)

    # Generate unique filename
    ext = file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg"
    filename = f"{current_user.id}_{uuid.uuid4()}.{ext}"
    filepath = os.path.join(AVATAR_UPLOAD_DIR, filename)

    # Save file
    with open(filepath, "wb") as f:
        f.write(content)

    # Update user avatar URL
    # In production, this would be a CDN URL or cloud storage URL
    avatar_url = f"{settings.APP_URL}/uploads/avatars/{filename}"
    current_user.avatar_url = avatar_url

    await db.commit()
    await db.refresh(current_user)

    return current_user


@router.post("/me/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    request: PasswordChangeRequest,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Change the current user's password."""
    # Verify current password
    if not verify_password(request.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Verify new passwords match
    if request.new_password != request.new_password_confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New passwords do not match",
        )

    # Update password
    current_user.hashed_password = get_password_hash(request.new_password)
    await db.commit()

    return {"message": "Password changed successfully"}


@router.post("/me/email-change", status_code=status.HTTP_202_ACCEPTED)
async def request_email_change(
    request: EmailChangeRequest,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """
    Request an email change. Sends a verification link to the new email address.
    The email will only be updated after the user clicks the verification link.
    """
    # Check if new email is already in use
    result = await db.execute(
        select(User).where(User.email == request.new_email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This email address is already in use",
        )

    # Generate verification token
    token = generate_verification_token()

    # Store the pending email change in user model
    # (In production, you might use a separate table for pending changes)
    current_user.pending_email = request.new_email
    current_user.email_change_token = token
    current_user.email_change_token_expires = datetime.now(timezone.utc) + timedelta(hours=24)

    await db.commit()

    # Send verification email to new address
    await send_verification_email(request.new_email, token)

    return {"message": "Verification email sent to your new email address"}


@router.post("/me/email-change/confirm", response_model=UserResponse)
async def confirm_email_change(
    request: EmailChangeConfirm,
    current_user: User = Depends(get_current_user_required),
    db: AsyncSession = Depends(get_db),
):
    """Confirm email change using the verification token."""
    # Validate token
    if not current_user.email_change_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pending email change request",
        )

    if current_user.email_change_token != request.token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token",
        )

    if current_user.email_change_token_expires and current_user.email_change_token_expires < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification token has expired",
        )

    # Update email
    current_user.email = current_user.pending_email
    current_user.pending_email = None
    current_user.email_change_token = None
    current_user.email_change_token_expires = None

    await db.commit()
    await db.refresh(current_user)

    return current_user
