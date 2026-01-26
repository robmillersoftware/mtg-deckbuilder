from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.models.user import User, VerificationToken, ResetToken, Preferences
from app.schemas.user import (
    UserCreate,
    UserResponse,
    UserLogin,
    Token,
    RefreshTokenRequest,
    PasswordReset,
    PasswordResetConfirm,
)
from app.core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    generate_verification_token,
    generate_reset_token,
)
from app.services.email import send_verification_email, send_password_reset_email

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user account. Only email and password required."""
    import re
    import secrets

    # Check if email already exists
    result = await db.execute(select(User).where(User.email == user_data.email.lower()))
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Generate username from email if not provided
    if user_data.username:
        username = user_data.username.lower()
    else:
        # Extract username from email (part before @)
        email_prefix = user_data.email.split("@")[0].lower()
        # Remove non-alphanumeric characters except underscore
        username = re.sub(r'[^a-z0-9_]', '', email_prefix)
        # Ensure minimum length
        if len(username) < 3:
            username = username + secrets.token_hex(2)

    # Check if username already exists, add suffix if needed
    base_username = username
    suffix = 0
    while True:
        result = await db.execute(select(User).where(User.username == username))
        if result.scalar_one_or_none() is None:
            break
        suffix += 1
        username = f"{base_username}{suffix}"
        if suffix > 100:  # Safety limit
            username = f"{base_username}_{secrets.token_hex(4)}"
            break

    # Create user
    user = User(
        email=user_data.email.lower(),
        username=username,
        hashed_password=get_password_hash(user_data.password),
        display_name=user_data.display_name,
    )
    db.add(user)
    await db.flush()

    # Create default preferences
    preferences = Preferences(user_id=user.id)
    db.add(preferences)

    # Create verification token
    token = generate_verification_token()
    verification_token = VerificationToken(
        user_id=user.id,
        token_hash=get_password_hash(token),
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )
    db.add(verification_token)

    await db.commit()
    await db.refresh(user)

    # Send verification email in background
    background_tasks.add_task(send_verification_email, user.email, token)

    return user


@router.post("/login", response_model=Token)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """Login with email and password."""
    result = await db.execute(
        select(User).where(User.email == credentials.email.lower())
    )
    user = result.scalar_one_or_none()

    # Generic error to not reveal whether email exists
    if user is None or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/verify/{token}")
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """Verify email with verification token."""
    # Find all non-expired verification tokens
    result = await db.execute(
        select(VerificationToken).where(
            VerificationToken.expires_at > datetime.utcnow()
        )
    )
    verification_tokens = result.scalars().all()

    # Find matching token
    for vt in verification_tokens:
        if verify_password(token, vt.token_hash):
            # Get user and mark as verified
            user_result = await db.execute(
                select(User).where(User.id == vt.user_id)
            )
            user = user_result.scalar_one_or_none()
            if user:
                user.is_verified = True
                await db.delete(vt)
                await db.commit()
                return {"message": "Email verified successfully"}

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid or expired verification token",
    )


@router.post("/password-reset/request")
async def request_password_reset(
    data: PasswordReset,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Request a password reset link."""
    result = await db.execute(
        select(User).where(User.email == data.email.lower())
    )
    user = result.scalar_one_or_none()

    # Always return success to not reveal whether email exists
    if user and user.is_verified:
        token = generate_reset_token()
        reset_token = ResetToken(
            user_id=user.id,
            token_hash=get_password_hash(token),
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        db.add(reset_token)
        await db.commit()

        background_tasks.add_task(send_password_reset_email, user.email, token)

    return {"message": "If the email exists, a reset link has been sent"}


@router.post("/password-reset/confirm")
async def confirm_password_reset(
    data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db),
):
    """Reset password with reset token."""
    if data.new_password != data.new_password_confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match",
        )

    # Find all non-expired, unused reset tokens
    result = await db.execute(
        select(ResetToken).where(
            ResetToken.expires_at > datetime.utcnow(),
            ResetToken.used_at.is_(None),
        )
    )
    reset_tokens = result.scalars().all()

    # Find matching token
    for rt in reset_tokens:
        if verify_password(data.token, rt.token_hash):
            # Get user and update password
            user_result = await db.execute(
                select(User).where(User.id == rt.user_id)
            )
            user = user_result.scalar_one_or_none()
            if user:
                user.hashed_password = get_password_hash(data.new_password)
                rt.used_at = datetime.utcnow()
                await db.commit()
                return {"message": "Password reset successfully"}

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid or expired reset token",
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """Refresh access token using refresh token."""
    from jose import jwt, JWTError
    from app.core.config import settings
    from uuid import UUID

    try:
        payload = jwt.decode(
            request.refresh_token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")

        if user_id is None or token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        user_uuid = UUID(user_id)
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    access_token = create_access_token(str(user.id))
    new_refresh_token = create_refresh_token(str(user.id))

    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
    )
