import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


async def send_verification_email(email: str, token: str) -> bool:
    """Send email verification link to user."""
    verification_url = f"{settings.APP_URL}/verify-email?token={token}"

    if not settings.SENDGRID_API_KEY:
        # Log for development
        logger.info(f"[DEV] Verification email for {email}: {verification_url}")
        return True

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        message = Mail(
            from_email=settings.SENDGRID_FROM_EMAIL,
            to_emails=email,
            subject=f"Verify your {settings.APP_NAME} account",
            html_content=f"""
            <h1>Welcome to {settings.APP_NAME}!</h1>
            <p>Please verify your email address by clicking the link below:</p>
            <p><a href="{verification_url}">Verify Email</a></p>
            <p>Or copy and paste this URL into your browser:</p>
            <p>{verification_url}</p>
            <p>This link will expire in 24 hours.</p>
            """,
        )

        sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
        response = sg.send(message)
        return response.status_code == 202

    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        return False


async def send_password_reset_email(email: str, token: str) -> bool:
    """Send password reset link to user."""
    reset_url = f"{settings.APP_URL}/reset-password?token={token}"

    if not settings.SENDGRID_API_KEY:
        # Log for development
        logger.info(f"[DEV] Password reset email for {email}: {reset_url}")
        return True

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        message = Mail(
            from_email=settings.SENDGRID_FROM_EMAIL,
            to_emails=email,
            subject=f"Reset your {settings.APP_NAME} password",
            html_content=f"""
            <h1>Password Reset Request</h1>
            <p>You requested to reset your password for {settings.APP_NAME}.</p>
            <p>Click the link below to reset your password:</p>
            <p><a href="{reset_url}">Reset Password</a></p>
            <p>Or copy and paste this URL into your browser:</p>
            <p>{reset_url}</p>
            <p>This link will expire in 1 hour.</p>
            <p>If you didn't request this, you can safely ignore this email.</p>
            """,
        )

        sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
        response = sg.send(message)
        return response.status_code == 202

    except Exception as e:
        logger.error(f"Failed to send password reset email: {e}")
        return False


async def send_alert_email(subject: str, body: str, to_email: Optional[str] = None) -> bool:
    """Send alert email to admin."""
    recipient = to_email or settings.ADMIN_EMAIL

    if not settings.SENDGRID_API_KEY:
        logger.info(f"[DEV] Alert email to {recipient}: {subject}")
        return True

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        message = Mail(
            from_email=settings.SENDGRID_FROM_EMAIL,
            to_emails=recipient,
            subject=f"[{settings.APP_NAME}] {subject}",
            html_content=f"<pre>{body}</pre>",
        )

        sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
        response = sg.send(message)
        return response.status_code == 202

    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")
        return False
