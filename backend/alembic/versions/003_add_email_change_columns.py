"""Add email change columns to users

Revision ID: 003
Revises: 002
Create Date: 2024-01-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add columns for email change functionality
    op.add_column('users', sa.Column('pending_email', sa.String(255), nullable=True))
    op.add_column('users', sa.Column('email_change_token', sa.String(255), nullable=True))
    op.add_column('users', sa.Column('email_change_token_expires', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'email_change_token_expires')
    op.drop_column('users', 'email_change_token')
    op.drop_column('users', 'pending_email')
