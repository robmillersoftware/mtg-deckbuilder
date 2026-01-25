"""Add commander column to decks table for cEDH support

Revision ID: 007
Revises: 006
Create Date: 2026-01-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '007'
down_revision: Union[str, None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add commander column for user-generated cEDH/Commander format decks
    op.add_column(
        'decks',
        sa.Column('commander', sa.String(255), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('decks', 'commander')
