"""Add commander column to decklists table for cEDH support

Revision ID: 006
Revises: 005
Create Date: 2026-01-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add commander column for cEDH/Commander format decks
    # Stores commander(s) as JSON array, e.g., ["Tymna the Weaver", "Kraum, Ludevic's Opus"]
    op.add_column(
        'decklists',
        sa.Column('commander', postgresql.JSONB(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('decklists', 'commander')
