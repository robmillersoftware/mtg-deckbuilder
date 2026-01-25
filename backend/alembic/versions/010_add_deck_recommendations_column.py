"""Add deck_recommendations column to simulation_runs

Revision ID: 010
Revises: 009
Create Date: 2026-01-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '010'
down_revision: Union[str, None] = '009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('simulation_runs', sa.Column('deck_recommendations', JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column('simulation_runs', 'deck_recommendations')
