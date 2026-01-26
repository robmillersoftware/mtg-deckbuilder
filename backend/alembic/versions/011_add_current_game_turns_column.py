"""Add current_game_turns column for live streaming

Revision ID: 011
Revises: 010
Create Date: 2024-01-26

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '011'
down_revision = '010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'simulation_runs',
        sa.Column('current_game_turns', JSONB, nullable=True)
    )


def downgrade() -> None:
    op.drop_column('simulation_runs', 'current_game_turns')
