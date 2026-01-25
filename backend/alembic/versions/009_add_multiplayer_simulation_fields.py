"""Add multiplayer simulation support fields

Revision ID: 009
Revises: 008
Create Date: 2026-01-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '009'
down_revision: Union[str, None] = '008'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add multiplayer configuration fields
    op.add_column('simulation_runs', sa.Column('num_players', sa.Integer, nullable=False, server_default='2'))
    op.add_column('simulation_runs', sa.Column('opponent_deck_names', JSONB, nullable=True))
    op.add_column('simulation_runs', sa.Column('opponent_archetypes', JSONB, nullable=True))
    op.add_column('simulation_runs', sa.Column('opponent_deck_snapshots', JSONB, nullable=True))

    # Add multiplayer results fields
    op.add_column('simulation_runs', sa.Column('first_place_count', sa.Integer, nullable=True))
    op.add_column('simulation_runs', sa.Column('your_placement_avg', sa.Numeric(3, 2), nullable=True))


def downgrade() -> None:
    op.drop_column('simulation_runs', 'your_placement_avg')
    op.drop_column('simulation_runs', 'first_place_count')
    op.drop_column('simulation_runs', 'opponent_deck_snapshots')
    op.drop_column('simulation_runs', 'opponent_archetypes')
    op.drop_column('simulation_runs', 'opponent_deck_names')
    op.drop_column('simulation_runs', 'num_players')
