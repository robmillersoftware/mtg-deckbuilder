"""Add simulation_runs table for persisted game simulations

Revision ID: 008
Revises: 007
Create Date: 2026-01-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision: str = '008'
down_revision: Union[str, None] = '007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'simulation_runs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),

        # Status
        sa.Column('status', sa.String(20), nullable=False, default='pending'),

        # Configuration
        sa.Column('your_deck_id', UUID(as_uuid=True), sa.ForeignKey('decks.id', ondelete='SET NULL'), nullable=True),
        sa.Column('your_deck_name', sa.String(255), nullable=False),
        sa.Column('your_deck_snapshot', JSONB, nullable=True),
        sa.Column('opponent_deck_name', sa.String(255), nullable=False),
        sa.Column('opponent_archetype', sa.String(255), nullable=True),
        sa.Column('opponent_deck_snapshot', JSONB, nullable=True),
        sa.Column('format', sa.String(50), nullable=False, default='standard'),
        sa.Column('num_games', sa.Integer, nullable=False, default=5),
        sa.Column('include_sideboard_games', sa.Integer, nullable=False, default=1),

        # Progress
        sa.Column('games_completed', sa.Integer, nullable=False, default=0),
        sa.Column('current_game_turn', sa.Integer, nullable=True),

        # Results
        sa.Column('your_wins', sa.Integer, nullable=True),
        sa.Column('opponent_wins', sa.Integer, nullable=True),
        sa.Column('win_rate', sa.Numeric(5, 4), nullable=True),
        sa.Column('average_game_length', sa.Numeric(5, 2), nullable=True),
        sa.Column('matchup_assessment', sa.String(20), nullable=True),

        # Detailed results
        sa.Column('games', JSONB, nullable=True),
        sa.Column('key_cards_for_you', JSONB, nullable=True),
        sa.Column('key_cards_against_you', JSONB, nullable=True),
        sa.Column('sideboard_guide', JSONB, nullable=True),
        sa.Column('strategic_advice', JSONB, nullable=True),
        sa.Column('mulligan_advice', sa.String(1000), nullable=True),

        # Error tracking
        sa.Column('error_message', sa.String(1000), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
    )

    # Create indexes
    op.create_index('idx_simulation_runs_user_status', 'simulation_runs', ['user_id', 'status'])
    op.create_index('idx_simulation_runs_user_created', 'simulation_runs', ['user_id', 'created_at'])


def downgrade() -> None:
    op.drop_index('idx_simulation_runs_user_created')
    op.drop_index('idx_simulation_runs_user_status')
    op.drop_table('simulation_runs')
