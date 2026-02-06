"""Add card_meta_stats table for individual card representation tracking

Revision ID: 012
Revises: f243f38bb0ca
Create Date: 2026-02-06

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision = '012'
down_revision = '011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'card_meta_stats',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('card_name', sa.String(255), nullable=False),
        sa.Column('card_id', UUID(as_uuid=True), nullable=True),
        sa.Column('format', sa.String(50), nullable=False, server_default='standard'),
        sa.Column('snapshot_date', sa.Date(), nullable=False),
        sa.Column('deck_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_decks', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('meta_percentage', sa.Numeric(5, 2), nullable=False, server_default='0'),
        sa.Column('main_deck_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('sideboard_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_copies', sa.Numeric(3, 1), nullable=False, server_default='0'),
        sa.Column('archetypes', JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint('card_name', 'format', 'snapshot_date', name='uq_card_meta_stats'),
    )

    op.create_index('idx_card_meta_stats_format_date', 'card_meta_stats', ['format', 'snapshot_date'])
    op.create_index('idx_card_meta_stats_percentage', 'card_meta_stats', ['format', 'snapshot_date', 'meta_percentage'])


def downgrade() -> None:
    op.drop_index('idx_card_meta_stats_percentage', table_name='card_meta_stats')
    op.drop_index('idx_card_meta_stats_format_date', table_name='card_meta_stats')
    op.drop_table('card_meta_stats')
