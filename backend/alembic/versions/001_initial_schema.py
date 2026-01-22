"""Initial schema

Revision ID: 001
Revises:
Create Date: 2024-01-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('username', sa.String(50), unique=True, nullable=False, index=True),
        sa.Column('display_name', sa.String(100)),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('avatar_url', sa.String(500)),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_verified', sa.Boolean(), default=False),
        sa.Column('is_superuser', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # User verification tokens
    op.create_table(
        'verification_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('token_hash', sa.String(255), nullable=False, index=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Password reset tokens
    op.create_table(
        'reset_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('token_hash', sa.String(255), nullable=False, index=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # User preferences
    op.create_table(
        'preferences',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('language', sa.String(10), default='en'),
        sa.Column('theme', sa.String(10), default='light'),
        sa.Column('default_format', sa.String(20), default='standard'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Cards table
    op.create_table(
        'cards',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('scryfall_id', sa.String(50), unique=True, nullable=False, index=True),
        sa.Column('oracle_id', sa.String(50), index=True),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('mana_cost', sa.String(50)),
        sa.Column('cmc', sa.Numeric(4, 1)),
        sa.Column('type_line', sa.String(255)),
        sa.Column('oracle_text', sa.Text()),
        sa.Column('power', sa.String(10)),
        sa.Column('toughness', sa.String(10)),
        sa.Column('colors', postgresql.ARRAY(sa.String(1))),
        sa.Column('color_identity', postgresql.ARRAY(sa.String(1))),
        sa.Column('keywords', postgresql.ARRAY(sa.String(50))),
        sa.Column('set_code', sa.String(10)),
        sa.Column('set_name', sa.String(100)),
        sa.Column('collector_number', sa.String(20)),
        sa.Column('rarity', sa.String(20)),
        sa.Column('image_uri', sa.String(500)),
        sa.Column('image_uri_small', sa.String(500)),
        sa.Column('image_uri_art_crop', sa.String(500)),
        sa.Column('price_usd', sa.Numeric(10, 2)),
        sa.Column('price_usd_foil', sa.Numeric(10, 2)),
        sa.Column('is_standard_legal', sa.Boolean(), default=False, index=True),
        sa.Column('legalities', postgresql.JSONB()),
        sa.Column('scryfall_uri', sa.String(500)),
        sa.Column('embedding', Vector(1536)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Decks table
    op.create_table(
        'decks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('owner_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('format', sa.String(50), default='standard'),
        sa.Column('archetype', sa.String(100)),
        sa.Column('main_deck', postgresql.JSONB(), default=[]),
        sa.Column('sideboard', postgresql.JSONB(), default=[]),
        sa.Column('strategy_summary', sa.Text()),
        sa.Column('card_explanations', postgresql.JSONB()),
        sa.Column('matchup_notes', postgresql.JSONB()),
        sa.Column('visibility', sa.String(20), default='private'),
        sa.Column('share_token', sa.String(50), unique=True, index=True),
        sa.Column('is_validated', sa.Boolean(), default=False),
        sa.Column('validation_errors', postgresql.JSONB()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Conversations table
    op.create_table(
        'conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), index=True),
        sa.Column('summary', sa.String(500)),
        sa.Column('messages', postgresql.JSONB(), default=[]),
        sa.Column('current_deck', postgresql.JSONB()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Events table (tournaments)
    op.create_table(
        'events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('mtgtop8_id', sa.String(50), unique=True, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('date', sa.Date()),
        sa.Column('format', sa.String(50)),
        sa.Column('player_count', sa.Integer()),
        sa.Column('source_url', sa.String(500)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Decklists table (tournament decklists)
    op.create_table(
        'decklists',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('event_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('events.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('mtgtop8_deck_id', sa.String(50), unique=True, index=True),
        sa.Column('archetype', sa.String(100)),
        sa.Column('placement', sa.Integer()),
        sa.Column('player_name', sa.String(255)),
        sa.Column('main_deck', postgresql.JSONB(), default=[]),
        sa.Column('sideboard', postgresql.JSONB(), default=[]),
        sa.Column('source_url', sa.String(500)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Meta snapshots table
    op.create_table(
        'meta_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('format', sa.String(50), nullable=False),
        sa.Column('archetype', sa.String(100), nullable=False),
        sa.Column('meta_percentage', sa.Numeric(5, 2)),
        sa.Column('sample_size', sa.Integer()),
        sa.Column('avg_finish', sa.Numeric(5, 2)),
        sa.Column('key_cards', postgresql.ARRAY(sa.String(100))),
        sa.Column('snapshot_date', sa.Date(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_meta_snapshots_format_archetype', 'meta_snapshots', ['format', 'archetype'])

    # Card co-occurrence table
    op.create_table(
        'card_cooccurrence',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('card_a', sa.String(255), nullable=False),
        sa.Column('card_b', sa.String(255), nullable=False),
        sa.Column('card1_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('cards.id', ondelete='SET NULL'), nullable=True),
        sa.Column('card2_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('cards.id', ondelete='SET NULL'), nullable=True),
        sa.Column('cooccurrence_count', sa.Integer(), default=0),
        sa.Column('format', sa.String(50)),
        sa.Column('last_updated', sa.Date()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('card1_id', 'card2_id', 'format', name='uq_card_cooccurrence'),
    )
    op.create_index('ix_card_cooccurrence_card_a', 'card_cooccurrence', ['card_a', 'format'])
    op.create_index('ix_card_cooccurrence_card1', 'card_cooccurrence', ['card1_id'])
    op.create_index('ix_card_cooccurrence_card2', 'card_cooccurrence', ['card2_id'])

    # Job runs table
    op.create_table(
        'job_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('job_name', sa.String(100), nullable=False),
        sa.Column('run_id', sa.String(50), unique=True, index=True),
        sa.Column('status', sa.String(20), default='pending'),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('ended_at', sa.DateTime(timezone=True)),
        sa.Column('duration_seconds', sa.Integer()),
        sa.Column('records_processed', sa.Integer()),
        sa.Column('records_inserted', sa.Integer()),
        sa.Column('records_updated', sa.Integer()),
        sa.Column('attempt_number', sa.Integer(), default=1),
        sa.Column('error_message', sa.Text()),
        sa.Column('result_summary', postgresql.JSONB()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_job_runs_job_name_created', 'job_runs', ['job_name', 'created_at'])


def downgrade() -> None:
    op.drop_table('job_runs')
    op.drop_table('card_cooccurrence')
    op.drop_table('meta_snapshots')
    op.drop_table('decklists')
    op.drop_table('events')
    op.drop_table('conversations')
    op.drop_table('decks')
    op.drop_table('cards')
    op.drop_table('preferences')
    op.drop_table('reset_tokens')
    op.drop_table('verification_tokens')
    op.drop_table('users')
    op.execute('DROP EXTENSION IF EXISTS vector')
