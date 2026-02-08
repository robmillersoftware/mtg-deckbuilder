"""Add context JSONB column to conversations for persistent strategy/phase tracking

Revision ID: 013
Revises: 012
Create Date: 2026-02-08

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '013'
down_revision = '012'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('conversations', sa.Column('context', JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column('conversations', 'context')
