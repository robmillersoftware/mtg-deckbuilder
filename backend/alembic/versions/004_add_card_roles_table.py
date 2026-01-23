"""Add card_roles table for role classification

Revision ID: 004
Revises: f243f38bb0ca
Create Date: 2026-01-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = 'f243f38bb0ca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'card_roles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('card_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('cards.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.String(50), nullable=False),
        sa.Column('efficiency', sa.Integer(), nullable=True),
        sa.Column('confidence', sa.Numeric(3, 2), nullable=True),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )

    # Indexes
    op.create_index('idx_card_roles_card_id', 'card_roles', ['card_id'])
    op.create_index('idx_card_roles_role', 'card_roles', ['role'])
    op.create_index('idx_card_role_lookup', 'card_roles', ['role', 'efficiency'])

    # Unique constraint
    op.create_unique_constraint('uq_card_role', 'card_roles', ['card_id', 'role'])


def downgrade() -> None:
    op.drop_constraint('uq_card_role', 'card_roles', type_='unique')
    op.drop_index('idx_card_role_lookup', 'card_roles')
    op.drop_index('idx_card_roles_role', 'card_roles')
    op.drop_index('idx_card_roles_card_id', 'card_roles')
    op.drop_table('card_roles')
