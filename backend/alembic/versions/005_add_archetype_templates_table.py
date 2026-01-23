"""Add archetype_templates table for deck generation

Revision ID: 005
Revises: 004
Create Date: 2026-01-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'archetype_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('archetype_category', sa.String(50), nullable=False),
        sa.Column('format', sa.String(50), nullable=False, server_default='standard'),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('computed_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('avg_lands', sa.Numeric(4, 1), nullable=False),
        sa.Column('avg_nonlands', sa.Numeric(4, 1), nullable=False),
        sa.Column('role_distribution', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('archetype_breakdown', postgresql.JSONB(), nullable=True),
    )

    # Indexes
    op.create_index('idx_archetype_template_format', 'archetype_templates', ['format'])

    # Unique constraint
    op.create_unique_constraint('uq_archetype_template', 'archetype_templates', ['archetype_category', 'format'])


def downgrade() -> None:
    op.drop_constraint('uq_archetype_template', 'archetype_templates', type_='unique')
    op.drop_index('idx_archetype_template_format', 'archetype_templates')
    op.drop_table('archetype_templates')
