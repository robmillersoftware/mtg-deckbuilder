"""Add job retry columns

Revision ID: 002
Revises: 001
Create Date: 2024-01-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add columns for retry functionality
    op.add_column('job_runs', sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('job_runs', sa.Column('error_stack', sa.Text(), nullable=True))
    op.add_column('job_runs', sa.Column('warnings', postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column('job_runs', 'warnings')
    op.drop_column('job_runs', 'error_stack')
    op.drop_column('job_runs', 'next_retry_at')
