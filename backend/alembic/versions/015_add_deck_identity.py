"""Add identity column to decks for Jev deck-fit (theme tags, key cards, user overrides)."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("decks", sa.Column("identity", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("decks", "identity")
