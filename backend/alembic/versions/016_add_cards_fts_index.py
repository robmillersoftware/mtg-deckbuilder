"""GIN full-text index on cards, for the semantic-search shortlist (works without OpenAI)."""

from alembic import op

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None

# Must match app.services.card_service.CARD_TSVECTOR exactly, or Postgres won't use the index.
CARD_TSVECTOR = (
    "to_tsvector('english', name || ' ' || coalesce(type_line, '') || ' ' || coalesce(oracle_text, ''))"
)


def upgrade() -> None:
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_cards_fts ON cards USING gin (({CARD_TSVECTOR}))")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_cards_fts")
