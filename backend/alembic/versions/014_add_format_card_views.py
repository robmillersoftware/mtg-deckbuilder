"""Add materialized views for format-based card filtering.

Each format gets its own materialized view containing only legal cards.
This makes format filtering structural — the LLM literally cannot access
cards that aren't legal in the selected format.

Views are refreshed after every Scryfall sync and mtgtop8 scrape.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None

# Formats and their corresponding Scryfall legality keys
FORMAT_LEGALITY_MAP = {
    "standard": "standard",
    "modern": "modern",
    "legacy": "legacy",
    "historic": "historic",
    "commander": "commander",  # used by cEDH
}


def upgrade() -> None:
    for view_suffix, legality_key in FORMAT_LEGALITY_MAP.items():
        view_name = f"cards_{view_suffix}"

        # Create materialized view (IF NOT EXISTS for idempotent re-runs
        # after partial failures)
        op.execute(f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS {view_name} AS
            SELECT *
            FROM cards
            WHERE legalities->>'{legality_key}' = 'legal'
        """)

        # Unique index on id — required for REFRESH MATERIALIZED VIEW CONCURRENTLY
        op.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{view_name}_id ON {view_name} (id)
        """)

        # Name index for text lookups
        op.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{view_name}_name ON {view_name} (name)
        """)

        # Lowercase name index for case-insensitive lookups
        op.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{view_name}_name_lower ON {view_name} (lower(name))
        """)

        # GIN index on colors for array containment queries
        op.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{view_name}_colors ON {view_name} USING gin (colors)
        """)

        # NOTE: HNSW vector indexes are omitted intentionally.
        # They consume too much disk/shared memory for constrained hosting
        # environments (e.g. Railway). The materialized views are already
        # small enough that sequential vector scans are fast.

    # Drop any HNSW indexes that may have been created by a previous
    # version of this migration before it failed mid-way.
    for view_suffix in FORMAT_LEGALITY_MAP:
        view_name = f"cards_{view_suffix}"
        op.execute(f"DROP INDEX IF EXISTS idx_{view_name}_embedding")


def downgrade() -> None:
    for view_suffix in FORMAT_LEGALITY_MAP:
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS cards_{view_suffix} CASCADE")
