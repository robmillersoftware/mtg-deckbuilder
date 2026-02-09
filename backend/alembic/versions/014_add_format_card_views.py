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

        # Create materialized view
        op.execute(f"""
            CREATE MATERIALIZED VIEW {view_name} AS
            SELECT *
            FROM cards
            WHERE legalities->>'{legality_key}' = 'legal'
        """)

        # Unique index on id — required for REFRESH MATERIALIZED VIEW CONCURRENTLY
        op.execute(f"""
            CREATE UNIQUE INDEX idx_{view_name}_id ON {view_name} (id)
        """)

        # Name index for text lookups
        op.execute(f"""
            CREATE INDEX idx_{view_name}_name ON {view_name} (name)
        """)

        # Lowercase name index for case-insensitive lookups
        op.execute(f"""
            CREATE INDEX idx_{view_name}_name_lower ON {view_name} (lower(name))
        """)

        # GIN index on colors for array containment queries
        op.execute(f"""
            CREATE INDEX idx_{view_name}_colors ON {view_name} USING gin (colors)
        """)

        # Vector index for semantic search (cosine distance)
        # Only index rows that actually have embeddings
        op.execute(f"""
            CREATE INDEX idx_{view_name}_embedding
            ON {view_name} USING hnsw (embedding vector_cosine_ops)
            WHERE embedding IS NOT NULL
        """)


def downgrade() -> None:
    for view_suffix in FORMAT_LEGALITY_MAP:
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS cards_{view_suffix} CASCADE")
