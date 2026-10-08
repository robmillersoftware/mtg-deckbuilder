"""Replace the LLM-narrated simulation_runs with Forge simulation runs (the old table holds no data)."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("simulation_runs")
    op.create_table(
        "simulation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("kind", sa.String(10), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("format", sa.String(50), nullable=False, server_default="standard"),
        sa.Column("deck", postgresql.JSONB, nullable=False),
        sa.Column("opponents", postgresql.JSONB, nullable=True),
        sa.Column("games_per_matchup", sa.Integer, nullable=False),
        sa.Column("options", postgresql.JSONB, nullable=True),
        sa.Column("progress", postgresql.JSONB, nullable=True),
        sa.Column("report", postgresql.JSONB, nullable=True),
        sa.Column("final_deck", postgresql.JSONB, nullable=True),
        sa.Column("stop_requested", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_simulation_runs_user", "simulation_runs", ["user_id", "created_at"])


def downgrade() -> None:
    # The LLM simulator's table is not restored; its code is gone.
    op.drop_index("idx_simulation_runs_user", table_name="simulation_runs")
    op.drop_table("simulation_runs")
