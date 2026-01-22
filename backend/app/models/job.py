import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    DateTime,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.session import Base


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class JobRun(Base):
    """
    Job execution log for scheduled tasks (Scryfall sync, mtgtop8 scrape, etc.).
    """

    __tablename__ = "job_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_name = Column(String(100), nullable=False, index=True)
    run_id = Column(String(100), nullable=False, index=True)
    status = Column(String(20), nullable=False, default=JobStatus.PENDING.value)

    # Timestamps
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)

    # Record counts
    records_inserted = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_deleted = Column(Integer, default=0)
    records_processed = Column(Integer, default=0)

    # Error tracking
    error_message = Column(Text, nullable=True)
    error_stack = Column(Text, nullable=True)
    warnings = Column(JSONB, nullable=True)

    # Retry info
    attempt_number = Column(Integer, default=1)
    next_retry_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_job_runs_name_status", "job_name", "status"),
        Index("idx_job_runs_created", "created_at"),
    )
