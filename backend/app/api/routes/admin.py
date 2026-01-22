from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4
import logging
import asyncio
import threading

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func

from app.db.session import get_db, async_session_factory
from app.models.user import User
from app.models.card import Card
from app.models.job import JobRun, JobStatus
from app.api.deps.auth import get_current_admin_user

logger = logging.getLogger(__name__)
router = APIRouter()


def run_async_job_in_thread(job_name: str):
    """Run an async job in a separate thread with its own event loop."""
    def thread_target():
        try:
            logger.info(f"Thread starting for job: {job_name}")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                from app.jobs.scheduler import run_job_manually
                result = loop.run_until_complete(run_job_manually(job_name))
                logger.info(f"Job {job_name} completed: {result}")
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Job {job_name} failed in thread: {e}", exc_info=True)

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    logger.info(f"Started thread for job: {job_name}")


@router.post("/jobs/{job_name}/run")
async def trigger_job(
    job_name: str,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Manually trigger a scheduled job.
    Only accessible to admin users.

    The job runs in the background with full retry logic and alerting.
    """
    valid_jobs = ["scryfall_sync", "mtgtop8_scrape"]
    if job_name not in valid_jobs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid job name. Valid options: {valid_jobs}",
        )

    from app.jobs.scheduler import run_job_manually

    # Create initial job run record
    run_id = str(uuid4())
    job_run = JobRun(
        job_name=job_name,
        run_id=run_id,
        status=JobStatus.PENDING.value,
    )
    db.add(job_run)
    await db.commit()

    # Run the job in a separate thread to ensure it executes
    run_async_job_in_thread(job_name)

    return {
        "run_id": run_id,
        "status": "running",
        "message": f"Job {job_name} has been started with retry logic enabled",
    }


@router.get("/jobs/history")
async def get_job_history(
    job_name: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get job execution history.
    Only accessible to admin users.
    """
    query = select(JobRun)

    if job_name:
        query = query.where(JobRun.job_name == job_name)
    if status_filter:
        query = query.where(JobRun.status == status_filter)

    query = query.order_by(desc(JobRun.created_at)).offset(offset).limit(limit)

    result = await db.execute(query)
    job_runs = result.scalars().all()

    return {
        "jobs": [
            {
                "id": str(jr.id),
                "job_name": jr.job_name,
                "run_id": jr.run_id,
                "status": jr.status,
                "started_at": jr.started_at.isoformat() if jr.started_at else None,
                "ended_at": jr.ended_at.isoformat() if jr.ended_at else None,
                "duration_seconds": jr.duration_seconds,
                "records_processed": jr.records_processed,
                "records_inserted": jr.records_inserted,
                "records_updated": jr.records_updated,
                "error_message": jr.error_message,
                "attempt_number": jr.attempt_number,
            }
            for jr in job_runs
        ],
        "total": len(job_runs),
        "limit": limit,
        "offset": offset,
    }


@router.get("/dashboard/jobs")
async def get_jobs_dashboard(
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get job metrics for the operations dashboard.
    Only accessible to admin users.
    """
    from sqlalchemy import func, case
    from datetime import timedelta

    job_names = ["scryfall_sync", "mtgtop8_scrape"]
    metrics = {}

    for job_name in job_names:
        # Get counts for last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)

        result = await db.execute(
            select(
                func.count().label("total"),
                func.sum(case((JobRun.status == JobStatus.SUCCESS.value, 1), else_=0)).label(
                    "success_count"
                ),
                func.sum(case((JobRun.status == JobStatus.FAILED.value, 1), else_=0)).label(
                    "failure_count"
                ),
            )
            .where(
                JobRun.job_name == job_name,
                JobRun.created_at >= thirty_days_ago,
            )
        )
        row = result.one()

        # Get last success timestamp
        last_success_result = await db.execute(
            select(JobRun.ended_at)
            .where(
                JobRun.job_name == job_name,
                JobRun.status == JobStatus.SUCCESS.value,
            )
            .order_by(desc(JobRun.ended_at))
            .limit(1)
        )
        last_success = last_success_result.scalar_one_or_none()

        metrics[job_name] = {
            "total_runs_30d": row.total or 0,
            "success_count_30d": row.success_count or 0,
            "failure_count_30d": row.failure_count or 0,
            "success_rate_30d": (
                (row.success_count / row.total * 100) if row.total and row.total > 0 else 0
            ),
            "last_success": last_success.isoformat() if last_success else None,
        }

    return metrics


@router.get("/embeddings/status")
async def get_embeddings_status(
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get status of card embeddings.
    """
    # Count total Standard-legal cards
    total_result = await db.execute(
        select(func.count()).select_from(Card).where(Card.is_standard_legal == True)
    )
    total_cards = total_result.scalar() or 0

    # Count cards with embeddings
    embedded_result = await db.execute(
        select(func.count())
        .select_from(Card)
        .where(Card.is_standard_legal == True)
        .where(Card.embedding.isnot(None))
    )
    embedded_cards = embedded_result.scalar() or 0

    return {
        "total_standard_cards": total_cards,
        "cards_with_embeddings": embedded_cards,
        "cards_missing_embeddings": total_cards - embedded_cards,
        "completion_percentage": (embedded_cards / total_cards * 100) if total_cards > 0 else 0,
    }


@router.post("/embeddings/backfill")
async def backfill_embeddings(
    batch_size: int = Query(50, ge=10, le=200),
    max_batches: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Backfill embeddings for cards that don't have them.
    Runs in the background to avoid timeout.
    """
    from app.services.embedding_service import get_embedding_service

    embedding_service = get_embedding_service()
    if not embedding_service.client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY environment variable.",
        )

    def run_backfill_in_thread():
        """Run backfill in a separate thread."""
        async def backfill_task():
            total_updated = 0
            for batch_num in range(max_batches):
                async with async_session_factory() as session:
                    result = await session.execute(
                        select(Card)
                        .where(Card.embedding.is_(None))
                        .where(Card.is_standard_legal == True)
                        .limit(batch_size)
                    )
                    cards = list(result.scalars().all())

                    if not cards:
                        logger.info(f"Backfill complete. Total cards updated: {total_updated}")
                        break

                    texts = []
                    for card in cards:
                        text = embedding_service._build_card_text(
                            name=card.name,
                            type_line=card.type_line,
                            oracle_text=card.oracle_text,
                            keywords=card.keywords,
                        )
                        texts.append(text)

                    embeddings = await embedding_service.get_embeddings_batch(texts, batch_size=batch_size)

                    batch_updated = 0
                    for card, embedding in zip(cards, embeddings):
                        if embedding:
                            card.embedding = embedding
                            batch_updated += 1

                    await session.commit()
                    total_updated += batch_updated
                    logger.info(f"Backfill batch {batch_num + 1}: updated {batch_updated} cards (total: {total_updated})")

        try:
            logger.info("Thread starting for embeddings backfill")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(backfill_task())
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Embeddings backfill failed in thread: {e}", exc_info=True)

    thread = threading.Thread(target=run_backfill_in_thread, daemon=True)
    thread.start()
    logger.info("Started thread for embeddings backfill")

    return {
        "status": "started",
        "message": f"Embedding backfill started. Processing up to {max_batches * batch_size} cards.",
        "batch_size": batch_size,
        "max_batches": max_batches,
    }
