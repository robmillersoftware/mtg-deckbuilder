from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func

from app.db.session import get_db, async_session_factory
from app.models.user import User
from app.models.card import Card
from app.models.job import JobRun, JobStatus
from app.api.deps.auth import get_current_admin_user
from app.core.queue import job_queue, get_queue_stats
from app.jobs.rq_tasks import get_task_function

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/jobs/{job_name}/run")
async def trigger_job(
    job_name: str,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Manually trigger a scheduled job via RQ (Redis Queue).
    Only accessible to admin users.

    The job runs in a separate RQ worker process, independent of the web server,
    ensuring it persists beyond HTTP request completion.
    """
    valid_jobs = ["scryfall_sync", "mtgtop8_scrape"]
    if job_name not in valid_jobs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid job name. Valid options: {valid_jobs}",
        )

    # Get the task function for this job
    try:
        task_func = get_task_function(job_name)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Create initial job run record
    run_id = str(uuid4())
    job_run = JobRun(
        job_name=job_name,
        run_id=run_id,
        status=JobStatus.PENDING.value,
    )
    db.add(job_run)
    await db.commit()

    # Enqueue the job to RQ - this runs in a separate worker process
    try:
        rq_job = job_queue.enqueue(
            task_func,
            job_timeout=3600,  # 1 hour timeout
            job_id=run_id,  # Use our run_id as RQ job ID for tracking
            meta={"run_id": run_id, "job_name": job_name},
        )
        logger.info(f"Enqueued job {job_name} with RQ job ID: {rq_job.id}")
    except Exception as e:
        logger.error(f"Failed to enqueue job {job_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to enqueue job. Is Redis running? Error: {str(e)}",
        )

    return {
        "run_id": run_id,
        "rq_job_id": rq_job.id,
        "status": "queued",
        "message": f"Job {job_name} has been queued for execution by RQ worker",
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
    Runs in RQ worker to avoid timeout.
    """
    from app.services.embedding_service import get_embedding_service
    from app.jobs.rq_tasks import run_embeddings_backfill_task

    embedding_service = get_embedding_service()
    if not embedding_service.client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY environment variable.",
        )

    # Enqueue to RQ
    try:
        run_id = str(uuid4())
        rq_job = job_queue.enqueue(
            run_embeddings_backfill_task,
            batch_size,
            max_batches,
            job_timeout=7200,  # 2 hour timeout for large backfills
            job_id=run_id,
            meta={"run_id": run_id, "job_name": "embeddings_backfill"},
        )
        logger.info(f"Enqueued embeddings backfill with RQ job ID: {rq_job.id}")
    except Exception as e:
        logger.error(f"Failed to enqueue embeddings backfill: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to enqueue job. Is Redis running? Error: {str(e)}",
        )

    return {
        "status": "queued",
        "rq_job_id": rq_job.id,
        "message": f"Embedding backfill queued. Processing up to {max_batches * batch_size} cards.",
        "batch_size": batch_size,
        "max_batches": max_batches,
    }


@router.get("/queue/status")
async def get_queue_status(
    current_user: User = Depends(get_current_admin_user),
):
    """
    Get RQ queue statistics.
    Shows pending, failed, and finished jobs.
    """
    try:
        stats = get_queue_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to get queue stats. Is Redis running? Error: {str(e)}",
        )


@router.get("/queue/job/{job_id}")
async def get_rq_job_status(
    job_id: str,
    current_user: User = Depends(get_current_admin_user),
):
    """
    Get status of a specific RQ job.
    """
    from rq.job import Job
    from app.core.queue import redis_conn

    try:
        job = Job.fetch(job_id, connection=redis_conn)
        return {
            "job_id": job.id,
            "status": job.get_status(),
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
            "result": job.result if job.is_finished else None,
            "exc_info": job.exc_info if job.is_failed else None,
            "meta": job.meta,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found or error fetching: {str(e)}",
        )
