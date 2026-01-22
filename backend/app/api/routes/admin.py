from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.db.session import get_db
from app.models.user import User
from app.models.job import JobRun, JobStatus
from app.api.deps.auth import get_current_admin_user

router = APIRouter()


@router.post("/jobs/{job_name}/run")
async def trigger_job(
    job_name: str,
    background_tasks: BackgroundTasks,
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

    # Schedule the job to run in the background with retry logic
    async def run_job_task():
        try:
            await run_job_manually(job_name)
        except Exception as e:
            # Error is already logged and handled by job runner
            pass

    background_tasks.add_task(run_job_task)

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
