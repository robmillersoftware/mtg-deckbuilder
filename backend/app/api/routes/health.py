from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.db.session import get_db
from app.models.job import JobRun, JobStatus

router = APIRouter()


@router.get("")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get("/jobs")
async def get_jobs_health(
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """
    Get health status of scheduled jobs.
    Returns 200 OK if all jobs are healthy, 503 if any are unhealthy.
    """
    job_names = ["scryfall_sync", "mtgtop8_scrape"]
    jobs_status = {}
    all_healthy = True

    for job_name in job_names:
        # Get last success
        success_result = await db.execute(
            select(JobRun)
            .where(
                JobRun.job_name == job_name,
                JobRun.status == JobStatus.SUCCESS.value,
            )
            .order_by(desc(JobRun.ended_at))
            .limit(1)
        )
        last_success = success_result.scalar_one_or_none()

        # Get last failure
        failure_result = await db.execute(
            select(JobRun)
            .where(
                JobRun.job_name == job_name,
                JobRun.status == JobStatus.FAILED.value,
            )
            .order_by(desc(JobRun.ended_at))
            .limit(1)
        )
        last_failure = failure_result.scalar_one_or_none()

        # Get next scheduled run (placeholder - would come from scheduler)
        next_scheduled = None

        # Determine health based on recent success
        is_healthy = True
        threshold = timedelta(hours=48) if job_name == "scryfall_sync" else timedelta(days=10)

        if last_success is None:
            is_healthy = False
        elif last_success.ended_at and (datetime.utcnow() - last_success.ended_at) > threshold:
            is_healthy = False

        if not is_healthy:
            all_healthy = False

        jobs_status[job_name] = {
            "status": "healthy" if is_healthy else "unhealthy",
            "last_success_timestamp": last_success.ended_at.isoformat() if last_success and last_success.ended_at else None,
            "last_failure_timestamp": last_failure.ended_at.isoformat() if last_failure and last_failure.ended_at else None,
            "next_scheduled_run": next_scheduled,
        }

    if not all_healthy:
        response.status_code = 503

    return jobs_status
