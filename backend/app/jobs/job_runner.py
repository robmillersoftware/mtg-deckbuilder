"""
Job Runner with Retry Logic and Alerting

Provides a wrapper for scheduled jobs that implements:
- Exponential backoff retry (1 min, 2 min, 4 min - up to 3 attempts)
- Alert notifications on final failure
- Structured logging for all attempts
"""

import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, Optional
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_factory
from app.models.job import JobRun, JobStatus
from app.services.email import send_alert_email
from app.core.config import settings

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BASE_RETRY_DELAY_SECONDS = 60  # 1 minute base delay


class JobRunner:
    """
    Wraps job functions with retry logic and alerting.
    """

    def __init__(
        self,
        job_name: str,
        job_func: Callable,
        max_retries: int = MAX_RETRIES,
        base_delay: int = BASE_RETRY_DELAY_SECONDS,
        run_id: Optional[str] = None,
    ):
        self.job_name = job_name
        self.job_func = job_func
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.run_id = run_id

    async def run(self) -> Dict[str, Any]:
        """
        Execute the job with retry logic.

        Returns:
            Dict containing job execution results and metadata
        """
        run_id = self.run_id or str(uuid4())
        start_time = datetime.utcnow()

        result = {
            "run_id": run_id,
            "job_name": self.job_name,
            "started_at": start_time.isoformat(),
            "attempts": [],
            "final_status": None,
            "error": None,
        }

        for attempt in range(1, self.max_retries + 1):
            attempt_start = datetime.utcnow()
            attempt_result = await self._execute_attempt(run_id, attempt)
            result["attempts"].append(attempt_result)

            if attempt_result["success"]:
                result["final_status"] = "success"
                result["data"] = attempt_result.get("data")
                logger.info(
                    f"Job {self.job_name} completed successfully on attempt {attempt}"
                )
                return result

            # Job failed - check if we should retry
            if attempt < self.max_retries:
                delay = self._calculate_delay(attempt)
                next_retry = datetime.utcnow() + timedelta(seconds=delay)

                logger.warning(
                    f"Job {self.job_name} failed on attempt {attempt}. "
                    f"Retrying in {delay} seconds at {next_retry.isoformat()}"
                )

                # Update job run with next retry time
                await self._update_job_run_retry(run_id, attempt, next_retry, attempt_result["error"])

                # Wait before retry
                await asyncio.sleep(delay)
            else:
                # All retries exhausted
                result["final_status"] = "failed"
                result["error"] = attempt_result["error"]

                logger.error(
                    f"Job {self.job_name} failed after {self.max_retries} attempts. "
                    f"Error: {attempt_result['error']}"
                )

                # Send alert notification
                await self._send_failure_alert(run_id, result)

        return result

    async def _execute_attempt(self, run_id: str, attempt: int) -> Dict[str, Any]:
        """Execute a single job attempt."""
        attempt_start = datetime.utcnow()

        async with async_session_factory() as db:
            # Create or update job run record
            job_run = await self._get_or_create_job_run(db, run_id, attempt)

            try:
                # Execute the actual job
                data = await self.job_func()

                # Update success status
                job_run.status = JobStatus.SUCCESS.value
                job_run.ended_at = datetime.utcnow()
                job_run.duration_seconds = int(
                    (job_run.ended_at - attempt_start).total_seconds()
                )

                # Extract record counts from job data if available
                if isinstance(data, dict):
                    job_run.records_processed = data.get("cards_processed", 0) or data.get("decklists_scraped", 0)
                    job_run.records_inserted = data.get("cards_updated", 0) or data.get("events_scraped", 0)

                await db.commit()

                return {
                    "attempt": attempt,
                    "success": True,
                    "started_at": attempt_start.isoformat(),
                    "ended_at": job_run.ended_at.isoformat(),
                    "data": data,
                }

            except Exception as e:
                error_message = str(e)
                error_stack = traceback.format_exc()

                # Update failure status
                job_run.status = JobStatus.FAILED.value
                job_run.ended_at = datetime.utcnow()
                job_run.duration_seconds = int(
                    (job_run.ended_at - attempt_start).total_seconds()
                )
                job_run.error_message = error_message
                job_run.error_stack = error_stack

                await db.commit()

                return {
                    "attempt": attempt,
                    "success": False,
                    "started_at": attempt_start.isoformat(),
                    "ended_at": job_run.ended_at.isoformat(),
                    "error": error_message,
                    "error_stack": error_stack,
                }

    async def _get_or_create_job_run(
        self, db: AsyncSession, run_id: str, attempt: int
    ) -> JobRun:
        """Get existing job run or create new one."""
        from sqlalchemy import select

        result = await db.execute(
            select(JobRun).where(JobRun.run_id == run_id)
        )
        job_run = result.scalar_one_or_none()

        if job_run:
            # Update for retry
            job_run.status = JobStatus.RUNNING.value
            job_run.attempt_number = attempt
            job_run.started_at = datetime.utcnow()
            job_run.ended_at = None
            job_run.error_message = None
        else:
            # Create new
            job_run = JobRun(
                job_name=self.job_name,
                run_id=run_id,
                status=JobStatus.RUNNING.value,
                started_at=datetime.utcnow(),
                attempt_number=attempt,
            )
            db.add(job_run)

        await db.flush()
        return job_run

    async def _update_job_run_retry(
        self,
        run_id: str,
        attempt: int,
        next_retry: datetime,
        error: str,
    ):
        """Update job run with next retry information."""
        async with async_session_factory() as db:
            from sqlalchemy import select

            result = await db.execute(
                select(JobRun).where(JobRun.run_id == run_id)
            )
            job_run = result.scalar_one_or_none()

            if job_run:
                job_run.next_retry_at = next_retry
                job_run.warnings = job_run.warnings or []
                if isinstance(job_run.warnings, list):
                    job_run.warnings.append({
                        "attempt": attempt,
                        "error": error,
                        "retry_at": next_retry.isoformat(),
                    })
                await db.commit()

    def _calculate_delay(self, attempt: int) -> int:
        """
        Calculate retry delay using exponential backoff.
        Attempt 1: 60 seconds (1 minute)
        Attempt 2: 120 seconds (2 minutes)
        Attempt 3: 240 seconds (4 minutes)
        """
        return self.base_delay * (2 ** (attempt - 1))

    async def _send_failure_alert(self, run_id: str, result: Dict[str, Any]):
        """Send alert notification after all retries exhausted."""
        subject = f"Job Failed: {self.job_name}"

        # Build alert body
        body_lines = [
            f"Job: {self.job_name}",
            f"Run ID: {run_id}",
            f"Started: {result['started_at']}",
            f"Final Status: {result['final_status']}",
            f"Total Attempts: {len(result['attempts'])}",
            "",
            "Error:",
            result.get("error", "Unknown error"),
            "",
            "Attempt History:",
        ]

        for attempt in result["attempts"]:
            status = "Success" if attempt["success"] else "Failed"
            body_lines.append(
                f"  Attempt {attempt['attempt']}: {status} at {attempt['started_at']}"
            )
            if not attempt["success"]:
                body_lines.append(f"    Error: {attempt.get('error', 'Unknown')}")

        body_lines.extend([
            "",
            f"Dashboard: {settings.APP_URL}/admin",
            f"Log URL: {settings.APP_URL}/admin?job_run={run_id}",
        ])

        body = "\n".join(body_lines)

        try:
            await send_alert_email(subject, body)
            logger.info(f"Sent failure alert for job {self.job_name}")
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")


# Factory functions for creating wrapped job runners
def create_scryfall_runner(run_id: Optional[str] = None) -> JobRunner:
    """Create a job runner for Scryfall sync."""
    from app.jobs.scryfall_sync import sync_scryfall_cards

    return JobRunner(
        job_name="scryfall_sync",
        job_func=sync_scryfall_cards,
        run_id=run_id,
    )


def create_mtgtop8_runner(run_id: Optional[str] = None) -> JobRunner:
    """Create a job runner for mtgtop8 scrape."""
    from app.jobs.mtgtop8_scrape import scrape_mtgtop8

    return JobRunner(
        job_name="mtgtop8_scrape",
        job_func=scrape_mtgtop8,
        run_id=run_id,
    )


async def run_job_with_retry(job_name: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a job by name with full retry logic.

    Args:
        job_name: Name of the job ("scryfall_sync" or "mtgtop8_scrape")
        run_id: Optional run ID to use (for tracking jobs triggered via API)

    Returns:
        Job execution results
    """
    if job_name == "scryfall_sync":
        runner = create_scryfall_runner(run_id=run_id)
    elif job_name == "mtgtop8_scrape":
        runner = create_mtgtop8_runner(run_id=run_id)
    else:
        raise ValueError(f"Unknown job: {job_name}")

    return await runner.run()
