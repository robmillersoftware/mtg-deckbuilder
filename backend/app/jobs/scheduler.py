"""
Job Scheduler Configuration

Uses APScheduler for running background jobs:
- Scryfall sync: Daily at 2:00 AM UTC
- mtgtop8 scrape: Sunday at 6:00 AM UTC

All jobs are wrapped with exponential backoff retry logic (3 attempts)
and send alert notifications on final failure.
"""

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.jobs.job_runner import run_job_with_retry

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def run_scryfall_with_retry():
    """Wrapper to run Scryfall sync with retry logic."""
    return await run_job_with_retry("scryfall_sync")


async def run_mtgtop8_with_retry():
    """Wrapper to run mtgtop8 scrape with retry logic."""
    return await run_job_with_retry("mtgtop8_scrape")


async def run_card_meta_stats_with_retry():
    """Wrapper to run card meta stats computation with retry logic."""
    return await run_job_with_retry("card_meta_stats")


def configure_scheduler() -> AsyncIOScheduler:
    """Configure and return the scheduler with all jobs."""

    # Scryfall daily sync - 2:00 AM UTC (with retry wrapper)
    scheduler.add_job(
        run_scryfall_with_retry,
        trigger=CronTrigger(hour=2, minute=0, timezone="UTC"),
        id="scryfall_sync",
        name="Scryfall Card Database Sync",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("Scheduled Scryfall sync job: daily at 02:00 UTC (with retry)")

    # mtgtop8 weekly scrape - Sunday 6:00 AM UTC (with retry wrapper)
    scheduler.add_job(
        run_mtgtop8_with_retry,
        trigger=CronTrigger(day_of_week="sun", hour=6, minute=0, timezone="UTC"),
        id="mtgtop8_scrape",
        name="mtgtop8 Tournament Data Scrape",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("Scheduled mtgtop8 scrape job: Sunday at 06:00 UTC (with retry)")

    # Card meta stats - Sunday 8:00 AM UTC (after mtgtop8 scrape completes)
    scheduler.add_job(
        run_card_meta_stats_with_retry,
        trigger=CronTrigger(day_of_week="sun", hour=8, minute=0, timezone="UTC"),
        id="card_meta_stats",
        name="Card Meta Stats Computation",
        replace_existing=True,
        max_instances=1,
    )
    logger.info("Scheduled card meta stats job: Sunday at 08:00 UTC (with retry)")

    return scheduler


def start_scheduler():
    """Start the scheduler."""
    if not scheduler.running:
        scheduler.start()
        logger.info("Scheduler started")


def shutdown_scheduler():
    """Shutdown the scheduler gracefully."""
    if scheduler.running:
        scheduler.shutdown(wait=True)
        logger.info("Scheduler shutdown complete")


def get_scheduled_jobs():
    """Get list of all scheduled jobs with their next run times."""
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger),
        })
    return jobs


async def run_job_manually(job_id: str) -> dict:
    """
    Manually trigger a job to run immediately with retry logic.

    Args:
        job_id: The job ID to run ("scryfall_sync", "mtgtop8_scrape", or "card_meta_stats")

    Returns:
        Job execution result with retry metadata
    """
    return await run_job_with_retry(job_id)
