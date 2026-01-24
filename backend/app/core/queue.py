"""
Redis Queue (RQ) Configuration

Provides RQ queue setup for background job execution that persists
beyond HTTP request completion.
"""

import logging
from redis import Redis
from rq import Queue

from app.core.config import settings

logger = logging.getLogger(__name__)

# Redis connection
redis_conn = Redis.from_url(settings.REDIS_URL)

# Default queue for background jobs
job_queue = Queue("spellbook_jobs", connection=redis_conn)

# High priority queue for urgent tasks
high_priority_queue = Queue("spellbook_high", connection=redis_conn)


def get_queue(priority: str = "default") -> Queue:
    """Get the appropriate queue based on priority."""
    if priority == "high":
        return high_priority_queue
    return job_queue


def enqueue_job(func, *args, job_timeout: int = 3600, **kwargs):
    """
    Enqueue a job to be executed by an RQ worker.

    Args:
        func: The function to execute
        *args: Positional arguments for the function
        job_timeout: Maximum job execution time in seconds (default: 1 hour)
        **kwargs: Keyword arguments for the function

    Returns:
        The RQ Job object
    """
    return job_queue.enqueue(
        func,
        *args,
        job_timeout=job_timeout,
        **kwargs
    )


def get_queue_stats():
    """Get statistics about the job queues."""
    return {
        "default_queue": {
            "name": job_queue.name,
            "pending": len(job_queue),
            "failed": len(job_queue.failed_job_registry),
            "finished": len(job_queue.finished_job_registry),
        },
        "high_priority_queue": {
            "name": high_priority_queue.name,
            "pending": len(high_priority_queue),
            "failed": len(high_priority_queue.failed_job_registry),
            "finished": len(high_priority_queue.finished_job_registry),
        },
    }
