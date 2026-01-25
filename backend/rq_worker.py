#!/usr/bin/env python
"""
RQ Worker for Spellbook Background Jobs

This worker runs as a separate process, independent of the web server.
It processes jobs from the Redis queue, ensuring they persist beyond
HTTP request completion.

Usage:
    python rq_worker.py              # Run worker for default queue
    python rq_worker.py --burst      # Process jobs and exit (good for testing)
    python rq_worker.py --verbose    # Verbose logging

In production, run this as a systemd service or in a separate container.
"""

import argparse
import logging
import sys
import os
import uuid
import socket

# Ensure the app module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from redis import Redis
from rq import Worker, Queue, Connection

from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Spellbook RQ Worker')
    parser.add_argument(
        '--burst',
        action='store_true',
        help='Run in burst mode (process jobs and exit)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--queues',
        nargs='+',
        default=['spellbook_high', 'spellbook_jobs'],
        help='Queues to listen on (default: spellbook_high, spellbook_jobs)'
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Connect to Redis (mask credentials in log)
    redis_url = settings.REDIS_URL
    # Parse URL to log host without credentials
    from urllib.parse import urlparse
    parsed = urlparse(redis_url)
    safe_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 6379}/{parsed.path.lstrip('/')}"
    logger.info(f"Connecting to Redis at {safe_url}")
    redis_conn = Redis.from_url(redis_url)

    # Test connection
    try:
        redis_conn.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)

    # Create queues
    queues = [Queue(name, connection=redis_conn) for name in args.queues]
    logger.info(f"Listening on queues: {[q.name for q in queues]}")

    # Clean up any stale workers before starting
    # This handles the case where a previous worker crashed without deregistering
    try:
        workers = Worker.all(connection=redis_conn)
        for w in workers:
            if w.name.startswith('spellbook'):
                # Check if the worker is actually dead
                try:
                    # Workers in 'busy' state with no heartbeat are likely dead
                    if w.state == 'idle' or not w.pid:
                        logger.info(f"Cleaning up stale worker: {w.name}")
                        w.register_death()
                except Exception as e:
                    logger.warning(f"Could not check worker {w.name}: {e}")
                    # Force cleanup of worker that we can't even check
                    try:
                        w.register_death()
                    except:
                        pass
    except Exception as e:
        logger.warning(f"Could not clean up stale workers: {e}")

    # Also directly clean up the known stale worker key if it exists
    try:
        stale_key = "rq:worker:spellbook-worker-1"
        if redis_conn.exists(stale_key):
            redis_conn.delete(stale_key)
            logger.info(f"Removed stale worker key: {stale_key}")
    except Exception as e:
        logger.warning(f"Could not remove stale worker key: {e}")

    # Generate a unique worker name using hostname + short UUID
    # This ensures uniqueness even in container environments where PID is always 1
    hostname = socket.gethostname()[:8]
    unique_id = uuid.uuid4().hex[:8]
    worker_name = f"spellbook-{hostname}-{unique_id}"

    # Start worker
    with Connection(redis_conn):
        worker = Worker(
            queues,
            connection=redis_conn,
            name=worker_name,
        )

        logger.info(f"Starting worker '{worker_name}' (burst={args.burst})")

        worker.work(
            burst=args.burst,
            logging_level='DEBUG' if args.verbose else 'INFO',
        )


if __name__ == '__main__':
    main()
