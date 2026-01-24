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

    # Connect to Redis
    redis_url = settings.REDIS_URL
    logger.info(f"Connecting to Redis at {redis_url}")
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

    # Start worker
    with Connection(redis_conn):
        worker = Worker(
            queues,
            connection=redis_conn,
            name=f"spellbook-worker-{os.getpid()}",
        )

        logger.info(f"Starting worker (burst={args.burst})")

        worker.work(
            burst=args.burst,
            logging_level='DEBUG' if args.verbose else 'INFO',
        )


if __name__ == '__main__':
    main()
