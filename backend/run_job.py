#!/usr/bin/env python3
"""
CLI script to manually run scraper jobs.

Usage:
    python run_job.py scryfall_sync        # Run Scryfall card sync
    python run_job.py mtgtop8_scrape       # Run mtgtop8 meta scrape
    python run_job.py card_meta_stats      # Compute per-card meta representation
    python run_job.py --list               # List available jobs
"""

import asyncio
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_scryfall():
    """Run Scryfall sync directly."""
    from app.jobs.scryfall_sync import sync_scryfall_cards

    logger.info("Starting Scryfall sync...")
    result = await sync_scryfall_cards()
    logger.info(f"Scryfall sync complete: {result}")
    return result


async def run_mtgtop8():
    """Run mtgtop8 scrape directly."""
    from app.jobs.mtgtop8_scrape import scrape_mtgtop8

    logger.info("Starting mtgtop8 scrape...")
    result = await scrape_mtgtop8()
    logger.info(f"mtgtop8 scrape complete: {result}")
    return result


async def run_card_meta_stats():
    """Compute per-card meta representation from tournament decklists."""
    from app.jobs.compute_card_meta_stats import compute_card_meta_stats

    logger.info("Starting card meta stats computation...")
    result = await compute_card_meta_stats()
    logger.info(f"Card meta stats complete: {result}")
    return result


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--list":
        print("Available jobs:")
        print("  scryfall_sync    - Sync cards from Scryfall API")
        print("  mtgtop8_scrape   - Scrape tournament data from mtgtop8.com")
        print("  card_meta_stats  - Compute per-card meta representation")
        sys.exit(0)

    job_name = sys.argv[1]

    valid_jobs = ["scryfall_sync", "mtgtop8_scrape", "card_meta_stats"]
    if job_name not in valid_jobs:
        print(f"Error: Unknown job '{job_name}'")
        print(f"Valid jobs: {', '.join(valid_jobs)}")
        sys.exit(1)

    try:
        if job_name == "scryfall_sync":
            result = asyncio.run(run_scryfall())
        elif job_name == "mtgtop8_scrape":
            result = asyncio.run(run_mtgtop8())
        elif job_name == "card_meta_stats":
            result = asyncio.run(run_card_meta_stats())

        print("\n" + "="*50)
        print("Job completed successfully!")
        if result:
            for key, value in result.items():
                print(f"  {key}: {value}")
        print("="*50)

    except KeyboardInterrupt:
        print("\nJob cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Job failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
