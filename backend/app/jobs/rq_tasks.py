"""
RQ Task Wrappers

Provides synchronous wrappers around async jobs for RQ worker execution.
RQ workers run in separate processes, independent of the web server,
ensuring jobs persist beyond HTTP request completion.
"""

import asyncio
import logging
from datetime import datetime

from rq import get_current_job

logger = logging.getLogger(__name__)


def _run_async(coro):
    """
    Run an async coroutine synchronously for RQ.
    Creates a new event loop for each job execution.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _get_run_id_from_job():
    """Get the run_id from the current RQ job's metadata or job ID."""
    job = get_current_job()
    if job:
        # First try to get from meta, then fall back to job.id
        run_id = job.meta.get("run_id") if job.meta else None
        if not run_id:
            run_id = job.id
        logger.info(f"[RQ] Using run_id: {run_id}")
        return run_id
    return None


def run_scryfall_sync_task():
    """
    RQ task wrapper for Scryfall sync job.
    This runs in a separate worker process.
    """
    run_id = _get_run_id_from_job()
    logger.info(f"[RQ] Starting Scryfall sync task at {datetime.utcnow().isoformat()} (run_id={run_id})")

    from app.jobs.job_runner import run_job_with_retry

    result = _run_async(run_job_with_retry("scryfall_sync", run_id=run_id))

    logger.info(f"[RQ] Scryfall sync completed: {result.get('final_status', 'unknown')}")
    return result


def run_mtgtop8_scrape_task():
    """
    RQ task wrapper for mtgtop8 scrape job.
    This runs in a separate worker process.
    """
    run_id = _get_run_id_from_job()
    logger.info(f"[RQ] Starting mtgtop8 scrape task at {datetime.utcnow().isoformat()} (run_id={run_id})")

    from app.jobs.job_runner import run_job_with_retry

    result = _run_async(run_job_with_retry("mtgtop8_scrape", run_id=run_id))

    logger.info(f"[RQ] mtgtop8 scrape completed: {result.get('final_status', 'unknown')}")
    return result


def run_embeddings_backfill_task(batch_size: int = 50, max_batches: int = 10):
    """
    RQ task wrapper for embeddings backfill.
    This runs in a separate worker process.
    """
    logger.info(f"[RQ] Starting embeddings backfill (batch_size={batch_size}, max_batches={max_batches})")

    async def backfill_embeddings():
        from app.db.session import async_session_factory
        from app.models.card import Card
        from app.services.embedding_service import get_embedding_service
        from sqlalchemy import select

        embedding_service = get_embedding_service()
        if not embedding_service.client:
            raise RuntimeError("OpenAI API key not configured")

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
                    logger.info(f"[RQ] Backfill complete. Total cards updated: {total_updated}")
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
                logger.info(f"[RQ] Backfill batch {batch_num + 1}: updated {batch_updated} cards (total: {total_updated})")

        return {"total_updated": total_updated}

    result = _run_async(backfill_embeddings())
    logger.info(f"[RQ] Embeddings backfill completed: {result}")
    return result


# Job name to task function mapping
TASK_REGISTRY = {
    "scryfall_sync": run_scryfall_sync_task,
    "mtgtop8_scrape": run_mtgtop8_scrape_task,
    "embeddings_backfill": run_embeddings_backfill_task,
}


def get_task_function(job_name: str):
    """Get the RQ task function for a given job name."""
    if job_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown job: {job_name}. Valid jobs: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[job_name]
