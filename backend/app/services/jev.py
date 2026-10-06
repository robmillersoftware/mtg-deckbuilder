"""
Shared Jev (TypeSafe System One) plumbing: the client factory, one concurrency
cap shared by every caller, and helpers that send requests through it.

Deck fit, role tagging, search re-ranking, chat routing and deck-request
parsing all call `ask`/`ask_many`, so concurrent users stay under Jev's
rate limit (80 req/s).
Design: docs/superpowers/specs/2026-10-05-jev-expansion-design.md
"""

import asyncio
import logging
import weakref
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from app.core.config import settings

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 2.0
RETRY_BUDGET = 5.0
FIT_DEADLINE = 6.0  # overall cap per Jev operation, across all waves and retries
# ponytail: fixed cap under Jev's 80 req/s limit; make it adaptive if 429s show up
MAX_CONCURRENT = 32

# One semaphore per event loop: asyncio primitives can't cross loops (tests and
# RQ jobs run their own), and within the app's loop every caller shares it.
_limiters: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _client() -> Optional[AsyncTypeSafeClient]:
    if not settings.TYPESAFE_API_KEY:
        return None
    return AsyncTypeSafeClient(
        api_key=settings.TYPESAFE_API_KEY,
        timeout=REQUEST_TIMEOUT,
        retry=RetryPolicy(api_timeout_error=False, timeout=RETRY_BUDGET),
    )


async def _close(client: Optional[AsyncTypeSafeClient]) -> None:
    """Safely close a client, logging and swallowing any errors."""
    if client is None:
        return
    try:
        await client.aclose()
    except Exception as e:
        logger.warning(f"Failed to close Jev client: {e}")


def limiter() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    gate = _limiters.get(loop)
    if gate is None:
        gate = _limiters[loop] = asyncio.Semaphore(MAX_CONCURRENT)
    return gate


@asynccontextmanager
async def session(client=None) -> AsyncIterator[Optional[Any]]:
    """Yield `client`, or a new client from settings (None when no key is
    configured). Closes only a client it created."""
    owned = client is None
    if owned:
        client = _client()
    try:
        yield client
    finally:
        if owned:
            await _close(client)


async def ask(client, state: Any, questions: Dict[str, Any], **kwargs) -> Any:
    """One System One request, counted against the shared cap."""
    async with limiter():
        return await client.system_one(state, questions, **kwargs)


async def ask_many(
    client,
    requests: Sequence[Tuple[Any, Dict[str, Any]]],
    deadline: Optional[float] = None,
    **kwargs,
) -> List[Any]:
    """Concurrent requests through the shared cap; results in input order.

    Raises on the first failure (the rest are cancelled) or when `deadline`
    seconds pass. Callers treat any exception as "Jev unavailable".
    """
    results: List[Any] = [None] * len(requests)

    async def one(i: int, state: Any, questions: Dict[str, Any]) -> None:
        results[i] = await ask(client, state, questions, **kwargs)

    async with asyncio.timeout(deadline):
        async with asyncio.TaskGroup() as tg:
            for i, (state, questions) in enumerate(requests):
                tg.create_task(one(i, state, questions))
    return results
