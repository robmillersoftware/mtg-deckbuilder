"""jev: shared client session, process-wide concurrency cap, concurrent requests."""

import asyncio

import pytest

from app.services import jev


class Client:
    """Echoes `state`; tracks in-flight requests, request kwargs, and close()."""

    def __init__(self, delay=0.01):
        self.delay, self.live, self.peak, self.closed, self.kwargs = delay, 0, 0, False, []

    async def system_one(self, state, questions, **kwargs):
        self.kwargs.append(kwargs)
        self.live += 1
        self.peak = max(self.peak, self.live)
        try:
            await asyncio.sleep(self.delay)
            if state == "boom":
                raise RuntimeError("boom")
            return state
        finally:
            self.live -= 1

    async def aclose(self):
        self.closed = True


async def test_cap_is_shared_across_concurrent_callers(monkeypatch):
    monkeypatch.setattr(jev, "MAX_CONCURRENT", 3)
    client = Client()
    a, b = await asyncio.gather(
        jev.ask_many(client, [(i, {}) for i in range(10)]),
        jev.ask_many(client, [(i, {}) for i in range(10, 20)]),
    )
    assert client.peak == 3
    assert a == list(range(10)) and b == list(range(10, 20))


async def test_ask_many_raises_on_first_failure():
    with pytest.raises(Exception):
        await jev.ask_many(Client(), [(1, {}), ("boom", {})])


async def test_ask_many_deadline():
    with pytest.raises(TimeoutError):
        await jev.ask_many(Client(delay=5), [(1, {})], deadline=0.01)


async def test_ask_passes_request_options():
    client = Client(delay=0)
    await jev.ask(client, 1, {}, timeout=15.0)
    assert client.kwargs == [{"timeout": 15.0}]


async def test_session_closes_only_a_client_it_created(monkeypatch):
    mine = Client()
    async with jev.session(mine) as c:
        assert c is mine
    assert not mine.closed

    made = Client()
    monkeypatch.setattr(jev, "_client", lambda: made)
    async with jev.session() as c:
        assert c is made
    assert made.closed


async def test_session_without_key_yields_none():
    async with jev.session() as c:  # conftest blanks TYPESAFE_API_KEY
        assert c is None
