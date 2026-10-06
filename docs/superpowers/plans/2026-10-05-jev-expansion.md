# Jev Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use Jev for card-role tagging, suggestion-time role checks, search re-ranking, chat routing and deck-request parsing, and stop low-fit cards from filling suggestion slots.

**Architecture:** A new `app/services/jev.py` owns the TypeSafe client factory and one process-wide concurrency cap. Every Jev caller sends its requests through it. Each feature asks Jev narrow questions in the module that already owns that feature (`deck_fit`, `classify_card_roles`, `card_service`, a new `chat_routing`, `deck_parsing`). Code turns the answers into decisions and falls back to today's behavior whenever Jev is missing, slow, or returns an incomplete answer.

**Tech Stack:** FastAPI, SQLAlchemy async, Alembic, Postgres full-text search, `typesafe-sdk==0.7.2` (`AsyncTypeSafeClient`, `Noul`, `Score`, `Choice`, `RetryPolicy`), pytest + pytest-asyncio (`asyncio_mode=auto`).

**Spec:** `docs/superpowers/specs/2026-10-05-jev-expansion-design.md`

## Global Constraints

- "Process-wide Jev concurrency cap ... shared by deck fit, role tagging, search re-ranking and routing, so concurrent users stay under Jev's rate limit (80 req/s; cap stays 32 in-flight)." `MAX_CONCURRENT = 32`.
- "`jev.py` also owns the client factory (`_client`, guarded `_close`, `REQUEST_TIMEOUT`, `RETRY_BUDGET`, `FIT_DEADLINE`), which move there from `deck_fit.py`. `deck_fit` imports them. Behavior is unchanged." Values: `REQUEST_TIMEOUT = 2.0`, `RETRY_BUDGET = 5.0`, `FIT_DEADLINE = 6.0`.
- Role tagging: "A role is assigned when its Noul is >= `ROLE_THRESHOLD` (0.5). Efficiency is `round(score) + 1` (1-5). `CardRole.confidence` stores the Noul, not the current hard-coded 0.9. `CardRole.reasoning` is left null." Cards are "saved in batches of 50". `get_unclassified_cards`, `save_card_roles`, `classify_all_cards` and `get_classification_stats` keep their contracts.
- Role check: "A candidate below `ROLE_FIT_CUTOFF` (0.5) for a role is dropped from that role's results only."
- Minimum fit: "`deck_fit.rank` also drops candidates whose plan fit is below `LOW_FIT_CUTOFF` (0.34, the same threshold `flag_low_fit` uses)." The fallback ordering is unchanged.
- Search: "`CardService.semantic_search(query, limit, standard_only, format, colors)` keeps its signature. All five callers are unchanged." Shortlist "up to `SHORTLIST_SIZE` (60) unique names"; re-rank "concurrent through the shared cap under `FIT_DEADLINE`"; "drop below `SEARCH_CUTOFF` (0.5)".
- Routing: "`ROUTE_CONFIDENCE` starts at 0.6 and is tuned on the eval." Colors and roles on noul >= 0.5; "count: the first integer in the message, default 6"; default roles are "the existing default three" (`threats`, `removal`, `card advantage`); state carries "the last 6 turns".
- Parsing: "`parse_deck_request(prompt, db)` keeps its return shape `{archetype, colors, colors_specified, strategy, specific_cards}`."
- Migration `016` (down_revision `015`): "a GIN expression index on that tsvector for `cards`".
- "OpenAI becomes optional at runtime."
- "Unchanged rule: a Jev failure never blocks suggestions, search, chat, generation or classification." Fallbacks: role tagging, batch skipped and logged; role check, whole fit round falls back to retrieval order; search re-rank, shortlist order; routing, LLM tool call; parsing, `fallback_parse`.
- Out of scope: replacing LLM generation, caching Jev answers, flavor-text themes, automatic re-tagging after Scryfall sync.
- Unit tests never touch the network. `backend/tests/conftest.py` (Task 1) blanks every API key, because the repo-root `.env` holds a real `TYPESAFE_API_KEY`. Eval scripts call real Jev and run only in their own steps, never in CI.
- Backend tests: `cd backend && python3 -m pytest` (baseline before Task 1: 123 passed).
- Every commit message ends with exactly these two lines:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN
  ```

### Resolved spec ambiguities

- **Per-loop limiter.** The spec describes "one module-level limiter". An `asyncio.Semaphore` cannot be shared across event loops, and tests and RQ jobs each run their own loop. So `jev.limiter()` keeps one semaphore per running loop. Inside the app's single loop, every caller shares one cap of 32.
- **`ROLE_MAP` moves to `app/models/card.py`**, next to `CARD_ROLES` and the new `ROLE_DEFINITIONS`. `deck_fit.role_description` needs it, and `guided_builder` already imports `deck_fit`, so leaving it in `guided_builder` would create an import cycle. `guided_builder` re-imports it, so `guided_builder.ROLE_MAP` still resolves.
- **Role-tagging timeout.** One card carries 44 questions, which can take longer than the 2 s chat timeout. Role tagging passes `timeout=TAG_REQUEST_TIMEOUT` (15 s) per request. Chat paths keep 2 s.
- **Full-text search runs on `cards`, not the format views.** The spec puts the GIN index on `cards`, and an index on `cards` does not cover the materialized views. So the full-text query hits `cards` with the same legality filter the vector path uses on `cards`.
- **The tsvector expression uses `coalesce`:** `to_tsvector('english', name || ' ' || coalesce(type_line, '') || ' ' || coalesce(oracle_text, ''))`. The spec's bare concatenation is NULL for cards with no oracle text, and those cards would never match. Stopwords are dropped by Postgres's `english` configuration, not by a Python list.
- **Popular cards** exclude basic lands (they top every frequency list) and match decklist names to `cards.name` exactly. DFC entries listed by front face alone are not matched. That is a known ceiling and is marked `ponytail:` in code.
- **Routing strategy text.** "The message plus the persisted strategy" is `"<persisted>; <message>"`, capped at 300 characters. The handlers persist it back to `context.strategy`, so without a cap it would grow every turn. Resolved card names that Jev says the user does *not* want are removed from that text. Otherwise `suggest_core` re-resolves them from the strategy into a "Build Around" group (for example, an opponent's card).
- **Routing colors fallback:** colors with noul >= 0.5, else the colors of the deck's mana costs, else the persisted `context.colors` (a no-deck conversation has no deck colors).
- **Count** is the first integer in the message, clamped to 1-20, default 6.
- **Confidence gates every Jev decision, `reply` included.** When `action` is `reply` with confidence >= `ROUTE_CONFIDENCE` and the forced-tool rule applies, the top non-reply tool is dispatched. Below the threshold, the existing `llm.chat_with_tools` path runs unchanged, including its own `require_tool`.
- **Text-only reply.** `llm.complete` takes one user string, so the conversation is sent as a `role: content` transcript, and the system prompt gets one extra line forbidding tools.
- **The `llm.is_configured()` gate stays first in `process_message`.** Without OpenRouter, chat still uses `_fallback_response`, as it does today.
- **Parsing colors.** `colors` holds the colors with noul >= 0.5, and only when the `colors_specified` noul is also >= 0.5; otherwise it is empty. (A pre-run showed Jev picking W/U/B/G for "a deck that beats mono-red".) Unlike `fallback_parse`, there is no `["R"]` default. `colors_specified` is true exactly when `colors` is non-empty.
- **Question ids.** Role Nouls in routing use `role:<role name>` (names may contain spaces; the model never sees ids). The role check in `score_fit` uses `role:<i>`, because guided role names are free text.

## Review Focus

1. **Jev answers only part of a request (a missing Choice or Noul) or times out mid-routing.** Expected: routing returns `None` and the turn takes the LLM tool-call path. It must not crash or dispatch with half-built arguments. Tests in Task 8 (`test_partial_answer_falls_back`, `test_route_none_uses_llm_tool_call`) and Task 7 (`test_partial_answer_falls_back`).
2. **A chat message names an opponent's card ("How do I beat Sheoldred?").** Expected: the card is not in `specific_cards`, and its name is stripped from the strategy text, so `suggest_core` does not add it as a "Build Around" group. Test in Task 8 (`test_unwanted_card_is_not_built_around`).
3. **Empty or oversized shortlists:** a query of only punctuation or stopwords, no hits at all, or more than 60 candidates. Expected: an empty query word list skips SQL, an empty shortlist returns `[]` without a Jev call, and the shortlist stops at 60 without querying lower-priority sources. Tests in Task 5 (`test_no_words_skips_the_query`) and Task 6 (`test_empty_shortlist_makes_no_jev_call`, `test_shortlist_capped_skips_lower_sources`).
4. **A guided role name that isn't in `ROLE_MAP`**, such as "cards with surveil", "Counterspells " with odd case or whitespace, or "payoffs". Expected: mapped names expand to the system role definitions after normalizing case and whitespace. Anything else is described by its own name. Tests in Task 3 (`test_role_description_expands_role_map_and_falls_back_to_name`, `test_role_nouls_per_candidate_roles`).
5. **Choice options with duplicate names:** a meta archetype repeated across snapshot dates or differing only in case, or a mechanic keyword equal to a role ("protection"). Expected: options are de-duplicated case-insensitively, the first description is kept, and every Choice still has at least one option ("none" for the opponent). Test in Task 8 (`test_choice_options_dedupe`).

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `backend/app/services/jev.py` (new) | Client factory, per-loop concurrency cap, `session`, `ask`, `ask_many` | 1 |
| `backend/tests/conftest.py` (new) | Blank API keys for every test | 1 |
| `backend/app/services/deck_fit.py` | Uses `jev`; `rank` minimum fit; role check in `score_fit`; `role_description` | 1, 2, 3 |
| `backend/app/models/card.py` | `ROLE_DEFINITIONS`, `ROLE_MAP` (moved) | 3 |
| `backend/app/services/guided_builder.py` | Role-check filter in the fit path; frequency lookup delegates to `CardService` | 3, 5 |
| `backend/tests/jev_fake.py` (new) | `FakeJev` stub with Noul, Score and Choice answers | 4 |
| `backend/app/jobs/classify_card_roles.py` | Jev role tagging | 4 |
| `backend/scripts/eval_roles.py` (new) | Role-tagging eval against real Jev | 4 |
| `backend/alembic/versions/016_add_cards_fts_index.py` (new) | GIN full-text index | 5 |
| `backend/app/services/card_service.py` | SQL filters, full-text names, popular names, frequency; shortlist + Jev re-rank | 5, 6 |
| `backend/app/services/ai/deck_parsing.py` | Jev deck-request parsing | 7 |
| `backend/app/services/chat_routing.py` (new) | Routing questions, argument assembly, `route()` | 8 |
| `backend/app/services/chat_service.py` | Calls routing before any LLM call | 8 |
| `backend/scripts/eval_routing.py` (new), `backend/scripts/eval_fit.py` | Routing/parsing eval; Lazotep Reaver relabel | 9 |

---

### Task 1: Shared Jev infrastructure, with `deck_fit` moved onto it

**Files:**
- Create: `backend/app/services/jev.py`
- Create: `backend/tests/conftest.py`
- Modify: `backend/app/services/deck_fit.py` (imports at lines 9-15; constants at lines 43-47; `_client`/`_close` at lines 154-171; the bodies of `infer_identity` and `score_fit`)
- Modify: `backend/tests/test_deck_fit.py:202-204` (`test_no_api_key_returns_empty`)
- Test: `backend/tests/test_jev.py`

**Interfaces:**
- Consumes: `settings.TYPESAFE_API_KEY` (already in `app/core/config.py`).
- Produces (in `app.services.jev`):
  - constants `REQUEST_TIMEOUT = 2.0`, `RETRY_BUDGET = 5.0`, `FIT_DEADLINE = 6.0`, `MAX_CONCURRENT = 32`
  - `_client() -> Optional[AsyncTypeSafeClient]` (None without a key)
  - `async _close(client) -> None` (None-safe; logs and swallows errors)
  - `limiter() -> asyncio.Semaphore` (one per running loop)
  - `session(client=None)`: an async context manager that yields the given client, or a new one from settings (None without a key), and closes only a client it created
  - `async ask(client, state, questions, **kwargs)`: one `system_one` call inside the cap; `kwargs` pass through (for example `timeout=`)
  - `async ask_many(client, requests: Sequence[Tuple[state, questions]], deadline: Optional[float] = None, **kwargs) -> List[response]`: concurrent, results in input order; raises on the first failure (cancelling the rest) or when the deadline passes
- `deck_fit` imports `FIT_DEADLINE`, `ask`, `ask_many` and `session` from `jev`. `deck_fit.FIT_DEADLINE` stays a module global there, so the existing `TestDeadline` monkeypatches still work.

- [ ] **Step 1: Write the failing tests**

`backend/tests/conftest.py`:
```python
"""Keep unit tests off the network: the repo-root .env holds real keys."""

import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def _no_external_keys(monkeypatch):
    for key in ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setattr(settings, key, None)
```

`backend/tests/test_jev.py`:
```python
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
```

In `backend/tests/test_deck_fit.py`, change `test_no_api_key_returns_empty` (the `deck_fit` module no longer imports `settings`):
```python
    async def test_no_api_key_returns_empty(self):
        # conftest blanks TYPESAFE_API_KEY, so no client can be built
        assert await deck_fit.score_fit(DeckIdentity(), [], [card("A")]) == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_jev.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'app.services.jev'`

- [ ] **Step 3: Write `jev.py`**

`backend/app/services/jev.py`:
```python
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
```

- [ ] **Step 4: Move `deck_fit` onto `jev`**

In `backend/app/services/deck_fit.py`, replace the import block:
```python
import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field
from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy, Score

from app.core.config import settings
```
with:
```python
import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field
from typesafe_sdk import Noul, Score

from app.services.jev import FIT_DEADLINE, ask, ask_many, session
```

Delete these lines (after `RE_INFER_AT = (6, 15, 30)`):
```python
REQUEST_TIMEOUT = 2.0
RETRY_BUDGET = 5.0
FIT_DEADLINE = 6.0  # overall cap per Jev operation, across all waves and retries
# ponytail: fixed concurrency cap under Jev's 80 req/s limit; make it adaptive if 429s show up
MAX_CONCURRENT = 32
```

Delete the `_client()` and `_close()` functions (they now live in `jev.py`).

Replace the whole `infer_identity` function with:
```python
async def infer_identity(
    cards: List[Dict[str, str]],
    request_text: Optional[str],
    overrides: Optional[IdentityOverrides] = None,
    client=None,
) -> Optional[DeckIdentity]:
    """Theme tags (one Noul each) and key cards (one Score per nonland card), in one request."""
    nonland = [c for c in cards if not is_land(c)]
    if not nonland and not request_text:
        return None

    questions: Dict[str, Any] = {
        f"tag:{tag}": Noul(instructions={
            "question": "Does the deck described by `deck.request` and `deck.cards` build around this theme?",
            "theme": definition,
        })
        for tag, definition in THEME_TAGS.items()
    }
    with_keys = len(nonland) >= RE_INFER_AT[0]
    if with_keys:
        for i in range(len(nonland)):
            questions[f"key:{i}"] = Score(
                instructions=f"How central is `deck.cards[{i}]` to this deck's plan?",
                criteria=KEY_CARD_LEVELS,
            )

    async with session(client) as client:
        if client is None:
            return None
        try:
            async with asyncio.timeout(FIT_DEADLINE):
                resp = await ask(
                    client, {"deck": {"request": request_text or "", "cards": nonland}}, questions)
            tags = [t for t in THEME_TAGS if resp.nouls[f"tag:{t}"].noul >= TAG_THRESHOLD]
            key_cards: List[str] = []
            if with_keys:
                n = 5 if len(nonland) < 40 else 8
                order = sorted(range(len(nonland)),
                               key=lambda i: resp.scores[f"key:{i}"].score, reverse=True)
                key_cards = [nonland[i]["name"] for i in order[:n]]

            identity = DeckIdentity(tags=tags, key_cards=key_cards, request_text=request_text,
                                    overrides=overrides or IdentityOverrides())
            return apply_overrides(identity, [c["name"] for c in nonland])
        except Exception as e:  # Jev must never break deck building
            logger.warning(f"Deck identity inference failed: {e}")
            return None
```

Replace the whole `score_fit` function with:
```python
async def score_fit(
    identity: DeckIdentity,
    key_cards: List[Dict[str, str]],
    candidates: List[Dict[str, str]],
    client=None,
) -> Dict[str, FitScore]:
    """One concurrent Jev request per unique candidate. Empty dict if any request fails."""
    unique = {c["name"]: c for c in candidates}
    if not unique:
        return {}

    deck = {"themes": identity.tags, "request": identity.request_text or "", "key_cards": key_cards}
    questions: Dict[str, Any] = {
        "plan_fit": Score(
            instructions="How well does `candidate` advance the plan described by `deck.themes` and `deck.request`?",
            criteria=PLAN_FIT_LEVELS,
        ),
        "anti_synergy": Noul(instructions=(
            "Does `candidate` actively work against the plan in `deck`, for example exiling "
            "its own graveyard in a graveyard deck or punishing its own token creation?"
        )),
    }
    if key_cards:
        questions["synergy"] = Score(
            instructions="Does `candidate`'s oracle text create, use, or reward what `deck.key_cards` do?",
            criteria=SYNERGY_LEVELS,
        )

    async with session(client) as client:
        if client is None:
            return {}
        cards = list(unique.values())
        try:
            responses = await ask_many(
                client, [({"deck": deck, "candidate": c}, questions) for c in cards], FIT_DEADLINE)
            return {
                c["name"]: FitScore(
                    plan_fit=r.scores["plan_fit"].score / 3,
                    synergy=r.scores["synergy"].score / 3 if key_cards else None,
                    anti_synergy=r.nouls["anti_synergy"].noul,
                )
                for c, r in zip(cards, responses)
            }
        except Exception as e:  # partial fit is worse than none: fall back to existing ordering
            logger.warning(f"Fit scoring failed, falling back: {e}")
            return {}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_jev.py tests/test_deck_fit.py -v`
Expected: all PASS, including the existing `TestScoreFit`, `TestDeadline` and cancellation tests, unchanged apart from the `test_no_api_key_returns_empty` edit.

Run: `cd backend && python3 -m pytest`
Expected: 129 passed (123 + 6 new).

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/jev.py backend/app/services/deck_fit.py \
  backend/tests/conftest.py backend/tests/test_jev.py backend/tests/test_deck_fit.py
git commit -m "Move Jev client and concurrency cap into a shared jev module

deck_fit's client factory, timeouts and per-call semaphore move to
app/services/jev.py. Every Jev caller now goes through one cap of 32
in-flight requests per event loop, so concurrent users share Jev's
rate limit. deck_fit behavior is unchanged.

Tests now blank all API keys via conftest, since the repo-root .env
holds a real TypeSafe key.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 2: Minimum plan fit in `rank`

**Files:**
- Modify: `backend/app/services/deck_fit.py` (`rank`)
- Test: `backend/tests/test_deck_fit.py` (`TestRank`)

**Interfaces:**
- Consumes: `LOW_FIT_CUTOFF = 0.34` (already in `deck_fit`).
- Produces: `rank(names, fit, freq) -> List[str]`, same signature. It now also drops names with `fit[n].plan_fit < LOW_FIT_CUTOFF`.

- [ ] **Step 1: Write the failing test and update the one it invalidates**

In `backend/tests/test_deck_fit.py`, `class TestRank`, replace `test_drops_anti_synergy_and_orders_by_total` (its "Meh" card had plan fit 0.3, which is now below the cutoff) and add a test:
```python
    def test_drops_anti_synergy_and_orders_by_total(self):
        fit = {"Good": fs(1.0, 1.0), "Meh": fs(0.4, 0.3), "Anti": fs(1.0, 1.0, anti=0.9)}
        assert deck_fit.rank(["Meh", "Anti", "Good"], fit, {}) == ["Good", "Meh"]

    def test_drops_plan_fit_below_cutoff_even_when_popular(self):
        fit = {"Fits": fs(0.34), "Low": fs(0.33, 1.0)}
        assert deck_fit.rank(["Low", "Fits"], fit, {"low": 99}) == ["Fits"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_deck_fit.py::TestRank -v`
Expected: `test_drops_plan_fit_below_cutoff_even_when_popular` FAILS with `assert ['Low', 'Fits'] == ['Fits']`

- [ ] **Step 3: Implement**

In `rank`, replace the docstring's first line and the `kept` line:
```python
def rank(names: Sequence[str], fit: Dict[str, FitScore], freq: Dict[str, int]) -> List[str]:
    """Drop anti-synergy and below-minimum plan fit, then sort by weighted fit +
    normalized meta frequency.

    `fit` must cover every name. `freq` is keyed by lowercase name.
    Sort is stable, so ties keep retrieval order.
    """
    top = max(freq.values(), default=0) or 1

    def total(n: str) -> float:
        f = fit[n]
        syn = f.plan_fit if f.synergy is None else f.synergy
        return (WEIGHTS["plan_fit"] * f.plan_fit + WEIGHTS["synergy"] * syn
                + WEIGHTS["meta"] * freq.get(n.lower(), 0) / top)

    kept = [n for n in names
            if fit[n].anti_synergy < ANTI_SYNERGY_CUTOFF and fit[n].plan_fit >= LOW_FIT_CUTOFF]
    return sorted(kept, key=total, reverse=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_deck_fit.py tests/test_guided_fit.py -v`
Expected: all PASS. The guided-fit tests use plan fits >= 0.4 for the cards they expect, except `D` (0.1) in the overlapping-roles test, which was never expected in the output.

Run: `cd backend && python3 -m pytest`
Expected: 130 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/deck_fit.py backend/tests/test_deck_fit.py
git commit -m "Drop candidates below minimum plan fit when ranking

rank() now also drops cards whose plan fit is below LOW_FIT_CUTOFF
(0.34, the flag_low_fit threshold), so a popular low-fit card such as
Together as One at 0.21 no longer fills a suggestion slot. A role may
show fewer cards. The fallback ordering is unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 3: Role check in `score_fit` and the guided fit path

**Files:**
- Modify: `backend/app/models/card.py` (typing import line 3; add `ROLE_DEFINITIONS` and `ROLE_MAP` after `CARD_ROLES`)
- Modify: `backend/app/services/guided_builder.py` (import line 14; delete the `ROLE_MAP` block at lines 64-89; fit path in `suggest_cards_for_strategy`, lines 493-514)
- Modify: `backend/app/services/deck_fit.py` (`FitScore`, new `ROLE_FIT_CUTOFF` and `role_description`, `score_fit`)
- Test: `backend/tests/test_deck_fit.py`, `backend/tests/test_guided_fit.py`

**Interfaces:**
- Consumes: `jev.session`, `jev.ask_many` (Task 1); `rank` (Task 2).
- Produces:
  - `app.models.card.ROLE_DEFINITIONS: Dict[str, str]` (keys == `CARD_ROLES`)
  - `app.models.card.ROLE_MAP: Dict[str, List[str]]` (moved; `guided_builder.ROLE_MAP` still resolves)
  - `deck_fit.ROLE_FIT_CUTOFF = 0.5`
  - `deck_fit.role_description(role: str) -> str`
  - `FitScore.roles: Dict[str, float]` (default `{}`), keyed by guided role name
  - `async score_fit(identity, key_cards, candidates, client=None, roles_by_candidate: Optional[Dict[str, List[str]]] = None) -> Dict[str, FitScore]`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_deck_fit.py`:
```python
class TestRoleCheck:
    async def test_role_nouls_per_candidate_roles(self):
        def answer(state, qs):
            nouls = {k: 0.0 for k, q in qs.items() if type(q).__name__ == "Noul"}
            nouls.update({k: 0.8 for k in qs if k.startswith("role:")})
            return nouls, {"plan_fit": 3.0}
        client = FakeClient(answer)
        fit = await deck_fit.score_fit(
            DeckIdentity(), [], [card("A"), card("B")], client=client,
            roles_by_candidate={"A": ["removal", "cards with surveil"]})
        assert fit["A"].roles == {"removal": 0.8, "cards with surveil": 0.8}
        assert fit["B"].roles == {}
        qs_a = next(qs for s, qs in client.calls if s["candidate"]["name"] == "A")
        assert qs_a["role:0"].instructions["role"] == deck_fit.role_description("removal")
        assert qs_a["role:1"].instructions["role"] == "cards with surveil"
        qs_b = next(qs for s, qs in client.calls if s["candidate"]["name"] == "B")
        assert not any(k.startswith("role:") for k in qs_b)


def test_role_description_expands_role_map_and_falls_back_to_name():
    from app.models.card import ROLE_DEFINITIONS
    assert deck_fit.role_description("Counterspells ") == ROLE_DEFINITIONS["counterspell"]
    removal = deck_fit.role_description("removal")
    assert ROLE_DEFINITIONS["removal_targeted"] in removal and ROLE_DEFINITIONS["removal_mass"] in removal
    assert deck_fit.role_description("cards with surveil") == "cards with surveil"
    assert deck_fit.role_description("payoffs") == "payoffs"


def test_role_map_targets_are_defined_roles():
    from app.models.card import CARD_ROLES, ROLE_DEFINITIONS, ROLE_MAP
    assert set(ROLE_DEFINITIONS) == set(CARD_ROLES)
    assert all(r in ROLE_DEFINITIONS for targets in ROLE_MAP.values() for r in targets)
```

In `backend/tests/test_guided_fit.py`, every fake `score` must accept the new keyword. Replace each occurrence of
```python
    async def score(identity, keys, cands, client=None):
```
with
```python
    async def score(identity, keys, cands, client=None, roles_by_candidate=None):
```
(There are six. Use your editor's replace-all.) Then append:
```python
async def test_role_check_drops_card_from_failing_role_only(analyzer, monkeypatch):
    a, pools = analyzer
    pools["lists"] = {"removal": ["A", "B"], "threats": ["A", "C"]}
    seen = {}
    role_fit = {"A": {"removal": 0.1, "threats": 0.9}, "B": {"removal": 0.9}, "C": {"threats": 0.9}}
    plan = {"A": 0.9, "B": 0.5, "C": 0.5}

    async def score(identity, keys, cands, client=None, roles_by_candidate=None):
        seen.update(roles_by_candidate)
        return {c["name"]: FitScore(plan_fit=plan[c["name"]], synergy=None, anti_synergy=0.0,
                                    roles=role_fit[c["name"]]) for c in cands}
    monkeypatch.setattr(guided_builder, "score_fit", score)
    out = await a.suggest_cards_for_strategy("s", [], ["removal", "threats"], [], cards_per_role=2,
                                             identity=DeckIdentity(tags=["x"]))
    assert seen == {"A": ["removal", "threats"], "B": ["removal"], "C": ["threats"]}
    assert [c["card_name"] for c in out["removal"]] == ["B"]
    assert [c["card_name"] for c in out["threats"]] == ["A", "C"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_fit.py tests/test_guided_fit.py -v`
Expected: `ImportError: cannot import name 'ROLE_DEFINITIONS'`, `AttributeError: module 'app.services.deck_fit' has no attribute 'role_description'`, `TypeError: score_fit() got an unexpected keyword argument 'roles_by_candidate'`, and `FitScore` rejecting `roles`.

- [ ] **Step 3: Add role definitions and move `ROLE_MAP`**

`backend/app/models/card.py`, line 3:
```python
from typing import Dict, List
```

Directly after the closing `]` of `CARD_ROLES`, add:
```python
# What each system role means. Used as Jev question text for role tagging and role checks.
ROLE_DEFINITIONS: Dict[str, str] = {
    "removal_targeted": "Destroys, exiles, deals damage to, or otherwise removes a single opposing creature or planeswalker",
    "removal_mass": "Board wipe: destroys, exiles, or kills multiple creatures at once",
    "removal_artifact_enchantment": "Destroys or exiles artifacts or enchantments",
    "card_draw": "Draws one or more cards (searching the library for a specific card or land is not card draw)",
    "card_selection": "Scry, surveil, looks at top cards of the library, or otherwise filters draws",
    "ramp": "Accelerates mana beyond one land per turn: a mana creature or rock, or putting extra lands onto the battlefield. A land tapping for mana is not ramp",
    "counterspell": "Counters spells",
    "discard": "Makes an opponent discard cards",
    "threat_cheap": "An efficient creature or threat with mana value 2 or less that pressures the opponent in combat (mana and utility creatures do not count)",
    "threat_midrange": "A value creature or planeswalker with mana value 3 or 4",
    "threat_finisher": "A game-ending threat with mana value 5 or more",
    "protection": "Gives hexproof, indestructible, or ward, or prevents damage to your permanents",
    "burn": "Deals direct damage to players or any target",
    "lifegain": "Gaining life is a main purpose of the card, not a side effect of another ability or of a land entering",
    "recursion": "Returns cards from a graveyard to hand or battlefield",
    "graveyard_hate": "Exiles cards from graveyards",
    "tutor": "Searches the library for a card of the player's choice (not only a basic land) and puts it into hand or onto the battlefield",
    "land_fixing_untapped": "A land producing two or more colors that can enter untapped",
    "land_fixing_tapped": "A land producing two or more colors that always enters tapped",
    "land_utility": "A land with a useful ability beyond making mana, such as destroying a land or drawing cards. Entering tapped, gaining 1 life on entry, or becoming a creature do not count",
    "land_creature": "A land that can become a creature",
    "land_basic": "A basic land (Plains, Island, Swamp, Mountain, Forest)",
}

# Map user-facing role names (chat tools, guided builder) to system role names in card_roles
ROLE_MAP: Dict[str, List[str]] = {
    "threats": ["threat_cheap", "threat_midrange", "threat_finisher"],
    "creatures": ["threat_cheap", "threat_midrange", "threat_finisher"],
    "removal": ["removal_targeted", "removal_mass", "removal_artifact_enchantment"],
    "card advantage": ["card_draw", "card_selection"],
    "card draw": ["card_draw", "card_selection"],
    "counterspells": ["counterspell"],
    "protection": ["protection"],
    "ramp": ["ramp"],
    "burn": ["burn"],
    "recursion": ["recursion"],
    "finishers": ["threat_finisher"],
    "interaction": ["removal_targeted", "counterspell"],
    "discard": ["discard"],
    "lifegain": ["lifegain"],
    "graveyard hate": ["graveyard_hate"],
    "tutors": ["tutor"],
    "sacrifice outlets": ["recursion"],
    "board wipes": ["removal_mass"],
    "spot removal": ["removal_targeted"],
    "cheap threats": ["threat_cheap"],
    "big threats": ["threat_finisher"],
    "top end": ["threat_finisher"],
    "early threats": ["threat_cheap"],
}
```

`backend/app/services/guided_builder.py`: delete the block from `# Map user-facing role names (from Claude tool calls) to system role names in card_roles table` through the closing `}` of `ROLE_MAP` (lines 64-89). Replace import line 14 with:
```python
from app.models.card import ROLE_MAP
from app.services.deck_fit import (
    ROLE_FIT_CUTOFF, DeckIdentity, card_payload, is_land, load_payloads, rank, score_fit,
)
```

- [ ] **Step 4: Add the role check to `deck_fit`**

In `backend/app/services/deck_fit.py`, add to the imports:
```python
from app.models.card import ROLE_DEFINITIONS, ROLE_MAP
```

After `LOW_FIT_CUTOFF = 0.34`, add:
```python
ROLE_FIT_CUTOFF = 0.5
```

Replace `class FitScore` with:
```python
class FitScore(BaseModel):
    plan_fit: float              # 0-1
    synergy: Optional[float]     # 0-1, None when the deck has no key cards
    anti_synergy: float          # noul
    roles: Dict[str, float] = Field(default_factory=dict)  # guided role name -> noul
```

After `is_land`, add:
```python
def role_description(role: str) -> str:
    """What a guided role asks for: ROLE_MAP names expand to their system roles'
    definitions; anything else (e.g. "cards with surveil") is its own description."""
    system = ROLE_MAP.get(role.lower().strip())
    return "; or ".join(ROLE_DEFINITIONS[r] for r in system) if system else role
```

Replace the whole `score_fit` function with:
```python
async def score_fit(
    identity: DeckIdentity,
    key_cards: List[Dict[str, str]],
    candidates: List[Dict[str, str]],
    client=None,
    roles_by_candidate: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, FitScore]:
    """One concurrent Jev request per unique candidate, plus one role Noul for each
    role in `roles_by_candidate[name]`. Empty dict if any request fails."""
    unique = {c["name"]: c for c in candidates}
    if not unique:
        return {}
    roles_by_candidate = roles_by_candidate or {}

    deck = {"themes": identity.tags, "request": identity.request_text or "", "key_cards": key_cards}
    questions: Dict[str, Any] = {
        "plan_fit": Score(
            instructions="How well does `candidate` advance the plan described by `deck.themes` and `deck.request`?",
            criteria=PLAN_FIT_LEVELS,
        ),
        "anti_synergy": Noul(instructions=(
            "Does `candidate` actively work against the plan in `deck`, for example exiling "
            "its own graveyard in a graveyard deck or punishing its own token creation?"
        )),
    }
    if key_cards:
        questions["synergy"] = Score(
            instructions="Does `candidate`'s oracle text create, use, or reward what `deck.key_cards` do?",
            criteria=SYNERGY_LEVELS,
        )

    def request(card: Dict[str, str]):
        qs = dict(questions)
        for i, role in enumerate(roles_by_candidate.get(card["name"], [])):
            qs[f"role:{i}"] = Noul(instructions={
                "question": "Does `candidate` fill this role in a deck?",
                "role": role_description(role),
            })
        return {"deck": deck, "candidate": card}, qs

    async with session(client) as client:
        if client is None:
            return {}
        cards = list(unique.values())
        try:
            responses = await ask_many(client, [request(c) for c in cards], FIT_DEADLINE)
            return {
                c["name"]: FitScore(
                    plan_fit=r.scores["plan_fit"].score / 3,
                    synergy=r.scores["synergy"].score / 3 if key_cards else None,
                    anti_synergy=r.nouls["anti_synergy"].noul,
                    roles={role: r.nouls[f"role:{i}"].noul
                           for i, role in enumerate(roles_by_candidate.get(c["name"], []))},
                )
                for c, r in zip(cards, responses)
            }
        except Exception as e:  # partial fit is worse than none: fall back to existing ordering
            logger.warning(f"Fit scoring failed, falling back: {e}")
            return {}
```

- [ ] **Step 5: Filter by role fit in the guided fit path**

In `backend/app/services/guided_builder.py`, `suggest_cards_for_strategy`, replace everything from `        # Collect each role independently and dedupe after ranking` to the final `return self._take_unique(ordered, pool, cards_per_role, fit)` with:
```python
        # Collect each role independently and dedupe after ranking, so one role's
        # discarded candidates don't starve later roles.
        pool = {}
        for role in roles:
            got = await self._collect_role_candidates(
                strategy, colors, [role], existing_cards, format,
                cards_per_role * self.FIT_POOL_MULTIPLIER)
            pool[role] = [c for c in got.get(role, []) if not is_land(card_payload(c))]
        candidates = [card_payload(c) for cards in pool.values() for c in cards]
        # The roles whose pools each candidate came from; Jev checks it fills each one.
        roles_by_candidate: Dict[str, List[str]] = {}
        for role, cards in pool.items():
            for c in cards:
                roles_by_candidate.setdefault(c["card_name"], []).append(role)
        keys = await load_payloads(self.db, identity.key_cards)
        fit = await score_fit(identity, keys, candidates, roles_by_candidate=roles_by_candidate)
        if not fit:
            return self._take_unique(
                {r: [c["card_name"] for c in cs] for r, cs in pool.items()},
                pool, cards_per_role, fit)

        freq = await self._rank_cards_by_tournament_frequency(list(fit), format=format)
        ordered = {
            role: rank([n for n in dict.fromkeys(c["card_name"] for c in cards)
                        if fit[n].roles.get(role, 1.0) >= ROLE_FIT_CUTOFF], fit, freq)
            for role, cards in pool.items()
        }
        return self._take_unique(ordered, pool, cards_per_role, fit)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_deck_fit.py tests/test_guided_fit.py tests/test_keyword_recommendations.py -v`
Expected: all PASS.

Run: `cd backend && python3 -m pytest`
Expected: 134 passed.

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/card.py backend/app/services/guided_builder.py \
  backend/app/services/deck_fit.py backend/tests/test_deck_fit.py backend/tests/test_guided_fit.py
git commit -m "Check each suggested card fills the role it is suggested for

The guided fit request now carries one Jev Noul per role whose pool a
candidate came from, described through ROLE_MAP and the new
ROLE_DEFINITIONS (or the role name itself for keyword roles). A card
below 0.5 for a role is dropped from that role only, so Basilisk
Collar stops showing up as removal. If the fit request fails, the round
still falls back to retrieval order.

ROLE_MAP moves to app/models/card.py next to CARD_ROLES to avoid an
import cycle; guided_builder re-exports it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---
### Task 4: Card-role tagging with Jev, plus the role eval

**Files:**
- Create: `backend/tests/jev_fake.py`
- Modify: `backend/app/jobs/classify_card_roles.py` (whole file)
- Create: `backend/scripts/eval_roles.py`
- Test: `backend/tests/test_classify_card_roles.py`

**Interfaces:**
- Consumes: `jev.session`, `jev.ask_many` (Task 1); `ROLE_DEFINITIONS`, `CARD_ROLES` (Task 3).
- Produces:
  - `tests/jev_fake.FakeJev(answer=None, fail=None, delay=0.0, fill=True)`: a test stub for `AsyncTypeSafeClient`. `answer(state, questions) -> Dict[str, value]`, where the value is a float for a Noul or Score and, for a Choice, a label string or a `{"choice", "confidence", "probabilities"}` dict. Unanswered questions get 0.0, or the first Choice label at confidence 1.0, unless `fill=False`, in which case they are omitted (a partial answer). `fail(state) -> bool` raises. `.calls` records `(state, questions, kwargs)`. Tasks 6-8 use it.
  - `classify_card_roles.ROLE_THRESHOLD = 0.5`, `TAG_REQUEST_TIMEOUT = 15.0`, `EFFICIENCY_LEVELS`
  - `async classify_cards_batch(cards: List[Card], client=None) -> List[Dict[str, Any]]`, which returns `[{"name", "roles": [{"role", "efficiency", "confidence"}]}]`, or `[]` when the batch is skipped
  - `save_card_roles(db, cards, classifications)`: same contract; now stores `confidence` from each role dict and `reasoning` as null
  - `get_unclassified_cards`, `classify_all_cards` and `get_classification_stats`: contracts unchanged

- [ ] **Step 1: Write the shared Jev test stub**

`backend/tests/jev_fake.py`:
```python
"""A stand-in for AsyncTypeSafeClient that answers Noul, Score and Choice questions."""

import asyncio
from types import SimpleNamespace


class FakeJev:
    """`answer(state, questions)` returns {question_id: value}: a float for a Noul or
    Score; for a Choice, a label or {"choice", "confidence", "probabilities"}.
    Unanswered questions get 0.0 / the first Choice label at confidence 1.0, or are
    left out entirely when fill=False (a partial answer). `fail(state)` raising
    simulates an API error."""

    def __init__(self, answer=None, fail=None, delay=0.0, fill=True):
        self.answer = answer or (lambda state, questions: {})
        self.fail = fail or (lambda state: False)
        self.delay, self.fill, self.calls = delay, fill, []

    async def system_one(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail(state):
            raise RuntimeError("jev unavailable")
        got = self.answer(state, questions)
        nouls, scores, choices = {}, {}, {}
        for qid, q in questions.items():
            if qid not in got and not self.fill:
                continue
            value = got.get(qid)
            kind = type(q).__name__
            if kind == "Noul":
                nouls[qid] = SimpleNamespace(noul=0.0 if value is None else value)
            elif kind == "Score":
                scores[qid] = SimpleNamespace(score=0.0 if value is None else value)
            else:
                if value is None:
                    value = next(iter(q.criteria))
                if isinstance(value, str):
                    value = {"choice": value, "confidence": 1.0, "probabilities": {value: 1.0}}
                choices[qid] = SimpleNamespace(**value)
        return SimpleNamespace(nouls=nouls, scores=scores, choices=choices)
```

- [ ] **Step 2: Write the failing tests**

`backend/tests/test_classify_card_roles.py`:
```python
"""Role tagging with Jev: thresholds, efficiency mapping, stored confidence, batch failure."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.dialects import postgresql

from app.jobs import classify_card_roles as job
from app.models.card import CARD_ROLES
from tests.jev_fake import FakeJev


def card(name, **kw):
    base = dict(name=name, mana_cost="{1}{B}{B}", cmc=3.0, type_line="Instant",
                oracle_text="Destroy target creature.", power=None, toughness=None)
    return SimpleNamespace(**{**base, **kw})


async def test_one_request_per_card_with_a_noul_and_score_per_role():
    client = FakeJev()
    await job.classify_cards_batch([card("Murder"), card("Shock")], client=client)
    assert len(client.calls) == 2
    state, questions, kwargs = client.calls[0]
    assert state == {"card": {"name": "Murder", "mana_cost": "{1}{B}{B}", "mana_value": 3.0,
                              "type_line": "Instant", "oracle_text": "Destroy target creature."}}
    assert set(questions) == {f"is:{r}" for r in CARD_ROLES} | {f"eff:{r}" for r in CARD_ROLES}
    assert kwargs == {"timeout": job.TAG_REQUEST_TIMEOUT}


async def test_power_toughness_included_for_creatures():
    client = FakeJev()
    await job.classify_cards_batch(
        [card("Bear", type_line="Creature — Bear", power="2", toughness="2")], client=client)
    assert client.calls[0][0]["card"]["power_toughness"] == "2/2"


async def test_threshold_efficiency_and_confidence():
    def answer(state, questions):
        return {"is:removal_targeted": 0.834, "eff:removal_targeted": 3.6,
                "is:burn": 0.5, "eff:burn": 0.0,
                "is:card_draw": 0.49, "eff:card_draw": 4.0}
    out = await job.classify_cards_batch([card("Murder")], client=FakeJev(answer))
    assert out == [{"name": "Murder", "roles": [
        {"role": "removal_targeted", "efficiency": 5, "confidence": 0.83},
        {"role": "burn", "efficiency": 1, "confidence": 0.5},
    ]}]


async def test_any_failure_skips_the_batch():
    client = FakeJev(fail=lambda state: state["card"]["name"] == "Shock")
    assert await job.classify_cards_batch([card("Murder"), card("Shock")], client=client) == []


async def test_partial_answer_skips_the_batch():
    assert await job.classify_cards_batch([card("Murder")], client=FakeJev(fill=False)) == []


async def test_no_key_skips_the_batch():
    assert await job.classify_cards_batch([card("Murder")]) == []


async def test_save_stores_noul_as_confidence_and_no_reasoning():
    ids = MagicMock()
    ids.all.return_value = [("id-1", "Murder")]
    db = MagicMock(execute=AsyncMock(side_effect=[ids, None]), commit=AsyncMock())
    saved = await job.save_card_roles(db, [card("Murder")], [
        {"name": "Murder", "roles": [{"role": "removal_targeted", "efficiency": 5, "confidence": 0.83}]}])
    assert saved == 1
    params = db.execute.call_args_list[1][0][0].compile(dialect=postgresql.dialect()).params
    assert params["confidence"] == 0.83
    assert params["efficiency"] == 5
    assert params["reasoning"] is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_classify_card_roles.py -v`
Expected: FAIL: `TypeError: classify_cards_batch() got an unexpected keyword argument 'client'`, and the save test fails on `confidence == 0.9`.

- [ ] **Step 4: Rewrite the job**

`backend/app/jobs/classify_card_roles.py` (whole file):
```python
"""
Card Role Classification Job
Schedule: Run manually or after Scryfall sync

Uses Jev (TypeSafe System One) to tag Standard-legal cards with functional
deck-building roles (removal, threats, ramp, etc.): one request per card,
with a Noul per role and an efficiency Score per role.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct
from sqlalchemy.dialects.postgresql import insert
from typesafe_sdk import Noul, Score

from app.db.session import async_session_factory
from app.models.card import Card, CardRole, CARD_ROLES, ROLE_DEFINITIONS
from app.services import jev

logger = logging.getLogger(__name__)

# Cards per batch; each card is one Jev request, all sent concurrently through the shared cap
BATCH_SIZE = 50

ROLE_THRESHOLD = 0.5
# ponytail: 44 questions per card can outlast the 2 s chat timeout; offline job, so allow longer
TAG_REQUEST_TIMEOUT = 15.0

EFFICIENCY_LEVELS = [
    "Pays far more mana or conditions than usual for this effect",
    "Below rate; the effect is narrow or overcosted",
    "Fair rate; a typical cost for this effect",
    "Strong rate; cheap or flexible for what it does",
    "Exceptional rate; the effect is far cheaper or stronger than typical",
]


def _card_state(card: Card) -> Dict[str, Any]:
    state = {
        "name": card.name,
        "mana_cost": card.mana_cost or "",
        "mana_value": card.cmc,
        "type_line": card.type_line or "",
        "oracle_text": card.oracle_text or "",
    }
    if card.power and card.toughness:
        state["power_toughness"] = f"{card.power}/{card.toughness}"
    return {"card": state}


def _role_questions() -> Dict[str, Any]:
    """One Noul per role plus a speculative efficiency Score per role, all in one request."""
    questions: Dict[str, Any] = {}
    for role, definition in ROLE_DEFINITIONS.items():
        questions[f"is:{role}"] = Noul(instructions={
            "question": "Does `card` fill this deck-building role, judged from its oracle text and stats?",
            "role": definition,
        })
        questions[f"eff:{role}"] = Score(
            instructions={
                "question": "Assuming `card` fills this role, how efficient is it at it relative to its mana cost?",
                "role": definition,
            },
            criteria=EFFICIENCY_LEVELS,
        )
    return questions


def _roles_from(resp: Any) -> List[Dict[str, Any]]:
    roles = []
    for role in ROLE_DEFINITIONS:
        p = resp.nouls[f"is:{role}"].noul
        if p >= ROLE_THRESHOLD:
            roles.append({
                "role": role,
                "efficiency": round(resp.scores[f"eff:{role}"].score) + 1,
                "confidence": round(p, 2),
            })
    return roles


async def get_unclassified_cards(db: AsyncSession, limit: int = 500) -> List[Card]:
    """Get Standard-legal cards that haven't been classified yet (unique names only).

    Returns one representative Card object per unique card name.
    This avoids processing the same card multiple times across different printings.
    """
    # Subquery to find card names that already have roles
    # Join CardRole -> Card to get names of classified cards
    classified_names_subquery = (
        select(Card.name)
        .join(CardRole, Card.id == CardRole.card_id)
        .distinct()
        .scalar_subquery()
    )

    # First get distinct unclassified card names
    names_result = await db.execute(
        select(distinct(Card.name))
        .where(Card.is_standard_legal == True)
        .where(Card.name.notin_(classified_names_subquery))
        .order_by(Card.name)
        .limit(limit)
    )
    unique_names = [row[0] for row in names_result.all()]

    if not unique_names:
        return []

    # Now fetch one card per name (any printing will do, they have same oracle text)
    result = await db.execute(
        select(Card)
        .where(Card.name.in_(unique_names))
        .distinct(Card.name)
        .order_by(Card.name, Card.id)  # Consistent ordering
    )
    return list(result.scalars().all())


async def classify_cards_batch(cards: List[Card], client=None) -> List[Dict[str, Any]]:
    """Tag a batch with Jev: one request per card, concurrent through the shared cap.

    Returns [] (the batch is skipped and its cards stay unclassified for the next
    run) when Jev is not configured, any request fails, or an answer is incomplete.
    """
    if not cards:
        return []
    questions = _role_questions()
    async with jev.session(client) as client:
        if client is None:
            logger.error("TYPESAFE_API_KEY not configured")
            return []
        try:
            responses = await jev.ask_many(
                client, [(_card_state(c), questions) for c in cards], timeout=TAG_REQUEST_TIMEOUT)
            return [{"name": c.name, "roles": _roles_from(r)} for c, r in zip(cards, responses)]
        except Exception as e:
            logger.error(f"Role tagging failed, skipping batch of {len(cards)} cards: {e}")
            return []


async def save_card_roles(
    db: AsyncSession,
    cards: List[Card],
    classifications: List[Dict[str, Any]]
) -> int:
    """Save classification results to database for all printings of each card."""
    # Get all card names from this batch
    card_names = [card.name for card in cards]

    # Fetch ALL card IDs for these names (all printings)
    all_cards_result = await db.execute(
        select(Card.id, Card.name)
        .where(Card.name.in_(card_names))
    )
    # Build lookup: card_name -> list of all card IDs with that name
    name_to_ids: Dict[str, List] = defaultdict(list)
    for card_id, card_name in all_cards_result.all():
        name_to_ids[card_name].append(card_id)

    saved = 0
    for classification in classifications:
        card_name = classification.get("name")
        card_ids = name_to_ids.get(card_name, [])

        if not card_ids:
            logger.warning(f"Card not found in batch: {card_name}")
            continue

        roles = classification.get("roles", [])
        for role_data in roles:
            role = role_data.get("role")

            # Validate role
            if role not in CARD_ROLES:
                logger.warning(f"Invalid role '{role}' for card {card_name}")
                continue

            efficiency = role_data.get("efficiency")
            confidence = role_data.get("confidence")
            reasoning = role_data.get("reasoning")

            # Save role for ALL printings of this card
            for card_id in card_ids:
                stmt = insert(CardRole).values(
                    card_id=card_id,
                    role=role,
                    efficiency=efficiency,
                    confidence=confidence,
                    reasoning=reasoning,
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_card_role",
                    set_={
                        "efficiency": stmt.excluded.efficiency,
                        "confidence": stmt.excluded.confidence,
                        "reasoning": stmt.excluded.reasoning,
                        "created_at": datetime.utcnow(),
                    },
                )
                await db.execute(stmt)
                saved += 1

    await db.commit()
    return saved


async def classify_all_cards() -> Dict[str, Any]:
    """
    Main classification function - classifies all unclassified Standard cards.

    Returns:
        Dict with classification statistics
    """
    start_time = datetime.utcnow()
    stats = {
        "started_at": start_time.isoformat(),
        "cards_processed": 0,
        "roles_saved": 0,
        "batches": 0,
        "errors": [],
    }

    async with async_session_factory() as db:
        try:
            # Get count of unique unclassified card NAMES (not printings)
            classified_names_subquery = (
                select(Card.name)
                .join(CardRole, Card.id == CardRole.card_id)
                .distinct()
                .scalar_subquery()
            )
            count_result = await db.execute(
                select(func.count(distinct(Card.name)))
                .where(Card.is_standard_legal == True)
                .where(Card.name.notin_(classified_names_subquery))
            )
            total_unclassified = count_result.scalar()
            logger.info(f"Found {total_unclassified} unique unclassified Standard cards")

            if total_unclassified == 0:
                logger.info("All cards already classified")
                stats["completed_at"] = datetime.utcnow().isoformat()
                return stats

            # Snapshot once and make a single pass: cards that fit no role are
            # never saved, so re-querying "unclassified" each batch would loop forever
            pending = await get_unclassified_cards(db, limit=total_unclassified)
            for i in range(0, len(pending), BATCH_SIZE):
                cards = pending[i:i + BATCH_SIZE]

                stats["batches"] += 1
                logger.info(f"Batch {stats['batches']}: Classifying {len(cards)} cards")

                classifications = await classify_cards_batch(cards)

                if classifications:
                    saved = await save_card_roles(db, cards, classifications)
                    stats["roles_saved"] += saved
                    logger.info(f"Batch {stats['batches']}: Saved {saved} roles")
                else:
                    logger.warning(f"Batch {stats['batches']}: No classifications returned")

                stats["cards_processed"] += len(cards)

        except Exception as e:
            logger.error(f"Classification job failed: {e}")
            stats["errors"].append(str(e))
            raise

    stats["completed_at"] = datetime.utcnow().isoformat()
    logger.info(f"Classification complete: {stats}")
    return stats


async def get_classification_stats() -> Dict[str, Any]:
    """Get statistics about current card classifications."""
    async with async_session_factory() as db:
        # Total unique Standard card names
        total_result = await db.execute(
            select(func.count(distinct(Card.name))).where(Card.is_standard_legal == True)
        )
        total_cards = total_result.scalar()

        # Unique card names with roles
        classified_result = await db.execute(
            select(func.count(distinct(Card.name)))
            .select_from(Card)
            .join(CardRole, Card.id == CardRole.card_id)
        )
        classified_cards = classified_result.scalar()

        # Roles by type (count unique card names per role)
        role_counts_result = await db.execute(
            select(CardRole.role, func.count(distinct(Card.name)))
            .select_from(CardRole)
            .join(Card, Card.id == CardRole.card_id)
            .group_by(CardRole.role)
            .order_by(func.count(distinct(Card.name)).desc())
        )
        role_counts = {row[0]: row[1] for row in role_counts_result.all()}

        return {
            "total_unique_standard_cards": total_cards,
            "classified_unique_cards": classified_cards,
            "unclassified_unique_cards": total_cards - classified_cards,
            "total_role_assignments": sum(role_counts.values()),
            "unique_cards_by_role": role_counts,
        }


if __name__ == "__main__":
    # Allow running directly for testing
    asyncio.run(classify_all_cards())
```

The 1-second sleep between batches is gone: the shared cap now does the rate limiting.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_classify_card_roles.py -v`
Expected: 7 PASS.

Run: `cd backend && python3 -m pytest`
Expected: 141 passed.

- [ ] **Step 6: Commit the job**

```bash
git add backend/tests/jev_fake.py backend/tests/test_classify_card_roles.py \
  backend/app/jobs/classify_card_roles.py
git commit -m "Tag card roles with Jev instead of free-form LLM JSON

One Jev request per card asks a Noul per role (assigned at >= 0.5)
and an efficiency Score per role (round(score) + 1, 1-5). The Noul is
stored as CardRole.confidence and reasoning is left null. Cards run
concurrently through the shared Jev cap; a batch with any failed
request is skipped and logged, and its cards stay unclassified for the
next run.

Re-tagging existing data is manual: DELETE FROM card_roles, then run
the job.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

- [ ] **Step 7: Write the role eval script**

`backend/scripts/eval_roles.py`:
```python
"""
Run hand-labeled cards through real Jev role tagging and report per-role
precision and recall.

Usage (needs TYPESAFE_API_KEY in .env):  cd backend && python3 scripts/eval_roles.py
"""

import asyncio
import os
import sys
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.jobs import classify_card_roles as job  # noqa: E402


def c(name, cost, cmc, type_line, text, pt=None):
    power, toughness = pt.split("/") if pt else (None, None)
    return SimpleNamespace(name=name, mana_cost=cost, cmc=cmc, type_line=type_line,
                           oracle_text=text, power=power, toughness=toughness)


# (card, expected roles)
CASES = [
    (c("Basilisk Collar", "{1}", 1, "Artifact — Equipment",
       "Equipped creature has deathtouch and lifelink.\nEquip {2}"), set()),
    (c("Murder", "{1}{B}{B}", 3, "Instant", "Destroy target creature."), {"removal_targeted"}),
    (c("Consider", "{U}", 1, "Instant", "Surveil 1.\nDraw a card."), {"card_selection", "card_draw"}),
    (c("Forest", "", 0, "Basic Land — Forest", "({T}: Add {G}.)"), {"land_basic"}),
    (c("Hallowed Fountain", "", 0, "Land — Plains Island",
       "({T}: Add {W} or {U}.)\nAs Hallowed Fountain enters the battlefield, you may pay 2 life. "
       "If you don't, it enters the battlefield tapped."), {"land_fixing_untapped"}),
    (c("Monastery Swiftspear", "{R}", 1, "Creature — Human Monk", "Haste\nProwess", "1/2"),
     {"threat_cheap"}),
    (c("Lightning Bolt", "{R}", 1, "Instant", "Lightning Bolt deals 3 damage to any target."),
     {"burn", "removal_targeted"}),
    (c("Wrath of God", "{2}{W}{W}", 4, "Sorcery", "Destroy all creatures. They can't be regenerated."),
     {"removal_mass"}),
    (c("Counterspell", "{U}{U}", 2, "Instant", "Counter target spell."), {"counterspell"}),
    (c("Negate", "{1}{U}", 2, "Instant", "Counter target noncreature spell."), {"counterspell"}),
    (c("Divination", "{2}{U}", 3, "Sorcery", "Draw two cards."), {"card_draw"}),
    (c("Opt", "{U}", 1, "Instant", "Scry 1.\nDraw a card."), {"card_selection", "card_draw"}),
    (c("Llanowar Elves", "{G}", 1, "Creature — Elf Druid", "{T}: Add {G}.", "1/1"), {"ramp"}),
    (c("Cultivate", "{2}{G}", 3, "Sorcery",
       "Search your library for up to two basic land cards, reveal those cards, put one onto the "
       "battlefield tapped and the other into your hand, then shuffle."), {"ramp"}),
    (c("Thoughtseize", "{B}", 1, "Sorcery",
       "Target player reveals their hand. You choose a nonland card from it. That player discards "
       "that card. You lose 2 life."), {"discard"}),
    (c("Disenchant", "{1}{W}", 2, "Instant", "Destroy target artifact or enchantment."),
     {"removal_artifact_enchantment"}),
    (c("Rest in Peace", "{1}{W}", 2, "Enchantment",
       "When Rest in Peace enters the battlefield, exile all graveyards.\nIf a card or token would be "
       "put into a graveyard from anywhere, exile it instead."), {"graveyard_hate"}),
    (c("Raise Dead", "{B}", 1, "Sorcery", "Return target creature card from your graveyard to your hand."),
     {"recursion"}),
    (c("Demonic Tutor", "{1}{B}", 2, "Sorcery",
       "Search your library for a card, put that card into your hand, then shuffle."), {"tutor"}),
    (c("Heroic Intervention", "{1}{G}", 2, "Instant",
       "Permanents you control gain hexproof and indestructible until end of turn."), {"protection"}),
    (c("Revitalize", "{1}{W}", 2, "Instant", "You gain 3 life.\nDraw a card."), {"lifegain", "card_draw"}),
    (c("Shivan Dragon", "{4}{R}{R}", 6, "Creature — Dragon",
       "Flying\n{R}: Shivan Dragon gets +1/+0 until end of turn.", "5/5"), {"threat_finisher"}),
    (c("Watchwolf", "{G}{W}", 2, "Creature — Wolf", "", "3/3"), {"threat_cheap"}),
    (c("Ravenous Chupacabra", "{2}{B}{B}", 4, "Creature — Beast Horror",
       "When Ravenous Chupacabra enters the battlefield, destroy target creature an opponent controls.",
       "2/2"), {"threat_midrange", "removal_targeted"}),
    (c("Sheoldred, the Apocalypse", "{2}{B}{B}", 4, "Legendary Creature — Phyrexian Praetor",
       "Deathtouch\nWhenever you draw a card, you gain 2 life.\nWhenever an opponent draws a card, "
       "they lose 2 life.", "4/5"), {"threat_midrange"}),
    (c("Lava Spike", "{R}", 1, "Sorcery", "Lava Spike deals 3 damage to target player or planeswalker."),
     {"burn"}),
    (c("Mutavault", "", 0, "Land",
       "{T}: Add {C}.\n{1}: Mutavault becomes a 2/2 creature with all creature types until end of turn. "
       "It's still a land."), {"land_creature"}),
    (c("Field of Ruin", "", 0, "Land",
       "{T}: Add {C}.\n{2}, {T}, Sacrifice Field of Ruin: Destroy target nonbasic land an opponent "
       "controls. Each player searches their library for a basic land card, puts it onto the "
       "battlefield, then shuffles."), {"land_utility"}),
    (c("Dismal Backwater", "", 0, "Land",
       "Dismal Backwater enters the battlefield tapped.\nWhen Dismal Backwater enters the battlefield, "
       "you gain 1 life.\n{T}: Add {U} or {B}."), {"land_fixing_tapped"}),
    (c("Ornithopter", "{0}", 0, "Artifact Creature — Thopter", "Flying", "0/2"), set()),
]

# Spec-required checks: (card name, predicate on the assigned role set, description)
REQUIRED = [
    ("Basilisk Collar", lambda got: "removal_targeted" not in got, "not removal_targeted"),
    ("Murder", lambda got: "removal_targeted" in got, "removal_targeted"),
    ("Consider", lambda got: "card_selection" in got, "card_selection"),
    ("Forest", lambda got: "land_basic" in got, "land_basic"),
    ("Hallowed Fountain", lambda got: any(r.startswith("land_fixing_") for r in got), "land_fixing_*"),
    ("Monastery Swiftspear", lambda got: "threat_cheap" in got, "threat_cheap"),
]


async def main() -> None:
    out = await job.classify_cards_batch([card for card, _ in CASES])
    if not out:
        print("ERROR: no results (no TYPESAFE_API_KEY, or Jev unavailable)")
        return
    got_by_name = {r["name"]: {x["role"] for x in r["roles"]} for r in out}
    tp, fp, fn = Counter(), Counter(), Counter()
    print(f"{'card':28} {'expected':45} got")
    for card, want in CASES:
        got = got_by_name[card.name]
        for r in got & want:
            tp[r] += 1
        for r in got - want:
            fp[r] += 1
        for r in want - got:
            fn[r] += 1
        mark = "" if got == want else "  <-"
        print(f"{card.name:28} {','.join(sorted(want)) or '-':45} {','.join(sorted(got)) or '-'}{mark}")

    print(f"\n{'role':30} precision  recall")
    for role in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[role] / (tp[role] + fp[role]) if tp[role] + fp[role] else float("nan")
        r = tp[role] / (tp[role] + fn[role]) if tp[role] + fn[role] else float("nan")
        print(f"{role:30} {p:9.2f}  {r:6.2f}")
    micro_p = sum(tp.values()) / max(sum(tp.values()) + sum(fp.values()), 1)
    micro_r = sum(tp.values()) / max(sum(tp.values()) + sum(fn.values()), 1)
    print(f"\nmicro precision {micro_p:.2f}  micro recall {micro_r:.2f}")

    failed = [f"{name}: {desc}" for name, ok, desc in REQUIRED if not ok(got_by_name[name])]
    for f in failed:
        print(f"REQUIRED FAILED  {f}")
    passed = not failed and micro_p >= 0.8 and micro_r >= 0.7
    print("PASS" if passed else "FAIL (need all required checks, precision >= 0.80, recall >= 0.70)")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 8: Run it against real Jev**

Run: `cd backend && python3 scripts/eval_roles.py`
Expected: a 30-row table, a per-role table, micro precision/recall, and `PASS`. There must be no `ERROR`. (A pre-run while writing this plan, with these definitions, gave micro precision 0.94, recall 1.00, and all required checks passing. The first-draft definitions scored 0.76 precision, mostly lands tagged `ramp` and Demonic Tutor tagged `card_draw`.) If there is, confirm `TYPESAFE_API_KEY` is set in the repo-root `.env`.

- [ ] **Step 9: Tune if it prints FAIL**

Change only these, one at a time, re-running after each:
1. The wording of the failing role's entry in `ROLE_DEFINITIONS` (`app/models/card.py`), or the `is:` question wording in `_role_questions`.
2. `ROLE_THRESHOLD`. If you change it, update the Global Constraints in this plan and the spec to match.

Re-run `cd backend && python3 -m pytest` after any change. If it still prints FAIL after three wording attempts, stop and report both tables to the human before continuing to Task 5.

- [ ] **Step 10: Commit the eval**

```bash
git add backend/scripts/eval_roles.py backend/app/jobs/classify_card_roles.py backend/app/models/card.py
git commit -m "Add Jev role-tagging eval script

30 hand-labeled cards run through real Jev, with per-role precision and
recall and the spec's required checks (Basilisk Collar is not removal,
Murder, Consider, a basic, a dual, a cheap threat).
Result: micro precision <P>, recall <R>. <one line on any tuning, or 'no tuning needed'>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```
Fill in the `<...>` parts from the actual run before committing.

---

### Task 5: Full-text index and the shortlist pieces in `CardService`

**Files:**
- Create: `backend/alembic/versions/016_add_cards_fts_index.py`
- Modify: `backend/app/services/card_service.py` (imports at lines 1-10; `_execute_vector_query` at lines 411-456; new methods after it)
- Modify: `backend/app/services/guided_builder.py` (`_rank_cards_by_tournament_frequency`, lines 285-323)
- Test: `backend/tests/test_card_shortlist.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (in `app.services.card_service`):
  - `CARD_TSVECTOR: str` (must equal migration 016's expression)
  - `SHORTLIST_SIZE = 60`
  - `CardService._filter_sql(format, standard_only, colors, legality=True, alias="") -> List[str]` (staticmethod; the WHERE fragments shared by the vector, full-text and popular queries)
  - `async CardService.text_search_names(query, format=None, standard_only=True, colors=None, limit=SHORTLIST_SIZE) -> List[str]`
  - `async CardService.popular_card_names(format=None, standard_only=True, colors=None, limit=SHORTLIST_SIZE) -> List[str]`
  - `async CardService.tournament_frequency(card_names: List[str], format="standard") -> Dict[str, int]` (moved from `DeckAnalyzer._rank_cards_by_tournament_frequency`, which now delegates to it)

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_card_shortlist.py`:
```python
"""CardService shortlist pieces: shared SQL filters, full-text names, popular names, FTS index."""

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from app.services import card_service as cs
from app.services.card_service import CardService


def svc(rows=()):
    s = CardService.__new__(CardService)
    result = MagicMock()
    result.all.return_value = list(rows)
    s.db = MagicMock(execute=AsyncMock(return_value=result))
    return s


def sql_of(s):
    stmt, params = s.db.execute.call_args[0]
    return str(stmt), params


class TestFilterSql:
    def test_format_legality_and_colors(self):
        assert CardService._filter_sql("modern", True, ["r", "x"]) == [
            "legalities->>'modern' = 'legal'",
            "(colors <@ ARRAY['R']::varchar[] OR colors = '{}' OR colors IS NULL)",
        ]

    def test_standard_only_without_format_with_alias(self):
        assert CardService._filter_sql(None, True, None, alias="c") == ["c.is_standard_legal = true"]

    def test_view_queries_skip_legality(self):
        assert CardService._filter_sql("standard", True, None, legality=False) == []


class TestTextSearchNames:
    async def test_ors_query_words_and_filters(self):
        s = svc([("Shock",), ("Lightning Strike",)])
        names = await s.text_search_names("Cheap red removal!", format="standard", colors=["R"])
        assert names == ["Shock", "Lightning Strike"]
        sql, params = sql_of(s)
        assert params["q"] == "cheap | red | removal"
        assert cs.CARD_TSVECTOR in sql and "FROM cards" in sql
        assert "legalities->>'standard' = 'legal'" in sql
        assert "colors <@ ARRAY['R']" in sql

    async def test_no_words_skips_the_query(self):
        s = svc()
        assert await s.text_search_names("?! --") == []
        s.db.execute.assert_not_called()


async def test_popular_names_exclude_basics_and_filter_on_card_alias():
    s = svc([("Burst Lightning", 45)])
    assert await s.popular_card_names(format="standard", colors=["R"]) == ["Burst Lightning"]
    sql, params = sql_of(s)
    assert "NOT LIKE 'Basic Land%'" in sql and "c.legalities->>'standard' = 'legal'" in sql
    assert params == {"format": "standard", "limit": cs.SHORTLIST_SIZE}


async def test_popular_names_default_format_is_standard():
    s = svc()
    await s.popular_card_names()
    assert sql_of(s)[1]["format"] == "standard"


async def test_tournament_frequency_lives_on_card_service():
    s = svc([("shock", 7)])
    assert await s.tournament_frequency(["Shock"], format="modern") == {"shock": 7}
    empty = svc()
    assert await empty.tournament_frequency([]) == {}
    empty.db.execute.assert_not_called()


def test_migration_index_matches_query_expression():
    path = Path(__file__).parents[1] / "alembic" / "versions" / "016_add_cards_fts_index.py"
    spec = importlib.util.spec_from_file_location("migration_016", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.revision == "016" and m.down_revision == "015"
    assert m.CARD_TSVECTOR == cs.CARD_TSVECTOR
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_card_shortlist.py -v`
Expected: FAIL with `AttributeError: type object 'CardService' has no attribute '_filter_sql'` (and similar for the other methods), and `FileNotFoundError` for the migration.

- [ ] **Step 3: Write the migration**

`backend/alembic/versions/016_add_cards_fts_index.py`:
```python
"""GIN full-text index on cards, for the semantic-search shortlist (works without OpenAI)."""

from alembic import op

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None

# Must match app.services.card_service.CARD_TSVECTOR exactly, or Postgres won't use the index.
CARD_TSVECTOR = (
    "to_tsvector('english', name || ' ' || coalesce(type_line, '') || ' ' || coalesce(oracle_text, ''))"
)


def upgrade() -> None:
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_cards_fts ON cards USING gin (({CARD_TSVECTOR}))")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_cards_fts")
```

- [ ] **Step 4: Add the shortlist pieces to `CardService`**

`backend/app/services/card_service.py`, replace the top of the file:
```python
from typing import Optional, List, Dict, Any
from uuid import UUID
import logging
```
with:
```python
from typing import Optional, List, Dict, Any
from uuid import UUID
import logging
import re
```

After the `FORMAT_VIEW_MAP` dict, add:
```python
# Full-text document for a card. Migration 016 indexes exactly this expression.
CARD_TSVECTOR = (
    "to_tsvector('english', name || ' ' || coalesce(type_line, '') || ' ' || coalesce(oracle_text, ''))"
)
SHORTLIST_SIZE = 60
VALID_COLORS = {"W", "U", "B", "R", "G"}
```

Replace the start of `_execute_vector_query`, from `"""Build and execute a single vector similarity SQL query."""` through `where_clause = " AND ".join(conditions)`, with:
```python
        """Build and execute a single vector similarity SQL query."""
        # The format views already hold only legal cards; the main table needs the filter.
        conditions = ["embedding IS NOT NULL", *self._filter_sql(
            format, standard_only, colors, legality=(source_table == "cards"))]
        where_clause = " AND ".join(conditions)
```
(The rest of `_execute_vector_query` is unchanged.)

After `_execute_vector_query`, add:
```python
    @staticmethod
    def _filter_sql(
        format: Optional[str],
        standard_only: bool,
        colors: Optional[List[str]],
        legality: bool = True,
        alias: str = "",
    ) -> List[str]:
        """WHERE fragments for format legality and deck colors (a card's colors must be
        a subset of the deck's, or colorless)."""
        p = f"{alias}." if alias else ""
        conditions = []
        if legality:
            if format and format in FORMAT_LEGALITY_MAP:
                conditions.append(f"{p}legalities->>'{FORMAT_LEGALITY_MAP[format]}' = 'legal'")
            elif standard_only:
                conditions.append(f"{p}is_standard_legal = true")
        valid = [c.upper() for c in colors or [] if c.upper() in VALID_COLORS]
        if valid:
            arr = "ARRAY[" + ",".join(f"'{c}'" for c in valid) + "]::varchar[]"
            conditions.append(f"({p}colors <@ {arr} OR {p}colors = '{{}}' OR {p}colors IS NULL)")
        return conditions

    async def text_search_names(
        self,
        query: str,
        format: Optional[str] = None,
        standard_only: bool = True,
        colors: Optional[List[str]] = None,
        limit: int = SHORTLIST_SIZE,
    ) -> List[str]:
        """Card names matching any query word (Postgres full text with english
        stemming and stopwords), best match first. Runs on `cards` so the
        migration-016 index applies."""
        words = re.findall(r"[a-z0-9]+", query.lower())
        if not words:
            return []
        where = " AND ".join([f"{CARD_TSVECTOR} @@ to_tsquery('english', :q)",
                              *self._filter_sql(format, standard_only, colors)])
        sql = text(f"""
            SELECT name
            FROM cards
            WHERE {where}
            GROUP BY name
            ORDER BY MAX(ts_rank({CARD_TSVECTOR}, to_tsquery('english', :q))) DESC, name
            LIMIT :limit
        """)
        result = await self.db.execute(sql, {"q": " | ".join(words), "limit": limit})
        return [row[0] for row in result.all()]

    async def popular_card_names(
        self,
        format: Optional[str] = None,
        standard_only: bool = True,
        colors: Optional[List[str]] = None,
        limit: int = SHORTLIST_SIZE,
    ) -> List[str]:
        """Nonbasic cards legal in the format and colors, most tournament decklists first."""
        # ponytail: exact name join misses DFCs listed by front face only; add a face match if it matters
        where = " AND ".join(["coalesce(c.type_line, '') NOT LIKE 'Basic Land%'",
                              *self._filter_sql(format, standard_only, colors, alias="c")])
        sql = text(f"""
            SELECT c.name, COUNT(DISTINCT d.id) AS freq
            FROM decklists d
            JOIN events e ON d.event_id = e.id
            CROSS JOIN LATERAL jsonb_array_elements(d.main_deck) AS card_entry
            JOIN cards c ON LOWER(c.name) = LOWER(card_entry->>'card_name')
            WHERE e.format = :format AND {where}
            GROUP BY c.name
            ORDER BY freq DESC, c.name
            LIMIT :limit
        """)
        result = await self.db.execute(sql, {"format": format or "standard", "limit": limit})
        return [row[0] for row in result.all()]

    async def tournament_frequency(
        self,
        card_names: List[str],
        format: str = "standard",
    ) -> Dict[str, int]:
        """
        Look up tournament decklist frequency for a list of card names.

        Returns a dict mapping lowercase card name -> frequency count.
        Cards not found in tournament data are absent.
        """
        if not card_names:
            return {}

        name_params: Dict[str, Any] = {}
        name_placeholders = []
        for i, name in enumerate(card_names):
            name_params[f"n_{i}"] = name.lower()
            name_placeholders.append(f":n_{i}")

        freq_sql = f"""
            SELECT
                LOWER(card_entry->>'card_name') as card_name,
                COUNT(DISTINCT d.id) as freq
            FROM decklists d
            JOIN events e ON d.event_id = e.id,
                 jsonb_array_elements(d.main_deck) as card_entry
            WHERE e.format = :format
              AND LOWER(card_entry->>'card_name') IN ({', '.join(name_placeholders)})
            GROUP BY LOWER(card_entry->>'card_name')
        """
        name_params["format"] = format

        result = await self.db.execute(text(freq_sql), name_params)
        return {row[0]: row[1] for row in result.all()}
```

`backend/app/services/guided_builder.py`: replace the whole body of `_rank_cards_by_tournament_frequency` (keep its signature and docstring) with:
```python
        return await CardService(self.db).tournament_frequency(card_names, format=format)
```
`test_keyword_recommendations.py` mocks `analyzer.db.execute`, and `CardService(self.db)` uses that same db, so those tests keep passing.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_card_shortlist.py tests/test_format_filtering.py tests/test_keyword_recommendations.py -v`
Expected: all PASS. The vector-query tests in `test_format_filtering.py` still see `cards_standard` with no legality filter, and `legalities->>'standard' = 'legal'` on `cards`.

Run: `cd backend && python3 -m pytest`
Expected: 150 passed.

- [ ] **Step 6: Apply the migration locally and check the index is used**

Run (stack from Task 10 Step 1 must be up; this only adds an index and does not recreate any container):
```bash
docker exec spellbook-backend alembic upgrade head
docker exec spellbook-db psql -U spellbook -d spellbook -c "EXPLAIN SELECT name FROM cards WHERE to_tsvector('english', name || ' ' || coalesce(type_line, '') || ' ' || coalesce(oracle_text, '')) @@ to_tsquery('english', 'removal | red')"
```
Expected: `Running upgrade 015 -> 016`, then a plan that mentions `idx_cards_fts` (a `Bitmap Index Scan`).

- [ ] **Step 7: Commit**

```bash
git add backend/alembic/versions/016_add_cards_fts_index.py backend/app/services/card_service.py \
  backend/app/services/guided_builder.py backend/tests/test_card_shortlist.py
git commit -m "Add full-text and popularity shortlist sources to CardService

Migration 016 adds a GIN index on a cards tsvector (name, type line,
oracle text). CardService gains text_search_names (query words OR'd,
english stemming and stopwords), popular_card_names (tournament
frequency in the format and colors, basics excluded) and
tournament_frequency (moved from the guided builder, which now
delegates). The legality and color WHERE fragments are shared with the
vector query.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---
### Task 6: `semantic_search`: shortlist, then Jev re-rank

**Files:**
- Modify: `backend/app/services/card_service.py` (imports; new `SEARCH_CUTOFF` and `score_search`; `semantic_search` at lines 325-369 replaced; new `_shortlist`)
- Modify: `backend/tests/test_format_filtering.py` (`test_full_semantic_search_routes_to_view`, last two lines)
- Test: `backend/tests/test_search_rerank.py`

**Interfaces:**
- Consumes: `jev.session`, `jev.ask_many`, `jev.FIT_DEADLINE` (Task 1); `deck_fit.card_payload`; `_vector_search`, `text_search_names`, `popular_card_names`, `SHORTLIST_SIZE` (Task 5); `get_cards_by_names` (existing).
- Produces:
  - `card_service.SEARCH_CUTOFF = 0.5`
  - `async card_service.score_search(query: str, cards: Sequence[Any], client=None) -> Optional[Dict[str, float]]`: `{}` for no cards; `None` when Jev is unavailable or any request fails
  - `CardService.semantic_search(query, limit=10, standard_only=True, format=None, colors=None) -> List[Card]`: signature unchanged
  - `async CardService._shortlist(query, format, standard_only, colors) -> List[str]`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_search_rerank.py`:
```python
"""semantic_search: shortlist merge (vector, full text, popular), Jev re-rank, fallbacks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services import card_service as cs
from app.services.card_service import CardService
from tests.jev_fake import FakeJev

REAL_SCORE_SEARCH = getattr(cs, "score_search", None)  # the svc fixture replaces cs.score_search


def card(name):
    return SimpleNamespace(name=name, mana_cost="{R}", type_line="Instant", oracle_text="Deal 2.")


@pytest.fixture
def svc(monkeypatch):
    s = CardService.__new__(CardService)
    s.db = MagicMock(rollback=AsyncMock())
    s.vec, s.fts, s.pop, s.calls = [], [], [], []
    s.embedding = None
    emb = SimpleNamespace(get_query_embedding=AsyncMock(side_effect=lambda q: s.embedding))
    monkeypatch.setattr(cs, "get_embedding_service", lambda: emb)

    async def vector(embedding, **kw):
        s.calls.append("vec")
        if isinstance(s.vec, Exception):
            raise s.vec
        return [SimpleNamespace(name=n) for n in s.vec]

    async def fts(query, format=None, standard_only=True, colors=None, limit=60):
        s.calls.append("fts")
        return list(s.fts)

    async def popular(format=None, standard_only=True, colors=None, limit=60):
        s.calls.append("pop")
        return list(s.pop)

    async def by_names(names):
        return {n.lower(): card(n) for n in names}

    s._vector_search, s.text_search_names, s.popular_card_names = vector, fts, popular
    s.get_cards_by_names = by_names
    s.scores = None

    async def score(query, cards, client=None):
        s.scored = [c.name for c in cards]
        return s.scores
    monkeypatch.setattr(cs, "score_search", score)
    return s


def names(cards):
    return [c.name for c in cards]


async def test_shortlist_merges_in_priority_order_and_dedupes(svc):
    svc.embedding = [0.1]
    svc.vec, svc.fts, svc.pop = ["A", "B"], ["B", "C"], ["C", "D"]
    assert names(await svc.semantic_search("burn", limit=10)) == ["A", "B", "C", "D"]


async def test_without_openai_skips_vector_search(svc):
    svc.vec, svc.fts = ["X"], ["A"]
    assert names(await svc.semantic_search("burn")) == ["A"]
    assert "vec" not in svc.calls


async def test_vector_failure_rolls_back_and_continues(svc):
    svc.embedding = [0.1]
    svc.vec, svc.fts = RuntimeError("no pgvector"), ["A"]
    assert names(await svc.semantic_search("burn")) == ["A"]
    svc.db.rollback.assert_awaited_once()


async def test_shortlist_capped_skips_lower_sources(svc, monkeypatch):
    monkeypatch.setattr(cs, "SHORTLIST_SIZE", 3)
    svc.fts, svc.pop = ["A", "B", "C", "D", "E"], ["F"]
    await svc.semantic_search("burn", limit=10)
    assert svc.scored == ["A", "B", "C"]
    assert "pop" not in svc.calls


async def test_rerank_orders_by_probability_and_cuts(svc):
    svc.fts = ["A", "B", "C"]
    svc.scores = {"A": 0.2, "B": 0.9, "C": 0.6}
    assert names(await svc.semantic_search("burn")) == ["B", "C"]


async def test_limit_applies_after_rerank(svc):
    svc.fts = ["A", "B", "C"]
    svc.scores = {"A": 0.6, "B": 0.9, "C": 0.7}
    assert names(await svc.semantic_search("burn", limit=1)) == ["B"]


async def test_jev_unavailable_returns_shortlist_order(svc):
    svc.fts, svc.pop = ["A", "B"], ["C"]
    svc.scores = None
    assert names(await svc.semantic_search("burn", limit=2)) == ["A", "B"]


async def test_empty_shortlist_makes_no_jev_call(svc, monkeypatch):
    client = FakeJev()

    async def score(query, cards):
        return await REAL_SCORE_SEARCH(query, cards, client=client)
    monkeypatch.setattr(cs, "score_search", score)
    assert await svc.semantic_search("?!") == []
    assert client.calls == []


class TestScoreSearch:
    async def test_one_noul_per_card(self):
        client = FakeJev(lambda state, qs: {"match": 0.7 if state["card"]["name"] == "A" else 0.1})
        assert await cs.score_search("burn", [card("A"), card("B")], client=client) == {"A": 0.7, "B": 0.1}
        assert len(client.calls) == 2
        state, qs, _ = client.calls[0]
        assert state["query"] == "burn" and set(qs) == {"match"}

    async def test_any_failure_returns_none(self):
        client = FakeJev(fail=lambda state: state["card"]["name"] == "B")
        assert await cs.score_search("burn", [card("A"), card("B")], client=client) is None

    async def test_no_key_returns_none(self):
        assert await cs.score_search("burn", [card("A")]) is None
```

In `backend/tests/test_format_filtering.py`, `test_full_semantic_search_routes_to_view`, the vector query is now the first of several queries. Replace its last two lines:
```python
        call_args = card_service.db.execute.call_args
        sql_text = str(call_args[0][0])
        assert "cards_commander" in sql_text
```
with:
```python
        sql_text = str(card_service.db.execute.call_args_list[0][0][0])
        assert "cards_commander" in sql_text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_search_rerank.py -v`
Expected: FAIL. The fixture's `monkeypatch.setattr(cs, "score_search", ...)` raises `AttributeError: <module 'app.services.card_service'> has no attribute 'score_search'`.

- [ ] **Step 3: Implement**

`backend/app/services/card_service.py`, add to the imports (after `import re`):
```python
from typing import Sequence

from typesafe_sdk import Noul

from app.services import jev
from app.services.deck_fit import card_payload
```
(`app.services.deck_fit` imports `card_service` only inside a function, so this creates no import cycle.)

After `VALID_COLORS`, add:
```python
SEARCH_CUTOFF = 0.5


async def score_search(query: str, cards: Sequence[Any], client=None) -> Optional[Dict[str, float]]:
    """One Jev Noul per card: does it serve `query`? Keyed by card name.

    {} for no cards. None when Jev is unavailable or any request fails, so the
    caller keeps its shortlist order.
    """
    if not cards:
        return {}
    question = {"match": Noul(instructions="Does `card` serve the request in `query`?")}
    payloads = [card_payload(c) for c in cards]
    async with jev.session(client) as client:
        if client is None:
            return None
        try:
            responses = await jev.ask_many(
                client, [({"query": query, "card": p}, question) for p in payloads], jev.FIT_DEADLINE)
            return {p["name"]: r.nouls["match"].noul for p, r in zip(payloads, responses)}
        except Exception as e:
            logger.warning(f"Search re-rank failed, using shortlist order: {e}")
            return None
```

Replace the whole `semantic_search` method with:
```python
    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        standard_only: bool = True,
        format: Optional[str] = None,
        colors: Optional[List[str]] = None,
    ) -> List[Card]:
        """
        Shortlist up to SHORTLIST_SIZE cards (vector hits when OpenAI is configured,
        then full-text hits, then popular cards), then have Jev re-rank them against
        the query and drop those below SEARCH_CUTOFF. Without Jev, returns the
        shortlist in merge order.
        """
        names = await self._shortlist(query, format, standard_only, colors)
        by_name = await self.get_cards_by_names(names)
        cards = [by_name[n.lower()] for n in names if n.lower() in by_name]
        scores = await score_search(query, cards)
        if scores is None:
            return cards[:limit]
        kept = [c for c in cards if scores.get(c.name, 0.0) >= SEARCH_CUTOFF]
        return sorted(kept, key=lambda c: scores[c.name], reverse=True)[:limit]

    async def _shortlist(
        self,
        query: str,
        format: Optional[str],
        standard_only: bool,
        colors: Optional[List[str]],
    ) -> List[str]:
        """Unique names, already filtered by format and colors, in priority order:
        vector hits, full-text hits, popular cards. Stops at SHORTLIST_SIZE."""
        names: Dict[str, None] = {}  # insertion-ordered set

        def add(found) -> None:
            for n in found:
                if len(names) >= SHORTLIST_SIZE:
                    return
                names.setdefault(n, None)

        embedding = await get_embedding_service().get_query_embedding(query)
        if embedding is not None:
            try:
                rows = await self._vector_search(
                    embedding, format=format, standard_only=standard_only,
                    colors=colors, limit=SHORTLIST_SIZE,
                )
                add(row.name for row in rows)
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                await self.db.rollback()  # the aborted transaction would fail the next query

        sources = (
            lambda: self.text_search_names(query, format, standard_only, colors),
            lambda: self.popular_card_names(format, standard_only, colors),
        )
        for source in sources:
            if len(names) >= SHORTLIST_SIZE:
                break
            try:
                add(await source())
            except Exception as e:
                logger.error(f"Shortlist query failed: {e}")
                await self.db.rollback()
        return list(names)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_search_rerank.py tests/test_format_filtering.py -v`
Expected: all PASS.

Run: `cd backend && python3 -m pytest`
Expected: 161 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/card_service.py backend/tests/test_search_rerank.py \
  backend/tests/test_format_filtering.py
git commit -m "Re-rank semantic search shortlists with Jev

semantic_search keeps its signature and callers. It now builds a
shortlist of up to 60 format- and color-filtered names (vector hits
when OpenAI is configured, then full-text hits, then popular cards),
asks Jev whether each card serves the query, and returns those at or
above 0.5, best first. Without Jev it returns the shortlist in merge
order, so search no longer comes back empty without OpenAI embeddings.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---
### Task 7: Deck-request parsing with Jev

**Files:**
- Modify: `backend/app/services/ai/deck_parsing.py` (imports at lines 1-12; `parse_deck_request` at lines 15-71)
- Test: `backend/tests/test_deck_parsing.py`

**Interfaces:**
- Consumes: `jev.session`, `jev.ask`, `jev.FIT_DEADLINE` (Task 1); `FakeJev` (Task 4); existing `extract_card_names_from_prompt(prompt, db, format="standard") -> List[str]` and `fallback_parse(prompt, db)`.
- Produces:
  - `async parse_deck_request(prompt: str, db, client=None) -> Dict[str, Any]` with keys `archetype`, `colors`, `colors_specified`, `strategy`, `specific_cards` (the fallback also adds `budget` and `focus`, as today)
  - `deck_parsing.ARCHETYPES: Dict[str, str]`, `COLOR_NAMES: Dict[str, str]`, `WANT_THRESHOLD = 0.5`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_deck_parsing.py`:
```python
"""parse_deck_request with Jev: shape, own-vs-opponent colors, wanted cards, fallback."""

import pytest

from app.services.ai import deck_parsing
from tests.jev_fake import FakeJev

SHEOLDRED = "Sheoldred, the Apocalypse"


@pytest.fixture
def names(monkeypatch):
    found = []

    async def extract(prompt, db, format="standard"):
        return list(found)
    monkeypatch.setattr(deck_parsing, "extract_card_names_from_prompt", extract)

    async def fallback(prompt, db):
        return {"fallback": True}
    monkeypatch.setattr(deck_parsing, "fallback_parse", fallback)
    return found


async def test_shape_colors_and_wanted_cards(names):
    names.extend(["Monastery Swiftspear", SHEOLDRED])
    client = FakeJev(lambda state, qs: {"archetype": "aggro", "color:R": 0.9, "colors_specified": 0.9,
                                        "wants:0": 0.9, "wants:1": 0.1})
    prompt = f"Mono-red aggro with Monastery Swiftspear that beats {SHEOLDRED}"
    out = await deck_parsing.parse_deck_request(prompt, db=None, client=client)
    assert out == {"archetype": "aggro", "colors": ["R"], "colors_specified": True,
                   "strategy": prompt, "specific_cards": ["Monastery Swiftspear"]}
    state, qs, _ = client.calls[0]
    assert state == {"request": prompt, "cards": ["Monastery Swiftspear", SHEOLDRED]}
    assert set(qs["archetype"].criteria) == {"aggro", "control", "midrange", "combo", "tempo"}


async def test_opponent_color_is_not_specified(names):
    client = FakeJev(lambda state, qs: {"archetype": "control", "color:R": 0.1, "colors_specified": 0.2})
    out = await deck_parsing.parse_deck_request("beat mono-red", db=None, client=client)
    assert out["colors"] == [] and out["colors_specified"] is False


async def test_unstated_colors_are_dropped(names):
    client = FakeJev(lambda state, qs: {"color:W": 0.8, "color:U": 0.7, "colors_specified": 0.1})
    out = await deck_parsing.parse_deck_request("A deck that beats mono-red", db=None, client=client)
    assert out["colors"] == [] and out["colors_specified"] is False


async def test_colors_specified_needs_a_color(names):
    client = FakeJev(lambda state, qs: {"colors_specified": 0.9})
    out = await deck_parsing.parse_deck_request("aggro deck", db=None, client=client)
    assert out["colors"] == [] and out["colors_specified"] is False


async def test_failure_falls_back(names):
    client = FakeJev(fail=lambda state: True)
    assert await deck_parsing.parse_deck_request("x", db=None, client=client) == {"fallback": True}


async def test_partial_answer_falls_back(names):
    client = FakeJev(lambda state, qs: {"archetype": "aggro"}, fill=False)
    assert await deck_parsing.parse_deck_request("x", db=None, client=client) == {"fallback": True}


async def test_no_key_falls_back_without_lookup(names, monkeypatch):
    async def boom(prompt, db, format="standard"):
        raise AssertionError("card lookup should not run without Jev")
    monkeypatch.setattr(deck_parsing, "extract_card_names_from_prompt", boom)
    assert await deck_parsing.parse_deck_request("x", db=None) == {"fallback": True}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_parsing.py -v`
Expected: FAIL with `TypeError: parse_deck_request() got an unexpected keyword argument 'client'`, and `test_no_key_falls_back_without_lookup` returns the fallback via the LLM-not-configured branch (it may pass; the others must fail).

- [ ] **Step 3: Implement**

In `backend/app/services/ai/deck_parsing.py`, replace the import block:
```python
import json
import logging
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.services import llm
```
with:
```python
import asyncio
import logging
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typesafe_sdk import Choice, Noul

from app.services import jev
```

Replace the whole `parse_deck_request` function with:
```python
ARCHETYPES = {
    "aggro": "Wins fast with cheap creatures and direct damage",
    "control": "Answers threats with removal, counterspells and card advantage, and wins late",
    "midrange": "Efficient threats plus interaction; trades resources and wins with sturdy, value-generating cards",
    "combo": "Assembles specific cards that win together",
    "tempo": "Cheap threats backed by cheap disruption to stay ahead",
}
COLOR_NAMES = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}
WANT_THRESHOLD = 0.5


def _parse_questions(n_cards: int) -> Dict[str, Any]:
    questions: Dict[str, Any] = {
        "archetype": Choice(
            instructions="Which archetype does `request` want for the user's own deck?",
            criteria=ARCHETYPES,
        ),
        "colors_specified": Noul(
            instructions="Did the user state colors for their own deck, not an opponent's?"),
    }
    for code, name in COLOR_NAMES.items():
        questions[f"color:{code}"] = Noul(instructions={
            "question": "Does the user want this color in their own deck (not an opponent's)?",
            "color": name,
        })
    for i in range(n_cards):
        questions[f"wants:{i}"] = Noul(
            instructions=f"Does the user want `cards[{i}]` in their deck, rather than only mentioning it?")
    return questions


async def parse_deck_request(prompt: str, db: AsyncSession, client=None) -> Dict[str, Any]:
    """
    Parse a natural-language deck request with one Jev request into
    {archetype, colors, colors_specified, strategy, specific_cards}.

    Card names are resolved from the database first (word n-grams of the prompt);
    Jev decides which of them the user wants in their deck and which colors are
    theirs rather than an opponent's. Falls back to keyword parsing when Jev is
    not configured, fails, or answers incompletely.
    """
    async with jev.session(client) as client:
        if client is not None:
            names = await extract_card_names_from_prompt(prompt, db)
            try:
                async with asyncio.timeout(jev.FIT_DEADLINE):
                    r = await jev.ask(client, {"request": prompt, "cards": names},
                                      _parse_questions(len(names)))
                colors = [c for c in COLOR_NAMES if r.nouls[f"color:{c}"].noul >= WANT_THRESHOLD]
                if r.nouls["colors_specified"].noul < WANT_THRESHOLD:
                    colors = []  # e.g. "beat mono-red": Jev leans toward non-red colors, but none were asked for
                return {
                    "archetype": r.choices["archetype"].choice,
                    "colors": colors,
                    "colors_specified": bool(colors),
                    "strategy": prompt,
                    "specific_cards": [n for i, n in enumerate(names)
                                       if r.nouls[f"wants:{i}"].noul >= WANT_THRESHOLD],
                }
            except Exception as e:
                logger.warning(f"Jev deck-request parse failed, using fallback: {e}")
    return await fallback_parse(prompt, db)
```

`fallback_parse`, `extract_card_names_from_prompt` and `get_commander_color_identity` are unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_deck_parsing.py -v`
Expected: 7 PASS.

Run: `cd backend && python3 -m pytest`
Expected: 168 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ai/deck_parsing.py backend/tests/test_deck_parsing.py
git commit -m "Parse deck requests with Jev instead of an LLM JSON prompt

One Jev request picks the archetype (Choice), the user's own colors
(a Noul per color, so 'beat mono-red' does not make the deck red),
whether colors were stated, and which database-resolved card names the
user wants in the deck. The return shape is unchanged; fallback_parse
still handles a missing key, a failure or an incomplete answer.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---
### Task 8: Chat routing with Jev

**Files:**
- Create: `backend/app/services/chat_routing.py`
- Modify: `backend/app/services/chat_service.py` (imports at lines 16-21; `suggest_core` `roles` description at line 59; `process_message` lines 409-415; new `_route_with_jev` and `_meta_archetypes` methods)
- Test: `backend/tests/test_chat_routing.py`, `backend/tests/test_chat_jev_dispatch.py`

**Interfaces:**
- Consumes: `jev.session`, `jev.ask`, `jev.FIT_DEADLINE` (Task 1); `deck_fit.role_description` (Task 3); `guided_builder._extract_mtg_keywords` (existing); `FakeJev` (Task 4); `ChatService._dispatch_tool(tool_name, tool_input, conversation, user_id, ai_text="")` (existing).
- Produces (in `app.services.chat_routing`):
  - constants `ROUTE_CONFIDENCE = 0.6`, `WANT_THRESHOLD = 0.5`, `REPLY = "reply"`, `CORE_ROLES: List[str]`, `DEFAULT_ROLES`, `DEFAULT_COUNT = 6`, `MAX_COUNT = 20`, `TURNS = 6`, `TURN_CHARS = 500`, `STRATEGY_CHARS = 300`
  - `@dataclass Route(action: str, confidence: float, probabilities: Dict[str, float], inputs: Dict[str, Dict[str, Any]])` with `top_tool() -> str` (the highest-probability tool, never `reply`)
  - `deck_summary(deck: Optional[Dict], fallback_colors: Sequence[str]) -> Dict` returning `{has_deck, unique_nonland_cards, colors}`
  - `build_questions(tools, card_names, mechanics, meta_archetypes) -> Dict[str, Question]`
  - `tool_inputs(resp, *, message, summary, strategy, card_names, tools) -> Dict[str, Dict[str, Any]]` (one input dict per tool name, holding only that tool's schema properties)
  - `async route(*, message, history, summary, strategy, card_names, meta_archetypes, tools, client=None) -> Optional[Route]`, where `history` is `conversation.messages` ending with the current message
- Produces (in `ChatService`): `async _meta_archetypes(format) -> List[str]` and `async _route_with_jev(message, conversation, deck, resolved_cards, format, require_tool, system_prompt, api_messages) -> Tuple[str, List[Tuple[str, Dict]]]`. A result of `("", [])` means "use the LLM tool call".

- [ ] **Step 1: Write the failing routing tests**

`backend/tests/test_chat_routing.py`:
```python
"""chat_routing: Jev questions, tool-argument assembly, and failure fallbacks."""

from app.services import chat_routing as cr
from app.services.chat_service import TOOLS
from tests.jev_fake import FakeJev

SHEOLDRED = "Sheoldred, the Apocalypse"
NO_DECK = {"has_deck": False, "unique_nonland_cards": 0, "colors": []}
BLACK_DECK = {"has_deck": True, "unique_nonland_cards": 12, "colors": ["B"]}


async def route(message, answers, summary=NO_DECK, strategy="", cards=(), meta=(), fill=True,
                history=None):
    client = FakeJev(lambda state, qs: answers, fill=fill)
    r = await cr.route(message=message, history=history or [{"role": "user", "content": message}],
                       summary=summary, strategy=strategy, card_names=list(cards),
                       meta_archetypes=list(meta), tools=TOOLS, client=client)
    return r, client


def test_deck_summary_colors_from_pips_else_fallback():
    deck = {"main_deck": [
        {"card_name": "Shock", "card": {"mana_cost": "{R}", "type_line": "Instant"}},
        {"card_name": "Mountain", "card": {"mana_cost": "", "type_line": "Basic Land — Mountain"}},
        {"card_name": "Murder", "card": {"mana_cost": "{1}{B}{B}", "type_line": "Instant"}},
    ]}
    assert cr.deck_summary(deck, ["G"]) == {"has_deck": True, "unique_nonland_cards": 2, "colors": ["B", "R"]}
    assert cr.deck_summary(None, ["G"]) == {"has_deck": False, "unique_nonland_cards": 0, "colors": ["G"]}


def test_choice_options_dedupe():
    qs = cr.build_questions(TOOLS, [SHEOLDRED], ["protection", "surveil"],
                            ["Izzet Prowess", "izzet prowess", "Mono-Red Aggro", ""])
    assert list(qs["opponent"].criteria) == ["Izzet Prowess", "Mono-Red Aggro", "none"]
    pkg = qs["package_role"].criteria
    assert list(pkg).count("protection") == 1
    assert pkg["surveil"] == "Cards with the surveil mechanic"
    assert set(qs["action"].criteria) == {t["name"] for t in TOOLS} | {cr.REPLY}
    assert "wants:0" in qs and "wants:1" not in qs
    assert all(f"role:{r}" in qs for r in cr.CORE_ROLES)


def test_opponent_choice_always_has_none():
    qs = cr.build_questions(TOOLS, [], [], [])
    assert list(qs["opponent"].criteria) == ["none"]


async def test_action_confidence_and_probabilities():
    answers = {"action": {"choice": "suggest_core", "confidence": 0.8,
                          "probabilities": {"suggest_core": 0.9, "reply": 0.1}}}
    r, _ = await route("Build me a mono-red aggro deck", answers)
    assert (r.action, r.confidence) == ("suggest_core", 0.8)
    assert r.probabilities == {"suggest_core": 0.9, "reply": 0.1}
    assert r.top_tool() == "suggest_core"


async def test_top_tool_never_returns_reply():
    r = cr.Route("reply", 0.9, {"reply": 0.7, "analyze_meta": 0.2, "suggest_core": 0.1},
                 {t["name"]: {} for t in TOOLS})
    assert r.top_tool() == "analyze_meta"


async def test_colors_from_nouls_else_deck_colors_and_opponent():
    r, _ = await route("beat mono-red", {"color:R": 0.1, "opponent": "Mono-Red Aggro"},
                       summary=BLACK_DECK, meta=["Mono-Red Aggro"])
    assert r.inputs["suggest_core"]["colors"] == ["B"]
    assert r.inputs["get_matchup_info"] == {"opponent_deck": "Mono-Red Aggro"}
    r, _ = await route("red aggro", {"color:R": 0.9}, summary=BLACK_DECK)
    assert r.inputs["suggest_core"]["colors"] == ["R"]


async def test_roles_package_role_and_count():
    r, _ = await route("give me 8 removal spells", {"role:removal": 0.9, "package_role": "removal"})
    assert r.inputs["suggest_core"]["roles"] == ["removal"]
    assert r.inputs["suggest_package"]["role"] == "removal"
    assert r.inputs["suggest_package"]["count"] == 8
    r, _ = await route("more stuff", {})
    assert r.inputs["suggest_core"]["roles"] == cr.DEFAULT_ROLES
    assert r.inputs["suggest_package"]["count"] == cr.DEFAULT_COUNT
    r, _ = await route("give me 100 cards", {})
    assert r.inputs["suggest_package"]["count"] == cr.MAX_COUNT


async def test_unwanted_card_is_not_built_around():
    msg = f"Beat {SHEOLDRED} with Monastery Swiftspear aggro"
    r, client = await route(msg, {"wants:0": 0.1, "wants:1": 0.9},
                            cards=[SHEOLDRED, "Monastery Swiftspear"])
    assert r.inputs["generate_full_deck"]["specific_cards"] == ["Monastery Swiftspear"]
    assert "Sheoldred" not in r.inputs["suggest_core"]["strategy"]
    assert "Apocalypse" not in r.inputs["suggest_core"]["strategy"]
    assert "Monastery Swiftspear" in r.inputs["suggest_core"]["strategy"]
    assert r.inputs["modify_deck"] == {"modification": msg}
    assert client.calls[0][0]["cards"] == [SHEOLDRED, "Monastery Swiftspear"]


async def test_strategy_joins_persisted_and_is_capped():
    r, _ = await route("more burn", {}, strategy="Mono-red aggro")
    assert r.inputs["suggest_core"]["strategy"] == "Mono-red aggro; more burn"
    r, _ = await route("more burn", {}, strategy="x" * 400)
    assert len(r.inputs["suggest_core"]["strategy"]) == cr.STRATEGY_CHARS


async def test_archetype_none_is_omitted():
    r, _ = await route("just build it", {"archetype": "none"})
    assert "archetype" not in r.inputs["generate_full_deck"]
    assert r.inputs["analyze_meta"] == {"focus": ""}
    r, _ = await route("just build it", {"archetype": "tempo"})
    assert r.inputs["generate_full_deck"]["archetype"] == "tempo"


async def test_state_carries_last_six_turns_before_this_message():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}" + "x" * 600}
               for i in range(9)] + [{"role": "user", "content": "now"}]
    _, client = await route("now", {}, history=history, strategy="s", meta=["A"])
    state = client.calls[0][0]
    assert [t["content"][:2] for t in state["recent_turns"]] == ["m3", "m4", "m5", "m6", "m7", "m8"]
    assert all(len(t["content"]) == cr.TURN_CHARS for t in state["recent_turns"])
    assert state["message"] == "now" and state["strategy"] == "s" and state["meta_archetypes"] == ["A"]


async def test_mechanics_from_message_become_package_options():
    _, client = await route("cards with surveil", {})
    state, qs, _ = client.calls[0]
    assert state["mechanics"] == ["surveil"] and "surveil" in qs["package_role"].criteria


async def test_partial_answer_falls_back():
    r, _ = await route("hi", {"action": "reply"}, fill=False)
    assert r is None


async def test_failure_and_missing_key_fall_back():
    client = FakeJev(fail=lambda state: True)
    assert await cr.route(message="hi", history=[], summary=NO_DECK, strategy="", card_names=[],
                          meta_archetypes=[], tools=TOOLS, client=client) is None
    assert await cr.route(message="hi", history=[], summary=NO_DECK, strategy="", card_names=[],
                          meta_archetypes=[], tools=TOOLS) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_chat_routing.py -v`
Expected: collection error, `ImportError: cannot import name 'chat_routing' from 'app.services'`

- [ ] **Step 3: Write `chat_routing.py`**

`backend/app/services/chat_routing.py`:
```python
"""
Chat routing with Jev: one System One request per chat turn picks the tool to
run (or a plain reply) and the judgments code needs to build its arguments.

Jev judges; code assembles tool inputs and decides whether the judgment is
confident enough to skip the LLM tool call.
Design: docs/superpowers/specs/2026-10-05-jev-expansion-design.md (section 5)
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from typesafe_sdk import Choice, Noul

from app.services import jev
from app.services.deck_fit import role_description
from app.services.guided_builder import _extract_mtg_keywords

logger = logging.getLogger(__name__)

ROUTE_CONFIDENCE = 0.6
WANT_THRESHOLD = 0.5
REPLY = "reply"
COLORS = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}
# The suggest_core role list (the tool description is built from this list)
CORE_ROLES = [
    "threats", "creatures", "removal", "card advantage", "card draw", "counterspells",
    "protection", "ramp", "burn", "recursion", "finishers", "interaction", "discard",
    "lifegain", "graveyard hate", "tutors", "sacrifice outlets", "board wipes",
    "spot removal", "cheap threats", "big threats",
]
DEFAULT_ROLES = ["threats", "removal", "card advantage"]
DEFAULT_COUNT = 6
MAX_COUNT = 20
TURNS = 6
TURN_CHARS = 500
STRATEGY_CHARS = 300
ARCHETYPES = {
    "aggro": "Wins fast with cheap creatures and direct damage",
    "control": "Answers threats with removal, counterspells and card advantage, and wins late",
    "midrange": "Efficient threats plus interaction; wins with sturdy, value-generating cards",
    "combo": "Assembles specific cards that win together",
    "tempo": "Cheap threats backed by cheap disruption to stay ahead",
    "none": "No archetype is stated or implied",
}
REPLY_DESCRIPTION = (
    "Answer in text only: a rules question, a card explanation, thanks, or advice that "
    "needs no card suggestions, deck changes, generated deck, or meta data"
)
PUNCTUATION = ",.!?\"'"


@dataclass
class Route:
    action: str
    confidence: float
    probabilities: Dict[str, float]
    inputs: Dict[str, Dict[str, Any]]  # tool name -> tool input

    def top_tool(self) -> str:
        """The most probable tool, for when a tool call is required but `reply` won."""
        return max(self.inputs, key=lambda name: self.probabilities.get(name, 0.0))


def deck_summary(deck: Optional[Dict[str, Any]], fallback_colors: Sequence[str]) -> Dict[str, Any]:
    """has_deck, unique nonland count, and colors from mana costs (else `fallback_colors`)."""
    entries = (deck or {}).get("main_deck") or []
    nonland = {e.get("card_name") for e in entries
               if "land" not in ((e.get("card") or {}).get("type_line") or "").lower()}
    costs = "".join((e.get("card") or {}).get("mana_cost") or "" for e in entries)
    colors = [c for c in COLORS if f"{{{c}}}" in costs] or list(fallback_colors)
    return {"has_deck": bool(entries), "unique_nonland_cards": len(nonland - {None}), "colors": colors}


def _options(names: Sequence[str], describe: Callable[[str], Any]) -> Dict[str, Any]:
    """Choice criteria; a repeated name (any case) keeps its first description."""
    out: Dict[str, Any] = {}
    seen = set()
    for name in names:
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out[name] = describe(name)
    return out


def build_questions(
    tools: List[Dict[str, Any]],
    card_names: Sequence[str],
    mechanics: Sequence[str],
    meta_archetypes: Sequence[str],
) -> Dict[str, Any]:
    questions: Dict[str, Any] = {
        "action": Choice(
            instructions="What should the deck-building assistant do with `message`, given `recent_turns`, `deck` and `strategy`?",
            criteria={**{t["name"]: t["description"] for t in tools}, REPLY: REPLY_DESCRIPTION},
        ),
        "package_role": Choice(
            instructions="Which role or mechanic does the user want more cards for?",
            criteria=_options([*CORE_ROLES, *mechanics], lambda n: (
                role_description(n) if n in CORE_ROLES else f"Cards with the {n} mechanic")),
        ),
        "archetype": Choice(
            instructions="Which archetype does the user want for their own deck?",
            criteria=ARCHETYPES,
        ),
        "opponent": Choice(
            instructions="Which opponent deck from `meta_archetypes` is the user asking about?",
            criteria={**_options(meta_archetypes, lambda n: None), "none": "No opponent deck is mentioned"},
        ),
    }
    for code, name in COLORS.items():
        questions[f"color:{code}"] = Noul(instructions={
            "question": "Does the user want this color in their own deck (not an opponent's)?",
            "color": name,
        })
    for role in CORE_ROLES:
        questions[f"role:{role}"] = Noul(instructions={
            "question": "Does the user want cards that fill this role now?",
            "role": role_description(role),
        })
    for i in range(len(card_names)):
        questions[f"wants:{i}"] = Noul(
            instructions=f"Does the user want `cards[{i}]` in their deck, rather than only mentioning it?")
    return questions


def _strip_names(text: str, names: Sequence[str]) -> str:
    """Remove capitalized words of `names` from `text`, so a card the user only
    mentioned (an opponent's) can't be re-resolved from the strategy downstream."""
    tokens = {w.strip(PUNCTUATION).lower() for n in names for w in n.split() if w[:1].isupper()}
    return " ".join(w for w in text.split()
                    if not (w[:1].isupper() and w.strip(PUNCTUATION).lower() in tokens))


def tool_inputs(
    resp: Any,
    *,
    message: str,
    summary: Dict[str, Any],
    strategy: str,
    card_names: Sequence[str],
    tools: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Each tool's input, from Jev's judgments plus the message. Raises KeyError
    when Jev left a question unanswered."""
    nouls, choices = resp.nouls, resp.choices
    colors = [c for c in COLORS if nouls[f"color:{c}"].noul >= WANT_THRESHOLD] or summary["colors"]
    roles = [r for r in CORE_ROLES if nouls[f"role:{r}"].noul >= WANT_THRESHOLD] or list(DEFAULT_ROLES)
    wanted = [n for i, n in enumerate(card_names) if nouls[f"wants:{i}"].noul >= WANT_THRESHOLD]
    text = _strip_names(message, [n for n in card_names if n not in wanted])
    found = re.search(r"\d+", message)
    archetype = choices["archetype"].choice
    opponent = choices["opponent"].choice
    args = {
        "focus": "" if archetype == "none" else archetype,
        "strategy": (f"{strategy}; {text}" if strategy else text)[:STRATEGY_CHARS],
        "colors": colors,
        "roles": roles,
        "role": choices["package_role"].choice,
        "count": min(max(int(found.group()), 1), MAX_COUNT) if found else DEFAULT_COUNT,
        "modification": message,
        "opponent_deck": "" if opponent == "none" else opponent,
        "archetype": archetype,
        "specific_cards": wanted,
    }
    inputs = {}
    for tool in tools:
        props = tool["input_schema"].get("properties", {})
        inputs[tool["name"]] = {k: v for k, v in args.items()
                                if k in props and not (k == "archetype" and v == "none")}
    return inputs


async def route(
    *,
    message: str,
    history: List[Dict[str, Any]],
    summary: Dict[str, Any],
    strategy: str,
    card_names: List[str],
    meta_archetypes: List[str],
    tools: List[Dict[str, Any]],
    client=None,
) -> Optional[Route]:
    """One Jev request deciding this turn's action and tool arguments.

    `history` is the conversation's messages, ending with the current one.
    Returns None when Jev is not configured, fails, or answers incompletely;
    the caller then uses the LLM tool call.
    """
    mechanics = _extract_mtg_keywords(message)
    state = {
        "message": message,
        "recent_turns": [{"role": m.get("role", ""), "content": (m.get("content") or "")[:TURN_CHARS]}
                         for m in history[-(TURNS + 1):-1]],
        "deck": summary,
        "strategy": strategy,
        "cards": card_names,
        "meta_archetypes": meta_archetypes,
        "mechanics": mechanics,
    }
    questions = build_questions(tools, card_names, mechanics, meta_archetypes)
    async with jev.session(client) as client:
        if client is None:
            return None
        try:
            async with asyncio.timeout(jev.FIT_DEADLINE):
                resp = await jev.ask(client, state, questions)
            action = resp.choices["action"]
            return Route(
                action=action.choice,
                confidence=action.confidence,
                probabilities=dict(action.probabilities),
                inputs=tool_inputs(resp, message=message, summary=summary, strategy=strategy,
                                   card_names=card_names, tools=tools),
            )
        except Exception as e:  # Jev must never block chat
            logger.warning(f"Jev routing failed, using the LLM tool call: {e}")
            return None
```

- [ ] **Step 4: Wire it into `ChatService` (description only, so `TOOLS` imports cleanly)**

In `backend/app/services/chat_service.py`, add to the imports (after `from app.services import llm`):
```python
from app.services import chat_routing
```

In `TOOLS`, `suggest_core`, replace the `roles` description string with one built from the shared list (the resulting text is identical):
```python
                    "description": "Role groups to suggest cards for. MUST use from: " + ", ".join(chat_routing.CORE_ROLES)
```

- [ ] **Step 5: Run the routing tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_chat_routing.py -v`
Expected: 14 PASS.

- [ ] **Step 6: Write the failing dispatch tests**

`backend/tests/test_chat_jev_dispatch.py`:
```python
"""process_message: confident Jev routes dispatch without an LLM tool call; otherwise the LLM decides."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.conversation import Conversation
from app.schemas.conversation import ChatResponse
from app.services import chat_routing, llm
from app.services.chat_service import TOOLS, ChatService

SHEOLDRED = {"name": "Sheoldred, the Apocalypse", "mana_cost": "{2}{B}{B}",
             "type_line": "Legendary Creature", "oracle_text": "Deathtouch", "colors": ["B"]}


def routed(action, confidence=0.9, probabilities=None):
    return chat_routing.Route(action, confidence, probabilities or {action: confidence},
                              {t["name"]: {"from": t["name"]} for t in TOOLS})


@pytest.fixture
def chat(monkeypatch):
    svc = ChatService.__new__(ChatService)
    svc.db = MagicMock(flush=AsyncMock(), commit=AsyncMock())
    conv = Conversation(messages=[], context={"strategy": "Mono-black midrange", "colors": ["B"]})
    conv.id = uuid4()
    svc._get_or_create_conversation = AsyncMock(return_value=conv)
    svc._meta_archetypes = AsyncMock(return_value=["Mono-Red Aggro"])
    svc.resolved = []

    async def resolve(message, format, check_legality=True):
        return svc.resolved if check_legality else []
    svc._resolve_card_mentions = resolve

    svc.dispatched = []

    async def dispatch(name, tool_input, conversation, user_id, ai_text=""):
        svc.dispatched.append((name, tool_input))
        return ChatResponse(response=f"ran {name}", conversation_id=conversation.id)
    svc._dispatch_tool = dispatch

    svc.llm_calls = []

    def chat_with_tools(**kwargs):
        svc.llm_calls.append("tools")
        return "llm text", []

    def complete(system, user, max_tokens=4096):
        svc.llm_calls.append("complete")
        svc.complete_args = (system, user)
        return "plain answer"
    monkeypatch.setattr(llm, "is_configured", lambda: True)
    monkeypatch.setattr(llm, "chat_with_tools", chat_with_tools)
    monkeypatch.setattr(llm, "complete", complete)

    svc.route = None

    async def route(**kwargs):
        svc.route_kwargs = kwargs
        return svc.route
    monkeypatch.setattr(chat_routing, "route", route)
    return svc


async def test_confident_tool_dispatches_without_llm(chat):
    chat.route = routed("suggest_core")
    resp = await chat.process_message("Build me a mono-black deck")
    assert resp.response == "ran suggest_core"
    assert chat.dispatched == [("suggest_core", {"from": "suggest_core"})]
    assert chat.llm_calls == []


async def test_route_state_comes_from_conversation(chat):
    chat.resolved = [SHEOLDRED]
    chat.route = routed("suggest_core")
    await chat.process_message("Build around Sheoldred")
    kw = chat.route_kwargs
    assert kw["card_names"] == ["Sheoldred, the Apocalypse"]
    assert kw["strategy"] == "Mono-black midrange"
    assert kw["summary"] == {"has_deck": False, "unique_nonland_cards": 0, "colors": ["B"]}
    assert kw["meta_archetypes"] == ["Mono-Red Aggro"]
    assert kw["history"][-1]["content"] == "Build around Sheoldred"
    assert kw["tools"] is TOOLS


async def test_forced_tool_overrides_reply(chat):
    chat.resolved = [SHEOLDRED]
    chat.route = routed("reply", 0.8, {"reply": 0.7, "get_matchup_info": 0.2, "suggest_core": 0.1})
    resp = await chat.process_message("Tell me about Sheoldred")
    assert chat.dispatched == [("get_matchup_info", {"from": "get_matchup_info"})]
    assert resp.response == "ran get_matchup_info"
    assert chat.llm_calls == []


async def test_confident_reply_uses_text_completion(chat):
    chat.route = routed("reply")
    resp = await chat.process_message("What does trample do?")
    assert resp.response == "plain answer"
    assert chat.llm_calls == ["complete"]
    system, user = chat.complete_args
    assert "no tools" in system
    assert user.endswith("user: What does trample do?")


async def test_low_confidence_uses_llm_tool_call(chat):
    chat.route = routed("suggest_core", confidence=chat_routing.ROUTE_CONFIDENCE - 0.01)
    resp = await chat.process_message("hmm, red maybe?")
    assert chat.llm_calls == ["tools"] and chat.dispatched == []
    assert resp.response == "llm text"


async def test_route_none_uses_llm_tool_call(chat):
    chat.route = None
    await chat.process_message("hello")
    assert chat.llm_calls == ["tools"]


async def test_empty_text_reply_falls_through_to_llm_tool_call(chat, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda system, user, max_tokens=4096: "")
    chat.route = routed("reply")
    await chat.process_message("What does trample do?")
    assert chat.llm_calls == ["tools"]
```

- [ ] **Step 7: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_chat_jev_dispatch.py -v`
Expected: the confident-route tests FAIL. `chat_with_tools` is called (`llm_calls == ['tools']`), and `route_kwargs` is never set (`AttributeError`). `test_low_confidence_uses_llm_tool_call` and `test_route_none_uses_llm_tool_call` may already pass.

- [ ] **Step 8: Route before the LLM call in `process_message`**

In `backend/app/services/chat_service.py`, add after `logger = logging.getLogger(__name__)`:
```python
TEXT_ONLY_REPLY = "\n\nFor this turn, reply in text only; no tools are available."
```

In `process_message`, replace:
```python
            response_text, tool_calls = llm.chat_with_tools(
                system=system_prompt,
                messages=api_messages,
                tools=TOOLS,
                require_tool=require_tool,
                max_tokens=2048,
            )
```
with:
```python
            response_text, tool_calls = await self._route_with_jev(
                message, conversation, deck, resolved_cards, format,
                require_tool, system_prompt, api_messages,
            )
            if not response_text and not tool_calls:
                logger.info("[ROUTE] LLM tool call (Jev unavailable, failed, or not confident)")
                response_text, tool_calls = llm.chat_with_tools(
                    system=system_prompt,
                    messages=api_messages,
                    tools=TOOLS,
                    require_tool=require_tool,
                    max_tokens=2048,
                )
```

Add these methods to `ChatService`, directly after `process_message`:
```python
    async def _route_with_jev(
        self,
        message: str,
        conversation: Conversation,
        deck: Optional[Dict[str, Any]],
        resolved_cards: List[Dict[str, Any]],
        format: str,
        require_tool: bool,
        system_prompt: str,
        api_messages: List[Dict[str, str]],
    ) -> Tuple[str, List[Tuple[str, Dict[str, Any]]]]:
        """(reply text, tool calls) decided by Jev, or ("", []) to use the LLM tool call."""
        ctx = conversation.get_context()
        routed = await chat_routing.route(
            message=message,
            history=conversation.messages or [],
            summary=chat_routing.deck_summary(deck, ctx.get("colors") or []),
            strategy=ctx.get("strategy") or "",
            card_names=[c["name"] for c in resolved_cards],
            meta_archetypes=await self._meta_archetypes(format),
            tools=TOOLS,
        )
        if routed is None or routed.confidence < chat_routing.ROUTE_CONFIDENCE:
            return "", []
        action = routed.action
        if action == chat_routing.REPLY and require_tool:
            action = routed.top_tool()
        logger.info(f"[ROUTE] Jev action={action} confidence={routed.confidence:.2f}")
        if action == chat_routing.REPLY:
            transcript = "\n\n".join(f"{m['role']}: {m['content']}" for m in api_messages)
            return llm.complete(system=system_prompt + TEXT_ONLY_REPLY, user=transcript), []
        if action not in routed.inputs:
            return "", []
        return "", [(action, routed.inputs[action])]

    async def _meta_archetypes(self, format: str) -> List[str]:
        """Up to 10 unique current meta archetype names, most-played first."""
        from app.models.meta import MetaSnapshot

        result = await self.db.execute(
            select(MetaSnapshot.archetype)
            .where(MetaSnapshot.format == format)
            .order_by(MetaSnapshot.meta_percentage.desc().nulls_last())
            .limit(30)
        )
        return list(dict.fromkeys(result.scalars().all()))[:10]
```

Change the typing import at line 1 to:
```python
from typing import Optional, List, Dict, Any, Tuple
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_chat_jev_dispatch.py tests/test_chat_routing.py tests/test_chat_fit_handlers.py tests/test_format_filtering.py -v`
Expected: all PASS.

Run: `cd backend && python3 -m pytest`
Expected: 189 passed.

- [ ] **Step 10: Commit**

```bash
git add backend/app/services/chat_routing.py backend/app/services/chat_service.py \
  backend/tests/test_chat_routing.py backend/tests/test_chat_jev_dispatch.py
git commit -m "Route chat turns with Jev before any LLM call

One Jev request per turn picks the action (a Choice over the seven
tools plus reply) and judges colors, roles, package role, archetype,
opponent and which mentioned cards the user wants. Code assembles each
tool's input from those answers. A confident tool choice dispatches
directly, a confident reply uses a text-only completion, and the
existing forced-tool rule overrides reply with the most likely tool.
Low confidence, a failure, or no key falls back to the LLM tool call
unchanged.

Cards the user only mentions (an opponent's) are kept out of
specific_cards and stripped from the strategy text.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---
### Task 9: Routing and parsing eval against real Jev; relabel the fit eval

**Files:**
- Create: `backend/scripts/eval_routing.py`
- Modify: `backend/scripts/eval_fit.py:47-48` (Lazotep Reaver label)

**Interfaces:**
- Consumes: `chat_routing.route`, `chat_routing.ROUTE_CONFIDENCE`, `Route` (Task 8); `chat_service.TOOLS`; `deck_parsing.parse_deck_request` (Task 7).
- Produces: nothing other tasks import. These are manual tuning tools, not run in CI.

- [ ] **Step 1: Relabel Lazotep Reaver**

In `backend/scripts/eval_fit.py`, change the Lazotep Reaver case's label from `"good"` to `"bad"`:
```python
    (GRAVEYARD, c("Lazotep Reaver", "Creature — Zombie Beast",
                  "When Lazotep Reaver enters the battlefield, amass 1.", "{1}{B}"), "bad"),
```

- [ ] **Step 2: Write the routing eval script**

`backend/scripts/eval_routing.py`:
```python
"""
Run labeled chat messages through real Jev routing, and deck requests through
Jev parsing, and report action/argument accuracy plus a ROUTE_CONFIDENCE sweep.

Usage (needs TYPESAFE_API_KEY in .env):  cd backend && python3 scripts/eval_routing.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import chat_routing  # noqa: E402
from app.services.ai import deck_parsing  # noqa: E402
from app.services.chat_service import TOOLS  # noqa: E402

META = ["Mono-Red Aggro", "Dimir Midrange", "Izzet Prowess", "Azorius Control"]
NO_DECK = {"has_deck": False, "unique_nonland_cards": 0, "colors": []}
RED_DECK = {"has_deck": True, "unique_nonland_cards": 14, "colors": ["R"]}
UB_DECK = {"has_deck": True, "unique_nonland_cards": 20, "colors": ["U", "B"]}
FULL_DECK = {"has_deck": True, "unique_nonland_cards": 36, "colors": ["R"]}
SHEOLDRED = "Sheoldred, the Apocalypse"

# (message, deck summary, persisted strategy, resolved card names, expected action, expected args)
# Args are checked on the expected action's input; lists compare as sets. Special keys:
#   _not_color: a color that must not be in suggest_core's colors
#   _has_role:  a role that must be in suggest_core's roles
#   _wants:     generate_full_deck's specific_cards, as a set
CASES = [
    ("Build me a mono-red aggro deck", NO_DECK, "", [], "suggest_core", {"colors": ["R"]}),
    ("beat mono-red", NO_DECK, "", [], "get_matchup_info",
     {"opponent_deck": "Mono-Red Aggro", "_not_color": "R"}),
    ("more removal", RED_DECK, "Mono-red aggro", [], "suggest_package", {"role": "removal"}),
    ("just build it", NO_DECK, "Mono-red aggro", [], "generate_full_deck", {}),
    ("what's good right now?", NO_DECK, "", [], "analyze_meta", {}),
    ("swap Shock for Lightning Strike", RED_DECK, "Mono-red aggro", ["Shock", "Lightning Strike"],
     "modify_deck", {"modification": "swap Shock for Lightning Strike"}),
    ("What does trample do?", RED_DECK, "Mono-red aggro", [], "reply", {}),
    ("How do I beat Dimir Midrange?", RED_DECK, "Mono-red aggro", [], "get_matchup_info",
     {"opponent_deck": "Dimir Midrange"}),
    ("Give me 8 cards with surveil", UB_DECK, "UB self-mill", [], "suggest_package",
     {"role": "surveil", "count": 8}),
    ("I need more card draw", UB_DECK, "UB control", [], "suggest_package", {"role": "card draw"}),
    ("Let's add the mana base", FULL_DECK, "Mono-red aggro", [], "finalize_mana_base", {"colors": ["R"]}),
    (f"Build around {SHEOLDRED}", NO_DECK, "", [SHEOLDRED], "suggest_core", {"_wants": [SHEOLDRED]}),
    (f"My opponent keeps casting {SHEOLDRED}, how do I deal with it?", RED_DECK, "Mono-red aggro",
     [SHEOLDRED], "get_matchup_info", {"_wants": []}),
    ("Make a Dimir control deck", NO_DECK, "", [], "suggest_core", {"colors": ["U", "B"]}),
    ("Skip the suggestions and give me a whole Boros deck", NO_DECK, "", [], "generate_full_deck",
     {"colors": ["R", "W"]}),
    ("Is Lightning Strike better than Shock?", RED_DECK, "Mono-red aggro",
     ["Lightning Strike", "Shock"], "reply", {}),
    ("Cut the four Shocks", RED_DECK, "Mono-red aggro", ["Shock"], "modify_deck", {}),
    ("What's the best aggro deck in the format?", NO_DECK, "", [], "analyze_meta", {}),
    ("What sideboard cards help against Izzet Prowess?", RED_DECK, "Mono-red aggro", [],
     "get_matchup_info", {"opponent_deck": "Izzet Prowess"}),
    ("Show me some counterspells for my deck", UB_DECK, "UB control", [], "suggest_package",
     {"role": "counterspells"}),
    ("I want to play green ramp", NO_DECK, "", [], "suggest_core", {"colors": ["G"], "_has_role": "ramp"}),
    ("Thanks!", RED_DECK, "Mono-red aggro", [], "reply", {}),
    ("Add lands", FULL_DECK, "Mono-red aggro", [], "finalize_mana_base", {}),
    ("Give me some cheap threats", RED_DECK, "Mono-red aggro", [], "suggest_package",
     {"role": "cheap threats"}),
    ("What archetypes are popular?", NO_DECK, "", [], "analyze_meta", {}),
]
# The spec's must-route cases: these may fall below the threshold (and go to the
# LLM) but must never be confidently wrong.
REQUIRED = {"beat mono-red", "more removal", "just build it", "what's good right now?",
            "swap Shock for Lightning Strike", "What does trample do?"}

# (prompt, resolved card names, expected fields). _not_color: a color that must not be picked.
PARSE_CASES = [
    ("Build a mono-red aggro deck", [],
     {"archetype": "aggro", "colors": ["R"], "colors_specified": True, "specific_cards": []}),
    ("A deck that beats mono-red", [], {"colors_specified": False, "_not_color": "R"}),
    (f"Dimir control with {SHEOLDRED}", [SHEOLDRED],
     {"archetype": "control", "colors": ["U", "B"], "colors_specified": True,
      "specific_cards": [SHEOLDRED]}),
    (f"Something to beat {SHEOLDRED} decks", [SHEOLDRED], {"specific_cards": []}),
    ("aggro deck", [], {"archetype": "aggro", "colors_specified": False}),
]


def same(a, b):
    return set(a) == set(b) if isinstance(b, list) else a == b


def args_ok(r: chat_routing.Route, action: str, expected: dict) -> bool:
    got = r.inputs.get(action, {})
    for key, want in expected.items():
        if key == "_not_color":
            if want in r.inputs["suggest_core"]["colors"]:
                return False
        elif key == "_has_role":
            if want not in r.inputs["suggest_core"]["roles"]:
                return False
        elif key == "_wants":
            if not same(r.inputs["generate_full_deck"]["specific_cards"], want):
                return False
        elif not same(got.get(key), want):
            return False
    return True


async def eval_routing() -> bool:
    rows = []
    print(f"{'message':46} {'expected':18} {'got':18} conf")
    for msg, deck, strategy, cards, want, expected in CASES:
        r = await chat_routing.route(
            message=msg, history=[{"role": "user", "content": msg}], summary=deck,
            strategy=strategy, card_names=cards, meta_archetypes=META, tools=TOOLS)
        if r is None:
            print(f"{msg[:46]:46} ERROR (no key or Jev unavailable)")
            return False
        ok = r.action == want
        ok_args = ok and args_ok(r, want, expected)
        rows.append((msg, r, ok, ok_args))
        verdict = ("ok" if ok_args else "args-MISS") if ok else "MISS"
        print(f"{msg[:46]:46} {want:18} {r.action:18} {r.confidence:.2f} {verdict}")

    n = len(rows)
    right = [row for row in rows if row[2]]
    print(f"\naction accuracy {len(right)}/{n}; argument accuracy {sum(row[3] for row in right)}/{len(right)}")
    for t in (0.4, 0.5, 0.6, 0.7, 0.8):
        confident = [row for row in rows if row[1].confidence >= t]
        good = sum(row[3] for row in confident)
        print(f"threshold {t:.1f}: {len(confident)}/{n} routed without the LLM, {good} fully correct")

    t = chat_routing.ROUTE_CONFIDENCE
    confident = [row for row in rows if row[1].confidence >= t]
    wrong_required = [row[0] for row in confident if row[0] in REQUIRED and not row[3]]
    for msg in wrong_required:
        print(f"REQUIRED CONFIDENTLY WRONG  {msg}")
    precision = sum(row[3] for row in confident) / max(len(confident), 1)
    print(f"at ROUTE_CONFIDENCE={t}: {precision:.0%} of confident routes fully correct")
    return precision >= 0.9 and not wrong_required


async def eval_parsing() -> bool:
    agree = 0
    print(f"\n{'prompt':46} result")
    for prompt, names, want in PARSE_CASES:
        async def extract(p, db, format="standard", _names=names):
            return list(_names)
        deck_parsing.extract_card_names_from_prompt = extract
        got = await deck_parsing.parse_deck_request(prompt, db=None)
        if "budget" in got:
            print(f"{prompt[:46]:46} FALLBACK (no key or Jev unavailable)")
            continue
        ok = all(want["_not_color"] not in got["colors"] if k == "_not_color" else same(got[k], v)
                 for k, v in want.items())
        agree += ok
        print(f"{prompt[:46]:46} {'ok' if ok else 'MISS'}  {got}")
    print(f"parsing: {agree}/{len(PARSE_CASES)}")
    return agree >= 4


async def main() -> None:
    routing_ok = await eval_routing()
    parsing_ok = await eval_parsing()
    print("PASS" if routing_ok and parsing_ok else
          "FAIL (need >= 90% of confident routes correct, no required case confidently wrong, parsing >= 4/5)")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Run it against real Jev**

Run: `cd backend && python3 scripts/eval_routing.py`
Expected: a 25-row routing table, the accuracy and threshold-sweep lines, a 5-row parsing table, and `PASS`. (A pre-run while writing this plan gave 25/25 actions, 24/25 arguments (the Boros colors missed), 96% of confident routes fully correct at 0.6, and parsing 5/5.) There must be no `ERROR` or `FALLBACK` rows. If there are, confirm `TYPESAFE_API_KEY` is set in the repo-root `.env`.

- [ ] **Step 4: Tune if it prints FAIL**

Look at the rows marked `MISS` or `args-MISS` and change only these, one at a time, re-running after each:
1. Wording in `chat_routing.py`: `REPLY_DESCRIPTION`, the `ARCHETYPES` descriptions, or a question's `instructions`. Do not edit the `TOOLS` descriptions in `chat_service.py`: the LLM fallback path uses them too.
2. Wording in `deck_parsing.py` (`ARCHETYPES`, question instructions) for parsing misses.
3. `ROUTE_CONFIDENCE`. Use the sweep: pick the lowest threshold where at least 90% of confident routes are fully correct and no required case is confidently wrong. If you change it, update the Global Constraints in this plan and the spec to match.

Re-run `cd backend && python3 -m pytest` after any change. If it still prints FAIL after three wording attempts, stop and report the tables to the human before continuing to Task 10.

- [ ] **Step 5: Run the fit eval with the new label**

Run: `cd backend && python3 scripts/eval_fit.py`
Expected: a 15-row table and an agreement line with no `ERROR` rows. Record the agreement in the commit message. The relabel is the change; no tuning is required here.

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/eval_routing.py backend/scripts/eval_fit.py \
  backend/app/services/chat_routing.py backend/app/services/ai/deck_parsing.py
git commit -m "Add Jev routing eval and relabel Lazotep Reaver in the fit eval

25 labeled chat messages and 5 deck requests run through real Jev,
reporting action and argument accuracy and a ROUTE_CONFIDENCE sweep.
Result: <A>/25 actions, <P>% of confident routes correct at
ROUTE_CONFIDENCE=<T>, parsing <K>/5. Fit eval: <F>/15.
<one line on any tuning, or 'no tuning needed'>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```
Fill in the `<...>` parts from the actual runs before committing.

---

### Task 10: End-to-end check against the running stack

**Files:** none, unless a bug is found.

**Interfaces:**
- Consumes: everything above, live. The backend container bind-mounts `./backend` and runs `uvicorn --reload`, so code changes are already live in it.

- [ ] **Step 1: Check the stack without disturbing the role job**

Run:
```bash
docker ps --filter name=spellbook --format '{{.Names}} {{.Status}}'
docker exec spellbook-backend sh -c 'tail -3 /tmp/classify.log'
```
Expected: `spellbook-backend`, `spellbook-db`, `spellbook-redis` and `spellbook-frontend` are `Up`. The log shows either batch progress (the LLM role job is still running) or `Classification complete`.

**Never recreate `spellbook-backend` while the log shows the job is still running.** Do not run `docker compose up`, `restart` or `down` against it; `docker exec` is safe. Only if `spellbook-backend` is *not* running, start the stack with Redis on host port 6380 and without the shell's OpenAI key:
```bash
env -u OPENAI_API_KEY docker compose \
  -f /private/tmp/claude-501/-Users-robmiller-Projects-mtg-deckbuilder/b8847f7b-d5b1-408b-9720-62bba18fb355/scratchpad/compose-local.yml \
  --project-directory /Users/robmiller/Projects/mtg-deckbuilder up -d
```

- [ ] **Step 2: Confirm keys, migration, and reload**

Run:
```bash
docker exec spellbook-backend sh -c 'test -n "$TYPESAFE_API_KEY" && echo jev-key-set; test -n "$OPENAI_API_KEY" && echo openai-set || echo openai-unset'
docker exec spellbook-backend alembic upgrade head
docker exec spellbook-backend alembic current
docker logs spellbook-backend --since 30m 2>&1 | grep -iE "reload|error|traceback" | tail -20
```
Expected: `jev-key-set`; `alembic current` prints `016 (head)`; the log shows reloads after the code changes and no import errors or tracebacks. Record whether OpenAI is set: if it is, the live chat uses vector hits in the shortlist, and Step 5 checks the no-OpenAI path separately.

- [ ] **Step 3: Live chat through Jev routing**

Run:
```bash
S=/private/tmp/claude-501/-Users-robmiller-Projects-mtg-deckbuilder/b8847f7b-d5b1-408b-9720-62bba18fb355/scratchpad
curl -s -X POST http://localhost:8000/api/conversations/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Build me a mono-red aggro deck", "format": "standard"}' > $S/chat.json
python3 - "$S/chat.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(r["response"][:200])
low = []
for g in r.get("card_suggestions") or []:
    print(f"{g['group_name']}: " + ", ".join(
        f"{c['card_name']} ({c['fit']['plan_fit']:.2f})" if c.get("fit") else c["card_name"]
        for c in g["cards"]))
    low += [c["card_name"] for c in g["cards"] if c.get("fit") and c["fit"]["plan_fit"] < 0.34]
print("groups:", len(r.get("card_suggestions") or []), "| below plan-fit cutoff:", low or "none")
EOF
docker logs spellbook-backend --since 5m 2>&1 | grep "\[ROUTE\]"
```
Expected: role groups print (for example Threats, Removal, Card Advantage) with plan-fit values, `below plan-fit cutoff: none`, and the log shows `[ROUTE] Jev action=suggest_core confidence=...` with no `[ROUTE] LLM tool call` line for this request. If the log shows the LLM line instead, re-check Task 9's threshold sweep for this message before calling it a bug.

- [ ] **Step 4: Role check and minimum fit in a follow-up turn**

Run (reuse the conversation):
```bash
CID=$(python3 -c "import json;print(json.load(open('$S/chat.json'))['conversation_id'])")
curl -s -X POST http://localhost:8000/api/conversations/chat -H 'Content-Type: application/json' \
  -d "{\"message\": \"more removal\", \"format\": \"standard\", \"conversation_id\": \"$CID\"}" \
  | python3 -c "import json,sys; r=json.load(sys.stdin); [print(g['group_name'], [c['card_name'] for c in g['cards']]) for g in r.get('card_suggestions') or []]"
```
Expected: one Removal group of cards that remove things. No equipment or auras such as Basilisk Collar.

- [ ] **Step 5: Search without OpenAI**

Run (a fresh process inside the container, with OpenAI blanked; the running server is untouched):
```bash
docker exec -e OPENAI_API_KEY= spellbook-backend python -c "
import asyncio
from app.db.session import async_session_factory
from app.services.card_service import CardService
async def main():
    async with async_session_factory() as db:
        cards = await CardService(db).semantic_search('cheap red removal', limit=5, format='standard', colors=['R'])
        print([c.name for c in cards])
asyncio.run(main())
"
```
Expected: a non-empty list of red removal spells.

- [ ] **Step 6: Report**

Record what each step showed: route log lines, groups, any below-cutoff cards, the removal group, and the search list. Fix any bug with a failing test first, then commit the fix separately.

For the PR description (do not run it here): re-tagging card roles with Jev after merge is manual. Run `DELETE FROM card_roles;` and then the role-classification job.
