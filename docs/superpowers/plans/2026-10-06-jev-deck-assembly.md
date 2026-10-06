# Jev Deck Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 60-card decks from a slot plan (taken from recent tournament decklists, or from the LLM for brews) with Jev choosing each slot's cards from cards played in the last 14 days, falling back to today's LLM path whenever Jev is unavailable or fails.

**Architecture:** `deck_plan.py` decides what the deck needs: it picks a reference archetype with one Jev `Choice`, then turns that archetype's decklists into role-and-mana-value slots, or asks the LLM for a brew plan. `deck_fill.py` fills the plan: one Jev `Choice` per slot over that slot's pool of played cards, code turning the ranking into copies, then requested cards, lands, basics, sideboard and a summary. `DeckGenerator.generate()` calls `deck_fill.assemble` in place of `ai_service.generate_deck` and keeps the LLM call as the fallback. The mtgtop8 scraper follows the two-week view's second page.

**Tech Stack:** FastAPI, SQLAlchemy async (`text()` SQL on Postgres JSONB), `typesafe-sdk==0.7.2` (`Choice`), OpenRouter via `app.services.llm`, BeautifulSoup, pytest + pytest-asyncio (`asyncio_mode=auto`).

**Spec:** `docs/superpowers/specs/2026-10-06-jev-deck-assembly-design.md`

## Global Constraints

- "**Tournament-playable** means the card appears in a decklist from the last 14 days for the format (`events.date >= today - 14`)." `WINDOW_DAYS = 14`.
- "**Only tournament-playable cards are eligible**, with one exception: cards the user explicitly requests (`specific_cards` from parsing) always go in, regardless of play data."
- "one Jev `Choice` over up to 255 candidates (the API maximum)". Reference candidates are "capped at 254" plus `none`; every slot pool "is ordered by play count and capped at 255". `MAX_OPTIONS = 255`.
- "If the pick isn't `none` and confidence is at least `REFERENCE_CONFIDENCE` (0.5), return the archetype name. Otherwise return `None`, meaning a brew." Reference candidates are archetypes with "at least 2 lists".
- Copy rules: "With a reference, copies are the card's average copies in the reference lists, rounded, at least 1. For brews, they are 4, 4, 3, 2, 1 by rank, then 1." "Copies are capped at 4 for nonbasics." Requested cards: "4 copies, or 1 if the type line contains `Legendary`."
- Land rules: "**Nonbasic count:** the reference lists' average nonbasic count. For brews it is 0 for mono-color, 4 for two colors and 6 for three or more." "**Pool:** lands played in the last 14 days whose `color_identity` ⊆ deck colors. That covers on-color duals and colorless utility lands (identity `{}`). Any off-color symbol excludes a land." "**Basics:** they fill the remaining land count, split by the colored mana symbols across the chosen nonland cards' mana costs. Every deck color gets at least 1 basic when basics are used." "Requested lands count toward the land total."
- Brew land counts: "The land count is 24 for control, 23 for midrange and 22 otherwise." "Brews get no sideboard."
- Slot planning: "A slot under 2 copies merges into the same role's nearest band, then into the same band's largest slot." Bands are "0-1, 2, 3, 4, 5+". "Sideboard slots are built the same way from the lists' sideboards, totalling 15."
- Option text: card name → `"{mana_cost} {type_line}. {oracle_text[:200]} [Played in N recent {format} tournament decklists]"`.
- "A Jev failure never blocks generation: the generator falls back to today's LLM path." "Jev calls go through `app/services/jev.py`, with its shared cap and deadline."
- "Formats other than 60-card constructed" are out of scope: "cEDH and Commander keep the LLM path."
- Unit tests never touch the network. `backend/tests/conftest.py` already blanks every API key; tests stub Jev with `tests/jev_fake.py` (`FakeJev`) and the database with `MagicMock`/`AsyncMock`. The live eval runs only in its own step, never in CI.
- Backend tests: `cd backend && python3 -m pytest` (baseline before Task 1: 215 passed).
- Every commit message ends with exactly these two lines:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN
  ```

### Resolved spec ambiguities

- **Decklist names vs card names.** Decklists list double-faced cards by front face ("Sephiroth, Fabled SOLDIER"); `cards` stores "Sephiroth, Fabled SOLDIER // Sephiroth, One-Winged Angel". Every join compares `lower(split_part(name, ' // ', 1))` on both sides, and decks use the full card name. In the live data, 35 of the 571 card names in recent Standard decklists are front-face-only.
- **A card's colors.** DFCs store colors and mana cost per face, so their top-level `colors` is `{}` and `mana_cost` is empty. "`colors` ⊆ deck colors" uses `coalesce(nullif(colors, '{}'), color_identity, '{}')` (`CARD_COLORS`), so an off-color DFC can't pass as colorless. The basics split counts each identity color once for a card with no mana cost.
- **Archetype names** are grouped by `lower(trim(archetype))` (the live data has "4/5C Control" and "4/5c Control", and "Weenie White " with a trailing space). An archetype literally named "none" is dropped, because `none` is the brew option.
- **`choose_reference` takes the archetype list.** Its signature is `choose_reference(client, request_text, colors, archetypes, format)`. `assemble` loads `recent_archetypes` once and raises when the format has no recent decklists (Historic has none), which sends that request to the LLM path.
- **60 cards exactly.** The nonland slots total `60 - lands`, where lands is the rounded average land count. Rounding the average nonland count separately can give 59 or 61. All rounding of slot sizes uses largest remainder.
- **Merging small slots.** A slot merged into the same role's nearest band widens that slot's mana range to cover both (ties go to the lower band). A slot merged into a different role's slot keeps the target's role. When there is no slot in the same band, it merges into the largest slot.
- **Reference colors.** When parsing found no colors, a reference deck uses its lists' colors: the colors played by at least half the lists, so one list's splash doesn't count. A brew with no colors raises, which sends it to the LLM path.
- **Reference copies for a card not in the reference lists** use the brew rank rule.
- **Slot order.** "Largest first" means the largest *remaining* slot, re-checked after each overflow. When the pools run out after the last slot, one catch-all slot (role `any`, any mana value, still played-only) takes the shortfall. Anything still short becomes basic lands, so the main deck stays at 60.
- **Requested cards** must be legal in the format; unknown or illegal names are skipped with a warning, like the LLM path's `find_valid_card`. A name repeated in the request goes in once. Requested nonbasic lands count toward the nonbasic target as well as the land total.
- **Sideboard.** Sideboard pools exclude main-deck cards, which keeps a card at no more than 4 copies across main and sideboard. Sideboard slot descriptions start with "Sideboard card: ", so Jev knows what it is picking.
- **Jev timeouts.** The client's default 2 s timeout is too short for a 255-option `Choice`, so each one passes `timeout=CHOICE_TIMEOUT` (10 s) through `jev.ask_many` with `deadline=CHOICE_DEADLINE` (15 s). A timeout raises, and the generator falls back. One of four live eval runs while writing this plan hit a 10 s timeout.
- **`llm.complete` is synchronous**, so the plan and summary calls run in `asyncio.to_thread`.
- **`assemble` also returns `reference` and `colors`.** `DeckGenerator` ignores the extra keys; the eval uses them.
- **The 14-day window is inlined** as `CURRENT_DATE - 14` (`RECENT`), because asyncpg can't bind an integer parameter in date arithmetic (`operator does not exist: date >= integer`).
- **60-card formats** are `standard`, `historic`, `modern` and `legacy` (the `FORMAT_LEGALITY_MAP` keys minus `cedh`). Others raise before any Jev or database call and go to the LLM path.
- **"When Jev is configured."** `DeckGenerator` always tries `assemble`, which raises when there is no `TYPESAFE_API_KEY`. That gives the same fallback with one code path.
- **Scraper pagination.** The scraper follows the page's "Next" link (`?f=ST&meta=50&cp=2`; the `meta` id differs per format). It stops when a page has no dated event rows or no "Next" link, and gives up after `MAX_EVENT_PAGES` (10) pages. Each event appears twice on page 1, so ids are deduplicated. On 2026-10-06 it found 25 Standard events over two pages; the spec's 22 was counted earlier.

### Known consequences (from the spec, not fixed here)

- Brews have no sideboard, so `DeckValidator` reports a `sideboard_size` error on brew decks and `is_validated` is false for them.
- Fetchlands have color identity `{}` and pass the land filter for any colors. Jev's land-slot description ("fixing for its colors, or utility") is the only guard. This is marked `ponytail:` in `land_pool`.

## Review Focus

1. **Double-faced cards** (listed by front face in decklists, empty top-level `colors` and `mana_cost` in `cards`). Expected: they join to their card rows, get filtered by color identity so an off-color DFC never passes as colorless, are found when requested by front face, and count toward basics by identity. Tests: Task 3 (`test_slots_lands_copies_and_sideboard` checks the front-face join), Task 5 (`test_filters_and_params` checks `CARD_COLORS`), Task 6 (`test_takes_four_copies_from_the_first_fitting_slot`, `test_legendary_is_one_copy_and_overflow_cuts_the_largest_slot`, `test_pips_count_hybrid_and_fall_back_to_identity`).
2. **A slot whose pool is empty or too thin** (an off-meta role, a narrow mana range, a mono-color deck). Expected: no Jev call for an empty pool; the shortfall moves to a same-role slot, then the largest slot, then one catch-all slot, then basics; the main deck is still 60. Tests: Task 5 (`test_empty_pool_makes_no_jev_call`, `test_largest_first_and_shortfall_moves_to_same_role`, `test_returns_what_cannot_be_filled`), Task 7 (`test_brew_has_no_sideboard_and_basic_land_split`).
3. **Requested cards that are odd**: a land, an off-color card, a Legendary DFC, an unknown or illegal name, or the same card twice. Expected: a land comes out of the land count, an off-color card adds its colors, a Legendary card is 1 copy, unknown or illegal names are skipped, and repeats go in once. Any copies beyond the fitting slot come out of the largest slot. Tests: Task 6 `TestReserveRequested`.
4. **Rounding and odd decklists** (a 23.5 average land count, 61-card lists, decklist names with no card row, untagged cards). Expected: nonland slots plus lands are exactly 60 and sideboard slots exactly 15. Untagged cards fall back to `creature` / `noncreature`, and unknown names count as `noncreature` at mana value 0. Tests: Task 2 (`test_largest_remainder_hits_the_total`), Task 3 (`test_untagged_cards_fall_back_to_type`, `test_slots_lands_copies_and_sideboard`), Task 7 (`test_reference_deck_is_60_and_15`).
5. **Archetype names as Choice options**: case and whitespace duplicates, an archetype named "None", more than 254 names, and one-list archetypes. Expected: grouped case-insensitively in SQL, "none" reserved, at most 255 options, and archetypes with fewer than 2 lists left out; no Jev call when nothing qualifies. Tests: Task 2 (`test_groups_names_case_insensitively_in_the_window`, `test_cap_and_reserved_none_name`, `test_no_candidates_skips_jev`).

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `backend/app/jobs/mtgtop8_scrape.py` | Event list follows the two-week view's pagination | 1 |
| `backend/app/services/deck_plan.py` (new) | `Slot`, `Plan`, `choose`, `recent_archetypes`, `choose_reference`, `plan_from_decklists`, `plan_with_llm`, default plans | 2, 3, 4 |
| `backend/app/services/deck_fill.py` (new) | `slot_pool`, `fill_slot`, `fill_slots`, `reserve_requested`, `land_pool`, `fill_lands`, basics, `summarize`, `assemble` | 5, 6, 7 |
| `backend/app/services/deck_generator.py` | Calls `deck_fill.assemble`; LLM path as fallback | 8 |
| `backend/scripts/eval_assembly.py` (new) | Live eval: three Standard decks against real Jev | 9 |
| `backend/tests/test_mtgtop8_pagination.py`, `test_deck_plan.py`, `test_deck_fill.py`, `test_deck_generator_assembly.py` (new) | Unit tests | 1-8 |

---

### Task 1: Scraper follows the two-week view's pagination

**Files:**
- Modify: `backend/app/jobs/mtgtop8_scrape.py` (import at line 13; constants at line 33; `scrape_recent_events` at lines 69-133)
- Test: `backend/tests/test_mtgtop8_pagination.py`

**Interfaces:**
- Consumes: `fetch_page(client, url) -> str` (unchanged), `MTGTOP8_BASE_URL`.
- Produces: `scrape_recent_events(client, format_id, format_name, days)` keeps its signature and return shape (`[{mtgtop8_id, name, date, format, url}]`); new `_dated_event_rows(soup) -> List[Tuple[str, str, date]]`; `MAX_EVENT_PAGES = 10`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_mtgtop8_pagination.py`:
```python
"""mtgtop8 event list: follow the two-week view's pagination."""

from datetime import date, timedelta

from app.jobs import mtgtop8_scrape as scrape


def day(days_ago: int) -> str:
    return (date.today() - timedelta(days=days_ago)).strftime("%d/%m/%y")


def row(event_id: str, name: str, when: str) -> str:
    return (f'<tr class=hover_tr><td></td><td class=S14><a href=event?e={event_id}&f=ST>{name}</a></td>'
            f'<td></td><td align=right class=S12>{when}</td></tr>')


def page(rows, nav: str = "") -> str:
    # an undated "Last major events" style row is on every real page
    undated = '<tr><td><a href=event?e=999&f=ST>The Decks to Beat</a></td></tr>'
    return f"<html><table>{''.join(rows)}{undated}</table><div>{nav}</div></html>"


NEXT = ('<div class=Nav_cur>1</div><div class=Nav_norm><a href=?f=ST&meta=50&cp=2>2</a></div>'
        '<div class=Nav_norm><a href=?f=ST&meta=50&cp=2>Next</a></div>')
PREV = '<div class=Nav_norm><a href=?f=ST&meta=50&cp=1>Prev</a></div>'


def fake_fetch(monkeypatch, pages):
    urls = []

    async def fetch(client, url):
        urls.append(url)
        return pages[url]

    monkeypatch.setattr(scrape, "fetch_page", fetch)
    return urls


async def test_follows_next_page_and_dedupes(monkeypatch):
    base = f"{scrape.MTGTOP8_BASE_URL}/format"
    urls = fake_fetch(monkeypatch, {
        f"{base}?f=ST": page([row("1", "League", day(1)), row("1", "League", day(1)),
                              row("2", "Challenge", day(3))], NEXT),
        f"{base}?f=ST&meta=50&cp=2": page([row("3", "RCQ", day(13))], PREV),
    })
    events = await scrape.scrape_recent_events(None, "ST", "standard", days=14)
    assert [e["mtgtop8_id"] for e in events] == ["1", "2", "3"]
    assert urls == [f"{base}?f=ST", f"{base}?f=ST&meta=50&cp=2"]
    assert events[2]["url"] == f"{scrape.MTGTOP8_BASE_URL}/event?e=3"


async def test_stops_at_a_page_without_dated_rows(monkeypatch):
    base = f"{scrape.MTGTOP8_BASE_URL}/format"
    urls = fake_fetch(monkeypatch, {
        f"{base}?f=ST": page([row("1", "League", day(1))], NEXT),
        f"{base}?f=ST&meta=50&cp=2": page([], NEXT),
    })
    events = await scrape.scrape_recent_events(None, "ST", "standard", days=14)
    assert [e["mtgtop8_id"] for e in events] == ["1"]
    assert len(urls) == 2


async def test_drops_events_older_than_the_window(monkeypatch):
    fake_fetch(monkeypatch, {
        f"{scrape.MTGTOP8_BASE_URL}/format?f=ST": page([row("1", "New", day(2)), row("2", "Old", day(20))]),
    })
    events = await scrape.scrape_recent_events(None, "ST", "standard", days=14)
    assert [e["name"] for e in events] == ["New"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_mtgtop8_pagination.py -v`
Expected: FAIL on `test_follows_next_page_and_dedupes` (only page 1 is fetched and event 1 appears twice) and `test_stops_at_a_page_without_dated_rows`; `test_drops_events_older_than_the_window` passes (today's behavior).

- [ ] **Step 3: Implement**

In `backend/app/jobs/mtgtop8_scrape.py`, change the datetime import:
```python
from datetime import date, datetime, timedelta
```

Below `REQUEST_DELAY = 1.0  # Be nice to the server`, add:
```python
MAX_EVENT_PAGES = 10  # ponytail: safety stop; the two-week view is 1-2 pages
```

Replace the whole `scrape_recent_events` function (from `async def scrape_recent_events(` down to, not including, `async def scrape_event_decklists(`) with:
```python
def _dated_event_rows(soup: BeautifulSoup) -> List[Tuple[str, str, date]]:
    """(event id, name, date) for each event link in a dated table row."""
    rows = []
    for link in soup.select("a[href*='event?e=']"):
        match = re.search(r"e=(\d+)", link.get("href", ""))
        row = link.find_parent("tr")
        if not match or row is None:
            continue
        cells = row.find_all("td")
        try:
            event_date = datetime.strptime(cells[-1].get_text(strip=True), "%d/%m/%y").date()
        except (ValueError, IndexError):
            continue
        rows.append((match.group(1), link.get_text(strip=True), event_date))
    return rows


async def scrape_recent_events(
    client: httpx.AsyncClient,
    format_id: str = STANDARD_FORMAT_ID,
    format_name: str = "standard",
    days: int = 14,
) -> List[Dict[str, Any]]:
    """Scrape recent events for a format from mtgtop8's two-week view, following
    its pagination (?f=X&meta=..&cp=N, via the "Next" link) until a page has no
    dated event rows or no next page."""
    events: List[Dict[str, Any]] = []
    seen = set()
    cutoff = datetime.now().date() - timedelta(days=days)
    url = f"{MTGTOP8_BASE_URL}/format?f={format_id}"
    for _ in range(MAX_EVENT_PAGES):
        soup = BeautifulSoup(await fetch_page(client, url), "html.parser")
        rows = _dated_event_rows(soup)
        if not rows:
            break
        for event_id, name, event_date in rows:
            if event_date < cutoff or event_id in seen:
                continue
            seen.add(event_id)
            events.append({
                "mtgtop8_id": event_id,
                "name": name,
                "date": event_date,
                "format": format_name,
                "url": f"{MTGTOP8_BASE_URL}/event?e={event_id}",
            })
        next_href = next((a.get("href") for a in soup.select("div.Nav_norm a")
                          if a.get_text(strip=True) == "Next"), None)
        if not next_href:
            break
        url = f"{MTGTOP8_BASE_URL}/format{next_href}"

    logger.info(f"Found {len(events)} recent {format_name} events")
    return events
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && python3 -m pytest tests/test_mtgtop8_pagination.py -v && python3 -m pytest -q`
Expected: 3 passed; full suite 218 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/jobs/mtgtop8_scrape.py backend/tests/test_mtgtop8_pagination.py
git commit -m "Follow mtgtop8's two-week event pagination

The format page shows about 20 events; the rest of the two-week view is on
?f=X&meta=..&cp=N. Follow the Next link until a page has no dated event
rows, and skip the duplicate event links each page carries.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 2: `deck_plan`: `Slot`, `Plan` and the reference archetype

**Files:**
- Create: `backend/app/services/deck_plan.py`
- Test: `backend/tests/test_deck_plan.py`

**Interfaces:**
- Consumes: `jev.ask_many(client, requests, deadline=None, **kwargs)`; `ROLE_DEFINITIONS` from `app.models.card`; `FakeJev` from `tests/jev_fake.py` (a Choice answer is a label or `{"choice", "confidence", "probabilities"}`).
- Produces (in `app.services.deck_plan`):
  - constants `WINDOW_DAYS = 14`, `REFERENCE_CONFIDENCE = 0.5`, `MAX_OPTIONS = 255`, `MIN_LISTS = 2`, `CHOICE_TIMEOUT = 10.0`, `CHOICE_DEADLINE = 15.0`, `NONE_OPTION = "none"`, `WUBRG = "WUBRG"`, `BANDS`, and `RECENT` (a SQL fragment for `e` = events with a `:format` parameter)
  - `@dataclass Slot(role: str, cmc_min: int, cmc_max: int, copies: int, description: str = "", type_contains: Optional[str] = None)`
  - `@dataclass Plan(slots: List[Slot], lands: int, nonbasic_lands: Optional[int] = None, sideboard: List[Slot] = [], copies: Dict[str, int] = {}, side_copies: Dict[str, int] = {}, colors: List[str] = [], reference: Optional[str] = None)`. `nonbasic_lands=None` means the brew rule applies.
  - `band_of(cmc) -> Tuple[int, int]`, `describe(role, cmc_min, cmc_max) -> str`, `largest_remainder(sizes, total) -> List[int]`
  - `async choose(client, state, question: Choice)` returns the answer (`.choice`, `.confidence`, `.probabilities`) and raises when Jev fails
  - `async recent_archetypes(db, format) -> List[Tuple[str, int]]`
  - `async choose_reference(client, request_text, colors, archetypes, format) -> Optional[str]`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_deck_plan.py`:
```python
"""deck_plan: reference archetype choice and slot plans."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.card import ROLE_DEFINITIONS
from app.services import deck_plan as dp
from app.services.deck_plan import Slot
from tests.jev_fake import FakeJev


def fake_db(rows=()):
    result = MagicMock()
    result.all.return_value = list(rows)
    return MagicMock(execute=AsyncMock(return_value=result))


def sql_of(db):
    stmt, params = db.execute.call_args[0]
    return str(stmt), params


class TestHelpers:
    def test_band_of(self):
        assert [dp.band_of(c) for c in (0, 1, 2, 3, 4, 5, 9)] == [
            (0, 1), (0, 1), (2, 2), (3, 3), (4, 4), (5, 99), (5, 99)]

    def test_describe(self):
        assert dp.describe("burn", 0, 1) == ROLE_DEFINITIONS["burn"] + " (mana value 0-1)"
        assert dp.describe("creature", 5, 99) == "A creature (mana value 5 or more)"
        assert dp.describe("noncreature", 3, 3) == "A noncreature spell (mana value 3)"

    def test_largest_remainder_hits_the_total(self):
        assert dp.largest_remainder([1.5, 1.5, 1.0], 4) == [2, 1, 1]
        assert sum(dp.largest_remainder([3.3, 7.1, 2.2, 9.9], 37)) == 37
        assert dp.largest_remainder([10, 0], 5) == [5, 0]


class TestRecentArchetypes:
    async def test_groups_names_case_insensitively_in_the_window(self):
        db = fake_db([("Dimir Aggro", 19), ("Boros Aggro", 2)])
        assert await dp.recent_archetypes(db, "standard") == [("Dimir Aggro", 19), ("Boros Aggro", 2)]
        sql, params = sql_of(db)
        assert params == {"format": "standard"}
        assert "e.date >= CURRENT_DATE - 14" in sql
        assert "GROUP BY lower(trim(d.archetype))" in sql


def picks(choice, confidence=0.9):
    return FakeJev(answer=lambda state, q: {
        "pick": {"choice": choice, "confidence": confidence, "probabilities": {choice: confidence}}})


ARCHETYPES = [("Dimir Aggro", 19), ("Boros Aggro", 2), ("Demone Scarso", 1)]


class TestChooseReference:
    async def test_picked_archetype(self):
        jev = picks("Boros Aggro")
        got = await dp.choose_reference(jev, "Build Boros aggro", ["R", "W"], ARCHETYPES, "standard")
        assert got == "Boros Aggro"
        state, questions, kwargs = jev.calls[0]
        assert state == {"request": "Build Boros aggro", "colors": ["R", "W"]}
        assert list(questions["pick"].criteria) == ["Dimir Aggro", "Boros Aggro", "none"]  # 1-list names out
        assert kwargs == {"timeout": dp.CHOICE_TIMEOUT}

    async def test_none_means_brew(self):
        assert await dp.choose_reference(picks("none"), "mono-red", ["R"], ARCHETYPES, "standard") is None

    async def test_low_confidence_means_brew(self):
        jev = picks("Dimir Aggro", confidence=0.4)
        assert await dp.choose_reference(jev, "blue black", ["U", "B"], ARCHETYPES, "standard") is None

    async def test_no_candidates_skips_jev(self):
        jev = picks("none")
        assert await dp.choose_reference(jev, "x", [], [("Rare Brew", 1)], "standard") is None
        assert jev.calls == []

    async def test_cap_and_reserved_none_name(self):
        many = [("None", 9)] + [(f"Deck {i}", 300 - i) for i in range(300)]
        jev = picks("Deck 0")
        await dp.choose_reference(jev, "x", [], many, "standard")
        criteria = jev.calls[0][1]["pick"].criteria
        assert len(criteria) == dp.MAX_OPTIONS
        assert "None" not in criteria and criteria["none"].startswith("None of these")

    async def test_jev_failure_raises(self):
        with pytest.raises(Exception):  # ask_many raises an ExceptionGroup
            await dp.choose_reference(FakeJev(fail=lambda s: True), "x", [], ARCHETYPES, "standard")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_plan.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'app.services.deck_plan'`.

- [ ] **Step 3: Implement**

`backend/app/services/deck_plan.py`:
```python
"""
Slot plans for Jev deck assembly: which kinds of cards a deck needs, and how many.

A plan comes from a reference archetype's recent decklists, or, for a brew,
from one LLM call (with fixed defaults when that fails).
Design: docs/superpowers/specs/2026-10-06-jev-deck-assembly-design.md
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Choice

from app.models.card import ROLE_DEFINITIONS
from app.services import jev

logger = logging.getLogger(__name__)

WINDOW_DAYS = 14  # "tournament-playable" = in a decklist from the last 14 days
REFERENCE_CONFIDENCE = 0.5
MAX_OPTIONS = 255  # Jev Choice maximum
MIN_LISTS = 2  # an archetype needs this many recent lists to be a reference
CHOICE_TIMEOUT = 10.0  # one Choice over 255 options can take a few seconds
CHOICE_DEADLINE = 15.0
NONE_OPTION = "none"
WUBRG = "WUBRG"
BANDS: List[Tuple[int, int]] = [(0, 1), (2, 2), (3, 3), (4, 4), (5, 99)]
# Decklists in the format from the last WINDOW_DAYS days (alias e = events). The
# day count is inlined: asyncpg cannot bind an integer into date arithmetic.
RECENT = f"e.format = :format AND e.date >= CURRENT_DATE - {WINDOW_DAYS}"


@dataclass
class Slot:
    role: str  # a non-land CARD_ROLES role, "creature" or "noncreature" (deck_fill adds "any", "land")
    cmc_min: int
    cmc_max: int
    copies: int
    description: str = ""
    type_contains: Optional[str] = None


@dataclass
class Plan:
    slots: List[Slot]
    lands: int
    nonbasic_lands: Optional[int] = None  # None: brew rule by color count
    sideboard: List[Slot] = field(default_factory=list)
    copies: Dict[str, int] = field(default_factory=dict)  # reference main-deck copies by card name
    side_copies: Dict[str, int] = field(default_factory=dict)
    colors: List[str] = field(default_factory=list)  # reference lists' colors, WUBRG order
    reference: Optional[str] = None


def band_of(cmc: float) -> Tuple[int, int]:
    return next(b for b in BANDS if cmc <= b[1])


def describe(role: str, cmc_min: int, cmc_max: int) -> str:
    what = {"creature": "A creature", "noncreature": "A noncreature spell"}.get(role) or ROLE_DEFINITIONS[role]
    mv = f"mana value {cmc_min}" if cmc_min == cmc_max else (
        f"mana value {cmc_min} or more" if cmc_max >= 99 else f"mana value {cmc_min}-{cmc_max}")
    return f"{what} ({mv})"


def largest_remainder(sizes: Sequence[float], total: int) -> List[int]:
    """Integers proportional to `sizes` that sum to `total`."""
    scale = total / sum(sizes) if sum(sizes) else 0
    raw = [s * scale for s in sizes]
    out = [int(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return out


async def choose(client, state: Dict[str, Any], question: Choice):
    """One Jev Choice through the shared cap and a deadline; returns the answer
    (`.choice`, `.confidence`, `.probabilities`). Raises when Jev fails."""
    (resp,) = await jev.ask_many(client, [(state, {"pick": question})],
                                 deadline=CHOICE_DEADLINE, timeout=CHOICE_TIMEOUT)
    return resp.choices["pick"]


async def recent_archetypes(db: AsyncSession, format: str) -> List[Tuple[str, int]]:
    """(archetype, decklist count) for the window, most lists first. Names are
    grouped case- and whitespace-insensitively."""
    sql = text(f"""
        SELECT MIN(trim(d.archetype)) AS name, COUNT(*) AS n
        FROM decklists d JOIN events e ON e.id = d.event_id
        WHERE {RECENT}
          AND coalesce(trim(d.archetype), '') <> ''
        GROUP BY lower(trim(d.archetype))
        ORDER BY n DESC, name
    """)
    result = await db.execute(sql, {"format": format})
    return [(row[0], row[1]) for row in result.all()]


async def choose_reference(client, request_text: str, colors: List[str],
                           archetypes: List[Tuple[str, int]], format: str) -> Optional[str]:
    """The current archetype the user is asking to build, or None for a brew."""
    names = [(a, n) for a, n in archetypes if n >= MIN_LISTS and a.lower() != NONE_OPTION]
    names = names[: MAX_OPTIONS - 1]
    if not names:
        return None
    criteria = {a: f"{a}: {n} recent {format} tournament decklists" for a, n in names}
    criteria[NONE_OPTION] = "None of these: the user wants a different or new deck of their own"
    answer = await choose(client, {"request": request_text, "colors": colors}, Choice(
        instructions="Which current archetype is the user asking to build?", criteria=criteria))
    if answer.choice == NONE_OPTION or answer.confidence < REFERENCE_CONFIDENCE:
        return None
    return answer.choice
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && python3 -m pytest tests/test_deck_plan.py -v && python3 -m pytest -q`
Expected: 10 passed; full suite 228 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/deck_plan.py backend/tests/test_deck_plan.py
git commit -m "Add deck_plan: slots and the Jev reference-archetype choice

One Jev Choice over the archetypes with at least 2 decklists in the last
14 days, plus none. A pick at confidence >= 0.5 is the reference; anything
else means a brew.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 3: Slot plan from the reference decklists

**Files:**
- Modify: `backend/app/services/deck_plan.py` (imports; append below `choose_reference`)
- Test: `backend/tests/test_deck_plan.py` (append)

**Interfaces:**
- Consumes: `Slot`, `Plan`, `RECENT`, `band_of`, `describe`, `largest_remainder`, `WUBRG` (Task 2).
- Produces:
  - constants `MIN_SLOT = 2`, `MAIN_SIZE = 60`, `SIDEBOARD_SIZE = 15`; `REFERENCE_SQL`
  - `is_land(type_line) -> bool` (front face only, so a spell // land MDFC is a spell)
  - `async plan_from_decklists(db, archetype, format) -> Optional[Plan]`: slots totalling `60 - lands`, `lands`, `nonbasic_lands`, `sideboard` totalling 15 (or `[]`), `copies` / `side_copies` (average copies per card name, rounded, at least 1), `colors` (played by at least half the lists), `reference=archetype`. `None` when the archetype has no lists in the window.
  - helpers `_slot_sizes`, `_merge_small`, `_to_slots`, `_avg_copies`

- [ ] **Step 1: Write the failing tests**

In `backend/tests/test_deck_plan.py`, add `from types import SimpleNamespace` as the first import line (above `from unittest.mock import AsyncMock, MagicMock`), then append:
```python
def entry(deck, name, qty, type_line, cmc, colors=(), role=None, section="main"):
    return SimpleNamespace(deck_id=deck, section=section, name=name, qty=qty, type_line=type_line,
                           cmc=cmc, colors=list(colors), role=role)


MOUNTAIN = "Basic Land — Mountain"
LISTS = [
    entry(1, "Hired Claw", 4, "Creature — Lizard Mercenary", 1, "R", "threat_cheap"),
    entry(1, "Burst Lightning", 4, "Instant", 1, "R", "burn"),
    entry(1, "Mystery Card", 1, None, None),  # decklist name with no card row
    entry(1, "Sacred Foundry", 4, "Land — Mountain Plains", 0),
    entry(1, "Mountain", 10, MOUNTAIN, 0),
    entry(1, "Rest in Peace", 2, "Enchantment", 2, "W", "graveyard_hate", "side"),
    entry(2, "Hired Claw", 3, "Creature — Lizard Mercenary", 1, "R", "threat_cheap"),
    entry(2, "Burst Lightning", 4, "Instant", 1, "R", "burn"),
    entry(2, "Lightning Helix", 2, "Instant", 2, "RW", "removal_targeted"),
    entry(2, "Sacred Foundry", 4, "Land — Mountain Plains", 0),
    entry(2, "Mountain", 9, MOUNTAIN, 0),
    entry(2, "Rest in Peace", 2, "Enchantment", 2, "W", "graveyard_hate", "side"),
]


class TestSlotSizes:
    def test_untagged_cards_fall_back_to_type(self):
        rows = [entry(1, "Bear", 4, "Creature — Bear", 2), entry(1, "Ponder", 2, "Sorcery", 1),
                entry(2, "Bear", 2, "Creature — Bear", 2)]
        assert dict(dp._slot_sizes(rows, 2)) == {("creature", (2, 2)): 3.0, ("noncreature", (0, 1)): 1.0}


class TestMergeSmall:
    def test_same_role_nearest_band_widens_the_range(self):
        got = dp._merge_small({("burn", (0, 1)): 6, ("burn", (3, 3)): 1, ("threat_cheap", (2, 2)): 8})
        assert got == [["burn", 0, 3, 7], ["threat_cheap", 2, 2, 8]]

    def test_same_band_largest_slot(self):
        got = dp._merge_small({("discard", (2, 2)): 1.5, ("threat_cheap", (2, 2)): 8,
                               ("burn", (2, 2)): 3, ("burn", (0, 1)): 4})
        assert ["threat_cheap", 2, 2, 9.5] in got and len(got) == 3

    def test_else_largest_slot(self):
        got = dp._merge_small({("tutor", (5, 99)): 1, ("burn", (0, 1)): 4, ("threat_cheap", (2, 2)): 8})
        assert got == [["burn", 0, 1, 4], ["threat_cheap", 2, 2, 9]]

    def test_a_lone_small_slot_stays(self):
        assert dp._merge_small({("burn", (0, 1)): 1}) == [["burn", 0, 1, 1]]


class TestPlanFromDecklists:
    async def test_slots_lands_copies_and_sideboard(self):
        db = fake_db(LISTS)
        plan = await dp.plan_from_decklists(db, "Boros Aggro", "standard")
        sql, params = sql_of(db)
        assert params == {"format": "standard", "archetype": "Boros Aggro"}
        assert "lower(trim(d.archetype)) = lower(trim(:archetype))" in sql
        assert "lower(split_part(c.name, ' // ', 1)) = n.k" in sql  # DFCs listed by front face
        assert "r.role NOT LIKE 'land%'" in sql

        assert plan.reference == "Boros Aggro"
        assert plan.lands == 14 and plan.nonbasic_lands == 4  # (14 + 13) / 2 rounds to 14
        # Averages: threat_cheap 0-1 3.5, burn 0-1 4, Helix 1, Mystery 0.5. Mystery joins its
        # band's largest slot (burn), then Helix the largest slot (burn): 3.5 vs 5.5, scaled to 46.
        assert {(s.role, s.cmc_min, s.cmc_max): s.copies for s in plan.slots} == {
            ("threat_cheap", 0, 1): 18, ("burn", 0, 1): 28}
        assert sum(s.copies for s in plan.slots) + plan.lands == dp.MAIN_SIZE
        assert plan.slots[0].description == dp.describe("threat_cheap", 0, 1)
        assert plan.copies["Hired Claw"] == 4 and plan.copies["Lightning Helix"] == 2
        assert plan.copies["Mountain"] == 10
        assert [(s.role, s.copies) for s in plan.sideboard] == [("graveyard_hate", 15)]
        assert plan.sideboard[0].description.startswith("Sideboard card: ")
        assert plan.side_copies == {"Rest in Peace": 2}

    async def test_colors_played_by_at_least_half_the_lists(self):
        rows = [entry(d, "Shock", 4, "Instant", 1, "R", "burn") for d in (1, 2, 3)]
        rows.append(entry(3, "Get Lost", 1, "Instant", 2, "W", "removal_targeted"))
        plan = await dp.plan_from_decklists(fake_db(rows), "Mono Red", "standard")
        assert plan.colors == ["R"]

    async def test_no_lists(self):
        assert await dp.plan_from_decklists(fake_db([]), "Gone", "standard") is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_plan.py -v`
Expected: the 8 new tests FAIL with `AttributeError: module 'app.services.deck_plan' has no attribute '_slot_sizes'` (and `_merge_small`, `plan_from_decklists`); the 10 Task 2 tests pass.

- [ ] **Step 3: Implement**

In `backend/app/services/deck_plan.py`, replace the import block (from `import logging` through `from app.services import jev`) with:
```python
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Choice

from app.models.card import ROLE_DEFINITIONS
from app.services import jev
```

Append to the end of the file:
```python
MIN_SLOT = 2  # slots under this many copies merge into a neighbor
MAIN_SIZE = 60
SIDEBOARD_SIZE = 15


# One row per decklist entry in the archetype's lists, resolved to a card name
# (decklists list DFCs by front face; cards store "Front // Back").
REFERENCE_SQL = text(f"""
    WITH lists AS (
        SELECT d.id, d.main_deck, d.sideboard
        FROM decklists d JOIN events e ON e.id = d.event_id
        WHERE {RECENT}
          AND lower(trim(d.archetype)) = lower(trim(:archetype))
    ),
    entries AS (
        SELECT l.id AS deck_id, 'main' AS section, x->>'card_name' AS entry, (x->>'quantity')::int AS qty
        FROM lists l CROSS JOIN LATERAL jsonb_array_elements(l.main_deck) x
        UNION ALL
        SELECT l.id, 'side', x->>'card_name', (x->>'quantity')::int
        FROM lists l CROSS JOIN LATERAL jsonb_array_elements(l.sideboard) x
    ),
    card AS (
        SELECT DISTINCT ON (k) k, c.name, c.type_line, c.cmc,
               coalesce(nullif(c.colors, '{{}}'), c.color_identity, '{{}}') AS colors
        FROM (SELECT DISTINCT lower(split_part(entry, ' // ', 1)) AS k FROM entries) n
        JOIN cards c ON lower(split_part(c.name, ' // ', 1)) = n.k
        ORDER BY k, c.name
    ),
    top_role AS (
        SELECT DISTINCT ON (c.name) c.name, r.role
        FROM card_roles r JOIN cards c ON c.id = r.card_id
        WHERE c.name IN (SELECT name FROM card) AND r.role NOT LIKE 'land%'
        ORDER BY c.name, r.confidence DESC NULLS LAST, r.efficiency DESC NULLS LAST
    )
    SELECT en.deck_id, en.section, coalesce(card.name, en.entry) AS name, en.qty,
           card.type_line, card.cmc, card.colors, top_role.role
    FROM entries en
    LEFT JOIN card ON card.k = lower(split_part(en.entry, ' // ', 1))
    LEFT JOIN top_role ON top_role.name = card.name
""")


def is_land(type_line: Optional[str]) -> bool:
    return "Land" in (type_line or "").split(" // ")[0]


def _slot_sizes(rows: Sequence[Any], n_lists: int) -> Dict[Tuple[str, Tuple[int, int]], float]:
    sums: Dict[Tuple[str, Tuple[int, int]], float] = defaultdict(float)
    for r in rows:
        role = r.role or ("creature" if "Creature" in (r.type_line or "") else "noncreature")
        sums[(role, band_of(r.cmc or 0))] += r.qty / n_lists
    return sums


def _merge_small(sizes: Dict[Tuple[str, Tuple[int, int]], float]) -> List[List[Any]]:
    """[role, cmc_min, cmc_max, avg copies] with every slot >= MIN_SLOT: a small
    slot merges into the same role's nearest band (widening its mana range),
    else into the same band's largest slot, else into the largest slot."""
    slots = [[role, lo, hi, avg] for (role, (lo, hi)), avg in sizes.items()]
    while len(slots) > 1:
        small = min(slots, key=lambda s: (s[3], s[0], s[1]))
        if small[3] >= MIN_SLOT:
            break
        others = [s for s in slots if s is not small]
        same_role = [s for s in others if s[0] == small[0]]
        if same_role:
            target = min(same_role, key=lambda s: (abs(s[1] - small[1]), s[1]))
            target[1], target[2] = min(target[1], small[1]), max(target[2], small[2])
        else:
            same_band = [s for s in others if (s[1], s[2]) == (small[1], small[2])]
            target = max(same_band or others, key=lambda s: s[3])
        target[3] += small[3]
        slots.remove(small)
    return slots


def _to_slots(sizes: Dict[Tuple[str, Tuple[int, int]], float], total: int, prefix: str = "") -> List[Slot]:
    merged = _merge_small(sizes)
    if not merged or total <= 0:
        return []
    counts = largest_remainder([s[3] for s in merged], total)
    return [Slot(role, lo, hi, n, prefix + describe(role, lo, hi))
            for (role, lo, hi, _), n in zip(merged, counts) if n > 0]


def _avg_copies(rows: Sequence[Any]) -> Dict[str, int]:
    """Average copies per card across the lists that play it, rounded, at least 1."""
    per: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        per[r.name].append(r.qty)
    return {name: max(1, round(sum(q) / len(q))) for name, q in per.items()}


async def plan_from_decklists(db: AsyncSession, archetype: str, format: str) -> Optional[Plan]:
    """Slot plan from the archetype's decklists in the window; None without lists."""
    rows = (await db.execute(REFERENCE_SQL, {
        "format": format, "archetype": archetype})).all()
    n_lists = len({r.deck_id for r in rows})
    if not n_lists:
        return None
    main = [r for r in rows if r.section == "main"]
    side = [r for r in rows if r.section == "side"]
    lands = [r for r in main if is_land(r.type_line)]
    spells = [r for r in main if not is_land(r.type_line)]
    land_count = round(sum(r.qty for r in lands) / n_lists)
    nonbasic = round(sum(r.qty for r in lands if not (r.type_line or "").startswith("Basic")) / n_lists)
    # A color counts when at least half the lists play it (splashes in one list don't).
    lists_with = {c: len({r.deck_id for r in spells if c in (r.colors or [])}) for c in WUBRG}
    return Plan(
        slots=_to_slots(_slot_sizes(spells, n_lists), MAIN_SIZE - land_count),
        lands=land_count,
        nonbasic_lands=min(nonbasic, land_count),
        sideboard=_to_slots(_slot_sizes(side, n_lists), SIDEBOARD_SIZE, "Sideboard card: ") if side else [],
        copies=_avg_copies(main),
        side_copies=_avg_copies(side),
        colors=[c for c in WUBRG if lists_with[c] * 2 >= n_lists],
        reference=archetype,
    )
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && python3 -m pytest tests/test_deck_plan.py -v && python3 -m pytest -q`
Expected: 18 passed; full suite 236 passed.

- [ ] **Step 5: Check the SQL against the real data (read-only)**

Run:
```bash
cd /Users/robmiller/Projects/mtg-deckbuilder && docker compose exec -T backend python - <<'EOF'
import asyncio
from app.db.session import async_session_factory
from app.services import deck_plan as dp
async def main():
    async with async_session_factory() as db:
        print((await dp.recent_archetypes(db, "standard"))[:5])
        for a in ("Boros Aggro", "Dimir Aggro"):
            p = await dp.plan_from_decklists(db, a, "standard")
            print(a, "lands", p.lands, "nonbasic", p.nonbasic_lands, "colors", p.colors,
                  "main", sum(s.copies for s in p.slots) + p.lands, "side", sum(s.copies for s in p.sideboard))
asyncio.run(main())
EOF
```
Expected (numbers move as the scraper adds events): the top archetypes with counts (for example `('Dimir Aggro', 19)`); `Boros Aggro lands 24 nonbasic 18 colors ['W', 'R'] main 60 side 15`; `Dimir Aggro lands 23 nonbasic 15 colors ['U', 'B'] main 60 side 15`. Every `main` must be 60 and every `side` 15.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/deck_plan.py backend/tests/test_deck_plan.py
git commit -m "Plan deck slots from a reference archetype's decklists

Each nonland card goes to its top card_roles role (creature/noncreature
when untagged) and a mana value band. Average copies per slot, small slots
merged, rounded to 60 minus the average land count. The sideboard is built
the same way, to 15. Decklist names match cards by front face, so DFCs join.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 4: Brew plan from the LLM, with default plans

**Files:**
- Modify: `backend/app/services/deck_plan.py` (imports; append)
- Test: `backend/tests/test_deck_plan.py` (append)

**Interfaces:**
- Consumes: `Slot`, `Plan`, `describe`, `largest_remainder`, `MAIN_SIZE`; `llm.is_configured()`, `llm.complete(system, user, max_tokens)` (synchronous); `CARD_ROLES`.
- Produces:
  - `TYPE_ROLES = ("creature", "noncreature")`, `SPELL_ROLES` (non-land `CARD_ROLES`), `LAND_COUNTS`, `DEFAULT_PLANS` (`aggro`, `midrange`, `control`), `PLAN_SYSTEM`
  - `lands_for(archetype_hint) -> int` (24 control, 23 midrange, else 22)
  - `parse_llm_plan(content, lands) -> Optional[List[Slot]]` (scaled to `60 - lands`; `None` if anything is invalid)
  - `default_plan(archetype_hint) -> List[Slot]` (unknown hints use `midrange`)
  - `async plan_with_llm(request_text, colors, archetype_hint) -> Plan` (no sideboard, `nonbasic_lands=None`, `reference=None`)

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_deck_plan.py`:
```python
LLM_PLAN = """Here you go:
[{"role": "threat_cheap", "cmc_min": 1, "cmc_max": 2, "type_contains": "Creature", "copies": 16,
  "description": "Cheap attackers"},
 {"role": "burn", "cmc_min": 1, "cmc_max": 3, "type_contains": null, "copies": 12, "description": "Burn"},
 {"role": "creature", "cmc_min": 3, "cmc_max": 4, "copies": 6, "description": "Top end"}]"""


class TestParseLlmPlan:
    def test_valid_plan_is_scaled_to_the_nonland_count(self):
        slots = dp.parse_llm_plan(LLM_PLAN, lands=22)
        assert [(s.role, s.cmc_min, s.cmc_max, s.type_contains) for s in slots] == [
            ("threat_cheap", 1, 2, "Creature"), ("burn", 1, 3, None), ("creature", 3, 4, None)]
        assert sum(s.copies for s in slots) == 38
        assert slots[0].description == "Cheap attackers"

    @pytest.mark.parametrize("bad", [
        "no json here",
        "[]",
        '[{"role": "land_basic", "copies": 4}]',
        '[{"role": "wizardry", "copies": 4}]',
        '[{"role": "burn", "copies": 0}]',
        '[{"role": "burn", "copies": "4"}]',
        '[{"role": "burn", "copies": true}]',
        '[{"role": "burn", "copies": 4, "cmc_min": 3, "cmc_max": 1}]',
        '[{"copies": 4}]',
        '["burn"]',
    ])
    def test_invalid_plans(self, bad):
        assert dp.parse_llm_plan(bad, lands=22) is None


class TestPlanWithLlm:
    def test_land_counts(self):
        assert [dp.lands_for(a) for a in ("control", "Midrange", "aggro", "combo", "")] == [24, 23, 22, 22, 22]

    async def test_without_llm_uses_the_archetype_default(self):
        plan = await dp.plan_with_llm("mono-red aggro", ["R"], "aggro")
        assert plan.lands == 22 and plan.sideboard == [] and plan.reference is None
        assert plan.nonbasic_lands is None  # brew rule applies at land time
        assert [s.role for s in plan.slots] == [r for r, *_ in dp.DEFAULT_PLANS["aggro"]]
        assert sum(s.copies for s in plan.slots) == 38

    async def test_llm_plan(self, monkeypatch):
        seen = {}

        def complete(system, user, max_tokens=4096):
            seen["user"] = user
            return LLM_PLAN

        monkeypatch.setattr(dp.llm, "is_configured", lambda: True)
        monkeypatch.setattr(dp.llm, "complete", complete)
        plan = await dp.plan_with_llm("UR control", ["U", "R"], "control")
        assert plan.lands == 24 and sum(s.copies for s in plan.slots) == 36
        assert plan.slots[0].description == "Cheap attackers"
        assert "UR control" in seen["user"] and "U, R" in seen["user"]

    async def test_bad_json_uses_the_default(self, monkeypatch):
        monkeypatch.setattr(dp.llm, "is_configured", lambda: True)
        monkeypatch.setattr(dp.llm, "complete", lambda *a, **k: '[{"role": "nonsense", "copies": 4}]')
        plan = await dp.plan_with_llm("control", ["U"], "control")
        assert [s.role for s in plan.slots] == [r for r, *_ in dp.DEFAULT_PLANS["control"]]
        assert sum(s.copies for s in plan.slots) == 36

    async def test_llm_error_uses_the_default(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(dp.llm, "is_configured", lambda: True)
        monkeypatch.setattr(dp.llm, "complete", boom)
        plan = await dp.plan_with_llm("tempo", ["U"], "tempo")  # no tempo default: midrange
        assert [s.role for s in plan.slots] == [r for r, *_ in dp.DEFAULT_PLANS["midrange"]]
        assert plan.lands == 22
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_plan.py -v`
Expected: the 16 new tests FAIL with `AttributeError: module 'app.services.deck_plan' has no attribute 'parse_llm_plan'` (and `lands_for`, `plan_with_llm`, `DEFAULT_PLANS`).

- [ ] **Step 3: Implement**

In `backend/app/services/deck_plan.py`, replace the import block (from `import logging` through `from app.services import jev`) with:
```python
import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Choice

from app.models.card import CARD_ROLES, ROLE_DEFINITIONS
from app.services import jev, llm
```

Append to the end of the file:
```python
TYPE_ROLES = ("creature", "noncreature")  # fallback roles for untagged cards
SPELL_ROLES = [r for r in CARD_ROLES if not r.startswith("land_")]
LAND_COUNTS = {"control": 24, "midrange": 23}  # brews; everything else 22

# Brew defaults when the LLM plan is missing or invalid: (role, cmc_min, cmc_max, copies).
DEFAULT_PLANS: Dict[str, List[Tuple[str, int, int, int]]] = {
    "aggro": [("threat_cheap", 0, 1, 8), ("threat_cheap", 2, 2, 12), ("threat_midrange", 3, 4, 8),
              ("removal_targeted", 0, 2, 6), ("burn", 0, 3, 4)],
    "midrange": [("threat_cheap", 0, 2, 8), ("threat_midrange", 3, 4, 10), ("threat_finisher", 5, 99, 4),
                 ("removal_targeted", 0, 3, 8), ("card_draw", 2, 4, 4), ("removal_mass", 3, 5, 3)],
    "control": [("removal_targeted", 0, 3, 8), ("counterspell", 1, 3, 6), ("removal_mass", 3, 5, 4),
                ("card_draw", 1, 4, 8), ("threat_finisher", 4, 99, 5), ("threat_midrange", 2, 4, 5)],
}

PLAN_SYSTEM = """You plan the nonland card slots of a 60-card Magic: The Gathering deck.
Reply with only a JSON list. Each item: {"role": one of ROLES, "cmc_min": int,
"cmc_max": int, "type_contains": a type-line word such as "Creature" or "Equipment" or null,
"copies": positive int, "description": one sentence on what the slot does}.
Use 6 to 10 slots. Copies across slots should total about the deck's nonland count.
ROLES: """ + ", ".join(SPELL_ROLES + list(TYPE_ROLES))


def lands_for(archetype_hint: str) -> int:
    return LAND_COUNTS.get((archetype_hint or "").lower(), 22)


def parse_llm_plan(content: str, lands: int) -> Optional[List[Slot]]:
    """Validated slots scaled to MAIN_SIZE - lands, or None if anything is off."""
    try:
        items = json.loads(content[content.index("["): content.rindex("]") + 1])
        slots = []
        for it in items:
            role, copies = it["role"], it["copies"]
            lo, hi = int(it.get("cmc_min", 0)), int(it.get("cmc_max", 99))
            if role not in SPELL_ROLES and role not in TYPE_ROLES:
                return None
            if not isinstance(copies, int) or isinstance(copies, bool) or copies <= 0 or lo > hi:
                return None
            slots.append(Slot(role, lo, hi, copies, str(it.get("description") or describe(role, lo, hi)),
                              it.get("type_contains") or None))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if not slots:
        return None
    for s, n in zip(slots, largest_remainder([s.copies for s in slots], MAIN_SIZE - lands)):
        s.copies = n
    return [s for s in slots if s.copies > 0]


def default_plan(archetype_hint: str) -> List[Slot]:
    spec = DEFAULT_PLANS.get((archetype_hint or "").lower(), DEFAULT_PLANS["midrange"])
    lands = lands_for(archetype_hint)
    counts = largest_remainder([c for *_, c in spec], MAIN_SIZE - lands)
    return [Slot(role, lo, hi, n, describe(role, lo, hi)) for (role, lo, hi, _), n in zip(spec, counts)]


async def plan_with_llm(request_text: str, colors: List[str], archetype_hint: str) -> Plan:
    """Brew plan from one LLM call; the archetype default when that fails. No sideboard."""
    lands = lands_for(archetype_hint)
    slots = None
    if llm.is_configured():
        try:
            content = await asyncio.to_thread(
                llm.complete, PLAN_SYSTEM,
                f"Deck request: {request_text}\nColors: {', '.join(colors) or 'any'}\n"
                f"Archetype: {archetype_hint or 'unspecified'}\nNonland cards: {MAIN_SIZE - lands}",
                1500)
            slots = parse_llm_plan(content, lands)
        except Exception as e:
            logger.warning(f"LLM slot plan failed, using the default: {e}")
    return Plan(slots=slots or default_plan(archetype_hint), lands=lands)
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && python3 -m pytest tests/test_deck_plan.py -v && python3 -m pytest -q`
Expected: 34 passed; full suite 252 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/deck_plan.py backend/tests/test_deck_plan.py
git commit -m "Plan brew decks with one LLM call and fixed defaults

The LLM returns slots as JSON; roles must be card roles or
creature/noncreature and copies positive integers. Copies scale to 60
minus 22/23/24 lands. Invalid output or no LLM uses the aggro, midrange
or control default.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 5: `deck_fill`: slot pools, Jev picks, copy rules and overflow

**Files:**
- Create: `backend/app/services/deck_fill.py`
- Test: `backend/tests/test_deck_fill.py`

**Interfaces:**
- Consumes: `Slot`, `choose`, `RECENT`, `MAX_OPTIONS` (Task 2); `FORMAT_LEGALITY_MAP` from `card_service`.
- Produces (in `app.services.deck_fill`):
  - constants `MAX_COPIES = 4`, `BREW_COPIES`, `ANY_SLOT`, `SLOT_QUESTION`, `CARD_COLORS`; `_plays_cte(sideboard)`, `_role_sql(slot)` (roles: any `CARD_ROLES` role, `creature`, `noncreature`, `any`)
  - `async slot_pool(db, slot, colors, format, chosen, sideboard=False) -> List[row]`; rows have `name, mana_cost, type_line, oracle_text, cmc, plays, color_identity`
  - `option_text(row, format) -> str`, `brew_copies(name, rank) -> int`
  - `async fill_slot(client, slot, pool, state, format, copies_for) -> List[Tuple[str, int]]`, where `copies_for(name, rank) -> int`
  - `class Build` with `copies: Dict[str, int]`, `rows: Dict[str, row]`, `add(name, qty, row=None)`, `total()`, `entries() -> [{card_name, quantity}]`
  - `async fill_slots(db, client, slots, build, colors, format, state, copies_for, exclude=(), sideboard=False) -> int` (copies left unfilled), where `state() -> dict`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_deck_fill.py`:
```python
"""deck_fill: slot pools, Jev picks, copy rules and slot overflow."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.services import deck_fill as df
from app.services.deck_plan import Slot
from tests.jev_fake import FakeJev


def card(name, type_line="Instant", cmc=1, colors="R", roles=(), plays=10, mana_cost="{R}",
         oracle="Deal 2 damage.", identity=None):
    return SimpleNamespace(name=name, type_line=type_line, cmc=cmc, colors=list(colors), roles=list(roles),
                           plays=plays, mana_cost=mana_cost, oracle_text=oracle,
                           color_identity=list(identity if identity is not None else colors))


def fake_db(rows=()):
    result = MagicMock()
    result.all.return_value = list(rows)
    return MagicMock(execute=AsyncMock(return_value=result))


def ranks(*names):
    """FakeJev whose Choice puts `names` first, in order."""
    def answer(state, questions):
        probs = {n: 1.0 - i / 100 for i, n in enumerate(names)}
        return {"pick": {"choice": names[0], "confidence": 0.9, "probabilities": probs}}
    return FakeJev(answer=answer)


def no_copies(name, rank):
    return 0


class TestSlotPool:
    async def test_filters_and_params(self):
        db = fake_db([card("Shock")])
        slot = Slot("burn", 0, 1, 8, "Burn")
        rows = await df.slot_pool(db, slot, ["R", "W"], "standard", ["Lightning Strike"])
        assert [r.name for r in rows] == ["Shock"]
        stmt, params = db.execute.call_args[0]
        sql = str(stmt)
        assert params == {"format": "standard", "legality": "standard", "colors": ["R", "W"],
                          "cmc_min": 0, "cmc_max": 1, "chosen": ["Lightning Strike"],
                          "role": "burn", "type_contains": None}
        assert "e.date >= CURRENT_DATE - 14" in sql  # played in the window only
        assert "jsonb_array_elements(d.main_deck)" in sql
        assert "JOIN played p" in sql  # inner join: unplayed cards never qualify
        assert "c.legalities->>:legality = 'legal'" in sql
        assert "coalesce(nullif(c.colors, '{}'), c.color_identity, '{}') <@ CAST(:colors AS varchar[])" in sql
        assert "NOT LIKE '%Land%'" in sql
        assert "c.cmc BETWEEN :cmc_min AND :cmc_max" in sql
        assert "r.role = :role" in sql
        assert "NOT (c.name = ANY(CAST(:chosen AS varchar[])))" in sql
        assert "ORDER BY plays DESC, c.name" in sql and "LIMIT 255" in sql

    async def test_type_roles_type_contains_and_sideboard_plays(self):
        db = fake_db()
        await df.slot_pool(db, Slot("creature", 2, 2, 4, "x", "Equipment"), ["R"], "modern", [], sideboard=True)
        sql = str(db.execute.call_args[0][0])
        assert "(c.type_line LIKE '%Creature%' OR c.type_line ILIKE '%' || :type_contains || '%')" in sql
        assert "jsonb_array_elements(d.main_deck || d.sideboard)" in sql
        assert db.execute.call_args[0][1]["legality"] == "modern"

    def test_option_text(self):
        row = card("Shock", mana_cost="{R}", type_line="Instant", oracle="Shock deals 2 damage.\nDraw.", plays=7)
        assert df.option_text(row, "standard") == (
            "{R} Instant. Shock deals 2 damage. Draw. [Played in 7 recent Standard tournament decklists]")
        assert len(df.option_text(card("Long", oracle="x" * 500), "standard")) < 300


class TestFillSlot:
    async def test_brew_copies_follow_jev_rank(self):
        pool = [card(n) for n in ("A", "B", "C", "D", "E", "F")]
        jev = ranks("F", "E", "D", "C", "B", "A")
        picks = await df.fill_slot(jev, Slot("burn", 0, 1, 15, "Burn"), pool, {"deck": {}}, "standard",
                                   df.brew_copies)
        assert picks == [("F", 4), ("E", 4), ("D", 3), ("C", 2), ("B", 1), ("A", 1)]
        state, questions, _ = jev.calls[0]
        q = questions["pick"]
        assert q.instructions["slot"] == "Burn" and list(q.criteria) == ["A", "B", "C", "D", "E", "F"]

    async def test_reference_copies_capped_at_four_and_at_the_slot(self):
        pool = [card("A"), card("B"), card("C")]
        ref = {"A": 6, "B": 3}
        picks = await df.fill_slot(ranks("A", "B", "C"), Slot("burn", 0, 1, 6, ""), pool, {}, "standard",
                                   lambda n, r: ref.get(n) or df.brew_copies(n, r))
        assert picks == [("A", 4), ("B", 2)]

    async def test_at_least_one_copy(self):
        picks = await df.fill_slot(ranks("A"), Slot("burn", 0, 1, 2, ""), [card("A"), card("B")], {},
                                   "standard", no_copies)
        assert picks == [("A", 1), ("B", 1)]

    async def test_empty_pool_makes_no_jev_call(self):
        jev = ranks("A")
        assert await df.fill_slot(jev, Slot("burn", 0, 1, 4, ""), [], {}, "standard", df.brew_copies) == []
        assert jev.calls == []

    async def test_pool_runs_out(self):
        picks = await df.fill_slot(ranks("A"), Slot("burn", 0, 1, 12, ""), [card("A"), card("B")], {},
                                   "standard", df.brew_copies)
        assert picks == [("A", 4), ("B", 4)]


CARDS = [
    card("Bolt A", roles=["burn"], cmc=1), card("Bolt B", roles=["burn"], cmc=1),
    card("Bear A", "Creature", cmc=2, roles=["threat_cheap"]), card("Bear B", "Creature", cmc=2, roles=["threat_cheap"]),
    card("Bear C", "Creature", cmc=2, roles=["threat_cheap"]), card("Ogre", "Creature", cmc=3, roles=["threat_midrange"]),
    card("Giant", "Creature", cmc=6, roles=["threat_finisher"]), card("Wrath", "Sorcery", cmc=4, roles=["removal_mass"]),
]


def fake_pool(cards):
    calls = []

    async def pool(db, slot, colors, format, chosen, sideboard=False):
        calls.append((slot.role, slot.copies, list(chosen), sideboard))
        return [c for c in cards
                if (slot.role == "any" or slot.role in c.roles)
                and slot.cmc_min <= c.cmc <= slot.cmc_max and c.name not in chosen
                and set(c.colors) <= set(colors)]
    pool.calls = calls
    return pool


class TestFillSlots:
    async def test_largest_first_and_shortfall_moves_to_same_role(self, monkeypatch):
        pool = fake_pool(CARDS)
        monkeypatch.setattr(df, "slot_pool", pool)
        build = df.Build()
        slots = [Slot("burn", 0, 1, 4, ""), Slot("threat_cheap", 2, 2, 16, ""), Slot("threat_cheap", 3, 4, 2, "")]
        short = await df.fill_slots(None, FakeJev(), slots, build, ["R"], "standard", dict, df.brew_copies)
        # threat_cheap 2-2 has 3 bears (4+4+3 = 11): 5 move to the other threat_cheap slot,
        # whose 3-4 pool has no threat_cheap cards, so 7 move to burn; burn has 2 cards (8).
        assert pool.calls[0][:2] == ("threat_cheap", 16)
        assert pool.calls[1][:2] == ("threat_cheap", 7)
        assert pool.calls[2][:2] == ("burn", 11)
        # 3 still short after the last slot: one catch-all slot over any played card
        assert pool.calls[3][:2] == ("any", 3)
        assert build.copies == {"Bear A": 4, "Bear B": 4, "Bear C": 3, "Bolt A": 4, "Bolt B": 4, "Ogre": 3}
        assert short == 0
        assert slots[1].copies == 16  # the plan's slots are not mutated

    async def test_returns_what_cannot_be_filled(self, monkeypatch):
        monkeypatch.setattr(df, "slot_pool", fake_pool([card("Bolt A", roles=["burn"])]))
        build = df.Build()
        short = await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], build, ["R"], "standard",
                                    dict, df.brew_copies)
        assert build.copies == {"Bolt A": 4} and short == 4

    async def test_exclude_and_sideboard_reach_the_pool(self, monkeypatch):
        pool = fake_pool(CARDS)
        monkeypatch.setattr(df, "slot_pool", pool)
        build = df.Build()
        await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 4, "")], build, ["R"], "standard", dict,
                            df.brew_copies, exclude=["Bolt A"], sideboard=True)
        assert pool.calls[0][2:] == (["Bolt A"], True)
        assert build.copies == {"Bolt B": 4}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_fill.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'app.services.deck_fill'`.

- [ ] **Step 3: Implement**

`backend/app/services/deck_fill.py`:
```python
"""
Jev deck assembly: fill a slot plan with cards played in recent tournaments.

Each slot is one Jev Choice over its pool of played cards; code turns the
ranking into copies. Requested cards always go in.
Design: docs/superpowers/specs/2026-10-06-jev-deck-assembly-design.md
"""

import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Choice

from app.services.card_service import FORMAT_LEGALITY_MAP
from app.services.deck_plan import MAX_OPTIONS, RECENT, Slot, choose

logger = logging.getLogger(__name__)

MAX_COPIES = 4
BREW_COPIES = [4, 4, 3, 2, 1]  # brew copies by Jev rank, then 1
ANY_SLOT = "Any card that makes this deck stronger"
SLOT_QUESTION = ("Which card best fills this slot in the deck described by `deck`? Prefer cards proven "
                 "in recent tournament play (see each option's play count) when they fit the slot.")

# A card's colors; DFCs store colors per face (top-level colors are empty), so
# fall back to color identity.
CARD_COLORS = "coalesce(nullif(c.colors, '{}'), c.color_identity, '{}')"


def _plays_cte(sideboard: bool) -> str:
    deck = "d.main_deck || d.sideboard" if sideboard else "d.main_deck"
    return f"""
        played AS (
            SELECT lower(split_part(x->>'card_name', ' // ', 1)) AS k, COUNT(DISTINCT d.id) AS plays
            FROM decklists d JOIN events e ON e.id = d.event_id
            CROSS JOIN LATERAL jsonb_array_elements({deck}) AS x
            WHERE {RECENT}
            GROUP BY 1
        )"""


def _role_sql(slot: Slot) -> str:
    if slot.role == "any":
        cond = "true"
    elif slot.role == "creature":
        cond = "c.type_line LIKE '%Creature%'"
    elif slot.role == "noncreature":
        cond = "c.type_line NOT LIKE '%Creature%'"
    else:
        cond = "EXISTS (SELECT 1 FROM card_roles r WHERE r.card_id = c.id AND r.role = :role)"
    if slot.type_contains:
        cond = f"({cond} OR c.type_line ILIKE '%' || :type_contains || '%')"
    return cond


async def slot_pool(db: AsyncSession, slot: Slot, colors: List[str], format: str,
                    chosen: Sequence[str], sideboard: bool = False) -> List[Any]:
    """Played, legal, on-color nonland cards that fit the slot, most played first
    (at most MAX_OPTIONS). Rows: name, mana_cost, type_line, oracle_text, cmc, plays."""
    sql = text(f"""
        WITH {_plays_cte(sideboard)}
        SELECT c.name, MAX(c.mana_cost) AS mana_cost, MAX(c.type_line) AS type_line,
               MAX(c.oracle_text) AS oracle_text, MAX(c.cmc) AS cmc, MAX(p.plays) AS plays,
               MAX(c.color_identity) AS color_identity
        FROM cards c JOIN played p ON p.k = lower(split_part(c.name, ' // ', 1))
        WHERE c.legalities->>:legality = 'legal'
          AND split_part(coalesce(c.type_line, ''), ' // ', 1) NOT LIKE '%Land%'
          AND {CARD_COLORS} <@ CAST(:colors AS varchar[])
          AND c.cmc BETWEEN :cmc_min AND :cmc_max
          AND {_role_sql(slot)}
          AND NOT (c.name = ANY(CAST(:chosen AS varchar[])))
        GROUP BY c.name
        ORDER BY plays DESC, c.name
        LIMIT {MAX_OPTIONS}
    """)
    params = {"format": format, "legality": FORMAT_LEGALITY_MAP[format], "colors": list(colors),
              "cmc_min": slot.cmc_min, "cmc_max": slot.cmc_max, "chosen": list(chosen),
              "role": slot.role, "type_contains": slot.type_contains}
    return list((await db.execute(sql, params)).all())


def option_text(row: Any, format: str) -> str:
    oracle = (row.oracle_text or "").replace("\n", " ")[:200]
    return (f"{row.mana_cost or ''} {row.type_line or ''}. {oracle} "
            f"[Played in {row.plays} recent {format.title()} tournament decklists]").strip()


def brew_copies(name: str, rank: int) -> int:
    return BREW_COPIES[rank] if rank < len(BREW_COPIES) else 1


async def fill_slot(client, slot: Slot, pool: Sequence[Any], state: Dict[str, Any], format: str,
                    copies_for: Callable[[str, int], int]) -> List[Tuple[str, int]]:
    """Jev ranks the pool for the slot; code takes copies down that ranking until
    the slot is full. Fewer than slot.copies when the pool runs out."""
    if not pool or slot.copies <= 0:
        return []
    answer = await choose(client, state, Choice(
        instructions={"question": SLOT_QUESTION, "slot": slot.description},
        criteria={r.name: option_text(r, format) for r in pool}))
    probs = answer.probabilities or {}
    ranked = sorted((r.name for r in pool), key=lambda n: -probs.get(n, 0.0))  # stable: ties keep play order
    picks, need = [], slot.copies
    for rank, name in enumerate(ranked):
        if need <= 0:
            break
        q = min(max(1, copies_for(name, rank)), MAX_COPIES, need)
        picks.append((name, q))
        need -= q
    return picks


class Build:
    """A deck in progress: copies by card name plus each card's row."""

    def __init__(self) -> None:
        self.copies: Dict[str, int] = {}
        self.rows: Dict[str, Any] = {}

    def add(self, name: str, qty: int, row: Any = None) -> None:
        self.copies[name] = self.copies.get(name, 0) + qty
        if row is not None:
            self.rows.setdefault(name, row)

    def total(self) -> int:
        return sum(self.copies.values())

    def entries(self) -> List[Dict[str, Any]]:
        return [{"card_name": n, "quantity": q} for n, q in self.copies.items() if q > 0]


async def fill_slots(db: AsyncSession, client, slots: List[Slot], build: Build, colors: List[str],
                     format: str, state: Callable[[], Dict[str, Any]],
                     copies_for: Callable[[str, int], int], exclude: Sequence[str] = (),
                     sideboard: bool = False) -> int:
    """Fill `slots` into `build`, largest remaining slot first. A slot's shortfall
    moves to a remaining slot with the same role, else the largest remaining slot;
    a final shortfall gets one catch-all slot. Returns copies still unfilled."""
    todo = [Slot(**vars(s)) for s in slots if s.copies > 0]  # copies: the plan stays intact
    short = 0
    while todo:
        slot = max(todo, key=lambda s: s.copies)
        todo.remove(slot)
        pool = await slot_pool(db, slot, colors, format, [*build.copies, *exclude], sideboard)
        rows = {r.name: r for r in pool}
        picks = await fill_slot(client, slot, pool, state(), format, copies_for)
        for name, q in picks:
            build.add(name, q, rows[name])
        short = slot.copies - sum(q for _, q in picks)
        if short and todo:
            target = next((s for s in todo if s.role == slot.role), None) or max(todo, key=lambda s: s.copies)
            target.copies += short
            short = 0
    if short:
        catch_all = Slot("any", 0, 99, short, ANY_SLOT)
        pool = await slot_pool(db, catch_all, colors, format, [*build.copies, *exclude], sideboard)
        rows = {r.name: r for r in pool}
        picks = await fill_slot(client, catch_all, pool, state(), format, copies_for)
        for name, q in picks:
            build.add(name, q, rows[name])
        short -= sum(q for _, q in picks)
    return short
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && python3 -m pytest tests/test_deck_fill.py -v && python3 -m pytest -q`
Expected: 11 passed; full suite 263 passed.

- [ ] **Step 5: Check one real pool (read-only)**

Run:
```bash
cd /Users/robmiller/Projects/mtg-deckbuilder && docker compose exec -T backend python - <<'EOF'
import asyncio
from app.db.session import async_session_factory
from app.services import deck_fill as df
from app.services.deck_plan import Slot
async def main():
    async with async_session_factory() as db:
        rows = await df.slot_pool(db, Slot("removal_targeted", 0, 2, 8, "Cheap removal"), ["R", "W"], "standard", [])
        print(len(rows), [(r.name, r.plays) for r in rows[:6]])
        print(df.option_text(rows[0], "standard"))
        rows = await df.slot_pool(db, Slot("any", 0, 99, 4, ""), ["B"], "standard", [])
        print([r.name for r in rows if " // " in r.name][:3])
asyncio.run(main())
EOF
```
Expected (counts move as the scraper adds events): about 20 red or white cheap removal spells, most played first (on 2026-10-06: `('Burst Lightning', 45), ('Erode', 39), ('Get Lost', 14), ...`), with no blue, black or green cards; one option line in the spec's format (`{R} Instant. Kicker {4} ... [Played in 45 recent Standard tournament decklists]`); and mono-black DFCs in full "Front // Back" form (on 2026-10-06: `Sephiroth, Fabled SOLDIER // Sephiroth, One-Winged Angel`, `Gumdrop Poisoner // Tempt with Treats`, `Virtue of Persistence // Locthwain Scorn`). Those show the front-face join and the identity color fallback both work.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/deck_fill.py backend/tests/test_deck_fill.py
git commit -m "Fill deck slots with Jev from cards played in the last 14 days

Each slot's pool is the legal, on-color, played nonland cards matching its
role and mana values, most played first, capped at 255. One Jev Choice
ranks it; code takes copies down the ranking (reference averages or
4/4/3/2/1 for brews, capped at 4). A short slot passes its remainder to
a same-role slot, then the largest, then one catch-all slot.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 6: Requested cards, lands and basics

**Files:**
- Modify: `backend/app/services/deck_fill.py` (imports; append)
- Test: `backend/tests/test_deck_fill.py` (import line; append)

**Interfaces:**
- Consumes: `Build`, `fill_slot`, `_plays_cte`, `CARD_COLORS`, `MAX_COPIES` (Task 5); `Plan`, `Slot`, `is_land`, `largest_remainder`, `WUBRG` (Tasks 2-3).
- Produces:
  - constants `BASICS`, `BREW_NONBASICS`, `LAND_SLOT`; `REQUESTED_SQL`
  - `async land_pool(db, colors, format, chosen) -> List[row]`
  - `async reserve_requested(db, names, plan, format, build) -> List[str]`: adds requested cards to `build`, shrinks `plan.slots` / `plan.lands` in place, and returns the requested cards' colors in WUBRG order
  - `pips(rows) -> Dict[str, int]`, `split_basics(counts, colors, n) -> Dict[basic name, int]`, `brew_nonbasics(colors) -> int`
  - `async fill_lands(db, client, build, plan, colors, format, state, copies_for) -> None`: adds exactly `plan.lands` lands to `build`

- [ ] **Step 1: Write the failing tests**

In `backend/tests/test_deck_fill.py`, change `from app.services.deck_plan import Slot` to:
```python
from app.services.deck_plan import Plan, Slot
```

Append:
```python
def requested_db(*rows):
    """execute().first() returns the next row (None: not found)."""
    results = [MagicMock(first=MagicMock(return_value=r)) for r in rows]
    return MagicMock(execute=AsyncMock(side_effect=results))


def plan(*slots, lands=22, nonbasic=None):
    return Plan(slots=list(slots), lands=lands, nonbasic_lands=nonbasic)


class TestReserveRequested:
    async def test_takes_four_copies_from_the_first_fitting_slot(self):
        p = plan(Slot("burn", 0, 1, 8), Slot("threat_cheap", 2, 2, 12))
        build = df.Build()
        db = requested_db(card("Shock", cmc=1, roles=["burn", "removal_targeted"]))
        colors = await df.reserve_requested(db, ["shock"], p, "standard", build)
        assert build.copies == {"Shock": 4} and colors == ["R"]
        assert [s.copies for s in p.slots] == [4, 12]
        stmt, params = db.execute.call_args[0]
        assert params == {"name": "shock", "legality": "standard"}
        assert "lower(split_part(c.name, ' // ', 1)) = lower(:name)" in str(stmt)  # front face finds a DFC
        assert df.CARD_COLORS + " AS colors" in str(stmt)

    async def test_legendary_is_one_copy_and_overflow_cuts_the_largest_slot(self):
        p = plan(Slot("burn", 0, 1, 2), Slot("threat_cheap", 2, 2, 12), Slot("removal_targeted", 0, 2, 6))
        build = df.Build()
        db = requested_db(card("Sephiroth, Fabled SOLDIER // Sephiroth, One-Winged Angel",
                               "Legendary Creature — Human // Legendary Creature — Angel", cmc=3,
                               colors="B", mana_cost=None, roles=["threat_midrange"]),  # SQL: identity
                          card("Big Burn", cmc=1, roles=["burn"]))
        colors = await df.reserve_requested(db, ["Sephiroth, Fabled SOLDIER", "Big Burn"], p, "standard", build)
        assert build.copies == {"Sephiroth, Fabled SOLDIER // Sephiroth, One-Winged Angel": 1, "Big Burn": 4}
        # Sephiroth fits no slot (threat_midrange 3): 1 off the largest. Big Burn: 2 from burn, 2 more off the largest.
        assert [s.copies for s in p.slots] == [0, 9, 6]
        assert colors == ["B", "R"]

    async def test_a_land_counts_toward_the_lands(self):
        p = plan(Slot("burn", 0, 1, 38), lands=22)
        build = df.Build()
        db = requested_db(card("Sacred Foundry", "Land — Mountain Plains", cmc=0, colors="RW",
                               mana_cost=None))  # SQL: a land's colors fall back to its identity
        colors = await df.reserve_requested(db, ["Sacred Foundry"], p, "standard", build)
        assert p.lands == 18 and p.slots[0].copies == 38
        assert colors == ["W", "R"]  # an off-color requested card brings its colors in

    async def test_unknown_or_illegal_names_and_repeats_are_skipped(self):
        p = plan(Slot("burn", 0, 1, 8))
        build = df.Build()
        db = requested_db(None, card("Shock"), card("Shock"))
        assert await df.reserve_requested(db, ["Nope", "Shock", "shock"], p, "standard", build) == ["R"]
        assert build.copies == {"Shock": 4} and p.slots[0].copies == 4


class TestLandPool:
    async def test_identity_subset_played_nonbasics(self):
        db = fake_db()
        await df.land_pool(db, ["R", "W"], "standard", ["Sacred Foundry"])
        stmt, params = db.execute.call_args[0]
        sql = str(stmt)
        assert params == {"format": "standard", "legality": "standard", "colors": ["R", "W"],
                          "chosen": ["Sacred Foundry"]}
        assert "coalesce(c.color_identity, '{}') <@ CAST(:colors AS varchar[])" in sql  # {} = colorless passes
        assert "LIKE '%Land%'" in sql and "c.type_line NOT LIKE 'Basic%'" in sql
        assert "JOIN played p" in sql and "LIMIT 255" in sql


class TestBasics:
    def test_pips_count_hybrid_and_fall_back_to_identity(self):
        rows = [card("A", mana_cost="{1}{R}{R}"), card("B", mana_cost="{R/W}{W}"),
                card("DFC", mana_cost=None, identity="B"), card("Artifact", mana_cost="{2}", colors="", identity="")]
        assert df.pips(rows) == {"W": 2, "U": 0, "B": 1, "R": 3, "G": 0}

    def test_split_by_pips_with_one_per_color(self):
        assert df.split_basics({"R": 30, "W": 0}, ["R", "W"], 10) == {"Plains": 1, "Mountain": 9}
        assert df.split_basics({"R": 6, "W": 2}, ["W", "R"], 9) == {"Plains": 3, "Mountain": 6}
        assert df.split_basics({}, ["U", "R"], 4) == {"Island": 2, "Mountain": 2}
        assert df.split_basics({"R": 1}, ["W", "U", "R"], 2) == {"Mountain": 2}  # too few for one each
        assert df.split_basics({"R": 1}, ["R"], 0) == {}

    def test_brew_nonbasics_by_color_count(self):
        assert [df.brew_nonbasics(c) for c in (["R"], ["R", "W"], ["U", "B", "R"], list("WUBRG"))] == [0, 4, 6, 6]


class TestFillLands:
    async def test_reference_nonbasics_then_basics_by_pips(self, monkeypatch):
        lands = [card("Sacred Foundry", "Land — Mountain Plains", 0, "", identity="RW", mana_cost=None),
                 card("Inspiring Vantage", "Land", 0, "", identity="RW", mana_cost=None)]
        monkeypatch.setattr(df, "land_pool", AsyncMock(return_value=lands))
        build = df.Build()
        build.add("Shock", 4, card("Shock", mana_cost="{R}"))
        build.add("Get Lost", 2, card("Get Lost", mana_cost="{1}{W}{W}", colors="W"))
        p = plan(lands=10, nonbasic=6)
        await df.fill_lands(None, ranks("Inspiring Vantage", "Sacred Foundry"), build, p, ["R", "W"],
                            "standard", dict, lambda n, r: 4)
        assert build.copies["Inspiring Vantage"] == 4 and build.copies["Sacred Foundry"] == 2
        assert build.copies["Mountain"] + build.copies["Plains"] == 4
        assert build.total() == 6 + 10

    async def test_requested_nonbasic_counts_and_mono_color_brew_gets_only_basics(self, monkeypatch):
        pool = AsyncMock(return_value=[])
        monkeypatch.setattr(df, "land_pool", pool)
        build = df.Build()
        build.add("Shock", 4, card("Shock"))
        await df.fill_lands(None, FakeJev(), build, plan(lands=22), ["R"], "standard", dict, df.brew_copies)
        pool.assert_not_called()  # brew rule: 0 nonbasics for one color
        assert build.copies == {"Shock": 4, "Mountain": 22}

        build = df.Build()
        build.add("Sacred Foundry", 4, card("Sacred Foundry", "Land", 0, "", identity="RW", mana_cost=None))
        await df.fill_lands(None, FakeJev(), build, plan(lands=18), ["R", "W"], "standard", dict, df.brew_copies)
        pool.assert_not_called()  # 2-color brew wants 4 nonbasics; the requested Foundry already is 4
        assert build.copies["Mountain"] + build.copies["Plains"] == 18
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_fill.py -v`
Expected: the 10 new tests FAIL with `AttributeError: module 'app.services.deck_fill' has no attribute 'reserve_requested'` (and `land_pool`, `pips`, `split_basics`, `brew_nonbasics`, `fill_lands`); the 11 Task 5 tests pass.

- [ ] **Step 3: Implement**

In `backend/app/services/deck_fill.py`, replace the import block (from `import logging` through `from app.services.deck_plan import ...`) with:
```python
import logging
import re
from typing import Any, Callable, Dict, List, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Choice

from app.services.card_service import FORMAT_LEGALITY_MAP
from app.services.deck_plan import MAX_OPTIONS, RECENT, WUBRG, Plan, Slot, choose, is_land, largest_remainder
```

Append to the end of the file:
```python
BASICS = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
BREW_NONBASICS = {1: 0, 2: 4}  # by deck color count; 3+ colors: 6
LAND_SLOT = "a land for this deck's mana: fixing for its colors, or utility"


async def land_pool(db: AsyncSession, colors: List[str], format: str, chosen: Sequence[str]) -> List[Any]:
    """Played, legal nonbasic lands whose color identity fits the deck (colorless
    included), most played first."""
    # ponytail: fetchlands have identity {} and pass for any colors; Jev's slot judgment is the only guard
    sql = text(f"""
        WITH {_plays_cte(False)}
        SELECT c.name, MAX(c.mana_cost) AS mana_cost, MAX(c.type_line) AS type_line,
               MAX(c.oracle_text) AS oracle_text, MAX(c.cmc) AS cmc, MAX(p.plays) AS plays,
               MAX(c.color_identity) AS color_identity
        FROM cards c JOIN played p ON p.k = lower(split_part(c.name, ' // ', 1))
        WHERE c.legalities->>:legality = 'legal'
          AND split_part(coalesce(c.type_line, ''), ' // ', 1) LIKE '%Land%'
          AND c.type_line NOT LIKE 'Basic%'
          AND coalesce(c.color_identity, '{{}}') <@ CAST(:colors AS varchar[])
          AND NOT (c.name = ANY(CAST(:chosen AS varchar[])))
        GROUP BY c.name
        ORDER BY plays DESC, c.name
        LIMIT {MAX_OPTIONS}
    """)
    params = {"format": format, "legality": FORMAT_LEGALITY_MAP[format], "colors": list(colors),
              "chosen": list(chosen)}
    return list((await db.execute(sql, params)).all())


REQUESTED_SQL = text(f"""
    SELECT c.name, c.mana_cost, c.type_line, c.oracle_text, c.cmc, c.color_identity,
           {CARD_COLORS} AS colors,
           ARRAY(SELECT DISTINCT r.role FROM card_roles r JOIN cards c2 ON c2.id = r.card_id
                 WHERE c2.name = c.name) AS roles
    FROM cards c
    WHERE (lower(c.name) = lower(:name) OR lower(split_part(c.name, ' // ', 1)) = lower(:name))
      AND c.legalities->>:legality = 'legal'
    ORDER BY c.name
    LIMIT 1
""")


def _fits(slot: Slot, row: Any) -> bool:
    role_ok = (slot.role in (row.roles or [])
               or (slot.role == "creature" and "Creature" in (row.type_line or ""))
               or (slot.role == "noncreature" and "Creature" not in (row.type_line or "")))
    return role_ok and slot.cmc_min <= (row.cmc or 0) <= slot.cmc_max


def _shrink_largest(slots: List[Slot], n: int) -> None:
    while n > 0 and any(s.copies for s in slots):
        largest = max(slots, key=lambda s: s.copies)
        cut = min(n, largest.copies)
        largest.copies -= cut
        n -= cut


async def reserve_requested(db: AsyncSession, names: Sequence[str], plan: Plan, format: str,
                            build: Build) -> List[str]:
    """Put each requested card in `build` (4 copies, 1 if Legendary) and take its
    space out of the plan: a land from the land counts, a spell from the first
    slot it fits (any excess from the largest slot). Returns the requested
    cards' colors. Unknown or format-illegal names are skipped."""
    colors: set = set()
    for name in names:
        row = (await db.execute(REQUESTED_SQL, {"name": name, "legality": FORMAT_LEGALITY_MAP[format]})).first()
        if row is None:
            logger.warning(f"[ASSEMBLY] Requested card not found or not legal in {format}: {name}")
            continue
        if row.name in build.copies:
            continue
        qty = 1 if "Legendary" in (row.type_line or "") else MAX_COPIES
        build.add(row.name, qty, row)
        colors.update(row.colors or [])
        if is_land(row.type_line):
            plan.lands = max(0, plan.lands - qty)
            continue
        slot = next((s for s in plan.slots if s.copies and _fits(s, row)), None)
        excess = qty
        if slot is not None:
            excess = max(0, qty - slot.copies)
            slot.copies = max(0, slot.copies - qty)
        _shrink_largest(plan.slots, excess)
    return [c for c in WUBRG if c in colors]


def pips(rows: Sequence[Any]) -> Dict[str, int]:
    """Colored mana symbols across the cards' mana costs (hybrid counts both
    colors). A card without a mana cost (DFCs store it per face) counts each
    color of its identity once."""
    out = {c: 0 for c in WUBRG}
    for r in rows:
        symbols = re.findall(r"\{([^}]*)\}", r.mana_cost or "")
        letters = [ch for s in symbols for ch in s if ch in WUBRG] if symbols else [
            ch for ch in (r.color_identity or "") if ch in WUBRG]
        for ch in letters:
            out[ch] += 1
    return out


def split_basics(counts: Dict[str, int], colors: Sequence[str], n: int) -> Dict[str, int]:
    """n basics split by pip share, at least 1 per deck color when n allows."""
    colors = [c for c in WUBRG if c in colors]
    if n <= 0 or not colors:
        return {}
    weights = [counts.get(c, 0) or 0 for c in colors]
    if not any(weights):
        weights = [1] * len(colors)
    floor = 1 if n >= len(colors) else 0
    extra = largest_remainder(weights, n - floor * len(colors))
    return {BASICS[c]: floor + e for c, e in zip(colors, extra) if floor + e > 0}


def brew_nonbasics(colors: Sequence[str]) -> int:
    return BREW_NONBASICS.get(len(colors), 6)


def _nonbasic_lands(build: Build) -> int:
    return sum(q for n, q in build.copies.items() if n in build.rows
               and is_land(build.rows[n].type_line) and not (build.rows[n].type_line or "").startswith("Basic"))


async def fill_lands(db: AsyncSession, client, build: Build, plan: Plan, colors: List[str], format: str,
                     state: Callable[[], Dict[str, Any]], copies_for: Callable[[str, int], int]) -> None:
    """Add plan.lands lands (requested lands already took their share): nonbasics
    by Jev pick up to the nonbasic count, then basics split by the spells' pips."""
    target = plan.nonbasic_lands if plan.nonbasic_lands is not None else brew_nonbasics(colors)
    need = max(0, min(target - _nonbasic_lands(build), plan.lands))
    picked = 0
    if need:
        pool = await land_pool(db, colors, format, list(build.copies))
        rows = {r.name: r for r in pool}
        for name, q in await fill_slot(client, Slot("land", 0, 99, need, LAND_SLOT), pool, state(),
                                       format, copies_for):
            build.add(name, q, rows[name])
            picked += q
    spells = [r for r in build.rows.values() if not is_land(r.type_line)]
    for name, q in split_basics(pips(spells), colors, plan.lands - picked).items():
        build.add(name, q)
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && python3 -m pytest tests/test_deck_fill.py -v && python3 -m pytest -q`
Expected: 21 passed; full suite 273 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/deck_fill.py backend/tests/test_deck_fill.py
git commit -m "Reserve requested cards and build the mana base

Requested cards go in first (4 copies, 1 if Legendary) and take their
space from the first slot they fit, or from the land count for a land;
their colors join the deck's. Nonbasic lands are a Jev pick from played
lands within the deck's color identity; basics fill the rest, split by
the spells' colored pips with at least one per color.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 7: `assemble`: the whole deck, sideboard and summary

**Files:**
- Modify: `backend/app/services/deck_fill.py` (imports; append)
- Test: `backend/tests/test_deck_fill.py` (imports; append)

**Interfaces:**
- Consumes: everything from Tasks 2-6; `jev.session(client)`; `llm.is_configured()`, `llm.complete`.
- Produces:
  - `SIXTY_CARD_FORMATS` (`standard`, `historic`, `modern`, `legacy`), `COLOR_WORDS`, `SUMMARY_SYSTEM`
  - `async summarize(main, side, request_text, reference, colors, format) -> Tuple[name, strategy_summary]`
  - `async assemble(db, request_text, colors, specific_cards, format="standard", include_sideboard=True, archetype="", client=None) -> {"name", "strategy_summary", "main_deck", "sideboard", "reference", "colors"}`. `main_deck` and `sideboard` are `[{card_name, quantity}]`. It raises when Jev is not configured or fails, the format is not 60-card, the format has no recent decklists, or a brew has no colors.

- [ ] **Step 1: Write the failing tests**

In `backend/tests/test_deck_fill.py`, add `import pytest` below `from unittest.mock import AsyncMock, MagicMock` (with a blank line between, as a third-party import group), then append:
```python
class TestSummarize:
    async def test_template_without_llm(self):
        main = df.Build()
        assert await df.summarize(main, df.Build(), "Boros aggro", "Boros Aggro", ["W", "R"], "standard") == (
            "Boros Aggro", "Built from recent Standard tournament lists of Boros Aggro.")
        assert await df.summarize(main, df.Build(), "mono-red burn", None, ["R"], "standard") == (
            "Red Deck", "A Standard brew for: mono-red burn")

    async def test_llm_name_and_summary(self, monkeypatch):
        seen = {}

        def complete(system, user, max_tokens=4096):
            seen["user"] = user
            return 'Sure: {"name": "Red Rush", "strategy_summary": "Attack early."}'

        monkeypatch.setattr(df.llm, "is_configured", lambda: True)
        monkeypatch.setattr(df.llm, "complete", complete)
        main, side = df.Build(), df.Build()
        main.add("Shock", 4)
        side.add("Abrade", 2)
        assert await df.summarize(main, side, "burn", None, ["R"], "standard") == ("Red Rush", "Attack early.")
        assert "4 Shock" in seen["user"] and "Sideboard:\n2 Abrade" in seen["user"]

    async def test_bad_llm_reply_uses_the_template(self, monkeypatch):
        monkeypatch.setattr(df.llm, "is_configured", lambda: True)
        monkeypatch.setattr(df.llm, "complete", lambda *a, **k: '{"name": ""}')
        assert (await df.summarize(df.Build(), df.Build(), "x", None, ["U", "R"], "standard"))[0] == "Blue-Red Deck"


MAIN_POOL = [
    card(f"Bear {i}", "Creature", cmc=2, roles=["threat_cheap"], colors="RW"[i % 2]) for i in range(20)
] + [card(f"Bolt {i}", cmc=1, roles=["burn"]) for i in range(10)] + [
    card(f"Hate {i}", "Enchantment", cmc=2, roles=["graveyard_hate"], colors="W") for i in range(8)]
RED = {c.name for c in MAIN_POOL if c.colors == ["R"]}
LANDS = [card("Sacred Foundry", "Land", 0, "", identity="RW", mana_cost=None),
         card("Inspiring Vantage", "Land", 0, "", identity="RW", mana_cost=None)]


def wire(monkeypatch, reference_plan=None, archetypes=(("Boros Aggro", 5),)):
    pool = fake_pool(MAIN_POOL)
    monkeypatch.setattr(df, "recent_archetypes", AsyncMock(return_value=list(archetypes)))
    monkeypatch.setattr(df, "plan_from_decklists", AsyncMock(return_value=reference_plan))
    monkeypatch.setattr(df, "slot_pool", pool)
    monkeypatch.setattr(df, "land_pool", AsyncMock(return_value=LANDS))
    return pool


def reference_plan():
    return Plan(slots=[Slot("threat_cheap", 2, 2, 20, "Bears"), Slot("burn", 0, 1, 16, "Burn")],
                lands=24, nonbasic_lands=8, sideboard=[Slot("graveyard_hate", 0, 3, 15, "Sideboard card: hate")],
                copies={"Bear 0": 4, "Bear 1": 2, "Sacred Foundry": 4}, side_copies={"Hate 0": 3},
                colors=["W", "R"], reference="Boros Aggro")


def total(entries):
    return sum(e["quantity"] for e in entries)


class TestAssemble:
    async def test_reference_deck_is_60_and_15(self, monkeypatch):
        pool = wire(monkeypatch, reference_plan())
        jev = FakeJev(answer=lambda state, q: (
            {"pick": {"choice": "Boros Aggro", "confidence": 0.9, "probabilities": {}}}
            if "none" in q["pick"].criteria else {}))
        deck = await df.assemble(None, "Boros aggro", ["R", "W"], [], "standard", True, "aggro", client=jev)
        assert total(deck["main_deck"]) == 60 and total(deck["sideboard"]) == 15
        assert deck["name"] == "Boros Aggro" and deck["reference"] == "Boros Aggro"
        assert deck["colors"] == ["W", "R"]
        main = {e["card_name"]: e["quantity"] for e in deck["main_deck"]}
        assert main["Bear 0"] == 4 and main["Bear 1"] == 2  # reference copies
        assert main["Sacred Foundry"] == 4
        side = {e["card_name"] for e in deck["sideboard"]}
        assert not side & set(main)  # copies stay within 4 across main and sideboard
        assert pool.calls[-1][3] is True  # sideboard pools count sideboard plays
        state = jev.calls[-1][0]["deck"]
        assert state["plan"] == "Boros Aggro, a current Standard archetype" and state["colors"] == ["W", "R"]

    async def test_brew_has_no_sideboard_and_basic_land_split(self, monkeypatch):
        wire(monkeypatch)
        brew = AsyncMock(return_value=Plan(slots=[Slot("threat_cheap", 2, 2, 18, "Bears"),
                                                  Slot("burn", 0, 1, 20, "Burn")], lands=22))
        monkeypatch.setattr(df, "plan_with_llm", brew)
        jev = FakeJev(answer=lambda state, q: (
            {"pick": {"choice": "none", "confidence": 0.9, "probabilities": {}}}
            if "none" in q["pick"].criteria else {}))
        deck = await df.assemble(None, "mono-red aggro", ["R"], [], "standard", True, "aggro", client=jev)
        main = {e["card_name"]: e["quantity"] for e in deck["main_deck"]}
        assert total(deck["main_deck"]) == 60 and deck["sideboard"] == [] and deck["reference"] is None
        assert main["Mountain"] == 22 and "Sacred Foundry" not in main  # mono-color brew: basics only
        assert {n for n in main if n.startswith("Bear")} <= RED  # on-color only
        brew.assert_awaited_once_with("mono-red aggro", ["R"], "aggro")

    async def test_reference_colors_when_none_were_parsed(self, monkeypatch):
        wire(monkeypatch, reference_plan())
        jev = FakeJev(answer=lambda state, q: (
            {"pick": {"choice": "Boros Aggro", "confidence": 0.9, "probabilities": {}}}
            if "none" in q["pick"].criteria else {}))
        await df.assemble(None, "Build the best deck", [], [], "standard", False, "", client=jev)
        assert jev.calls[-1][0]["deck"]["colors"] == ["W", "R"]

    async def test_not_sixty_card_format(self):
        with pytest.raises(ValueError):
            await df.assemble(None, "x", ["B"], [], "cedh", False, client=FakeJev())

    async def test_no_jev_key(self):
        with pytest.raises(RuntimeError):
            await df.assemble(None, "x", ["R"], [], "standard")  # conftest blanks TYPESAFE_API_KEY

    async def test_no_recent_decklists(self, monkeypatch):
        wire(monkeypatch, archetypes=())
        with pytest.raises(ValueError):
            await df.assemble(None, "x", ["R"], [], "standard", client=FakeJev())

    async def test_brew_without_colors(self, monkeypatch):
        wire(monkeypatch)
        jev = FakeJev(answer=lambda state, q: {"pick": "none"})
        with pytest.raises(ValueError):
            await df.assemble(None, "something fun", [], [], "standard", client=jev)

    async def test_jev_failure_propagates(self, monkeypatch):
        wire(monkeypatch, reference_plan())
        with pytest.raises(Exception):
            await df.assemble(None, "x", ["R"], [], "standard", client=FakeJev(fail=lambda s: True))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_fill.py -v`
Expected: the 11 new tests FAIL with `AttributeError: module 'app.services.deck_fill' has no attribute 'summarize'` (and `assemble`, `recent_archetypes` in `wire`).

- [ ] **Step 3: Implement**

In `backend/app/services/deck_fill.py`, replace the import block (from `import logging` through `from app.services.deck_plan import ...`) with:
```python
import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typesafe_sdk import Choice

from app.services import jev, llm
from app.services.card_service import FORMAT_LEGALITY_MAP
from app.services.deck_plan import (
    MAX_OPTIONS, RECENT, WUBRG, Plan, Slot, choose, choose_reference, is_land, largest_remainder,
    plan_from_decklists, plan_with_llm, recent_archetypes,
)
```

Append to the end of the file:
```python
SIXTY_CARD_FORMATS = {f for f in FORMAT_LEGALITY_MAP if f != "cedh"}
COLOR_WORDS = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
SUMMARY_SYSTEM = """You name and describe a finished Magic: The Gathering deck.
Reply with only JSON: {"name": "a short deck name", "strategy_summary": "2-4 sentences on how the deck plays and wins"}."""


async def summarize(main: Build, side: Build, request_text: str, reference: Optional[str],
                    colors: List[str], format: str) -> Tuple[str, str]:
    """(deck name, strategy summary) from one LLM call; a template without the LLM."""
    name = reference or f"{'-'.join(COLOR_WORDS[c] for c in colors)} Deck"
    summary = (f"Built from recent {format.title()} tournament lists of {reference}." if reference
               else f"A {format.title()} brew for: {request_text}")
    if not llm.is_configured():
        return name, summary
    listing = "\n".join(f"{q} {n}" for n, q in main.copies.items())
    if side.copies:
        listing += "\nSideboard:\n" + "\n".join(f"{q} {n}" for n, q in side.copies.items())
    try:
        content = await asyncio.to_thread(llm.complete, SUMMARY_SYSTEM,
                                          f"Request: {request_text}\n\n{listing}", 600)
        data = json.loads(content[content.index("{"): content.rindex("}") + 1])
        if isinstance(data.get("name"), str) and isinstance(data.get("strategy_summary"), str) \
                and data["name"].strip() and data["strategy_summary"].strip():
            return data["name"].strip(), data["strategy_summary"].strip()
    except Exception as e:
        logger.warning(f"[ASSEMBLY] Summary LLM call failed, using the template: {e}")
    return name, summary


async def assemble(db: AsyncSession, request_text: str, colors: Optional[List[str]],
                   specific_cards: Optional[List[str]], format: str = "standard",
                   include_sideboard: bool = True, archetype: str = "", client=None) -> Dict[str, Any]:
    """Build a deck: reference or brew plan, requested cards, Jev-filled slots,
    lands, sideboard (reference only), summary. Returns {name, strategy_summary,
    main_deck, sideboard} like ai_service.generate_deck, plus the reference
    archetype (None for a brew) and the deck colors. Raises when Jev is not
    configured or fails, the format is not 60-card, the format has no recent
    decklists, or a brew has no colors; the caller falls back to the LLM path."""
    if format not in SIXTY_CARD_FORMATS:
        raise ValueError(f"Jev assembly builds 60-card formats only, not {format}")
    async with jev.session(client) as client:
        if client is None:
            raise RuntimeError("Jev is not configured")
        archetypes = await recent_archetypes(db, format)
        if not archetypes:
            raise ValueError(f"No recent {format} decklists")
        reference = await choose_reference(client, request_text, colors or [], archetypes, format)
        plan = await plan_from_decklists(db, reference, format) if reference else None
        if plan is None:
            reference = None
            plan = await plan_with_llm(request_text, colors or [], archetype)

        main, side = Build(), Build()
        requested_colors = await reserve_requested(db, specific_cards or [], plan, format, main)
        deck_colors = [c for c in WUBRG if c in {*(colors or plan.colors), *requested_colors}]
        if not deck_colors:
            raise ValueError("A brew needs colors")

        def state() -> Dict[str, Any]:
            return {"deck": {
                "plan": f"{reference}, a current {format.title()} archetype" if reference else request_text,
                "request": request_text, "colors": deck_colors,
                "chosen": [f"{q}x {n}" for n, q in main.copies.items()],
                "sideboard": [f"{q}x {n}" for n, q in side.copies.items()],
            }}

        def main_copies(name: str, rank: int) -> int:
            return plan.copies.get(name) or brew_copies(name, rank)

        def side_copies(name: str, rank: int) -> int:
            return plan.side_copies.get(name) or brew_copies(name, rank)

        short = await fill_slots(db, client, plan.slots, main, deck_colors, format, state, main_copies)
        plan.lands += short  # a pool too thin to fill the spells: basics keep the deck at 60
        await fill_lands(db, client, main, plan, deck_colors, format, state, main_copies)
        if include_sideboard and plan.sideboard:
            await fill_slots(db, client, plan.sideboard, side, deck_colors, format, state, side_copies,
                             exclude=list(main.copies), sideboard=True)

    name, summary = await summarize(main, side, request_text, reference, deck_colors, format)
    logger.info(f"[ASSEMBLY] {name}: reference={reference} colors={deck_colors} "
                f"main={main.total()} sideboard={side.total()}")
    return {"name": name, "strategy_summary": summary, "main_deck": main.entries(),
            "sideboard": side.entries(), "reference": reference, "colors": deck_colors}
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && python3 -m pytest tests/test_deck_fill.py -v && python3 -m pytest -q`
Expected: 32 passed; full suite 284 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/deck_fill.py backend/tests/test_deck_fill.py
git commit -m "Assemble whole decks with Jev: plan, slots, lands, sideboard, summary

assemble picks a reference archetype or a brew plan, reserves requested
cards, fills slots and lands with Jev, fills a 15-card sideboard for a
reference deck, and has the LLM name and summarize the finished list (a
template without it). It raises for non-60-card formats, missing Jev, no
recent decklists or a colorless brew, so callers can fall back.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 8: `DeckGenerator` uses Jev assembly, with the LLM path as fallback

**Files:**
- Modify: `backend/app/services/deck_generator.py` (import at line 23; `generate()` lines 104-130)
- Test: `backend/tests/test_deck_generator_assembly.py`

**Interfaces:**
- Consumes: `deck_fill.assemble(db, request_text, colors, specific_cards, format=, include_sideboard=, archetype=)` (Task 7).
- Produces: `DeckGenerator.generate(...)` keeps its signature and response. It sets `deck_data` from `assemble`, or from `ai_service.generate_deck` when `assemble` raises. Card validation, deck validation, Jev fit flags and explanations run unchanged on either result. `/decks/generate` and chat's `generate_full_deck` both go through this method.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_deck_generator_assembly.py`:
```python
"""DeckGenerator.generate builds with Jev assembly and falls back to the LLM path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services import deck_generator as dg
from app.services.deck_generator import DeckGenerator

ASSEMBLED = {
    "name": "Boros Aggro", "strategy_summary": "Built from recent lists.",
    "main_deck": [{"card_name": "Hired Claw", "quantity": 4}],
    "sideboard": [{"card_name": "Rest in Peace", "quantity": 2}],
}


@pytest.fixture
def generator(monkeypatch):
    g = DeckGenerator.__new__(DeckGenerator)
    g.db = MagicMock(commit=AsyncMock(), refresh=AsyncMock())
    g._get_or_create_conversation = AsyncMock(return_value=MagicMock(id="11111111-1111-1111-1111-111111111111"))
    g._get_meta_context = AsyncMock(return_value={})
    g._get_archetype_template = AsyncMock(return_value=None)
    g._validate_and_fix_cards = AsyncMock(side_effect=lambda cards, format="standard": cards)
    g._format_deck_response = MagicMock(return_value="ok")
    g.validator = SimpleNamespace(validate=AsyncMock(return_value=SimpleNamespace(is_valid=True, errors=[])))
    g.ai_service = SimpleNamespace(
        parse_deck_request=AsyncMock(return_value={
            "archetype": "aggro", "colors": ["R", "W"], "colors_specified": True,
            "strategy": "", "specific_cards": ["Hired Claw"]}),
        generate_deck=AsyncMock(return_value={
            "name": "LLM Deck", "strategy_summary": "llm",
            "main_deck": [{"card_name": "Shock", "quantity": 4}], "sideboard": []}),
        generate_card_explanations=AsyncMock(return_value={}),
    )
    monkeypatch.setattr(dg.deck_fit, "review_deck", AsyncMock(return_value=(None, {})))
    return g


async def test_uses_jev_assembly(generator, monkeypatch):
    assemble = AsyncMock(return_value=ASSEMBLED)
    monkeypatch.setattr(dg.deck_fill, "assemble", assemble)
    result = await generator.generate("Build me Boros aggro", include_sideboard=True)
    assemble.assert_awaited_once_with(
        generator.db, "Build me Boros aggro", ["R", "W"], ["Hired Claw"],
        format="standard", include_sideboard=True, archetype="aggro")
    generator.ai_service.generate_deck.assert_not_awaited()
    generator._get_meta_context.assert_not_awaited()
    assert result.deck.name == "Boros Aggro"
    assert result.deck.main_deck == ASSEMBLED["main_deck"] and result.deck.sideboard == ASSEMBLED["sideboard"]
    assert result.strategy_summary == "Built from recent lists."


async def test_falls_back_to_the_llm_path_when_assembly_raises(generator, monkeypatch):
    monkeypatch.setattr(dg.deck_fill, "assemble", AsyncMock(side_effect=RuntimeError("jev down")))
    result = await generator.generate("Build me Boros aggro")
    generator.ai_service.generate_deck.assert_awaited_once()
    assert result.deck.name == "LLM Deck"


async def test_cedh_keeps_the_llm_path(generator):
    # the real assemble rejects cEDH before touching Jev or the database
    generator.ai_service.get_commander_color_identity = AsyncMock(return_value=["B"])
    result = await generator.generate("cEDH Tymna", format="cedh", colors=["W", "B"])
    generator.ai_service.generate_deck.assert_awaited_once()
    assert result.deck.name == "LLM Deck"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_generator_assembly.py -v`
Expected: `test_uses_jev_assembly` and `test_falls_back_to_the_llm_path_when_assembly_raises` FAIL with `AttributeError: module 'app.services.deck_generator' has no attribute 'deck_fill'`; `test_cedh_keeps_the_llm_path` passes (today's behavior).

- [ ] **Step 3: Implement**

In `backend/app/services/deck_generator.py`, change `from app.services import deck_fit` to:
```python
from app.services import deck_fill, deck_fit
```

In `generate()`, replace this block (from `# Get meta data for context` through the closing `)` of the `generate_deck` call):
```python
        # Get meta data for context
        meta_data = await self._get_meta_context(format=format)

        # Get archetype template for role distribution guidance
        archetype_template = await self._get_archetype_template(
            parsed_request.get("archetype", ""),
            format=format,
        )
        if archetype_template:
            logger.info(
                f"Using {archetype_template['archetype_category']} template "
                f"(from {archetype_template['sample_size']} tournament decks)"
            )

        # Generate the deck
        # Preserve colors if user explicitly specified them (prevents tournament data override)
        deck_data = await self.ai_service.generate_deck(
            archetype=parsed_request.get("archetype", ""),
            colors=colors,
            strategy=parsed_request.get("strategy", ""),
            meta_context=meta_data,
            include_sideboard=include_sideboard,
            specific_cards=specific_cards,
            archetype_template=archetype_template,
            format=format,
            preserve_colors=colors_specified,
        )
```
with:
```python
        # Jev assembly builds 60-card decks from recent tournament cards. It raises
        # when Jev is unavailable or fails, or for cEDH/Commander: then the LLM path.
        deck_data = None
        try:
            deck_data = await deck_fill.assemble(
                self.db,
                prompt,
                colors,
                specific_cards,
                format=format,
                include_sideboard=include_sideboard,
                archetype=parsed_request.get("archetype", ""),
            )
        except Exception as e:
            logger.warning(f"[DECK-GEN] Jev assembly unavailable, using the LLM path: {e}")

        if deck_data is None:
            # Get meta data for context
            meta_data = await self._get_meta_context(format=format)

            # Get archetype template for role distribution guidance
            archetype_template = await self._get_archetype_template(
                parsed_request.get("archetype", ""),
                format=format,
            )
            if archetype_template:
                logger.info(
                    f"Using {archetype_template['archetype_category']} template "
                    f"(from {archetype_template['sample_size']} tournament decks)"
                )

            # Preserve colors if user explicitly specified them (prevents tournament data override)
            deck_data = await self.ai_service.generate_deck(
                archetype=parsed_request.get("archetype", ""),
                colors=colors,
                strategy=parsed_request.get("strategy", ""),
                meta_context=meta_data,
                include_sideboard=include_sideboard,
                specific_cards=specific_cards,
                archetype_template=archetype_template,
                format=format,
                preserve_colors=colors_specified,
            )
```

- [ ] **Step 4: Run the tests**

Run: `cd backend && python3 -m pytest tests/test_deck_generator_assembly.py tests/test_deck_generator_explanations.py -v && python3 -m pytest -q`
Expected: 6 passed (the explanations tests still pass: with no key, `assemble` raises and they take the LLM mock); full suite 287 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/deck_generator.py backend/tests/test_deck_generator_assembly.py
git commit -m "Generate decks with Jev assembly, falling back to the LLM

DeckGenerator.generate calls deck_fill.assemble first. When Jev is missing
or fails, or for cEDH/Commander, it logs and runs the existing
ai_service.generate_deck path. Validation, fit flags and explanations are
unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 9: Live eval against real Jev

**Files:**
- Create: `backend/scripts/eval_assembly.py`

**Interfaces:**
- Consumes: `deck_fill.assemble`, `parse_deck_request`, `RECENT`, `BASIC_LANDS`, `async_session_factory`. It runs inside the backend container, which has the database and both keys; it only reads.

- [ ] **Step 1: Write the eval**

`backend/scripts/eval_assembly.py`:
```python
"""
Build three Standard decks with real Jev assembly and check each one: legal,
60 main (and 15 sideboard with a reference), every card played in the last 14
days or requested, nonbasic lands within the deck colors, at most 4 copies.
Prints the lists for human review. Not run in CI.

Usage (inside the backend container, which has the DB and keys):
  docker compose exec -T backend python scripts/eval_assembly.py
"""

import asyncio
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from app.db.session import async_session_factory  # noqa: E402
from app.services import deck_fill  # noqa: E402
from app.services.ai.deck_parsing import parse_deck_request  # noqa: E402
from app.services.deck_plan import RECENT  # noqa: E402
from app.services.deck_validator import BASIC_LANDS  # noqa: E402

# (request, must have a reference archetype)
CASES = [
    ("Build me a Boros aggro deck", True),
    ("mono-red aggro", False),
    ("A black-red midrange deck built around Sephiroth, Fabled SOLDIER", False),
]

CARD_SQL = text("""
    SELECT c.name, bool_or(c.legalities->>'standard' = 'legal') AS legal,
           MAX(c.type_line) AS type_line, MAX(c.color_identity) AS identity
    FROM cards c WHERE c.name = ANY(CAST(:names AS varchar[])) GROUP BY c.name
""")
PLAYED_SQL = text(f"""
    SELECT DISTINCT lower(split_part(x->>'card_name', ' // ', 1))
    FROM decklists d JOIN events e ON e.id = d.event_id
    CROSS JOIN LATERAL jsonb_array_elements(d.main_deck || d.sideboard) AS x
    WHERE {RECENT}
""")


async def check(db, deck, requested, need_reference) -> list:
    main = Counter({e["card_name"]: e["quantity"] for e in deck["main_deck"]})
    side = Counter({e["card_name"]: e["quantity"] for e in deck["sideboard"]})
    names = sorted(set(main) | set(side))
    cards = {r.name: r for r in (await db.execute(CARD_SQL, {"names": names})).all()}
    played = {r[0] for r in (await db.execute(PLAYED_SQL, {"format": "standard"})).all()}
    asked = {n.lower() for n in requested}
    problems = []
    if need_reference and not deck["reference"]:
        problems.append("no reference archetype")
    if sum(main.values()) != 60:
        problems.append(f"main is {sum(main.values())}")
    if deck["reference"] and sum(side.values()) != 15:
        problems.append(f"sideboard is {sum(side.values())}")
    for n in names:
        row = cards.get(n)
        if row is None or not row.legal:
            problems.append(f"not legal: {n}")
            continue
        if n not in BASIC_LANDS and n.lower() not in asked \
                and n.split(" // ")[0].lower() not in played:
            problems.append(f"not played or requested: {n}")
        if "Land" in row.type_line.split(" // ")[0] and n not in BASIC_LANDS \
                and not set(row.identity or []) <= set(deck["colors"]):
            problems.append(f"off-color land: {n} {row.identity}")
        if n not in BASIC_LANDS and main[n] + side[n] > 4:
            problems.append(f"{main[n] + side[n]} copies: {n}")
    return problems


async def main() -> None:
    failed = 0
    async with async_session_factory() as db:
        for request, need_reference in CASES:
            parsed = await parse_deck_request(request, db)
            start = time.time()
            try:
                deck = await deck_fill.assemble(db, request, parsed["colors"], parsed["specific_cards"],
                                                "standard", True, parsed["archetype"])
            except Exception as e:  # in the app this falls back to the LLM path
                failed += 1
                print(f"\n=== {request}\nERROR (Jev assembly raised): {e!r}")
                continue
            took = time.time() - start
            problems = await check(db, deck, parsed["specific_cards"], need_reference)
            failed += bool(problems)
            print(f"\n=== {request}  ({took:.1f}s)")
            print(f"reference={deck['reference']} colors={deck['colors']} "
                  f"requested={parsed['specific_cards']}")
            print(f"{deck['name']}: {deck['strategy_summary']}")
            for e in deck["main_deck"]:
                print(f"  {e['quantity']} {e['card_name']}")
            if deck["sideboard"]:
                print("  Sideboard:")
                for e in deck["sideboard"]:
                    print(f"  {e['quantity']} {e['card_name']}")
            print("OK" if not problems else "PROBLEMS: " + "; ".join(problems))
    print("\nPASS" if not failed else f"\nFAIL ({failed} of {len(CASES)} decks)")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run it against real Jev**

Run: `cd /Users/robmiller/Projects/mtg-deckbuilder && docker compose exec -T backend python scripts/eval_assembly.py`
Expected: three decklists, each followed by `OK`, then `PASS`. "Build me a Boros aggro deck" shows `reference=Boros Aggro colors=['W', 'R']` with 60 main and a 15-card sideboard. "mono-red aggro" shows `reference=None colors=['R']` with 22 Mountains and no sideboard. The Sephiroth request lists the full DFC name under `requested=` and has exactly 1 copy of it. Each deck takes about 9-20 s. (Pre-runs while writing this plan: PASS on three of four runs, 8.7-20.5 s per deck. The fourth run hit one `TypeSafeAPITimeoutError` at 10 s.)

- [ ] **Step 3: Stop and report if it does not pass**

If the only problem is an `ERROR` line from a Jev timeout (`TypeSafeAPITimeoutError`), re-run once. Otherwise, if it prints `FAIL` or any `PROBLEMS:` line, **stop**. Do not tune thresholds, prompts or SQL, and do not continue to Task 10. Report the full output (the decklists and every problem line) to the human and wait.

Also read the three lists as a player would, and note anything odd in the report even when it passes: off-plan cards, a curve that is too high, many one-ofs, or more than one sweeper in an aggro deck.

- [ ] **Step 4: Commit**

```bash
git add backend/scripts/eval_assembly.py
git commit -m "Add the Jev deck assembly eval

Builds Boros aggro (reference), mono-red aggro (brew) and a deck around a
requested DFC with real Jev, and checks legality, 60/15, played-or-requested
cards, on-color nonbasic lands and copy limits. Result: <PASS/FAIL>,
<per-deck seconds>.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```
Fill in the `<...>` parts from the actual run before committing.

---

### Task 10: End-to-end check through `POST /api/decks/generate`

**Files:** none, unless a bug is found.

**Interfaces:**
- Consumes: everything above, live. The backend container bind-mounts `./backend` and runs `uvicorn --reload`, so the code is already live in it. **Do not recreate, restart or `docker compose up` any container.** `docker compose exec` and `curl` are safe.

- [ ] **Step 1: Confirm the stack and the reload**

Run:
```bash
cd /Users/robmiller/Projects/mtg-deckbuilder
docker ps --filter name=spellbook --format '{{.Names}} {{.Status}}'
docker compose exec -T backend sh -c 'test -n "$TYPESAFE_API_KEY" && echo jev-key-set; test -n "$OPENROUTER_API_KEY" && echo llm-key-set'
docker logs spellbook-backend --since 30m 2>&1 | grep -iE "reload|error|traceback" | tail -20
```
Expected: `spellbook-backend`, `spellbook-db` and `spellbook-redis` are `Up`; `jev-key-set` and `llm-key-set` print; the log shows reloads after the code changes and no import errors or tracebacks.

- [ ] **Step 2: A reference deck through the API**

Run:
```bash
S=/private/tmp/claude-501/-Users-robmiller-Projects-mtg-deckbuilder/b8847f7b-d5b1-408b-9720-62bba18fb355/scratchpad
curl -s -w '\n%{time_total}s\n' -X POST http://localhost:8000/api/decks/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Build me a Boros aggro deck", "include_explanations": false}' > $S/gen-boros.txt
tail -1 $S/gen-boros.txt
head -1 $S/gen-boros.txt | python3 -c "
import json, sys
d = json.load(sys.stdin)['deck']
q = lambda es: sum(e['quantity'] for e in es)
print(d['name'], '| main', q(d['main_deck']), '| side', q(d['sideboard']), '| valid', d['is_validated'], d['validation_errors'])
print('; '.join(f\"{e['quantity']} {e['card_name']}\" for e in d['main_deck']))"
docker logs spellbook-backend --since 5m 2>&1 | grep -E "\[ASSEMBLY\]|Jev assembly unavailable"
```
Expected: about 10-20 s; `main 60 | side 15 | valid True None`; a list that matches the Task 9 Boros deck in character; and one log line `[ASSEMBLY] <name>: reference=Boros Aggro colors=['W', 'R'] main=60 sideboard=15`, with no `Jev assembly unavailable` line.

- [ ] **Step 3: A brew through the API**

Run:
```bash
curl -s -X POST http://localhost:8000/api/decks/generate -H 'Content-Type: application/json' \
  -d '{"prompt": "Build me a mono-red aggro deck", "include_explanations": false}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)['deck']
print(d['name'], sum(e['quantity'] for e in d['main_deck']), sum(e['quantity'] for e in d['sideboard']),
      [e['type'] for e in d['validation_errors'] or []])"
docker logs spellbook-backend --since 2m 2>&1 | grep -E "\[ASSEMBLY\]"
```
Expected: `<name> 60 0 ['sideboard_size']` (a brew has no sideboard, so the validator flags only that; see Known consequences), and `[ASSEMBLY] ... reference=None colors=['R'] main=60 sideboard=0`.

- [ ] **Step 4: Report**

Record what each step showed: timings, totals, validation results, the `[ASSEMBLY]` lines, and the Boros list. Fix any bug with a failing test first, then commit the fix separately. If a step shows `Jev assembly unavailable`, quote the logged exception in the report.
