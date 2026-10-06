# Brew Relatives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plan and fill a brew (a request that matches no single current archetype) from the on-color part of the current archetypes Jev judges closest to it, instead of from an LLM slot plan, with the LLM plan kept as the fallback when there are no relatives.

**Architecture:** `deck_plan.relative_candidates` finds the current archetypes whose lists play enough spells in the deck's colors. `deck_plan.choose_relatives` asks one Jev `Noul` per candidate, with the archetype's most-played cards in the state, and keeps those at or above 0.5. `plan_from_decklists` now takes a list of archetypes and an optional color filter, so a single reference passes `[archetype]` and a brew passes its relatives and colors. In `deck_fill`, every pool that was scoped to one archetype is scoped to a set (`lower(trim(d.archetype)) = ANY(:archetypes)`), and `assemble` wires the brew path: relatives' cards first, then the format's, for both the main deck and the sideboard.

**Tech Stack:** FastAPI, SQLAlchemy async (`text()` SQL on Postgres JSONB), `typesafe-sdk` (`Noul`, `Choice`) through `app/services/jev.py`, pytest + pytest-asyncio (`asyncio_mode=auto`).

**Spec:** `docs/superpowers/specs/2026-10-06-brew-relatives-design.md` (builds on `docs/superpowers/specs/2026-10-06-jev-deck-assembly-design.md`)

## Global Constraints

- Hard rule: "No card names, archetype names or land names appear anywhere in production code. That covers constants, prompts, filters and defaults."
- "Related archetypes, their cards, slot sizes, copy counts and land counts all come from the last 14 days of decklists in the database."
- "Jev judges relatedness from the request and from the archetype's actual decklist contents."
- "Only thresholds and counts are constants, such as minimum lists, the relatedness cutoff, the maximum number of relatives and the on-color share." The constants are `MIN_LISTS` (2), `MIN_ON_COLOR_SPELLS` (8), `RELATIVE_THRESHOLD` (0.5), `MAX_RELATIVES` (5), and 25 cards per candidate in the Jev state (`TOP_CARDS`).
- "Tests may use names, because they are fixtures."
- Candidates: "the current archetypes with at least `MIN_LISTS` (2) lists in the window whose lists average at least `MIN_ON_COLOR_SPELLS` (8) main-deck spell copies with colors ⊆ the deck colors, colorless included. Colors use the same `CARD_COLORS` rule as the pools, with front-face matching. If there are no deck colors, there are no candidates."
- Relatives question, verbatim: "Does `archetype` play the same kind of game as the deck the user asked for (for example fast aggro, midrange, control or ramp)? Judge its game plan from its cards; ignore its other colors." One Noul per candidate, "run concurrently through the shared cap and deadline".
- Relatives state: "`{"request", "colors", "archetype": {"name", "lists", "cards"}}`. `cards` holds that archetype's 25 most-played main-deck spell names with their average copies".
- "Relatives are the candidates at or above `RELATIVE_THRESHOLD` (0.5), ranked by probability, capped at `MAX_RELATIVES` (5)."
- Plan from relatives: "counting only main-deck spells whose colors fit the deck colors", "the same top-role and mana-band aggregation, merge rule and largest-remainder rounding as a single reference". Slot sizes are "scaled so the nonland total is 60 minus the land count". "**Land count:** the relatives' average land count, rounded." "**Nonbasic count:** the brew rule, the lower quartile by color count." "The one-color utility-land check still applies." "**Copies:** each card's average copies across the relatives' lists that play it, rounded, at least 1." "**Plan colors:** the deck colors." "**Reference:** a joined label for logs only. It is not shown to Jev as an archetype name."
- Fill: "Main-deck slots are filled from cards played in the relatives' lists first, then the whole format for any shortfall." "Lands are unchanged." Sideboard: "filled from the relatives' sideboard cards first, then the format's, with the existing color filter."
- Pools: "`slot_pool`, `sideboard_pool` and the plays CTE accept a set of archetypes. The SQL becomes `lower(trim(d.archetype)) = ANY(:archetypes)`, bound."
- Jev state for slot picks: "the user's request plus 'built from the current archetypes closest to it'. Archetype names are not used as instructions."
- Failure: "Any Jev failure in the relatives step propagates like other Jev failures. `assemble` then raises and the generator falls back to the LLM path. A brew with zero relatives is not a failure: it uses the LLM plan."
- Out of scope: "Changing single-archetype reference decks." "Fit-flag swapping." "New data sources beyond mtgtop8."
- Unit tests never touch the network. `backend/tests/conftest.py` blanks every API key; Jev is `tests/jev_fake.py` (`FakeJev`), the database is `MagicMock`/`AsyncMock`. The live eval and the end-to-end check run only in their own tasks.
- Backend tests: `cd backend && python3 -m pytest` (baseline before Task 1: 345 passed).
- Every commit message ends with exactly these two lines:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN
  ```

### Resolved spec ambiguities

- **Basic land names.** `deck_fill.BASICS` ("Plains", "Island", "Swamp", "Mountain", "Forest") and the fetchland guard's basic land types stay in code. They are rules of the game, not the metagame, and a deck can't be built without them. The no-names guard test (Task 5) leaves them out.
- **Which deck colors pick the candidates.** The parsed `colors` from the request. Requested cards' colors are only known after `reserve_requested`, which needs a plan first. A brew with no parsed colors gets no candidates and uses the LLM plan, as before. In the live data "Gruul aggro" parses to `[]`, while "a red-green aggro deck" parses to `['R', 'G']`.
- **"On-color share"** is the sum, over the archetype's main-deck spells whose `CARD_COLORS` ⊆ deck colors, of the card's total copies divided by the archetype's list count. Colorless spells count. Decklist names with no card row are skipped. "Spell" means the front face is not a Land (`is_land`'s rule).
- **The 25 cards in the state** are the archetype's most-played main-deck spells of any color, by average copies per list (rounded to 0.1). Off-color cards are included so Jev can see that a two-color archetype is two colors. The state holds them as `{card name: average copies}`.
- **Archetype names** are compared as `lower(trim(...))`. `deck_plan.archetype_keys` lowers, trims and deduplicates them before binding. The bind is `ANY(CAST(:archetypes AS varchar[]))`, like the other array binds in these files.
- **Copies from relatives** use every row of the relatives' lists, lands and off-color cards included. Off-color cards never reach an on-color pool, so their entries are harmless.
- **Brew slot rows with no card row** (decklist names the `cards` table doesn't know) don't count toward brew slots, because they can't be filled. A single reference keeps today's behavior, where they count as `noncreature`.
- **Ties between relatives** keep candidate order (most lists first): the sort is stable.
- **`assemble` returns a new key, `relatives`** (empty for a reference or an LLM-planned brew). `reference` stays `None` for every brew, so the summary and the deck name keep the brew wording. `DeckGenerator` ignores the extra key.
- **The relatives Nouls** use `CHOICE_TIMEOUT` (10 s) and `CHOICE_DEADLINE` (15 s), like `utility_lands`.
- **The eval's "no card outside the relatives' lists unless the slot needed a format top-up (reported)"** prints every nonland card (main or sideboard) that is not in any relative's main deck or sideboard as a format top-up. It is information, not a failure, because the eval can't tell which slot ran short. Lands are left out because lands still come from the whole format.
- **The eval doesn't fail a brew with zero relatives**, since the spec says that is not a failure. It prints the relatives for every brew.

### The live data this plan was checked against (Standard, 2026-10-06)

- 22 archetypes have at least 2 lists in the window.
- For `["R"]`, 10 pass the on-color filter (average on-color spell copies per list in brackets): Boros Dragons (18 lists, 20.4), Izzet Spellementals (11, 8.9), 4/5c Control (9, 10.8), UR Aggro (8, 17.6), Rakdos Aggro (4, 19.3), Boros Dwarves (3, 25.3), Roaming Elementals (3, 11.0), Boros Aggro (2, 14.0), Jeskai Control (2, 13.0), Mardu Aggro (2, 12.5). Jund Sacrifice (2.9) and UW Control (2.3) are out. `["R", "G"]` has 16 candidates and `["B", "R"]` has 17. The query takes under 0.25 s.
- Jev's scores for "mono-red aggro" in one run: Boros Dwarves 0.47, Roaming Elementals 0.41, Mardu Aggro 0.41, Boros Aggro 0.28, Rakdos Aggro 0.27, Boros Dragons 0.24, UR Aggro 0.11, the rest 0.05 or less. Over six runs, mono-red got `['Boros Dwarves']` three times and no relatives three times. "a red-green aggro deck" got `['Roaming Elementals']` in all six runs (once with Boros Dwarves too, at 0.71 and about 0.5). The black-red Sephiroth request got no relatives (best: Mardu Aggro 0.43).

## Review Focus

1. **A relative whose lists mostly play off-color cards** (for mono-red, Boros Dragons plays 20.4 red or colorless spell copies of 36; Jund Sacrifice plays 2.9 of 38). Expected: the archetype is a candidate only with at least 8 on-color copies per list, and only its on-color spells shape the slots and enter the pools, so no white or black card reaches a mono-red deck. Tests: Task 2 `test_only_on_color_spells_make_slots_totalling_60`, Task 3 `test_on_color_share_and_top_cards` (7.9 on-color copies is out).
2. **Deck colors that no archetype covers, or no parsed colors** ("Gruul aggro" parses to no colors). Expected: no candidates and no relatives Jev calls; the brew uses the LLM plan exactly as today, and a brew still without colors raises so the generator falls back. Tests: Task 3 `test_no_colors_no_candidates`; the existing Task 4 assemble tests `test_brew_gets_a_format_sideboard_and_basic_land_split` (no candidates) and `test_brew_without_colors`.
3. **Archetype names that differ only by case or spaces** ("4/5C Control" and "4/5c Control", "Weenie White " with a trailing space). Expected: one candidate, one bound key, and the plan and the pools read the lists of every spelling. Tests: Task 1 `test_archetype_keys_lower_trim_and_dedupe` and `test_a_set_of_archetypes_is_one_bound_list`, Task 3 `test_on_color_share_and_top_cards` (`lower(trim(d.archetype)) AS a`).
4. **Every candidate scores below 0.5** (common in the live data: the black-red request's best is 0.43, mono-red's is about 0.47-0.5). Expected: zero relatives is not an error; the brew uses the LLM plan, format pools, a format sideboard and the plain request as the Jev plan. Tests: Task 3 `test_all_below_the_threshold_is_no_relatives`, Task 4 `test_no_relatives_uses_the_llm_plan`.
5. **Double-faced cards in the on-color filter** (top-level `colors` is `{}`, which would pass as colorless; decklists list them by front face). Expected: candidates and the relatives' plan use `CARD_COLORS` (color identity when `colors` is empty) and the front-face join, so an off-color DFC never counts as on-color. Tests: Task 2 `test_slots_lands_and_copies` (`CARD_COLORS` in `REFERENCE_SQL`), Task 3 `test_on_color_share_and_top_cards`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `backend/app/services/deck_plan.py` | `archetype_keys`; `CARD_COLORS` (moved here); `plan_from_decklists` over a list of archetypes with an optional color filter; `relative_candidates`, `choose_relatives` and their constants | 1, 2, 3 |
| `backend/app/services/deck_fill.py` | Pools and fills scoped to a set of archetypes; `assemble` wires the brew-relatives path | 1, 2, 4 |
| `backend/tests/test_deck_plan.py`, `backend/tests/test_deck_fill.py` | Unit tests | 1-4 |
| `backend/tests/test_no_hardcoded_names.py` (new) | Guard: no fixture card or archetype name in `deck_plan.py` or `deck_fill.py` | 5 |
| `backend/scripts/eval_assembly.py` | Live eval: a second brew, relatives and format top-ups printed | 6 |

---

### Task 1: Pools and fills scoped to a set of archetypes

A reference deck's pools are scoped to one archetype today. This task makes the scope a list, bound once as `ANY(...)`, and keeps the reference path's behavior by passing `[reference]`.

**Files:**
- Modify: `backend/app/services/deck_plan.py` (new `archetype_keys` above `band_of`, line 59)
- Modify: `backend/app/services/deck_fill.py` (imports lines 21-24; `_plays_cte` lines 46-57; `slot_pool` lines 74-99; `_fill_one` lines 158-174; `fill_slots` lines 177-199; `sideboard_pool` lines 436-469; `fill_sideboard` lines 472-490; `assemble` lines 559-564)
- Test: `backend/tests/test_deck_plan.py`, `backend/tests/test_deck_fill.py`

**Interfaces:**
- Consumes: today's `deck_fill` pools and fills.
- Produces:
  - `deck_plan.archetype_keys(names: Sequence[str]) -> List[str]`: lower-cased, trimmed, deduplicated, sorted.
  - `deck_fill.ARCHETYPE_SCOPE`: the SQL fragment `AND lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))`.
  - `slot_pool(db, slot, colors, format, chosen, archetypes: Sequence[str] = ()) -> List[Any]`
  - `sideboard_pool(db, colors, format, chosen, archetypes: Sequence[str] = ()) -> List[Any]`
  - `fill_slots(db, client, slots, build, colors, format, state, copies_for, scope: Sequence[str] = ()) -> int`
  - `fill_sideboard(db, client, main, side, colors, format, state, scope: Sequence[str]) -> None`
  - `_fill_one(db, client, slot, build, colors, format, state, copies_for, scope: Sequence[str], allow_none=True) -> int`
  - Each pool binds `archetypes` only when the list is non-empty; an empty scope means the whole format. `assemble` passes `scope = [reference] if reference else []`. A plain string must never be passed as a scope (it would be read as a list of letters).

- [ ] **Step 1: Write the failing tests**

The new tests pin the bound list and its normalization. The existing fakes (`fake_pool`, `fake_side_pool`) take `archetypes` and record them joined by `", "`, so the existing scope assertions (`"Boros Aggro"`, `None`) keep their meaning. Callers that passed a string scope now pass a list.

In `backend/tests/test_deck_plan.py`, replace:

```python
    def test_largest_remainder_hits_the_total(self):
```

with:

```python
    def test_archetype_keys_lower_trim_and_dedupe(self):
        assert dp.archetype_keys(["Boros Aggro", " boros aggro", "4/5C Control", "4/5c Control "]) == [
            "4/5c control", "boros aggro"]
        assert dp.archetype_keys([]) == []

    def test_largest_remainder_hits_the_total(self):
```

In `backend/tests/test_deck_fill.py`, replace:

```python
    async def test_archetype_scope_is_bound(self):
        db = fake_db()
        await df.slot_pool(db, Slot("burn", 0, 1, 8, "Burn"), ["R"], "standard", [], "Boros Aggro")
        stmt, params = db.execute.call_args[0]
        assert "AND lower(trim(d.archetype)) = lower(trim(:archetype))" in str(stmt)
        assert params["archetype"] == "Boros Aggro"
        db = fake_db()
        await df.slot_pool(db, Slot("burn", 0, 1, 8, "Burn"), ["R"], "standard", [])
        assert "archetype" not in str(db.execute.call_args[0][0])
        assert "archetype" not in db.execute.call_args[0][1]

```

with:

```python
    async def test_archetype_scope_is_bound(self):
        db = fake_db()
        await df.slot_pool(db, Slot("burn", 0, 1, 8, "Burn"), ["R"], "standard", [], ["Boros Aggro"])
        stmt, params = db.execute.call_args[0]
        assert "AND lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))" in str(stmt)
        assert params["archetypes"] == ["boros aggro"]
        db = fake_db()
        await df.slot_pool(db, Slot("burn", 0, 1, 8, "Burn"), ["R"], "standard", [])
        assert "archetype" not in str(db.execute.call_args[0][0])
        assert "archetypes" not in db.execute.call_args[0][1]

    async def test_a_set_of_archetypes_is_one_bound_list(self):
        db = fake_db()
        await df.slot_pool(db, Slot("burn", 0, 1, 8, "Burn"), ["R"], "standard", [],
                           ["Rakdos Aggro", " rakdos aggro ", "4/5C Control", "4/5c Control"])
        stmt, params = db.execute.call_args[0]
        assert params["archetypes"] == ["4/5c control", "rakdos aggro"]  # case and space duplicates merge
        assert str(stmt).count(":archetypes") == 1

```

In `backend/tests/test_deck_fill.py`, replace:

```python
def fake_pool(cards, ref_names=()):
    """Pool over `cards`; with an archetype only `ref_names` qualify."""
    calls, scopes = [], []

    async def pool(db, slot, colors, format, chosen, archetype=None):
        calls.append((slot.role, slot.copies, list(chosen)))
        scopes.append(archetype)
        return [c for c in cards
                if (archetype is None or c.name in ref_names)
```

with:

```python
def fake_pool(cards, ref_names=()):
    """Pool over `cards`; with archetypes only `ref_names` qualify. `scopes`
    records each call's archetypes joined by ", " (None: the whole format)."""
    calls, scopes = [], []

    async def pool(db, slot, colors, format, chosen, archetypes=()):
        calls.append((slot.role, slot.copies, list(chosen)))
        scopes.append(", ".join(archetypes) or None)
        return [c for c in cards
                if (not archetypes or c.name in ref_names)
```

In `backend/tests/test_deck_fill.py`, replace:

```python
        short = await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], build, ["R"], "standard", dict,
                                    df.brew_copies, "Boros Aggro")
        assert pool.scopes == ["Boros Aggro", None]
```

with:

```python
        short = await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], build, ["R"], "standard", dict,
                                    df.brew_copies, ["Boros Aggro"])
        assert pool.scopes == ["Boros Aggro", None]
```

In `backend/tests/test_deck_fill.py`, replace:

```python
        await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], df.Build(), ["R"], "standard", dict,
                            df.brew_copies, "Boros Aggro")
        assert pool.scopes == ["Boros Aggro"]

    async def test_brew_uses_the_format_pool_only(self, monkeypatch):
```

with:

```python
        await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], df.Build(), ["R"], "standard", dict,
                            df.brew_copies, ["Boros Aggro"])
        assert pool.scopes == ["Boros Aggro"]

    async def test_a_set_of_relatives_is_one_scope(self, monkeypatch):
        pool = fake_pool(CARDS, ref_names={"Bolt A"})
        monkeypatch.setattr(df, "slot_pool", pool)
        await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], df.Build(), ["R"], "standard", dict,
                            df.brew_copies, ["Rakdos Aggro", "Boros Aggro"])
        assert pool.scopes == ["Rakdos Aggro, Boros Aggro", None]  # one pool over all of them, then the format

    async def test_brew_uses_the_format_pool_only(self, monkeypatch):
```

In `backend/tests/test_deck_fill.py`, replace:

```python
        short = await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], build, ["R"], "standard", dict,
                                    df.brew_copies, "Boros Aggro")
        assert build.copies == {"Bolt A": 4, "Ogre": 4} and short == 0
```

with:

```python
        short = await df.fill_slots(None, FakeJev(), [Slot("burn", 0, 1, 8, "")], build, ["R"], "standard", dict,
                                    df.brew_copies, ["Boros Aggro"])
        assert build.copies == {"Bolt A": 4, "Ogre": 4} and short == 0
```

In `backend/tests/test_deck_fill.py`, replace:

```python
def fake_side_pool(by_scope):
    calls = []

    async def pool(db, colors, format, chosen, archetype=None):
        calls.append((archetype, list(chosen)))
        return [c for c in by_scope.get(archetype, []) if c.name not in chosen and set(c.colors) <= set(colors)]
```

with:

```python
def fake_side_pool(by_scope):
    """`by_scope` keys: archetypes joined by ", ", or None for the whole format."""
    calls = []

    async def pool(db, colors, format, chosen, archetypes=()):
        key = ", ".join(archetypes) or None
        calls.append((key, list(chosen)))
        return [c for c in by_scope.get(key, []) if c.name not in chosen and set(c.colors) <= set(colors)]
```

In `backend/tests/test_deck_fill.py`, replace:

```python
        rows = await df.sideboard_pool(db, ["R", "W"], "standard", ["Shock"], "Boros Aggro")
```

with:

```python
        rows = await df.sideboard_pool(db, ["R", "W"], "standard", ["Shock"], ["Boros Aggro"])
```

In `backend/tests/test_deck_fill.py`, replace:

```python
                          "chosen": ["Shock"], "archetype": "Boros Aggro"}
```

with:

```python
                          "chosen": ["Shock"], "archetypes": ["boros aggro"]}
```

In `backend/tests/test_deck_fill.py`, replace:

```python
        assert "lower(trim(d.archetype)) = lower(trim(:archetype))" in sql
        assert df.CARD_COLORS
```

with:

```python
        assert "lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))" in sql
        assert df.CARD_COLORS
```

In `backend/tests/test_deck_fill.py`, replace:

```python
        assert "d.archetype" not in str(stmt) and "archetype" not in params
```

with:

```python
        assert "d.archetype" not in str(stmt) and "archetypes" not in params
```

In `backend/tests/test_deck_fill.py`, replace:

```python
                                lambda: {"deck": {"chosen": ["4x Shock"]}}, "Boros Aggro")
```

with:

```python
                                lambda: {"deck": {"chosen": ["4x Shock"]}}, ["Boros Aggro"])
```

In `backend/tests/test_deck_fill.py`, replace:

```python
        await df.fill_sideboard(None, FakeJev(), df.Build(), side, ["W", "U"], "standard", dict, "UW Control")
```

with:

```python
        await df.fill_sideboard(None, FakeJev(), df.Build(), side, ["W", "U"], "standard", dict, ["UW Control"])
```

In `backend/tests/test_deck_fill.py`, replace:

```python
        async def pool(db, slot, colors, format, chosen, archetype=None):
            return {"card_draw"
```

with:

```python
        async def pool(db, slot, colors, format, chosen, archetypes=()):
            return {"card_draw"
```


- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest -q`
Expected: `19 failed, 329 passed`. `test_archetype_keys_lower_trim_and_dedupe` fails with `AttributeError: ... no attribute 'archetype_keys'`, `test_archetype_scope_is_bound` and `test_reference_sideboards` fail on the SQL and the `archetypes` param, and the fill, sideboard and assemble tests fail because the old code passes `None` as a scope to the new fakes.

- [ ] **Step 3: Implement**

In `backend/app/services/deck_plan.py`, replace:

```python
def band_of(cmc: float)
```

with:

```python
def archetype_keys(names: Sequence[str]) -> List[str]:
    """Archetype names as the SQL compares them (lower case, trimmed), deduplicated:
    bind as :archetypes against lower(trim(d.archetype))."""
    return sorted({n.strip().lower() for n in names})


def band_of(cmc: float)
```

In `backend/app/services/deck_fill.py`, replace:

```python
    CHOICE_DEADLINE, CHOICE_TIMEOUT, MAX_OPTIONS, RECENT, WUBRG, Plan, Slot, choose, choose_reference, is_land, largest_remainder,
    plan_from_decklists, plan_with_llm, recent_archetypes,
```

with:

```python
    CHOICE_DEADLINE, CHOICE_TIMEOUT, MAX_OPTIONS, RECENT, WUBRG, Plan, Slot, archetype_keys, choose, choose_reference,
    is_land, largest_remainder, plan_from_decklists, plan_with_llm, recent_archetypes,
```

In `backend/app/services/deck_fill.py`, replace:

```python
def _plays_cte(archetype: Optional[str] = None) -> str:
    """Main-deck plays per card (front face, lower case) in the window, within
    one archetype's lists when `archetype` is given (bind :archetype)."""
    scope = "AND lower(trim(d.archetype)) = lower(trim(:archetype))" if archetype else ""

```

with:

```python
# Decklists of a set of archetypes (bind :archetypes to archetype_keys(...)).
ARCHETYPE_SCOPE = "AND lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))"


def _plays_cte(scoped: bool = False) -> str:
    """Main-deck plays per card (front face, lower case) in the window, within
    the :archetypes lists when `scoped`."""
    scope = ARCHETYPE_SCOPE if scoped else ""

```

In `backend/app/services/deck_fill.py`, replace:

```python
                    chosen: Sequence[str], archetype: Optional[str] = None) -> List[Any]:
    """Played, legal, on-color nonland cards that fit the slot, most played first
    (at most MAX_OPTIONS); plays are counted within `archetype`'s lists when given. Rows: name, mana_cost, type_line, oracle_text, cmc, plays."""
    sql = text(f"""
        WITH {_plays_cte(archetype)}
```

with:

```python
                    chosen: Sequence[str], archetypes: Sequence[str] = ()) -> List[Any]:
    """Played, legal, on-color nonland cards that fit the slot, most played first
    (at most MAX_OPTIONS); plays are counted within the `archetypes`' lists when
    given. Rows: name, mana_cost, type_line, oracle_text, cmc, plays."""
    sql = text(f"""
        WITH {_plays_cte(bool(archetypes))}
```

In `backend/app/services/deck_fill.py`, replace:

```python
              "role": slot.role, "type_contains": slot.type_contains}
    if archetype:
        params["archetype"] = archetype

```

with:

```python
              "role": slot.role, "type_contains": slot.type_contains}
    if archetypes:
        params["archetypes"] = archetype_keys(archetypes)

```

In `backend/app/services/deck_fill.py`, replace:

```python
                    reference: Optional[str], allow_none: bool = True) -> int:
    """Fill one slot: from the reference archetype's cards first, then (for the
    shortfall) from the whole format. Returns copies still unfilled."""
    short = slot.copies
    for archetype in ([reference] if reference else []) + [None]:
        if short <= 0:
            break
        part = Slot(**{**vars(slot), "copies": short})
        pool = await slot_pool(db, part, colors, format, list(build.copies), archetype)
```

with:

```python
                    scope: Sequence[str], allow_none: bool = True) -> int:
    """Fill one slot: from the `scope` archetypes' cards first, then (for the
    shortfall) from the whole format. Returns copies still unfilled."""
    short = slot.copies
    for archetypes in ([scope] if scope else []) + [()]:
        if short <= 0:
            break
        part = Slot(**{**vars(slot), "copies": short})
        pool = await slot_pool(db, part, colors, format, list(build.copies), archetypes)
```

In `backend/app/services/deck_fill.py`, replace:

```python
                     copies_for: Callable[[str, int], int], reference: Optional[str] = None) -> int:
    """Fill `slots` into `build`, largest remaining slot first, each from the
    reference archetype's played cards first (when `reference`), then the
    format's.
```

with:

```python
                     copies_for: Callable[[str, int], int], scope: Sequence[str] = ()) -> int:
    """Fill `slots` into `build`, largest remaining slot first, each from the
    `scope` archetypes' played cards first (the reference, or a brew's
    relatives), then the format's.
```

In `backend/app/services/deck_fill.py`, replace:

```python
        short = await _fill_one(db, client, slot, build, colors, format, state, copies_for, reference)
```

with:

```python
        short = await _fill_one(db, client, slot, build, colors, format, state, copies_for, scope)
```

In `backend/app/services/deck_fill.py`, replace:

```python
                                copies_for, reference, allow_none=False)
```

with:

```python
                                copies_for, scope, allow_none=False)
```

In `backend/app/services/deck_fill.py`, replace:

```python
                         archetype: Optional[str] = None) -> List[Any]:
    """Legal, on-color, nonbasic cards from sideboards in the window: the
    archetype's lists, or every list in the format when archetype is None. Most
    played first, at most MAX_OPTIONS. Rows add `copies`: the card's average
    sideboard copies there, rounded, at least 1."""
    scope = "AND lower(trim(d.archetype)) = lower(trim(:archetype))" if archetype else ""

```

with:

```python
                         archetypes: Sequence[str] = ()) -> List[Any]:
    """Legal, on-color, nonbasic cards from sideboards in the window: the
    `archetypes`' lists, or every list in the format when none are given. Most
    played first, at most MAX_OPTIONS. Rows add `copies`: the card's average
    sideboard copies there, rounded, at least 1."""
    scope = ARCHETYPE_SCOPE if archetypes else ""

```

In `backend/app/services/deck_fill.py`, replace:

```python
              "chosen": list(chosen)}
    if archetype:
        params["archetype"] = archetype
    return list((await db.execute(sql, params)).all())
```

with:

```python
              "chosen": list(chosen)}
    if archetypes:
        params["archetypes"] = archetype_keys(archetypes)
    return list((await db.execute(sql, params)).all())
```

In `backend/app/services/deck_fill.py`, replace:

```python
                         state: Callable[[], Dict[str, Any]], reference: Optional[str]) -> None:
    """Fill `side` to SIDEBOARD_SIZE: one Jev ranking over the reference lists'
    sideboard cards, then (to top up a short pool, or alone for a brew) one over
    the format's sideboard cards.
```

with:

```python
                         state: Callable[[], Dict[str, Any]], scope: Sequence[str]) -> None:
    """Fill `side` to SIDEBOARD_SIZE: one Jev ranking over the `scope` archetypes'
    sideboard cards, then (to top up a short pool, or alone when there is no
    scope) one over the format's sideboard cards.
```

In `backend/app/services/deck_fill.py`, replace:

```python
    for archetype in ([reference] if reference else []) + [None]:
        need = SIDEBOARD_SIZE - side.total()
        if need <= 0:
            return
        pool = await sideboard_pool(db, colors, format, [*main.copies, *side.copies], archetype)
```

with:

```python
    for archetypes in ([scope] if scope else []) + [()]:
        need = SIDEBOARD_SIZE - side.total()
        if need <= 0:
            return
        pool = await sideboard_pool(db, colors, format, [*main.copies, *side.copies], archetypes)
```

In `backend/app/services/deck_fill.py`, replace:

```python
        short = await fill_slots(db, client, plan.slots, main, deck_colors, format, state, main_copies,
                                 reference)
```

with:

```python
        scope = [reference] if reference else []
        short = await fill_slots(db, client, plan.slots, main, deck_colors, format, state, main_copies, scope)
```

In `backend/app/services/deck_fill.py`, replace:

```python
            await fill_sideboard(db, client, main, side, deck_colors, format, state, reference)
```

with:

```python
            await fill_sideboard(db, client, main, side, deck_colors, format, state, scope)
```


- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest -q`
Expected: `348 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/deck_plan.py backend/app/services/deck_fill.py backend/tests/test_deck_plan.py backend/tests/test_deck_fill.py
git commit -m "Scope assembly pools to a set of archetypes

slot_pool, sideboard_pool and the plays CTE took one archetype; they now
take a list, bound once as lower(trim(d.archetype)) = ANY(:archetypes)
after archetype_keys lowers, trims and dedupes the names. fill_slots and
fill_sideboard take that list as their scope. A reference deck passes
[reference], so its behavior is unchanged; brews will pass their
relatives.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 2: Slot plan from a list of archetypes, with an optional color filter

**Files:**
- Modify: `backend/app/services/deck_plan.py` (`RECENT` line 36; `REFERENCE_SQL` lines 123-154; `plan_from_decklists` lines 208-229)
- Modify: `backend/app/services/deck_fill.py` (`CARD_COLORS` moves out, lines 41-43; imports; the `plan_from_decklists` call in `assemble`)
- Test: `backend/tests/test_deck_plan.py`, `backend/tests/test_deck_fill.py`

**Interfaces:**
- Consumes: `archetype_keys` (Task 1).
- Produces:
  - `deck_plan.CARD_COLORS` (moved from `deck_fill`, which now imports it; `df.CARD_COLORS` still resolves).
  - `plan_from_decklists(db, archetypes: Sequence[str], format: str, colors: Optional[Sequence[str]] = None) -> Optional[Plan]`. With `colors=None` (a reference) it behaves as before. With colors (a brew's relatives), slots count only spells whose colors ⊆ `colors`, `nonbasic_lands` is `None` (the brew rule applies in `fill_lands`), `colors` is the deck colors in WUBRG order, and `reference` is the archetypes joined by `" + "` (a log label).
  - `REFERENCE_SQL` binds `:archetypes` (a list from `archetype_keys`) instead of `:archetype`.
  - `assemble` calls `plan_from_decklists(db, [reference], format)`.

- [ ] **Step 1: Write the failing tests**

`entry()` gains a way to build a row with no card match (`colors=None`, as the SQL returns for an unknown name). The relatives fixture is three lists of two archetypes with an off-color card, a colorless card, an unknown name and an off-color dual land.

In `backend/tests/test_deck_plan.py`, replace:

```python
    return SimpleNamespace(deck_id=deck, name=name, qty=qty, type_line=type_line,
                           cmc=cmc, colors=list(colors), role=role)
```

with:

```python
    return SimpleNamespace(deck_id=deck, name=name, qty=qty, type_line=type_line,
                           cmc=cmc, colors=None if colors is None else list(colors), role=role)
```

In `backend/tests/test_deck_plan.py`, replace:

```python
        plan = await dp.plan_from_decklists(db, "Boros Aggro", "standard")
        sql, params = sql_of(db)
        assert params == {"format": "standard", "archetype": "Boros Aggro"}
        assert "lower(trim(d.archetype)) = lower(trim(:archetype))" in sql
```

with:

```python
        plan = await dp.plan_from_decklists(db, ["Boros Aggro"], "standard")
        sql, params = sql_of(db)
        assert params == {"format": "standard", "archetypes": ["boros aggro"]}
        assert "lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))" in sql
        assert dp.CARD_COLORS + " AS colors" in sql  # a DFC's colors fall back to its identity
```

In `backend/tests/test_deck_plan.py`, replace:

```python
        plan = await dp.plan_from_decklists(fake_db(rows), "Mono Red", "standard")
        assert plan.colors == ["R"]

    async def test_no_lists(self):
        assert await dp.plan_from_decklists(fake_db([]), "Gone", "standard") is None
```

with:

```python
        plan = await dp.plan_from_decklists(fake_db(rows), ["Mono Red"], "standard")
        assert plan.colors == ["R"]

    async def test_no_lists(self):
        assert await dp.plan_from_decklists(fake_db([]), ["Gone"], "standard") is None


# Two relatives of a mono-red brew: lists 1-2 one archetype, list 3 another.
RELATIVE_LISTS = [
    entry(1, "Hired Claw", 4, "Creature — Lizard Mercenary", 1, "R", "threat_cheap"),
    entry(1, "Burst Lightning", 4, "Instant", 1, "R", "burn"),
    entry(1, "Lightning Helix", 4, "Instant", 2, "RW", "removal_targeted"),  # off-color for R
    entry(1, "Sacred Foundry", 4, "Land — Mountain Plains", 0, "RW"),
    entry(1, "Mountain", 18, MOUNTAIN, 0),
    entry(2, "Hired Claw", 2, "Creature — Lizard Mercenary", 1, "R", "threat_cheap"),
    entry(2, "Burst Lightning", 4, "Instant", 1, "R", "burn"),
    entry(2, "Lightning Helix", 4, "Instant", 2, "RW", "removal_targeted"),
    entry(2, "Mountain", 20, MOUNTAIN, 0),
    entry(3, "Hired Claw", 4, "Creature — Lizard Mercenary", 1, "R", "threat_cheap"),
    entry(3, "Fear of Missing Out", 4, "Enchantment Creature — Nightmare", 2, "BR", "threat_cheap"),
    entry(3, "Patchwork Beastie", 4, "Artifact Creature — Beast", 1, "", "threat_cheap"),  # colorless fits
    entry(3, "Mystery Card", 2, None, None, colors=None),  # no card row (SQL colors NULL): not counted
    entry(3, "Mountain", 24, MOUNTAIN, 0),
]


class TestPlanFromRelatives:
    async def plan(self):
        db = fake_db(RELATIVE_LISTS)
        plan = await dp.plan_from_decklists(db, ["Boros Aggro", "Rakdos Aggro"], "standard", ["R"])
        return plan, sql_of(db)

    async def test_one_query_over_all_the_relatives(self):
        _, (sql, params) = await self.plan()
        assert params == {"format": "standard", "archetypes": ["boros aggro", "rakdos aggro"]}

    async def test_only_on_color_spells_make_slots_totalling_60(self):
        plan, _ = await self.plan()
        # On-color per list: threat_cheap 0-1 Claw (4+2+4)/3 + Beastie 4/3 = 14/3; burn 0-1 8/3.
        # Helix and Fear of Missing Out are off-color; Mystery Card has no row. 38 nonland, 14:8.
        assert {(s.role, s.cmc_min, s.cmc_max): s.copies for s in plan.slots} == {
            ("threat_cheap", 0, 1): 24, ("burn", 0, 1): 14}
        assert sum(s.copies for s in plan.slots) + plan.lands == dp.MAIN_SIZE

    async def test_lands_copies_colors_and_label_from_the_relatives(self):
        plan, _ = await self.plan()
        assert plan.lands == 22  # (22 + 20 + 24) / 3 lands per list, Sacred Foundry included
        assert plan.nonbasic_lands is None  # brew rule (lower quartile by color count) at land time
        assert plan.copies["Hired Claw"] == 3  # (4 + 2 + 4) / 3 across the lists that play it
        assert plan.copies["Patchwork Beastie"] == 4
        assert plan.colors == ["R"]
        assert plan.reference == "Boros Aggro + Rakdos Aggro"  # a log label, never shown to Jev

    async def test_no_colors_keeps_the_reference_rules(self):
        plan = await dp.plan_from_decklists(fake_db(RELATIVE_LISTS), ["Boros Aggro", "Rakdos Aggro"], "standard")
        assert plan.nonbasic_lands == 1  # 4 Foundries over 3 lists
        assert ("removal_targeted", 2, 2) in {(s.role, s.cmc_min, s.cmc_max) for s in plan.slots}
```

In `backend/tests/test_deck_fill.py`, replace:

```python
        state = jev.calls[-1][0]["deck"]
        assert state["plan"] == "Boros Aggro, a current Standard archetype" and state["colors"] == ["W", "R"]

```

with:

```python
        state = jev.calls[-1][0]["deck"]
        assert state["plan"] == "Boros Aggro, a current Standard archetype" and state["colors"] == ["W", "R"]
        df.plan_from_decklists.assert_awaited_once_with(None, ["Boros Aggro"], "standard")

```


- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest -q`
Expected: `5 failed, 347 passed`: `test_slots_lands_and_copies` (the `:archetypes` param and SQL), three `TestPlanFromRelatives` tests (no `colors` parameter yet), and `test_reference_deck_is_60_and_15` (`assemble` still passes a string).

- [ ] **Step 3: Implement**

In `backend/app/services/deck_plan.py`, replace:

```python
RECENT = f"e.format = :format AND e.date >= CURRENT_DATE - {WINDOW_DAYS}"

```

with:

```python
RECENT = f"e.format = :format AND e.date >= CURRENT_DATE - {WINDOW_DAYS}"
# A card's colors; DFCs store colors per face (top-level colors are empty), so
# fall back to color identity.
CARD_COLORS = "coalesce(nullif(c.colors, '{}'), c.color_identity, '{}')"

```

In `backend/app/services/deck_plan.py`, replace:

```python
# One row per main-deck entry in the archetype's lists, resolved to a card name
# (decklists list DFCs by front face; cards store "Front // Back").
```

with:

```python
# One row per main-deck entry in the :archetypes lists (archetype_keys), resolved
# to a card name (decklists list DFCs by front face; cards store "Front // Back").
```

In `backend/app/services/deck_plan.py`, replace:

```python
          AND lower(trim(d.archetype)) = lower(trim(:archetype))
    ),
```

with:

```python
          AND lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))
    ),
```

In `backend/app/services/deck_plan.py`, replace:

```python
               coalesce(nullif(c.colors, '{{}}'), c.color_identity, '{{}}') AS colors
```

with:

```python
               {CARD_COLORS} AS colors
```

In `backend/app/services/deck_plan.py`, replace:

```python
async def plan_from_decklists(db: AsyncSession, archetype: str, format: str) -> Optional[Plan]:
    """Main-deck slot plan from the archetype's decklists in the window; None
    without lists. Sideboards are not slot-planned (deck_fill.fill_sideboard)."""
    rows = (await db.execute(REFERENCE_SQL, {
        "format": format, "archetype": archetype})).all()
    n_lists = len({r.deck_id for r in rows})
    if not n_lists:
        return None
    lands = [r for r in rows if is_land(r.type_line)]
    spells = [r for r in rows if not is_land(r.type_line)]
    land_count
```

with:

```python
async def plan_from_decklists(db: AsyncSession, archetypes: Sequence[str], format: str,
                              colors: Optional[Sequence[str]] = None) -> Optional[Plan]:
    """Main-deck slot plan from the `archetypes`' decklists in the window; None
    without lists. Sideboards are not slot-planned (deck_fill.fill_sideboard).

    A reference passes [archetype] and no colors. A brew passes its relatives and
    the deck colors: slots then count only spells whose colors fit the deck
    (colorless included), the nonbasic count is left to the brew rule, and the
    plan's colors are the deck colors."""
    rows = (await db.execute(REFERENCE_SQL, {
        "format": format, "archetypes": archetype_keys(archetypes)})).all()
    n_lists = len({r.deck_id for r in rows})
    if not n_lists:
        return None
    lands = [r for r in rows if is_land(r.type_line)]
    spells = [r for r in rows if not is_land(r.type_line)]
    if colors is not None:  # unknown names (colors None) can't be filled, so they don't count
        spells = [r for r in spells if r.colors is not None and set(r.colors) <= set(colors)]
    land_count
```

In `backend/app/services/deck_plan.py`, replace:

```python
        nonbasic_lands=min(nonbasic, land_count),
        copies=_avg_copies(rows),
        colors=[c for c in WUBRG if lists_with[c] * 2 >= n_lists],
        reference=archetype,
    )
```

with:

```python
        nonbasic_lands=None if colors is not None else min(nonbasic, land_count),
        copies=_avg_copies(rows),
        colors=([c for c in WUBRG if c in colors] if colors is not None
                else [c for c in WUBRG if lists_with[c] * 2 >= n_lists]),
        reference=" + ".join(archetypes),
    )
```

In `backend/app/services/deck_fill.py`, replace:

```python
                 "sweeper that kills its own creatures in a creature deck).")

# A card's colors; DFCs store colors per face (top-level colors are empty), so
# fall back to color identity.
CARD_COLORS = "coalesce(nullif(c.colors, '{}'), c.color_identity, '{}')"


# Decklists
```

with:

```python
                 "sweeper that kills its own creatures in a creature deck).")

# Decklists
```

In `backend/app/services/deck_fill.py`, replace:

```python
    CHOICE_DEADLINE, CHOICE_TIMEOUT, MAX_OPTIONS, RECENT, WUBRG, Plan, Slot, archetype_keys, choose, choose_reference,
    is_land, largest_remainder, plan_from_decklists, plan_with_llm, recent_archetypes,
```

with:

```python
    CARD_COLORS, CHOICE_DEADLINE, CHOICE_TIMEOUT, MAX_OPTIONS, RECENT, WUBRG, Plan, Slot, archetype_keys, choose,
    choose_reference, is_land, largest_remainder, plan_from_decklists, plan_with_llm, recent_archetypes,
```

In `backend/app/services/deck_fill.py`, replace:

```python
        plan = await plan_from_decklists(db, reference, format) if reference else None
```

with:

```python
        plan = await plan_from_decklists(db, [reference], format) if reference else None
```


- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest -q`
Expected: `352 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/deck_plan.py backend/app/services/deck_fill.py backend/tests/test_deck_plan.py backend/tests/test_deck_fill.py
git commit -m "Plan a deck from several archetypes' lists, on-color spells only

plan_from_decklists takes a list of archetypes and optional deck colors.
A reference passes [archetype] and no colors and plans as before. With
colors (a brew's relatives), slot sizes count only spells whose colors
fit the deck, colorless included, and the nonbasic count is left to the
brew rule, since relatives may run fixing this deck doesn't need. Land
count and copies are the relatives' averages. CARD_COLORS moves to
deck_plan so both modules share the DFC color rule.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 3: Relative candidates and Jev's relatives

**Files:**
- Modify: `backend/app/services/deck_plan.py` (import `Noul`, line 18; constants, `CANDIDATES_SQL`, `relative_candidates` and `choose_relatives` above `TYPE_ROLES`, line 232)
- Test: `backend/tests/test_deck_plan.py`

**Interfaces:**
- Consumes: `CARD_COLORS` (Task 2), `RECENT`, `MIN_LISTS`, `CHOICE_TIMEOUT`, `CHOICE_DEADLINE`, `jev.ask_many(client, requests, deadline=..., timeout=...)`.
- Produces:
  - Constants `MIN_ON_COLOR_SPELLS = 8`, `RELATIVE_THRESHOLD = 0.5`, `MAX_RELATIVES = 5`, `TOP_CARDS = 25`, `RELATIVE_QUESTION` (the spec's question, verbatim).
  - `relative_candidates(db, colors: Sequence[str], format: str) -> List[Tuple[str, int, List[Tuple[str, float]]]]`: `(archetype, list count, [(card name, average copies per list)])`, most lists first, at most `TOP_CARDS` cards each. `[]` without a query when `colors` is empty.
  - `choose_relatives(client, request_text: str, colors: Sequence[str], candidates) -> List[str]`: archetype names, most likely first. `[]` without a Jev call when there are no candidates. Raises when Jev fails.
  - Jev state per candidate: `{"request": request_text, "colors": [...], "archetype": {"name": ..., "lists": n, "cards": {name: copies}}}`; question id `"relative"`.

- [ ] **Step 1: Write the failing tests**

In `backend/tests/test_deck_plan.py`, replace:

```python
    async def test_no_colors_keeps_the_reference_rules(self):
        plan = await dp.plan_from_decklists(fake_db(RELATIVE_LISTS), ["Boros Aggro", "Rakdos Aggro"], "standard")
        assert plan.nonbasic_lands == 1  # 4 Foundries over 3 lists
        assert ("removal_targeted", 2, 2) in {(s.role, s.cmc_min, s.cmc_max) for s in plan.slots}
```

with:

```python
    async def test_no_colors_keeps_the_reference_rules(self):
        plan = await dp.plan_from_decklists(fake_db(RELATIVE_LISTS), ["Boros Aggro", "Rakdos Aggro"], "standard")
        assert plan.nonbasic_lands == 1  # 4 Foundries over 3 lists
        assert ("removal_targeted", 2, 2) in {(s.role, s.cmc_min, s.cmc_max) for s in plan.slots}


def cand(archetype, lists, name, avg_copies, on_color=True):
    return SimpleNamespace(archetype=archetype, lists=lists, name=name, avg_copies=avg_copies, on_color=on_color)


CANDIDATE_ROWS = [
    # 8 on-color copies per list: exactly MIN_ON_COLOR_SPELLS, so a candidate
    cand("Boros Dragons", 18, "Lightning Helix", 4.0, on_color=False),
    cand("Boros Dragons", 18, "Burst Lightning", 4.0),
    cand("Boros Dragons", 18, "Hired Claw", 3.5),
    cand("Boros Dragons", 18, "Patchwork Beastie", 0.5),  # colorless counts as on-color
    # 7.9 on-color: out, however many lists it has
    cand("Dimir Aggro", 19, "Fear of Missing Out", 7.9),
    cand("Dimir Aggro", 19, "Get Lost", 4.0, on_color=False),
] + [cand("Rakdos Aggro", 4, f"Spell {i}", 1.0) for i in range(30)]  # 30 on-color spells


class TestRelativeCandidates:
    async def test_on_color_share_and_top_cards(self):
        db = fake_db(CANDIDATE_ROWS)
        got = await dp.relative_candidates(db, ["R"], "standard")
        assert [(a, n) for a, n, _ in got] == [("Boros Dragons", 18), ("Rakdos Aggro", 4)]
        # cards are the archetype's most-played spells of any color, so Jev sees off-color ones too
        assert got[0][2] == [("Lightning Helix", 4.0), ("Burst Lightning", 4.0), ("Hired Claw", 3.5),
                             ("Patchwork Beastie", 0.5)]
        assert len(got[1][2]) == dp.TOP_CARDS == 25
        sql, params = sql_of(db)
        assert params == {"format": "standard", "colors": ["R"], "min_lists": 2}
        assert "e.date >= CURRENT_DATE - 14" in sql
        assert "lower(trim(d.archetype)) AS a" in sql  # case and space duplicates are one archetype
        assert "HAVING COUNT(*) >= :min_lists" in sql
        assert "lower(split_part(c.name, ' // ', 1)) = n.k" in sql  # DFCs listed by front face
        assert dp.CARD_COLORS + " AS colors" in sql  # a DFC's colors fall back to its identity
        assert "colors <@ CAST(:colors AS varchar[])" in sql
        assert "NOT LIKE '%Land%'" in sql  # spells only

    async def test_no_colors_no_candidates(self):
        db = fake_db(CANDIDATE_ROWS)
        assert await dp.relative_candidates(db, [], "standard") == []
        db.execute.assert_not_called()


CANDIDATES = [(f"Deck {i}", 10 - i, [(f"Card {i}", 4.0)]) for i in range(7)]


def judges(probabilities):
    return FakeJev(answer=lambda state, q: {"relative": probabilities.get(state["archetype"]["name"], 0.0)})


class TestChooseRelatives:
    async def test_threshold_cap_and_order(self):
        jev = judges({"Deck 0": 0.5, "Deck 1": 0.9, "Deck 2": 0.49, "Deck 3": 0.7, "Deck 4": 0.8,
                      "Deck 5": 0.6, "Deck 6": 0.95})
        got = await dp.choose_relatives(jev, "mono-red aggro", ["R"], CANDIDATES)
        assert got == ["Deck 6", "Deck 1", "Deck 4", "Deck 3", "Deck 5"]  # 0.5 is in but past the cap of 5
        assert len(jev.calls) == len(CANDIDATES)

    async def test_state_holds_the_archetype_cards(self):
        jev = judges({})
        await dp.choose_relatives(jev, "mono-red aggro", ["R"], [("Boros Dragons", 18, [("Burst Lightning", 4.0),
                                                                                         ("Hired Claw", 3.5)])])
        state, questions, kwargs = jev.calls[0]
        assert state == {"request": "mono-red aggro", "colors": ["R"], "archetype": {
            "name": "Boros Dragons", "lists": 18, "cards": {"Burst Lightning": 4.0, "Hired Claw": 3.5}}}
        assert questions["relative"].instructions == dp.RELATIVE_QUESTION
        assert kwargs == {"timeout": dp.CHOICE_TIMEOUT}

    async def test_all_below_the_threshold_is_no_relatives(self):
        assert await dp.choose_relatives(judges({"Deck 0": 0.49}), "x", ["R"], CANDIDATES) == []

    async def test_no_candidates_skips_jev(self):
        jev = judges({})
        assert await dp.choose_relatives(jev, "x", ["R"], []) == [] and jev.calls == []

    async def test_jev_failure_raises(self):
        with pytest.raises(Exception):  # ask_many raises an ExceptionGroup
            await dp.choose_relatives(FakeJev(fail=lambda s: True), "x", ["R"], CANDIDATES)
```


- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_plan.py -q`
Expected: 6 of the 7 new tests fail with `AttributeError: module 'app.services.deck_plan' has no attribute 'relative_candidates'` (or `choose_relatives`). `test_jev_failure_raises` passes already, because `pytest.raises(Exception)` also catches the `AttributeError`; it pins the behavior after Step 3.

- [ ] **Step 3: Implement**

In `backend/app/services/deck_plan.py`, replace:

```python
from typesafe_sdk import Choice

```

with:

```python
from typesafe_sdk import Choice, Noul

```

In `backend/app/services/deck_plan.py`, replace:

```python
TYPE_ROLES = ("creature", "noncreature")
```

with:

```python
MIN_ON_COLOR_SPELLS = 8  # a brew relative's lists average this many on-color main-deck spells
RELATIVE_THRESHOLD = 0.5
MAX_RELATIVES = 5
TOP_CARDS = 25  # cards per candidate shown to Jev
RELATIVE_QUESTION = ("Does `archetype` play the same kind of game as the deck the user asked for "
                     "(for example fast aggro, midrange, control or ramp)? Judge its game plan from its "
                     "cards; ignore its other colors.")

# Per (archetype, main-deck spell) in the window: the archetype's list count, the
# spell's average copies per list, and whether its colors fit :colors (colorless
# included). Archetypes with fewer than :min_lists lists and unknown names are left out.
CANDIDATES_SQL = text(f"""
    WITH lists AS (
        SELECT d.id, lower(trim(d.archetype)) AS a, trim(d.archetype) AS label, d.main_deck
        FROM decklists d JOIN events e ON e.id = d.event_id
        WHERE {RECENT}
          AND coalesce(trim(d.archetype), '') <> ''
    ),
    sizes AS (
        SELECT a, MIN(label) AS archetype, COUNT(*) AS lists FROM lists GROUP BY a HAVING COUNT(*) >= :min_lists
    ),
    entries AS (
        SELECT l.a, x->>'card_name' AS entry, (x->>'quantity')::int AS qty
        FROM lists l JOIN sizes s ON s.a = l.a
        CROSS JOIN LATERAL jsonb_array_elements(l.main_deck) x
    ),
    card AS (
        SELECT DISTINCT ON (k) k, c.name, c.type_line, {CARD_COLORS} AS colors
        FROM (SELECT DISTINCT lower(split_part(entry, ' // ', 1)) AS k FROM entries) n
        JOIN cards c ON lower(split_part(c.name, ' // ', 1)) = n.k
        ORDER BY k, c.name
    )
    SELECT s.archetype, s.lists, card.name, SUM(en.qty)::float / s.lists AS avg_copies,
           bool_and(card.colors <@ CAST(:colors AS varchar[])) AS on_color
    FROM entries en
    JOIN card ON card.k = lower(split_part(en.entry, ' // ', 1))
    JOIN sizes s ON s.a = en.a
    WHERE split_part(coalesce(card.type_line, ''), ' // ', 1) NOT LIKE '%Land%'
    GROUP BY s.archetype, s.lists, card.name
    ORDER BY s.lists DESC, s.archetype, avg_copies DESC, card.name
""")


async def relative_candidates(db: AsyncSession, colors: Sequence[str],
                              format: str) -> List[Tuple[str, int, List[Tuple[str, float]]]]:
    """(archetype, list count, top cards) for each current archetype with at least
    MIN_LISTS lists whose lists average at least MIN_ON_COLOR_SPELLS main-deck
    spell copies with colors within `colors`, most lists first. Top cards are its
    TOP_CARDS most-played main-deck spells, any color, with average copies per
    list. No colors: no candidates."""
    if not colors:
        return []
    rows = (await db.execute(CANDIDATES_SQL, {
        "format": format, "colors": list(colors), "min_lists": MIN_LISTS})).all()
    found: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        a = found.setdefault(r.archetype, {"lists": r.lists, "on_color": 0.0, "cards": []})
        a["on_color"] += r.avg_copies if r.on_color else 0.0
        a["cards"].append((r.name, round(r.avg_copies, 1)))
    return [(name, a["lists"], a["cards"][:TOP_CARDS]) for name, a in found.items()
            if a["on_color"] >= MIN_ON_COLOR_SPELLS]


async def choose_relatives(client, request_text: str, colors: Sequence[str],
                           candidates: Sequence[Tuple[str, int, List[Tuple[str, float]]]]) -> List[str]:
    """The candidates Jev judges close relatives of the requested deck (one Noul
    each, judged from the archetype's cards, not only its name): at or above
    RELATIVE_THRESHOLD, most likely first, at most MAX_RELATIVES. Raises when Jev
    fails."""
    if not candidates:
        return []
    requests = [({"request": request_text, "colors": list(colors),
                  "archetype": {"name": name, "lists": lists, "cards": dict(cards)}},
                 {"relative": Noul(instructions=RELATIVE_QUESTION)}) for name, lists, cards in candidates]
    answers = await jev.ask_many(client, requests, deadline=CHOICE_DEADLINE, timeout=CHOICE_TIMEOUT)
    scored = [(a.nouls["relative"].noul, name) for a, (name, _, _) in zip(answers, candidates)]
    ranked = sorted((s for s in scored if s[0] >= RELATIVE_THRESHOLD), key=lambda s: -s[0])  # stable
    return [name for _, name in ranked[:MAX_RELATIVES]]


TYPE_ROLES = ("creature", "noncreature")
```


- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest -q`
Expected: `359 passed`.

- [ ] **Step 5: Check the candidates query against the live data (read-only)**

The backend container bind-mounts `./backend` and runs `uvicorn --reload`, so the code is already in it. Do not restart or recreate any container.

Run:
```bash
cd /Users/robmiller/Projects/mtg-deckbuilder
docker compose exec -T backend python - <<'EOF'
import asyncio, time
from app.db.session import async_session_factory
from app.services import deck_plan as dp

async def main():
    async with async_session_factory() as db:
        for colors in (["R"], ["R", "G"], []):
            start = time.time()
            got = await dp.relative_candidates(db, colors, "standard")
            print(colors, f"{time.time() - start:.2f}s", [(a, n) for a, n, _ in got])

asyncio.run(main())
EOF
```
Expected (data from 2026-10-06; later data shifts the names, not the shape): `['R']` lists about 10 archetypes, among them Boros Dragons, UR Aggro, Rakdos Aggro, Boros Dwarves and Boros Aggro, and not Jund Sacrifice, UW Control or any mono-green deck; `['R', 'G']` about 16; `[]` prints `[]`. Each query takes well under a second. If `['R']` lists a deck that plays no red or colorless spells, stop and report.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/deck_plan.py backend/tests/test_deck_plan.py
git commit -m "Find brew relatives: on-color candidates judged by Jev

relative_candidates takes the current archetypes with at least 2 lists
whose lists average at least 8 main-deck spell copies within the deck's
colors (colorless included, DFCs by color identity). choose_relatives
asks one Jev Noul per candidate, with the archetype's 25 most-played
spells and their average copies in the state, so Jev judges what the
lists play rather than the name. Relatives are those at 0.5 or more,
most likely first, at most 5.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 4: `assemble` plans and fills brews from their relatives

**Files:**
- Modify: `backend/app/services/deck_fill.py` (imports; `assemble` docstring, plan step, `state()`, the `scope` line, the log line and the return)
- Test: `backend/tests/test_deck_fill.py`

**Interfaces:**
- Consumes: `relative_candidates`, `choose_relatives` (Task 3), `plan_from_decklists(db, archetypes, format, colors)` (Task 2), `fill_slots(..., scope)`, `fill_sideboard(..., scope)` (Task 1).
- Produces: `assemble(...)` returns `{name, strategy_summary, main_deck, sideboard, reference, relatives, colors}`. `relatives` is the list of archetypes the brew was planned from (empty for a reference deck or an LLM-planned brew). The `[ASSEMBLY]` log line adds `relatives=[...]`.
- Flow after `choose_reference`:
  1. A reference with lists: `scope = [reference]`, as before.
  2. Otherwise `reference = None`; candidates from the parsed colors; Jev's relatives; with relatives, `plan_from_decklists(db, relatives, format, colors)` and `scope = relatives`.
  3. With no plan yet, `plan_with_llm(...)` and an empty scope, as today.
  4. The slot-pick state's `plan` is `"<request>, built from the current archetypes closest to it"` for a brew with relatives; it never names an archetype.

- [ ] **Step 1: Write the failing tests**

`wire()` also stubs `relative_candidates` (no candidates unless a test passes some), so the existing assemble tests keep their database fakes.

In `backend/tests/test_deck_fill.py`, replace:

```python
def wire(monkeypatch, reference_plan=None, archetypes=(("Boros Aggro", 5),)):
    pool = fake_pool(MAIN_POOL)
    monkeypatch.setattr(df, "recent_archetypes", AsyncMock(return_value=list(archetypes)))
```

with:

```python
def wire(monkeypatch, reference_plan=None, archetypes=(("Boros Aggro", 5),), candidates=()):
    pool = fake_pool(MAIN_POOL)
    monkeypatch.setattr(df, "recent_archetypes", AsyncMock(return_value=list(archetypes)))
    monkeypatch.setattr(df, "relative_candidates", AsyncMock(return_value=list(candidates)))
```

In `backend/tests/test_deck_fill.py`, replace:

```python
    async def test_slow_jev_hits_the_deadline(self, monkeypatch):
```

with:

```python
    async def test_brew_with_relatives_is_planned_and_filled_from_them(self, monkeypatch):
        candidates = [("Boros Aggro", 5, [("Bear 1", 4.0)]), ("Rakdos Aggro", 4, [("Bolt 0", 4.0)])]
        pool, side_pool = wire(monkeypatch, candidates=candidates)
        relatives_plan = Plan(slots=[Slot("threat_cheap", 2, 2, 18, "Bears"), Slot("burn", 0, 1, 20, "Burn")],
                              lands=22, copies={"Bolt 0": 2}, colors=["R"], reference="Boros Aggro")
        monkeypatch.setattr(df, "plan_from_decklists", AsyncMock(return_value=relatives_plan))
        llm_plan = AsyncMock()
        monkeypatch.setattr(df, "plan_with_llm", llm_plan)

        def answer(state, q):
            if "relative" in q:
                return {"relative": 0.9 if state["archetype"]["name"] == "Boros Aggro" else 0.2}
            return {"pick": {"choice": "none", "confidence": 0.9, "probabilities": {}}} \
                if "none" in q["pick"].criteria else {}
        jev = FakeJev(answer=answer)
        deck = await df.assemble(None, "mono-red aggro", ["R"], [], "standard", True, "aggro", client=jev)

        assert deck["reference"] is None and deck["relatives"] == ["Boros Aggro"]
        df.relative_candidates.assert_awaited_once_with(None, ["R"], "standard")
        df.plan_from_decklists.assert_awaited_once_with(None, ["Boros Aggro"], "standard", ["R"])
        llm_plan.assert_not_called()
        assert pool.scopes[0] == "Boros Aggro"  # relatives' cards first, then the format
        assert side_pool.calls[0][0] == "Boros Aggro"
        main = {e["card_name"]: e["quantity"] for e in deck["main_deck"]}
        assert total(deck["main_deck"]) == 60 and total(deck["sideboard"]) == 15
        assert main["Bolt 0"] == 2  # copies from the relatives' lists
        assert not {"Sacred Foundry", "Inspiring Vantage"} & set(main)  # brew nonbasic rule: 0 for one color
        slot_states = [state["deck"] for state, _, _ in jev.calls if "deck" in state]
        assert slot_states and all(
            s["plan"] == "mono-red aggro, built from the current archetypes closest to it" for s in slot_states)
        assert not any("Boros Aggro" in str(s) for s in slot_states)  # archetype names are not instructions

    async def test_no_relatives_uses_the_llm_plan(self, monkeypatch):
        pool, side_pool = wire(monkeypatch, candidates=[("Boros Aggro", 5, [("Bear 1", 4.0)])])
        monkeypatch.setattr(df, "plan_with_llm", AsyncMock(return_value=Plan(
            slots=[Slot("threat_cheap", 2, 2, 18, "Bears"), Slot("burn", 0, 1, 20, "Burn")], lands=22)))
        jev = FakeJev(answer=lambda state, q: {"relative": 0.3} if "relative" in q else (
            {"pick": {"choice": "none", "confidence": 0.9, "probabilities": {}}}
            if "none" in q["pick"].criteria else {}))
        deck = await df.assemble(None, "mono-red aggro", ["R"], [], "standard", True, "aggro", client=jev)
        assert deck["relatives"] == [] and total(deck["main_deck"]) == 60 and total(deck["sideboard"]) == 15
        df.plan_from_decklists.assert_not_called()
        df.plan_with_llm.assert_awaited_once_with("mono-red aggro", ["R"], "aggro")
        assert set(pool.scopes) == {None} and [c[0] for c in side_pool.calls] == [None]
        assert all(s["deck"]["plan"] == "mono-red aggro" for s, _, _ in jev.calls if "deck" in s)

    async def test_relatives_jev_failure_propagates(self, monkeypatch):
        wire(monkeypatch, candidates=[("Boros Aggro", 5, [("Bear 1", 4.0)])])
        jev = FakeJev(answer=lambda state, q: {"pick": "none"} if "pick" in q else {},
                      fail=lambda state: "archetype" in state)
        with pytest.raises(ExceptionGroup) as err:
            await df.assemble(None, "mono-red aggro", ["R"], [], "standard", True, "aggro", client=jev)
        assert err.group_contains(RuntimeError, match="jev unavailable")

    async def test_slow_jev_hits_the_deadline(self, monkeypatch):
```


- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_fill.py -q`
Expected: `11 failed`: every test that calls `wire()` fails with `AttributeError: <module 'app.services.deck_fill' ...> has no attribute 'relative_candidates'`.

- [ ] **Step 3: Implement**

In `backend/app/services/deck_fill.py`, replace:

```python
    CARD_COLORS, CHOICE_DEADLINE, CHOICE_TIMEOUT, MAX_OPTIONS, RECENT, WUBRG, Plan, Slot, archetype_keys, choose,
    choose_reference, is_land, largest_remainder, plan_from_decklists, plan_with_llm, recent_archetypes,
```

with:

```python
    CARD_COLORS, CHOICE_DEADLINE, CHOICE_TIMEOUT, MAX_OPTIONS, RECENT, WUBRG, Plan, Slot, archetype_keys, choose,
    choose_reference, choose_relatives, is_land, largest_remainder, plan_from_decklists, plan_with_llm,
    recent_archetypes, relative_candidates,
```

In `backend/app/services/deck_fill.py`, replace:

```python
    """Build a deck: reference or brew plan, requested cards, Jev-filled slots,
    lands, sideboard, summary. Returns {name, strategy_summary,
    main_deck, sideboard} like ai_service.generate_deck, plus the reference
    archetype (None for a brew) and the deck colors. Raises
```

with:

```python
    """Build a deck: reference or brew plan, requested cards, Jev-filled slots,
    lands, sideboard, summary. A brew is planned and filled from its relatives
    (the current archetypes Jev judges closest to the request), or from the LLM
    plan when it has none. Returns {name, strategy_summary, main_deck,
    sideboard} like ai_service.generate_deck, plus the reference archetype (None
    for a brew), the brew's relatives and the deck colors. Raises
```

In `backend/app/services/deck_fill.py`, replace:

```python
        reference = await choose_reference(client, request_text, colors or [], archetypes, format)
        plan = await plan_from_decklists(db, [reference], format) if reference else None
        if plan is None:
            reference = None
            plan = await plan_with_llm(request_text, colors or [], archetype)

```

with:

```python
        reference = await choose_reference(client, request_text, colors or [], archetypes, format)
        plan = await plan_from_decklists(db, [reference], format) if reference else None
        scope = [reference] if plan else []  # archetypes whose cards fill the deck first
        relatives: List[str] = []
        if plan is None:
            reference = None
            candidates = await relative_candidates(db, colors or [], format)
            relatives = await choose_relatives(client, request_text, colors or [], candidates)
            plan = await plan_from_decklists(db, relatives, format, colors) if relatives else None
            relatives = relatives if plan else []
            scope = relatives
        if plan is None:
            plan = await plan_with_llm(request_text, colors or [], archetype)

```

In `backend/app/services/deck_fill.py`, replace:

```python
                "plan": f"{reference}, a current {format.title()} archetype" if reference else request_text,
```

with:

```python
                "plan": (f"{reference}, a current {format.title()} archetype" if reference
                         else f"{request_text}, built from the current archetypes closest to it" if relatives
                         else request_text),
```

In `backend/app/services/deck_fill.py`, replace:

```python
        scope = [reference] if reference else []
        short = await fill_slots(
```

with:

```python
        short = await fill_slots(
```

In `backend/app/services/deck_fill.py`, replace:

```python
    logger.info(f"[ASSEMBLY] {name}: reference={reference} colors={deck_colors} "
                f"main={main.total()} sideboard={side.total()}")
    return {"name": name, "strategy_summary": summary, "main_deck": main.entries(),
            "sideboard": side.entries(), "reference": reference, "colors": deck_colors}
```

with:

```python
    logger.info(f"[ASSEMBLY] {name}: reference={reference} relatives={relatives} colors={deck_colors} "
                f"main={main.total()} sideboard={side.total()}")
    return {"name": name, "strategy_summary": summary, "main_deck": main.entries(),
            "sideboard": side.entries(), "reference": reference, "relatives": relatives, "colors": deck_colors}
```


- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest -q`
Expected: `362 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/deck_fill.py backend/tests/test_deck_fill.py
git commit -m "Build brews from their relatives' lists, LLM plan as the fallback

When no single archetype matches, assemble asks Jev which on-color
archetypes are close relatives of the request and plans from their
on-color spells. Main-deck slots and the sideboard come from the
relatives' cards first, then the format's. The Jev plan text is the
request 'built from the current archetypes closest to it', never an
archetype name. With no relatives the brew uses the LLM plan as before;
a Jev failure still raises so the generator falls back.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 5: Guard against hardcoded names

**Files:**
- Create: `backend/tests/test_no_hardcoded_names.py`

**Interfaces:**
- Consumes: the fixtures `tests/test_deck_plan.py` (`LISTS`, `RELATIVE_LISTS`, `CANDIDATE_ROWS`, `ARCHETYPES`) and `tests/test_deck_fill.py` (`CARDS`, `MAIN_POOL`, `LANDS`, `LAND_TEXTS`, `reference_plan()`), and `deck_fill.BASICS`.
- Produces: a test that fails when any string literal in `deck_plan.py` or `deck_fill.py` (docstrings and f-string parts included) contains a fixture card or archetype name as a whole word, case-insensitively. Basic land names are allowed.

- [ ] **Step 1: Write the test**

Create `backend/tests/test_no_hardcoded_names.py`:

```python
"""The metagame changes every week, so deck assembly code names no card, archetype
or land: everything comes from recent decklists. Basic land names are rules of
the game, not the metagame, and are allowed."""

import ast
import re
from pathlib import Path

from app.services import deck_fill as df
from tests import test_deck_fill as tdf
from tests import test_deck_plan as tdp

PRODUCTION = [Path(df.__file__), Path(df.__file__).with_name("deck_plan.py")]

FIXTURE_NAMES = (
    {r.name for r in tdp.LISTS + tdp.RELATIVE_LISTS} | {r.name for r in tdp.CANDIDATE_ROWS}
    | {r.archetype for r in tdp.CANDIDATE_ROWS} | {a for a, _ in tdp.ARCHETYPES}
    | {c.name for c in tdf.CARDS + tdf.MAIN_POOL + tdf.LANDS} | set(tdf.LAND_TEXTS)
    | {tdf.reference_plan().reference, "Rakdos Aggro", "UW Control", "Candy Trail"}
) - set(df.BASICS.values())


def literals(path: Path) -> list:
    """Every string literal in the file, f-string parts and docstrings included."""
    return [n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def named(text: str) -> set:
    return {n for n in FIXTURE_NAMES if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text, re.IGNORECASE)}


def test_fixture_names_are_collected():
    assert {"Hired Claw", "Boros Aggro", "Sacred Foundry", "Rakdos Aggro"} <= FIXTURE_NAMES
    assert "Mountain" not in FIXTURE_NAMES
    assert named('x = "Boros aggro lists"') == {"Boros Aggro"}  # the check itself works


def test_assembly_code_names_no_card_archetype_or_land():
    found = {f"{path.name}: {name}" for path in PRODUCTION for lit in literals(path) for name in named(lit)}
    assert not found
```


- [ ] **Step 2: Run it**

Run: `cd backend && python3 -m pytest tests/test_no_hardcoded_names.py -v`
Expected: `2 passed`. This is a guard on the code after Tasks 1-4, so it passes at once.

- [ ] **Step 3: Check that it bites**

Append the line `EXAMPLE = "Hired Claw"` to the end of `backend/app/services/deck_plan.py`, then run `cd backend && python3 -m pytest tests/test_no_hardcoded_names.py -q`.
Expected: `test_assembly_code_names_no_card_archetype_or_land` fails with `deck_plan.py: Hired Claw` in the assertion. Delete the line again, run the full suite (`cd backend && python3 -m pytest -q`) and confirm `364 passed` and that `git diff backend/app` is empty.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_no_hardcoded_names.py
git commit -m "Guard deck assembly code against hardcoded card and archetype names

The metagame changes every week, so deck_plan.py and deck_fill.py must
take every card and archetype from recent decklists. The test fails when
any string literal there (docstrings and f-strings included) names a
card or archetype from the test fixtures. Basic land names are rules,
not metagame, and are allowed.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```

---

### Task 6: Live eval with relatives

**Files:**
- Modify: `backend/scripts/eval_assembly.py`

**Interfaces:**
- Consumes: `assemble(...)["relatives"]` (Task 4), `deck_plan.archetype_keys` (Task 1).
- Produces: the eval builds four decks (Boros aggro, mono-red aggro, a red-green aggro brew, the Sephiroth request). For each brew it prints `relatives=[...]` and, when there are relatives, `format top-ups: [...]`. The checks are unchanged: legal, 60 + 15, every card played or requested, on-color nonbasic lands, at most 4 copies.

- [ ] **Step 1: Update the eval**

In `backend/scripts/eval_assembly.py`, replace:

```python
"""
Build three Standard decks with real Jev assembly and check each one: legal,
60 main and 15 sideboard, every card played in the last 14
days or requested, nonbasic lands within the deck colors, at most 4 copies.
Prints the lists for human review. Not run in CI.

```

with:

```python
"""
Build four Standard decks with real Jev assembly and check each one: legal,
60 main and 15 sideboard, every card played in the last 14
days or requested, nonbasic lands within the deck colors, at most 4 copies.
Brews print the relatives Jev chose and every nonland card from outside the
relatives' lists (a format top-up). Prints the lists for human review. Not run in CI.

```

In `backend/scripts/eval_assembly.py`, replace:

```python
from app.services.deck_plan import RECENT  # noqa: E402
```

with:

```python
from app.services.deck_plan import RECENT, archetype_keys  # noqa: E402
```

In `backend/scripts/eval_assembly.py`, replace:

```python
# (request, must have a reference archetype)
CASES = [
    ("Build me a Boros aggro deck", True),
    ("mono-red aggro", False),
    ("A black-red midrange deck built around Sephiroth, Fabled SOLDIER", False),
]
```

with:

```python
# (request, must have a reference archetype). Brews print their relatives; none is
# not a failure (the brew then uses the LLM plan).
CASES = [
    ("Build me a Boros aggro deck", True),
    ("mono-red aggro", False),
    ("a red-green aggro deck", False),
    ("A black-red midrange deck built around Sephiroth, Fabled SOLDIER", False),
]
```

In `backend/scripts/eval_assembly.py`, replace:

```python
    WHERE {RECENT}
""")


async def check(
```

with:

```python
    WHERE {RECENT}
""")
RELATIVES_SQL = text(f"""
    SELECT DISTINCT lower(split_part(x->>'card_name', ' // ', 1))
    FROM decklists d JOIN events e ON e.id = d.event_id
    CROSS JOIN LATERAL jsonb_array_elements(d.main_deck || d.sideboard) AS x
    WHERE {RECENT} AND lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))
""")


async def top_ups(db, deck) -> list:
    """Nonland cards from outside the relatives' lists: format top-ups."""
    if not deck["relatives"]:
        return []
    params = {"format": "standard", "archetypes": archetype_keys(deck["relatives"])}
    theirs = {r[0] for r in (await db.execute(RELATIVES_SQL, params)).all()}
    names = [e["card_name"] for e in deck["main_deck"] + deck["sideboard"]]
    cards = {r.name: r for r in (await db.execute(CARD_SQL, {"names": names})).all()}
    return [n for n in names if n.split(" // ")[0].lower() not in theirs
            and n in cards and "Land" not in cards[n].type_line.split(" // ")[0]]


async def check(
```

In `backend/scripts/eval_assembly.py`, replace:

```python
            print(f"reference={deck['reference']} colors={deck['colors']} "
                  f"requested={parsed['specific_cards']}")
```

with:

```python
            print(f"reference={deck['reference']} relatives={deck['relatives']} colors={deck['colors']} "
                  f"requested={parsed['specific_cards']}")
            if deck["relatives"]:
                print(f"format top-ups: {await top_ups(db, deck) or 'none'}")
```


- [ ] **Step 2: Run the unit tests**

Run: `cd backend && python3 -m pytest -q`
Expected: `364 passed` (the script is not imported by any test).

- [ ] **Step 3: Run it against real Jev**

Run: `cd /Users/robmiller/Projects/mtg-deckbuilder && docker compose exec -T backend python scripts/eval_assembly.py`
Expected: four decklists, each followed by `OK`, then `PASS`. From two runs while writing this plan (each deck 6-41 s):
- "Build me a Boros aggro deck": `reference=Boros Aggro relatives=[] colors=['W', 'R']`, as before.
- "mono-red aggro": `reference=None`, `colors=['R']`, and `relatives=['Boros Dwarves']` or `relatives=[]` (about half of the runs each, because Jev scores Boros Dwarves at about 0.47-0.5). With Boros Dwarves, the deck had the Dwarves' red and colorless cards (4 Dwarven Mauler, 4 Thorin, Mountain-king, 4 Chainsaw, 4 Lavaspur Boots, 4 The Lonely Mountain) and 16 format top-ups (Burst Lightning, Sear, Wild Ride and some one-ofs). Without relatives it used the LLM plan, as today.
- "a red-green aggro deck": `relatives=['Roaming Elementals']` (once also Boros Dwarves) and 6-10 format top-ups.
- The Sephiroth request: `relatives=[]` (LLM plan), exactly 1 copy of the DFC. This deck took 30-41 s, the slowest of the four.

- [ ] **Step 4: Stop and report if it does not pass**

If the only problem is an `ERROR` line from a Jev timeout (`TypeSafeAPITimeoutError` or `TimeoutError`), run it once more. Otherwise, if it prints `FAIL` or any `PROBLEMS:` line, **stop**. Do not tune `RELATIVE_THRESHOLD`, `MIN_ON_COLOR_SPELLS`, the question, the state or the SQL, and do not continue to Task 7. Those values come from the spec, and changing them needs a spec change. Report the full output (every decklist and problem line) to the human and wait.

Report these even when it passes:
- the relatives each brew got, and whether mono-red got any;
- the format top-ups;
- anything a player would find odd: off-plan cards, a curve that is too high, many one-ofs, or more than one sweeper in an aggro deck;
- any deck over 45 s, because `ASSEMBLY_DEADLINE` is 60 s.

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/eval_assembly.py
git commit -m "Eval brews built from relatives

Adds a red-green aggro brew and prints, for each brew, the relatives Jev
chose and the cards that came from the format instead of the relatives'
lists. Result: <PASS/FAIL>; mono-red relatives: <list>; red-green
relatives: <list>; <per-deck seconds>.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRcdjJFCWo3bHUFugg7oPN"
```
Fill in the `<...>` parts from the actual run before committing.

---

### Task 7: End-to-end check through the Build page's chat

**Files:** none, unless a bug is found.

**Interfaces:**
- Consumes: everything above, live, through `POST /api/conversations/chat` with `"mode": "build"` (the Build page). On that page a deck request becomes `generate_full_deck`, which calls `DeckGenerator.generate` and so `deck_fill.assemble`. The backend container bind-mounts `./backend` and runs `uvicorn --reload`, so the code is already live. **Do not recreate, restart or `docker compose up` any container.** `docker compose exec`, `docker logs` and `curl` are safe. Each request saves a conversation and a deck, as the app always does.

- [ ] **Step 1: Confirm the stack and the reload**

Run:
```bash
cd /Users/robmiller/Projects/mtg-deckbuilder
docker ps --filter name=spellbook --format '{{.Names}} {{.Status}}'
docker compose exec -T backend sh -c 'test -n "$TYPESAFE_API_KEY" && echo jev-key-set; test -n "$OPENROUTER_API_KEY" && echo llm-key-set'
docker logs spellbook-backend --since 30m 2>&1 | grep -iE "reload|error|traceback" | tail -20
```
Expected: `spellbook-backend`, `spellbook-db` and `spellbook-redis` are `Up`; `jev-key-set` and `llm-key-set` print; the log shows reloads after the code changes and no import errors or tracebacks.

- [ ] **Step 2: Two brews through the chat**

Run:
```bash
cd /Users/robmiller/Projects/mtg-deckbuilder
S=/private/tmp/claude-501/-Users-robmiller-Projects-mtg-deckbuilder/b8847f7b-d5b1-408b-9720-62bba18fb355/scratchpad
for ask in "Build me a mono-red aggro deck" "Build me a red-green aggro deck"; do
  curl -s -w '\n%{time_total}s\n' -X POST http://localhost:8000/api/conversations/chat \
    -H 'Content-Type: application/json' \
    -d "{\"message\": \"$ask\", \"mode\": \"build\", \"format\": \"standard\"}" > $S/chat.txt
  tail -1 $S/chat.txt
  head -1 $S/chat.txt | python3 -c '
import json, sys
d = json.load(sys.stdin).get("deck")
if not d:
    sys.exit("no deck in the chat response")
count = lambda es: sum(e["quantity"] for e in es or [])
print(d["name"], "| main", count(d["main_deck"]), "| side", count(d["sideboard"]))
print("; ".join(str(e["quantity"]) + " " + e["card_name"] for e in d["main_deck"]))'
done
docker logs spellbook-backend --since 5m 2>&1 | grep -E "\[ASSEMBLY\]|Jev assembly unavailable"
```
Expected: each request takes about 15-45 s and prints `main 60 | side 15` and its list. The log has two `[ASSEMBLY]` lines with `reference=None`: mono-red with `relatives=['Boros Dwarves']` or `relatives=[]`, red-green with `relatives=['Roaming Elementals']` (possibly with another relative), and no `Jev assembly unavailable` line. The chat builds its own request text, so its colors and relatives can differ from the eval's; report what they were.

- [ ] **Step 3: Report**

Record what each step showed: timings, totals, the `[ASSEMBLY]` lines and both lists. If `no deck in the chat response` prints, quote the response's `response` text, because the chat routed the message somewhere other than `generate_full_deck`. If a step shows `Jev assembly unavailable`, quote the logged exception. Fix any bug with a failing test first, then commit the fix on its own.
