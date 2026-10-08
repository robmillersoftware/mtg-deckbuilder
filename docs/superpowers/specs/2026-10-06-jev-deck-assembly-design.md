# Jev deck assembly

Date: 2026-10-06
Status: approved in conversation, pending spec review. Revised the same day: brews get a sideboard, sideboards are not slot-planned, and a fetchland guard on the land pool.
Builds on: `2026-10-05-jev-expansion-design.md`; PR #51 (deck generation fixes)

## Goal

Generated decks are legal but play poorly: the LLM picks a high curve, off-plan cards and many one-ofs. Replace LLM card picking with:

- a slot plan taken from recent tournament decklists (or, for brews, from the LLM),
- Jev choosing the cards for each slot from cards played in the last two weeks.

The LLM only plans brews and writes the summary text.

## Feasibility (spike, 2026-10-06)

- **Pool size:** one Jev `Choice` over up to 255 candidates (the API maximum) ranks the whole pool.
- **Cost and speed:** pools of 32-205 cards used 2.4k-15.9k input tokens, in 0.2-0.65 s.
- **Play counts matter:** without play counts in the option text, picks were casual (Fanatical Firebrand over Hired Claw). With "played in N recent Standard decklists" in each option, the top picks were what a strong player would choose: Burst Lightning, Sear and Abrade; Hired Claw; Smaug and Magmatic Hellkite. Jev still weighed fit to the slot over raw popularity.

## Rules

- **Tournament-playable** means the card appears in a decklist from the last 14 days for the format (`events.date >= today - 14`).
- **Only tournament-playable cards are eligible**, with one exception: cards the user explicitly requests (`specific_cards` from parsing) always go in, regardless of play data.
- A Jev failure never blocks generation: the generator falls back to today's LLM path.

## Pipeline

All of this runs inside `DeckGenerator.generate()`, replacing its `ai_service.generate_deck(...)` call. The steps that follow it stay unchanged: card validation, deck validation, Jev fit flags and optional explanations. `/decks/generate` and chat's `generate_full_deck` both use this path.

### 1. Reference archetype

`deck_plan.choose_reference(request_text, colors, format)`:

- Candidates are the archetype names from decklists in the last 14 days, by count, including names with at least 2 lists, capped at 254.
- One Jev `Choice` over those names plus `none`: "Which current archetype is the user asking to build?" The state holds the request text and the parsed colors.
- If the pick isn't `none` and confidence is at least `REFERENCE_CONFIDENCE` (0.5), return the archetype name. Otherwise return `None`, meaning a brew.

### 2a. Slot plan from decklists

`deck_plan.plan_from_decklists(archetype, format)` uses that archetype's decklists from the last 14 days.

For each nonland main-deck card:

- **role:** its highest-confidence `card_roles` row. Untagged cards fall back to `creature` / `noncreature` by type line.
- **band:** its mana value band, one of 0-1, 2, 3, 4, 5+.

Slot sizes:

- Average copies per (role, band) across the lists gives the slot sizes.
- A slot under 2 copies merges into the same role's nearest band, then into the same band's largest slot.
- Rounding is adjusted so the total equals the average nonland count, rounded.
- The land count is the average land count, rounded.

Only the main deck is slot-planned; the sideboard is picked separately (step 6).

### 2b. Slot plan for brews

`deck_plan.plan_with_llm(request_text, colors, archetype_hint)` makes one `llm.complete` call.

- It returns JSON: a list of `{role, cmc_min, cmc_max, type_contains, copies, description}`.
- Validation: roles must be in `CARD_ROLES` or be `creature` / `noncreature`, and copies are positive integers.
- Copies are scaled so the nonland total is 60 minus the land count. The land count is 24 for control, 23 for midrange and 22 otherwise.
- Brews get a sideboard from the format's sideboard cards (step 6).
- If the JSON is invalid, a fixed archetype default plan is used for aggro, midrange and control. These live in code.

### 3. Requested cards

- They are reserved before any slot is filled: 4 copies, or 1 if the type line contains `Legendary`.
- Each one reduces the copies of the first slot it fits by role and band, or is added beyond the plan if none fits. Overflow then reduces the largest slot.
- Requested lands count toward the land total.
- Deck colors are the parsed colors plus the colors of any requested card, so a requested off-color card brings its color in.

### 4. Fill slots with Jev

For each slot, largest first, `deck_fill.slot_pool(slot, colors, format, chosen)` builds the pool: cards in last-14-day decklists for the format that meet all of these:

- legal in the format;
- `colors` ⊆ deck colors;
- match the slot's role (`card_roles`) or `type_contains`;
- within the slot's mana value range;
- not already chosen.

Each card carries its play count, meaning the number of distinct decklists it appears in. The pool is ordered by play count and capped at 255.

`deck_fill.fill_slot(...)` asks one Jev `Choice`:

- **State:** the deck plan (reference archetype or request text, colors) and the cards chosen so far, with copies.
- **Question:** "Which card best fills this slot?" plus the slot description.
- **Options:** card name → `"{mana_cost} {type_line}. {oracle_text[:200]} [Played in N recent {format} tournament decklists]"`. Names are unique because the pool groups by name; a double-faced card uses its full name.

The options also include "none of these fit". Cards Jev ranks below it are not taken, and the unfilled count moves to other slots like any shortfall. The last-resort catch-all slot and the sideboard don't offer it, so the deck still reaches 60 + 15. (Without it, a brew slot like red "card draw" was filled with Candy Trail, since a Choice always picks something.)

Code then walks the cards by descending probability:

- It takes each card's copies until the slot's count is reached.
- With a reference, copies are the card's average copies in the reference lists, rounded, at least 1. For brews, they are 4, 4, 3, 2, 1 by rank, then 1.
- Copies are capped at 4 for nonbasics.

If the pool runs out, the remaining count moves to the next slot with the same role, then the largest remaining slot.

### 5. Lands

- **Nonbasic count:** the reference lists' average nonbasic count. For brews it is the lower quartile of the nonbasic counts of recent decklists with the same number of colors (6 for mono-color on 2026-10-06; a median let land-heavy green landfall lists set every mono brew to 11). Without data, the fixed rule applies: 0 for mono-color, 4 for two colors, 6 for three or more.
- **Pool:** lands played in the last 14 days whose `color_identity` ⊆ deck colors. That covers on-color duals and colorless utility lands (identity `{}`). Any off-color symbol excludes a land. The pool is ordered by play count.
- **Fetchland guard:** a land whose oracle text searches for a land with basic land types must name at least one of the deck colors' basic types (W Plains, U Island, B Swamp, R Mountain, G Forest) to enter the land pool. Lands that search only for a generic "basic land" stay allowed.
- **Pick:** Jev picks with the same `Choice` mechanism. The slot description is "a land for this deck's mana: fixing for its colors, or utility".
- **Basics:** they fill the remaining land count, split by the colored mana symbols across the chosen nonland cards' mana costs. Every deck color gets at least 1 basic when basics are used.

### 6. Sideboard

Sideboards are not slot-planned.

- **Candidates:** the cards in the reference lists' sideboards; for a brew, the format's sideboard cards from the last 14 days, any archetype. They are color-filtered by the same rule as the main pools (including the DFC `color_identity` fallback), played-only, exclude main-deck cards, ordered by play count and capped at 255.
- **Pick:** one Jev `Choice` ranks them: "Which card best belongs in this deck's sideboard?", with the deck plan and the main deck in the state.
- **Copies:** code walks the cards by descending probability, taking the card's average sideboard copies in the reference lists, rounded, at least 1 (brews: its average sideboard copies across the format), capped at 4 total copies across main and sideboard, until exactly 15.
- Every deck, brew or reference, is 60 + 15 and validates.

### 7. Summary

One `llm.complete` call writes `strategy_summary` and the deck name from the finished list. If the LLM is unavailable, a templated name and summary are used instead.

## Components

- `backend/app/services/deck_plan.py`: `Slot` dataclass, `choose_reference`, `plan_from_decklists`, `plan_with_llm`, archetype default plans.
- `backend/app/services/deck_fill.py`: `slot_pool`, `fill_slot`, `fill_lands`, `sideboard_pool`, `fill_sideboard`, `assemble(request_text, colors, specific_cards, format, include_sideboard) -> {name, strategy_summary, main_deck, sideboard}`, the same shape `ai_service.generate_deck` returns.
- `backend/app/services/deck_generator.py`: calls `deck_fill.assemble` when Jev is configured. It falls back to `ai_service.generate_deck` when Jev is unavailable or `assemble` raises.
- `backend/app/jobs/mtgtop8_scrape.py`: follows the two-week view's pagination (`format?f=X&meta=..&cp=N`) until a page has no dated event rows. Standard yields 22 events instead of 20.
- Jev calls go through `app/services/jev.py`, with its shared cap and deadline.

## Speed

There are about 10-15 sequential `Choice` calls, at roughly 0.5 s each, plus one or two sideboard `Choice` calls and one LLM summary call. That makes about 10-15 s per deck, against about 56 s today.

## Testing

**Unit tests** (stubbed Jev via `tests/jev_fake.py`, fake DB rows):

- reference choice: picked, `none`, low confidence;
- slot planning: role and band aggregation, merging small slots, totals;
- LLM plan: validation and default on bad JSON;
- requested-card reservation;
- pool filters: played-only, color subset, role and type, mana value, excluding chosen, 255 cap;
- copy rules for reference and brew decks;
- slot overflow;
- lands: identity subset including colorless, the fetchland guard (a Plains/Island fetch excluded from mono-red, a generic basic-land fetch allowed), basics split;
- sideboard: reference sideboard candidates, format candidates for brews, average sideboard copies, the 4-copy cap across main and sideboard, exactly 15;
- `assemble` totals of 60 and 15;
- fallback to the LLM path on Jev failure.

**Live eval** (`backend/scripts/eval_assembly.py`, not in CI) builds three Standard decks:

- "Boros aggro" (has a reference);
- "mono-red aggro" (likely a brew);
- a request naming one specific card.

Each must pass all of these checks:

- legal;
- 60 main and 15 sideboard;
- every card played in the last 14 days or requested;
- every nonbasic land's identity a subset of the deck colors;
- no more than 4 copies of a nonbasic.

The script prints the lists for human review.

## Out of scope

- The `archetype_templates` job is broken: it mixes formats and its roles and land counts are empty. This flow doesn't use it.
- The guided builder's mana base.
- Formats other than 60-card constructed. cEDH and Commander keep the LLM path.
- Off-meta or new-set cards without play data, unless requested.
