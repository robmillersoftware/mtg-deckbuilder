# Anchor brews to related archetypes

Date: 2026-10-06
Status: approved in conversation, pending spec review
Builds on: `2026-10-06-jev-deck-assembly-design.md` (PRs #52–#55)

## Goal

A brew is a request that matches no single current archetype, such as "mono-red aggro" when nobody plays exactly that. Brews are planned by an LLM slot plan today. That plan sometimes asks for roles that real decks of that kind don't play ("card draw", "protection", "ramp"), and Jev must then fill those slots with off-plan cards. Recent examples are Candy Trail, Lavaspur Boots, Dragonfire Blade and Flamebraider. Copy counts are also guessed by rank (4, 4, 3, 2, 1), which gives brews many one-ofs.

Instead, plan brews from the on-color part of the current archetypes most related to the request.

## Hard rule: no hardcoded names

The format changes every set and every week, so nothing about the metagame may be written into code.

- No card names, archetype names or land names appear anywhere in production code. That covers constants, prompts, filters and defaults.
- Related archetypes, their cards, slot sizes, copy counts and land counts all come from the last 14 days of decklists in the database.
- Jev judges relatedness by game plan, from the request and from the archetype's actual decklist contents. Colors are enforced by code (the on-color filter), not by the question; asking about colors made Jev score every two-color relative of a mono-color request below 0.5 (spike 2026-10-06: best 0.34 vs. 0.61-0.76 for the four aggro relatives with the game-plan question).
- Only thresholds and counts are constants, such as minimum lists, the relatedness cutoff, the maximum number of relatives and the on-color share.
- Tests may use names, because they are fixtures.

## Pipeline change

Step 1 is unchanged: `choose_reference` picks a single archetype or `none`. On `none`, the brew path now runs these steps:

1. **Candidates.** Code takes the current archetypes with at least `MIN_LISTS` (2) lists in the window whose lists average at least `MIN_ON_COLOR_SPELLS` (8) main-deck spell copies with colors ⊆ the deck colors, colorless included. Colors use the same `CARD_COLORS` rule as the pools, with front-face matching. If there are no deck colors, there are no candidates.
2. **Relatives.** Each candidate gets one Jev Noul, run concurrently through the shared cap and deadline: "Does `archetype` play the same kind of game as the deck the user asked for (for example fast aggro, midrange, control or ramp)? Judge its game plan from its cards; ignore its other colors."
   - The state is `{"request", "colors", "archetype": {"name", "lists", "cards"}}`. `cards` holds that archetype's 25 most-played main-deck spell names with their average copies, so Jev judges from contents, not the name alone.
   - Relatives are the candidates at or above `RELATIVE_THRESHOLD` (0.5), ranked by probability, capped at `MAX_RELATIVES` (5).
3. **Plan from relatives.** `plan_from_decklists` generalises to take a list of archetypes and an optional color filter. It aggregates over all the relatives' lists, counting only main-deck spells whose colors fit the deck colors. It uses the same top-role and mana-band aggregation, merge rule and largest-remainder rounding as a single reference.
   - **Slot sizes:** the average on-color copies per list in each (role, band). They are then scaled so the nonland total is 60 minus the land count.
   - **Land count:** the relatives' average land count, rounded.
   - **Nonbasic count:** the brew rule, the lower quartile by color count. Relatives may play off-color fixing that this deck doesn't need. The one-color utility-land check still applies.
   - **Copies:** each card's average copies across the relatives' lists that play it, rounded, at least 1.
   - **Plan colors:** the deck colors.
   - **Reference:** a joined label for logs only. It is not shown to Jev as an archetype name.
4. **Fill.** Main-deck slots are filled from cards played in the relatives' lists first, then the whole format for any shortfall. This is the existing two-stage fill with the scope widened from one archetype to a set. Lands are unchanged.
5. **Sideboard.** It is filled from the relatives' sideboard cards first, then the format's, with the existing color filter.
6. **Fallback.** With no relatives, the brew uses today's LLM slot plan and defaults, unchanged.

The Jev state for slot picks shows the deck plan as the user's request plus "built from the current archetypes closest to it". Archetype names are not used as instructions.

## Components

- **`deck_plan.py`:**
  - `relative_candidates(db, colors, format) -> List[(archetype, lists, top_cards)]`
  - `choose_relatives(client, request, colors, candidates) -> List[str]`
  - `plan_from_decklists(db, archetypes: Sequence[str], format, colors=None)`. A single reference passes `[archetype]`, which keeps current behavior.
- **`deck_fill.py`:**
  - `slot_pool`, `sideboard_pool` and the plays CTE accept a set of archetypes. The SQL becomes `lower(trim(d.archetype)) = ANY(:archetypes)`, bound.
  - `assemble` wires the brew path.

## Failure behavior

Any Jev failure in the relatives step propagates like other Jev failures. `assemble` then raises and the generator falls back to the LLM path. A brew with zero relatives is not a failure: it uses the LLM plan.

## Testing

**Unit tests:**
- candidate filtering (on-color share, `MIN_LISTS`, no colors);
- relatives (threshold, cap, ordering, state contains cards and not only the name);
- the multi-archetype plan (on-color filter, totals of 60, copies from relatives, land count from relatives, nonbasics from the brew rule);
- scoped pools with a set of archetypes;
- a single reference unchanged;
- the zero-relatives fallback;
- a grep-style test asserting that `deck_plan.py` and `deck_fill.py` contain no string literal matching a card or archetype name from the local test fixtures.

**Live eval:** `eval_assembly.py` gains "mono-red aggro" and one other brew. For each brew it prints the relatives Jev chose. Checks: 60 + 15, every card played or requested, at most 4 copies, and no card outside the relatives' lists unless the slot needed a format top-up (reported). Lists are printed for human review.

## Out of scope

- Changing single-archetype reference decks.
- Fit-flag swapping (tried and dropped: the flags vary between runs).
- New data sources beyond mtgtop8.
