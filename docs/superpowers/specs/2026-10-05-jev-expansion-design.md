# Jev expansion: roles, search, and chat routing

Date: 2026-10-05
Status: approved in conversation, pending spec review
Builds on: `2026-10-05-deck-fit-design.md` (deck fit) and the OpenRouter LLM switch (PR #49)

## Goal

Close the gaps seen in the first live guided-build run:

- wrong role assignments (Basilisk Collar suggested as removal),
- low-fit cards filling suggestion slots (Together as One at plan fit 0.21),
- empty results when OpenAI embeddings are unavailable,
- an hour-long, free-form-JSON role classification job,
- every chat turn paying for an LLM tool call to make routing decisions.

Jev answers narrow judgments. Code keeps retrieval, legality, colors, counts, and control flow. The LLM keeps generation: conversational text, decklists, explanations, sideboard plans, game narration.

## Shared infrastructure

- **Process-wide Jev concurrency cap.** Replace the per-call `asyncio.Semaphore(MAX_CONCURRENT)` in `deck_fit.score_fit` with one module-level limiter in a new `app/services/jev.py`. It is shared by deck fit, role tagging, search re-ranking and routing, so concurrent users stay under Jev's rate limit (80 req/s; cap stays 32 in-flight).
- `jev.py` also owns the client factory (`_client`, guarded `_close`, `REQUEST_TIMEOUT`, `RETRY_BUDGET`, `FIT_DEADLINE`), which move there from `deck_fit.py`. `deck_fit` imports them. Behavior is unchanged.

## 1. Card-role tagging with Jev

Replaces the LLM call in `app/jobs/classify_card_roles.py`. `get_unclassified_cards`, `save_card_roles`, `classify_all_cards` and `get_classification_stats` keep their contracts.

Per unique card, one Jev request:

- State: `{"card": {name, mana_cost, mana_value, type_line, oracle_text, power_toughness?}}`.
- For each of the 22 roles in `CARD_ROLES`: a Noul `is:<role>` ("Does `card` fill this deck-building role, judged from its oracle text and stats?" with the role definition as structured instructions) and a Score `eff:<role>` (efficiency relative to mana cost, 5 levels described by rate, not metagame knowledge).
- A role is assigned when its Noul is >= `ROLE_THRESHOLD` (0.5). Efficiency is `round(score) + 1` (1-5). `CardRole.confidence` stores the Noul, not the current hard-coded 0.9. `CardRole.reasoning` is left null.

Cards run concurrently through the shared cap and are saved in batches of 50. The single-pass snapshot fix from PR #49 stays.

Cost estimate: about 3k input tokens per card, roughly $0.65 for the 5,185 unique Standard names at $0.042/Mtok. Expected wall time is minutes.

**Eval:** `backend/scripts/eval_roles.py`, about 30 hand-labeled cards against real Jev, reporting per-role precision and recall. It must include Basilisk Collar (not `removal_targeted`), Murder (`removal_targeted`), Consider (`card_selection`), a basic land (`land_basic`), a dual land (`land_fixing_*`) and a cheap aggressive creature (`threat_cheap`). Tune `ROLE_THRESHOLD` and question wording on it.

The LLM classification currently running is left to finish. Re-running with Jev after merge requires clearing `card_roles` (`DELETE FROM card_roles`) and running the job; this is a manual step in the PR description, not a migration.

## 2. Role check at suggestion time

In the guided builder's fit path, each candidate already gets one Jev fit request. Add one Noul per role whose pool the candidate came from: "Does `candidate` fill this role in a deck: `<role description>`?"

- The role description is the guided role name (for example `removal`, `card advantage`), expanded through `ROLE_MAP` to the system roles' definitions when the name maps, otherwise the role name itself (covers keyword roles such as "cards with surveil").
- A candidate below `ROLE_FIT_CUTOFF` (0.5) for a role is dropped from that role's results only.
- `score_fit` gains an optional `roles_by_candidate: Dict[str, List[str]]`. `FitScore` gains `roles: Dict[str, float]`, defaulting to empty.
- If the fit request fails, the round falls back as today.

## 3. Minimum fit in ranking

`deck_fit.rank` also drops candidates whose plan fit is below `LOW_FIT_CUTOFF` (0.34, the same threshold `flag_low_fit` uses). A role may then show fewer cards. The fallback ordering, used when fit is unavailable, is unchanged.

## 4. Semantic search: shortlist, then Jev re-rank

`CardService.semantic_search(query, limit, standard_only, format, colors)` keeps its signature. All five callers are unchanged.

1. **Shortlist**: up to `SHORTLIST_SIZE` (60) unique names, already filtered by format and colors, merged in this priority order:
   - vector hits, only when OpenAI is configured and the query embedding succeeds;
   - Postgres full-text hits: `to_tsvector('english', name || ' ' || type_line || ' ' || oracle_text)` matched against the query words OR'd together (stopwords dropped), ordered by `ts_rank`;
   - popular cards in the requested colors and format, by tournament decklist frequency (existing `_rank_cards_by_tournament_frequency` SQL, moved to a `CardService` method).
2. **Re-rank**: one Jev Noul per shortlisted card, "Does `card` serve the request in `query`?", concurrent through the shared cap under `FIT_DEADLINE`. Sort by probability and drop below `SEARCH_CUTOFF` (0.5). Return the top `limit`.
3. **Fallback**: if Jev fails or is not configured, return the shortlist in its merge order, truncated to `limit`.

Migration `016`: a GIN expression index on that tsvector for `cards`.

OpenAI becomes optional at runtime. The Scryfall sync still computes embeddings when a key is configured.

## 5. Chat routing with Jev

In `ChatService.process_message`, after card-mention resolution and before any LLM call, one Jev request:

- State: the current message, the last 6 turns, a deck summary (`has_deck`, unique nonland count, deck colors), the persisted strategy context, database-resolved card names, the current meta archetype names, and code-detected mechanic keywords (`_extract_mtg_keywords`).
- Questions:
  - `action`: a Choice over the 7 tool names plus `reply`, each option with a structured description adapted from the tool's description.
  - `color:<C>` for each of W, U, B, R, G: a Noul, "Does the user want this color in their own deck (not an opponent's)?"
  - `role:<r>` for each role in the suggest_core role list: a Noul.
  - `package_role`: a Choice over the role list plus the detected mechanics.
  - `archetype`: a Choice over aggro, control, midrange, combo, tempo, none.
  - `opponent`: a Choice over the meta archetypes plus none.
  - `wants:<i>` for each resolved card name: a Noul, "Does the user want this card in their deck, rather than only mentioning it?"
- Code assembles the tool input:
  - colors: the colors with noul >= 0.5, else the deck's colors;
  - roles: the roles >= 0.5, else the existing default three;
  - strategy: the message plus the persisted strategy;
  - modification: the message;
  - count: the first integer in the message, default 6;
  - specific_cards: the wanted resolved names.
- Dispatch rules:
  - if `action` confidence >= `ROUTE_CONFIDENCE` and the action is a tool, dispatch it directly with no LLM call;
  - if `action` is `reply`, use `llm.complete` with the conversation for a text-only answer;
  - the existing forced-tool rule (a resolved card or build intent and no deck) overrides `reply` with the highest-probability tool;
  - if confidence is low, or Jev fails or is not configured, use the current `llm.chat_with_tools` path unchanged.
- `ROUTE_CONFIDENCE` starts at 0.6 and is tuned on the eval.

## 6. Deck-request parsing with Jev

`parse_deck_request(prompt, db)` keeps its return shape `{archetype, colors, colors_specified, strategy, specific_cards}`.

One Jev request:

- `archetype`: a Choice.
- `color:<C>`: a Noul per color.
- `colors_specified`: a Noul, "Did the user state colors for their own deck, not an opponent's?"
- `wants:<i>`: a Noul per database-resolved card name. Resolution uses the existing exact and fuzzy card-name lookup over the prompt's word n-grams.
- `strategy`: the prompt text.
- Fallback: the existing `fallback_parse` when Jev fails or is not configured.

## Eval scripts (not run in CI; require a key)

- `scripts/eval_roles.py`, as in section 1.
- `scripts/eval_routing.py`: about 25 labeled chat messages with expected action and arguments, also covering `parse_deck_request`. Must include "beat mono-red" (red not the user's color), "more removal" (`suggest_package`, role removal), "just build it" (`generate_full_deck`), "what's good right now?" (`analyze_meta`), "swap X for Y" (`modify_deck`), and a plain question (`reply`). Reports action accuracy and argument accuracy, and tunes `ROUTE_CONFIDENCE`.
- `scripts/eval_fit.py` stays; relabel Lazotep Reaver to `bad` (a deferred minor from the deck-fit work).

## Failure behavior

Unchanged rule: a Jev failure never blocks suggestions, search, chat, generation or classification.

| Path | Fallback |
|---|---|
| Role tagging | Batch skipped and logged; cards stay unclassified for the next run |
| Role check | Whole fit round falls back to retrieval order |
| Search re-rank | Shortlist order |
| Routing | LLM tool call |
| Parsing | `fallback_parse` |

## Testing

- Unit tests with a stubbed Jev client:
  - role tagging (assignment threshold, efficiency mapping, confidence stored);
  - role check (per-role drop, multi-role candidates);
  - `rank` minimum fit;
  - search (shortlist merge and dedupe, re-rank order, cutoff, fallback when Jev fails and when OpenAI is absent);
  - routing (confident dispatch, the forced-tool override of `reply`, the low-confidence LLM fallback, argument assembly including the color and opponent cases);
  - parsing (shape, the colors_specified case, fallback).
- Integration: a live chat request "Build me a mono-red aggro deck" against the running stack returns role groups with no plan-fit-below-cutoff cards, and the log shows no LLM tool call when routing is confident.

## Out of scope

- Replacing LLM generation (decklists, explanations, sideboards, narration).
- Caching Jev answers.
- Flavor-text themes.
- Re-running role tagging automatically after Scryfall sync (stays manual or admin-triggered as today).
