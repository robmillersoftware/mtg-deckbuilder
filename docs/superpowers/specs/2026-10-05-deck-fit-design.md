# Deck fit with Jev

Date: 2026-10-05
Status: approved in conversation, pending spec review

## Goal

Judge whether a card belongs in a specific deck: its theme, play pattern, and mechanics. Use the judgment in three places: guided-builder suggestions, review of an existing deck, and a check on Claude-generated decks.

Jev (TypeSafe System One, `jev-latest`) answers only narrow judgment questions. Code owns retrieval, legality, curve and role counts, metagame strength, and the final ranking.

## What Jev knows

Nothing about Magic beyond what the request carries. Every question includes the evidence:

- The candidate card: name, mana cost, mana value, type line, oracle text, power/toughness. Scryfall oracle text carries reminder text for non-evergreen keywords (verified: Doom Whisperer reads "Surveil 2. (Look at the top two cards...)"), so Jev can read mechanics it has never heard of.
- The deck identity (below): theme tags, the user's request text if any, and the oracle text of the key cards.

Jev is never asked how strong a card is in the metagame. Tournament frequency (mtgtop8), rarity, and mana value already cover that in code.

## Deck identity

```python
class DeckIdentity(BaseModel):
    tags: list[str]              # active theme tags
    key_cards: list[str]         # card names, 5-8
    request_text: str | None     # user's original description
    overrides: IdentityOverrides # user edits, applied last

class IdentityOverrides(BaseModel):
    tags_on: list[str] = []
    tags_off: list[str] = []
    pinned: list[str] = []       # always key cards
    unpinned: list[str] = []     # never key cards
```

### Inference

`infer_identity(cards, request_text)` makes one Jev request. The state is the full decklist (name, type line, oracle text per unique card) plus `request_text`. A 100-card Commander list fits under the 32k-token state limit.

- **Theme tags**: one Noul per tag. "Does the deck in `deck.cards` build around this theme?" with the tag's definition as structured instructions. A tag is on at noul >= 0.5. Tag list: `graveyard`, `tokens`, `counters`, `spellslinger`, `lifegain`, `artifacts`, `enchantments`, `sacrifice`, `tribal`, `aggro`, `midrange`, `control`, `ramp`. Each has a one-line definition in code.
- **Key cards**: one Score per unique card. "How central is `deck.cards[i]` to this deck's plan?" with levels: incidental / supporting / important / defines the plan. Top 8 by score become key cards (5 for decks under 40 unique cards). Lands are excluded.

Overrides apply after inference: `tags_on` and `pinned` are added, `tags_off` and `unpinned` removed.

With no decklist yet (start of a guided build), identity is `request_text` alone and the tag Nouls run against it. Key cards stay empty until the deck has at least 6 nonland cards.

## Fit scoring

`score_fit(identity, candidates) -> dict[str, FitScore]` sends one Jev request per candidate, all concurrent, using the async client.

State per request:

```json
{
  "deck": {"themes": ["graveyard", "sacrifice"], "request": "...", "key_cards": [{"name": "...", "type_line": "...", "oracle_text": "..."}]},
  "candidate": {"name": "...", "mana_cost": "...", "type_line": "...", "oracle_text": "..."}
}
```

Questions:

| ID | Type | Instructions | Criteria |
|---|---|---|---|
| `plan_fit` | Score | How well does `candidate` advance the plan described by `deck.themes` and `deck.request`? | does nothing for the plan / incidental help / clearly supports the plan / a core piece of the plan |
| `synergy` | Score | Does `candidate`'s oracle text create, use, or reward what `deck.key_cards` do? | no interaction / loose overlap / direct interaction with one key card / direct interaction with several key cards |
| `anti_synergy` | Noul | Does `candidate` actively work against the plan (for example exiling your own graveyard in a graveyard deck, or punishing your own token creation)? | |

```python
class FitScore(BaseModel):
    plan_fit: float     # 0-1 (score / 3)
    synergy: float      # 0-1 (score / 3)
    anti_synergy: float # noul
```

When `key_cards` is empty, `synergy` is skipped and its weight moves to `plan_fit`.

## Ranking

`rank(candidates, fit, meta, weights)` is plain code:

1. Drop candidates with `anti_synergy >= 0.7`.
2. `total = 0.4 * plan_fit + 0.3 * synergy + 0.3 * meta_norm`, where `meta_norm` is tournament frequency normalized to 0-1 within the pool.
3. Ties break on the existing rarity and mana-value order.

The weights and the 0.7 cutoff are starting values. The eval script (Testing) is how they get tuned.

## Integration

### 1. Guided builder

`DeckAnalyzer.suggest_cards_for_strategy` (`backend/app/services/guided_builder.py`) keeps its retrieval paths (role map, keyword search, semantic search, co-occurrence, mechanical synergy) but collects `3 * cards_per_role` candidates per role. It then calls `score_fit` once on the combined pool, runs `rank`, and keeps the top `cards_per_role` per role. Each suggestion dict gains `fit: {plan_fit, synergy}`.

The identity for an unsaved guided build is kept in `Conversation.current_deck["identity"]` and re-inferred when the nonland count crosses 6, 15, and 30.

Frontend: `GuidedBuilder.tsx` suggestion cards show plan fit and synergy as two small badges.

### 2. Deck review

- Migration: `Deck.identity` JSONB, nullable.
- `POST /decks/{id}/fit` infers or loads the identity, scores every unique nonland card, and returns `{identity, cards: {name: FitScore}}`.
- `PATCH /decks/{id}/identity` saves overrides.
- `DeckView.tsx` gets a Fit panel: active tags (toggleable), key cards (pin/unpin), and the deck's cards sorted by fit with the bottom five highlighted.

### 3. Generator check

In `DeckGenerator.generate` (`backend/app/services/deck_generator.py`), after `_validate_and_fix_cards`, infer identity from the generated list plus the request, score the cards, and store `identity` and per-card fit on the deck. Version one flags cards and does not swap them. Swapping in the best-fitting same-role candidate is a follow-up once the scores are trusted.

## Failure handling

If `TYPESAFE_API_KEY` is unset, or a request raises `TypeSafeError` or times out (2 s per request), fit is treated as unavailable: `rank` falls back to the current ordering, endpoints return `fit: null`, and the UI hides fit badges. A Jev failure never blocks suggestions, deck saves, or generation.

## Configuration

- `TYPESAFE_API_KEY` in `Settings`, `.env.example`, and docker-compose.
- `typesafe-sdk==0.7.2` in `requirements.txt`.
- Model `jev-latest`. Pin a versioned ID once thresholds are tuned.

## Testing

- Unit tests with a stubbed client: override merge, anti-synergy cutoff, weighting and normalization, the empty-key-cards path, and the fallback when the client raises.
- `backend/scripts/eval_fit.py`: about 15 hand-labeled (deck, card, should_fit) cases run against real Jev, printing scores and agreement. Examples: graveyard deck vs. a card that exiles your own graveyard; tokens deck vs. an anthem; control deck vs. a vanilla 2-drop. Requires a key. Not run in CI.

## Out of scope

- Caching fit scores.
- Replacing the Claude card-role classifier with Jev (a shelved stash on this branch; separate work).
- Auto-swapping low-fit cards in generated decks.
- Flavor-text themes (flavor text is not stored today).
