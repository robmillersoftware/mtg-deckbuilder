# Forge simulation: playtested decks and real matchup tests

Date: 2026-10-06
Status: draft for review
Replaces: the LLM-narrated simulator (`game_simulator.py`, `/simulate`)
Builds on: Jev deck assembly (PR #60)

## Goal

Deck quality has plateaued because nothing in the builder plays Magic. Rules text, play counts, LLM judgment and Jev judgment are all stand-ins for knowing how cards perform together. This design adds a real rules engine that plays games, and uses it two ways:

1. **Test a deck.** Play any deck against the current meta and report how it does, with evidence.
2. **Playtest a build.** Every deck built from a prompt is improved by playing it against the meta, swapping cards, and keeping the swaps that win more.

The user sees what is happening while it happens, and gets a report they can act on afterwards.

## Feasibility (spike, 2026-10-06)

Forge daily snapshot 2.0.16 (10.06), headless `sim` mode:

- **Coverage:** 570 of the 571 cards in the last 14 days of Standard decklists are scripted. The one miss is a naming mismatch (`Unholy Annex / Ritual Chamber`).
- **Speed:** about 0.75 s per game in one JVM. 100 games took 53 s across 4 parallel JVMs on a 10-core machine. JVM startup is about 4 s.
- **Interface:** `sim -D <deck dir> -d a.dck b.dck -n <games> -f constructed [-q] [-s seed]`. Game logs go to stdout; the last line per game is `Match Result: <a>: N <b>: M`.
- **Play quality:** rules-correct but passive. In one game the AI made a land into a hasty creature and did not attack with it. Dimir Aggro, the top meta deck, lost 30-70 to Boros Dragons.

## What the simulation can and cannot tell us

Results measure a deck as piloted by Forge's AI. They are good evidence about mana, curve, whether synergies fire at all, and how a deck fares against the meta's threats. They undervalue decks that need skilled play (tempo, ninjutsu, combo, counterspell control). Every report says so in one sentence, and the builder never discards a card that real tournament lists play heavily on sim evidence alone (see Search, cut rules).

## Rules

- **Formats:** 60-card constructed (the formats `deck_fill` supports). Commander and cEDH are out of scope; the old 3-4 player mode is removed.
- **Games are best-of-one, no sideboarding.** Forge's AI does not sideboard well. Sideboards come from assembly as today and are not tested.
- **Gauntlet:** the top `GAUNTLET_SIZE` (5) archetypes by current meta share, each represented by its most recent best-placing list from the last 14 days. A deck's overall win rate is the gauntlet win rates weighted by meta share (renormalized over the 5).
- **Statistics:** win rates are reported with a 95% Wilson interval. Two decks are compared on the same game seeds (paired seeds per matchup) to cut noise.
- **No hardcoded card or archetype names,** as before. The gauntlet, cut candidates and add candidates all come from data.
- **A Forge failure never blocks a build:** the user gets the assembled deck, marked "not playtested", with the reason.

## Components

### Forge engine (`app/services/forge.py`)

- `available() -> bool`, `card_names() -> set[str]` (front faces, lowercased, read once from the cardsfolder zip).
- `write_deck(deck, path)`: our main deck as a `.dck` file. Names map to Forge's spelling through `card_names()`; a card Forge lacks is reported, never silently dropped.
- `play(deck_a, deck_b, games, seed) -> MatchResult`: runs `sim` in a subprocess with a timeout, splits `games` across `FORGE_WORKERS` JVMs (default: CPU count − 2), and parses the logs.
- `MatchResult`: wins, losses, draws, and per game: winner, turns, mulligans per player, and every card each player cast or played, by turn.
- The log parser is the only code that knows Forge's log format, and it is tested against captured real logs.

Runtime: Forge and a headless JRE 17 go into the `rq-worker` image, pinned by a `FORGE_SNAPSHOT` build arg. A `scripts/update_forge.sh` bumps it when a new set lands. The worker's `platform: linux/amd64` pin is removed for this image if possible, because Java under emulation on Apple Silicon is several times slower; this is measured in the first plan task.

### Card performance (`app/services/sim_stats.py`)

From a set of `MatchResult`s, per card in the tested deck:

- **cast rate:** games in which it was cast or played, per copy in the deck;
- **win rate when cast:** wins in the games where it was cast;
- **turn first cast**, median.

Plus per deck: average turn of a win, mulligan rate, and games lost with 3 or fewer lands played by turn 5 (mana screw) or 7 or more lands and an empty board by turn 8 (flood). These feed both the report and the search.

### Test a deck (replaces `/simulate`)

- **Request:** a saved deck or the deck in the current conversation; opponents are "the meta" (the gauntlet) or one or more chosen archetypes; games per matchup (default 50, max 200).
- **Runs** as an RQ job on the worker, not a FastAPI background task: it is CPU-heavy and can take minutes.
- **Report:** see Feedback.

### Playtest a build (`app/services/deck_search.py`)

Runs after assembly whenever Forge is available. Assembly stays the seed.

1. **Baseline:** play the seed against the gauntlet, `SCREEN_GAMES` (20) per matchup.
2. **Round** (up to `MAX_ROUNDS` = 6, within `BUILD_BUDGET` = 6 minutes):
   - **Cut candidates:** the 3 nonland cards with the lowest win rate when cast, among those cast in at least 5 games, plus any nonland card cast in under 15% of games (stuck in hand or uncastable). Never cut: a requested card; a synergy-slot card when the cut would leave the slot under 8 copies; or a card in more than half of the relatives' or reference's real lists.
   - **Add candidates:** for each cut card, one Jev Choice over the played pool for the cut card's slot (the existing `slot_pool`/`fill_slot` path). The state includes the sim evidence: the matchups the deck loses, and how the cut card underperformed.
   - **Land candidate:** if screw or flood games exceed 15%, one candidate moves the land count by ±1.
   - **Each candidate** is the seed with N copies of A swapped for N copies of B (N = A's copies, capped by B's limit). Up to `CANDIDATES_PER_ROUND` = 6 are screened on paired seeds.
   - **Accept** the best candidate if its weighted win rate beats the current deck by more than one standard error. Otherwise stop.
3. **Confirm:** the final deck and the seed play `CONFIRM_GAMES` (50) per matchup on fresh paired seeds. If the final deck is not better, the seed is returned and the report says the changes did not hold up.

Budget: one round is about 6 candidates × 5 matchups × 20 games = 600 games, roughly 1 minute on 8 JVMs.

## Feedback

This is where most of the user-facing work is.

### While it runs

- **The first deck shows right away.** Assembly takes about 25 s; the chat shows that deck immediately with "Playtesting against the top 5 decks, about 5 minutes. The list updates as it improves."
- **A live progress panel** under the deck, updated by polling every 2 s:
  - the current stage: "Baseline", "Round 2 of 6: trying 6 swaps", "Confirming";
  - a progress bar and time remaining, from games played over games planned;
  - the gauntlet table filling in live: each opponent, its meta share, wins-losses so far, and win rate;
  - an event feed in plain words: "Erode won 31% of the games it was cast in: trying Burst Lightning instead", "Kept −2 Erode +2 Burst Lightning: 48% → 53% against the meta", "Tried 6 swaps, none clearly better: stopping".
- **The deck updates in place** when a swap is kept: added cards are highlighted, cut cards appear struck through for one round.
- **A Stop button** ends the search and keeps the best deck so far. A queued job shows its place in the queue.

### The report (build and test)

- **Headline:** overall win rate against the meta with its range ("53% against the top 5 decks, likely 48-58%"), and the change from the first draft for a build.
- **Matchups:** per opponent, the win rate with range and a favored / even / unfavored label (above 55%, 45-55%, below 45%), and the average turn games end.
- **What changed** (builds): each kept swap with its before/after win rate overall and against the matchup it helped most.
- **Card performance:** the 5 strongest and 5 weakest cards by win rate when cast, with cast counts, so a weak result is traceable to cards. Cards with too few casts to judge are listed as such rather than ranked.
- **Mana:** mulligan rate, screw and flood rates, with a plain sentence when one is high ("Lost 18% of games stuck on 3 lands: consider a 24th land").
- **Watch a game:** one representative win and one loss as a condensed turn-by-turn log (lands, casts, attacks, life totals), rendered from the parsed log.
- **Limits, one line:** "Played by Forge's AI, which plays straightforwardly; decks that rely on tricky play may do better in real games."
- **Not simulated:** any card Forge cannot run, named, and the sideboard.

### When something fails

Each failure is reported in plain words, with what the user still has:

- Forge unavailable or crashed: "Couldn't playtest this deck (the simulator isn't running). Here is the untested build."
- A card Forge lacks: "Forge can't play [[Card]] yet, so it was left out of testing and kept in your list."
- Budget hit: the report says the search stopped on time and shows the best deck found.

## Data and API

- **`simulation_runs`** is replaced (it has no rows): `id, user_id (nullable, anonymous allowed as for conversations), kind ('test' | 'build'), status, deck (JSONB), opponents (JSONB), progress (JSONB: stage, games_done, games_planned, gauntlet table, events), report (JSONB), seed_deck, final_deck, error, created_at, updated_at`. One Alembic migration drops the old table and creates this one.
- `POST /simulations` (test), `GET /simulations/{id}`, `POST /simulations/{id}/stop`, `GET /simulations` (the user's runs).
- Chat's `generate_full_deck` returns the assembled deck plus `simulation_id`; the Build page polls it.
- Access rules match conversations: the owner, or anyone holding the id of an anonymous run.

## Removed

`game_simulator.py`, its schemas and routes, the LLM-narrated game turns, and the old Simulation page. The DeckView "Simulate" button opens the new test page.

## Testing

- **Unit:**
  - log parser against captured Forge logs: winner, turns, mulligans, casts per turn, draws, a timeout;
  - `.dck` writing and Forge name mapping, including split and double-faced cards and an unknown card;
  - Wilson intervals and weighted win rates;
  - search with a fake engine: cut and add selection, never-cut rules, acceptance threshold, stop on no improvement, budget stop, confirmation rejecting a non-improvement;
  - progress events and report shape;
  - API access rules.
- **Live, not CI:** `scripts/eval_playtest.py` builds the three decks that failed in review (mono-red aggro, the Weapons Manufacturing build-around, "beat the meta"), prints seed vs final with reports, and checks: the final deck is legal and 60 cards, no requested card was cut, every swap logged, and the confirmed win rate is not below the seed's.

## Out of scope

- Sideboarding and best-of-three.
- Commander, cEDH and multiplayer.
- Improving Forge's AI.
- Searching from scratch without an assembled seed.
