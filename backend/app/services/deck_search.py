"""Playtest search: play the deck against the gauntlet, swap out what underperforms,
keep a swap only when it clearly wins more, and confirm the result on fresh games."""

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Set

from app.services.deck_fill import BASICS
from app.services.forge import GameRecord
from app.services.gauntlet import Opponent
from app.services.sim_stats import CardStat, MatchupStats, card_stats, mana_stats, matchup_stats, overall, pct

BASIC_NAMES = set(BASICS.values())
MAX_COPIES = 4
CONFIRM_SEED_OFFSET = 1000

Evaluate = Callable[[Dict[str, int], int, int], Awaitable[Dict[str, List[GameRecord]]]]
FindAdd = Callable[[Dict[str, int], str, List[str]], Awaitable[Optional[str]]]
Protect = Callable[[Dict[str, int], str], bool]


class Stopped(Exception):
    """The user asked to stop."""


@dataclass
class SearchConfig:
    screen_games: int = 8  # per matchup, for the baseline and each candidate
    confirm_games: int = 12  # per matchup, for the final check
    max_rounds: int = 3
    candidates: int = 3  # swaps tried per round
    budget_s: float = 240.0
    min_casts: int = 5  # games cast before a card's win rate is trusted
    stuck_rate: float = 0.15  # cast in fewer games than this: stuck in hand or uncastable
    problem_rate: float = 0.15  # screw or flood rate that earns a land-count candidate


@dataclass
class Swap:
    cut: str
    add: str
    copies: int
    reason: str

    def apply(self, main: Dict[str, int]) -> Dict[str, int]:
        new = dict(main)
        new[self.cut] = new.get(self.cut, 0) - self.copies
        if new[self.cut] <= 0:
            del new[self.cut]
        new[self.add] = new.get(self.add, 0) + self.copies
        return new


@dataclass
class SearchResult:
    main: Dict[str, int]
    baseline: List[MatchupStats]
    final: List[MatchupStats]
    records: Dict[str, List[GameRecord]]
    changes: List[Dict] = field(default_factory=list)
    stopped: str = "max_rounds"


def _flat(records: Dict[str, List[GameRecord]]) -> List[GameRecord]:
    return [r for rs in records.values() for r in rs]


def land_swap(main: Dict[str, int], records: Sequence[GameRecord], cards: Sequence[CardStat],
              protect: Protect, cfg: SearchConfig) -> Optional[Swap]:
    """One land more (screw) or fewer (flood), traded with a spell."""
    mana = mana_stats(records)
    basics = [n for n in main if n in BASIC_NAMES]
    judged = [c for c in cards if c.games_cast >= cfg.min_casts]
    if not basics or not judged:
        return None
    basic = max(basics, key=main.get)
    lands = sum(q for n, q in main.items() if n in BASIC_NAMES)
    if mana["screw_rate"] > cfg.problem_rate:
        weakest = min((c for c in judged if not protect(main, c.name)), key=lambda c: c.win_rate_when_cast,
                      default=None)
        if weakest:
            return Swap(weakest.name, basic, 1, f"Lost {pct(mana['screw_rate'])} of games short on lands: "
                                                f"trying one more {basic} instead of a {weakest.name}")
    if mana["flood_rate"] > cfg.problem_rate and lands > 0 and not protect(main, basic):
        best = max((c for c in judged if main[c.name] < MAX_COPIES), key=lambda c: c.win_rate_when_cast,
                   default=None)
        if best:
            return Swap(basic, best.name, 1, f"Lost {pct(mana['flood_rate'])} of games drawing mostly lands: "
                                             f"trying another {best.name} instead of a {basic}")
    return None


async def _propose(main: Dict[str, int], records: Sequence[GameRecord], stats: Sequence[MatchupStats],
                   lands: Set[str], find_add: FindAdd, protect: Protect, cfg: SearchConfig,
                   tried: Set) -> List[Swap]:
    cards = card_stats(records, main, lands)
    judged = sorted((c for c in cards if c.games_cast >= cfg.min_casts), key=lambda c: c.win_rate_when_cast)
    weakest = judged[:3]
    stuck = [c for c in cards if c.cast_share < cfg.stuck_rate and c not in weakest]
    losing = [m.opponent for m in sorted(stats, key=lambda m: m.win_rate)[:2]]
    swaps: List[Swap] = []
    for card in weakest + stuck:
        if len(swaps) >= cfg.candidates - 1:  # leave room for a land candidate
            break
        if protect(main, card.name):
            continue
        add = await find_add(main, card.name, losing)
        if not add or (card.name, add) in tried:
            continue
        copies = min(main[card.name], MAX_COPIES - main.get(add, 0))
        if copies <= 0:
            continue
        why = (f"{card.name} won {pct(card.win_rate_when_cast)} of the games it was cast in" if card in weakest
               else f"{card.name} was cast in only {pct(card.cast_share)} of games")
        swaps.append(Swap(card.name, add, copies, f"{why}: trying {add} instead"))
        tried.add((card.name, add))
    land = land_swap(main, records, cards, protect, cfg)
    if land and (land.cut, land.add) not in tried:
        swaps.append(land)
        tried.add((land.cut, land.add))
    return swaps[:cfg.candidates]


def _change(swap: Swap, before: Sequence[MatchupStats], after: Sequence[MatchupStats],
            before_rate: float, after_rate: float) -> Dict:
    pairs = [(b, a) for b in before for a in after if a.opponent == b.opponent]
    if not pairs:
        return {"cut": swap.cut, "add": swap.add, "copies": swap.copies, "before": before_rate, "after": after_rate,
                "best_matchup": None}
    b, a = max(pairs, key=lambda p: p[1].win_rate - p[0].win_rate)
    return {"cut": swap.cut, "add": swap.add, "copies": swap.copies, "before": before_rate, "after": after_rate,
            "best_matchup": {"opponent": a.opponent, "before": b.win_rate, "after": a.win_rate}}


async def playtest(seed_main: Dict[str, int], opponents: Sequence[Opponent], lands: Set[str],
                   evaluate: Evaluate, find_add: FindAdd, protect: Protect, progress,
                   cfg: SearchConfig = SearchConfig(), clock: Callable[[], float] = time.monotonic,
                   rng_seed: int = 0) -> SearchResult:
    start = clock()

    def stats_of(records: Dict[str, List[GameRecord]]) -> List[MatchupStats]:
        return [matchup_stats(o.archetype, o.share, records.get(o.archetype, [])) for o in opponents]

    await progress.stage("Baseline: playing the first draft against the meta")
    try:
        current_records = await evaluate(seed_main, cfg.screen_games, rng_seed)
    except Stopped:
        return SearchResult(dict(seed_main), [], [], {}, [], "user")
    current, current_main = stats_of(current_records), dict(seed_main)
    first_stats = current  # save baseline for honest reporting if search ends early
    first = overall(current)
    rate, se = first.win_rate, first.se
    await progress.set_matchups(current)
    await progress.event(f"First draft: {pct(rate)} against the meta")

    changes: List[Dict] = []
    stopped, tried = "max_rounds", set()
    try:
        for round_no in range(1, cfg.max_rounds + 1):
            if clock() - start > cfg.budget_s:
                stopped = "budget"
                await progress.event("Out of time: keeping the best deck so far")
                break
            if await progress.stop_requested():
                raise Stopped()
            swaps = await _propose(current_main, _flat(current_records), current, lands, find_add, protect, cfg, tried)
            if not swaps:
                stopped = "no_candidates"
                await progress.event("No more swaps worth trying")
                break
            await progress.stage(f"Round {round_no} of {cfg.max_rounds}: trying {len(swaps)} swaps")
            for swap in swaps:
                await progress.event(swap.reason, "tried")
            best = None
            for swap in swaps:
                records = await evaluate(swap.apply(current_main), cfg.screen_games, rng_seed)
                stats = stats_of(records)
                candidate_rate = overall(stats).win_rate
                if best is None or candidate_rate > best[1]:
                    best = (swap, candidate_rate, records, stats)
            swap, candidate_rate, records, stats = best
            if candidate_rate - rate > se:
                changes.append(_change(swap, current, stats, rate, candidate_rate))
                await progress.event(f"Kept −{swap.copies} {swap.cut} +{swap.copies} {swap.add}: "
                                     f"{pct(rate)} → {pct(candidate_rate)} against the meta", "kept")
                current_main, current_records, current = swap.apply(current_main), records, stats
                rate, se = candidate_rate, overall(stats).se
                await progress.set_matchups(current)
                await progress.deck(current_main)
            else:
                stopped = "no_improvement"
                await progress.event(f"Tried {len(swaps)} swaps; none was clearly better")
                break
    except Stopped:
        stopped = "user"
        await progress.event("Stopped: keeping the best deck so far")

    if not changes:
        return SearchResult(current_main, current, current, current_records, changes, stopped)
    if stopped == "user":
        return SearchResult(current_main, first_stats, current, current_records, changes, stopped)

    await progress.stage("Confirming: replaying the first draft and the final deck on new games")
    try:
        confirm_seed = rng_seed + CONFIRM_SEED_OFFSET
        seed_records = await evaluate(seed_main, cfg.confirm_games, confirm_seed)
        final_records = await evaluate(current_main, cfg.confirm_games, confirm_seed)
    except Stopped:
        return SearchResult(current_main, first_stats, current, current_records, changes, "user")
    baseline, final = stats_of(seed_records), stats_of(final_records)
    if overall(final).win_rate <= overall(baseline).win_rate:
        await progress.event("The changes didn't hold up over more games, so the first draft stands")
        await progress.set_matchups(baseline)
        await progress.deck(seed_main)
        return SearchResult(dict(seed_main), baseline, baseline, seed_records, [], "reverted")
    await progress.set_matchups(final)
    await progress.event(f"Confirmed: {pct(overall(baseline).win_rate)} → {pct(overall(final).win_rate)} "
                         "against the meta", "kept")
    return SearchResult(current_main, baseline, final, final_records, changes, stopped)
