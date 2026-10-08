"""Playtest search: cut what underperforms, keep swaps that clearly win more, confirm."""

import pytest

from app.services import deck_search as ds
from app.services.forge import OPPONENT, TESTED, GameRecord
from app.services.gauntlet import Opponent

OPPONENTS = [Opponent("A", 20.0, {}), Opponent("B", 10.0, {})]
LANDS = {"Forest"}
EFFECT = {"Good": 0.06, "Dud": -0.06}  # win-rate change per copy
SEED = {"Forest": 24, "Dud": 4, "Meh": 4, "Filler": 28}


def rate_of(main):
    return min(0.95, max(0.05, 0.5 + sum(EFFECT.get(n, 0) * q for n, q in main.items())))


def fake_evaluate(calls, flip_on_confirm=False):
    """Deterministic games: Dud is cast only in losses, Good only in wins."""
    async def evaluate(main, games, seed):
        calls.append((dict(main), games, seed))
        rate = rate_of(main)
        if flip_on_confirm and games == ds.SearchConfig().confirm_games:
            rate = 1 - rate
        out = {}
        for opp in OPPONENTS:
            wins = round(games * rate)
            records = []
            for i in range(games):
                won = i < wins
                casts = [(2, n) for n in main if n not in LANDS and (n != "Dud" or not won) and (n != "Good" or won)]
                records.append(GameRecord(
                    winner=TESTED if won else OPPONENT, turns=8, own_turns={TESTED: 8, OPPONENT: 8},
                    mulligans={TESTED: 0, OPPONENT: 0}, casts={TESTED: casts, OPPONENT: []},
                    lands={TESTED: [(t, "Forest") for t in range(1, 6)], OPPONENT: []}))
            out[opp.archetype] = records
        return out
    return evaluate


async def find_add(main, cut, losing_to):
    return "Good" if cut == "Dud" else "Meh2"


class FakeProgress:
    def __init__(self, stop_after=None):
        self.events, self.stages, self.decks, self.matchups = [], [], [], []
        self.stop_after, self.checks = stop_after, 0

    async def stage(self, text):
        self.stages.append(text)

    async def event(self, text, kind="info"):
        self.events.append((kind, text))

    async def set_matchups(self, stats):
        self.matchups.append(stats)

    async def deck(self, main):
        self.decks.append(dict(main))

    async def stop_requested(self):
        self.checks += 1
        return self.stop_after is not None and self.checks > self.stop_after


async def test_keeps_a_clearly_better_swap_and_confirms_it():
    calls, progress = [], FakeProgress()
    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate(calls), find_add,
                               lambda main, cut: False, progress, rng_seed=5)
    assert result.main == {"Forest": 24, "Meh": 4, "Filler": 28, "Good": 4}
    assert [(c["cut"], c["add"], c["copies"]) for c in result.changes] == [("Dud", "Good", 4)]
    change = result.changes[0]
    assert change["before"] == pytest.approx(0.25) and change["after"] == pytest.approx(0.75)
    assert result.stopped == "no_improvement"
    screen = {seed for _, games, seed in calls if games == ds.SearchConfig().screen_games}
    confirm = [(m, seed) for m, games, seed in calls if games == ds.SearchConfig().confirm_games]
    assert screen == {5} and [m for m, _ in confirm] == [SEED, result.main] and len({s for _, s in confirm}) == 1
    assert any(kind == "kept" and "Dud" in text and "Good" in text for kind, text in progress.events)
    assert any(kind == "tried" and "Dud won 0% of the games it was cast in" in text for kind, text in progress.events)
    assert progress.decks[-1] == result.main


async def test_protected_cards_are_never_cut():
    calls, progress = [], FakeProgress()
    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate(calls), find_add,
                               lambda main, cut: cut == "Dud", progress)
    assert result.main == SEED and result.changes == []
    assert all(c[0]["Dud"] == 4 for c in calls)


async def test_changes_that_fail_confirmation_are_reverted():
    calls, progress = [], FakeProgress()
    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate(calls, flip_on_confirm=True), find_add,
                               lambda main, cut: False, progress)
    assert result.main == SEED and result.changes == [] and result.stopped == "reverted"
    assert progress.decks[-1] == SEED


async def test_budget_and_user_stop():
    times = iter([0.0, 1000.0, 1000.0])
    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate([]), find_add, lambda m, c: False,
                               FakeProgress(), clock=lambda: next(times))
    assert result.stopped == "budget" and result.main == SEED

    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate([]), find_add, lambda m, c: False,
                               FakeProgress(stop_after=0))
    assert result.stopped == "user" and result.main == SEED


async def test_stop_during_the_baseline_returns_the_seed():
    async def evaluate(main, games, seed):
        raise ds.Stopped()
    result = await ds.playtest(SEED, OPPONENTS, LANDS, evaluate, find_add, lambda m, c: False, FakeProgress())
    assert result.main == SEED and result.stopped == "user" and result.final == []


def test_swap_apply_respects_existing_copies():
    assert ds.Swap("A", "B", 2, "").apply({"A": 2, "B": 1}) == {"B": 3}


def test_land_swap_for_screw():
    screwed = GameRecord(winner=OPPONENT, turns=6, own_turns={TESTED: 6, OPPONENT: 6},
                         mulligans={TESTED: 0, OPPONENT: 0}, casts={TESTED: [(2, "Weak")], OPPONENT: []},
                         lands={TESTED: [(1, "Forest"), (2, "Forest")], OPPONENT: []})
    records = [screwed] * 10
    from app.services.sim_stats import card_stats
    main = {"Forest": 22, "Weak": 4}
    cards = card_stats(records, main, {"Forest"})
    swap = ds.land_swap(main, records, cards, lambda m, c: False, ds.SearchConfig())
    assert (swap.cut, swap.add, swap.copies) == ("Weak", "Forest", 1)
    assert "short on lands" in swap.reason


async def test_stop_after_a_kept_change():
    calls, progress = [], FakeProgress(stop_after=1)
    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate(calls), find_add,
                               lambda main, cut: False, progress, rng_seed=5)
    assert result.stopped == "user"
    assert len(result.changes) == 1 and result.changes[0]["cut"] == "Dud"
    assert result.main == {"Forest": 24, "Meh": 4, "Filler": 28, "Good": 4}
    # baseline is first-draft stats, final is after the swap
    assert result.baseline[0].win_rate != result.final[0].win_rate
    # no confirmation calls
    confirm = [(m, games) for m, games, _ in calls if games == ds.SearchConfig().confirm_games]
    assert len(confirm) == 0


async def test_no_candidates():
    async def no_find_add(main, cut, losing_to):
        return None

    calls, progress = [], FakeProgress()
    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate(calls), no_find_add,
                               lambda main, cut: False, progress)
    assert result.stopped == "no_candidates"
    assert result.main == SEED
    assert result.changes == []


async def test_max_rounds():
    calls, progress = [], FakeProgress()
    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate(calls), find_add,
                               lambda main, cut: False, progress,
                               cfg=ds.SearchConfig(max_rounds=1), rng_seed=5)
    assert result.stopped == "max_rounds"
    assert len(result.changes) == 1  # round 1 keeps the swap
    # confirmation still runs
    confirm = [(m, games) for m, games, _ in calls if games == ds.SearchConfig().confirm_games]
    assert len(confirm) == 2  # seed and final


async def test_flood_land_swap():
    flooded = GameRecord(winner=OPPONENT, turns=7, own_turns={TESTED: 7, OPPONENT: 7},
                         mulligans={TESTED: 0, OPPONENT: 0}, casts={TESTED: [(2, "Best")], OPPONENT: []},
                         lands={TESTED: [(i, "Forest") for i in range(1, 9)], OPPONENT: []})
    records = [flooded] * 10
    from app.services.sim_stats import card_stats
    main = {"Forest": 28, "Best": 3, "Filler": 29}
    cards = card_stats(records, main, {"Forest"})
    swap = ds.land_swap(main, records, cards, lambda m, c: False, ds.SearchConfig())
    assert swap is not None
    assert swap.cut == "Forest" and swap.copies == 1
    assert swap.add == "Best"
    assert "drawing mostly lands" in swap.reason


async def test_four_copy_cap():
    calls, progress = [], FakeProgress()
    # modify seed so Good starts at 3 copies
    seed_with_good = {"Forest": 24, "Dud": 4, "Meh": 4, "Filler": 25, "Good": 3}

    result = await ds.playtest(seed_with_good, OPPONENTS, LANDS, fake_evaluate(calls), find_add,
                               lambda main, cut: False, progress, rng_seed=5)
    # at most 1 copy of Dud can be swapped for Good (since Good is at 3)
    if result.changes:
        assert result.changes[0]["copies"] == 1
    # verify all evaluated decks total 60 cards
    for main, games, _ in calls:
        assert sum(main.values()) == 60
