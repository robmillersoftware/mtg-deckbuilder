"""Win rates with uncertainty, and the card and mana evidence behind them."""

import pytest

from app.services.forge import OPPONENT, TESTED, GameRecord
from app.services import sim_stats as ss


def game(winner=TESTED, casts=(), lands=5, own=8, mulligans=0, turns=8):
    return GameRecord(winner=winner, turns=turns, own_turns={TESTED: own, OPPONENT: own},
                      mulligans={TESTED: mulligans, OPPONENT: 0},
                      casts={TESTED: list(casts), OPPONENT: []},
                      lands={TESTED: [(t, "Forest") for t in range(1, lands + 1)], OPPONENT: []})


class TestWinRates:
    def test_wilson_interval(self):
        lo, hi = ss.wilson(10, 20)
        assert lo == pytest.approx(0.299, abs=0.001) and hi == pytest.approx(0.701, abs=0.001)
        assert ss.wilson(0, 0) == (0.0, 1.0)

    def test_draws_count_half(self):
        m = ss.matchup_stats("A", 10.0, [game(TESTED), game(OPPONENT), game(None, turns=30)])
        assert (m.wins, m.losses, m.draws, m.games) == (1, 1, 1, 3)
        assert m.win_rate == pytest.approx(0.5)
        assert m.avg_turns == pytest.approx(46 / 3)
        d = m.as_dict()
        assert d["label"] == "even" and d["opponent"] == "A" and d["games"] == 3 and d["lo"] < 0.5 < d["hi"]

    def test_labels(self):
        assert ss.label(0.56) == "favored" and ss.label(0.55) == "even" and ss.label(0.44) == "unfavored"
        assert ss.label(0.45) == "even"

    def test_overall_weights_by_meta_share(self):
        a = ss.matchup_stats("A", 30.0, [game(TESTED)] * 8 + [game(OPPONENT)] * 2)
        b = ss.matchup_stats("B", 10.0, [game(TESTED)] * 2 + [game(OPPONENT)] * 8)
        o = ss.overall([a, b])
        assert o.win_rate == pytest.approx(0.75 * 0.8 + 0.25 * 0.2)
        assert o.games == 20 and o.lo < o.win_rate < o.hi and o.se > 0

    def test_overall_with_no_games(self):
        o = ss.overall([ss.matchup_stats("A", 10.0, [])])
        assert (o.win_rate, o.games) == (0.0, 0)

    def test_overall_sweep_keeps_an_interval(self):
        o = ss.overall([ss.matchup_stats("A", 10.0, [game(TESTED)] * 20)])
        assert o.win_rate == 1.0 and o.se > 0 and o.lo < 0.95 and o.hi == 1.0


class TestCardStats:
    def test_win_rate_when_cast_maps_front_faces_back(self):
        main = {"Forest": 20, "Bear": 4, "Esper Origins // Summon: Esper Maduin": 4, "Never Cast": 4}
        records = [game(TESTED, [(2, "Bear"), (3, "Esper Origins"), (4, "Bear")]),
                   game(OPPONENT, [(2, "Bear")]),
                   game(TESTED, [(5, "Esper Origins")])]
        stats = {c.name: c for c in ss.card_stats(records, main, lands={"Forest"})}
        assert set(stats) == {"Bear", "Esper Origins // Summon: Esper Maduin", "Never Cast"}  # no lands
        bear = stats["Bear"]
        assert (bear.games_cast, bear.win_rate_when_cast, bear.median_turn) == (2, 0.5, 2)
        assert bear.cast_share == pytest.approx(2 / 3)
        assert stats["Esper Origins // Summon: Esper Maduin"].win_rate_when_cast == 1.0
        assert stats["Never Cast"].games_cast == 0 and stats["Never Cast"].win_rate_when_cast is None


class TestMana:
    def test_screw_flood_and_mulligans(self):
        screwed = game(OPPONENT, lands=2, own=6)
        flooded = game(OPPONENT, lands=8, own=8, casts=[(2, "Bear")])
        short_loss = game(OPPONENT, lands=2, own=3)  # lost before its 5th turn: not counted as screw
        records = [screwed, flooded, short_loss, game(TESTED, mulligans=1)]
        m = ss.mana_stats(records)
        assert m == {"mulligan_rate": 0.25, "screw_rate": 0.25, "flood_rate": 0.25}
        advice = ss.mana_advice(m)
        assert any("3 or fewer lands" in a for a in advice) and any("one fewer land" in a for a in advice)
        assert ss.mana_advice({"mulligan_rate": 0.0, "screw_rate": 0.0, "flood_rate": 0.0}) == []
