"""The report a user reads after a test or a build."""

from app.services.forge import OPPONENT, TESTED, GameRecord
from app.services.sim_report import LIMITS, build_report
from app.services.sim_stats import matchup_stats


def game(winner, casts, turns, log):
    return GameRecord(winner=winner, turns=turns, own_turns={TESTED: turns, OPPONENT: turns},
                      mulligans={TESTED: 0, OPPONENT: 0}, casts={TESTED: casts, OPPONENT: []},
                      lands={TESTED: [(t, "Forest") for t in range(1, 6)], OPPONENT: []}, log=log)


def test_report_sections():
    wins = [game(TESTED, [(2, "Good")], 6 + i % 3, [f"win {i}"]) for i in range(6)]
    losses = [game(OPPONENT, [(2, "Bad")], 9, [f"loss {i}"]) for i in range(6)]
    records = wins + losses
    stats = [matchup_stats("A", 10.0, records), matchup_stats("B", 5.0, [])]
    main = {"Forest": 24, "Good": 4, "Bad": 4, "Rare": 4}
    report = build_report(stats, records, main, {"Forest"}, ["Unknown Card"],
                          changes=[{"cut": "Bad", "add": "Good"}], stopped="no_improvement")
    assert report["overall"]["win_rate"] == 0.5 and report["overall"]["games"] == 12
    assert report["baseline"] is None
    assert [(m["opponent"], m["label"]) for m in report["matchups"]] == [("A", "even")]  # B played no games
    assert [c["name"] for c in report["cards"]["strongest"]] == ["Good"]
    assert [c["name"] for c in report["cards"]["weakest"]] == ["Bad"]
    assert report["cards"]["too_few"] == ["Rare"]
    assert report["games"]["win"] == ["win 1"]  # the win closest to the median win length (7 turns)
    assert report["games"]["loss"] == ["loss 0"]
    assert report["not_simulated"] == {"cards": ["Unknown Card"], "sideboard": True}
    assert report["limits"] == LIMITS
    assert report["changes"] == [{"cut": "Bad", "add": "Good"}] and report["stopped"] == "no_improvement"
    assert set(report["mana"]) == {"mulligan_rate", "screw_rate", "flood_rate", "advice"}
