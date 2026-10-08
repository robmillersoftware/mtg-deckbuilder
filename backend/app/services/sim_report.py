"""The report shown after a deck test or a build playtest."""

import statistics
from typing import Dict, List, Optional, Sequence, Set

from app.services.forge import TESTED, GameRecord
from app.services.sim_stats import MatchupStats, card_stats, mana_advice, mana_stats, overall

LIMITS = ("Played by Forge's AI, which plays straightforwardly; decks that rely on tricky play "
          "may do better in real games.")
MIN_CASTS = 5  # a card needs this many games cast to be ranked
SHOWN_CARDS = 5
LOG_LINES = 300


def representative(records: Sequence[GameRecord], won: bool) -> Optional[List[str]]:
    """The condensed log of the win (or loss) closest to the median length."""
    games = [r for r in records if r.winner is not None and (r.winner == TESTED) == won]
    if not games:
        return None
    mid = statistics.median(r.turns for r in games)
    return min(games, key=lambda r: abs(r.turns - mid)).log[:LOG_LINES]


def build_report(matchups: Sequence[MatchupStats], records: Sequence[GameRecord], main: Dict[str, int],
                 lands: Set[str], missing: Sequence[str], baseline: Optional[Sequence[MatchupStats]] = None,
                 changes: Sequence[Dict] = (), stopped: Optional[str] = None) -> Dict:
    cards = card_stats(records, main, lands)
    judged = sorted((c for c in cards if c.games_cast >= MIN_CASTS),
                    key=lambda c: c.win_rate_when_cast, reverse=True)
    half = (len(judged) + 1) // 2  # the top half can be "strongest", the bottom half "weakest"
    strongest = judged[:min(half, SHOWN_CARDS)]
    weakest = list(reversed(judged[half:]))[:SHOWN_CARDS]
    mana = mana_stats(records)
    return {
        "overall": overall(matchups).as_dict(),
        "baseline": overall(baseline).as_dict() if baseline else None,
        "matchups": [m.as_dict() for m in matchups if m.games],  # no games, nothing to label
        "changes": list(changes),
        "cards": {"strongest": [c.as_dict() for c in strongest], "weakest": [c.as_dict() for c in weakest],
                  "too_few": sorted(c.name for c in cards if c.games_cast < MIN_CASTS)},
        "mana": {**mana, "advice": mana_advice(mana)},
        "games": {"win": representative(records, True), "loss": representative(records, False)},
        "not_simulated": {"cards": list(missing), "sideboard": True},
        "limits": LIMITS,
        "stopped": stopped,
    }
