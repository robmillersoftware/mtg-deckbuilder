"""Win rates with uncertainty (Wilson intervals, meta-share weighting), and the
per-card and mana evidence from Forge games that explains them."""

import math
import re
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from app.services.forge import OPPONENT, TESTED, GameRecord

Z = 1.96  # 95% intervals
FAVORED, UNFAVORED = 0.55, 0.45
SCREW_TURN, SCREW_LANDS = 5, 3  # lost after reaching own turn 5 with 3 or fewer lands played
FLOOD_TURN, FLOOD_LANDS, FLOOD_SPELLS = 7, 7, 4  # lost with 7+ lands and 4 or fewer spells by own turn 7
ADVICE_RATE = 0.15
MULLIGAN_ADVICE_RATE = 0.25


def pct(x: float) -> str:
    return f"{round(100 * x)}%"


def wilson(score: float, n: int) -> Tuple[float, float]:
    """95% interval for a win rate of score/n (draws count half)."""
    if n == 0:
        return (0.0, 1.0)
    p = score / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def label(rate: float) -> str:
    return "favored" if rate > FAVORED else "unfavored" if rate < UNFAVORED else "even"


@dataclass
class MatchupStats:
    opponent: str
    share: float
    wins: int
    losses: int
    draws: int
    avg_turns: float

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def as_dict(self) -> Dict:
        lo, hi = wilson(self.wins + 0.5 * self.draws, self.games)
        return {**asdict(self), "games": self.games, "win_rate": self.win_rate, "lo": lo, "hi": hi,
                "label": label(self.win_rate)}


def matchup_stats(opponent: str, share: float, records: Sequence[GameRecord]) -> MatchupStats:
    return MatchupStats(
        opponent=opponent, share=share,
        wins=sum(r.winner == TESTED for r in records),
        losses=sum(r.winner == OPPONENT for r in records),
        draws=sum(r.winner is None for r in records),
        avg_turns=statistics.fmean(r.turns for r in records) if records else 0.0)


@dataclass
class Overall:
    win_rate: float
    lo: float
    hi: float
    se: float
    games: int

    def as_dict(self) -> Dict:
        return {"win_rate": self.win_rate, "lo": self.lo, "hi": self.hi, "games": self.games}


def overall(matchups: Sequence[MatchupStats]) -> Overall:
    """Win rate against the field: matchups weighted by meta share."""
    played = [m for m in matchups if m.games]
    total = sum(m.share for m in played)
    if not played or total <= 0:
        return Overall(0.0, 0.0, 1.0, 0.5, 0)
    rate = sum(m.share / total * m.win_rate for m in played)
    se = math.sqrt(sum((m.share / total) ** 2 * m.win_rate * (1 - m.win_rate) / m.games for m in played))
    return Overall(rate, max(0.0, rate - Z * se), min(1.0, rate + Z * se), se, sum(m.games for m in played))


@dataclass
class CardStat:
    name: str
    copies: int
    games_cast: int
    cast_share: float  # share of games in which it was cast
    win_rate_when_cast: Optional[float]
    median_turn: Optional[float]  # own turn it was first cast, median

    def as_dict(self) -> Dict:
        return asdict(self)


_FACES = re.compile(r"\s+//?\s+")


def card_stats(records: Sequence[GameRecord], main: Dict[str, int], lands: Set[str]) -> List[CardStat]:
    """Per nonland card of `main`: how often it was cast and how those games went.
    Forge logs a split or double-faced card by its front face."""
    back: Dict[str, str] = {}
    for name in main:
        back[name.lower()] = name
        back.setdefault(_FACES.split(name)[0].lower(), name)
    seen: Dict[str, List[Tuple[Optional[str], int]]] = {n: [] for n in main if n not in lands}
    for record in records:
        first: Dict[str, int] = {}
        for turn, card in record.casts[TESTED]:
            name = back.get(card.lower())
            if name in seen and name not in first:
                first[name] = turn
        for name, turn in first.items():
            seen[name].append((record.winner, turn))
    stats = []
    for name, games in seen.items():
        score = sum(1.0 if w == TESTED else 0.5 if w is None else 0.0 for w, _ in games)
        stats.append(CardStat(
            name=name, copies=main[name], games_cast=len(games),
            cast_share=len(games) / len(records) if records else 0.0,
            win_rate_when_cast=score / len(games) if games else None,
            median_turn=statistics.median(t for _, t in games) if games else None))
    return stats


def _by(entries: Sequence[Tuple[int, str]], turn: int) -> int:
    return sum(1 for t, _ in entries if t <= turn)


def mana_stats(records: Sequence[GameRecord]) -> Dict[str, float]:
    n = len(records) or 1
    screw = flood = 0
    for r in records:
        if r.winner != OPPONENT:
            continue
        if r.own_turns[TESTED] >= SCREW_TURN and _by(r.lands[TESTED], SCREW_TURN) <= SCREW_LANDS:
            screw += 1
        elif (r.own_turns[TESTED] >= FLOOD_TURN and _by(r.lands[TESTED], FLOOD_TURN) >= FLOOD_LANDS
              and _by(r.casts[TESTED], FLOOD_TURN) <= FLOOD_SPELLS):
            flood += 1
    return {"mulligan_rate": sum(r.mulligans[TESTED] > 0 for r in records) / n,
            "screw_rate": screw / n, "flood_rate": flood / n}


def mana_advice(mana: Dict[str, float]) -> List[str]:
    advice = []
    if mana["screw_rate"] > ADVICE_RATE:
        advice.append(f"Lost {pct(mana['screw_rate'])} of games stuck on 3 or fewer lands by turn 5: "
                      "consider one more land.")
    if mana["flood_rate"] > ADVICE_RATE:
        advice.append(f"Lost {pct(mana['flood_rate'])} of games drawing mostly lands: consider one fewer land.")
    if mana["mulligan_rate"] > MULLIGAN_ADVICE_RATE:
        advice.append(f"Mulliganed {pct(mana['mulligan_rate'])} of games: the mana or the curve may be awkward.")
    return advice
