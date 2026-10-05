"""
Deck fit: Jev (TypeSafe System One) judgments on whether cards fit a deck.

Jev answers narrow questions (theme tags, card centrality, plan fit, synergy,
anti-synergy). Code owns ranking, legality, and metagame strength.
Design: docs/superpowers/specs/2026-10-05-deck-fit-design.md
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

THEME_TAGS: Dict[str, str] = {
    "graveyard": "Fills its own graveyard and gets value from cards there (self-mill, surveil, recursion, flashback, escape)",
    "tokens": "Creates many tokens and benefits from having them",
    "counters": "Puts +1/+1 or other counters on its permanents and rewards having them",
    "spellslinger": "Casts many instants and sorceries and rewards casting them",
    "lifegain": "Gains life repeatedly and rewards gaining life",
    "artifacts": "Plays many artifacts and rewards controlling or casting them",
    "enchantments": "Plays many enchantments and rewards controlling or casting them",
    "sacrifice": "Sacrifices its own permanents for value and rewards creatures dying",
    "tribal": "Focuses on one creature type and rewards sharing that type",
    "aggro": "Wins fast with cheap creatures and direct damage before the opponent stabilizes",
    "midrange": "Trades resources efficiently and wins with sturdy, value-generating threats",
    "control": "Answers threats with removal, counterspells, and card advantage, and wins late",
    "ramp": "Accelerates its mana to cast expensive, powerful spells early",
}

TAG_THRESHOLD = 0.5
ANTI_SYNERGY_CUTOFF = 0.7
LOW_FIT_CUTOFF = 0.34
WEIGHTS = {"plan_fit": 0.4, "synergy": 0.3, "meta": 0.3}
ORACLE_CHAR_LIMIT = 600
RE_INFER_AT = (6, 15, 30)


class IdentityOverrides(BaseModel):
    tags_on: List[str] = Field(default_factory=list)
    tags_off: List[str] = Field(default_factory=list)
    pinned: List[str] = Field(default_factory=list)
    unpinned: List[str] = Field(default_factory=list)


class DeckIdentity(BaseModel):
    tags: List[str] = Field(default_factory=list)
    key_cards: List[str] = Field(default_factory=list)
    request_text: Optional[str] = None
    overrides: IdentityOverrides = Field(default_factory=IdentityOverrides)


class FitScore(BaseModel):
    plan_fit: float              # 0-1
    synergy: Optional[float]     # 0-1, None when the deck has no key cards
    anti_synergy: float          # noul


def card_payload(card: Any) -> Dict[str, str]:
    """The card fields Jev sees. Accepts a Card ORM object or a suggestion dict."""
    if isinstance(card, dict):
        get = card.get
        name = card.get("card_name") or card.get("name")
    else:
        get = lambda k: getattr(card, k, None)  # noqa: E731
        name = card.name
    return {
        "name": name,
        "mana_cost": get("mana_cost") or "",
        "type_line": get("type_line") or "",
        "oracle_text": (get("oracle_text") or "")[:ORACLE_CHAR_LIMIT],
    }


def is_land(payload: Dict[str, str]) -> bool:
    return "Land" in payload.get("type_line", "")


def apply_overrides(identity: DeckIdentity, deck_names: Sequence[str]) -> DeckIdentity:
    """Apply user edits; unknown tags and pinned cards not in the deck are ignored."""
    o = identity.overrides
    tags = [t for t in identity.tags if t not in o.tags_off]
    tags += [t for t in o.tags_on if t in THEME_TAGS and t not in tags]
    in_deck = set(deck_names)
    keys = [k for k in identity.key_cards if k not in o.unpinned]
    keys += [p for p in o.pinned if p in in_deck and p not in keys]
    return identity.model_copy(update={"tags": tags, "key_cards": keys})


def _avg(f: FitScore) -> float:
    return f.plan_fit if f.synergy is None else (f.plan_fit + f.synergy) / 2


def rank(names: Sequence[str], fit: Dict[str, FitScore], freq: Dict[str, int]) -> List[str]:
    """Drop anti-synergy cards, then sort by weighted fit + normalized meta frequency.

    `fit` must cover every name. `freq` is keyed by lowercase name.
    Sort is stable, so ties keep retrieval order.
    """
    top = max(freq.values(), default=0) or 1

    def total(n: str) -> float:
        f = fit[n]
        syn = f.plan_fit if f.synergy is None else f.synergy
        return (WEIGHTS["plan_fit"] * f.plan_fit + WEIGHTS["synergy"] * syn
                + WEIGHTS["meta"] * freq.get(n.lower(), 0) / top)

    kept = [n for n in names if fit[n].anti_synergy < ANTI_SYNERGY_CUTOFF]
    return sorted(kept, key=total, reverse=True)


def flag_low_fit(fit: Dict[str, FitScore], limit: int = 5) -> List[str]:
    """Cards that work against the plan or barely fit it, worst first."""
    bad = [n for n, f in fit.items()
           if f.anti_synergy >= ANTI_SYNERGY_CUTOFF or _avg(f) < LOW_FIT_CUTOFF]
    return sorted(bad, key=lambda n: _avg(fit[n]))[:limit]


def bucket(nonland_count: int) -> int:
    """Which re-inference threshold a deck size has crossed (0-3)."""
    return sum(nonland_count >= t for t in RE_INFER_AT)
