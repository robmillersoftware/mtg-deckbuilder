"""
Deck fit: Jev (TypeSafe System One) judgments on whether cards fit a deck.

Jev answers narrow questions (theme tags, card centrality, plan fit, synergy,
anti-synergy). Code owns ranking, legality, and metagame strength.
Design: docs/superpowers/specs/2026-10-05-deck-fit-design.md
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field
from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy, Score

from app.core.config import settings

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

REQUEST_TIMEOUT = 2.0
RETRY_BUDGET = 5.0
FIT_DEADLINE = 6.0  # overall cap per Jev operation, across all waves and retries
# ponytail: fixed concurrency cap under Jev's 80 req/s limit; make it adaptive if 429s show up
MAX_CONCURRENT = 32

KEY_CARD_LEVELS = [
    "Incidental: could be swapped for a generic card without changing the plan",
    "Supporting: helps the plan but is replaceable",
    "Important: one of the main ways the deck executes its plan",
    "Defines the plan: the deck is built around this card",
]
PLAN_FIT_LEVELS = [
    "Does nothing for this deck's plan",
    "Incidental help: generically useful, not specific to the plan",
    "Clearly supports the plan",
    "A core piece of the plan",
]
SYNERGY_LEVELS = [
    "No interaction with the key cards",
    "Loose overlap: same general strategy, no direct interaction",
    "Directly creates, uses, or rewards what one key card does",
    "Directly creates, uses, or rewards what several key cards do",
]


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


def _client() -> Optional[AsyncTypeSafeClient]:
    if not settings.TYPESAFE_API_KEY:
        return None
    return AsyncTypeSafeClient(
        api_key=settings.TYPESAFE_API_KEY,
        timeout=REQUEST_TIMEOUT,
        retry=RetryPolicy(api_timeout_error=False, timeout=RETRY_BUDGET),
    )


async def _close(client: Optional[AsyncTypeSafeClient]) -> None:
    """Safely close a client, logging and swallowing any errors."""
    if client is None:
        return
    try:
        await client.aclose()
    except Exception as e:
        logger.warning(f"Failed to close Jev client: {e}")


async def infer_identity(
    cards: List[Dict[str, str]],
    request_text: Optional[str],
    overrides: Optional[IdentityOverrides] = None,
    client=None,
) -> Optional[DeckIdentity]:
    """Theme tags (one Noul each) and key cards (one Score per nonland card), in one request."""
    nonland = [c for c in cards if not is_land(c)]
    if not nonland and not request_text:
        return None

    questions: Dict[str, Any] = {
        f"tag:{tag}": Noul(instructions={
            "question": "Does the deck described by `deck.request` and `deck.cards` build around this theme?",
            "theme": definition,
        })
        for tag, definition in THEME_TAGS.items()
    }
    with_keys = len(nonland) >= RE_INFER_AT[0]
    if with_keys:
        for i in range(len(nonland)):
            questions[f"key:{i}"] = Score(
                instructions=f"How central is `deck.cards[{i}]` to this deck's plan?",
                criteria=KEY_CARD_LEVELS,
            )

    owned = client is None
    client = client or _client()
    if client is None:
        return None
    try:
        async with asyncio.timeout(FIT_DEADLINE):
            resp = await client.system_one(
                {"deck": {"request": request_text or "", "cards": nonland}}, questions)
        tags = [t for t in THEME_TAGS if resp.nouls[f"tag:{t}"].noul >= TAG_THRESHOLD]
        key_cards: List[str] = []
        if with_keys:
            n = 5 if len(nonland) < 40 else 8
            order = sorted(range(len(nonland)), key=lambda i: resp.scores[f"key:{i}"].score, reverse=True)
            key_cards = [nonland[i]["name"] for i in order[:n]]

        identity = DeckIdentity(tags=tags, key_cards=key_cards, request_text=request_text,
                                overrides=overrides or IdentityOverrides())
        return apply_overrides(identity, [c["name"] for c in nonland])
    except Exception as e:  # Jev must never break deck building
        logger.warning(f"Deck identity inference failed: {e}")
        return None
    finally:
        if owned:
            await _close(client)


async def score_fit(
    identity: DeckIdentity,
    key_cards: List[Dict[str, str]],
    candidates: List[Dict[str, str]],
    client=None,
) -> Dict[str, FitScore]:
    """One concurrent Jev request per unique candidate. Empty dict if any request fails."""
    unique = {c["name"]: c for c in candidates}
    if not unique:
        return {}

    deck = {"themes": identity.tags, "request": identity.request_text or "", "key_cards": key_cards}
    questions: Dict[str, Any] = {
        "plan_fit": Score(
            instructions="How well does `candidate` advance the plan described by `deck.themes` and `deck.request`?",
            criteria=PLAN_FIT_LEVELS,
        ),
        "anti_synergy": Noul(instructions=(
            "Does `candidate` actively work against the plan in `deck`, for example exiling "
            "its own graveyard in a graveyard deck or punishing its own token creation?"
        )),
    }
    if key_cards:
        questions["synergy"] = Score(
            instructions="Does `candidate`'s oracle text create, use, or reward what `deck.key_cards` do?",
            criteria=SYNERGY_LEVELS,
        )

    owned = client is None
    client = client or _client()
    if client is None:
        return {}
    gate = asyncio.Semaphore(MAX_CONCURRENT)
    results = {}

    async def one(card: Dict[str, str]) -> None:
        async with gate:
            r = await client.system_one({"deck": deck, "candidate": card}, questions)
        results[card["name"]] = FitScore(
            plan_fit=r.scores["plan_fit"].score / 3,
            synergy=r.scores["synergy"].score / 3 if key_cards else None,
            anti_synergy=r.nouls["anti_synergy"].noul,
        )

    try:
        async with asyncio.timeout(FIT_DEADLINE):
            async with asyncio.TaskGroup() as tg:
                for card in unique.values():
                    tg.create_task(one(card))
    except Exception as e:  # partial fit is worse than none: fall back to existing ordering
        logger.warning(f"Fit scoring failed, falling back: {e}")
        return {}
    finally:
        if owned:
            await _close(client)
    return results


async def load_payloads(db, names: Sequence[str]) -> List[Dict[str, str]]:
    from app.services.card_service import CardService

    found = await CardService(db).get_cards_by_names(list(names))
    seen, payloads = set(), []
    for card in found.values():  # DFC faces map to the same Card; keep one payload each
        if card.name not in seen:
            seen.add(card.name)
            payloads.append(card_payload(card))
    return payloads


async def review_deck(
    db,
    entries: List[Dict[str, Any]],
    request_text: Optional[str],
    overrides: Optional[IdentityOverrides] = None,
    client=None,
) -> Tuple[Optional[DeckIdentity], Dict[str, FitScore]]:
    """Infer a deck's identity and score its nonland cards against it."""
    payloads = await load_payloads(db, [e["card_name"] for e in entries if e.get("card_name")])
    nonland = [p for p in payloads if not is_land(p)]
    if not nonland:
        return None, {}
    identity = await infer_identity(payloads, request_text, overrides, client=client)
    if identity is None:
        return None, {}
    keys = [p for p in nonland if p["name"] in identity.key_cards]
    keys.sort(key=lambda p: identity.key_cards.index(p["name"]))
    fit = await score_fit(identity, keys, nonland, client=client)
    if not fit:  # scoring failed: report unavailable rather than "everything fits"
        return None, {}
    return identity, fit
