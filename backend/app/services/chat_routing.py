"""
Chat routing with Jev: one System One request per chat turn picks the tool to
run (or a plain reply) and the judgments code needs to build its arguments.

Jev judges; code assembles tool inputs and decides whether the judgment is
confident enough to skip the LLM tool call.
Design: docs/superpowers/specs/2026-10-05-jev-expansion-design.md (section 5)
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from typesafe_sdk import Choice, Noul

from app.services import jev
from app.services.deck_fit import role_description
from app.services.guided_builder import _extract_mtg_keywords

logger = logging.getLogger(__name__)

ROUTE_CONFIDENCE = 0.6
WANT_THRESHOLD = 0.5
REPLY = "reply"
COLORS = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}
# The suggest_core role list (the tool description is built from this list)
CORE_ROLES = [
    "threats", "creatures", "removal", "card advantage", "card draw", "counterspells",
    "protection", "ramp", "burn", "recursion", "finishers", "interaction", "discard",
    "lifegain", "graveyard hate", "tutors", "sacrifice outlets", "board wipes",
    "spot removal", "cheap threats", "big threats",
]
DEFAULT_ROLES = ["threats", "removal", "card advantage"]
DEFAULT_COUNT = 6
MAX_COUNT = 20
TURNS = 6
TURN_CHARS = 500
STRATEGY_CHARS = 300
ARCHETYPES = {
    "aggro": "Wins fast with cheap creatures and direct damage",
    "control": "Answers threats with removal, counterspells and card advantage, and wins late",
    "midrange": "Efficient threats plus interaction; wins with sturdy, value-generating cards",
    "combo": "Assembles specific cards that win together",
    "tempo": "Cheap threats backed by cheap disruption to stay ahead",
    "none": "No archetype is stated or implied",
}
REPLY_DESCRIPTION = (
    "Answer in text only: a rules question, a card explanation, thanks, or advice that "
    "needs no card suggestions, deck changes, generated deck, or meta data"
)
PUNCTUATION = ",.!?\"'"


@dataclass
class Route:
    action: str
    confidence: float
    probabilities: Dict[str, float]
    inputs: Dict[str, Dict[str, Any]]  # tool name -> tool input

    def top_tool(self) -> str:
        """The most probable tool, for when a tool call is required but `reply` won."""
        return max(self.inputs, key=lambda name: self.probabilities.get(name, 0.0))


def deck_summary(deck: Optional[Dict[str, Any]], fallback_colors: Sequence[str]) -> Dict[str, Any]:
    """has_deck, unique nonland count, and colors from mana costs (else `fallback_colors`)."""
    entries = (deck or {}).get("main_deck") or []
    nonland = {e.get("card_name") for e in entries
               if "land" not in ((e.get("card") or {}).get("type_line") or "").lower()}
    costs = "".join((e.get("card") or {}).get("mana_cost") or "" for e in entries)
    colors = [c for c in COLORS if f"{{{c}}}" in costs] or list(fallback_colors)
    return {"has_deck": bool(entries), "unique_nonland_cards": len(nonland - {None}), "colors": colors}


def _options(names: Sequence[str], describe: Callable[[str], Any]) -> Dict[str, Any]:
    """Choice criteria; a repeated name (any case) keeps its first description."""
    out: Dict[str, Any] = {}
    seen = set()
    for name in names:
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out[name] = describe(name)
    return out


def build_questions(
    tools: List[Dict[str, Any]],
    card_names: Sequence[str],
    mechanics: Sequence[str],
    meta_archetypes: Sequence[str],
) -> Dict[str, Any]:
    questions: Dict[str, Any] = {
        "action": Choice(
            instructions="What should the deck-building assistant do with `message`, given `recent_turns`, `deck` and `strategy`?",
            criteria={**{t["name"]: t["description"] for t in tools}, REPLY: REPLY_DESCRIPTION},
        ),
        "package_role": Choice(
            instructions="Which role or mechanic does the user want more cards for?",
            criteria=_options([*CORE_ROLES, *mechanics], lambda n: (
                role_description(n) if n in CORE_ROLES else f"Cards with the {n} mechanic")),
        ),
        "archetype": Choice(
            instructions="Which archetype does the user want for their own deck?",
            criteria=ARCHETYPES,
        ),
        "opponent": Choice(
            instructions="Which opponent deck from `meta_archetypes` is the user asking about?",
            criteria={**_options(meta_archetypes, lambda n: None), "none": "No opponent deck is mentioned"},
        ),
    }
    for code, name in COLORS.items():
        questions[f"color:{code}"] = Noul(instructions={
            "question": "Does the user want this color in their own deck (not an opponent's)?",
            "color": name,
        })
    for role in CORE_ROLES:
        questions[f"role:{role}"] = Noul(instructions={
            "question": "Does the user want cards that fill this role now?",
            "role": role_description(role),
        })
    for i in range(len(card_names)):
        questions[f"wants:{i}"] = Noul(
            instructions=f"Does the user want `cards[{i}]` in their deck, rather than only mentioning it?")
    return questions


def _strip_names(text: str, names: Sequence[str]) -> str:
    """Remove capitalized words of `names` from `text`, so a card the user only
    mentioned (an opponent's) can't be re-resolved from the strategy downstream."""
    tokens = {w.strip(PUNCTUATION).lower() for n in names for w in n.split() if w[:1].isupper()}
    return " ".join(w for w in text.split()
                    if not (w[:1].isupper() and w.strip(PUNCTUATION).lower() in tokens))


def tool_inputs(
    resp: Any,
    *,
    message: str,
    summary: Dict[str, Any],
    strategy: str,
    card_names: Sequence[str],
    tools: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Each tool's input, from Jev's judgments plus the message. Raises KeyError
    when Jev left a question unanswered."""
    nouls, choices = resp.nouls, resp.choices
    colors = [c for c in COLORS if nouls[f"color:{c}"].noul >= WANT_THRESHOLD] or summary["colors"]
    roles = [r for r in CORE_ROLES if nouls[f"role:{r}"].noul >= WANT_THRESHOLD] or list(DEFAULT_ROLES)
    wanted = [n for i, n in enumerate(card_names) if nouls[f"wants:{i}"].noul >= WANT_THRESHOLD]
    text = _strip_names(message, [n for n in card_names if n not in wanted])
    found = re.search(r"\d+", message)
    archetype = choices["archetype"].choice
    opponent = choices["opponent"].choice
    args = {
        "focus": "" if archetype == "none" else archetype,
        "strategy": (f"{strategy}; {text}" if strategy else text)[:STRATEGY_CHARS],
        "colors": colors,
        "roles": roles,
        "role": choices["package_role"].choice,
        "count": min(max(int(found.group()), 1), MAX_COUNT) if found else DEFAULT_COUNT,
        "modification": message,
        "opponent_deck": "" if opponent == "none" else opponent,
        "archetype": archetype,
        "specific_cards": wanted,
    }
    inputs = {}
    for tool in tools:
        props = tool["input_schema"].get("properties", {})
        inputs[tool["name"]] = {k: v for k, v in args.items()
                                if k in props and not (k == "archetype" and v == "none")}
    return inputs


async def route(
    *,
    message: str,
    history: List[Dict[str, Any]],
    summary: Dict[str, Any],
    strategy: str,
    card_names: List[str],
    meta_archetypes: List[str],
    tools: List[Dict[str, Any]],
    client=None,
) -> Optional[Route]:
    """One Jev request deciding this turn's action and tool arguments.

    `history` is the conversation's messages, ending with the current one.
    Returns None when Jev is not configured, fails, or answers incompletely;
    the caller then uses the LLM tool call.
    """
    mechanics = _extract_mtg_keywords(message)
    state = {
        "message": message,
        "recent_turns": [{"role": m.get("role", ""), "content": (m.get("content") or "")[:TURN_CHARS]}
                         for m in history[-(TURNS + 1):-1]],
        "deck": summary,
        "strategy": strategy,
        "cards": card_names,
        "meta_archetypes": meta_archetypes,
        "mechanics": mechanics,
    }
    questions = build_questions(tools, card_names, mechanics, meta_archetypes)
    async with jev.session(client) as client:
        if client is None:
            return None
        try:
            async with asyncio.timeout(jev.FIT_DEADLINE):
                resp = await jev.ask(client, state, questions)
            action = resp.choices["action"]
            return Route(
                action=action.choice,
                confidence=action.confidence,
                probabilities=dict(action.probabilities),
                inputs=tool_inputs(resp, message=message, summary=summary, strategy=strategy,
                                   card_names=card_names, tools=tools),
            )
        except Exception as e:  # Jev must never block chat
            logger.warning(f"Jev routing failed, using the LLM tool call: {e}")
            return None
