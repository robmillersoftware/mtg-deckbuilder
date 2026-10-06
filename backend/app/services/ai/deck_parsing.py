"""Deck request parsing utilities."""

import asyncio
import logging
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typesafe_sdk import Choice, Noul

from app.services import jev

logger = logging.getLogger(__name__)


ARCHETYPES = {
    "aggro": "Wins fast with cheap creatures and direct damage",
    "control": "Answers threats with removal, counterspells and card advantage, and wins late",
    "midrange": "Efficient threats plus interaction; trades resources and wins with sturdy, value-generating cards",
    "combo": "Assembles specific cards that win together",
    "tempo": "Cheap threats backed by cheap disruption to stay ahead",
}
COLOR_NAMES = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}
WANT_THRESHOLD = 0.5


def _parse_questions(n_cards: int) -> Dict[str, Any]:
    questions: Dict[str, Any] = {
        "archetype": Choice(
            instructions="Which archetype does `request` want for the user's own deck?",
            criteria=ARCHETYPES,
        ),
        "colors_specified": Noul(
            instructions="Did the user state colors for their own deck, not an opponent's?"),
    }
    for code, name in COLOR_NAMES.items():
        questions[f"color:{code}"] = Noul(instructions={
            "question": "Does the user want this color in their own deck (not an opponent's)?",
            "color": name,
        })
    for i in range(n_cards):
        questions[f"wants:{i}"] = Noul(
            instructions=f"Does the user want `cards[{i}]` in their deck, rather than only mentioning it?")
    return questions


async def parse_deck_request(prompt: str, db: AsyncSession, client=None) -> Dict[str, Any]:
    """
    Parse a natural-language deck request with one Jev request into
    {archetype, colors, colors_specified, strategy, specific_cards}.

    Card names are resolved from the database first (word n-grams of the prompt);
    Jev decides which of them the user wants in their deck and which colors are
    theirs rather than an opponent's. Falls back to keyword parsing when Jev is
    not configured, fails, or answers incompletely.
    """
    async with jev.session(client) as client:
        names = None
        if client is not None:
            names = await extract_card_names_from_prompt(prompt, db)
            try:
                async with asyncio.timeout(jev.FIT_DEADLINE):
                    r = await jev.ask(client, {"request": prompt, "cards": names},
                                      _parse_questions(len(names)))
                colors = [c for c in COLOR_NAMES if r.nouls[f"color:{c}"].noul >= WANT_THRESHOLD]
                if r.nouls["colors_specified"].noul < WANT_THRESHOLD:
                    colors = []  # e.g. "beat mono-red": Jev leans toward non-red colors, but none were asked for
                return {
                    "archetype": r.choices["archetype"].choice,
                    "colors": colors,
                    "colors_specified": bool(colors),
                    "strategy": prompt,
                    "specific_cards": [n for i, n in enumerate(names)
                                       if r.nouls[f"wants:{i}"].noul >= WANT_THRESHOLD],
                }
            except Exception as e:
                logger.warning(f"Jev deck-request parse failed, using fallback: {e}")
    return await fallback_parse(prompt, db, names)


async def fallback_parse(prompt: str, db: AsyncSession, names: List[str] = None) -> Dict[str, Any]:
    """Simple keyword-based parsing as fallback. `names` skips the card lookup when already resolved."""
    prompt_lower = prompt.lower()

    # Detect colors
    colors = []
    color_keywords = {
        "white": "W", "plains": "W",
        "blue": "U", "island": "U",
        "black": "B", "swamp": "B",
        "red": "R", "mountain": "R",
        "green": "G", "forest": "G",
        # Guild names (2-color)
        "azorius": ["W", "U"],
        "dimir": ["U", "B"],
        "rakdos": ["B", "R"],
        "gruul": ["R", "G"],
        "selesnya": ["G", "W"],
        "orzhov": ["W", "B"],
        "izzet": ["U", "R"],
        "golgari": ["B", "G"],
        "boros": ["R", "W"],
        "simic": ["G", "U"],
        # Shard names (3-color)
        "esper": ["W", "U", "B"],
        "grixis": ["U", "B", "R"],
        "jund": ["B", "R", "G"],
        "naya": ["R", "G", "W"],
        "bant": ["G", "W", "U"],
        # Wedge names (3-color)
        "abzan": ["W", "B", "G"],
        "jeskai": ["U", "R", "W"],
        "sultai": ["B", "G", "U"],
        "mardu": ["R", "W", "B"],
        "temur": ["G", "U", "R"],
        # Mono-color
        "mono-red": ["R"],
        "mono-white": ["W"],
        "mono-blue": ["U"],
        "mono-black": ["B"],
        "mono-green": ["G"],
        "monored": ["R"],
        "monowhite": ["W"],
        "monoblue": ["U"],
        "monoblack": ["B"],
        "monogreen": ["G"],
    }

    # Track if colors were mentioned in a "build this" context vs "beat this" context
    colors_specified = False
    negative_context_patterns = ["beat", "against", "counter", "hate", "matchup", "versus", "vs"]
    has_negative_context = any(pattern in prompt_lower for pattern in negative_context_patterns)

    for keyword, color in color_keywords.items():
        if keyword in prompt_lower:
            # Check if this color mention is in a negative context
            # Simple heuristic: if "beat/against/etc" appears before the color, it's likely the opponent's color
            keyword_pos = prompt_lower.find(keyword)
            in_negative_context = False
            for pattern in negative_context_patterns:
                pattern_pos = prompt_lower.find(pattern)
                if pattern_pos != -1 and pattern_pos < keyword_pos:
                    in_negative_context = True
                    break

            if not in_negative_context:
                if isinstance(color, list):
                    colors.extend(color)
                else:
                    colors.append(color)
                colors_specified = True  # At least one color is in positive context

    colors = list(set(colors))
    # If all color mentions were in negative context, colors_specified should be False
    if not colors:
        colors_specified = False

    # Detect archetype
    archetype = "midrange"
    if any(word in prompt_lower for word in ["aggro", "aggressive", "fast", "burn", "rush"]):
        archetype = "aggro"
    elif any(word in prompt_lower for word in ["control", "counter", "remove", "wrath"]):
        archetype = "control"
    elif any(word in prompt_lower for word in ["combo", "infinite", "win condition"]):
        archetype = "combo"
    elif any(word in prompt_lower for word in ["tempo", "disruptive"]):
        archetype = "tempo"

    # Try to extract specific card names from the prompt
    specific_cards = names if names is not None else await extract_card_names_from_prompt(prompt, db)

    return {
        "archetype": archetype,
        "colors": colors or ["R"],
        "colors_specified": colors_specified,  # True if colors were mentioned in positive context
        "strategy": prompt,
        "specific_cards": specific_cards,
        "budget": "competitive",
        "focus": "speed" if archetype == "aggro" else "value",
    }


MAX_PROMPT_WORDS = 40  # a pasted decklist must not fan out into hundreds of queries
MAX_CARD_NAMES = 10


async def extract_card_names_from_prompt(
    prompt: str, db: AsyncSession, format: str = "standard"
) -> List[str]:
    """Extract card names mentioned in the prompt by checking against database."""
    from app.models.card import Card

    specific_cards = []
    words = prompt.split()[:MAX_PROMPT_WORDS]
    potential_names = []

    # Try to find multi-word card names
    for i in range(len(words)):
        # Single word
        if words[i][0:1].isupper() or words[i].lower() in [
            "tezzeret", "jace", "liliana", "chandra", "nissa",
            "garruk", "ajani", "nicol", "bolas", "atraxa"
        ]:
            potential_names.append(words[i].strip(",.!?"))
        # Two words
        if i < len(words) - 1:
            two_word = f"{words[i]} {words[i+1]}".strip(",.!?")
            potential_names.append(two_word)
        # Three words
        if i < len(words) - 2:
            three_word = f"{words[i]} {words[i+1]} {words[i+2]}".strip(",.!?")
            potential_names.append(three_word)
        # Four words (for cards like "Atraxa, Grand Unifier")
        if i < len(words) - 3:
            four_word = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}".strip(",.!?")
            potential_names.append(four_word)

    # Check each potential name against the database
    from app.services.card_service import get_format_legality_condition

    for name in potential_names:
        if len(specific_cards) >= MAX_CARD_NAMES:
            break
        if len(name) < 3:
            continue
        query = select(Card.name).where(
            func.lower(Card.name).like(f"%{name.lower()}%"),
            get_format_legality_condition(format)
        )
        query = query.limit(1)
        result = await db.execute(query)
        card = result.scalar_one_or_none()
        if card and card not in specific_cards:
            specific_cards.append(card)
            logger.info(f"Extracted card name from prompt: {card}")

    return specific_cards


async def get_commander_color_identity(card_name: str, db: AsyncSession) -> List[str]:
    """Get a commander's color identity from the database with fuzzy matching."""
    from app.models.card import Card
    from rapidfuzz import fuzz

    # 1. Try exact match
    query = select(Card).where(
        func.lower(Card.name) == card_name.lower()
    ).limit(1)
    result = await db.execute(query)
    card = result.scalar_one_or_none()

    if not card:
        # 2. Try partial/substring match
        query = select(Card).where(
            func.lower(Card.name).like(f"%{card_name.lower()}%")
        ).limit(1)
        result = await db.execute(query)
        card = result.scalar_one_or_none()

    if not card:
        # 3. Try fuzzy match
        first_word = card_name.split()[0].split(",")[0] if card_name else ""
        if first_word and len(first_word) >= 3:
            query = select(Card).where(
                func.lower(Card.name).like(f"{first_word.lower()}%")
            ).limit(20)
            result = await db.execute(query)
            candidates = result.scalars().all()

            if candidates:
                best_match = None
                best_score = 0
                for c in candidates:
                    score = fuzz.ratio(card_name.lower(), c.name.lower())
                    if score > best_score and score >= 70:
                        best_score = score
                        best_match = c
                if best_match:
                    logger.info(f"Fuzzy matched '{card_name}' to '{best_match.name}' (score: {best_score})")
                    card = best_match

    if card and card.color_identity:
        logger.info(f"Found color identity for {card.name}: {card.color_identity}")
        return card.color_identity
    elif card and card.colors:
        logger.info(f"Using colors for {card.name}: {card.colors}")
        return card.colors

    logger.warning(f"Could not find color identity for {card_name}, defaulting to WUBRG")
    return ["W", "U", "B", "R", "G"]
