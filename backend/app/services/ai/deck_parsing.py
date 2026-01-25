"""Deck request parsing utilities."""

import json
import logging
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.config import settings

logger = logging.getLogger(__name__)


async def parse_deck_request(prompt: str, db: AsyncSession) -> Dict[str, Any]:
    """
    Parse a natural language deck request to extract:
    - Archetype (aggro, control, midrange, combo)
    - Colors
    - Strategy focus
    - Specific card requests

    Uses Haiku for fast parsing (~0.5s vs 2-3s with Sonnet).
    """
    if not settings.ANTHROPIC_API_KEY:
        return await fallback_parse(prompt, db)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            system="""Parse MTG deck request into JSON:
{"archetype": "aggro|control|midrange|combo|tempo", "colors": ["W","U","B","R","G"], "strategy": "brief description", "specific_cards": ["card names mentioned"]}

Color codes: W=White, U=Blue, B=Black, R=Red, G=Green
Guild names: Azorius=WU, Dimir=UB, Rakdos=BR, Gruul=RG, Selesnya=GW, Orzhov=WB, Izzet=UR, Golgari=BG, Boros=RW, Simic=GU""",
            messages=[{"role": "user", "content": prompt}],
        )

        if response.content:
            content = response.content[0].text
            if "{" in content:
                json_start = content.index("{")
                json_end = content.rindex("}") + 1
                return json.loads(content[json_start:json_end])

    except Exception as e:
        logger.warning(f"Haiku parse failed, using fallback: {e}")

    return await fallback_parse(prompt, db)


async def fallback_parse(prompt: str, db: AsyncSession) -> Dict[str, Any]:
    """Simple keyword-based parsing as fallback."""
    prompt_lower = prompt.lower()

    # Detect colors
    colors = []
    color_keywords = {
        "white": "W", "plains": "W",
        "blue": "U", "island": "U",
        "black": "B", "swamp": "B",
        "red": "R", "mountain": "R",
        "green": "G", "forest": "G",
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
        "mono-red": ["R"],
        "mono-white": ["W"],
        "mono-blue": ["U"],
        "mono-black": ["B"],
        "mono-green": ["G"],
    }

    for keyword, color in color_keywords.items():
        if keyword in prompt_lower:
            if isinstance(color, list):
                colors.extend(color)
            else:
                colors.append(color)
    colors = list(set(colors))

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
    specific_cards = await extract_card_names_from_prompt(prompt, db)

    return {
        "archetype": archetype,
        "colors": colors or ["R"],
        "strategy": prompt,
        "specific_cards": specific_cards,
        "budget": "competitive",
        "focus": "speed" if archetype == "aggro" else "value",
    }


async def extract_card_names_from_prompt(
    prompt: str, db: AsyncSession, format: str = "standard"
) -> List[str]:
    """Extract card names mentioned in the prompt by checking against database."""
    from app.models.card import Card

    specific_cards = []
    words = prompt.split()
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
