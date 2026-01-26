from typing import Optional, List, Dict, Any
from collections import defaultdict
import logging
import json
import time

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.services.card_service import CardService
from app.services.deck_validator import BASIC_LANDS
from app.models.meta import Decklist, Event, CardCooccurrence


from app.services.ai.mana_base import ManaBaseMixin
from app.services.ai.deck_validation import DeckValidationMixin
from app.services.ai.metagame import MetagameMixin
from app.services.ai.card_selection import CardSelectionMixin
from app.services.ai.json_helpers import repair_json, extract_deck_from_malformed_json
from app.services.ai.deck_parsing import (
    parse_deck_request,
    fallback_parse,
    extract_card_names_from_prompt,
    get_commander_color_identity,
)
logger = logging.getLogger(__name__)


# Maximum number of full decklist examples to include in prompt
MAX_DECKLIST_EXAMPLES = 3

# Cache for tournament cards (shared across requests)
_tournament_cards_cache: Dict[str, Any] = {
    "data": None,
    "timestamp": 0,
    "ttl": 300,  # 5 minutes
}


class AIService(ManaBaseMixin, DeckValidationMixin, MetagameMixin, CardSelectionMixin):
    """
    Service for AI-powered deck building using Claude API.
    Implements the constrained card selection system to prevent hallucination.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.card_service = CardService(db)


    # Delegate to module functions
    def _repair_json(self, json_str: str) -> str:
        return repair_json(json_str)

    def _extract_deck_from_malformed_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        return extract_deck_from_malformed_json(json_str)

    async def parse_deck_request(self, prompt: str) -> Dict[str, Any]:
        return await parse_deck_request(prompt, self.db)

    async def _fallback_parse(self, prompt: str) -> Dict[str, Any]:
        return await fallback_parse(prompt, self.db)

    async def _extract_card_names_from_prompt(self, prompt: str, format: str = "standard") -> List[str]:
        return await extract_card_names_from_prompt(prompt, self.db, format)

    async def get_commander_color_identity(self, card_name: str) -> List[str]:
        return await get_commander_color_identity(card_name, self.db)

    async def generate_deck(
        self,
        archetype: str,
        colors: List[str],
        strategy: str,
        meta_context: Dict[str, Any],
        include_sideboard: bool = True,
        specific_cards: List[str] = None,
        archetype_template: Dict[str, Any] = None,
        format: str = "standard",
    ) -> Dict[str, Any]:
        """
        Generate a complete deck using AI with constrained card selection.

        archetype_template: Optional dict with role distribution targets from tournament data:
            - archetype_category: aggro/midrange/control/combo
            - sample_size: number of tournament decks analyzed
            - avg_lands: average land count
            - avg_nonlands: average nonland count
            - role_distribution: {role: avg_count} targets
        format: Game format (standard, historic, modern, legacy, cedh)
        """
        specific_cards = specific_cards or []

        if not settings.ANTHROPIC_API_KEY:
            return await self._generate_fallback_deck(archetype, colors, strategy, format=format)

        try:
            import asyncio
            import anthropic

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            # Phase 1: Handle specific cards and strategy-based archetype matching (may change colors)
            # NOTE: For cEDH, colors come from the commander's color identity and should NOT be overridden
            template_decks = []
            template_cards = []
            detected_themes = []
            preserve_colors = format == "cedh"  # Commander color identity must be preserved
            logger.debug(f"[AI-SERVICE] Specific cards from parse: {specific_cards}")
            logger.debug(f"[AI-SERVICE] Format: {format}, preserve_colors: {preserve_colors}")
            if specific_cards:
                template_decks = await self._find_decks_with_cards(specific_cards, format=format)
                logger.debug(f"[AI-SERVICE] Found {len(template_decks)} tournament decks with specific cards")
                if template_decks:
                    if not preserve_colors:
                        template_colors = self._extract_colors_from_decks(template_decks)
                        if template_colors:
                            logger.debug(f"[AI-SERVICE] Overriding colors from {colors} to {template_colors} based on tournament decks")
                            colors = template_colors
                    template_cards = self._extract_cards_from_decks(template_decks)
                elif format != "cedh":
                    # Theme detection only for 60-card formats, not cEDH
                    # cEDH builds around commander + combos, not casual themes
                    detected_themes = await self._detect_card_themes(specific_cards)
                    logger.debug(f"[AI-SERVICE] Detected themes: {detected_themes}")
                    if detected_themes:
                        template_cards = await self._get_tournament_synergy_cards(detected_themes, format=format)
                        logger.debug(f"[AI-SERVICE] Found {len(template_cards)} tournament-played synergy cards")

            # Phase 1b: If no specific cards, try to find archetype decklists based on strategy
            # This allows strategy keywords like "graveyard" to find "Reanimator" decks and use their colors
            # NOTE: Skip color override for cEDH - commander color identity is fixed
            if not template_decks and strategy:
                strategy_decks = await self._get_archetype_decklists(archetype, [], strategy, limit=5, format=format)
                if strategy_decks:
                    logger.debug(f"[AI-SERVICE] Found {len(strategy_decks)} tournament decks matching strategy '{strategy}'")
                    if not preserve_colors:
                        strategy_colors = self._extract_colors_from_decks(strategy_decks)
                        if strategy_colors:
                            logger.debug(f"[AI-SERVICE] Overriding colors from {colors} to {strategy_colors} based on strategy-matched decks")
                            colors = strategy_colors
                    template_decks = strategy_decks
                    template_cards = self._extract_cards_from_decks(strategy_decks)

            # Phase 2: Run independent queries in parallel
            parallel_tasks = {
                'available_cards': self._get_available_cards(colors, format),
                'sideboard_patterns': self._get_sideboard_patterns(archetype, colors, format=format),
                'composition': self._get_deck_composition_from_meta(archetype, colors, format=format),
                'mana_base_data': self._get_mana_base_from_meta(archetype, colors, format=format),
                'semantic_cards': self._semantic_card_search(strategy, colors, format=format),
                'tournament_cards': self._get_all_tournament_cards(format=format),
            }

            # Add conditional parallel tasks
            if specific_cards:
                parallel_tasks['cooccurrence_cards'] = self._get_cooccurrence_cards(specific_cards, colors, format=format)
            if not template_cards:
                parallel_tasks['meta_cards'] = self._get_meta_cards(archetype, strategy, format=format)
            if not template_decks and archetype:
                parallel_tasks['example_decklists'] = self._get_archetype_decklists(archetype, colors, strategy, format=format)

            # Execute all queries in parallel
            task_names = list(parallel_tasks.keys())
            results = await asyncio.gather(*parallel_tasks.values(), return_exceptions=True)
            parallel_results = {}
            for name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    logger.warning(f"Parallel query {name} failed: {result}")
                    parallel_results[name] = [] if name != 'composition' else {'sample_size': 0}
                else:
                    parallel_results[name] = result

            # Unpack results
            available_cards = parallel_results.get('available_cards', [])
            sideboard_patterns = parallel_results.get('sideboard_patterns', {})
            composition = parallel_results.get('composition', {'sample_size': 0})
            mana_base_data = parallel_results.get('mana_base_data', {})
            semantic_cards = parallel_results.get('semantic_cards', [])
            tournament_cards = parallel_results.get('tournament_cards', [])
            cooccurrence_cards = parallel_results.get('cooccurrence_cards', [])
            meta_cards = template_cards if template_cards else parallel_results.get('meta_cards', [])
            example_decklists = template_decks if template_decks else parallel_results.get('example_decklists', [])

            logger.debug(f"[AI-SERVICE] Parallel queries complete: {len(available_cards)} available, {len(tournament_cards)} tournament cards")

            # Add template cards to available cards if needed
            if template_cards:
                template_card_names = {c["name"] for c in template_cards}
                existing_names = {c["name"] for c in available_cards}
                missing_names = template_card_names - existing_names
                if missing_names:
                    from app.models.card import Card
                    from sqlalchemy import func
                    from app.services.card_service import get_format_legality_condition

                    use_format_legality = format in ["cedh", "commander", "legacy", "modern", "historic"]
                    if use_format_legality:
                        missing_query = select(Card).where(
                            func.lower(Card.name).in_([n.lower() for n in missing_names]),
                            get_format_legality_condition(format)
                        )
                    else:
                        missing_query = select(Card).where(
                            func.lower(Card.name).in_([n.lower() for n in missing_names]),
                            Card.is_standard_legal == True
                        )
                    missing_result = await self.db.execute(missing_query)
                    missing_cards_raw = missing_result.scalars().all()
                    seen = set()
                    for c in missing_cards_raw:
                        if c.name not in seen:
                            seen.add(c.name)
                            available_cards.append({
                                "name": c.name,
                                "mana_cost": c.mana_cost,
                                "type_line": c.type_line,
                                "oracle_text": c.oracle_text,
                                "cmc": float(c.cmc) if c.cmc else 0,
                            })

            recommended_lands = mana_base_data.get("recommended_lands", [])

            # Format the land recommendations
            if recommended_lands:
                land_recommendations = "\n".join(
                    f"- {land['name']}: {land['recommended_quantity']} copies (used in {land['frequency']} decks)"
                    for land in recommended_lands
                )
            else:
                land_recommendations = "No specific land data available - use appropriate dual lands and basics."

            # Build specific cards requirement
            specific_cards_text = ""
            if specific_cards:
                specific_cards_text = f"""
REQUIRED CARDS - The user specifically requested these cards be included:
{chr(10).join(f"- {card}" for card in specific_cards)}
You MUST include these cards in the deck (use 4 copies unless the card is legendary or there's a good reason for fewer).
"""

            # Format full decklist examples (already fetched in parallel)
            decklist_examples_text = ""
            if example_decklists:
                decklist_examples_text = f"""
WINNING TOURNAMENT DECKLISTS - Study and emulate these actual winning decklists:
{self._format_decklists_as_examples(example_decklists)}

Use these decklists as your primary reference. Copy their card choices and quantities.
"""

            # Build format-aware prompt sections from the queried data
            # All queries are now format-aware, so they return data for the specified format
            meta_cards_text = ""
            if template_cards and template_decks:
                # We found tournament decks with the requested cards - use them as strong guidance
                meta_cards_text = f"""
TOURNAMENT DECK TEMPLATE - These cards are played in {len(template_decks)} tournament decks that use the same cards you requested:
{chr(10).join(f"- {c['name']}: typically {c['recommended_quantity']} copies (played in {c['frequency']}/{len(template_decks)} decks)" for c in template_cards[:60])}

BUILD THE DECK USING THESE CARDS. This is what competitive players actually use with the requested cards.
"""
            elif template_cards and detected_themes:
                # We found tournament-played synergy cards
                theme_str = ", ".join(detected_themes)
                meta_cards_text = f"""
TOURNAMENT-PLAYED {theme_str.upper()} CARDS - These cards match your theme AND see actual tournament play:
{chr(10).join(f"- {c['name']}: {c['recommended_quantity']} copies (played in {c['frequency']} tournament decks)" for c in template_cards[:40])}

USE THESE CARDS. They are competitively proven for the {theme_str} theme.
"""
            elif meta_cards:
                meta_cards_text = f"""
COMPETITIVELY PROVEN CARDS - These cards are commonly played in tournament {archetype} decks:
{chr(10).join(f"- {c['name']}: typically {c['recommended_quantity']} copies (played in {c['frequency']} decks)" for c in meta_cards[:50])}

Prioritize these cards when building the deck - they have proven competitive performance.
"""

            # Format co-occurrence synergy section
            cooccurrence_text = ""
            if cooccurrence_cards:
                cooccurrence_text = f"""
SYNERGY CARDS (cards that frequently appear with your requested cards in winning decks):
{chr(10).join(f"- {c['name']} (appeared together {c['cooccurrence_count']} times)" for c in cooccurrence_cards[:20])}

These cards have proven synergy - prioritize them.
"""

            # Format semantic search results
            semantic_text = ""
            if semantic_cards:
                semantic_text = f"""
STRATEGY-RELEVANT CARDS (semantically matched to your request):
{chr(10).join(f"- {c['name']}: {c['type_line']}" for c in semantic_cards[:15])}
"""

            # Format sideboard patterns (skip for Commander formats which have no sideboard)
            sideboard_guide_text = ""
            if format not in ["cedh", "commander"] and sideboard_patterns.get("sideboard_staples"):
                sideboard_guide_text = f"""
SIDEBOARD GUIDE - These cards appear most frequently in tournament sideboards for this archetype:
{chr(10).join(f"- {c['name']}: {c['avg_quantity']} copies (in {c['frequency']}/{c['total_decks']} decks)" for c in sideboard_patterns['sideboard_staples'][:15])}

Use these as your sideboard template - they are what competitive players actually use.
"""

            # Format deck composition guidance
            composition_text = ""
            if composition.get("sample_size", 0) > 0:
                composition_text = f"""
DECK COMPOSITION (based on {composition['sample_size']} tournament {archetype} decks):
- Creatures: ~{composition['avg_creatures']} cards
- Spells (instants/sorceries/enchantments/artifacts/planeswalkers): ~{composition['avg_spells']} cards
- Lands: ~{composition['avg_lands']} cards

Follow this composition - it's what winning decks actually use.
"""

            # Format role distribution guidance from archetype template
            role_distribution_text = ""
            if archetype_template and archetype_template.get("role_distribution"):
                role_dist = archetype_template["role_distribution"]
                category = archetype_template.get("archetype_category", archetype)
                sample_size = archetype_template.get("sample_size", 0)
                avg_lands = archetype_template.get("avg_lands", 24)

                # Get cards grouped by role for actionable guidance
                # Pass archetype category to filter tournament decks by similar archetypes
                role_cards = await self._get_cards_by_role(colors, role_dist, category)
                logger.info(f"Found cards for {len(role_cards)} roles (filtered by {category} archetypes)")

                # Human-readable role names
                role_labels = {
                    "threat_cheap": "CHEAP THREATS (CMC ≤2)",
                    "threat_midrange": "MIDRANGE THREATS (CMC 3-4)",
                    "threat_finisher": "FINISHERS (CMC 5+)",
                    "removal_targeted": "TARGETED REMOVAL",
                    "removal_mass": "BOARD WIPES",
                    "removal_artifact_enchantment": "ARTIFACT/ENCHANTMENT REMOVAL",
                    "card_draw": "CARD DRAW",
                    "card_selection": "CARD SELECTION",
                    "ramp": "MANA RAMP",
                    "counterspell": "COUNTERSPELLS",
                    "discard": "DISCARD",
                    "burn": "BURN/DIRECT DAMAGE",
                    "protection": "PROTECTION",
                    "lifegain": "LIFEGAIN",
                    "recursion": "RECURSION",
                    "graveyard_hate": "GRAVEYARD HATE",
                    "tutor": "TUTORS",
                }

                # Build role-based card selection sections
                role_sections = []
                for role, target_count in sorted(role_dist.items(), key=lambda x: -x[1]):
                    if role.startswith("land_"):
                        continue  # Skip lands, handled separately
                    if target_count < 1.0:
                        continue  # Skip minor roles

                    label = role_labels.get(role, role.replace("_", " ").upper())
                    cards_for_role = role_cards.get(role, [])

                    if cards_for_role:
                        # Show tournament-proven cards first with their counts
                        proven_cards = [c for c in cards_for_role if c.get("tournament_count", 0) > 0]
                        other_cards = [c for c in cards_for_role if c.get("tournament_count", 0) == 0]

                        card_list = []
                        for c in proven_cards[:10]:
                            card_list.append(f"{c['name']} ({c['tournament_count']} decks)")
                        # Add a few non-tournament cards as alternatives
                        for c in other_cards[:3]:
                            card_list.append(c["name"])

                        role_sections.append(
                            f"{label} (target: ~{int(target_count)} cards):\n"
                            f"  PROVEN: {', '.join(card_list)}"
                        )
                    else:
                        role_sections.append(f"{label} (target: ~{int(target_count)} cards)")

                total_deck_size = 99 if format == "cedh" else 60
                role_distribution_text = f"""
=== ROLE-BASED DECK CONSTRUCTION (from {sample_size} winning {category.upper()} decks) ===

*** LAND COUNT: APPROXIMATELY {int(avg_lands)} LANDS ***
Tournament {category} decks run approximately {int(avg_lands)} lands.
You have {total_deck_size - int(avg_lands)} slots for non-land cards.

Fill these functional slots from the options listed:

{chr(10).join(role_sections)}

REMINDER: ~{int(avg_lands)} LANDS + ~{total_deck_size - int(avg_lands)} NON-LANDS = {total_deck_size} CARDS TOTAL.
"""
                logger.info(f"Using {category} role distribution from {sample_size} tournament decks")

            # Log what we're sending to the AI
            logger.info(f"Sending {len(available_cards)} available cards to AI for {archetype} deck")
            if available_cards:
                logger.info(f"Sample available cards: {[c['name'] for c in available_cards[:10]]}")

            # Filter tournament cards (already fetched in parallel) to deck's colors
            tournament_card_names = {c.lower() for c in tournament_cards}
            on_color_tournament = await self._filter_tournament_cards_by_color(tournament_card_names, colors)
            logger.debug(f"[AI-SERVICE] {len(on_color_tournament)} tournament cards match colors {colors}")

            # Build format-specific rules
            if format == "cedh":
                from app.services.cedh_knowledge import get_cedh_system_prompt, CEDH_STAPLES

                # Use actual tournament data for land count when available
                # Priority: archetype_template > composition from meta > reasonable default
                if archetype_template and archetype_template.get('avg_lands'):
                    target_lands = int(archetype_template['avg_lands'])
                    logger.debug(f"[AI-SERVICE] Using archetype template land count: {target_lands}")
                elif composition.get('sample_size', 0) > 0:
                    target_lands = composition.get('avg_lands', 29)
                    logger.debug(f"[AI-SERVICE] Using tournament meta land count: {target_lands} (from {composition['sample_size']} decks)")
                else:
                    # Fallback only when no tournament data available
                    target_lands = 29
                    logger.debug(f"[AI-SERVICE] No tournament data, using default land count: {target_lands}")

                # Get commander name from specific_cards if available
                commander_name = specific_cards[0] if specific_cards else None

                # Get comprehensive cEDH knowledge (now with strategy awareness)
                cedh_knowledge = get_cedh_system_prompt(
                    commander=commander_name,
                    colors=colors,
                    strategy=strategy
                )

                format_header = f"""You are Spellbook, an expert cEDH (competitive Commander) deck builder.

{cedh_knowledge}

YOU MUST BUILD A COMPLETE 99-CARD SINGLETON DECK (no sideboard).
Commander: {commander_name or 'Not specified'}
Colors: {', '.join(colors)}"""

                format_rules = f"""DECK BUILDING RULES (cEDH / Commander):

*** CRITICAL: YOU MUST OUTPUT EXACTLY 99 CARDS. NOT 80, NOT 90, EXACTLY 99. ***
Count your cards before outputting. If you have fewer than 99, add more staples.

1. Main deck = EXACTLY 99 cards (the commander is separate, not counted)
2. NO sideboard - Commander format does not use sideboards
3. SINGLETON: Maximum 1 copy of each card (except basic lands)
4. LANDS: approximately {int(target_lands)} lands (based on tournament data)
5. ONLY use cards legal in Commander format (NOT BANNED)
6. Every card's color identity must fit within colors {colors}
7. Include ALL applicable cEDH staples listed above (free counters, tutors, fast mana)
8. Include a clear win condition (usually Thassa's Oracle + Demonic Consultation/Tainted Pact)
9. If running Tainted Pact: NO duplicate card names (use 1 of each basic, or snow + regular)
10. BANNED CARDS (DO NOT USE): Mana Crypt, Jeweled Lotus, Flash, Paradox Engine
11. DO NOT include casual/"fun" cards - every card must be competitively viable

CARD COUNT CHECKLIST (must add up to 99):
- ~{target_lands} lands (fetches, duals, shocks, rainbow lands)
- ~12 mana rocks/dorks
- ~12 counterspells (free counters + cheap counters)
- ~10 tutors
- ~8 card draw engines
- ~6 win condition pieces
- ~6 removal/interaction
- ~{99 - target_lands - 54} flex slots (hatebears, protection)
= 99 TOTAL"""

                format_json = """Return JSON with reasoning for key cards:
{{
    "name": "Deck Name",
    "strategy_summary": "2-3 sentence strategy description including win condition",
    "main_deck": [
        {{"card_name": "Card Name", "quantity": 1, "reason": "Why this card"}}
    ],
    "sideboard": []
}}"""
            else:
                target_lands = archetype_template.get('avg_lands', composition.get('avg_lands', 24)) if archetype_template else composition.get('avg_lands', 24)
                format_header = f"""You are Spellbook, an expert Magic: The Gathering deck builder for Standard format.

YOU MUST BUILD A COMPLETE 60-CARD DECK + 15-CARD SIDEBOARD.

Colors: {', '.join(colors)}"""

                format_rules = f"""DECK BUILDING RULES:
1. Main deck = EXACTLY 60 cards
2. Sideboard = EXACTLY 15 cards
3. *** LANDS: EXACTLY {target_lands} LANDS *** (this is what tournament decks use - do NOT use fewer!)
4. Max 4 copies of non-basic cards
5. ONLY use cards from the PROVEN tournament lists above - these are what actually win
6. Every non-land card must match colors {colors}
7. For {len(colors)}-color decks: {"Use mostly basic lands. Fetch lands and fixing are unnecessary." if len(colors) == 1 else "Use appropriate dual lands for fixing."}
8. EVERY card must have a clear purpose - if you can't explain why it's in the deck, don't include it"""

                format_json = """Return JSON with reasoning for key cards:
{{
    "name": "Deck Name",
    "strategy_summary": "2-3 sentence strategy description",
    "main_deck": [
        {{"card_name": "Card Name", "quantity": 4, "reason": "Why this card"}}
    ],
    "sideboard": [
        {{"card_name": "Card Name", "quantity": 3, "reason": "What matchups"}}
    ]
}}"""

            # Build tournament data sections
            # For cEDH, the cedh_knowledge prompt already provides comprehensive staple/mana guidance
            # For other formats, use tournament data if available
            if format == "cedh":
                tournament_section = ""  # cedh_knowledge in format_header provides comprehensive guidance
                mana_section = ""  # cedh_knowledge handles mana base recommendations
            else:
                tournament_section = ""
                if on_color_tournament:
                    tournament_section = f"""
TOURNAMENT-PLAYED CARDS IN YOUR COLORS (USE THESE):
{chr(10).join(f"- {name}" for name in on_color_tournament[:150])}
"""
                mana_section = ""
                if mana_base_data.get('sample_size', 0) > 0:
                    mana_section = f"""
MANA BASE from {mana_base_data.get('sample_size', 0)} tournament decks:
{land_recommendations}
"""

            # For cEDH, don't pass strategy/archetype - just build the best pile of cards
            if format == "cedh":
                user_request_text = f"Build a cEDH deck for {commander_name or 'this commander'} in colors {', '.join(colors)}"
                user_message = f"Build the most competitive cEDH deck possible. Ignore any themes - just play the best cards."
            else:
                user_request_text = f"USER REQUEST: {archetype} deck"
                user_message = strategy

            system_prompt = f"""{format_header}
{specific_cards_text}
{decklist_examples_text}
{meta_cards_text}
{cooccurrence_text}
{semantic_text}
{composition_text}
{role_distribution_text}
{sideboard_guide_text}
{tournament_section}
{mana_section}
{format_rules}

{user_request_text}

{format_json}"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            if not response.content:
                logger.error("AI returned empty content")
                raise ValueError("Empty AI response")

            content = response.content[0].text

            # Extract JSON with error recovery
            if "{" in content:
                json_start = content.index("{")
                json_end = content.rindex("}") + 1
                json_str = content[json_start:json_end]

                try:
                    deck_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error: {e}, attempting repair")
                    # Try to repair common JSON issues
                    repaired = self._repair_json(json_str)
                    try:
                        deck_data = json.loads(repaired)
                        logger.info("JSON repair successful")
                    except json.JSONDecodeError as e2:
                        logger.error(f"JSON repair failed: {e2}")
                        # Last resort: try to extract just the main_deck array
                        deck_data = self._extract_deck_from_malformed_json(json_str)
                        if not deck_data:
                            raise ValueError(f"Could not parse AI response: {e2}")

                # Helper to validate card exists with proper format legality
                # Direct database lookup - no pre-fetching needed
                async def find_valid_card(card_name: str) -> Optional[str]:
                    from app.models.card import Card
                    from app.services.card_service import FORMAT_LEGALITY_MAP, get_format_legality_condition
                    from sqlalchemy import func

                    # Direct database lookup for exact name match with format legality
                    query = select(Card.name).where(
                        func.lower(Card.name) == card_name.lower(),
                    )
                    if format in FORMAT_LEGALITY_MAP:
                        query = query.where(get_format_legality_condition(format))
                    else:
                        query = query.where(Card.is_standard_legal == True)
                    query = query.limit(1)
                    result = await self.db.execute(query)
                    db_name = result.scalar_one_or_none()
                    if db_name:
                        return db_name

                    # Try fuzzy/partial match in database as fallback
                    query = select(Card.name).where(
                        func.lower(Card.name).like(f"%{card_name.lower()}%"),
                    )
                    if format in FORMAT_LEGALITY_MAP:
                        query = query.where(get_format_legality_condition(format))
                    else:
                        query = query.where(Card.is_standard_legal == True)
                    query = query.limit(1)
                    result = await self.db.execute(query)
                    fuzzy_name = result.scalar_one_or_none()
                    if fuzzy_name:
                        return fuzzy_name

                    return None

                # Filter main deck to only valid cards
                filtered_main = []
                for entry in deck_data.get("main_deck", []):
                    card_name = entry.get("card_name", "")
                    valid_name = await find_valid_card(card_name)
                    if valid_name:
                        entry["card_name"] = valid_name  # Use proper name
                        filtered_main.append(entry)
                    else:
                        logger.debug(f"[AI-SERVICE] Removed invalid card: {card_name}")

                # Filter sideboard to only valid cards
                filtered_sideboard = []
                for entry in deck_data.get("sideboard", []):
                    card_name = entry.get("card_name", "")
                    valid_name = await find_valid_card(card_name)
                    if valid_name:
                        entry["card_name"] = valid_name
                        filtered_sideboard.append(entry)
                    else:
                        logger.debug(f"[AI-SERVICE] Removed invalid sideboard card: {card_name}")

                deck_data["main_deck"] = filtered_main
                deck_data["sideboard"] = filtered_sideboard

                # Filter out cards that don't match the deck's colors
                deck_data = await self._filter_by_color(deck_data, colors)

                # Fix mana base - cEDH needs special handling for fetches/duals
                if format == "cedh":
                    # Store target_lands and commander in deck_data for use in _fix_deck_counts
                    deck_data["target_lands"] = target_lands
                    deck_data["commander"] = commander_name
                    await self._ensure_cedh_mana_base(deck_data, colors)
                else:
                    target_lands = int(archetype_template.get('avg_lands', 24)) if archetype_template else 24
                    await self._fix_land_count(deck_data, target_lands, colors)

                # Fix card counts to match format requirements
                deck_data = await self._fix_deck_counts(deck_data, colors, available_cards, format=format)

                # Validate total counts
                main_count = sum(e.get("quantity", 0) for e in deck_data.get("main_deck", []))
                side_count = sum(e.get("quantity", 0) for e in deck_data.get("sideboard", []))

                logger.debug(f"[AI-SERVICE] Generated deck after fixing: {main_count} main, {side_count} sideboard")

                # Enrich deck entries with card type information for frontend categorization
                deck_data = await self._enrich_deck_with_card_data(deck_data)

                return deck_data

        except Exception as e:
            logger.error(f"AI deck generation error: {e}", exc_info=True)

        return await self._generate_fallback_deck(archetype, colors, strategy, format=format)

    async def generate_sideboard_matrix(
        self,
        deck_data: Dict[str, Any],
        archetype: str,
        meta_archetypes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate a complete sideboard guide matrix for all meta matchups.
        """
        if not settings.ANTHROPIC_API_KEY:
            return {"matchups": [], "general_sideboard_notes": "AI service not configured"}

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            main_deck = deck_data.get("main_deck", [])
            sideboard = deck_data.get("sideboard", [])

            main_cards = [f"{e.get('quantity', 1)}x {e.get('card_name', '')}" for e in main_deck]
            sideboard_cards = [f"{e.get('quantity', 1)}x {e.get('card_name', '')}" for e in sideboard]

            # Format meta matchups with key cards
            matchup_list = "\n".join([
                f"- {m.get('archetype', 'Unknown')}: {m.get('meta_percentage', 0):.1f}% of meta. Key cards: {', '.join(m.get('key_cards', [])[:5]) or 'unknown'}"
                for m in meta_archetypes[:8]  # Top 8 matchups
            ])

            system_prompt = f"""You are creating a comprehensive sideboard guide for a Magic: The Gathering deck.

DECK: {deck_data.get('name', 'Unknown')} ({archetype})

MAIN DECK ({sum(e.get('quantity', 0) for e in main_deck)} cards):
{chr(10).join(main_cards)}

SIDEBOARD ({sum(e.get('quantity', 0) for e in sideboard)} cards):
{chr(10).join(sideboard_cards)}

META MATCHUPS TO ADDRESS:
{matchup_list}

For EACH matchup, provide:
1. What cards to bring IN from sideboard (with quantities and reasoning)
2. What cards to take OUT from main deck (with quantities and reasoning)
3. Brief strategy notes for the matchup
4. Key cards to find/keep in opening hand
5. Opponent's key cards to play around

IMPORTANT: Cards in must equal cards out for each matchup!

Return JSON:
{{
    "matchups": [
        {{
            "matchup": "Archetype Name",
            "matchup_description": "Brief description of the matchup",
            "cards_in": [
                {{"card_name": "Card", "quantity": 2, "reasoning": "Why bring this in"}}
            ],
            "cards_out": [
                {{"card_name": "Card", "quantity": 2, "reasoning": "Why take this out"}}
            ],
            "strategy_notes": "How to approach this matchup post-board",
            "key_cards_to_find": ["Card1", "Card2"],
            "cards_to_play_around": ["Opponent's Card1", "Opponent's Card2"]
        }}
    ],
    "general_sideboard_notes": "General advice about sideboarding with this deck"
}}"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Generate sideboard plans for all {len(meta_archetypes[:8])} matchups."}],
            )

            if not response.content:
                return {"matchups": [], "general_sideboard_notes": "Failed to generate"}

            content = response.content[0].text

            if "{" in content:
                json_start = content.index("{")
                json_end = content.rindex("}") + 1
                result = json.loads(content[json_start:json_end])
                logger.info(f"Generated sideboard matrix with {len(result.get('matchups', []))} matchups")
                return result

        except Exception as e:
            logger.error(f"Failed to generate sideboard matrix: {e}")

        return {"matchups": [], "general_sideboard_notes": "Error generating sideboard guide"}

    async def generate_card_explanations(
        self,
        deck_data: Dict[str, Any],
        archetype: str,
        strategy: str,
    ) -> Dict[str, str]:
        """
        Generate context-aware explanations for each card in the deck.
        Explains WHY each card is in THIS specific deck.
        """
        if not settings.ANTHROPIC_API_KEY:
            return {}

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            # Collect unique card names from main deck and sideboard
            main_deck = deck_data.get("main_deck", [])
            sideboard = deck_data.get("sideboard", [])

            main_cards = [f"{e.get('quantity', 1)}x {e.get('card_name', '')}" for e in main_deck]
            sideboard_cards = [f"{e.get('quantity', 1)}x {e.get('card_name', '')}" for e in sideboard]

            all_card_names = list(set(
                [e.get("card_name", "") for e in main_deck] +
                [e.get("card_name", "") for e in sideboard]
            ))

            if not all_card_names:
                return {}

            system_prompt = f"""You are explaining card choices in a Magic: The Gathering deck.

DECK: {deck_data.get('name', 'Unknown')}
ARCHETYPE: {archetype}
STRATEGY: {strategy}

MAIN DECK:
{chr(10).join(main_cards)}

SIDEBOARD:
{chr(10).join(sideboard_cards)}

For each card, explain in 1-2 sentences:
1. Its specific role in THIS deck (not generic card description)
2. What matchups it's important against
3. When you might sideboard it out (for main deck cards) or in (for sideboard cards)

Return JSON mapping card names to explanations:
{{
    "Card Name": "This is your primary removal spell, crucial against creature-based decks. Side out against control where creatures are sparse.",
    ...
}}

Be specific to this deck's strategy. Don't just describe what the card does - explain why it's HERE."""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Generate explanations for all {len(all_card_names)} cards."}],
            )

            if not response.content:
                return {}

            content = response.content[0].text

            # Extract JSON
            if "{" in content:
                json_start = content.index("{")
                json_end = content.rindex("}") + 1
                explanations = json.loads(content[json_start:json_end])
                logger.info(f"Generated explanations for {len(explanations)} cards")
                return explanations

        except Exception as e:
            logger.error(f"Failed to generate card explanations: {e}")

        return {}

    async def iterate_deck(
        self,
        current_deck: Dict[str, Any],
        modification_request: str,
    ) -> Dict[str, Any]:
        """
        Suggest modifications to an existing deck.
        """
        if not settings.ANTHROPIC_API_KEY:
            return {"changes": [], "summary": "AI service not configured"}

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            # Get available cards
            available_cards = await self.card_service.search(standard_only=True, limit=100)
            available_names = [c.name for c in available_cards]

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=f"""You are a Magic: The Gathering deck modification assistant.

Current deck:
Main deck: {json.dumps(current_deck.get('main_deck', []))}
Sideboard: {json.dumps(current_deck.get('sideboard', []))}

Available cards to add (only use these): {', '.join(available_names[:100])}

Return modifications as JSON:
{{
    "changes": [
        {{"action": "remove|add", "card_name": "Name", "quantity": N, "target": "main_deck|sideboard", "reasoning": "Why"}}
    ],
    "summary": "Brief summary of changes"
}}""",
                messages=[{"role": "user", "content": modification_request}],
            )

            content = response.content[0].text
            if "{" in content:
                json_start = content.index("{")
                json_end = content.rindex("}") + 1
                return json.loads(content[json_start:json_end])

        except Exception as e:
            logger.error(f"AI iteration error: {e}")

        return {"changes": [], "summary": "Unable to process modification"}

    async def _get_cards_by_role(
        self,
        colors: List[str],
        role_distribution: Dict[str, float],
        archetype_category: str = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get cards grouped by their functional roles, filtered by color.
        Returns dict mapping role -> list of cards that fulfill that role.
        Cards must be colorless OR have colors that are a subset of deck colors.

        PRIORITIZES cards that appear in SIMILAR tournament decklists
        (matching by color and archetype category).
        """
        from sqlalchemy import text

        # Build color filter - cards must be colorless or ONLY have colors in the deck
        color_list = [c.upper() for c in colors]
        color_placeholders = ", ".join([f"'{c}'" for c in color_list])

        # Build archetype filter for tournament decks
        # Match decks that have similar colors OR similar archetype keywords
        color_keywords = []
        for c in color_list:
            color_map = {"R": "red", "W": "white", "U": "blue", "B": "black", "G": "green"}
            if c in color_map:
                color_keywords.append(color_map[c])

        # Also match mono-X patterns
        if len(color_list) == 1:
            color_keywords.append(f"mono")

        # Archetype category keywords for filtering similar decks
        archetype_keywords = {
            "aggro": ["aggro", "rdw", "red deck wins", "burn", "sligh", "weenie"],
            "control": ["control"],
            "midrange": ["midrange", "ramp", "value"],
            "combo": ["combo", "storm", "reanimator"],
        }
        arch_filters = archetype_keywords.get(archetype_category, [])

        # Build the archetype/color LIKE conditions
        archetype_conditions = []
        for kw in color_keywords + arch_filters:
            archetype_conditions.append(f"LOWER(d.archetype) LIKE '%{kw}%'")

        archetype_filter = " OR ".join(archetype_conditions) if archetype_conditions else "TRUE"

        role_cards = {}

        for role in role_distribution.keys():
            # Skip land roles - handled separately
            if role.startswith("land_"):
                continue

            # Query cards that have this role, match color requirements,
            # and cross-reference with SIMILAR tournament decklists for priority
            # Handle DFCs by matching on the front face name (before " // ")
            query = text(f"""
                WITH similar_deck_cards AS (
                    -- Get cards from tournament decks with similar colors/archetype
                    SELECT
                        card_entry->>'card_name' as card_name,
                        COUNT(DISTINCT d.id) as tournament_count
                    FROM decklists d,
                         jsonb_array_elements(d.main_deck) as card_entry
                    WHERE {archetype_filter}
                    GROUP BY card_entry->>'card_name'
                ),
                role_cards_deduped AS (
                    -- Get unique cards with this role (dedupe by name)
                    SELECT DISTINCT ON (c.name)
                        c.name,
                        -- Extract front face name for DFCs (before " // ")
                        SPLIT_PART(c.name, ' // ', 1) as front_face_name,
                        c.mana_cost,
                        c.type_line,
                        cr.efficiency,
                        c.cmc,
                        c.colors
                    FROM cards c
                    JOIN card_roles cr ON c.id = cr.card_id
                    WHERE cr.role = :role
                      AND c.is_standard_legal = true
                    ORDER BY c.name, cr.efficiency DESC NULLS LAST
                )
                SELECT
                    rc.name,
                    rc.mana_cost,
                    rc.type_line,
                    rc.efficiency,
                    rc.cmc,
                    COALESCE(sdc.tournament_count, 0) as tournament_count
                FROM role_cards_deduped rc
                -- Match on front face name OR full name (handles DFCs)
                LEFT JOIN similar_deck_cards sdc ON
                    LOWER(sdc.card_name) = LOWER(rc.name) OR
                    LOWER(sdc.card_name) = LOWER(rc.front_face_name)
                WHERE rc.colors = ARRAY[]::varchar[]
                   OR rc.colors <@ ARRAY[{color_placeholders}]::varchar[]
                ORDER BY tournament_count DESC NULLS LAST, rc.efficiency DESC NULLS LAST
                LIMIT 50
            """)

            result = await self.db.execute(query, {"role": role})
            cards = result.all()

            if cards:
                # Sort by: tournament appearances (desc), then efficiency (desc), then name
                sorted_cards = sorted(
                    cards,
                    key=lambda x: (-(x.tournament_count or 0), -(x.efficiency or 0), x.name)
                )
                role_cards[role] = [
                    {
                        "name": c.name,
                        "mana_cost": c.mana_cost,
                        "type_line": c.type_line,
                        "efficiency": c.efficiency,
                        "cmc": float(c.cmc) if c.cmc else 0,
                        "tournament_count": c.tournament_count,
                    }
                    for c in sorted_cards[:20]  # Top 20 per role
                ]

        return role_cards

    async def _get_available_cards(self, colors: List[str], format: str = "standard") -> List[Dict[str, Any]]:
        """Get available cards for the specified colors and format."""
        all_cards = []

        # Determine if we should use format-based legality or standard_only
        # For cEDH/Commander, use format-based legality to get eternal cards
        use_format_legality = format in ["cedh", "commander", "legacy", "modern", "historic"]

        logger.info(f"Getting available cards for colors={colors}, format={format}, use_format_legality={use_format_legality}")

        # Get cards for each color separately (not requiring all colors)
        # Use high limits to ensure we get enough unique cards after deduplication
        for color in colors:
            if use_format_legality:
                color_cards = await self.card_service.search(
                    colors=[color],
                    standard_only=False,
                    format=format,
                    limit=500,
                )
            else:
                color_cards = await self.card_service.search(
                    colors=[color],
                    standard_only=True,
                    limit=500,
                )
            all_cards.extend(color_cards)
            logger.info(f"Found {len(color_cards)} cards for color {color}")

        # Get lands separately - search for land type
        if use_format_legality:
            lands = await self.card_service.search(
                card_type="land",
                standard_only=False,
                format=format,
                limit=500,
            )
        else:
            lands = await self.card_service.search(
                card_type="land",
                standard_only=True,
                limit=500,
            )
        all_cards.extend(lands)
        logger.info(f"Found {len(lands)} lands")

        # Get colorless artifacts (includes Sol Ring, Mana Crypt for eternal formats)
        if use_format_legality:
            colorless_artifacts = await self.card_service.search(
                card_type="artifact",
                standard_only=False,
                format=format,
                limit=500,
            )
        else:
            colorless_artifacts = await self.card_service.search(
                card_type="artifact",
                standard_only=True,
                limit=500,
            )
        all_cards.extend(colorless_artifacts)
        logger.info(f"Found {len(colorless_artifacts)} artifacts")

        # Get colorless non-artifact cards (like colorless planeswalkers)
        # Search for cards with no colors (empty array)
        from sqlalchemy import select, func
        from app.models.card import Card
        from app.services.card_service import get_format_legality_condition

        base_conditions = [
            Card.colors == [],  # Empty colors array
            ~Card.type_line.ilike("%land%"),  # Exclude lands
            ~Card.type_line.ilike("%artifact%"),  # Exclude artifacts (already fetched)
        ]

        if use_format_legality:
            colorless_query = select(Card).where(
                get_format_legality_condition(format),
                *base_conditions
            ).order_by(Card.name).limit(5000)
        else:
            colorless_query = select(Card).where(
                Card.is_standard_legal == True,
                *base_conditions
            ).order_by(Card.name).limit(5000)

        result = await self.db.execute(colorless_query)
        colorless_raw = result.scalars().all()

        # Deduplicate colorless cards
        colorless_seen = set()
        colorless_others = []
        for c in colorless_raw:
            if c.name not in colorless_seen:
                colorless_seen.add(c.name)
                colorless_others.append(c)
                if len(colorless_others) >= 200:
                    break

        all_cards.extend(colorless_others)
        logger.info(f"Found {len(colorless_others)} colorless non-artifact cards")
        if colorless_others:
            logger.info(f"Sample colorless: {[c.name for c in colorless_others[:5]]}")

        # Add basic lands explicitly (they don't have color identity)
        basic_land_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
        for color in colors:
            basic_name = basic_land_map.get(color.upper())
            if basic_name:
                basic = await self.card_service.get_by_name(basic_name, standard_only=False)
                if basic:
                    all_cards.append(basic)

        # Deduplicate by name
        seen_names = set()
        unique_cards = []
        for c in all_cards:
            if c.name not in seen_names:
                seen_names.add(c.name)
                unique_cards.append(c)

        return [
            {
                "name": c.name,
                "mana_cost": c.mana_cost,
                "type_line": c.type_line,
                "oracle_text": c.oracle_text,
                "cmc": float(c.cmc) if c.cmc else 0,
            }
            for c in unique_cards
        ]

    async def _fix_deck_counts(
        self,
        deck_data: Dict[str, Any],
        colors: List[str],
        available_cards: List[Dict[str, Any]],
        format: str = "standard",
    ) -> Dict[str, Any]:
        """Fix deck to match format requirements (60 main + 15 side for standard, 99 main + 0 side for cEDH)."""
        from app.models.card import Card
        from app.services.deck_validator import FORMAT_RULES
        from sqlalchemy import func

        rules = FORMAT_RULES.get(format, FORMAT_RULES["standard"])
        target_main = rules["main_deck_size"]
        target_side = rules["sideboard_size"]
        max_copies = rules["max_copies"]

        main_deck = deck_data.get("main_deck", [])
        sideboard = deck_data.get("sideboard", [])

        # For cEDH, clear the sideboard since Commander doesn't use one
        if format == "cedh":
            sideboard = []
            deck_data["sideboard"] = []

        # For cEDH, remove the commander from the main deck if present
        # The commander is separate and not counted in the 99 cards
        if format == "cedh":
            commander_name = deck_data.get("commander")
            if commander_name:
                original_count = len(main_deck)
                main_deck = [
                    entry for entry in main_deck
                    if entry.get("card_name", "").lower() != commander_name.lower()
                ]
                if len(main_deck) < original_count:
                    logger.debug(f"[AI-SERVICE] Removed commander '{commander_name}' from main deck (it's separate)")
                deck_data["main_deck"] = main_deck

        # For cEDH, check for anti-synergies with expensive commanders
        # Commanders costing 4+ mana need artifact mana - don't include artifact hate
        if format == "cedh":
            commander_name = deck_data.get("commander")
            if commander_name:
                # Look up commander's mana value (limit 1 since multiple printings exist)
                commander_query = select(Card.cmc).where(
                    func.lower(Card.name) == commander_name.lower()
                ).limit(1)
                result = await self.db.execute(commander_query)
                commander_cmc = result.scalar_one_or_none()

                if commander_cmc and commander_cmc >= 4:
                    # Remove artifact hate cards that would prevent casting the commander
                    artifact_hate_cards = {
                        "collector ouphe", "null rod", "stony silence",
                        "karn, the great creator", "kataki, war's wage"
                    }
                    original_count = len(main_deck)
                    main_deck = [
                        entry for entry in main_deck
                        if entry.get("card_name", "").lower() not in artifact_hate_cards
                    ]
                    removed_count = original_count - len(main_deck)
                    if removed_count > 0:
                        logger.debug(f"[AI-SERVICE] Removed {removed_count} artifact hate cards (anti-synergy with {commander_cmc}-mana commander)")
                    deck_data["main_deck"] = main_deck

        # Enforce singleton for cEDH
        if max_copies == 1:
            seen = set()
            deduped = []
            for entry in main_deck:
                card_name = entry.get("card_name", "")
                if card_name in BASIC_LANDS:
                    deduped.append(entry)
                elif card_name not in seen:
                    seen.add(card_name)
                    entry["quantity"] = 1
                    deduped.append(entry)
                else:
                    logger.debug(f"[AI-SERVICE] Removed duplicate in singleton deck: {card_name}")
            main_deck = deduped
            deck_data["main_deck"] = main_deck

        # Get basic land for the colors
        basic_land_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
        primary_basic = basic_land_map.get(colors[0].upper() if colors else "R", "Mountain")

        # First, ensure we have enough lands
        archetype = deck_data.get("archetype", "").lower()
        if format == "cedh":
            # Use target_lands from deck_data if available (set during generation from tournament data)
            # Otherwise use a reasonable minimum that doesn't over-constrain
            min_lands = deck_data.get("target_lands", 28)
            # Ensure minimum sanity check - cEDH rarely goes below 27
            min_lands = max(min_lands - 2, 27)  # Allow some variance below target
        else:
            min_lands = 20 if archetype in ["aggro", "burn", "red deck wins"] else 22

        # Count current lands by querying card types
        all_card_names = [entry.get("card_name", "") for entry in main_deck]
        land_query = select(Card.name).where(
            func.lower(Card.name).in_([n.lower() for n in all_card_names]),
            Card.type_line.ilike("%land%")
        )
        land_result = await self.db.execute(land_query)
        land_names = {row[0].lower() for row in land_result.all()}

        current_land_count = 0
        land_entries = []
        non_land_entries = []
        for entry in main_deck:
            card_name = entry.get("card_name", "")
            qty = entry.get("quantity", 0)
            if card_name.lower() in land_names:
                current_land_count += qty
                land_entries.append(entry)
            else:
                non_land_entries.append(entry)

        logger.debug(f"[AI-SERVICE] Current land count: {current_land_count}, minimum: {min_lands}")

        # If we don't have enough lands, add more basics and trim spells
        if current_land_count < min_lands:
            lands_to_add = min_lands - current_land_count
            logger.debug(f"[AI-SERVICE] Need to add {lands_to_add} more lands")

            # Find or create basic land entry
            existing_basic = None
            for entry in land_entries:
                if entry.get("card_name") == primary_basic:
                    existing_basic = entry
                    break

            if existing_basic:
                existing_basic["quantity"] += lands_to_add
            else:
                land_entries.append({"card_name": primary_basic, "quantity": lands_to_add})

            # Trim non-land cards to make room (remove from cards with highest quantities first)
            non_land_entries.sort(key=lambda x: x.get("quantity", 0), reverse=True)
            remaining_to_trim = lands_to_add
            while remaining_to_trim > 0 and non_land_entries:
                # Find a card we can trim (prefer trimming 4-ofs to 3-ofs, etc.)
                trimmed = False
                for entry in non_land_entries:
                    if entry.get("quantity", 0) > 1:
                        trim_amount = min(entry["quantity"] - 1, remaining_to_trim)
                        entry["quantity"] -= trim_amount
                        remaining_to_trim -= trim_amount
                        logger.debug(f"[AI-SERVICE] Trimmed {trim_amount}x {entry.get('card_name')} to make room for lands")
                        trimmed = True
                        if remaining_to_trim <= 0:
                            break
                if not trimmed:
                    # Remove entire cards if we can't trim quantities
                    if non_land_entries:
                        removed = non_land_entries.pop()
                        remaining_to_trim -= removed.get("quantity", 0)
                        logger.debug(f"[AI-SERVICE] Removed {removed.get('card_name')} to make room for lands")

            # Rebuild main_deck with lands first, then non-lands
            main_deck = land_entries + [e for e in non_land_entries if e.get("quantity", 0) > 0]
            deck_data["main_deck"] = main_deck

        main_count = sum(e.get("quantity", 0) for e in main_deck)
        side_count = sum(e.get("quantity", 0) for e in sideboard)

        # Fix main deck to target size
        if main_count < target_main:
            deficit = target_main - main_count
            logger.debug(f"[AI-SERVICE] Deck has {main_count} cards, need {deficit} more (target: {target_main})")

            # First, try to add more non-land cards
            existing_cards = {entry.get("card_name", "").lower() for entry in main_deck}

            # For cEDH, add cards from tournament metagame decklists
            if format == "cedh":
                from app.models.meta import Decklist, Event
                from app.models.card import Card
                from app.services.card_service import get_format_legality_condition
                from sqlalchemy import func
                from collections import Counter

                # Normalize deck colors for comparison
                deck_colors = {c.upper() for c in colors} if colors else set()
                logger.debug(f"[AI-SERVICE] Deck colors for filler filtering: {deck_colors}")

                # Query cards from tournament decklists in the metagame
                # Get recent cEDH decklists ordered by date and extract card frequencies
                decklist_query = (
                    select(Decklist.main_deck)
                    .join(Event, Decklist.event_id == Event.id)
                    .where(Event.format == "cedh")
                    .order_by(Event.date.desc())
                    .limit(200)  # Sample recent decklists
                )
                decklist_result = await self.db.execute(decklist_query)
                decklists = decklist_result.scalars().all()

                # Count card frequencies across all decklists
                card_frequency = Counter()
                for decklist_cards in decklists:
                    if decklist_cards:
                        for entry in decklist_cards:
                            card_name = entry.get("card_name", "")
                            if card_name and card_name.lower() not in existing_cards:
                                card_frequency[card_name] += 1

                # Sort by frequency (most played cards first)
                metagame_cards = [card for card, _ in card_frequency.most_common()]
                logger.debug(f"[AI-SERVICE] Found {len(metagame_cards)} unique cards from {len(decklists)} metagame decklists")

                # Add cards from metagame (with color identity check and legality validation)
                cards_added = 0
                for card_name in metagame_cards:
                    if deficit <= 0:
                        break

                    # Query card with color identity check
                    query = select(Card.name, Card.color_identity, Card.type_line).where(
                        func.lower(Card.name) == card_name.lower(),
                        get_format_legality_condition(format)
                    ).limit(1)
                    result = await self.db.execute(query)
                    row = result.first()

                    if not row:
                        continue

                    valid_name, card_color_identity, type_line = row

                    # Skip lands - we handle those separately
                    if type_line and "land" in type_line.lower().split(" — ")[0].lower():
                        continue

                    # Check color identity - card must be playable in the deck's colors
                    if card_color_identity:
                        card_colors = {c.upper() for c in card_color_identity}
                        if not card_colors.issubset(deck_colors):
                            continue

                    if valid_name.lower() not in existing_cards:
                        main_deck.append({"card_name": valid_name, "quantity": 1})
                        existing_cards.add(valid_name.lower())
                        deficit -= 1
                        cards_added += 1
                        logger.debug(f"[AI-SERVICE] Added metagame card: {valid_name} (played in {card_frequency[card_name]} decks)")

                logger.debug(f"[AI-SERVICE] Added {cards_added} cards from metagame decklists")

                # If still short and no metagame data, fall back to known cEDH staples
                if deficit > 0:
                    from app.services.cedh_knowledge import CEDH_STAPLES

                    logger.debug(f"[AI-SERVICE] Still need {deficit} cards, checking cEDH staples...")

                    # Build prioritized list of staples to add
                    staples_to_add = []
                    categories = ["free_counterspells", "tutors", "fast_mana", "efficient_counterspells",
                                  "card_advantage", "interaction", "mana_dorks", "hatebears",
                                  "utility_creatures", "combo_creatures"]

                    for category in categories:
                        if category in CEDH_STAPLES:
                            for card_name in CEDH_STAPLES[category].get("cards", []):
                                if card_name.lower() not in existing_cards:
                                    staples_to_add.append(card_name)

                    for card_name in staples_to_add:
                        if deficit <= 0:
                            break

                        # Query card with color identity check
                        query = select(Card.name, Card.color_identity).where(
                            func.lower(Card.name) == card_name.lower(),
                            get_format_legality_condition(format)
                        ).limit(1)
                        result = await self.db.execute(query)
                        row = result.first()

                        if not row:
                            continue

                        valid_name, card_color_identity = row

                        # Check color identity
                        if card_color_identity:
                            card_colors = {c.upper() for c in card_color_identity}
                            if not card_colors.issubset(deck_colors):
                                continue

                        if valid_name.lower() not in existing_cards:
                            main_deck.append({"card_name": valid_name, "quantity": 1})
                            existing_cards.add(valid_name.lower())
                            deficit -= 1
                            logger.debug(f"[AI-SERVICE] Added cEDH staple: {valid_name}")

            # If still short, add from available cards (non-cEDH formats only)
            # For cEDH, we skip this and add lands instead - the available_cards pool can have garbage
            if deficit > 0 and format != "cedh":
                available_nonlands = [
                    c for c in available_cards
                    if c.get("name", "").lower() not in existing_cards
                    and not (c.get("type_line") or "").lower().startswith("land")
                    and "land" not in (c.get("type_line") or "").lower().split(" — ")[0].lower()
                ]

                for card in available_nonlands:
                    if deficit <= 0:
                        break
                    qty_to_add = min(max_copies, deficit)
                    main_deck.append({"card_name": card["name"], "quantity": qty_to_add})
                    deficit -= qty_to_add
                    logger.debug(f"[AI-SERVICE] Added {qty_to_add}x {card['name']} to fill deck")

            # If still short, add basic lands as last resort
            if deficit > 0:
                existing_basic = None
                for entry in main_deck:
                    if entry.get("card_name") == primary_basic:
                        existing_basic = entry
                        break
                if existing_basic:
                    existing_basic["quantity"] += deficit
                else:
                    main_deck.append({"card_name": primary_basic, "quantity": deficit})
                logger.debug(f"[AI-SERVICE] Added {deficit} {primary_basic} as filler (last resort)")

        elif main_count > target_main:
            # Remove cards from the end (typically less important)
            excess = main_count - target_main
            while excess > 0 and main_deck:
                last_entry = main_deck[-1]
                qty = last_entry.get("quantity", 1)
                if qty <= excess:
                    excess -= qty
                    main_deck.pop()
                    logger.debug(f"[AI-SERVICE] Removed {qty}x {last_entry.get('card_name')} to reduce main deck")
                else:
                    last_entry["quantity"] -= excess
                    logger.debug(f"[AI-SERVICE] Reduced {last_entry.get('card_name')} by {excess} to reach {target_main}")
                    excess = 0

        # Fix sideboard to target size
        if target_side > 0:
            if side_count < target_side:
                # Add basic lands or duplicate existing sideboard cards
                deficit = target_side - side_count
                if sideboard:
                    # Try to add more copies of existing sideboard cards
                    for entry in sideboard:
                        current_qty = entry.get("quantity", 0)
                        can_add = max_copies - current_qty
                        if can_add > 0:
                            add = min(can_add, deficit)
                            entry["quantity"] = current_qty + add
                            deficit -= add
                            if deficit <= 0:
                                break
                if deficit > 0:
                    # Add basic land as filler
                    sideboard.append({"card_name": primary_basic, "quantity": deficit})
                logger.debug(f"[AI-SERVICE] Fixed sideboard to {target_side} cards")

            elif side_count > target_side:
                # Remove cards from sideboard
                excess = side_count - target_side
                while excess > 0 and sideboard:
                    last_entry = sideboard[-1]
                    qty = last_entry.get("quantity", 1)
                    if qty <= excess:
                        excess -= qty
                        sideboard.pop()
                    else:
                        last_entry["quantity"] -= excess
                        excess = 0
                logger.debug(f"[AI-SERVICE] Trimmed sideboard to {target_side} cards")

        deck_data["main_deck"] = main_deck
        deck_data["sideboard"] = sideboard
        return deck_data

