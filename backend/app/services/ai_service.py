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

logger = logging.getLogger(__name__)


# Maximum number of full decklist examples to include in prompt
MAX_DECKLIST_EXAMPLES = 3

# Cache for tournament cards (shared across requests)
_tournament_cards_cache: Dict[str, Any] = {
    "data": None,
    "timestamp": 0,
    "ttl": 300,  # 5 minutes
}


class AIService:
    """
    Service for AI-powered deck building using Claude API.
    Implements the constrained card selection system to prevent hallucination.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.card_service = CardService(db)

    async def parse_deck_request(self, prompt: str) -> Dict[str, Any]:
        """
        Parse a natural language deck request to extract:
        - Archetype (aggro, control, midrange, combo)
        - Colors
        - Strategy focus
        - Specific card requests

        Uses Haiku for fast parsing (~0.5s vs 2-3s with Sonnet).
        """
        if not settings.ANTHROPIC_API_KEY:
            return await self._fallback_parse(prompt)

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            response = client.messages.create(
                model="claude-3-5-haiku-20241022",  # Fast model for parsing
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

        return await self._fallback_parse(prompt)

    async def _fallback_parse(self, prompt: str) -> Dict[str, Any]:
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
        archetype = "midrange"  # default
        if any(word in prompt_lower for word in ["aggro", "aggressive", "fast", "burn", "rush"]):
            archetype = "aggro"
        elif any(word in prompt_lower for word in ["control", "counter", "remove", "wrath"]):
            archetype = "control"
        elif any(word in prompt_lower for word in ["combo", "infinite", "win condition"]):
            archetype = "combo"
        elif any(word in prompt_lower for word in ["tempo", "disruptive"]):
            archetype = "tempo"

        # Try to extract specific card names from the prompt
        specific_cards = await self._extract_card_names_from_prompt(prompt)

        return {
            "archetype": archetype,
            "colors": colors or ["R"],  # Default to red if no colors specified
            "strategy": prompt,
            "specific_cards": specific_cards,
            "budget": "competitive",
            "focus": "speed" if archetype == "aggro" else "value",
        }

    async def _extract_card_names_from_prompt(self, prompt: str) -> List[str]:
        """Extract card names mentioned in the prompt by checking against database."""
        from app.models.card import Card
        from sqlalchemy import func

        specific_cards = []
        prompt_lower = prompt.lower()

        # Split prompt into potential card name candidates
        # Look for capitalized words that might be card names
        words = prompt.split()
        potential_names = []

        # Try to find multi-word card names (e.g., "Lightning Bolt", "Tezzeret, Cruel Captain")
        for i in range(len(words)):
            # Single word
            if words[i][0:1].isupper() or words[i].lower() in ["tezzeret", "jace", "liliana", "chandra", "nissa", "garruk", "ajani", "nicol", "bolas"]:
                potential_names.append(words[i].strip(",.!?"))
            # Two words
            if i < len(words) - 1:
                two_word = f"{words[i]} {words[i+1]}".strip(",.!?")
                potential_names.append(two_word)
            # Three words
            if i < len(words) - 2:
                three_word = f"{words[i]} {words[i+1]} {words[i+2]}".strip(",.!?")
                potential_names.append(three_word)

        # Check each potential name against the database
        for name in potential_names:
            if len(name) < 3:  # Skip very short strings
                continue
            query = select(Card.name).where(
                func.lower(Card.name).like(f"%{name.lower()}%"),
                Card.is_standard_legal == True
            ).limit(1)
            result = await self.db.execute(query)
            card = result.scalar_one_or_none()
            if card and card not in specific_cards:
                specific_cards.append(card)
                logger.info(f"Extracted card name from prompt: {card}")

        return specific_cards

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
            template_decks = []
            template_cards = []
            detected_themes = []
            print(f"[AI-SERVICE] Specific cards from parse: {specific_cards}")
            if specific_cards:
                template_decks = await self._find_decks_with_cards(specific_cards)
                print(f"[AI-SERVICE] Found {len(template_decks)} tournament decks with specific cards")
                if template_decks:
                    template_colors = self._extract_colors_from_decks(template_decks)
                    if template_colors:
                        print(f"[AI-SERVICE] Overriding colors from {colors} to {template_colors} based on tournament decks")
                        colors = template_colors
                    template_cards = self._extract_cards_from_decks(template_decks)
                else:
                    detected_themes = await self._detect_card_themes(specific_cards)
                    print(f"[AI-SERVICE] Detected themes: {detected_themes}")
                    if detected_themes:
                        template_cards = await self._get_tournament_synergy_cards(detected_themes)
                        print(f"[AI-SERVICE] Found {len(template_cards)} tournament-played synergy cards")

            # Phase 1b: If no specific cards, try to find archetype decklists based on strategy
            # This allows strategy keywords like "graveyard" to find "Reanimator" decks and use their colors
            if not template_decks and strategy:
                strategy_decks = await self._get_archetype_decklists(archetype, [], strategy, limit=5)
                if strategy_decks:
                    print(f"[AI-SERVICE] Found {len(strategy_decks)} tournament decks matching strategy '{strategy}'")
                    strategy_colors = self._extract_colors_from_decks(strategy_decks)
                    if strategy_colors:
                        print(f"[AI-SERVICE] Overriding colors from {colors} to {strategy_colors} based on strategy-matched decks")
                        colors = strategy_colors
                    template_decks = strategy_decks
                    template_cards = self._extract_cards_from_decks(strategy_decks)

            # Phase 2: Run independent queries in parallel
            parallel_tasks = {
                'available_cards': self._get_available_cards(colors),
                'sideboard_patterns': self._get_sideboard_patterns(archetype, colors),
                'composition': self._get_deck_composition_from_meta(archetype, colors),
                'mana_base_data': self._get_mana_base_from_meta(archetype, colors),
                'semantic_cards': self._semantic_card_search(strategy, colors),
                'tournament_cards': self._get_all_tournament_cards(),
            }

            # Add conditional parallel tasks
            if specific_cards:
                parallel_tasks['cooccurrence_cards'] = self._get_cooccurrence_cards(specific_cards, colors)
            if not template_cards:
                parallel_tasks['meta_cards'] = self._get_meta_cards(archetype, strategy)
            if not template_decks and archetype:
                parallel_tasks['example_decklists'] = self._get_archetype_decklists(archetype, colors, strategy)

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

            print(f"[AI-SERVICE] Parallel queries complete: {len(available_cards)} available, {len(tournament_cards)} tournament cards")

            # Add template cards to available cards if needed
            if template_cards:
                template_card_names = {c["name"] for c in template_cards}
                existing_names = {c["name"] for c in available_cards}
                missing_names = template_card_names - existing_names
                if missing_names:
                    from app.models.card import Card
                    from sqlalchemy import func
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

            # Format meta cards section
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
Do NOT use cards like Solemn Simulacrum, Prophetic Prism, or other "classic" artifacts that don't see current Standard tournament play.
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

            # Format sideboard patterns
            sideboard_guide_text = ""
            if sideboard_patterns.get("sideboard_staples"):
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
            print(f"[AI-SERVICE] {len(on_color_tournament)} tournament cards match colors {colors}")

            # Build format-specific rules
            if format == "cedh":
                target_lands = archetype_template.get('avg_lands', 30) if archetype_template else 30
                format_header = f"""You are Spellbook, an expert Magic: The Gathering deck builder for cEDH (competitive Commander).

YOU MUST BUILD A COMPLETE 99-CARD SINGLETON DECK (no sideboard).
This is a Commander deck - every card except basic lands must be a 1-of.

Colors: {', '.join(colors)}"""

                format_rules = f"""DECK BUILDING RULES (cEDH / Commander):
1. Main deck = EXACTLY 99 cards (the commander is separate)
2. NO sideboard - Commander format does not use sideboards
3. SINGLETON: Maximum 1 copy of each card (except basic lands)
4. *** LANDS: approximately {int(target_lands)} LANDS ***
5. ONLY use cards legal in Commander format
6. Every card's color identity must fit within colors {colors}
7. For {len(colors)}-color decks: {"Use mostly basic lands." if len(colors) == 1 else "Use appropriate dual lands, fetch lands, and color fixing."}
8. Include staple cEDH cards: fast mana (Sol Ring, Mana Vault, Chrome Mox, Mox Diamond, Lotus Petal, etc.), efficient interaction, and win conditions
9. EVERY card must have a clear purpose"""

                format_json = """Return JSON with reasoning for key cards:
{{
    "name": "Deck Name",
    "strategy_summary": "2-3 sentence strategy description",
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

            system_prompt = f"""{format_header}
{specific_cards_text}
{decklist_examples_text}
{meta_cards_text}
{cooccurrence_text}
{semantic_text}
{composition_text}
{role_distribution_text}
{sideboard_guide_text}

TOURNAMENT-PLAYED CARDS IN YOUR COLORS (USE THESE):
{chr(10).join(f"- {name}" for name in on_color_tournament[:150])}

MANA BASE from {mana_base_data.get('sample_size', 0)} tournament decks:
{land_recommendations}

{format_rules}

USER REQUEST: {archetype} deck

{format_json}"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": strategy}],
            )

            if not response.content:
                logger.error("AI returned empty content")
                raise ValueError("Empty AI response")

            content = response.content[0].text

            # Extract JSON
            if "{" in content:
                json_start = content.index("{")
                json_end = content.rindex("}") + 1
                deck_data = json.loads(content[json_start:json_end])

                # Build set of valid card names (case-insensitive)
                valid_card_names = {c["name"].lower() for c in available_cards}
                # Also add the on-color tournament cards
                valid_card_names.update(n.lower() for n in on_color_tournament)

                # Helper to find card with fuzzy matching
                async def find_valid_card(card_name: str) -> Optional[str]:
                    # Exact match
                    if card_name.lower() in valid_card_names:
                        return card_name
                    # Partial match (e.g., "Tezzeret" matches "Tezzeret, Cruel Captain")
                    for valid_name in valid_card_names:
                        if card_name.lower() in valid_name or valid_name in card_name.lower():
                            # Get the proper name from database
                            from app.models.card import Card
                            from app.services.card_service import FORMAT_LEGALITY_MAP, get_format_legality_condition
                            from sqlalchemy import func
                            query = select(Card.name).where(
                                func.lower(Card.name).like(f"%{card_name.lower()}%"),
                            )
                            if format in FORMAT_LEGALITY_MAP:
                                query = query.where(get_format_legality_condition(format))
                            else:
                                query = query.where(Card.is_standard_legal == True)
                            query = query.limit(1)
                            result = await self.db.execute(query)
                            proper_name = result.scalar_one_or_none()
                            if proper_name:
                                return proper_name
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
                        print(f"[AI-SERVICE] Removed invalid card: {card_name}")

                # Filter sideboard to only valid cards
                filtered_sideboard = []
                for entry in deck_data.get("sideboard", []):
                    card_name = entry.get("card_name", "")
                    valid_name = await find_valid_card(card_name)
                    if valid_name:
                        entry["card_name"] = valid_name
                        filtered_sideboard.append(entry)
                    else:
                        print(f"[AI-SERVICE] Removed invalid sideboard card: {card_name}")

                deck_data["main_deck"] = filtered_main
                deck_data["sideboard"] = filtered_sideboard

                # Validate and fix land count
                target_lands = int(archetype_template.get('avg_lands', 24)) if archetype_template else 24
                await self._fix_land_count(deck_data, target_lands, colors)

                # Filter out cards that don't match the deck's colors
                deck_data = await self._filter_by_color(deck_data, colors)

                # Fix card counts to match format requirements
                deck_data = await self._fix_deck_counts(deck_data, colors, available_cards, format=format)

                # Validate total counts
                main_count = sum(e.get("quantity", 0) for e in deck_data.get("main_deck", []))
                side_count = sum(e.get("quantity", 0) for e in deck_data.get("sideboard", []))

                print(f"[AI-SERVICE] Generated deck after fixing: {main_count} main, {side_count} sideboard")

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

    async def _fix_land_count(
        self,
        deck_data: Dict[str, Any],
        target_lands: int,
        colors: List[str],
    ) -> None:
        """
        Validate and fix the land count in a generated deck.
        If the deck has too few lands, add basics. If too many, remove some.
        Modifies deck_data in place.
        """
        main_deck = deck_data.get("main_deck", [])

        # Count current lands
        land_count = 0
        land_entries = []
        nonland_entries = []

        for entry in main_deck:
            card_name = entry.get("card_name", "").lower()
            # Check if it's a land (basic or otherwise)
            is_land = any(basic in card_name for basic in ["plains", "island", "swamp", "mountain", "forest"]) or \
                      "land" in card_name.lower()

            # Also check by querying the database for type_line
            if not is_land:
                from app.models.card import Card
                from sqlalchemy import func
                query = select(Card.type_line).where(
                    func.lower(Card.name) == card_name.lower()
                ).limit(1)
                result = await self.db.execute(query)
                type_line = result.scalar_one_or_none()
                if type_line and "land" in type_line.lower():
                    is_land = True

            if is_land:
                land_count += entry.get("quantity", 1)
                land_entries.append(entry)
            else:
                nonland_entries.append(entry)

        logger.info(f"Land count validation: {land_count} lands (target: {target_lands})")

        if land_count < target_lands:
            # Need to add more lands - use basics
            lands_needed = target_lands - land_count

            # Map colors to basic lands
            basic_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
            basics_to_add = [basic_map[c] for c in colors if c in basic_map]

            if basics_to_add:
                # Distribute evenly among colors
                per_basic = lands_needed // len(basics_to_add)
                remainder = lands_needed % len(basics_to_add)

                for i, basic in enumerate(basics_to_add):
                    qty = per_basic + (1 if i < remainder else 0)
                    if qty > 0:
                        # Check if this basic already exists
                        existing = next((e for e in land_entries if e.get("card_name", "").lower() == basic.lower()), None)
                        if existing:
                            existing["quantity"] = existing.get("quantity", 0) + qty
                        else:
                            land_entries.append({"card_name": basic, "quantity": qty})

                logger.info(f"Added {lands_needed} basic lands to reach target of {target_lands}")

            # May need to remove some nonland cards to make room
            total_nonlands = sum(e.get("quantity", 1) for e in nonland_entries)
            target_nonlands = 60 - target_lands

            if total_nonlands > target_nonlands:
                # Remove lowest-priority cards (from the end of the list)
                cards_to_remove = total_nonlands - target_nonlands
                while cards_to_remove > 0 and nonland_entries:
                    last_entry = nonland_entries[-1]
                    qty = last_entry.get("quantity", 1)
                    if qty <= cards_to_remove:
                        nonland_entries.pop()
                        cards_to_remove -= qty
                        logger.info(f"Removed {qty}x {last_entry.get('card_name')} to make room for lands")
                    else:
                        last_entry["quantity"] = qty - cards_to_remove
                        logger.info(f"Reduced {last_entry.get('card_name')} by {cards_to_remove} to make room for lands")
                        cards_to_remove = 0

        elif land_count > target_lands + 2:
            # Too many lands - remove some basics
            lands_to_remove = land_count - target_lands
            for entry in land_entries:
                if lands_to_remove <= 0:
                    break
                card_name = entry.get("card_name", "").lower()
                if card_name in ["plains", "island", "swamp", "mountain", "forest"]:
                    qty = entry.get("quantity", 1)
                    remove_qty = min(qty, lands_to_remove)
                    entry["quantity"] = qty - remove_qty
                    lands_to_remove -= remove_qty

            # Remove entries with 0 quantity
            land_entries = [e for e in land_entries if e.get("quantity", 0) > 0]

        # Rebuild main deck
        deck_data["main_deck"] = nonland_entries + land_entries

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

    async def _get_available_cards(self, colors: List[str]) -> List[Dict[str, Any]]:
        """Get available cards for the specified colors."""
        all_cards = []

        # Get cards for each color separately (not requiring all colors)
        # Use high limits to ensure we get enough unique cards after deduplication
        for color in colors:
            color_cards = await self.card_service.search(
                colors=[color],
                standard_only=True,
                limit=500,  # Increased to get more cards
            )
            all_cards.extend(color_cards)
            logger.info(f"Found {len(color_cards)} cards for color {color}")

        # Get lands separately - search for land type
        # Need high limit to cover all unique Standard lands
        lands = await self.card_service.search(
            card_type="land",
            standard_only=True,
            limit=500,
        )
        all_cards.extend(lands)
        logger.info(f"Found {len(lands)} lands")

        # Get colorless artifacts
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
        colorless_query = select(Card).where(
            Card.is_standard_legal == True,
            Card.colors == [],  # Empty colors array
            ~Card.type_line.ilike("%land%"),  # Exclude lands
            ~Card.type_line.ilike("%artifact%"),  # Exclude artifacts (already fetched)
        ).order_by(Card.name).limit(5000)  # Fetch many to dedupe
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

    async def _get_mana_base_from_meta(self, archetype: str, colors: List[str]) -> Dict[str, Any]:
        """Get recommended mana base by analyzing similar tournament decks."""
        from app.models.card import Card
        from sqlalchemy import func

        colors_upper = [c.upper() for c in colors]
        is_mono_color = len(colors_upper) == 1

        # Find similar decklists from tournaments
        query = select(Decklist).join(Event).where(Event.format == "standard")

        # Try to match archetype name (fuzzy)
        if archetype:
            query = query.where(Decklist.archetype.ilike(f"%{archetype}%"))

        query = query.limit(50)  # Get more to filter by color
        result = await self.db.execute(query)
        decklists = result.scalars().all()

        if not decklists:
            # Fallback: get any recent decklists
            query = select(Decklist).join(Event).where(Event.format == "standard").limit(50)
            result = await self.db.execute(query)
            decklists = result.scalars().all()

        # Collect all unique card names from decklists
        all_card_names = set()
        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    all_card_names.add(card_name)

        # Batch query to find lands and their color identity
        land_query = select(Card).where(
            func.lower(Card.name).in_([n.lower() for n in all_card_names]),
            Card.type_line.ilike("%land%")
        )
        land_result = await self.db.execute(land_query)
        all_lands = land_result.scalars().all()

        # Build a map of land name -> color identity, filtering for our colors
        valid_lands = {}
        for land in all_lands:
            land_colors = land.color_identity or []
            land_name_lower = land.name.lower()

            # Skip if we already processed this land
            if land_name_lower in valid_lands:
                continue

            # For mono-color decks, only allow:
            # - Basic lands of the color
            # - Colorless lands (empty color identity)
            # - Lands that only produce our color
            if is_mono_color:
                if len(land_colors) == 0:
                    # Colorless utility lands are fine
                    valid_lands[land_name_lower] = land.name
                elif land_colors == colors_upper:
                    # Exact color match (e.g., a red-only land for mono-red)
                    valid_lands[land_name_lower] = land.name
                # Skip multi-color lands like Restless Spire for mono-color decks
            else:
                # For multi-color decks, allow lands that fit within our colors
                if len(land_colors) == 0 or all(c in colors_upper for c in land_colors):
                    valid_lands[land_name_lower] = land.name

        # Count land occurrences across decklists (only valid lands)
        land_counts = defaultdict(lambda: {"count": 0, "total_quantity": 0})

        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name.lower() in valid_lands:
                    quantity = entry.get("quantity", 1)
                    # Use the canonical name from the database
                    canonical_name = valid_lands[card_name.lower()]
                    land_counts[canonical_name]["count"] += 1
                    land_counts[canonical_name]["total_quantity"] += quantity

        # Calculate average quantity for each land
        recommended_lands = []
        for land_name, data in land_counts.items():
            if data["count"] >= 2:  # Only include lands that appear in multiple decks
                avg_qty = round(data["total_quantity"] / data["count"])
                recommended_lands.append({
                    "name": land_name,
                    "recommended_quantity": avg_qty,
                    "frequency": data["count"],
                })

        # Sort by frequency (most common first)
        recommended_lands.sort(key=lambda x: x["frequency"], reverse=True)

        logger.info(f"Mana base for {colors}: {len(recommended_lands)} valid lands from {len(decklists)} decklists")

        return {
            "sample_size": len(decklists),
            "recommended_lands": recommended_lands[:15],  # Top 15 lands
        }

    async def _get_meta_cards(self, archetype: str, strategy: str = "") -> List[Dict[str, Any]]:
        """Get commonly played non-land cards from tournament decks for the given archetype."""
        from app.models.card import Card
        from sqlalchemy import or_

        # Map strategy keywords to archetype search terms
        strategy_archetype_map = {
            "graveyard": ["reanimator", "graveyard", "dredge"],
            "reanimate": ["reanimator"],
            "reanimation": ["reanimator"],
            "mill": ["mill", "reanimator"],
            "self-mill": ["mill", "reanimator"],
            "discard": ["reanimator", "madness"],
            "sacrifice": ["sacrifice", "aristocrats"],
            "tokens": ["tokens", "go-wide"],
            "ramp": ["ramp", "big mana"],
            "control": ["control"],
            "aggro": ["aggro", "red deck"],
            "burn": ["burn", "red deck", "aggro"],
        }

        # Build archetype search terms from both archetype and strategy
        search_terms = []
        if archetype:
            search_terms.append(archetype)

        # Add strategy-based search terms
        strategy_lower = strategy.lower()
        for keyword, terms in strategy_archetype_map.items():
            if keyword in strategy_lower:
                search_terms.extend(terms)

        # Find similar decklists from tournaments
        query = select(Decklist).join(Event).where(Event.format == "standard")

        if search_terms:
            # Search for any matching archetype term
            conditions = [Decklist.archetype.ilike(f"%{term}%") for term in search_terms]
            query = query.where(or_(*conditions))

        query = query.limit(20)
        result = await self.db.execute(query)
        decklists = result.scalars().all()

        if not decklists:
            return []

        # Collect all card names and quantities
        card_counts = defaultdict(lambda: {"count": 0, "total_quantity": 0})
        all_card_names = set()

        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    all_card_names.add(card_name)
                    quantity = entry.get("quantity", 1)
                    card_counts[card_name]["count"] += 1
                    card_counts[card_name]["total_quantity"] += quantity

        # Batch query to find which cards are NOT lands
        from sqlalchemy import func
        land_query = select(Card.name).where(
            func.lower(Card.name).in_([n.lower() for n in all_card_names]),
            Card.type_line.ilike("%land%")
        ).distinct(Card.name)
        land_result = await self.db.execute(land_query)
        land_names = {row[0].lower() for row in land_result.all()}

        # Filter to non-land cards that appear in multiple decks
        meta_cards = []
        for card_name, data in card_counts.items():
            if card_name.lower() not in land_names and data["count"] >= 2:
                avg_qty = round(data["total_quantity"] / data["count"])
                meta_cards.append({
                    "name": card_name,
                    "recommended_quantity": avg_qty,
                    "frequency": data["count"],
                })

        # Sort by frequency (most common first)
        meta_cards.sort(key=lambda x: x["frequency"], reverse=True)

        return meta_cards

    async def _filter_tournament_cards_by_color(
        self,
        tournament_card_names: set,
        colors: List[str],
    ) -> List[str]:
        """Filter tournament cards to only those matching the deck's colors."""
        from app.models.card import Card
        from sqlalchemy import func

        colors_upper = [c.upper() for c in colors]
        is_mono_color = len(colors_upper) == 1
        matching_cards = []

        # Query all cards that are in the tournament set
        query = select(Card).where(
            func.lower(Card.name).in_([n.lower() for n in tournament_card_names])
        )
        result = await self.db.execute(query)
        cards = result.scalars().all()

        seen = set()
        for card in cards:
            if card.name in seen:
                continue
            seen.add(card.name)

            card_colors = card.colors or []
            is_land = "land" in (card.type_line or "").lower()
            is_colorless = len(card_colors) == 0
            matches_colors = all(c in colors_upper for c in card_colors)

            # For lands, be more selective
            if is_land:
                oracle = (card.oracle_text or "").lower()
                name_lower = card.name.lower()

                # For mono-color decks, exclude fetch lands and multi-color fixing
                if is_mono_color:
                    # Skip fetch lands (search for basic land)
                    if "search" in oracle and "basic land" in oracle:
                        continue
                    # Skip lands that choose colors or produce any color
                    if "any color" in oracle or "choose" in oracle:
                        continue
                    # Skip obvious multi-color lands
                    if any(x in name_lower for x in ["passage", "gate", "plaza", "junction"]):
                        continue

                matching_cards.append(card.name)
            elif is_colorless or matches_colors:
                matching_cards.append(card.name)

        return matching_cards

    async def _get_all_tournament_cards(self) -> List[str]:
        """Get all unique card names that appear in tournament decklists. Uses cache."""
        global _tournament_cards_cache

        # Check cache
        now = time.time()
        if (_tournament_cards_cache["data"] is not None and
            now - _tournament_cards_cache["timestamp"] < _tournament_cards_cache["ttl"]):
            logger.debug("Using cached tournament cards")
            return _tournament_cards_cache["data"]

        # Fetch from database
        query = select(Decklist).join(Event).where(Event.format == "standard").limit(200)
        result = await self.db.execute(query)
        decklists = result.scalars().all()

        all_cards = set()
        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    all_cards.add(card_name)
            for entry in (decklist.sideboard or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    all_cards.add(card_name)

        cards_list = list(all_cards)

        # Update cache
        _tournament_cards_cache["data"] = cards_list
        _tournament_cards_cache["timestamp"] = now
        logger.info(f"Cached {len(cards_list)} tournament cards")

        return cards_list

    async def _find_decks_with_cards(self, card_names: List[str]) -> List[Decklist]:
        """Find tournament decklists that contain the specified cards."""
        # Search for decklists containing any of the specified cards
        query = select(Decklist).join(Event).where(Event.format == "standard")

        result = await self.db.execute(query.limit(100))
        all_decklists = result.scalars().all()

        # Filter to decklists that contain the specified cards
        matching_decklists = []
        card_names_lower = [name.lower() for name in card_names]

        for decklist in all_decklists:
            deck_card_names = [
                entry.get("card_name", "").lower()
                for entry in (decklist.main_deck or [])
            ]
            # Check if any of the specified cards are in this deck
            if any(card in deck_card_names for card in card_names_lower):
                matching_decklists.append(decklist)

        logger.info(f"Found {len(matching_decklists)} decklists containing {card_names}")
        return matching_decklists

    def _extract_colors_from_decks(self, decklists: List[Decklist]) -> List[str]:
        """Extract colors by analyzing the actual land base in decklists."""
        from collections import Counter

        # Count colors based on dual lands in the decks
        color_counts = Counter()

        # Map of dual land names to their colors
        dual_land_colors = {
            # Shock lands
            "hallowed fountain": ["W", "U"], "watery grave": ["U", "B"],
            "blood crypt": ["B", "R"], "stomping ground": ["R", "G"],
            "temple garden": ["G", "W"], "godless shrine": ["W", "B"],
            "steam vents": ["U", "R"], "overgrown tomb": ["B", "G"],
            "sacred foundry": ["R", "W"], "breeding pool": ["G", "U"],
            # Slow lands
            "deserted beach": ["W", "U"], "shipwreck marsh": ["U", "B"],
            "haunted ridge": ["B", "R"], "rockfall vale": ["R", "G"],
            "overgrown farmland": ["G", "W"], "shattered sanctum": ["W", "B"],
            "stormcarved coast": ["U", "R"], "deathcap glade": ["B", "G"],
            "sundown pass": ["R", "W"], "dreamroot cascade": ["G", "U"],
            # Pain lands
            "adarkar wastes": ["W", "U"], "underground river": ["U", "B"],
            "sulfurous springs": ["B", "R"], "karplusan forest": ["R", "G"],
            "brushland": ["G", "W"], "caves of koilos": ["W", "B"],
            "shivan reef": ["U", "R"], "llanowar wastes": ["B", "G"],
            "battlefield forge": ["R", "W"], "yavimaya coast": ["G", "U"],
            # Surveil lands
            "meticulous archive": ["W", "U"], "undercity sewers": ["U", "B"],
            "raucous theater": ["B", "R"], "commercial district": ["R", "G"],
            "lush portico": ["G", "W"], "shadowy backstreet": ["W", "B"],
            "thundering falls": ["U", "R"], "underground mortuary": ["B", "G"],
            "elegant parlor": ["R", "W"], "hedge maze": ["G", "U"],
            # Fast lands
            "seachrome coast": ["W", "U"], "darkslick shores": ["U", "B"],
            "blackcleave cliffs": ["B", "R"], "copperline gorge": ["R", "G"],
            "razorverge thicket": ["G", "W"], "concealed courtyard": ["W", "B"],
            "spirebluff canal": ["U", "R"], "blooming marsh": ["B", "G"],
            "inspiring vantage": ["R", "W"], "botanical sanctum": ["G", "U"],
            # Pathways and other common duals
            "hengegate pathway": ["W", "U"], "clearwater pathway": ["U", "B"],
            "blightstep pathway": ["B", "R"], "cragcrown pathway": ["R", "G"],
            "branchloft pathway": ["G", "W"], "brightclimb pathway": ["W", "B"],
            "riverglide pathway": ["U", "R"], "darkbore pathway": ["B", "G"],
            "needleverge pathway": ["R", "W"], "barkchannel pathway": ["G", "U"],
            # Verge lands
            "sunlit verge": ["W", "U"], "gloomlake verge": ["U", "B"],
            "blazemire verge": ["B", "R"], "thornspire verge": ["R", "G"],
            "willowrush verge": ["G", "W"], "shadowed verge": ["W", "B"],
            "stormwild verge": ["U", "R"], "wastewood verge": ["B", "G"],
            "emberfall verge": ["R", "W"], "flooded verge": ["G", "U"],
        }

        for decklist in decklists:
            deck_colors = set()
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "").lower()
                if card_name in dual_land_colors:
                    for color in dual_land_colors[card_name]:
                        deck_colors.add(color)
                # Also check basic lands
                elif card_name == "plains":
                    deck_colors.add("W")
                elif card_name == "island":
                    deck_colors.add("U")
                elif card_name == "swamp":
                    deck_colors.add("B")
                elif card_name == "mountain":
                    deck_colors.add("R")
                elif card_name == "forest":
                    deck_colors.add("G")

            if deck_colors:
                color_combos = tuple(sorted(deck_colors))
                color_counts[color_combos] += 1

        if color_counts:
            most_common = color_counts.most_common(1)[0][0]
            print(f"[AI-SERVICE] Extracted colors {list(most_common)} from {len(decklists)} decklists")
            return list(most_common)

        print(f"[AI-SERVICE] Could not extract colors from decklists")
        return []

    def _extract_cards_from_decks(self, decklists: List[Decklist]) -> List[Dict[str, Any]]:
        """Extract the most commonly played cards from decklists."""
        card_counts = defaultdict(lambda: {"count": 0, "total_quantity": 0})

        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    quantity = entry.get("quantity", 1)
                    card_counts[card_name]["count"] += 1
                    card_counts[card_name]["total_quantity"] += quantity

        # Build list of commonly played cards
        common_cards = []
        for card_name, data in card_counts.items():
            avg_qty = round(data["total_quantity"] / data["count"])
            common_cards.append({
                "name": card_name,
                "recommended_quantity": avg_qty,
                "frequency": data["count"],
            })

        # Sort by frequency
        common_cards.sort(key=lambda x: x["frequency"], reverse=True)
        return common_cards

    async def _get_archetype_decklists(
        self,
        archetype: str,
        colors: List[str],
        strategy: str = "",
        limit: int = MAX_DECKLIST_EXAMPLES
    ) -> List[Decklist]:
        """Get tournament decklists for a given archetype."""
        from app.models.card import Card
        from sqlalchemy import func, or_

        # Map strategy keywords to archetype search terms
        strategy_archetype_map = {
            "graveyard": ["reanimator", "graveyard", "dredge"],
            "reanimate": ["reanimator"],
            "reanimation": ["reanimator"],
            "mill": ["mill", "reanimator"],
            "self-mill": ["mill", "reanimator"],
            "discard": ["reanimator", "madness"],
            "sacrifice": ["sacrifice", "aristocrats"],
            "tokens": ["tokens", "go-wide"],
            "ramp": ["ramp", "big mana"],
            "control": ["control"],
            "aggro": ["aggro", "red deck"],
            "burn": ["burn", "red deck", "aggro"],
        }

        # Build archetype search terms from both archetype and strategy
        search_terms = []
        if archetype:
            search_terms.append(archetype)

        # Add strategy-based search terms
        strategy_lower = strategy.lower()
        for keyword, terms in strategy_archetype_map.items():
            if keyword in strategy_lower:
                search_terms.extend(terms)

        query = select(Decklist).join(Event).where(
            Event.format == "standard"
        )

        # Match archetype name (fuzzy) with any search term
        if search_terms:
            conditions = [Decklist.archetype.ilike(f"%{term}%") for term in search_terms]
            query = query.where(or_(*conditions))

        # Order by placement (best finishes first)
        query = query.order_by(Decklist.placement.asc().nullslast()).limit(limit * 3)

        result = await self.db.execute(query)
        decklists = result.scalars().all()

        if not decklists:
            # Fallback: get any recent top-placing decklists
            query = select(Decklist).join(Event).where(
                Event.format == "standard"
            ).order_by(Decklist.placement.asc().nullslast()).limit(limit * 3)
            result = await self.db.execute(query)
            decklists = result.scalars().all()

        # Filter to decklists that roughly match the requested colors
        if colors and decklists:
            colors_upper = set(c.upper() for c in colors)
            matching_decklists = []
            for decklist in decklists:
                # Infer deck colors from archetype name
                deck_colors = self._infer_colors_from_archetype(decklist.archetype or "")
                if not deck_colors or deck_colors == colors_upper or deck_colors.issubset(colors_upper):
                    matching_decklists.append(decklist)
                    if len(matching_decklists) >= limit:
                        break
            return matching_decklists

        return decklists[:limit]

    def _infer_colors_from_archetype(self, archetype: str) -> set:
        """Infer color identity from archetype name."""
        archetype_lower = archetype.lower()
        colors = set()

        color_keywords = {
            "white": "W", "mono-w": "W", "mono white": "W",
            "blue": "U", "mono-u": "U", "mono blue": "U",
            "black": "B", "mono-b": "B", "mono black": "B",
            "red": "R", "mono-r": "R", "mono red": "R",
            "green": "G", "mono-g": "G", "mono green": "G",
            "azorius": "WU", "dimir": "UB", "rakdos": "BR",
            "gruul": "RG", "selesnya": "GW", "orzhov": "WB",
            "izzet": "UR", "golgari": "BG", "boros": "RW",
            "simic": "GU", "esper": "WUB", "grixis": "UBR",
            "jund": "BRG", "naya": "RGW", "bant": "GWU",
            "abzan": "WBG", "jeskai": "URW", "sultai": "BGU",
            "mardu": "RWB", "temur": "GUR",
        }

        for keyword, color_str in color_keywords.items():
            if keyword in archetype_lower:
                colors.update(color_str)

        return colors

    def _format_decklists_as_examples(self, decklists: List[Decklist], max_examples: int = MAX_DECKLIST_EXAMPLES) -> str:
        """Format tournament decklists as full examples for the AI prompt."""
        if not decklists:
            return ""

        # Sort by placement (best finishes first) and limit
        sorted_decklists = sorted(
            decklists,
            key=lambda d: d.placement if d.placement else 999
        )[:max_examples]

        examples = []
        for decklist in sorted_decklists:
            # Format main deck
            main_deck_lines = []
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                quantity = entry.get("quantity", 0)
                if card_name and quantity:
                    main_deck_lines.append(f"{quantity} {card_name}")

            # Format sideboard
            sideboard_lines = []
            for entry in (decklist.sideboard or []):
                card_name = entry.get("card_name", "")
                quantity = entry.get("quantity", 0)
                if card_name and quantity:
                    sideboard_lines.append(f"{quantity} {card_name}")

            # Build example string
            finish_str = f" ({decklist.finish_position})" if decklist.finish_position else ""
            archetype_str = decklist.archetype or "Unknown"
            player_str = decklist.player_name or "Unknown"

            example = f"""
--- {archetype_str}{finish_str} by {player_str} ---
Main Deck ({sum(e.get('quantity', 0) for e in decklist.main_deck or [])} cards):
{chr(10).join(main_deck_lines)}

Sideboard ({sum(e.get('quantity', 0) for e in decklist.sideboard or [])} cards):
{chr(10).join(sideboard_lines)}
"""
            examples.append(example)

        return "\n".join(examples)

    async def _get_cooccurrence_cards(
        self,
        card_names: List[str],
        colors: List[str],
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Get cards that frequently co-occur with the given cards in tournament decks."""
        from app.models.card import Card
        from sqlalchemy import func, or_, union_all, literal_column

        if not card_names:
            return []

        card_names_lower = [n.lower() for n in card_names]

        # Co-occurrence stores pairs in sorted order, so we need to check both directions
        # Query 1: where card_a matches, return card_b
        query_a = select(
            CardCooccurrence.card_b.label("partner"),
            CardCooccurrence.cooccurrence_count.label("count")
        ).where(
            CardCooccurrence.format == "standard",
            func.lower(CardCooccurrence.card_a).in_(card_names_lower)
        )

        # Query 2: where card_b matches, return card_a
        query_b = select(
            CardCooccurrence.card_a.label("partner"),
            CardCooccurrence.cooccurrence_count.label("count")
        ).where(
            CardCooccurrence.format == "standard",
            func.lower(CardCooccurrence.card_b).in_(card_names_lower)
        )

        # Union and aggregate
        combined = union_all(query_a, query_b).subquery()
        final_query = select(
            combined.c.partner,
            func.sum(combined.c.count).label("total_count")
        ).group_by(
            combined.c.partner
        ).order_by(
            func.sum(combined.c.count).desc()
        ).limit(limit * 2)

        result = await self.db.execute(final_query)
        cooccurrence_results = result.all()

        if not cooccurrence_results:
            logger.info(f"No co-occurrence data found for {card_names}")
            return []

        # Get card info to filter by color
        card_names_to_check = [row[0] for row in cooccurrence_results]
        colors_upper = [c.upper() for c in colors]

        card_query = select(Card).where(
            func.lower(Card.name).in_([n.lower() for n in card_names_to_check]),
            Card.is_standard_legal == True
        )
        card_result = await self.db.execute(card_query)
        cards = card_result.scalars().all()

        # Build map of card name to colors
        card_color_map = {}
        for card in cards:
            card_colors = card.colors or []
            is_colorless = len(card_colors) == 0
            is_land = "land" in (card.type_line or "").lower()
            matches_colors = all(c in colors_upper for c in card_colors)
            if is_colorless or is_land or matches_colors:
                card_color_map[card.name.lower()] = card

        # Filter results by color and build output
        synergy_cards = []
        for card_b, count in cooccurrence_results:
            card_b_lower = card_b.lower()
            if card_b_lower in card_color_map and card_b_lower not in [n.lower() for n in card_names]:
                synergy_cards.append({
                    "name": card_color_map[card_b_lower].name,
                    "cooccurrence_count": count,
                    "recommended_quantity": 4,  # Will be refined later
                })
                if len(synergy_cards) >= limit:
                    break

        logger.info(f"Found {len(synergy_cards)} co-occurring cards for {card_names}")
        return synergy_cards

    async def _get_sideboard_patterns(
        self,
        archetype: str,
        colors: List[str]
    ) -> Dict[str, Any]:
        """Analyze sideboard patterns from tournament decks to identify hate cards."""
        from app.models.card import Card
        from sqlalchemy import func

        colors_upper = [c.upper() for c in colors]

        # Get decklists with sideboards
        query = select(Decklist).join(Event).where(
            Event.format == "standard"
        )
        if archetype:
            query = query.where(Decklist.archetype.ilike(f"%{archetype}%"))
        query = query.limit(100)

        result = await self.db.execute(query)
        decklists = result.scalars().all()

        if not decklists:
            return {"sideboard_staples": [], "matchup_cards": {}}

        # Count sideboard card frequencies
        sideboard_counts = defaultdict(lambda: {"count": 0, "total_quantity": 0})
        all_sideboard_names = set()

        for decklist in decklists:
            for entry in (decklist.sideboard or []):
                card_name = entry.get("card_name", "")
                quantity = entry.get("quantity", 0)
                if card_name:
                    all_sideboard_names.add(card_name)
                    sideboard_counts[card_name]["count"] += 1
                    sideboard_counts[card_name]["total_quantity"] += quantity

        # Filter to cards matching the deck's colors
        valid_sideboard_cards = set()
        if all_sideboard_names:
            card_query = select(Card).where(
                func.lower(Card.name).in_([n.lower() for n in all_sideboard_names]),
                Card.is_standard_legal == True
            )
            card_result = await self.db.execute(card_query)
            cards = card_result.scalars().all()

            for card in cards:
                card_colors = card.colors or []
                is_colorless = len(card_colors) == 0
                matches_colors = all(c in colors_upper for c in card_colors)
                if is_colorless or matches_colors:
                    valid_sideboard_cards.add(card.name.lower())

        # Build sideboard staples list
        sideboard_staples = []
        for card_name, data in sideboard_counts.items():
            if card_name.lower() in valid_sideboard_cards and data["count"] >= 3:
                avg_qty = round(data["total_quantity"] / data["count"])
                sideboard_staples.append({
                    "name": card_name,
                    "frequency": data["count"],
                    "avg_quantity": avg_qty,
                    "total_decks": len(decklists),
                })

        # Sort by frequency
        sideboard_staples.sort(key=lambda x: x["frequency"], reverse=True)

        return {
            "sideboard_staples": sideboard_staples[:20],
            "sample_size": len(decklists),
        }

    async def _get_deck_composition_from_meta(
        self,
        archetype: str,
        colors: List[str]
    ) -> Dict[str, Any]:
        """Analyze actual tournament decks to derive typical composition ratios."""
        from app.models.card import Card
        from sqlalchemy import func

        query = select(Decklist).join(Event).where(Event.format == "standard")
        if archetype:
            query = query.where(Decklist.archetype.ilike(f"%{archetype}%"))
        query = query.limit(50)

        result = await self.db.execute(query)
        decklists = result.scalars().all()

        if not decklists:
            return {
                "avg_creatures": 20,
                "avg_spells": 16,
                "avg_lands": 24,
                "sample_size": 0
            }

        # Collect all card names to query types
        all_card_names = set()
        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                all_card_names.add(entry.get("card_name", ""))

        # Query card types
        card_query = select(Card.name, Card.type_line).where(
            func.lower(Card.name).in_([n.lower() for n in all_card_names])
        )
        card_result = await self.db.execute(card_query)
        card_types = {row[0].lower(): row[1] for row in card_result.all()}

        # Analyze composition
        compositions = []
        for decklist in decklists:
            creatures = 0
            lands = 0
            other_spells = 0

            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                quantity = entry.get("quantity", 0)
                type_line = card_types.get(card_name.lower(), "").lower()

                if "land" in type_line:
                    lands += quantity
                elif "creature" in type_line:
                    creatures += quantity
                else:
                    other_spells += quantity

            compositions.append({
                "creatures": creatures,
                "lands": lands,
                "spells": other_spells,
            })

        # Calculate averages
        avg_creatures = round(sum(c["creatures"] for c in compositions) / len(compositions))
        avg_lands = round(sum(c["lands"] for c in compositions) / len(compositions))
        avg_spells = round(sum(c["spells"] for c in compositions) / len(compositions))

        logger.info(f"Deck composition for {archetype}: {avg_creatures} creatures, {avg_spells} spells, {avg_lands} lands (from {len(decklists)} decks)")

        return {
            "avg_creatures": avg_creatures,
            "avg_spells": avg_spells,
            "avg_lands": avg_lands,
            "sample_size": len(decklists),
        }

    async def _semantic_card_search(
        self,
        strategy: str,
        colors: List[str],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Use semantic search to find relevant cards based on strategy description."""
        try:
            cards = await self.card_service.semantic_search(
                query=strategy,
                limit=limit,
                standard_only=True,
                colors=colors if colors else None,
            )

            return [
                {
                    "name": c.name,
                    "type_line": c.type_line,
                    "oracle_text": c.oracle_text,
                    "relevance": "semantic_match",
                }
                for c in cards
            ]
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    async def _detect_card_themes(self, card_names: List[str]) -> List[str]:
        """Detect themes from requested cards' oracle text and types."""
        from app.models.card import Card
        from sqlalchemy import func

        themes = set()

        # Look up the requested cards
        for card_name in card_names:
            query = select(Card).where(func.lower(Card.name).ilike(f"%{card_name.lower()}%")).limit(1)
            result = await self.db.execute(query)
            card = result.scalar_one_or_none()

            if not card:
                continue

            oracle = (card.oracle_text or "").lower()
            type_line = (card.type_line or "").lower()

            # Detect artifact theme
            if "artifact" in oracle or "artifact" in type_line:
                themes.add("artifact")

            # Detect +1/+1 counter theme
            if "+1/+1 counter" in oracle:
                themes.add("counters")

            # Detect tribal themes
            creature_types = ["goblin", "merfolk", "elf", "zombie", "vampire", "human", "angel", "dragon", "elemental"]
            for creature_type in creature_types:
                if creature_type in oracle or creature_type in type_line:
                    themes.add(creature_type)

            # Detect graveyard theme
            if "graveyard" in oracle or "dies" in oracle or "from your graveyard" in oracle:
                themes.add("graveyard")

            # Detect tokens theme
            if "create" in oracle and ("token" in oracle or "tokens" in oracle):
                themes.add("tokens")

            # Detect spellslinger theme
            if "instant" in oracle or "sorcery" in oracle or "noncreature spell" in oracle:
                themes.add("spellslinger")

        logger.info(f"Detected themes from {card_names}: {themes}")
        return list(themes)

    async def _get_tournament_synergy_cards(self, themes: List[str]) -> List[Dict[str, Any]]:
        """Get cards that match the theme AND appear in tournament decklists."""
        from app.models.card import Card
        from sqlalchemy import or_, func

        # First get all cards from tournament decklists with their frequency
        query = select(Decklist).join(Event).where(Event.format == "standard").limit(200)
        result = await self.db.execute(query)
        decklists = result.scalars().all()

        # Count card frequencies in tournaments
        card_frequencies = defaultdict(lambda: {"count": 0, "total_quantity": 0})
        for decklist in decklists:
            for entry in (decklist.main_deck or []):
                card_name = entry.get("card_name", "")
                if card_name:
                    card_frequencies[card_name]["count"] += 1
                    card_frequencies[card_name]["total_quantity"] += entry.get("quantity", 1)

        tournament_card_names = set(card_frequencies.keys())

        # Now find which of these match our themes
        synergy_cards = []

        for theme in themes:
            conditions = [
                Card.is_standard_legal == True,
                func.lower(Card.name).in_([n.lower() for n in tournament_card_names])
            ]

            if theme == "artifact":
                conditions.append(
                    or_(
                        Card.type_line.ilike("%artifact%"),
                        Card.oracle_text.ilike("%artifact%"),
                    )
                )
            elif theme == "counters":
                conditions.append(Card.oracle_text.ilike("%+1/+1 counter%"))
            elif theme == "graveyard":
                conditions.append(
                    or_(
                        Card.oracle_text.ilike("%graveyard%"),
                        Card.oracle_text.ilike("%from your graveyard%"),
                    )
                )
            elif theme == "tokens":
                conditions.append(Card.oracle_text.ilike("%create%token%"))
            elif theme == "spellslinger":
                conditions.append(
                    or_(
                        Card.oracle_text.ilike("%instant%"),
                        Card.oracle_text.ilike("%sorcery%"),
                    )
                )

            query = select(Card).where(*conditions).limit(100)
            result = await self.db.execute(query)
            cards_raw = result.scalars().all()

            # Deduplicate and add frequency data
            seen = set()
            for c in cards_raw:
                if c.name not in seen:
                    seen.add(c.name)
                    freq_data = card_frequencies.get(c.name, {"count": 0, "total_quantity": 0})
                    if freq_data["count"] > 0:
                        avg_qty = round(freq_data["total_quantity"] / freq_data["count"])
                        synergy_cards.append({
                            "name": c.name,
                            "recommended_quantity": avg_qty,
                            "frequency": freq_data["count"],
                            "theme": theme,
                        })

        # Sort by tournament frequency (most played first)
        synergy_cards.sort(key=lambda x: x["frequency"], reverse=True)
        return synergy_cards

    async def _get_synergy_cards(self, themes: List[str], colors: List[str]) -> List[Dict[str, Any]]:
        """Find cards that synergize with the detected themes."""
        from app.models.card import Card
        from sqlalchemy import or_, func

        synergy_cards = []

        for theme in themes:
            conditions = [Card.is_standard_legal == True]

            if theme == "artifact":
                # Find artifacts and cards that reference artifacts
                conditions.append(
                    or_(
                        Card.type_line.ilike("%artifact%"),
                        Card.oracle_text.ilike("%artifact%"),
                    )
                )
            elif theme == "counters":
                conditions.append(Card.oracle_text.ilike("%+1/+1 counter%"))
            elif theme == "graveyard":
                conditions.append(
                    or_(
                        Card.oracle_text.ilike("%graveyard%"),
                        Card.oracle_text.ilike("%from your graveyard%"),
                    )
                )
            elif theme == "tokens":
                conditions.append(Card.oracle_text.ilike("%create%token%"))
            elif theme == "spellslinger":
                conditions.append(
                    or_(
                        Card.oracle_text.ilike("%instant%"),
                        Card.oracle_text.ilike("%sorcery%"),
                        Card.type_line.ilike("%instant%"),
                        Card.type_line.ilike("%sorcery%"),
                    )
                )
            elif theme in ["goblin", "merfolk", "elf", "zombie", "vampire", "human", "angel", "dragon", "elemental"]:
                # Tribal - find creatures of that type
                conditions.append(
                    or_(
                        Card.type_line.ilike(f"%{theme}%"),
                        Card.oracle_text.ilike(f"%{theme}%"),
                    )
                )

            query = select(Card).where(*conditions).order_by(Card.name).limit(5000)
            result = await self.db.execute(query)
            cards_raw = result.scalars().all()

            # Deduplicate
            seen = set()
            for c in cards_raw:
                if c.name not in seen:
                    seen.add(c.name)
                    synergy_cards.append({
                        "name": c.name,
                        "recommended_quantity": 4 if "legendary" not in (c.type_line or "").lower() else 2,
                        "frequency": 1,  # No tournament data, so frequency is 1
                        "theme": theme,
                    })

        logger.info(f"Found {len(synergy_cards)} synergy cards for themes {themes}")
        return synergy_cards

    async def _filter_by_color(
        self,
        deck_data: Dict[str, Any],
        colors: List[str],
    ) -> Dict[str, Any]:
        """Remove cards that don't match the deck's color identity."""
        from app.models.card import Card
        from sqlalchemy import func

        colors_upper = [c.upper() for c in colors]

        async def filter_list(card_list: List[Dict]) -> List[Dict]:
            filtered = []
            for entry in card_list:
                card_name = entry.get("card_name", "")
                # Look up the card's colors
                query = select(Card).where(func.lower(Card.name) == card_name.lower()).limit(1)
                result = await self.db.execute(query)
                card = result.scalar_one_or_none()

                if card:
                    card_colors = card.colors or []
                    card_color_identity = card.color_identity or []
                    is_land = "land" in (card.type_line or "").lower()
                    is_colorless = len(card_colors) == 0 and len(card_color_identity) == 0

                    # For lands, check color identity (not colors, since lands have no mana cost)
                    if is_land:
                        # Allow lands with no color identity or color identity within our colors
                        land_fits = len(card_color_identity) == 0 or all(c in colors_upper for c in card_color_identity)
                        if land_fits:
                            filtered.append(entry)
                        else:
                            print(f"[AI-SERVICE] Removed off-color land: {card_name} (color_identity: {card_color_identity})")
                    elif is_colorless:
                        # Colorless non-land cards are always allowed
                        filtered.append(entry)
                    elif all(c in colors_upper for c in card_colors):
                        # Card colors match deck colors
                        filtered.append(entry)
                    else:
                        print(f"[AI-SERVICE] Removed off-color card: {card_name} (colors: {card_colors})")
                else:
                    filtered.append(entry)  # Keep if we can't verify

            return filtered

        deck_data["main_deck"] = await filter_list(deck_data.get("main_deck", []))
        deck_data["sideboard"] = await filter_list(deck_data.get("sideboard", []))

        return deck_data

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
                    print(f"[AI-SERVICE] Removed duplicate in singleton deck: {card_name}")
            main_deck = deduped
            deck_data["main_deck"] = main_deck

        # Get basic land for the colors
        basic_land_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
        primary_basic = basic_land_map.get(colors[0].upper() if colors else "R", "Mountain")

        # First, ensure we have enough lands
        archetype = deck_data.get("archetype", "").lower()
        if format == "cedh":
            min_lands = 28  # cEDH runs fewer lands due to fast mana
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

        print(f"[AI-SERVICE] Current land count: {current_land_count}, minimum: {min_lands}")

        # If we don't have enough lands, add more basics and trim spells
        if current_land_count < min_lands:
            lands_to_add = min_lands - current_land_count
            print(f"[AI-SERVICE] Need to add {lands_to_add} more lands")

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
                        print(f"[AI-SERVICE] Trimmed {trim_amount}x {entry.get('card_name')} to make room for lands")
                        trimmed = True
                        if remaining_to_trim <= 0:
                            break
                if not trimmed:
                    # Remove entire cards if we can't trim quantities
                    if non_land_entries:
                        removed = non_land_entries.pop()
                        remaining_to_trim -= removed.get("quantity", 0)
                        print(f"[AI-SERVICE] Removed {removed.get('card_name')} to make room for lands")

            # Rebuild main_deck with lands first, then non-lands
            main_deck = land_entries + [e for e in non_land_entries if e.get("quantity", 0) > 0]
            deck_data["main_deck"] = main_deck

        main_count = sum(e.get("quantity", 0) for e in main_deck)
        side_count = sum(e.get("quantity", 0) for e in sideboard)

        # Fix main deck to target size
        if main_count < target_main:
            deficit = target_main - main_count
            print(f"[AI-SERVICE] Deck has {main_count} cards, need {deficit} more (target: {target_main})")

            # First, try to add more non-land cards from available cards
            existing_cards = {entry.get("card_name", "").lower() for entry in main_deck}

            # Get non-land available cards that aren't already in the deck
            available_nonlands = [
                c for c in available_cards
                if c.get("name", "").lower() not in existing_cards
                and not (c.get("type_line") or "").lower().startswith("land")
                and "land" not in (c.get("type_line") or "").lower().split(" — ")[0].lower()
            ]

            # Add cards from available pool (prefer cards that appear in tournament decks)
            cards_added = 0
            for card in available_nonlands:
                if deficit <= 0:
                    break
                # Singleton for cEDH, up to 4 for other formats
                qty_to_add = min(max_copies, deficit)
                main_deck.append({"card_name": card["name"], "quantity": qty_to_add})
                deficit -= qty_to_add
                cards_added += qty_to_add
                print(f"[AI-SERVICE] Added {qty_to_add}x {card['name']} to fill deck")

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
                print(f"[AI-SERVICE] Added {deficit} {primary_basic} as filler (last resort)")

        elif main_count > target_main:
            # Remove cards from the end (typically less important)
            excess = main_count - target_main
            while excess > 0 and main_deck:
                last_entry = main_deck[-1]
                qty = last_entry.get("quantity", 1)
                if qty <= excess:
                    excess -= qty
                    main_deck.pop()
                    print(f"[AI-SERVICE] Removed {qty}x {last_entry.get('card_name')} to reduce main deck")
                else:
                    last_entry["quantity"] -= excess
                    print(f"[AI-SERVICE] Reduced {last_entry.get('card_name')} by {excess} to reach {target_main}")
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
                print(f"[AI-SERVICE] Fixed sideboard to {target_side} cards")

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
                print(f"[AI-SERVICE] Trimmed sideboard to {target_side} cards")

        deck_data["main_deck"] = main_deck
        deck_data["sideboard"] = sideboard
        return deck_data

    async def _generate_fallback_deck(
        self,
        archetype: str,
        colors: List[str],
        strategy: str,
        format: str = "standard",
    ) -> Dict[str, Any]:
        """Generate a basic deck structure as fallback."""
        # Get available cards
        available = await self.card_service.search(
            colors=colors,
            format=format,
            limit=100,
        )

        main_deck = []
        sideboard = []

        # Simple deck building heuristic
        creatures = [c for c in available if "creature" in (c.type_line or "").lower()]
        instants = [c for c in available if "instant" in (c.type_line or "").lower()]
        sorceries = [c for c in available if "sorcery" in (c.type_line or "").lower()]
        lands = [c for c in available if "land" in (c.type_line or "").lower()]

        # Add creatures (20-24)
        for i, card in enumerate(creatures[:6]):
            main_deck.append({"card_name": card.name, "quantity": 4})

        # Add spells (12-16)
        for card in instants[:2]:
            main_deck.append({"card_name": card.name, "quantity": 4})
        for card in sorceries[:1]:
            main_deck.append({"card_name": card.name, "quantity": 4})

        # Add lands (20-24)
        basic_land_map = {"R": "Mountain", "U": "Island", "B": "Swamp", "W": "Plains", "G": "Forest"}
        if colors:
            primary_color = colors[0]
            land_name = basic_land_map.get(primary_color, "Mountain")
            main_deck.append({"card_name": land_name, "quantity": 20})

        # Sideboard
        for card in instants[2:5]:
            sideboard.append({"card_name": card.name, "quantity": 3})

        deck_data = {
            "name": f"{archetype.title()} {'/'.join(colors)}",
            "strategy_summary": f"A {archetype} deck focusing on {strategy}",
            "main_deck": main_deck,
            "sideboard": sideboard,
            "slot_recommendations": [],
            "sideboard_guide": [],
        }

        # Enrich deck entries with card type information
        deck_data = await self._enrich_deck_with_card_data(deck_data)

        return deck_data

    async def _enrich_deck_with_card_data(
        self,
        deck_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enrich deck entries with card data (type_line) for frontend categorization.
        """
        from app.models.card import Card
        from sqlalchemy import func

        # Collect all card names from the deck
        all_card_names = set()
        for entry in deck_data.get("main_deck", []):
            all_card_names.add(entry.get("card_name", ""))
        for entry in deck_data.get("sideboard", []):
            all_card_names.add(entry.get("card_name", ""))

        if not all_card_names:
            return deck_data

        # Fetch all cards in one query
        query = select(Card).where(
            func.lower(Card.name).in_([n.lower() for n in all_card_names])
        )
        result = await self.db.execute(query)
        cards = result.scalars().all()

        # Build a lookup map by lowercase name
        card_map = {}
        for card in cards:
            card_map[card.name.lower()] = {
                "type_line": card.type_line,
                "mana_cost": card.mana_cost,
                "oracle_text": card.oracle_text,
                "colors": card.colors,
                "image_uri": card.image_uri,
            }

        # Enrich main deck entries
        for entry in deck_data.get("main_deck", []):
            card_name = entry.get("card_name", "")
            card_data = card_map.get(card_name.lower(), {})
            if card_data:
                entry["card"] = card_data

        # Enrich sideboard entries
        for entry in deck_data.get("sideboard", []):
            card_name = entry.get("card_name", "")
            card_data = card_map.get(card_name.lower(), {})
            if card_data:
                entry["card"] = card_data

        return deck_data
