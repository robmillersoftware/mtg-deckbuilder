from typing import Optional, List, Dict, Any
from collections import defaultdict
import logging
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.services.card_service import CardService
from app.models.meta import Decklist, Event

logger = logging.getLogger(__name__)


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
        """
        if not settings.ANTHROPIC_API_KEY:
            # Fallback parsing without API
            return await self._fallback_parse(prompt)

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system="""You are a Magic: The Gathering deck building assistant.
Parse the user's deck request and extract the following information in JSON format:
{
    "archetype": "aggro|control|midrange|combo|tempo",
    "colors": ["W", "U", "B", "R", "G"],
    "strategy": "brief description of the strategy",
    "specific_cards": ["any cards specifically mentioned by name"],
    "budget": "competitive|budget|any",
    "focus": "speed|value|resilience|interaction"
}

IMPORTANT: If the user mentions a card name (like "Tezzeret", "Lightning Bolt", etc.), add it to specific_cards.""",
                messages=[{"role": "user", "content": prompt}],
            )

            print(f"[AI-SERVICE] Parse response: {response.content[0].text if response.content else 'EMPTY'}")

            # Extract JSON from response
            if not response.content:
                raise ValueError("Empty AI response")
            content = response.content[0].text
            # Try to find JSON in response
            if "{" in content:
                json_start = content.index("{")
                json_end = content.rindex("}") + 1
                return json.loads(content[json_start:json_end])

        except Exception as e:
            logger.error(f"AI parse error: {e}", exc_info=True)

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
    ) -> Dict[str, Any]:
        """
        Generate a complete deck using AI with constrained card selection.
        """
        specific_cards = specific_cards or []

        if not settings.ANTHROPIC_API_KEY:
            return await self._generate_fallback_deck(archetype, colors, strategy)

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            # If specific cards are requested, find tournament decks that use them
            # and use those as templates for colors and card choices
            template_decks = []
            template_cards = []
            detected_themes = []
            print(f"[AI-SERVICE] Specific cards from parse: {specific_cards}")
            if specific_cards:
                template_decks = await self._find_decks_with_cards(specific_cards)
                print(f"[AI-SERVICE] Found {len(template_decks)} tournament decks with specific cards")
                if template_decks:
                    # Override colors based on what tournament decks actually use
                    template_colors = self._extract_colors_from_decks(template_decks)
                    if template_colors:
                        print(f"[AI-SERVICE] Overriding colors from {colors} to {template_colors} based on tournament decks")
                        colors = template_colors
                    # Get the most played cards from these decks
                    template_cards = self._extract_cards_from_decks(template_decks)
                    print(f"[AI-SERVICE] Found {len(template_decks)} tournament decks using {specific_cards}")
                else:
                    # No tournament data - detect themes from the requested cards
                    print(f"[AI-SERVICE] No tournament decks found for {specific_cards}, detecting themes...")
                    detected_themes = await self._detect_card_themes(specific_cards)
                    print(f"[AI-SERVICE] Detected themes: {detected_themes}")
                    if detected_themes:
                        # Get synergy cards that ALSO appear in tournament play
                        template_cards = await self._get_tournament_synergy_cards(detected_themes)
                        print(f"[AI-SERVICE] Found {len(template_cards)} tournament-played synergy cards for themes {detected_themes}")

            # Get available cards for the colors
            available_cards = await self._get_available_cards(colors)

            # If we have template cards, add them to available cards to ensure they pass hallucination filter
            if template_cards:
                template_card_names = {c["name"] for c in template_cards}
                existing_names = {c["name"] for c in available_cards}
                # Find template cards not already in available cards and fetch them
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
                    # Deduplicate by name
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
                    logger.info(f"Added {len(seen)} template cards to available cards list")

            # Get cards commonly played in similar tournament decks
            meta_cards = await self._get_meta_cards(archetype) if not template_cards else template_cards

            # Get mana base recommendations from tournament decks
            mana_base_data = await self._get_mana_base_from_meta(archetype, colors)
            recommended_lands = mana_base_data.get("recommended_lands", [])
            logger.info(f"Mana base recommendations for {archetype}: {recommended_lands}")

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

            # Log what we're sending to the AI
            logger.info(f"Sending {len(available_cards)} available cards to AI for {archetype} deck")
            if available_cards:
                logger.info(f"Sample available cards: {[c['name'] for c in available_cards[:10]]}")

            # Get cards that appear in tournament decklists for quality reference
            tournament_cards = await self._get_all_tournament_cards()
            tournament_card_names = {c.lower() for c in tournament_cards}
            print(f"[AI-SERVICE] Found {len(tournament_card_names)} unique cards from tournament decklists")

            # Filter tournament cards to only those matching the deck's colors
            on_color_tournament = await self._filter_tournament_cards_by_color(tournament_card_names, colors)
            print(f"[AI-SERVICE] {len(on_color_tournament)} tournament cards match colors {colors}")

            system_prompt = f"""You are Spellbook, an expert Magic: The Gathering deck builder for Standard format.

YOU MUST BUILD A COMPLETE 60-CARD DECK + 15-CARD SIDEBOARD.

Colors: {', '.join(colors)}
{specific_cards_text}
{meta_cards_text}

TOURNAMENT-PLAYED CARDS IN YOUR COLORS (USE THESE):
{chr(10).join(f"- {name}" for name in on_color_tournament[:150])}

MANA BASE from {mana_base_data.get('sample_size', 0)} tournament decks:
{land_recommendations}

DECK BUILDING RULES:
1. Main deck = EXACTLY 60 cards (count them!)
2. Sideboard = EXACTLY 15 cards
3. Use 22-26 lands typically
4. Max 4 copies of non-basic cards
5. ONLY use cards from the tournament list above or basic lands
6. Every card must match colors {colors} or be colorless/land

USER REQUEST: {archetype} deck

Return JSON with EXACTLY 60 main deck cards and 15 sideboard cards:
{{
    "name": "Deck Name",
    "strategy_summary": "Brief strategy description",
    "main_deck": [
        {{"card_name": "Card Name", "quantity": 4}}
    ],
    "sideboard": [
        {{"card_name": "Card Name", "quantity": 3}}
    ]
}}"""

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
                            from sqlalchemy import func
                            query = select(Card.name).where(
                                func.lower(Card.name).like(f"%{card_name.lower()}%"),
                                Card.is_standard_legal == True
                            ).limit(1)
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

                # Filter out cards that don't match the deck's colors
                deck_data = await self._filter_by_color(deck_data, colors)

                # Fix card counts to ensure 60/15
                deck_data = await self._fix_deck_counts(deck_data, colors, available_cards)

                # Validate total counts
                main_count = sum(e.get("quantity", 0) for e in deck_data.get("main_deck", []))
                side_count = sum(e.get("quantity", 0) for e in deck_data.get("sideboard", []))

                print(f"[AI-SERVICE] Generated deck after fixing: {main_count} main, {side_count} sideboard")

                # Enrich deck entries with card type information for frontend categorization
                deck_data = await self._enrich_deck_with_card_data(deck_data)

                return deck_data

        except Exception as e:
            logger.error(f"AI deck generation error: {e}", exc_info=True)

        return await self._generate_fallback_deck(archetype, colors, strategy)

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

    async def _get_meta_cards(self, archetype: str) -> List[Dict[str, Any]]:
        """Get commonly played non-land cards from tournament decks for the given archetype."""
        from app.models.card import Card

        # Find similar decklists from tournaments
        query = select(Decklist).join(Event).where(Event.format == "standard")

        if archetype:
            query = query.where(Decklist.archetype.ilike(f"%{archetype}%"))

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

            if is_land or is_colorless or matches_colors:
                matching_cards.append(card.name)

        return matching_cards

    async def _get_all_tournament_cards(self) -> List[str]:
        """Get all unique card names that appear in tournament decklists."""
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

        return list(all_cards)

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
        """Extract the most common color combination from decklists."""
        from collections import Counter

        # Count color combinations
        color_combos = Counter()
        for decklist in decklists:
            # Try to infer colors from archetype name or cards
            archetype = (decklist.archetype or "").lower()

            colors = []
            if "red" in archetype or "mono-r" in archetype or "gruul" in archetype or "izzet" in archetype or "rakdos" in archetype or "boros" in archetype:
                colors.append("R")
            if "blue" in archetype or "mono-u" in archetype or "izzet" in archetype or "dimir" in archetype or "azorius" in archetype or "simic" in archetype:
                colors.append("U")
            if "black" in archetype or "mono-b" in archetype or "dimir" in archetype or "rakdos" in archetype or "golgari" in archetype or "orzhov" in archetype:
                colors.append("B")
            if "white" in archetype or "mono-w" in archetype or "azorius" in archetype or "boros" in archetype or "selesnya" in archetype or "orzhov" in archetype:
                colors.append("W")
            if "green" in archetype or "mono-g" in archetype or "gruul" in archetype or "simic" in archetype or "golgari" in archetype or "selesnya" in archetype:
                colors.append("G")

            if colors:
                color_combos[tuple(sorted(set(colors)))] += 1

        if color_combos:
            most_common = color_combos.most_common(1)[0][0]
            return list(most_common)

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
    ) -> Dict[str, Any]:
        """Fix deck to have exactly 60 main deck and 15 sideboard cards with proper land count."""
        from app.models.card import Card
        from sqlalchemy import func

        main_deck = deck_data.get("main_deck", [])
        sideboard = deck_data.get("sideboard", [])

        # Get basic land for the colors
        basic_land_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
        primary_basic = basic_land_map.get(colors[0].upper() if colors else "R", "Mountain")

        # First, ensure we have enough lands (minimum 20 for aggro, 22 for others)
        archetype = deck_data.get("archetype", "").lower()
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

        # Fix main deck
        if main_count < 60:
            # Add basic lands to reach 60
            deficit = 60 - main_count
            # Check if we already have the basic land in the deck
            existing_basic = None
            for entry in main_deck:
                if entry.get("card_name") == primary_basic:
                    existing_basic = entry
                    break
            if existing_basic:
                existing_basic["quantity"] += deficit
            else:
                main_deck.append({"card_name": primary_basic, "quantity": deficit})
            print(f"[AI-SERVICE] Added {deficit} {primary_basic} to reach 60 cards")

        elif main_count > 60:
            # Remove cards from the end (typically less important)
            excess = main_count - 60
            while excess > 0 and main_deck:
                last_entry = main_deck[-1]
                qty = last_entry.get("quantity", 1)
                if qty <= excess:
                    excess -= qty
                    main_deck.pop()
                    print(f"[AI-SERVICE] Removed {qty}x {last_entry.get('card_name')} to reduce main deck")
                else:
                    last_entry["quantity"] -= excess
                    print(f"[AI-SERVICE] Reduced {last_entry.get('card_name')} by {excess} to reach 60")
                    excess = 0

        # Fix sideboard
        if side_count < 15:
            # Add basic lands or duplicate existing sideboard cards
            deficit = 15 - side_count
            if sideboard:
                # Try to add more copies of existing sideboard cards
                for entry in sideboard:
                    current_qty = entry.get("quantity", 0)
                    can_add = 4 - current_qty  # Max 4 copies
                    if can_add > 0:
                        add = min(can_add, deficit)
                        entry["quantity"] = current_qty + add
                        deficit -= add
                        if deficit <= 0:
                            break
            if deficit > 0:
                # Add basic land as filler
                sideboard.append({"card_name": primary_basic, "quantity": deficit})
            print(f"[AI-SERVICE] Fixed sideboard to 15 cards")

        elif side_count > 15:
            # Remove cards from sideboard
            excess = side_count - 15
            while excess > 0 and sideboard:
                last_entry = sideboard[-1]
                qty = last_entry.get("quantity", 1)
                if qty <= excess:
                    excess -= qty
                    sideboard.pop()
                else:
                    last_entry["quantity"] -= excess
                    excess = 0
            print(f"[AI-SERVICE] Trimmed sideboard to 15 cards")

        deck_data["main_deck"] = main_deck
        deck_data["sideboard"] = sideboard
        return deck_data

    async def _generate_fallback_deck(
        self,
        archetype: str,
        colors: List[str],
        strategy: str,
    ) -> Dict[str, Any]:
        """Generate a basic deck structure as fallback."""
        # Get available cards
        available = await self.card_service.search(
            colors=colors,
            standard_only=True,
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
