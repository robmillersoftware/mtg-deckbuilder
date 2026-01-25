from typing import Optional, List, Dict, Any
from uuid import UUID
import logging
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.conversation import Conversation
from app.models.card import Card
from app.schemas.conversation import (
    ChatResponse,
    CardExplanationResponse,
)
from app.services.card_service import CardService
from app.services.deck_generator import DeckGenerator
from app.services.ai_service import AIService
from app.core.config import settings

logger = logging.getLogger(__name__)

# Tool definitions for Claude
TOOLS = [
    {
        "name": "generate_best_meta_deck",
        "description": "Analyze the current metagame and build the best deck for favorable matchups. Use this when the user says things like 'Whatever is best', 'build me the best deck', 'analyze the meta and build something', or doesn't specify what deck they want. This tool autonomously picks the best option based on current tournament data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "optimization_goal": {
                    "type": "string",
                    "enum": ["best_winrate", "beat_top_decks", "underplayed_strong", "balanced"],
                    "description": "What to optimize for. 'best_winrate'=highest performing, 'beat_top_decks'=counter the meta, 'underplayed_strong'=good but under the radar, 'balanced'=solid all-around"
                }
            },
            "required": []
        }
    },
    {
        "name": "generate_deck",
        "description": "Generate a new Magic: The Gathering deck based on SPECIFIC user requirements (colors, archetype). Only use this when the user has specified what colors or archetype they want. If the user is vague or says 'whatever is best', use generate_best_meta_deck instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "colors": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["W", "U", "B", "R", "G"]},
                    "description": "The color(s) for the deck. W=White, U=Blue, B=Black, R=Red, G=Green"
                },
                "archetype": {
                    "type": "string",
                    "enum": ["aggro", "control", "midrange", "combo", "tempo"],
                    "description": "The deck archetype/strategy"
                },
                "strategy": {
                    "type": "string",
                    "description": "Brief description of the deck strategy or theme"
                },
                "specific_cards": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific cards the user wants included"
                }
            },
            "required": ["colors", "archetype"]
        }
    },
    {
        "name": "modify_deck",
        "description": "Modify the current deck in the conversation. Use when user asks to change, add, remove, swap cards, or adjust the deck.",
        "input_schema": {
            "type": "object",
            "properties": {
                "modification": {
                    "type": "string",
                    "description": "Description of the modification to make"
                }
            },
            "required": ["modification"]
        }
    },
    {
        "name": "get_matchup_info",
        "description": "Get matchup and meta information for the current deck. Use when user asks about matchups, how to beat certain decks, sideboard advice, or meta positioning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "opponent_deck": {
                    "type": "string",
                    "description": "The opponent deck archetype to analyze (optional)"
                }
            }
        }
    }
]


class ChatService:
    """
    Chat service for processing user messages and generating responses.
    Uses Claude with tools to intelligently route requests.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.card_service = CardService(db)
        self.ai_service = AIService(db)
        self.deck_generator = DeckGenerator(db)

    async def process_message(
        self,
        message: str,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        format: str = "standard",
    ) -> ChatResponse:
        """
        Process a user message using Claude with tools to determine the appropriate action.
        """
        # Store format for use in handlers
        self._current_format = format

        # Get or create conversation
        conversation = await self._get_or_create_conversation(conversation_id, user_id)

        # Get current deck context if available
        current_deck = conversation.current_deck

        # Add user message to conversation
        conversation.add_message("user", message)

        # Use Claude with tools to determine what to do
        if not settings.ANTHROPIC_API_KEY:
            # Fallback without API
            return await self._fallback_response(message, conversation, user_id)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            # Build context for Claude
            deck_context = ""
            if current_deck:
                main_deck_list = current_deck.get('main_deck', [])
                sideboard_list = current_deck.get('sideboard', [])

                # Format card lists
                main_cards_str = ", ".join(
                    f"{e.get('quantity', 1)}x {e.get('card_name', 'Unknown')}"
                    for e in main_deck_list[:20]  # Limit for context size
                )
                if len(main_deck_list) > 20:
                    main_cards_str += f"... and {len(main_deck_list) - 20} more"

                sideboard_cards_str = ", ".join(
                    f"{e.get('quantity', 1)}x {e.get('card_name', 'Unknown')}"
                    for e in sideboard_list
                )

                deck_context = f"""
Current deck in conversation:
- Name: {current_deck.get('name', 'Unnamed')}
- Archetype: {current_deck.get('archetype', 'Unknown')}
- Main deck ({sum(e.get('quantity', 1) for e in main_deck_list)} cards): {main_cards_str}
- Sideboard ({sum(e.get('quantity', 1) for e in sideboard_list)} cards): {sideboard_cards_str}
"""

            # Format-specific guidance
            format_name = "cEDH" if format == "cedh" else format.capitalize()
            format_guidance = ""
            if format == "cedh":
                format_guidance = """
IMPORTANT - cEDH RULES:
- cEDH decks have exactly 100 cards (99 + commander)
- Singleton format: only 1 copy of each non-basic land card
- Must specify a commander (legendary creature or planeswalker with "can be your commander")
- Color identity: deck can only contain cards matching the commander's color identity
- If user mentions a commander, use that card's color identity for the deck"""
            elif format in ["modern", "legacy"]:
                format_guidance = f"""
Note: Building for {format_name} format - 60 card minimum, 4 copies max of any non-basic land card."""

            system_prompt = f"""You are Spellbook, an expert Magic: The Gathering deck building assistant for {format_name} format.

{deck_context}
{format_guidance}

Your role:
1. If the user wants a deck but is VAGUE about specifics (says "whatever is best", "build me something good", "analyze meta and build", "surprise me", or doesn't specify colors/archetype) - use the generate_best_meta_deck tool IMMEDIATELY. Do NOT ask clarifying questions.
2. If the user EXPLICITLY specifies colors AND/OR archetype (like "red aggro" or "blue-white control") - use the generate_deck tool
3. If the user asks to modify/change/adjust the current deck - use the modify_deck tool
4. If the user asks about matchups, meta, how to beat something, or sideboard advice - use the get_matchup_info tool
5. For general questions about MTG strategy, cards, or rules - respond directly with helpful information

CRITICAL: When a user says "Whatever is best" or similar vague responses, this is NOT a request for more questions - it means "YOU decide for me". Use generate_best_meta_deck immediately.

IMPORTANT: Questions like "How do I beat X?" or "What's good against Y?" are strategy questions - do NOT generate a deck for these. Either use get_matchup_info if there's a current deck, or just provide strategy advice.

Be decisive and action-oriented. Focus on competitive {format_name} play."""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=system_prompt,
                tools=TOOLS,
                messages=[{"role": "user", "content": message}],
            )

            # Process response
            if response.stop_reason == "tool_use":
                # Claude wants to use a tool
                for content in response.content:
                    if content.type == "tool_use":
                        tool_name = content.name
                        tool_input = content.input

                        print(f"[CHAT-SERVICE] Claude called tool: {tool_name} with input: {tool_input}")

                        if tool_name == "generate_best_meta_deck":
                            return await self._handle_meta_deck_generation(
                                tool_input, conversation, user_id
                            )
                        elif tool_name == "generate_deck":
                            return await self._handle_deck_generation(
                                tool_input, conversation, user_id
                            )
                        elif tool_name == "modify_deck":
                            return await self._handle_deck_modification(
                                tool_input, conversation, user_id
                            )
                        elif tool_name == "get_matchup_info":
                            return await self._handle_matchup_query(
                                tool_input, conversation
                            )

            # Claude responded with text (no tool use)
            response_text = ""
            for content in response.content:
                if hasattr(content, "text"):
                    response_text += content.text

            if response_text:
                conversation.add_message("assistant", response_text)
                await self.db.commit()

                return ChatResponse(
                    response=response_text,
                    conversation_id=conversation.id,
                    suggestions=self._get_suggestions(current_deck),
                )

        except Exception as e:
            logger.error(f"Chat processing error: {e}", exc_info=True)

        # Fallback
        return await self._fallback_response(message, conversation, user_id)

    async def _get_or_create_conversation(
        self, conversation_id: Optional[UUID], user_id: Optional[UUID]
    ) -> Conversation:
        """Get existing conversation or create new one."""
        if conversation_id:
            result = await self.db.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conversation = result.scalar_one_or_none()
            if conversation:
                return conversation

        conversation = Conversation(user_id=user_id, messages=[])
        self.db.add(conversation)
        await self.db.flush()
        return conversation

    async def _handle_meta_deck_generation(
        self,
        tool_input: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[UUID],
    ) -> ChatResponse:
        """Handle autonomous meta-based deck generation."""
        from app.models.meta import MetaSnapshot

        optimization_goal = tool_input.get("optimization_goal", "balanced")
        format = getattr(self, "_current_format", "standard")

        # Get current meta data
        result = await self.db.execute(
            select(MetaSnapshot)
            .where(MetaSnapshot.format == format)
            .order_by(MetaSnapshot.meta_percentage.desc())
            .limit(10)
        )
        snapshots = result.scalars().all()

        # For cEDH, we need to track the commander name separately
        # since the "archetype" in meta is actually the commander name
        commander_name = None
        chosen_archetype_name = None  # The full name from meta (e.g., "Kinnan, Bonder Prodigy")

        # Analyze meta and pick best deck to build
        if not snapshots:
            # No meta data - build a generally good deck
            archetype = "midrange"
            colors = ["R", "G"]
            reasoning = "Building Gruul Midrange as a solid all-around choice."
        else:
            # Pick based on optimization goal
            top_decks = [(s.archetype, float(s.meta_percentage or 0)) for s in snapshots]

            if optimization_goal == "best_winrate":
                # Build the top performing deck
                best = snapshots[0]
                chosen_archetype_name = best.archetype
                archetype = self._classify_deck_type(best.archetype or "midrange")
                colors = self._extract_colors_from_archetype(best.archetype or "")
                reasoning = f"Building {best.archetype} - currently the #1 deck in the meta at {float(best.meta_percentage or 0):.1f}% of the field."

            elif optimization_goal == "beat_top_decks":
                # Find what beats the top decks
                top_type = self._classify_deck_type(snapshots[0].archetype or "")
                if top_type == "aggro":
                    archetype = "midrange"
                    colors = ["B", "G"]
                    reasoning = f"The meta is aggro-heavy ({snapshots[0].archetype} at top). Building Golgari Midrange to prey on aggressive decks with removal and lifegain."
                elif top_type == "control":
                    archetype = "aggro"
                    colors = ["R"]
                    reasoning = f"The meta is control-heavy ({snapshots[0].archetype} at top). Building Mono-Red Aggro to go under them before they stabilize."
                elif top_type == "midrange":
                    archetype = "control"
                    colors = ["W", "U"]
                    reasoning = f"The meta is midrange-heavy ({snapshots[0].archetype} at top). Building Azorius Control to go over them with card advantage and sweepers."
                else:
                    archetype = "midrange"
                    colors = ["R", "G"]
                    reasoning = "Building Gruul Midrange as a flexible counter to the field."

            elif optimization_goal == "underplayed_strong":
                # Find a deck that's good but under-represented
                for snap in snapshots[3:8]:  # Look at positions 4-8
                    if snap.meta_percentage and float(snap.meta_percentage) > 3:
                        chosen_archetype_name = snap.archetype
                        archetype = self._classify_deck_type(snap.archetype or "midrange")
                        colors = self._extract_colors_from_archetype(snap.archetype or "")
                        reasoning = f"Building {snap.archetype} - solid performance but less expected at {float(snap.meta_percentage):.1f}% meta share."
                        break
                else:
                    archetype = "tempo"
                    colors = ["U", "R"]
                    reasoning = "Building Izzet Tempo - a strong but underrepresented strategy."

            else:  # balanced
                # Pick something solid with good matchup spread
                archetype = "midrange"
                # Look for a midrange deck in the meta
                for snap in snapshots[:5]:
                    if "midrange" in (snap.archetype or "").lower():
                        chosen_archetype_name = snap.archetype
                        colors = self._extract_colors_from_archetype(snap.archetype or "")
                        reasoning = f"Building {snap.archetype} - balanced matchups across the field."
                        break
                else:
                    colors = ["B", "G"]
                    reasoning = "Building Golgari Midrange - strong, flexible, and good against most of the field."

        # For cEDH, the archetype name IS the commander name
        # We need to pass it as specific_cards and look up the actual color identity
        specific_cards = None
        if format == "cedh" and chosen_archetype_name:
            commander_name = chosen_archetype_name
            specific_cards = [commander_name]
            # Look up the commander's actual color identity
            commander_colors = await self.ai_service.get_commander_color_identity(commander_name)
            if commander_colors:
                colors = commander_colors
                logger.info(f"[CHAT-SERVICE] Using commander {commander_name} color identity: {colors}")

        # Build the prompt
        color_names = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
        color_str = "/".join(color_names.get(c, c) for c in colors)
        format_display = "cEDH" if format == "cedh" else format.capitalize()

        if format == "cedh" and commander_name:
            prompt = f"Build a competitive {commander_name} cEDH deck"
        else:
            prompt = f"Build a competitive {color_str} {archetype} deck optimized for the current {format_display} meta"

        logger.info(f"[CHAT-SERVICE] _handle_meta_deck_generation: format={format}, colors={colors}, commander={commander_name}")

        result = await self.deck_generator.generate(
            prompt=prompt,
            user_id=user_id,
            conversation_id=conversation.id,
            include_sideboard=(format != "cedh"),  # cEDH doesn't use sideboard
            format=format,
            colors=colors if colors else None,
            specific_cards=specific_cards,  # Pass commander name for cEDH
        )

        meta_summary = "\n\n**Current Meta:**\n"
        for snap in snapshots[:5]:
            pct = f"{float(snap.meta_percentage):.1f}%" if snap.meta_percentage else "?"
            meta_summary += f"- {snap.archetype}: {pct}\n"

        return ChatResponse(
            response=f"**{reasoning}**\n\n{result.strategy_summary or ''}{meta_summary}",
            conversation_id=result.conversation_id,
            deck={
                "name": result.deck.name,
                "format": result.deck.format,
                "commander": result.deck.commander,
                "main_deck": result.deck.main_deck,
                "sideboard": result.deck.sideboard,
                "archetype": result.deck.archetype,
            },
            suggestions=[
                "Show matchup analysis",
                "Build something to beat this",
                "Make it more aggressive",
            ],
        )

    def _extract_colors_from_archetype(self, archetype: str) -> List[str]:
        """Extract likely colors from an archetype name."""
        arch_lower = archetype.lower()

        # Check for explicit color mentions
        color_map = {
            "white": "W", "azorius": "WU", "orzhov": "WB", "boros": "WR", "selesnya": "WG",
            "blue": "U", "dimir": "UB", "izzet": "UR", "simic": "UG",
            "black": "B", "rakdos": "BR", "golgari": "BG",
            "red": "R", "gruul": "RG",
            "green": "G",
            "esper": "WUB", "grixis": "UBR", "jund": "BRG", "naya": "WRG", "bant": "WUG",
            "abzan": "WBG", "jeskai": "WUR", "sultai": "UBG", "mardu": "WBR", "temur": "URG",
            "mono-red": "R", "mono-white": "W", "mono-blue": "U", "mono-black": "B", "mono-green": "G",
        }

        for key, colors in color_map.items():
            if key in arch_lower:
                return list(colors)

        # Default based on archetype type
        if "aggro" in arch_lower or "burn" in arch_lower or "rdw" in arch_lower:
            return ["R"]
        elif "control" in arch_lower:
            return ["W", "U"]
        elif "ramp" in arch_lower:
            return ["G"]

        return ["R", "G"]  # Default to Gruul

    async def _handle_deck_generation(
        self,
        tool_input: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[UUID],
    ) -> ChatResponse:
        """Handle deck generation from tool call."""
        colors = tool_input.get("colors", [])
        archetype = tool_input.get("archetype", "midrange")
        strategy = tool_input.get("strategy", "")
        specific_cards = tool_input.get("specific_cards", [])
        format = getattr(self, "_current_format", "standard")
        format_display = "cEDH" if format == "cedh" else format.capitalize()

        logger.info(f"[CHAT-SERVICE] _handle_deck_generation: format={format}, colors={colors}, specific_cards={specific_cards}")

        # Build a prompt from the parsed data
        prompt = f"Build a {' '.join(colors) if colors else ''} {archetype} deck"
        if strategy:
            prompt += f" focused on {strategy}"
        if specific_cards:
            prompt += f" including {', '.join(specific_cards)}"

        result = await self.deck_generator.generate(
            prompt=prompt,
            user_id=user_id,
            conversation_id=conversation.id,
            include_sideboard=(format != "cedh"),  # cEDH doesn't use sideboard
            format=format,
            colors=colors if colors else None,  # Pass explicit colors to avoid re-parsing
            specific_cards=specific_cards if specific_cards else None,  # Pass explicit specific_cards
        )

        return ChatResponse(
            response=f"I've built a {result.deck.archetype or format_display} deck for you!\n\n"
            + (result.strategy_summary or ""),
            conversation_id=result.conversation_id,
            deck={
                "name": result.deck.name,
                "format": result.deck.format,
                "commander": result.deck.commander,
                "main_deck": result.deck.main_deck,
                "sideboard": result.deck.sideboard,
                "archetype": result.deck.archetype,
            },
            suggestions=[
                "Explain the sideboard" if format != "cedh" else "Explain key cards",
                "Show matchups",
                "Make it faster",
            ],
        )

    async def _handle_deck_modification(
        self,
        tool_input: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[UUID],
    ) -> ChatResponse:
        """Handle deck modification from tool call."""
        modification = tool_input.get("modification", "")

        if not conversation.current_deck:
            return ChatResponse(
                response="I don't have a deck to modify. Would you like me to build one first?",
                conversation_id=conversation.id,
                suggestions=["Build me a deck", "Import my deck"],
            )

        result = await self.deck_generator.iterate(
            modification=modification,
            conversation_id=conversation.id,
            user_id=user_id,
        )

        return ChatResponse(
            response=result.summary,
            conversation_id=conversation.id,
            deck={
                "name": result.deck.name,
                "format": result.deck.format,
                "commander": result.deck.commander,
                "main_deck": result.deck.main_deck,
                "sideboard": result.deck.sideboard,
                "archetype": result.deck.archetype,
            },
            suggestions=["Undo changes", "Show the full list", "Explain changes"],
        )

    async def _handle_matchup_query(
        self,
        tool_input: Dict[str, Any],
        conversation: Conversation,
    ) -> ChatResponse:
        """Handle matchup/meta query from tool call."""
        opponent_deck = tool_input.get("opponent_deck", "")

        if not conversation.current_deck:
            # No deck context - give general advice
            response = await self._get_general_meta_advice(opponent_deck)
        else:
            response = await self._get_matchup_analysis(
                conversation.current_deck, opponent_deck
            )

        conversation.add_message("assistant", response)
        await self.db.commit()

        return ChatResponse(
            response=response,
            conversation_id=conversation.id,
            suggestions=["Show sideboard guide", "Build a counter deck"],
        )

    async def _get_general_meta_advice(self, opponent_deck: str) -> str:
        """Get general meta advice when no deck context is available."""
        from app.models.meta import MetaSnapshot

        # Get current meta
        result = await self.db.execute(
            select(MetaSnapshot)
            .where(MetaSnapshot.format == "standard")
            .order_by(MetaSnapshot.meta_percentage.desc())
            .limit(10)
        )
        snapshots = result.scalars().all()

        response = "**Current Standard Meta:**\n\n"

        if snapshots:
            for snap in snapshots:
                pct = f"{float(snap.meta_percentage):.1f}%" if snap.meta_percentage else "?"
                response += f"- **{snap.archetype}** ({pct})\n"
            response += "\n"

        if opponent_deck:
            # Give strategy advice for the mentioned deck
            response += f"**Tips against {opponent_deck}:**\n\n"

            opp_lower = opponent_deck.lower()
            if "aggro" in opp_lower or "red" in opp_lower:
                response += "- Play efficient removal for early threats\n"
                response += "- Include lifegain to stabilize\n"
                response += "- Board wipes like Wrath effects are very effective\n"
            elif "control" in opp_lower:
                response += "- Apply early pressure before they stabilize\n"
                response += "- Include resilient threats that dodge single-target removal\n"
                response += "- Save key spells for when they tap out\n"
            elif "midrange" in opp_lower:
                response += "- Go under them with aggro or over them with control\n"
                response += "- Card advantage is key in grindy games\n"
                response += "- Planeswalkers can break parity\n"
            else:
                response += "- Study their key cards and plan interaction accordingly\n"
                response += "- Sideboard cards specific to their strategy\n"

        response += "\nWould you like me to build a deck to counter a specific archetype?"

        return response

    async def _get_matchup_analysis(
        self, deck: Dict[str, Any], opponent_deck: str
    ) -> str:
        """Get matchup analysis for a specific deck."""
        from app.models.meta import MetaSnapshot

        deck_archetype = deck.get("archetype", "").lower()
        our_type = self._classify_deck_type(deck_archetype)

        # Get meta decks
        result = await self.db.execute(
            select(MetaSnapshot)
            .where(MetaSnapshot.format == "standard")
            .order_by(MetaSnapshot.meta_percentage.desc())
            .limit(15)
        )
        snapshots = result.scalars().all()

        # Classify matchups
        favorable = []
        even = []
        challenging = []

        for snap in snapshots:
            if not snap.archetype:
                continue
            meta_type = self._classify_deck_type(snap.archetype.lower())

            if snap.archetype.lower() == deck_archetype:
                even.append(f"{snap.archetype} (mirror)")
                continue

            # Rock-paper-scissors matchup logic
            if our_type == "aggro":
                if meta_type in ["control", "ramp"]:
                    favorable.append(snap.archetype)
                elif meta_type == "midrange":
                    challenging.append(snap.archetype)
                else:
                    even.append(snap.archetype)
            elif our_type == "control":
                if meta_type == "midrange":
                    favorable.append(snap.archetype)
                elif meta_type == "aggro":
                    challenging.append(snap.archetype)
                else:
                    even.append(snap.archetype)
            elif our_type == "midrange":
                if meta_type == "aggro":
                    favorable.append(snap.archetype)
                elif meta_type == "control":
                    challenging.append(snap.archetype)
                else:
                    even.append(snap.archetype)
            else:
                even.append(snap.archetype)

        response = f"**Matchup Analysis for {deck.get('name', 'your deck')}**\n\n"

        if favorable:
            response += "**Favorable:**\n"
            for d in favorable[:4]:
                response += f"- {d}\n"
            response += "\n"

        if even:
            response += "**Even:**\n"
            for d in even[:3]:
                response += f"- {d}\n"
            response += "\n"

        if challenging:
            response += "**Challenging:**\n"
            for d in challenging[:4]:
                response += f"- {d}\n"
            response += "\n"

        # Specific opponent advice
        if opponent_deck:
            opp_type = self._classify_deck_type(opponent_deck.lower())
            response += f"\n**Specific tips vs {opponent_deck}:**\n"

            if our_type == "aggro" and opp_type == "midrange":
                response += "- Focus on reach/burn to close games\n"
                response += "- Don't overextend into their removal\n"
            elif our_type == "control" and opp_type == "aggro":
                response += "- Prioritize early removal and lifegain\n"
                response += "- Sweepers are your best friend\n"
            elif our_type == "midrange" and opp_type == "control":
                response += "- Deploy threats carefully, bait counters\n"
                response += "- Card advantage helps in long games\n"

        return response

    def _classify_deck_type(self, archetype: str) -> str:
        """Classify a deck archetype into a general category."""
        archetype = archetype.lower()

        if any(kw in archetype for kw in ["aggro", "burn", "red deck", "rdw", "mono-red", "sligh", "goblins"]):
            return "aggro"
        elif any(kw in archetype for kw in ["control", "uw", "azorius", "esper", "dimir"]):
            return "control"
        elif any(kw in archetype for kw in ["ramp", "stompy", "mono-green"]):
            return "ramp"
        elif any(kw in archetype for kw in ["combo", "storm"]):
            return "combo"
        elif any(kw in archetype for kw in ["midrange", "jund", "abzan", "gruul"]):
            return "midrange"
        return "unknown"

    def _get_suggestions(self, deck: Optional[Dict[str, Any]]) -> List[str]:
        """Get contextual suggestions based on current state."""
        if deck:
            return ["Show matchups", "Make changes", "Export deck"]
        return ["Build me a deck", "What's the current meta?", "Help"]

    async def _fallback_response(
        self,
        message: str,
        conversation: Conversation,
        user_id: Optional[UUID],
    ) -> ChatResponse:
        """Fallback response when AI is not available."""
        response = (
            "I'm Spellbook, your MTG deck building assistant! I can help you:\n\n"
            "- **Build decks**: 'Build me a mono-red aggro deck'\n"
            "- **Modify decks**: 'Make it faster' or 'Add blue for counterspells'\n"
            "- **Analyze matchups**: 'How do I beat control?'\n\n"
            "What would you like to do?"
        )

        conversation.add_message("assistant", response)
        await self.db.commit()

        return ChatResponse(
            response=response,
            conversation_id=conversation.id,
            suggestions=["Build me a competitive Standard deck", "What's the current meta?"],
        )

    async def explain_card(
        self,
        card_name: str,
        deck: Optional[Dict[str, Any]] = None,
    ) -> CardExplanationResponse:
        """Explain a card's role in a deck."""
        card = await self.card_service.get_by_name(card_name)

        if not card:
            similar = await self.card_service.fuzzy_search_by_name(card_name, limit=1)
            if similar:
                card = similar[0]

        if not card:
            return CardExplanationResponse(
                card_name=card_name,
                role="Unknown",
                explanation=f"Card '{card_name}' was not found.",
                synergies=[],
                alternatives=[],
            )

        role = "Spell"
        if "creature" in (card.type_line or "").lower():
            role = "Creature"
        elif "land" in (card.type_line or "").lower():
            role = "Land"
        elif "planeswalker" in (card.type_line or "").lower():
            role = "Planeswalker"

        explanation = f"{card.name} is a {card.type_line}."
        if card.oracle_text:
            explanation += f"\n\n{card.oracle_text}"

        return CardExplanationResponse(
            card_name=card.name,
            role=role,
            explanation=explanation,
            synergies=[],
            alternatives=[],
        )
