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
        "name": "generate_deck",
        "description": "Generate a new Magic: The Gathering deck based on user requirements. Use this when the user explicitly asks to BUILD, CREATE, or MAKE a deck. Do NOT use this for questions about strategy, matchups, or how to beat something.",
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
    ) -> ChatResponse:
        """
        Process a user message using Claude with tools to determine the appropriate action.
        """
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

            system_prompt = f"""You are Spellbook, an expert Magic: The Gathering deck building assistant for Standard format.

{deck_context}

Your role:
1. If the user EXPLICITLY asks to build/create/make a deck, use the generate_deck tool
2. If the user asks to modify/change/adjust the current deck, use the modify_deck tool
3. If the user asks about matchups, meta, how to beat something, or sideboard advice, use the get_matchup_info tool
4. For general questions about MTG strategy, cards, or rules - just respond directly with helpful information

IMPORTANT: Questions like "How do I beat X?" or "What's good against Y?" are strategy questions - do NOT generate a deck for these. Either use get_matchup_info if there's a current deck, or just provide strategy advice.

Be concise and helpful. Focus on competitive Standard play."""

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

                        if tool_name == "generate_deck":
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
            include_sideboard=True,
        )

        return ChatResponse(
            response=f"I've built a {result.deck.archetype or 'Standard'} deck for you!\n\n"
            + (result.strategy_summary or ""),
            conversation_id=result.conversation_id,
            deck={
                "name": result.deck.name,
                "main_deck": result.deck.main_deck,
                "sideboard": result.deck.sideboard,
                "archetype": result.deck.archetype,
            },
            suggestions=[
                "Explain the sideboard",
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
                "main_deck": result.deck.main_deck,
                "sideboard": result.deck.sideboard,
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
