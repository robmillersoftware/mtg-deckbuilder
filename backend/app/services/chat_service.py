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
from app.services.guided_builder import DeckAnalyzer
from app.core.config import settings

logger = logging.getLogger(__name__)

# Tool definitions for Claude - incremental collaborative builder
TOOLS = [
    {
        "name": "analyze_meta",
        "description": "Analyze the current metagame and recommend archetypes/directions. Use when the user is exploring options, asks 'what's good?', or wants meta analysis before committing to a build. Do NOT generate a deck - just provide analysis and recommendations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "Optional focus area: 'aggro', 'control', 'midrange', 'combo', or empty for general overview"
                }
            },
            "required": []
        }
    },
    {
        "name": "suggest_core",
        "description": "Suggest the core cards for a deck in role-based groups. Use when the user has decided on a direction (named a card, chose colors, picked an archetype) and you're starting the build. Returns 3-5 groups of 4-8 cards each for the user to review and add. For cEDH, use larger groups (10-15 cards, ~5 groups).",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "description": "The deck strategy/theme (e.g. 'red aggro', 'blue-white control', 'Atraxa superfriends')"
                },
                "colors": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["W", "U", "B", "R", "G"]},
                    "description": "The color(s) for the deck"
                },
                "roles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Role groups to suggest cards for (e.g. ['threats', 'removal', 'card advantage', 'protection'])"
                }
            },
            "required": ["strategy", "colors", "roles"]
        }
    },
    {
        "name": "suggest_package",
        "description": "Suggest a focused package of cards for a specific role/gap in an existing deck. Use when the user asks for cards to fill a specific need like 'what removal should I play?' or 'I need more card draw'. Returns 1 group.",
        "input_schema": {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": "The role to fill (e.g. 'removal', 'card draw', 'finishers', 'counterspells', 'ramp')"
                },
                "strategy": {
                    "type": "string",
                    "description": "Context about the deck strategy for better suggestions"
                },
                "colors": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["W", "U", "B", "R", "G"]},
                    "description": "The deck's colors"
                },
                "count": {
                    "type": "integer",
                    "description": "How many cards to suggest (default 6)"
                }
            },
            "required": ["role", "colors"]
        }
    },
    {
        "name": "finalize_mana_base",
        "description": "Generate a complete mana base for the current deck. Use when the nonland card count is near the target (e.g. ~36 nonland cards for a 60-card deck, ~65 for cEDH). Returns a batch of lands the user can add all at once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "colors": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["W", "U", "B", "R", "G"]},
                    "description": "The deck's colors"
                }
            },
            "required": ["colors"]
        }
    },
    {
        "name": "modify_deck",
        "description": "Modify the current deck in the conversation. Use when user asks to swap, cut, or adjust specific cards.",
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
    },
    {
        "name": "generate_full_deck",
        "description": "Generate a complete deck in one shot. ONLY use this when the user EXPLICITLY asks to skip the collaborative process, e.g. 'just build it', 'skip suggestions', 'give me the whole deck'. Never use this by default.",
        "input_schema": {
            "type": "object",
            "properties": {
                "colors": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["W", "U", "B", "R", "G"]},
                    "description": "The color(s) for the deck"
                },
                "archetype": {
                    "type": "string",
                    "enum": ["aggro", "control", "midrange", "combo", "tempo"],
                    "description": "The deck archetype"
                },
                "strategy": {
                    "type": "string",
                    "description": "Brief description of the deck strategy"
                },
                "specific_cards": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific cards to include"
                }
            },
            "required": []
        }
    },
]


class ChatService:
    """
    Chat service for processing user messages and generating responses.
    Uses Claude with tools for incremental collaborative deck building.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.card_service = CardService(db)
        self.ai_service = AIService(db)
        self.deck_generator = DeckGenerator(db)
        self.deck_analyzer = DeckAnalyzer(db)

    async def process_message(
        self,
        message: str,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        format: str = "standard",
        current_deck: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """
        Process a user message using Claude with tools for incremental deck building.
        """
        self._current_format = format

        # Get or create conversation
        conversation = await self._get_or_create_conversation(conversation_id, user_id)

        # Sync local deck state from frontend if provided
        if current_deck is not None:
            conversation.current_deck = current_deck
            await self.db.flush()

        # Get current deck context
        deck = conversation.current_deck

        # Add user message to conversation
        conversation.add_message("user", message)

        if not settings.ANTHROPIC_API_KEY:
            return await self._fallback_response(message, conversation, user_id)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            # Build deck context string
            deck_context = self._build_deck_context(deck)

            # Format-specific guidance
            format_name = "cEDH" if format == "cedh" else format.capitalize()
            format_guidance = self._get_format_guidance(format, format_name)

            system_prompt = f"""You are Spellbook, a collaborative deck building partner for {format_name} format.
You work WITH the user, not FOR them. Suggest cards, explain choices, let them decide.

{deck_context}
{format_guidance}

FLOW:
1. EXPLORE: Help settle on a direction. If vague, use analyze_meta. If they name a card/colors, move to step 2.
2. BUILD CORE: Use suggest_core to propose key cards in role-based groups (e.g. threats, removal, card advantage, protection).
3. FILL GAPS: After user adds cards, use suggest_package for missing roles.
4. LANDS: When nonland count is near target, use finalize_mana_base.
5. REFINE: Help with cuts, sideboard, matchups using modify_deck or get_matchup_info.

RULES:
- NEVER use generate_full_deck unless user explicitly asks to skip suggestions (e.g. "just build it", "skip suggestions").
- When using suggest_core, pick 3-5 meaningful role names for the groups.
- For cEDH: suggest in larger batches (10-15 per group, ~5 groups).
- For questions about matchups, strategy, or "how do I beat X" - use get_matchup_info or respond with text advice. Do NOT generate cards.
- When the user says "what's good" or similar vague exploration, use analyze_meta.
- Be concise in your text responses. Focus on actionable advice.
- Always explain your reasoning briefly when suggesting a direction."""

            # Build conversation history - send last 8 messages for multi-turn context
            recent = conversation.messages[-8:] if conversation.messages else []
            api_messages = []
            for m in recent:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role in ("user", "assistant") and content:
                    api_messages.append({"role": role, "content": content})

            # Ensure we end with the current user message and messages alternate properly
            if not api_messages or api_messages[-1]["role"] != "user":
                api_messages.append({"role": "user", "content": message})

            # Fix: ensure messages alternate (Claude API requirement)
            api_messages = self._fix_message_alternation(api_messages)

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=system_prompt,
                tools=TOOLS,
                messages=api_messages,
            )

            # Capture text content alongside tool use
            response_text = ""
            for content in response.content:
                if hasattr(content, "text"):
                    response_text += content.text

            # Process tool calls
            if response.stop_reason == "tool_use":
                for content in response.content:
                    if content.type == "tool_use":
                        tool_name = content.name
                        tool_input = content.input

                        logger.debug(f"[CHAT-SERVICE] Claude called tool: {tool_name} with input: {tool_input}")

                        result = await self._dispatch_tool(
                            tool_name, tool_input, conversation, user_id, response_text
                        )
                        if result:
                            return result

            # Claude responded with text only (no tool use)
            if response_text:
                conversation.add_message("assistant", response_text)
                await self.db.commit()

                return ChatResponse(
                    response=response_text,
                    conversation_id=conversation.id,
                    suggestions=self._get_suggestions(deck),
                )

        except Exception as e:
            logger.error(f"Chat processing error: {e}", exc_info=True)

        return await self._fallback_response(message, conversation, user_id)

    def _build_deck_context(self, deck: Optional[Dict[str, Any]]) -> str:
        """Build deck context string for the system prompt."""
        if not deck:
            return "No deck started yet."

        main_deck_list = deck.get('main_deck', [])
        sideboard_list = deck.get('sideboard', [])

        main_cards_str = ", ".join(
            f"{e.get('quantity', 1)}x {e.get('card_name', 'Unknown')}"
            for e in main_deck_list[:30]
        )
        if len(main_deck_list) > 30:
            main_cards_str += f"... and {len(main_deck_list) - 30} more"

        sideboard_cards_str = ", ".join(
            f"{e.get('quantity', 1)}x {e.get('card_name', 'Unknown')}"
            for e in sideboard_list
        )

        nonland_count = 0
        land_count = 0
        for e in main_deck_list:
            card_data = e.get("card", {}) or {}
            type_line = (card_data.get("type_line") or "").lower()
            qty = e.get("quantity", 0)
            if "land" in type_line:
                land_count += qty
            else:
                nonland_count += qty

        return f"""Current deck in conversation:
- Name: {deck.get('name', 'Unnamed')}
- Archetype: {deck.get('archetype', 'Unknown')}
- Main deck ({sum(e.get('quantity', 1) for e in main_deck_list)} cards, {nonland_count} nonland, {land_count} lands): {main_cards_str}
- Sideboard ({sum(e.get('quantity', 1) for e in sideboard_list)} cards): {sideboard_cards_str}"""

    def _get_format_guidance(self, format: str, format_name: str) -> str:
        """Get format-specific guidance for the system prompt."""
        if format == "cedh":
            return """IMPORTANT - cEDH RULES:
- cEDH decks have exactly 100 cards (99 + commander)
- Singleton format: only 1 copy of each non-basic land card
- Must specify a commander (legendary creature)
- Color identity: deck can only contain cards matching the commander's color identity
- Suggest cards in larger batches (~10-15 per group, 5 groups)
- Target ~34 lands, ~65 nonland cards"""
        elif format in ["modern", "legacy"]:
            return f"Building for {format_name} format - 60 card minimum, 4 copies max of any non-basic land card."
        return "Building for Standard format - 60 card minimum, 4 copies max of any non-basic land card."

    def _fix_message_alternation(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Ensure messages alternate user/assistant as required by Claude API."""
        if not messages:
            return messages

        fixed = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == fixed[-1]["role"]:
                # Merge consecutive same-role messages
                fixed[-1]["content"] += "\n\n" + msg["content"]
            else:
                fixed.append(msg)

        # Must start with user message
        if fixed[0]["role"] != "user":
            fixed.insert(0, {"role": "user", "content": "(continuing conversation)"})

        return fixed

    async def _dispatch_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[UUID],
        ai_text: str = "",
    ) -> Optional[ChatResponse]:
        """Dispatch a tool call to the appropriate handler."""
        handlers = {
            "analyze_meta": self._handle_analyze_meta,
            "suggest_core": self._handle_suggest_core,
            "suggest_package": self._handle_suggest_package,
            "finalize_mana_base": self._handle_finalize_mana_base,
            "modify_deck": self._handle_deck_modification,
            "get_matchup_info": self._handle_matchup_query,
            "generate_full_deck": self._handle_generate_full_deck,
        }

        handler = handlers.get(tool_name)
        if not handler:
            logger.warning(f"Unknown tool: {tool_name}")
            return None

        # Handlers that need user_id
        if tool_name in ("modify_deck", "generate_full_deck"):
            return await handler(tool_input, conversation, user_id, ai_text)
        elif tool_name in ("suggest_core", "suggest_package", "finalize_mana_base", "analyze_meta"):
            return await handler(tool_input, conversation, ai_text)
        else:
            return await handler(tool_input, conversation, ai_text)

    async def _handle_analyze_meta(
        self,
        tool_input: Dict[str, Any],
        conversation: Conversation,
        ai_text: str = "",
    ) -> ChatResponse:
        """Analyze the current metagame and recommend directions."""
        from app.models.meta import MetaSnapshot

        format = getattr(self, "_current_format", "standard")
        focus = tool_input.get("focus", "")

        result = await self.db.execute(
            select(MetaSnapshot)
            .where(MetaSnapshot.format == format)
            .order_by(MetaSnapshot.meta_percentage.desc())
            .limit(10)
        )
        snapshots = result.scalars().all()

        format_name = "cEDH" if format == "cedh" else format.capitalize()
        response = ai_text + "\n\n" if ai_text else ""
        response += f"**Current {format_name} Meta:**\n\n"

        if snapshots:
            for snap in snapshots:
                pct = f"{float(snap.meta_percentage):.1f}%" if snap.meta_percentage else "?"
                response += f"- **{snap.archetype}** ({pct})\n"
        else:
            response += "No meta data available yet.\n"

        response += "\nWhat direction interests you? Name a card, pick colors, or choose an archetype and I'll start suggesting cards."

        conversation.add_message("assistant", response)
        await self.db.commit()

        return ChatResponse(
            response=response,
            conversation_id=conversation.id,
            suggestions=[
                "Build around the top deck",
                "I want to play aggro",
                "What beats the top decks?",
            ],
        )

    async def _handle_suggest_core(
        self,
        tool_input: Dict[str, Any],
        conversation: Conversation,
        ai_text: str = "",
    ) -> ChatResponse:
        """Suggest core cards in role-based groups."""
        format = getattr(self, "_current_format", "standard")
        strategy = tool_input.get("strategy", "")
        colors = tool_input.get("colors", [])
        roles = tool_input.get("roles", ["threats", "removal", "card advantage"])

        # Get existing card names from deck
        deck = conversation.current_deck or {}
        existing = [e.get("card_name", "") for e in deck.get("main_deck", [])]

        cards_per_role = 12 if format == "cedh" else 6

        # Query cards grouped by role
        role_cards = await self.deck_analyzer.suggest_cards_for_strategy(
            strategy=strategy,
            colors=colors,
            roles=roles,
            existing_cards=existing,
            format=format,
            cards_per_role=cards_per_role,
        )

        # Build card suggestion groups
        card_suggestions = []
        for role, cards in role_cards.items():
            group = {
                "group_name": role.replace("_", " ").title(),
                "role": role,
                "is_batch": False,
                "cards": [
                    {
                        "card_name": c["card_name"],
                        "quantity": 1 if format == "cedh" else min(4, 3),
                        "mana_cost": c.get("mana_cost"),
                        "type_line": c.get("type_line"),
                        "image_uri": c.get("image_uri"),
                        "reasoning": None,  # Could be enriched with AI reasoning in future
                    }
                    for c in cards
                ],
            }
            card_suggestions.append(group)

        response = ai_text + "\n\n" if ai_text else ""
        response += f"Here are my suggestions for your **{strategy}** deck. Review each group and add the cards you like!"

        conversation.add_message("assistant", response)
        await self.db.commit()

        return ChatResponse(
            response=response,
            conversation_id=conversation.id,
            card_suggestions=card_suggestions if card_suggestions else None,
            suggestions=[
                "I need more removal",
                "What about card draw?",
                "Let's add the mana base",
            ],
        )

    async def _handle_suggest_package(
        self,
        tool_input: Dict[str, Any],
        conversation: Conversation,
        ai_text: str = "",
    ) -> ChatResponse:
        """Suggest a focused package for a specific role."""
        format = getattr(self, "_current_format", "standard")
        role = tool_input.get("role", "removal")
        strategy = tool_input.get("strategy", "")
        colors = tool_input.get("colors", [])
        count = tool_input.get("count", 6)

        deck = conversation.current_deck or {}
        existing = [e.get("card_name", "") for e in deck.get("main_deck", [])]

        role_cards = await self.deck_analyzer.suggest_cards_for_strategy(
            strategy=strategy or role,
            colors=colors,
            roles=[role],
            existing_cards=existing,
            format=format,
            cards_per_role=count,
        )

        card_suggestions = []
        cards = role_cards.get(role, [])
        if cards:
            group = {
                "group_name": role.replace("_", " ").title(),
                "role": role,
                "is_batch": False,
                "cards": [
                    {
                        "card_name": c["card_name"],
                        "quantity": 1 if format == "cedh" else min(4, 3),
                        "mana_cost": c.get("mana_cost"),
                        "type_line": c.get("type_line"),
                        "image_uri": c.get("image_uri"),
                        "reasoning": None,
                    }
                    for c in cards
                ],
            }
            card_suggestions.append(group)

        response = ai_text + "\n\n" if ai_text else ""
        response += f"Here are some **{role}** options for your deck:"

        conversation.add_message("assistant", response)
        await self.db.commit()

        return ChatResponse(
            response=response,
            conversation_id=conversation.id,
            card_suggestions=card_suggestions if card_suggestions else None,
            suggestions=[
                "Suggest more options",
                "Let's add the mana base",
                "What else am I missing?",
            ],
        )

    async def _handle_finalize_mana_base(
        self,
        tool_input: Dict[str, Any],
        conversation: Conversation,
        ai_text: str = "",
    ) -> ChatResponse:
        """Generate mana base for the current deck."""
        format = getattr(self, "_current_format", "standard")
        colors = tool_input.get("colors", [])

        deck = conversation.current_deck or {}
        main_deck = deck.get("main_deck", [])

        lands = await self.deck_analyzer.compute_mana_base(
            main_deck=main_deck,
            colors=colors,
            format=format,
        )

        card_suggestions = []
        if lands:
            group = {
                "group_name": "Mana Base",
                "role": "lands",
                "is_batch": True,
                "cards": [
                    {
                        "card_name": l["card_name"],
                        "quantity": l["quantity"],
                        "mana_cost": l.get("mana_cost"),
                        "type_line": l.get("type_line"),
                        "image_uri": l.get("image_uri"),
                        "reasoning": None,
                    }
                    for l in lands
                ],
            }
            card_suggestions.append(group)

        total_lands = sum(l["quantity"] for l in lands)
        response = ai_text + "\n\n" if ai_text else ""
        response += f"Here's a **{total_lands}-land mana base** for your deck. You can add them all at once or adjust quantities."

        conversation.add_message("assistant", response)
        await self.db.commit()

        return ChatResponse(
            response=response,
            conversation_id=conversation.id,
            card_suggestions=card_suggestions if card_suggestions else None,
            suggestions=[
                "Adjust the lands",
                "Build the sideboard",
                "Show matchup analysis",
            ],
        )

    async def _handle_deck_modification(
        self,
        tool_input: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[UUID],
        ai_text: str = "",
    ) -> ChatResponse:
        """Handle deck modification from tool call."""
        modification = tool_input.get("modification", "")

        if not conversation.current_deck:
            return ChatResponse(
                response="I don't have a deck to modify. Would you like to start building one?",
                conversation_id=conversation.id,
                suggestions=["Build me a deck", "What's good right now?"],
            )

        result = await self.deck_generator.iterate(
            modification=modification,
            conversation_id=conversation.id,
            user_id=user_id,
        )

        response = ai_text + "\n\n" if ai_text else ""
        response += result.summary

        return ChatResponse(
            response=response,
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
        ai_text: str = "",
    ) -> ChatResponse:
        """Handle matchup/meta query from tool call."""
        opponent_deck = tool_input.get("opponent_deck", "")

        if not conversation.current_deck:
            response = await self._get_general_meta_advice(opponent_deck)
        else:
            response = await self._get_matchup_analysis(
                conversation.current_deck, opponent_deck
            )

        full_response = (ai_text + "\n\n" if ai_text else "") + response

        conversation.add_message("assistant", full_response)
        await self.db.commit()

        return ChatResponse(
            response=full_response,
            conversation_id=conversation.id,
            suggestions=["Show sideboard guide", "Build a counter deck"],
        )

    async def _handle_generate_full_deck(
        self,
        tool_input: Dict[str, Any],
        conversation: Conversation,
        user_id: Optional[UUID],
        ai_text: str = "",
    ) -> ChatResponse:
        """Generate a complete deck in one shot (escape hatch)."""
        colors = tool_input.get("colors", [])
        archetype = tool_input.get("archetype", "midrange")
        strategy = tool_input.get("strategy", "")
        specific_cards = tool_input.get("specific_cards", [])
        format = getattr(self, "_current_format", "standard")
        format_display = "cEDH" if format == "cedh" else format.capitalize()

        # If no colors specified, try to pick from meta
        if not colors:
            from app.models.meta import MetaSnapshot
            result = await self.db.execute(
                select(MetaSnapshot)
                .where(MetaSnapshot.format == format)
                .order_by(MetaSnapshot.meta_percentage.desc())
                .limit(1)
            )
            top = result.scalar_one_or_none()
            if top and top.archetype:
                colors = self._extract_colors_from_archetype(top.archetype)
                if not strategy:
                    strategy = top.archetype

        if not colors:
            colors = ["R", "G"]

        prompt = f"Build a {' '.join(colors) if colors else ''} {archetype} deck"
        if strategy:
            prompt += f" focused on {strategy}"
        if specific_cards:
            prompt += f" including {', '.join(specific_cards)}"

        result = await self.deck_generator.generate(
            prompt=prompt,
            user_id=user_id,
            conversation_id=conversation.id,
            include_sideboard=(format != "cedh"),
            format=format,
            colors=colors if colors else None,
            specific_cards=specific_cards if specific_cards else None,
        )

        response = ai_text + "\n\n" if ai_text else ""
        response += f"Here's your complete **{result.deck.archetype or format_display}** deck!\n\n"
        response += result.strategy_summary or ""

        return ChatResponse(
            response=response,
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

    # --- Helper methods ---

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

    def _extract_colors_from_archetype(self, archetype: str) -> List[str]:
        """Extract likely colors from an archetype name."""
        arch_lower = archetype.lower()

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

        if "aggro" in arch_lower or "burn" in arch_lower or "rdw" in arch_lower:
            return ["R"]
        elif "control" in arch_lower:
            return ["W", "U"]
        elif "ramp" in arch_lower:
            return ["G"]

        return ["R", "G"]

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

    async def _get_general_meta_advice(self, opponent_deck: str) -> str:
        """Get general meta advice when no deck context is available."""
        from app.models.meta import MetaSnapshot

        format = getattr(self, "_current_format", "standard")
        result = await self.db.execute(
            select(MetaSnapshot)
            .where(MetaSnapshot.format == format)
            .order_by(MetaSnapshot.meta_percentage.desc())
            .limit(10)
        )
        snapshots = result.scalars().all()

        format_name = "cEDH" if format == "cedh" else format.capitalize()
        response = f"**Current {format_name} Meta:**\n\n"

        if snapshots:
            for snap in snapshots:
                pct = f"{float(snap.meta_percentage):.1f}%" if snap.meta_percentage else "?"
                response += f"- **{snap.archetype}** ({pct})\n"
            response += "\n"

        if opponent_deck:
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

        format = getattr(self, "_current_format", "standard")
        deck_archetype = deck.get("archetype", "").lower()
        our_type = self._classify_deck_type(deck_archetype)

        result = await self.db.execute(
            select(MetaSnapshot)
            .where(MetaSnapshot.format == format)
            .order_by(MetaSnapshot.meta_percentage.desc())
            .limit(15)
        )
        snapshots = result.scalars().all()

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

    async def _fallback_response(
        self,
        message: str,
        conversation: Conversation,
        user_id: Optional[UUID],
    ) -> ChatResponse:
        """Fallback response when AI is not available."""
        response = (
            "I'm Spellbook, your MTG deck building partner! I can help you:\n\n"
            "- **Explore the meta**: 'What's good right now?'\n"
            "- **Build incrementally**: 'I want to build around [card name]'\n"
            "- **Get a full deck**: 'Just build me the best deck'\n"
            "- **Fill gaps**: 'I need more removal'\n"
            "- **Analyze matchups**: 'How do I beat control?'\n\n"
            "What would you like to do?"
        )

        conversation.add_message("assistant", response)
        await self.db.commit()

        return ChatResponse(
            response=response,
            conversation_id=conversation.id,
            suggestions=["What's good right now?", "Build around a card", "Just build me a deck"],
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
