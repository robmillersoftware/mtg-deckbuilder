from typing import Optional, List, Dict, Any
from uuid import UUID
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from sqlalchemy import func as sqlfunc

from app.models.card import Card
from app.models.conversation import Conversation
from app.schemas.conversation import (
    ChatResponse,
    CardExplanationResponse,
)
from app.services.card_service import CardService, get_format_legality_condition
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

    async def _resolve_card_mentions(
        self, message: str, format: str, check_legality: bool = True
    ) -> List[Dict[str, str]]:
        """
        Look up card names mentioned in the user's message against the database.
        Returns a list of dicts with card details for any matches found.

        If check_legality is False, skips format legality filtering (used as
        a fallback to detect cards that exist but aren't legal in the format).
        """
        words = message.split()
        potential_names = []

        # Build candidate card names (1-4 word combinations starting with uppercase)
        for i in range(len(words)):
            cleaned = words[i].strip(",.!?\"'")
            if cleaned and cleaned[0:1].isupper():
                potential_names.append(cleaned)
                if i < len(words) - 1:
                    potential_names.append(f"{cleaned} {words[i+1].strip(',.!?')}")
                if i < len(words) - 2:
                    potential_names.append(f"{cleaned} {words[i+1]} {words[i+2]}".strip(",.!?"))
                if i < len(words) - 3:
                    potential_names.append(f"{cleaned} {words[i+1]} {words[i+2]} {words[i+3]}".strip(",.!?"))

        resolved = []
        seen_names = set()

        legality_filter = get_format_legality_condition(format) if check_legality else None

        for name in potential_names:
            if len(name) < 3:
                continue

            # Try exact match first
            conditions = [sqlfunc.lower(Card.name) == name.lower()]
            if legality_filter is not None:
                conditions.append(legality_filter)
            query = select(Card.name, Card.type_line, Card.oracle_text, Card.mana_cost, Card.colors).where(
                *conditions
            ).limit(1)
            result = await self.db.execute(query)
            row = result.first()

            if not row:
                # Try partial match, but count how many distinct cards match
                partial_conditions = [sqlfunc.lower(Card.name).like(f"%{name.lower()}%")]
                if legality_filter is not None:
                    partial_conditions.append(legality_filter)
                count_query = select(sqlfunc.count(sqlfunc.distinct(Card.name))).where(
                    *partial_conditions
                )
                count_result = await self.db.execute(count_query)
                match_count = count_result.scalar() or 0

                if match_count == 1:
                    # Only one card matches - safe to use it
                    query = select(Card.name, Card.type_line, Card.oracle_text, Card.mana_cost, Card.colors).where(
                        *partial_conditions
                    ).limit(1)
                    result = await self.db.execute(query)
                    row = result.first()

            if row and row.name not in seen_names:
                seen_names.add(row.name)
                resolved.append({
                    "name": row.name,
                    "type_line": row.type_line or "",
                    "oracle_text": row.oracle_text or "",
                    "mana_cost": row.mana_cost or "",
                    "colors": row.colors or [],
                })

        return resolved

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

        # Ensure conversation context exists
        if conversation.context is None:
            conversation.context = {}

        # Add user message to conversation
        conversation.add_message("user", message)

        if not settings.ANTHROPIC_API_KEY:
            return await self._fallback_response(message, conversation, user_id)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

            # Resolve card names mentioned in the message
            resolved_cards = await self._resolve_card_mentions(message, format)

            # Fallback: if no cards found with format filter, try without format
            # restriction to detect cards that exist but aren't legal in the
            # selected format.  This prevents the AI from hallucinating card
            # details and lets us warn it about legality.
            format_illegal_cards: List[Dict[str, str]] = []
            if not resolved_cards:
                all_cards = await self._resolve_card_mentions(
                    message, format, check_legality=False
                )
                format_illegal_cards = all_cards

            # Build card context if any cards were found
            card_context = ""
            if resolved_cards:
                card_lines = []
                for c in resolved_cards:
                    card_lines.append(
                        f"- {c['name']} {c['mana_cost']} — {c['type_line']}: {c['oracle_text'][:200]}"
                    )
                card_context = (
                    "\n\nCARD REFERENCES (verified from database):\n"
                    + "\n".join(card_lines)
                    + "\nThese cards have been verified. Do NOT ask for clarification about them."
                )
            elif format_illegal_cards:
                format_name_upper = "cEDH" if format == "cedh" else format.capitalize()
                card_lines = []
                for c in format_illegal_cards:
                    card_lines.append(
                        f"- {c['name']} {c['mana_cost']} — {c['type_line']}: {c['oracle_text'][:200]}"
                    )
                card_context = (
                    f"\n\nCARD REFERENCES (found but NOT LEGAL in {format_name_upper}):\n"
                    + "\n".join(card_lines)
                    + f"\nThese cards are NOT legal in {format_name_upper}. "
                    + f"Do NOT build a deck with them. Instead, help the user find "
                    + f"{format_name_upper}-legal cards that enable a similar strategy. "
                    + f"Call suggest_core with a strategy description inspired by what "
                    + f"these cards do, using only {format_name_upper}-legal cards."
                )

            # Build deck context string
            deck_context = self._build_deck_context(deck)

            # Build conversation context string from persisted state
            conv_context = self._build_conversation_context(conversation)

            # Build conversation summary for long conversations
            conv_summary = self._build_conversation_summary(conversation)

            # Format-specific guidance
            format_name = "cEDH" if format == "cedh" else format.capitalize()
            format_guidance = self._get_format_guidance(format, format_name)

            system_prompt = f"""You are Spellbook, a collaborative deck building partner for {format_name} format.
You work WITH the user, not FOR them. Suggest cards, explain choices, let them decide.

{format_guidance}

{conv_context}
{deck_context}
{conv_summary}

CRITICAL RULE - ALWAYS USE TOOLS TO RECOMMEND CARDS:
When the user names a specific card to build around, or picks colors/an archetype, you MUST call suggest_core. Card recommendations MUST go through the tools so they appear in the interactive UI.

FLOW:
1. EXPLORE: Help settle on a direction. ONLY if the user is truly vague (e.g. "help me", "what's good"), use analyze_meta. If they name ANY specific card, colors, or archetype, skip directly to step 2.
2. BUILD CORE: Call suggest_core with the strategy, colors (inferred from the card if needed), and 3-5 role groups. You may include a BRIEF (1-2 sentence) introduction before the tool call, but the tool call is MANDATORY.
3. FILL GAPS: After user adds cards, use suggest_package for missing roles.
4. LANDS: When nonland count is near target, use finalize_mana_base. ONLY use finalize_mana_base for lands.
5. REFINE: Help with cuts, sideboard, matchups using modify_deck or get_matchup_info.

RULES:
- Only use generate_full_deck when the user explicitly asks to skip suggestions (e.g. "just build it", "skip suggestions").
- When using suggest_core, pick 3-5 meaningful role names for the groups.
- Reserve all land suggestions for finalize_mana_base. Keep suggest_core and suggest_package focused on nonland cards only.
- For cEDH: suggest in larger batches (10-15 per group, ~5 groups).
- For questions about matchups, strategy, or "how do I beat X" - use get_matchup_info or respond with text advice only.
- When the user says "what's good" or similar vague exploration, use analyze_meta.
- Be concise in your text responses. Focus on actionable advice.
- If CARD REFERENCES are provided below, the card has already been identified from the database. Proceed directly with suggest_core using the resolved card.
- IMPORTANT: When the user asks for "more support", "more help", "more options", or similar continuation requests, continue building on the CURRENT STRATEGY described in the conversation context above. Suggest more cards, offer alternative approaches within the strategy, or advance to the next phase.{card_context}"""

            # Build conversation history - send last 20 messages for multi-turn context
            recent = conversation.messages[-20:] if conversation.messages else []
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

            # When a card has been resolved from the database or the user
            # expresses clear deck-building intent, force tool use so card
            # recommendations go through the interactive UI.
            build_intent_keywords = [
                "build around", "build with", "i want to build",
                "i want to play", "deck with", "deck around",
                "deck featuring", "brew around", "brew with",
            ]
            has_build_intent = any(
                kw in message.lower() for kw in build_intent_keywords
            )
            api_kwargs = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2048,
                "system": system_prompt,
                "tools": TOOLS,
                "messages": api_messages,
            }
            if (resolved_cards or format_illegal_cards or has_build_intent) and not deck:
                # User named a card or expressed build intent and no deck
                # exists yet -> force a tool call so cards go through the UI
                api_kwargs["tool_choice"] = {"type": "any"}

            response = client.messages.create(**api_kwargs)

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
                    suggestions=self._get_suggestions(deck, conversation),
                )

        except Exception as e:
            logger.error(f"Chat processing error: {e}", exc_info=True)

        return await self._fallback_response(message, conversation, user_id)

    def _build_conversation_context(self, conversation: Conversation) -> str:
        """Build a structured context string from persisted conversation state.

        This ensures Claude always knows what strategy is being built,
        what phase the conversation is in, and what key decisions have been made,
        even if the early messages have scrolled out of the 20-message window.
        """
        ctx = conversation.get_context()
        if not ctx:
            return ""

        parts = ["CONVERSATION CONTEXT (persisted state):"]

        if ctx.get("strategy"):
            parts.append(f"- Strategy: {ctx['strategy']}")
        if ctx.get("colors"):
            color_str = ", ".join(ctx["colors"])
            parts.append(f"- Colors: {color_str}")
        if ctx.get("archetype"):
            parts.append(f"- Archetype: {ctx['archetype']}")
        if ctx.get("build_around_cards"):
            cards_str = ", ".join(ctx["build_around_cards"])
            parts.append(f"- Build-around cards: {cards_str}")
        if ctx.get("phase"):
            parts.append(f"- Current phase: {ctx['phase']}")
        if ctx.get("roles_suggested"):
            roles_str = ", ".join(ctx["roles_suggested"])
            parts.append(f"- Roles already suggested: {roles_str}")
        if ctx.get("user_preferences"):
            parts.append(f"- User preferences: {ctx['user_preferences']}")

        if len(parts) == 1:
            return ""

        parts.append("Use this context to maintain continuity. The user expects you to remember what you're building together.")
        return "\n".join(parts)

    def _build_conversation_summary(self, conversation: Conversation) -> str:
        """Build a summary of older messages that have scrolled out of the recent window.

        When the conversation exceeds 20 messages, the earlier messages are dropped
        from the API call. This method provides a text summary of those older messages
        so important context isn't lost.
        """
        messages = conversation.messages or []
        if len(messages) <= 20:
            return ""

        # Summarize messages that won't be in the recent window
        older_messages = messages[:-20]

        # Extract key decisions and actions from older messages
        key_points = []
        for msg in older_messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            if role == "user":
                # Capture user's stated goals and preferences
                lower = content.lower()
                if any(kw in lower for kw in ["i want", "build around", "focus on", "i like", "i prefer", "let's build"]):
                    # Truncate very long messages
                    key_points.append(f"User said: \"{content[:150]}\"")
            elif role == "assistant":
                # Capture key suggestions/decisions from assistant
                if "suggestions for your" in content.lower():
                    # This was a suggest_core response - capture the strategy
                    idx = content.lower().find("suggestions for your")
                    snippet = content[idx:idx+100].split("deck")[0] + "deck"
                    key_points.append(f"Assistant provided {snippet}")
                elif "meta:" in content.lower():
                    key_points.append("Assistant provided meta analysis")

        if not key_points:
            return ""

        summary = "EARLIER CONVERSATION SUMMARY (messages before the recent window):\n"
        summary += "\n".join(f"- {point}" for point in key_points[:10])
        return summary

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
        # All non-Commander 60-card formats
        return f"""FORMAT: {format_name.upper()} (60-card constructed)
- 60 card minimum main deck, 15 card sideboard.
- Up to 4 copies of any non-basic land card.
- Only suggest cards that are legal in {format_name}.
- Think in terms of 4-of playsets, curve optimization, and {format_name} metagame archetypes.
- When a user says "build around [card]", they want a 60-card {format_name} deck featuring that card as a 4-of.
- Target ~24 lands, ~36 nonland cards (varies by archetype)."""

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

        # Persist context from tool inputs so future turns remember what we're building
        self._update_context_from_tool(tool_name, tool_input, conversation)

        # Handlers that need user_id
        if tool_name in ("modify_deck", "generate_full_deck"):
            return await handler(tool_input, conversation, user_id, ai_text)
        elif tool_name in ("suggest_core", "suggest_package", "finalize_mana_base", "analyze_meta"):
            return await handler(tool_input, conversation, ai_text)
        else:
            return await handler(tool_input, conversation, ai_text)

    def _update_context_from_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        conversation: Conversation,
    ) -> None:
        """Extract and persist conversation context from tool call parameters.

        This is the key to maintaining context across turns. When Claude calls
        suggest_core with strategy="Moonshadow graveyard synergy" and colors=["B","G"],
        we persist that so the next turn's system prompt includes it.
        """
        if tool_name == "suggest_core":
            conversation.update_context(
                strategy=tool_input.get("strategy"),
                colors=tool_input.get("colors"),
                phase="building_core",
            )
            # Track roles suggested so far
            roles = tool_input.get("roles", [])
            existing_roles = conversation.get_context().get("roles_suggested", [])
            all_roles = list(set(existing_roles + roles))
            conversation.update_context(roles_suggested=all_roles)

            # Track build-around cards from strategy
            if tool_input.get("strategy"):
                conversation.update_context(
                    archetype=tool_input.get("strategy")
                )

        elif tool_name == "suggest_package":
            conversation.update_context(
                phase="filling_gaps",
            )
            # Update strategy/colors if provided (may refine from earlier)
            if tool_input.get("strategy"):
                conversation.update_context(strategy=tool_input["strategy"])
            if tool_input.get("colors"):
                conversation.update_context(colors=tool_input["colors"])
            # Track additional roles
            role = tool_input.get("role")
            if role:
                existing_roles = conversation.get_context().get("roles_suggested", [])
                if role not in existing_roles:
                    conversation.update_context(roles_suggested=existing_roles + [role])

        elif tool_name == "finalize_mana_base":
            conversation.update_context(phase="lands")
            if tool_input.get("colors"):
                conversation.update_context(colors=tool_input["colors"])

        elif tool_name == "generate_full_deck":
            conversation.update_context(
                strategy=tool_input.get("strategy"),
                colors=tool_input.get("colors"),
                archetype=tool_input.get("archetype"),
                phase="complete",
            )
            if tool_input.get("specific_cards"):
                conversation.update_context(
                    build_around_cards=tool_input["specific_cards"]
                )

        elif tool_name == "analyze_meta":
            conversation.update_context(phase="exploring")

        elif tool_name == "modify_deck":
            conversation.update_context(phase="refining")

        elif tool_name == "get_matchup_info":
            conversation.update_context(phase="refining")

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

    @staticmethod
    def _is_land_card(card: Dict[str, Any]) -> bool:
        """Check if a card is a land based on its type_line."""
        type_line = (card.get("type_line") or "").lower()
        return "land" in type_line

    @staticmethod
    def _filter_lands_from_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove land cards from a list of card suggestions."""
        return [c for c in cards if not ChatService._is_land_card(c)]

    async def _handle_suggest_core(
        self,
        tool_input: Dict[str, Any],
        conversation: Conversation,
        ai_text: str = "",
    ) -> ChatResponse:
        """Suggest core cards in role-based groups, incorporating tournament data."""
        format = getattr(self, "_current_format", "standard")
        strategy = tool_input.get("strategy", "")
        colors = tool_input.get("colors", [])
        roles = tool_input.get("roles", ["threats", "removal", "card advantage"])

        # Filter out land-related roles - lands come from finalize_mana_base
        land_role_keywords = {"land", "lands", "mana base", "manabase", "mana_base"}
        roles = [r for r in roles if r.lower() not in land_role_keywords]
        if not roles:
            roles = ["threats", "removal", "card advantage"]

        # Get existing card names from deck
        deck = conversation.current_deck or {}
        existing = [e.get("card_name", "") for e in deck.get("main_deck", [])]
        existing_lower = {n.lower() for n in existing}

        cards_per_role = 12 if format == "cedh" else 6

        # Extract card names from the strategy to find co-occurring cards
        # (e.g., strategy="Moonshadow enchantment aggro" -> look for "Moonshadow")
        build_around_cards = []
        if strategy:
            resolved = await self._resolve_card_mentions(strategy, format)
            build_around_cards = [c["name"] for c in resolved]

        # Persist build-around cards to conversation context
        if build_around_cards:
            conversation.update_context(build_around_cards=build_around_cards)

        # Get co-occurrence based synergy cards if we have build-around targets
        synergy_group = None
        if build_around_cards:
            try:
                cooccurrence_cards = await self.ai_service._get_cooccurrence_cards(
                    card_names=build_around_cards,
                    colors=colors,
                    limit=cards_per_role,
                    format=format,
                )
                if cooccurrence_cards:
                    # Fetch full card data for co-occurrence results
                    synergy_cards = []
                    for cc in cooccurrence_cards:
                        if cc["name"].lower() in existing_lower:
                            continue
                        card = await self.card_service.get_by_name(cc["name"], format=format)
                        if card and "land" not in (card.type_line or "").lower():
                            synergy_cards.append({
                                "card_name": card.name,
                                "quantity": 1 if format == "cedh" else 4,
                                "mana_cost": card.mana_cost,
                                "type_line": card.type_line,
                                "image_uri": card.image_uri,
                                "reasoning": None,
                            })
                        if len(synergy_cards) >= cards_per_role:
                            break
                    if synergy_cards:
                        synergy_group = {
                            "group_name": f"Synergy with {', '.join(build_around_cards)}",
                            "role": "synergy",
                            "is_batch": False,
                            "cards": synergy_cards,
                        }
            except Exception as e:
                logger.warning(f"Co-occurrence lookup failed: {e}")

        # Query cards grouped by role (now tournament-aware)
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

        # Add synergy group first if we have build-around cards
        if synergy_group:
            card_suggestions.append(synergy_group)

        for role, cards in role_cards.items():
            # Filter out any land cards that slipped through semantic search
            filtered_cards = self._filter_lands_from_cards(cards)
            if not filtered_cards:
                continue
            group = {
                "group_name": role.replace("_", " ").title(),
                "role": role,
                "is_batch": False,
                "cards": [
                    {
                        "card_name": c["card_name"],
                        "quantity": 1 if format == "cedh" else 4,
                        "mana_cost": c.get("mana_cost"),
                        "type_line": c.get("type_line"),
                        "image_uri": c.get("image_uri"),
                        "reasoning": None,
                    }
                    for c in filtered_cards
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
        # Filter out any land cards - lands come from finalize_mana_base
        cards = self._filter_lands_from_cards(cards)
        if cards:
            group = {
                "group_name": role.replace("_", " ").title(),
                "role": role,
                "is_batch": False,
                "cards": [
                    {
                        "card_name": c["card_name"],
                        "quantity": 1 if format == "cedh" else 4,
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

    def _get_suggestions(self, deck: Optional[Dict[str, Any]], conversation: Optional[Conversation] = None) -> List[str]:
        """Get contextual suggestions based on current state and conversation context."""
        ctx = conversation.get_context() if conversation else {}
        phase = ctx.get("phase")
        strategy = ctx.get("strategy")

        if deck:
            main_count = sum(e.get("quantity", 1) for e in deck.get("main_deck", []))
            has_lands = any(
                "land" in (e.get("card", {}) or {}).get("type_line", "").lower()
                for e in deck.get("main_deck", [])
            )

            if phase == "building_core" or (main_count < 20):
                return ["Suggest more cards", "I need removal", "I need card draw"]
            elif phase == "filling_gaps" or (main_count < 36 and not has_lands):
                return ["What am I missing?", "Let's add the mana base", "Show matchups"]
            elif not has_lands:
                return ["Let's add the mana base", "Show matchups", "Make changes"]
            else:
                return ["Show matchups", "Make changes", "Build the sideboard"]

        if strategy:
            return ["Continue building", "Show more options", "Start over"]

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
        """Fallback response when AI is not available. Context-aware."""
        ctx = conversation.get_context()
        strategy = ctx.get("strategy")
        phase = ctx.get("phase")

        # If we have context, provide a context-aware fallback instead of generic help
        if strategy:
            response = (
                f"I'm still working with you on your **{strategy}** deck. "
                "I had a brief hiccup processing that. Could you rephrase what you'd like?\n\n"
            )
            if phase in ("building_core", "filling_gaps"):
                response += (
                    "- **More suggestions**: 'Suggest more cards for this strategy'\n"
                    "- **Fill a role**: 'I need more removal' or 'I need card draw'\n"
                    "- **Add lands**: 'Let's add the mana base'\n"
                )
                suggestions = [
                    "Suggest more cards",
                    "I need more removal",
                    "Let's add the mana base",
                ]
            elif phase == "lands":
                response += (
                    "- **Adjust**: 'Change the land count'\n"
                    "- **Sideboard**: 'Build the sideboard'\n"
                    "- **Matchups**: 'How do I beat aggro?'\n"
                )
                suggestions = ["Build the sideboard", "Show matchups", "Adjust the lands"]
            else:
                response += (
                    "- **Keep building**: 'Suggest more cards'\n"
                    "- **Matchups**: 'How does this do against control?'\n"
                    "- **Modify**: 'I want to make changes'\n"
                )
                suggestions = ["Suggest more cards", "Show matchups", "Make changes"]
        else:
            response = (
                "I'm Spellbook, your MTG deck building partner! I can help you:\n\n"
                "- **Explore the meta**: 'What's good right now?'\n"
                "- **Build incrementally**: 'I want to build around [card name]'\n"
                "- **Get a full deck**: 'Just build me the best deck'\n"
                "- **Fill gaps**: 'I need more removal'\n"
                "- **Analyze matchups**: 'How do I beat control?'\n\n"
                "What would you like to do?"
            )
            suggestions = ["What's good right now?", "Build around a card", "Just build me a deck"]

        conversation.add_message("assistant", response)
        await self.db.commit()

        return ChatResponse(
            response=response,
            conversation_id=conversation.id,
            suggestions=suggestions,
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
