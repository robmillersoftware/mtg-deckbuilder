from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
import logging
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.deck import Deck
from app.models.conversation import Conversation
from app.models.meta import MetaSnapshot, ArchetypeTemplate
from app.schemas.deck import (
    DeckResponse,
    DeckGenerateResponse,
    DeckIterateResponse,
    SlotRecommendation,
    SideboardEntry,
    ChangeLogEntry,
)
from app.services.card_service import CardService
from app.services.deck_validator import DeckValidator
from app.services.ai_service import AIService
from app.core.security import generate_share_token

logger = logging.getLogger(__name__)


class DeckGenerator:
    """
    AI-powered deck generation service.
    Uses LLM to interpret requests and build competitive decks.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.card_service = CardService(db)
        self.validator = DeckValidator(db)
        self.ai_service = AIService(db)

    async def generate(
        self,
        prompt: str,
        user_id: Optional[UUID] = None,
        conversation_id: Optional[UUID] = None,
        include_sideboard: bool = True,
        format: str = "standard",
        colors: Optional[List[str]] = None,
        specific_cards: Optional[List[str]] = None,
    ) -> DeckGenerateResponse:
        """
        Generate a deck based on natural language prompt.

        Args:
            prompt: Natural language description of desired deck
            user_id: Optional user ID for saving conversation
            conversation_id: Optional existing conversation to continue
            include_sideboard: Whether to generate sideboard
            format: Game format (standard, historic, modern, legacy, cedh)
            colors: Optional explicit colors (skip parsing if provided)
            specific_cards: Optional explicit specific cards (skip parsing if provided)

        Returns:
            DeckGenerateResponse with complete deck and strategy
        """
        # Get or create conversation
        conversation = await self._get_or_create_conversation(conversation_id, user_id)

        # Add user message to conversation
        conversation.add_message("user", prompt)

        # Parse the request to understand intent (if colors/specific_cards not provided explicitly)
        parsed_request = await self.ai_service.parse_deck_request(prompt)

        # Use explicit colors if provided, otherwise use parsed colors
        if colors is None:
            colors = parsed_request.get("colors", [])

        # Use explicit specific_cards if provided, otherwise use parsed specific_cards
        if specific_cards is None:
            specific_cards = parsed_request.get("specific_cards", [])

        logger.info(f"[DECK-GEN] Format: {format}, Colors: {colors}, Specific cards: {specific_cards}")

        # For cEDH/Commander: look up commander color identity from specific cards
        # Only do this if colors weren't explicitly provided (i.e., came from parsing)
        if format == "cedh" and specific_cards and not colors:
            for card_name in specific_cards:
                commander_colors = await self.ai_service.get_commander_color_identity(card_name)
                if commander_colors:
                    logger.info(f"Using commander {card_name} color identity: {commander_colors}")
                    colors = commander_colors
                    break  # Use the first card's color identity (assume it's the commander)

        # Get meta data for context
        meta_data = await self._get_meta_context(format=format)

        # Get archetype template for role distribution guidance
        archetype_template = await self._get_archetype_template(
            parsed_request.get("archetype", ""),
            format=format,
        )
        if archetype_template:
            logger.info(
                f"Using {archetype_template['archetype_category']} template "
                f"(from {archetype_template['sample_size']} tournament decks)"
            )

        # Generate the deck
        deck_data = await self.ai_service.generate_deck(
            archetype=parsed_request.get("archetype", ""),
            colors=colors,
            strategy=parsed_request.get("strategy", ""),
            meta_context=meta_data,
            include_sideboard=include_sideboard,
            specific_cards=specific_cards,
            archetype_template=archetype_template,
            format=format,
        )

        # Validate all cards exist and are legal
        main_deck = deck_data.get("main_deck", [])
        sideboard = deck_data.get("sideboard", [])

        validated_main = await self._validate_and_fix_cards(main_deck, format=format)
        validated_sideboard = await self._validate_and_fix_cards(sideboard, format=format)

        # Run deck validation
        validation = await self.validator.validate(validated_main, validated_sideboard, format=format)

        # Generate unique deck name if user is logged in
        deck_name = deck_data.get("name", "Generated Deck")
        if user_id:
            deck_name = await self._get_unique_deck_name(user_id, deck_name)

        # Determine commander for cEDH/Commander decks
        commander_name = None
        if format in ["cedh", "commander"] and specific_cards:
            # The first specific card mentioned is typically the commander
            commander_name = specific_cards[0]
            logger.info(f"Setting commander for {format} deck: {commander_name}")

        # Create deck object (not saved to DB - user must explicitly save)
        deck = Deck(
            id=uuid4(),
            owner_id=user_id,
            name=deck_name,
            format=format,
            archetype=parsed_request.get("archetype"),
            commander=commander_name,
            main_deck=validated_main,
            sideboard=validated_sideboard,
            strategy_summary=deck_data.get("strategy_summary", ""),
            is_validated=validation.is_valid,
            validation_errors=[e.model_dump() for e in validation.errors] if validation.errors else None,
        )

        # Update conversation with deck
        # Commander needs to be stored as a DeckEntry dict for frontend compatibility
        commander_entry = None
        if commander_name:
            commander_entry = {
                "card_name": commander_name,
                "quantity": 1,
                "card": None,
            }
        conversation.current_deck = {
            "name": deck.name,
            "archetype": deck.archetype,
            "commander": commander_entry,
            "main_deck": deck.main_deck,
            "sideboard": deck.sideboard,
            "format": format,
        }

        # Generate response message
        response_message = self._format_deck_response(deck, deck_data)
        conversation.add_message("assistant", response_message)
        conversation.summary = f"Deck: {deck.name}"

        await self.db.commit()
        await self.db.refresh(conversation)

        # Build slot recommendations
        slot_recommendations = [
            SlotRecommendation(
                slot_type=rec.get("slot_type", ""),
                role_description=rec.get("role", ""),
                card_name=rec.get("card_name", ""),
                quantity=rec.get("quantity", 1),
                reasoning=rec.get("reasoning", ""),
            )
            for rec in deck_data.get("slot_recommendations", [])
        ]

        # Build sideboard guide
        sideboard_guide = [
            SideboardEntry(
                card_name=entry.get("card_name", ""),
                quantity=entry.get("quantity", 1),
                matchups=entry.get("matchups", []),
                reasoning=entry.get("reasoning", ""),
            )
            for entry in deck_data.get("sideboard_guide", [])
        ]

        return DeckGenerateResponse(
            deck=DeckResponse(
                id=deck.id,
                owner_id=user_id or uuid4(),
                name=deck.name,
                description=deck.description,
                format=deck.format,
                archetype=deck.archetype,
                commander=deck.commander,
                main_deck=deck.main_deck,
                sideboard=deck.sideboard,
                strategy_summary=deck.strategy_summary,
                card_explanations=deck.card_explanations,
                matchup_notes=deck.matchup_notes,
                visibility=deck.visibility or "private",
                share_token=deck.share_token,
                is_validated=deck.is_validated,
                validation_errors=deck.validation_errors,
                created_at=deck.created_at or datetime.utcnow(),
                updated_at=deck.updated_at or datetime.utcnow(),
            ),
            conversation_id=conversation.id,
            strategy_summary=deck.strategy_summary or "",
            slot_recommendations=slot_recommendations,
            sideboard_guide=sideboard_guide,
        )

    async def iterate(
        self,
        modification: str,
        conversation_id: Optional[UUID] = None,
        deck_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        format: str = "standard",
    ) -> DeckIterateResponse:
        """
        Modify an existing deck based on natural language instructions.

        Args:
            modification: Description of desired changes
            conversation_id: Conversation with current deck context
            deck_id: Alternatively, specify deck ID directly
            user_id: Optional user ID
            format: Game format (standard, historic, modern, legacy, cedh)

        Returns:
            DeckIterateResponse with updated deck and change log
        """
        # Get current deck
        current_deck = None
        deck_format = format  # Use provided format as default

        if conversation_id:
            result = await self.db.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conversation = result.scalar_one_or_none()
            if conversation and conversation.current_deck:
                current_deck = conversation.current_deck
                # Use format from conversation if available
                deck_format = current_deck.get("format", format)
        elif deck_id:
            result = await self.db.execute(select(Deck).where(Deck.id == deck_id))
            deck = result.scalar_one_or_none()
            if deck:
                current_deck = {
                    "name": deck.name,
                    "archetype": deck.archetype,
                    "commander": deck.commander,
                    "main_deck": deck.main_deck,
                    "sideboard": deck.sideboard,
                    "format": deck.format,
                }
                deck_format = deck.format or format
                conversation = await self._get_or_create_conversation(None, user_id)

        if current_deck is None:
            raise ValueError("No deck found to iterate on")

        # Add user message
        conversation.add_message("user", modification)

        # Get AI to suggest modifications
        changes = await self.ai_service.iterate_deck(
            current_deck=current_deck,
            modification_request=modification,
        )

        # Apply changes
        new_main_deck = list(current_deck.get("main_deck", []))
        new_sideboard = list(current_deck.get("sideboard", []))
        change_log = []

        for change in changes.get("changes", []):
            action = change.get("action")
            card_name = change.get("card_name")
            quantity = change.get("quantity", 1)
            target = change.get("target", "main_deck")

            if action == "remove":
                deck_list = new_main_deck if target == "main_deck" else new_sideboard
                for entry in deck_list[:]:
                    if entry.get("card_name") == card_name:
                        old_qty = entry.get("quantity", 0)
                        if quantity >= old_qty:
                            deck_list.remove(entry)
                        else:
                            entry["quantity"] = old_qty - quantity
                        change_log.append(
                            ChangeLogEntry(
                                action="removed",
                                card_name=card_name,
                                old_quantity=old_qty,
                                new_quantity=max(0, old_qty - quantity),
                                reasoning=change.get("reasoning", ""),
                            )
                        )
                        break

            elif action == "add":
                deck_list = new_main_deck if target == "main_deck" else new_sideboard
                # Check if card already exists
                existing = next((e for e in deck_list if e.get("card_name") == card_name), None)
                if existing:
                    old_qty = existing.get("quantity", 0)
                    existing["quantity"] = old_qty + quantity
                    change_log.append(
                        ChangeLogEntry(
                            action="changed",
                            card_name=card_name,
                            old_quantity=old_qty,
                            new_quantity=old_qty + quantity,
                            reasoning=change.get("reasoning", ""),
                        )
                    )
                else:
                    deck_list.append({"card_name": card_name, "quantity": quantity})
                    change_log.append(
                        ChangeLogEntry(
                            action="added",
                            card_name=card_name,
                            old_quantity=0,
                            new_quantity=quantity,
                            reasoning=change.get("reasoning", ""),
                        )
                    )

        # Validate the result
        validated_main = await self._validate_and_fix_cards(new_main_deck, format=deck_format)
        validated_sideboard = await self._validate_and_fix_cards(new_sideboard, format=deck_format)
        validation = await self.validator.validate(validated_main, validated_sideboard, format=deck_format)

        # Update conversation
        # Preserve commander in DeckEntry dict format
        existing_commander = current_deck.get("commander")
        # If commander is a string (legacy format), convert to dict
        if isinstance(existing_commander, str):
            existing_commander = {
                "card_name": existing_commander,
                "quantity": 1,
                "card": None,
            }
        conversation.current_deck = {
            "name": current_deck.get("name", "Modified Deck"),
            "archetype": current_deck.get("archetype"),
            "commander": existing_commander,
            "main_deck": validated_main,
            "sideboard": validated_sideboard,
            "format": deck_format,
        }

        response_message = f"I've made the following changes:\n" + "\n".join(
            f"- {c.action.capitalize()} {c.new_quantity - (c.old_quantity or 0) if c.action == 'added' else (c.old_quantity or 0) - (c.new_quantity or 0)}x {c.card_name}: {c.reasoning}"
            for c in change_log
        )
        conversation.add_message("assistant", response_message)

        await self.db.commit()

        # Create response deck
        deck = Deck(
            id=uuid4(),
            owner_id=user_id or uuid4(),
            name=current_deck.get("name", "Modified Deck"),
            format=deck_format,
            archetype=current_deck.get("archetype"),
            commander=current_deck.get("commander"),
            main_deck=validated_main,
            sideboard=validated_sideboard,
            is_validated=validation.is_valid,
        )

        return DeckIterateResponse(
            deck=DeckResponse(
                id=deck.id,
                owner_id=deck.owner_id,
                name=deck.name,
                description=deck.description,
                format=deck.format,
                archetype=deck.archetype,
                commander=deck.commander,
                main_deck=deck.main_deck,
                sideboard=deck.sideboard,
                strategy_summary=deck.strategy_summary,
                card_explanations=deck.card_explanations,
                matchup_notes=deck.matchup_notes,
                visibility=deck.visibility or "private",
                share_token=deck.share_token,
                is_validated=deck.is_validated,
                validation_errors=deck.validation_errors,
                created_at=deck.created_at or datetime.utcnow(),
                updated_at=deck.updated_at or datetime.utcnow(),
            ),
            changes=change_log,
            summary=changes.get("summary", "Deck has been modified"),
        )

    async def _get_or_create_conversation(
        self,
        conversation_id: Optional[UUID],
        user_id: Optional[UUID],
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

    async def _get_unique_deck_name(self, user_id: UUID, base_name: str) -> str:
        """Generate a unique deck name for the user."""
        from app.models.deck import Deck

        # Check if base name exists
        result = await self.db.execute(
            select(Deck).where(
                Deck.owner_id == user_id,
                Deck.name == base_name,
            )
        )
        if result.scalar_one_or_none() is None:
            return base_name

        # Find unique name with suffix
        for i in range(2, 100):
            candidate = f"{base_name} #{i}"
            result = await self.db.execute(
                select(Deck).where(
                    Deck.owner_id == user_id,
                    Deck.name == candidate,
                )
            )
            if result.scalar_one_or_none() is None:
                return candidate

        # Fallback to timestamp
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        return f"{base_name} ({timestamp})"

    async def _get_meta_context(self, format: str = "standard") -> Dict[str, Any]:
        """Get current meta information for AI context."""
        result = await self.db.execute(
            select(MetaSnapshot)
            .where(MetaSnapshot.format == format)
            .order_by(MetaSnapshot.meta_percentage.desc())
            .limit(10)
        )
        snapshots = result.scalars().all()

        return {
            "archetypes": [
                {
                    "name": s.archetype,
                    "meta_percentage": float(s.meta_percentage) if s.meta_percentage else 0,
                    "key_cards": s.key_cards or [],
                }
                for s in snapshots
            ]
        }

    async def _get_archetype_template(self, archetype: str, format: str = "standard") -> Optional[Dict[str, Any]]:
        """Get the archetype template for role distribution guidance."""
        # Map archetype to category
        archetype_lower = archetype.lower() if archetype else ""

        # Categorization rules (same as compute_archetype_templates.py)
        category = "other"
        if any(kw in archetype_lower for kw in ["aggro", "weenie", "red deck wins", "rdw", "burn", "sligh"]):
            category = "aggro"
        elif any(kw in archetype_lower for kw in ["control"]):
            category = "control"
        elif any(kw in archetype_lower for kw in ["midrange", "landfall", "rhythm", "ramp", "value"]):
            category = "midrange"
        elif any(kw in archetype_lower for kw in ["combo", "reanimate", "reanimator", "storm"]):
            category = "combo"

        result = await self.db.execute(
            select(ArchetypeTemplate)
            .where(ArchetypeTemplate.archetype_category == category)
            .where(ArchetypeTemplate.format == format)
        )
        template = result.scalar_one_or_none()

        if not template:
            logger.info(f"No archetype template found for category '{category}'")
            return None

        return {
            "archetype_category": template.archetype_category,
            "sample_size": template.sample_size,
            "avg_lands": float(template.avg_lands),
            "avg_nonlands": float(template.avg_nonlands),
            "role_distribution": template.role_distribution,
        }

    async def _validate_and_fix_cards(
        self,
        card_list: List[Dict[str, Any]],
        format: str = "standard",
    ) -> List[Dict[str, Any]]:
        """Validate cards exist and get their IDs. Includes card type data for frontend."""
        validated = []

        for entry in card_list:
            card_name = entry.get("card_name", "")
            quantity = entry.get("quantity", 1)

            card = await self.card_service.get_by_name(card_name, format=format)
            if card:
                validated.append({
                    "card_id": str(card.id),
                    "card_name": card.name,  # Use canonical name
                    "quantity": quantity,
                    "set_code": card.set_code,
                    "collector_number": card.collector_number,
                    "card": {
                        "type_line": card.type_line,
                        "mana_cost": card.mana_cost,
                        "colors": card.colors,
                        "image_uri": card.image_uri,
                    },
                })
            else:
                # Try fuzzy match
                similar = await self.card_service.fuzzy_search_by_name(card_name, limit=1)
                if similar:
                    card = similar[0]
                    validated.append({
                        "card_id": str(card.id),
                        "card_name": card.name,
                        "quantity": quantity,
                        "set_code": card.set_code,
                        "collector_number": card.collector_number,
                        "card": {
                            "type_line": card.type_line,
                            "mana_cost": card.mana_cost,
                            "colors": card.colors,
                            "image_uri": card.image_uri,
                        },
                    })
                else:
                    logger.warning(f"Card not found: {card_name}")
                    validated.append({
                        "card_name": card_name,
                        "quantity": quantity,
                    })

        return validated

    def _format_deck_response(self, deck: Deck, deck_data: Dict[str, Any]) -> str:
        """Format deck as response message."""
        lines = [
            f"## {deck.name}",
            "",
            "### Strategy",
            deck_data.get("strategy_summary", ""),
            "",
            f"### Main Deck ({sum(e.get('quantity', 0) for e in deck.main_deck)})",
            "",
        ]

        # Group by type
        creatures = []
        spells = []
        lands = []

        for entry in deck.main_deck:
            card_name = entry.get("card_name", "")
            quantity = entry.get("quantity", 1)
            line = f"{quantity} {card_name}"

            # Simple classification (would be improved with actual card data)
            if "land" in card_name.lower():
                lands.append(line)
            elif any(t in card_name.lower() for t in ["creature", "goblin", "wizard"]):
                creatures.append(line)
            else:
                spells.append(line)

        if creatures:
            lines.append("**Creatures**")
            lines.extend(creatures)
            lines.append("")

        if spells:
            lines.append("**Spells**")
            lines.extend(spells)
            lines.append("")

        if lands:
            lines.append("**Lands**")
            lines.extend(lands)
            lines.append("")

        if deck.sideboard:
            lines.append(f"### Sideboard ({sum(e.get('quantity', 0) for e in deck.sideboard)})")
            lines.append("")
            for entry in deck.sideboard:
                card_name = entry.get("card_name", "")
                quantity = entry.get("quantity", 1)
                lines.append(f"{quantity} {card_name}")

        return "\n".join(lines)
