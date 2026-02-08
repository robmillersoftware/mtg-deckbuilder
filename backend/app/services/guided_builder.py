"""
Guided deck building service.

Walks users through deck construction step-by-step with AI assistance
at each stage: strategy, colors, core cards, support, mana base, sideboard.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.conversation import Conversation
from app.models.meta import MetaSnapshot
from app.services.ai_service import AIService
from app.services.card_service import CardService
from app.services.deck_validator import DeckValidator
from app.schemas.guided_build import (
    GuidedBuildStep,
    STEP_ORDER,
    GuidedBuildStepResponse,
    GuidedBuildCompleteResponse,
    ArchetypeOption,
    ColorOption,
    CardRecommendation,
    CardSlotGroup,
    LandRecommendation,
    SideboardRecommendation,
)

logger = logging.getLogger(__name__)

STEP_TITLES = {
    GuidedBuildStep.STRATEGY: "Choose Your Strategy",
    GuidedBuildStep.COLORS: "Pick Your Colors",
    GuidedBuildStep.CORE: "Select Core Cards",
    GuidedBuildStep.SUPPORT: "Add Support Cards",
    GuidedBuildStep.MANA_BASE: "Build Your Mana Base",
    GuidedBuildStep.SIDEBOARD: "Craft Your Sideboard",
    GuidedBuildStep.REVIEW: "Review & Finalize",
}

STEP_DESCRIPTIONS = {
    GuidedBuildStep.STRATEGY: "What kind of deck do you want to play? Choose an archetype that fits your playstyle.",
    GuidedBuildStep.COLORS: "Which colors best support your strategy? Each combination has unique strengths.",
    GuidedBuildStep.CORE: "These are the engine of your deck - the key threats and synergies that define your gameplan.",
    GuidedBuildStep.SUPPORT: "Fill in removal, card draw, and utility to round out your deck.",
    GuidedBuildStep.MANA_BASE: "A solid mana base is critical. Let's make sure you can cast your spells on curve.",
    GuidedBuildStep.SIDEBOARD: "Prepare for the meta. Your sideboard lets you adapt to different matchups.",
    GuidedBuildStep.REVIEW: "Review your complete deck. Make any final adjustments before saving.",
}


class GuidedBuilder:
    """Step-by-step guided deck building with AI assistance."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.ai_service = AIService(db)
        self.card_service = CardService(db)
        self.validator = DeckValidator(db)

    async def start_session(
        self,
        user_id: Optional[UUID],
        format: str = "standard",
    ) -> GuidedBuildStepResponse:
        """Start a new guided build session. Returns strategy options."""
        # Create conversation to persist session state
        conversation = Conversation(
            user_id=user_id,
            messages=[],
            summary="Guided Build",
        )
        # Store guided build state in current_deck JSONB
        conversation.current_deck = {
            "guided_build": True,
            "format": format,
            "step": GuidedBuildStep.STRATEGY.value,
            "selections": {},
            "main_deck": [],
            "sideboard": [],
        }
        conversation.add_message("system", f"Starting guided deck build for {format} format.")
        self.db.add(conversation)
        await self.db.flush()

        # Get strategy options
        step_data = await self._build_strategy_step(format)

        return GuidedBuildStepResponse(
            session_id=conversation.id,
            current_step=GuidedBuildStep.STRATEGY,
            step_index=0,
            total_steps=len(STEP_ORDER),
            step_title=STEP_TITLES[GuidedBuildStep.STRATEGY],
            step_description=STEP_DESCRIPTIONS[GuidedBuildStep.STRATEGY],
            data=step_data,
            ai_message="Let's build a deck together! First, what kind of strategy appeals to you? "
            "Each archetype has a different playstyle - pick the one that sounds most fun.",
        )

    async def advance_step(
        self,
        session_id: UUID,
        selections: Dict[str, Any],
        user_id: Optional[UUID] = None,
    ) -> GuidedBuildStepResponse:
        """Process current step selections and advance to the next step."""
        conversation = await self._get_session(session_id, user_id)
        state = conversation.current_deck
        current_step = GuidedBuildStep(state["step"])
        current_index = STEP_ORDER.index(current_step)

        # Store selections for current step
        state.setdefault("selections", {})[current_step.value] = selections

        # Process selections based on current step
        await self._process_step_selections(state, current_step, selections)

        # Move to next step
        next_index = current_index + 1
        if next_index >= len(STEP_ORDER):
            # Already at last step
            next_step = STEP_ORDER[-1]
        else:
            next_step = STEP_ORDER[next_index]

        state["step"] = next_step.value
        conversation.current_deck = state
        await self.db.commit()

        # Build next step data
        step_data, ai_message = await self._build_step_data(state, next_step)

        return GuidedBuildStepResponse(
            session_id=session_id,
            current_step=next_step,
            step_index=next_index,
            total_steps=len(STEP_ORDER),
            step_title=STEP_TITLES[next_step],
            step_description=STEP_DESCRIPTIONS[next_step],
            data=step_data,
            ai_message=ai_message,
        )

    async def go_back(
        self,
        session_id: UUID,
        user_id: Optional[UUID] = None,
    ) -> GuidedBuildStepResponse:
        """Go back to the previous step."""
        conversation = await self._get_session(session_id, user_id)
        state = conversation.current_deck
        current_step = GuidedBuildStep(state["step"])
        current_index = STEP_ORDER.index(current_step)

        if current_index <= 0:
            prev_step = STEP_ORDER[0]
            prev_index = 0
        else:
            prev_index = current_index - 1
            prev_step = STEP_ORDER[prev_index]

        state["step"] = prev_step.value
        conversation.current_deck = state
        await self.db.commit()

        step_data, ai_message = await self._build_step_data(state, prev_step)

        return GuidedBuildStepResponse(
            session_id=session_id,
            current_step=prev_step,
            step_index=prev_index,
            total_steps=len(STEP_ORDER),
            step_title=STEP_TITLES[prev_step],
            step_description=STEP_DESCRIPTIONS[prev_step],
            data=step_data,
            ai_message=ai_message,
        )

    async def complete_build(
        self,
        session_id: UUID,
        deck_name: Optional[str],
        user_id: Optional[UUID] = None,
        save: bool = False,
    ) -> GuidedBuildCompleteResponse:
        """Finalize the guided build and optionally save the deck."""
        conversation = await self._get_session(session_id, user_id)
        state = conversation.current_deck

        main_deck = state.get("main_deck", [])
        sideboard = state.get("sideboard", [])
        format = state.get("format", "standard")
        archetype = state.get("selections", {}).get("strategy", {}).get("archetype", "Custom")
        colors = state.get("selections", {}).get("colors", {}).get("colors", [])
        name = deck_name or f"{archetype} Deck"

        # Validate
        validation = await self.validator.validate(main_deck, sideboard, format)
        errors = [e.message for e in validation.errors] if validation.errors else []

        # Generate strategy summary
        strategy_summary = await self._generate_strategy_summary(state)

        deck_id = None
        if save and user_id:
            from app.models.deck import Deck
            deck = Deck(
                id=uuid4(),
                owner_id=user_id,
                name=name,
                format=format,
                archetype=archetype,
                main_deck=main_deck,
                sideboard=sideboard,
                strategy_summary=strategy_summary,
                is_validated=validation.is_valid,
                validation_errors=[e.model_dump(mode='json') for e in validation.errors] if validation.errors else None,
            )
            self.db.add(deck)
            await self.db.commit()
            await self.db.refresh(deck)
            deck_id = deck.id

        return GuidedBuildCompleteResponse(
            session_id=session_id,
            deck_id=deck_id,
            deck_name=name,
            main_deck=main_deck,
            sideboard=sideboard,
            strategy_summary=strategy_summary,
            archetype=archetype,
            colors=colors,
            format=format,
            is_valid=validation.is_valid,
            validation_errors=errors,
        )

    # --- Step builders ---

    async def _build_strategy_step(self, format: str) -> Dict[str, Any]:
        """Build strategy step data with meta-informed archetype options."""
        meta_data = await self._get_meta_context(format)

        # Build archetype options from meta + general archetypes
        archetypes = []

        # Add meta-informed archetypes
        for arch in meta_data.get("archetypes", [])[:5]:
            archetypes.append(ArchetypeOption(
                name=arch["name"],
                description=f"A proven archetype making up {arch['meta_percentage']:.1f}% of the meta.",
                playstyle=self._classify_playstyle(arch["name"]),
                meta_percentage=arch["meta_percentage"],
                example_cards=arch.get("key_cards", [])[:4],
            ))

        # Always include general archetypes
        general_archetypes = [
            ArchetypeOption(
                name="Aggro",
                description="Win fast with efficient creatures and burn. Get under slower decks before they stabilize.",
                playstyle="Fast and aggressive. Deploy threats early and close the game quickly.",
                example_cards=[],
            ),
            ArchetypeOption(
                name="Midrange",
                description="Powerful, versatile threats that can adapt to any matchup. Trade efficiently and grind value.",
                playstyle="Flexible and powerful. Play the best cards at every mana cost.",
                example_cards=[],
            ),
            ArchetypeOption(
                name="Control",
                description="Answer everything. Counter spells, remove threats, and win with overwhelming card advantage.",
                playstyle="Patient and reactive. Survive the early game, dominate the late game.",
                example_cards=[],
            ),
            ArchetypeOption(
                name="Combo",
                description="Assemble a devastating combination of cards that wins the game on the spot.",
                playstyle="Focused and explosive. Find your pieces, protect them, and win.",
                example_cards=[],
            ),
        ]

        # Add general archetypes that aren't already represented
        existing_names = {a.name.lower() for a in archetypes}
        for ga in general_archetypes:
            if ga.name.lower() not in existing_names:
                archetypes.append(ga)

        return {
            "archetypes": [a.model_dump() for a in archetypes],
            "meta_summary": f"Current {format} meta has {len(meta_data.get('archetypes', []))} tracked archetypes."
            if meta_data.get("archetypes") else f"No meta data available for {format} yet.",
            "format": format,
        }

    async def _build_color_step(self, state: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        """Build color options based on chosen archetype."""
        archetype = state.get("selections", {}).get("strategy", {}).get("archetype", "Midrange")
        format = state.get("format", "standard")

        color_options = self._get_color_options_for_archetype(archetype)

        ai_msg = (
            f"Great choice with {archetype}! Now let's pick your colors. "
            f"Each color combination brings different tools to the table. "
            f"I've highlighted the combinations that work best for {archetype} strategies."
        )

        return {
            "options": [c.model_dump() for c in color_options],
            "recommendation": f"For {archetype}, I'd suggest colors that give you access to the tools this strategy needs most.",
            "archetype": archetype,
        }, ai_msg

    async def _build_core_step(self, state: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        """Build core card recommendations - the engine of the deck."""
        archetype = state.get("selections", {}).get("strategy", {}).get("archetype", "Midrange")
        colors = state.get("selections", {}).get("colors", {}).get("colors", [])
        format = state.get("format", "standard")

        slots = []

        if archetype.lower() in ["aggro", "red deck wins", "rdw"]:
            slot_configs = [
                ("1-Drop Threats", "Aggressive one-mana creatures to start the pressure early", 8),
                ("2-Drop Threats", "Efficient two-mana creatures that hit hard", 8),
                ("3-Drop Finishers", "Top-end threats that close out the game", 4),
                ("Burn / Reach", "Direct damage to finish off opponents or remove blockers", 6),
            ]
        elif archetype.lower() in ["control"]:
            slot_configs = [
                ("Counterspells", "Permission spells to deny your opponent's key plays", 6),
                ("Removal", "Efficient answers to creatures and other threats", 6),
                ("Card Advantage", "Draw spells and engines to keep your hand full", 6),
                ("Win Conditions", "Powerful finishers to close the game once you've stabilized", 4),
            ]
        elif archetype.lower() in ["combo"]:
            slot_configs = [
                ("Combo Pieces", "The key cards that form your winning combination", 8),
                ("Enablers / Tutors", "Cards that find or fuel your combo", 6),
                ("Protection", "Ways to protect your combo from disruption", 4),
                ("Interaction", "Removal and counters to survive until you combo off", 6),
            ]
        else:  # Midrange or specific archetypes
            slot_configs = [
                ("Threats", "Powerful creatures and planeswalkers that pressure your opponent", 10),
                ("Removal", "Efficient answers to opposing threats", 6),
                ("Card Advantage", "Ways to keep your hand full and out-resource opponents", 4),
                ("Utility", "Flexible cards that support your gameplan", 4),
            ]

        for slot_name, description, target in slot_configs:
            cards = await self._get_card_recommendations(
                colors=colors,
                role=slot_name.lower(),
                archetype=archetype,
                format=format,
                limit=6,
            )
            slots.append(CardSlotGroup(
                slot_name=slot_name,
                description=description,
                target_count=target,
                recommendations=cards,
            ))

        total_needed = sum(s.target_count for s in slots)
        ai_msg = (
            f"Now for the fun part - picking your cards! I've organized recommendations into "
            f"role-based slots. You need around {total_needed} nonland cards. "
            f"Select the ones that excite you, or adjust quantities as you see fit."
        )

        return {
            "slots": [s.model_dump() for s in slots],
            "strategy_note": f"Building a {' '.join(colors)} {archetype} deck.",
            "cards_needed": total_needed,
        }, ai_msg

    async def _build_support_step(self, state: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        """Build support card recommendations based on what's already chosen."""
        colors = state.get("selections", {}).get("colors", {}).get("colors", [])
        archetype = state.get("selections", {}).get("strategy", {}).get("archetype", "Midrange")
        format = state.get("format", "standard")
        main_deck = state.get("main_deck", [])

        current_count = sum(e.get("quantity", 0) for e in main_deck)
        # Standard: 60 cards, ~24 lands = 36 nonland slots
        target_nonlands = 36
        remaining = max(0, target_nonlands - current_count)

        # Figure out what roles are missing
        slots = []
        existing_names = {e.get("card_name", "").lower() for e in main_deck}

        support_configs = [
            ("Additional Removal", "More answers to deal with the meta's top threats", max(2, remaining // 3)),
            ("Card Selection", "Filtering and card draw to find what you need", max(2, remaining // 4)),
            ("Flexible Slots", "Versatile cards that can fill multiple roles", max(2, remaining - remaining // 3 - remaining // 4)),
        ]

        for slot_name, description, target in support_configs:
            cards = await self._get_card_recommendations(
                colors=colors,
                role=slot_name.lower().replace("additional ", ""),
                archetype=archetype,
                format=format,
                limit=5,
                exclude=existing_names,
            )
            slots.append(CardSlotGroup(
                slot_name=slot_name,
                description=description,
                target_count=target,
                recommendations=cards,
            ))

        ai_msg = (
            f"Your core is looking solid with {current_count} cards! "
            f"You need about {remaining} more nonland cards to round things out. "
            f"Let's fill in the gaps with support cards."
        )

        return {
            "slots": [s.model_dump() for s in slots],
            "current_deck_size": current_count,
            "remaining_nonland_slots": remaining,
        }, ai_msg

    async def _build_mana_base_step(self, state: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        """Build mana base recommendations."""
        colors = state.get("selections", {}).get("colors", {}).get("colors", [])
        format = state.get("format", "standard")
        main_deck = state.get("main_deck", [])

        nonland_count = sum(e.get("quantity", 0) for e in main_deck)
        total_lands = max(20, 60 - nonland_count)

        # Count color requirements from current deck
        color_reqs = self._estimate_color_requirements(main_deck, colors)

        lands = await self._get_land_recommendations(
            colors=colors,
            format=format,
            total_lands=total_lands,
            color_reqs=color_reqs,
        )

        ai_msg = (
            f"Time for the mana base! With {nonland_count} nonland cards, "
            f"you'll want {total_lands} lands. I've balanced dual lands and basics "
            f"to make sure you can cast everything on curve."
        )

        return {
            "lands": [l.model_dump() for l in lands],
            "total_lands": total_lands,
            "color_requirements": color_reqs,
            "mana_curve_note": f"Targeting {total_lands} lands for a {nonland_count}-spell deck.",
        }, ai_msg

    async def _build_sideboard_step(self, state: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        """Build sideboard recommendations based on meta."""
        colors = state.get("selections", {}).get("colors", {}).get("colors", [])
        archetype = state.get("selections", {}).get("strategy", {}).get("archetype", "Midrange")
        format = state.get("format", "standard")

        meta_data = await self._get_meta_context(format)
        meta_matchups = meta_data.get("archetypes", [])[:5]

        recommendations = await self._get_sideboard_recommendations(
            colors=colors,
            archetype=archetype,
            format=format,
            meta_matchups=meta_matchups,
        )

        ai_msg = (
            f"Your sideboard is your secret weapon. These 15 cards let you adapt "
            f"to different matchups after game 1. I've picked cards that shore up "
            f"your {archetype} strategy against the most popular decks."
        )

        return {
            "recommendations": [r.model_dump() for r in recommendations],
            "meta_matchups": meta_matchups,
            "sideboard_strategy": f"Sideboard plan for {archetype} in the current {format} meta.",
        }, ai_msg

    async def _build_review_step(self, state: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        """Build the final review step."""
        main_deck = state.get("main_deck", [])
        sideboard = state.get("sideboard", [])
        archetype = state.get("selections", {}).get("strategy", {}).get("archetype", "Custom")
        colors = state.get("selections", {}).get("colors", {}).get("colors", [])
        format = state.get("format", "standard")

        validation = await self.validator.validate(main_deck, sideboard, format)
        errors = [e.message for e in validation.errors] if validation.errors else []

        main_count = sum(e.get("quantity", 0) for e in main_deck)
        sb_count = sum(e.get("quantity", 0) for e in sideboard)

        strengths, weaknesses = self._analyze_deck_profile(main_deck, archetype)

        ai_msg = (
            f"Here's your complete {archetype} deck! "
            f"{main_count} main deck cards, {sb_count} sideboard cards. "
        )
        if errors:
            ai_msg += f"There are {len(errors)} validation issue(s) to address. "
        else:
            ai_msg += "Everything looks valid and ready to play! "
        ai_msg += "You can go back to any step to make changes, or save your deck."

        return {
            "deck_name": f"{' '.join(colors)} {archetype}" if colors else archetype,
            "archetype": archetype,
            "colors": colors,
            "strategy_summary": f"A {format} {archetype} deck built around {', '.join(colors)} cards." if colors else f"A {format} {archetype} deck.",
            "main_deck": main_deck,
            "sideboard": sideboard,
            "main_deck_count": main_count,
            "sideboard_count": sb_count,
            "validation_errors": errors,
            "strengths": strengths,
            "weaknesses": weaknesses,
        }, ai_msg

    async def _build_step_data(
        self,
        state: Dict[str, Any],
        step: GuidedBuildStep,
    ) -> tuple[Dict[str, Any], str]:
        """Route to the correct step builder."""
        if step == GuidedBuildStep.STRATEGY:
            data = await self._build_strategy_step(state.get("format", "standard"))
            return data, "What strategy appeals to you?"
        elif step == GuidedBuildStep.COLORS:
            return await self._build_color_step(state)
        elif step == GuidedBuildStep.CORE:
            return await self._build_core_step(state)
        elif step == GuidedBuildStep.SUPPORT:
            return await self._build_support_step(state)
        elif step == GuidedBuildStep.MANA_BASE:
            return await self._build_mana_base_step(state)
        elif step == GuidedBuildStep.SIDEBOARD:
            return await self._build_sideboard_step(state)
        elif step == GuidedBuildStep.REVIEW:
            return await self._build_review_step(state)

        return {}, ""

    async def _process_step_selections(
        self,
        state: Dict[str, Any],
        step: GuidedBuildStep,
        selections: Dict[str, Any],
    ) -> None:
        """Process and apply user selections to the session state."""
        if step == GuidedBuildStep.CORE:
            # User selected core cards - add to main_deck
            cards = selections.get("cards", [])
            state["main_deck"] = cards
        elif step == GuidedBuildStep.SUPPORT:
            # User selected support cards - merge with existing main_deck
            cards = selections.get("cards", [])
            existing = state.get("main_deck", [])
            existing_names = {e.get("card_name") for e in existing}
            for card in cards:
                if card.get("card_name") not in existing_names:
                    existing.append(card)
                else:
                    # Update quantity
                    for e in existing:
                        if e.get("card_name") == card.get("card_name"):
                            e["quantity"] = card.get("quantity", e.get("quantity", 1))
            state["main_deck"] = existing
        elif step == GuidedBuildStep.MANA_BASE:
            # User selected lands - add to main_deck
            lands = selections.get("lands", [])
            nonlands = [e for e in state.get("main_deck", []) if not self._is_land(e)]
            state["main_deck"] = nonlands + lands
        elif step == GuidedBuildStep.SIDEBOARD:
            # User selected sideboard cards
            state["sideboard"] = selections.get("cards", [])

    # --- Helper methods ---

    async def _get_session(self, session_id: UUID, user_id: Optional[UUID]) -> Conversation:
        """Get and validate a guided build session."""
        result = await self.db.execute(
            select(Conversation).where(Conversation.id == session_id)
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise ValueError("Guided build session not found")
        if not conversation.current_deck or not conversation.current_deck.get("guided_build"):
            raise ValueError("Not a guided build session")
        return conversation

    async def _get_meta_context(self, format: str) -> Dict[str, Any]:
        """Get current meta information."""
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

    async def _get_card_recommendations(
        self,
        colors: List[str],
        role: str,
        archetype: str,
        format: str,
        limit: int = 6,
        exclude: set = None,
    ) -> List[CardRecommendation]:
        """Get card recommendations for a specific role."""
        exclude = exclude or set()

        # Map role names to card type hints for searching
        type_hints = {
            "threats": "creature",
            "1-drop threats": "creature",
            "2-drop threats": "creature",
            "3-drop finishers": "creature",
            "removal": None,
            "counterspells": "instant",
            "card advantage": None,
            "card selection": None,
            "win conditions": None,
            "combo pieces": None,
            "enablers / tutors": None,
            "protection": "instant",
            "interaction": None,
            "burn / reach": None,
            "utility": None,
            "flexible slots": None,
        }

        card_type = type_hints.get(role.lower())

        # Search for relevant cards
        try:
            # Use semantic search for better results
            query = f"{role} cards for {archetype} deck in {' '.join(colors)} colors"
            cards = await self.card_service.semantic_search(
                query=query,
                colors=colors if colors else None,
                format=format,
                limit=limit * 2,
            )
        except Exception:
            # Fallback to basic search
            cards = await self.card_service.search(
                colors=colors if colors else None,
                card_type=card_type,
                standard_only=(format == "standard"),
                format=format,
                limit=limit * 2,
            )

        recommendations = []
        for card in cards:
            if card.name.lower() in exclude:
                continue
            if len(recommendations) >= limit:
                break

            recommendations.append(CardRecommendation(
                card_name=card.name,
                card_id=str(card.id),
                quantity=self._suggest_quantity(card, role),
                role=role,
                reasoning=self._generate_card_reasoning(card, role, archetype),
                image_uri=card.image_uri,
                mana_cost=card.mana_cost,
                type_line=card.type_line,
            ))

        return recommendations

    async def _get_land_recommendations(
        self,
        colors: List[str],
        format: str,
        total_lands: int,
        color_reqs: Dict[str, int],
    ) -> List[LandRecommendation]:
        """Get land recommendations for the mana base."""
        lands = []

        if len(colors) <= 1:
            # Mono color - mostly basics
            color = colors[0] if colors else "W"
            basic = self._color_to_basic(color)
            lands.append(LandRecommendation(
                card_name=basic,
                quantity=total_lands,
                category="basic",
                reasoning=f"Mono-color deck runs all basics for consistency.",
            ))
            return lands

        # Multi-color: search for dual lands
        try:
            dual_cards = await self.card_service.search(
                colors=colors,
                card_type="land",
                standard_only=(format == "standard"),
                limit=20,
            )
        except Exception:
            dual_cards = []

        # Categorize and add dual lands
        dual_count = 0
        max_duals = min(total_lands - len(colors) * 2, 16)  # Leave room for basics

        for card in dual_cards:
            if dual_count >= max_duals:
                break
            if not card.name or "basic" in (card.type_line or "").lower():
                continue

            qty = 4 if dual_count + 4 <= max_duals else max_duals - dual_count
            if qty <= 0:
                break

            category = "dual"
            if "fetch" in (card.oracle_text or "").lower() or "search" in (card.oracle_text or "").lower():
                category = "fetch"
            elif any(kw in (card.oracle_text or "").lower() for kw in ["channel", "ability", "counter", "draw"]):
                category = "utility"

            lands.append(LandRecommendation(
                card_name=card.name,
                card_id=str(card.id),
                quantity=qty,
                category=category,
                reasoning=f"Provides color fixing for your {'/'.join(colors)} mana base.",
                image_uri=card.image_uri,
            ))
            dual_count += qty

        # Fill remaining with basics
        remaining = total_lands - dual_count
        if remaining > 0 and colors:
            per_color = remaining // len(colors)
            extra = remaining % len(colors)
            for i, color in enumerate(colors):
                basic = self._color_to_basic(color)
                qty = per_color + (1 if i < extra else 0)
                if qty > 0:
                    lands.append(LandRecommendation(
                        card_name=basic,
                        quantity=qty,
                        category="basic",
                        reasoning=f"Basic land for {color} mana.",
                    ))

        return lands

    async def _get_sideboard_recommendations(
        self,
        colors: List[str],
        archetype: str,
        format: str,
        meta_matchups: List[Dict[str, Any]],
    ) -> List[SideboardRecommendation]:
        """Get sideboard recommendations based on meta."""
        recommendations = []
        cards_used = set()

        # General sideboard categories
        sb_roles = [
            ("graveyard hate", ["graveyard", "reanimator", "dredge"]),
            ("artifact/enchantment removal", ["artifacts", "enchantments"]),
            ("extra removal", ["aggro", "creatures"]),
            ("counterspells", ["combo", "control"]),
            ("board wipes", ["aggro", "tokens", "go-wide"]),
        ]

        for role, matchups in sb_roles:
            try:
                query = f"sideboard {role} for {format}"
                cards = await self.card_service.semantic_search(
                    query=query,
                    colors=colors if colors else None,
                    format=format,
                    limit=4,
                )
            except Exception:
                cards = []

            for card in cards:
                if card.name.lower() in cards_used:
                    continue
                if len(recommendations) >= 15:
                    break
                cards_used.add(card.name.lower())
                recommendations.append(SideboardRecommendation(
                    card_name=card.name,
                    card_id=str(card.id),
                    quantity=min(3, 15 - sum(r.quantity for r in recommendations)),
                    target_matchups=matchups,
                    reasoning=f"Effective {role} for post-board games.",
                    image_uri=card.image_uri,
                ))

            if len(recommendations) >= 15:
                break

        return recommendations[:15]

    async def _generate_strategy_summary(self, state: Dict[str, Any]) -> str:
        """Generate a strategy summary for the completed deck."""
        archetype = state.get("selections", {}).get("strategy", {}).get("archetype", "Custom")
        colors = state.get("selections", {}).get("colors", {}).get("colors", [])
        main_deck = state.get("main_deck", [])
        card_count = sum(e.get("quantity", 0) for e in main_deck)
        card_names = [e.get("card_name", "") for e in main_deck[:10]]

        return (
            f"This {'/'.join(colors)} {archetype} deck aims to "
            f"{'pressure opponents quickly' if archetype.lower() == 'aggro' else 'control the game' if archetype.lower() == 'control' else 'outvalue opponents'} "
            f"with {card_count} carefully selected cards. "
            f"Key cards include {', '.join(card_names[:3])}."
        )

    def _classify_playstyle(self, archetype_name: str) -> str:
        """Classify an archetype's playstyle."""
        name_lower = archetype_name.lower()
        if any(kw in name_lower for kw in ["aggro", "red deck", "burn", "weenie", "sligh"]):
            return "Aggressive - deploy threats and attack relentlessly."
        elif any(kw in name_lower for kw in ["control", "azorius", "esper"]):
            return "Reactive - answer threats and win with inevitability."
        elif any(kw in name_lower for kw in ["combo", "storm", "reanimate"]):
            return "Combo - assemble pieces for a powerful finish."
        elif any(kw in name_lower for kw in ["tempo"]):
            return "Tempo - efficient threats backed by disruption."
        return "Midrange - play powerful cards at every point on the curve."

    def _get_color_options_for_archetype(self, archetype: str) -> List[ColorOption]:
        """Get recommended color combinations for an archetype."""
        arch_lower = archetype.lower()

        # All two-color combos with MTG names
        all_pairs = [
            (["W", "U"], "Azorius", "Control and flying creatures",
             ["Strong removal", "Counterspells", "Card draw"], ["Slow starts", "Vulnerable to aggro"]),
            (["U", "B"], "Dimir", "Disruption and card advantage",
             ["Hand disruption", "Card draw", "Removal"], ["Few board wipes", "Slow threats"]),
            (["B", "R"], "Rakdos", "Aggressive disruption",
             ["Efficient removal", "Fast threats", "Burn"], ["No enchantment removal", "Limited card draw"]),
            (["R", "G"], "Gruul", "Big creatures and ramp",
             ["Large threats", "Ramp", "Reach"], ["No counterspells", "Weak removal"]),
            (["G", "W"], "Selesnya", "Tokens and going wide",
             ["Board presence", "Lifegain", "Anthems"], ["No card draw", "Weak to wraths"]),
            (["W", "B"], "Orzhov", "Lifegain and grinding",
             ["Best removal suite", "Lifegain", "Recursion"], ["No counterspells", "Slow"]),
            (["U", "R"], "Izzet", "Spells and tempo",
             ["Counterspells", "Burn", "Card draw"], ["Weak to resolved threats", "No lifegain"]),
            (["B", "G"], "Golgari", "Graveyard value",
             ["Removal", "Recursion", "Value creatures"], ["No counterspells", "Slow"]),
            (["R", "W"], "Boros", "Aggressive and fast",
             ["Fast creatures", "Burn", "Removal"], ["No card draw", "Runs out of gas"]),
            (["G", "U"], "Simic", "Ramp and card advantage",
             ["Ramp", "Card draw", "Big threats"], ["Weak removal", "No lifegain"]),
        ]

        # Also include mono colors
        monos = [
            (["W"], "Mono White", "Efficient creatures and removal",
             ["Low curve", "Good removal", "Lifegain"], ["Limited card draw"]),
            (["U"], "Mono Blue", "Counterspells and tempo",
             ["Counterspells", "Card draw", "Tempo"], ["Weak to resolved threats"]),
            (["B"], "Mono Black", "Disruption and recursion",
             ["Hand disruption", "Removal", "Recursion"], ["No artifact removal"]),
            (["R"], "Mono Red", "Burn and aggression",
             ["Speed", "Direct damage", "Low curve"], ["Runs out of gas"]),
            (["G"], "Mono Green", "Big creatures and ramp",
             ["Large creatures", "Ramp", "Fight effects"], ["No removal spells"]),
        ]

        options = []
        for colors, name, desc, strengths, weaknesses in monos + all_pairs:
            options.append(ColorOption(
                colors=colors,
                name=name,
                description=desc,
                strengths=strengths,
                weaknesses=weaknesses,
            ))

        return options

    def _suggest_quantity(self, card, role: str) -> int:
        """Suggest how many copies of a card to include."""
        cmc = card.cmc or 0
        if cmc <= 2:
            return 4
        elif cmc <= 4:
            return 3
        else:
            return 2

    def _generate_card_reasoning(self, card, role: str, archetype: str) -> str:
        """Generate a brief reasoning for why this card fits."""
        parts = []
        if card.mana_cost:
            parts.append(f"Costs {card.mana_cost}")
        if card.type_line:
            parts.append(card.type_line)
        parts.append(f"fills the {role} role in your {archetype} strategy")
        return ". ".join(parts) + "."

    def _estimate_color_requirements(
        self,
        main_deck: List[Dict[str, Any]],
        colors: List[str],
    ) -> Dict[str, int]:
        """Estimate color pip requirements from the deck."""
        reqs = {c: 0 for c in colors}
        color_map = {"W": "W", "U": "U", "B": "B", "R": "R", "G": "G"}

        for entry in main_deck:
            card_data = entry.get("card", {})
            mana_cost = card_data.get("mana_cost", "") if card_data else ""
            qty = entry.get("quantity", 1)
            for color in colors:
                pip = f"{{{color}}}"
                count = mana_cost.count(pip) if mana_cost else 0
                reqs[color] = reqs.get(color, 0) + (count * qty)

        return reqs

    def _color_to_basic(self, color: str) -> str:
        """Convert color letter to basic land name."""
        return {
            "W": "Plains",
            "U": "Island",
            "B": "Swamp",
            "R": "Mountain",
            "G": "Forest",
        }.get(color, "Plains")

    def _is_land(self, entry: Dict[str, Any]) -> bool:
        """Check if a deck entry is a land."""
        card_data = entry.get("card", {})
        if card_data and card_data.get("type_line"):
            return "land" in card_data["type_line"].lower()
        name = entry.get("card_name", "").lower()
        return name in ["plains", "island", "swamp", "mountain", "forest"] or "land" in name

    def _analyze_deck_profile(
        self,
        main_deck: List[Dict[str, Any]],
        archetype: str,
    ) -> tuple[List[str], List[str]]:
        """Analyze deck strengths and weaknesses."""
        card_count = sum(e.get("quantity", 0) for e in main_deck)
        strengths = []
        weaknesses = []

        if card_count >= 58:
            strengths.append("Deck size is close to the standard 60 cards")
        else:
            weaknesses.append(f"Deck only has {card_count} cards (need 60)")

        if archetype.lower() in ["aggro", "red deck wins"]:
            strengths.append("Aggressive strategy can win before opponents set up")
            weaknesses.append("May struggle in longer games")
        elif archetype.lower() == "control":
            strengths.append("Strong late game with lots of answers")
            weaknesses.append("Can be overwhelmed by fast aggro starts")
        elif archetype.lower() == "midrange":
            strengths.append("Flexible gameplan that adapts to matchups")

        return strengths, weaknesses
