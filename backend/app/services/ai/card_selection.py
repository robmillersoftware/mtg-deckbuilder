"""Card selection and synergy utilities."""

import logging
import re
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text, or_

from app.services.card_service import get_format_view

logger = logging.getLogger(__name__)


# Mechanical synergy patterns: each theme maps to a list of (pattern, weight) tuples.
# Patterns are matched against oracle_text (lowercased). Weight 1.0 = core synergy,
# 0.5 = supporting synergy, 0.25 = tangential.
SYNERGY_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
    "graveyard": [
        # Core: directly interact with graveyard
        ("return.*from.*graveyard", 1.0),
        ("put.*into.*graveyard", 1.0),
        ("from your graveyard", 1.0),
        ("cards in your graveyard", 1.0),
        ("mill", 1.0),
        ("self-mill", 1.0),
        ("dredge", 1.0),
        # Recursion keywords
        ("flashback", 1.0),
        ("unearth", 1.0),
        ("escape", 1.0),
        ("embalm", 1.0),
        ("eternalize", 1.0),
        ("disturb", 1.0),
        ("aftermath", 0.75),
        ("retrace", 0.75),
        # Death/sacrifice synergy (tangential for graveyard; core in "sacrifice" theme)
        ("when.*dies", 0.25),
        ("whenever.*dies", 0.25),
        ("sacrifice a creature", 0.25),
        ("sacrifice a permanent", 0.25),
        ("sacrifice another", 0.25),
        # Reanimation
        ("return.*creature.*from.*graveyard.*to the battlefield", 1.0),
        ("reanimate", 1.0),
        ("exile.*from.*graveyard", 0.5),
        # Self-enabling
        ("discard a card", 0.5),
        ("discard.*card.*draw", 0.5),
        ("put the rest into your graveyard", 0.75),
        ("into your graveyard", 0.75),
        # Delve / cost reduction from graveyard
        ("delve", 0.75),
        ("exile.*cards? from your graveyard", 0.5),
    ],
    "sacrifice": [
        ("sacrifice a creature", 1.0),
        ("sacrifice a permanent", 1.0),
        ("sacrifice another", 1.0),
        ("sacrifice a token", 0.75),
        ("whenever you sacrifice", 1.0),
        ("whenever.*is sacrificed", 1.0),
        ("when.*dies", 1.0),
        ("whenever.*dies", 1.0),
        ("whenever another creature", 0.75),
        ("create.*token", 0.75),
        ("blood token", 0.75),
        ("food token", 0.75),
        ("treasure token", 0.5),
        ("death trigger", 1.0),
        ("return.*from.*graveyard", 0.5),
        ("when.*leaves the battlefield", 0.5),
    ],
    "tokens": [
        ("create.*token", 1.0),
        ("creature token", 1.0),
        ("populate", 1.0),
        ("whenever.*token", 0.75),
        ("number of creatures you control", 0.75),
        ("each creature you control", 0.75),
        ("creatures you control get", 1.0),
        ("go wide", 1.0),
        ("anthem", 0.75),
        ("convoke", 0.5),
        ("for each creature", 0.75),
        ("sacrifice a token", 0.5),
        ("token.*enters the battlefield", 0.75),
    ],
    "+1/+1 counters": [
        ("\\+1/\\+1 counter", 1.0),
        ("proliferate", 1.0),
        ("modular", 1.0),
        ("adapt", 0.75),
        ("evolve", 0.75),
        ("enters.*with.*counter", 0.75),
        ("put.*counter.*on", 0.75),
        ("remove.*counter.*from", 0.5),
        ("counter.*among", 0.75),
        ("whenever.*counter", 0.5),
        ("double.*counter", 1.0),
    ],
    "enchantment": [
        ("enchantment", 0.5),
        ("aura", 0.75),
        ("constellation", 1.0),
        ("enchantress", 1.0),
        ("whenever.*enchantment.*enters", 1.0),
        ("whenever you cast an enchantment", 1.0),
        ("enchantment.*you control", 0.75),
        ("enchant creature", 0.5),
        ("bestow", 0.75),
    ],
    "artifact": [
        ("artifact", 0.5),
        ("whenever.*artifact.*enters", 1.0),
        ("affinity for artifacts", 1.0),
        ("metalcraft", 1.0),
        ("improvise", 0.75),
        ("artifact.*you control", 0.75),
        ("sacrifice an artifact", 0.75),
        ("whenever you cast an artifact", 1.0),
        ("treasure token", 0.5),
        ("equipment", 0.5),
    ],
    "spellslinger": [
        ("whenever you cast.*instant", 1.0),
        ("whenever you cast.*sorcery", 1.0),
        ("whenever you cast a noncreature", 0.75),
        ("magecraft", 1.0),
        ("prowess", 0.75),
        ("storm", 1.0),
        ("copy.*spell", 0.75),
        ("instant.*sorcery.*in your graveyard", 0.75),
        ("flashback", 0.5),
        ("cost.*less to cast", 0.5),
    ],
    "tribal": [
        # Generic tribal patterns - works with any creature type
        ("creatures you control get", 0.75),
        ("other.*you control get", 0.75),
        ("lord", 0.75),
        ("whenever another.*enters the battlefield", 0.5),
    ],
}

# Map strategy keywords to their synergy axes for richer detection
STRATEGY_THEME_MAP: Dict[str, List[str]] = {
    "graveyard": ["graveyard", "sacrifice"],
    "reanimator": ["graveyard"],
    "aristocrats": ["sacrifice", "graveyard"],
    "sacrifice": ["sacrifice", "graveyard"],
    "value engine": ["graveyard", "sacrifice"],
    "tokens": ["tokens"],
    "go wide": ["tokens"],
    "counters": ["+1/+1 counters"],
    "enchantress": ["enchantment"],
    "artifacts": ["artifact"],
    "spells": ["spellslinger"],
    "storm": ["spellslinger"],
}


class CardSelectionMixin:
    """Mixin providing card selection and synergy methods."""

    db: AsyncSession

    async def _semantic_card_search(
        self,
        strategy: str,
        colors: List[str],
        limit: int = 20,
        format: str = "standard"
    ) -> List[Dict[str, Any]]:
        """Search for cards that match the strategy semantically."""
        from app.models.card import Card
        from app.services.card_service import get_format_legality_condition

        keywords = strategy.lower().split()
        results = []
        seen = set()

        for keyword in keywords:
            if len(keyword) < 3:
                continue

            query = select(Card).where(
                Card.oracle_text.ilike(f"%{keyword}%") |
                Card.type_line.ilike(f"%{keyword}%") |
                Card.name.ilike(f"%{keyword}%"),
                get_format_legality_condition(format)
            ).limit(10)

            result = await self.db.execute(query)
            cards = result.scalars().all()

            for card in cards:
                if card.name not in seen:
                    seen.add(card.name)
                    results.append({
                        "name": card.name,
                        "mana_cost": card.mana_cost,
                        "type_line": card.type_line,
                        "oracle_text": card.oracle_text,
                    })

            if len(results) >= limit:
                break

        return results[:limit]

    async def _detect_card_themes(
        self, card_names: List[str], strategy: str = ""
    ) -> List[str]:
        """Detect themes from a list of card names and optional strategy text.

        Returns themes sorted by relevance (most pattern matches first).
        """
        from app.models.card import Card

        if not card_names and not strategy:
            return []

        # Track theme scores for ranking
        theme_scores: Dict[str, float] = defaultdict(float)

        # Detect themes from strategy text first
        strategy_lower = strategy.lower()
        for keyword, mapped_themes in STRATEGY_THEME_MAP.items():
            if keyword in strategy_lower:
                for theme in mapped_themes:
                    theme_scores[theme] += 2.0  # Strategy match is high signal

        if card_names:
            query = select(Card).where(
                func.lower(Card.name).in_([n.lower() for n in card_names])
            )
            result = await self.db.execute(query)
            cards = result.scalars().all()

            for card in cards:
                oracle = (card.oracle_text or "").lower()
                type_line = (card.type_line or "").lower()
                keywords = [k.lower() for k in (card.keywords or [])]

                # Creature types (for tribal detection)
                for creature_type in [
                    "angel", "demon", "dragon", "elf", "goblin", "human",
                    "merfolk", "vampire", "zombie", "warrior", "wizard",
                    "knight", "spirit", "beast", "elemental", "faerie",
                    "rat", "cat", "dog", "dinosaur", "pirate",
                ]:
                    if creature_type in type_line:
                        theme_scores[creature_type] += 1.0

                # Score each synergy axis by checking how many patterns match
                for theme, patterns in SYNERGY_PATTERNS.items():
                    for pattern, weight in patterns:
                        if re.search(pattern, oracle):
                            theme_scores[theme] += weight

                # Keyword-based detection (from Scryfall keywords field)
                keyword_theme_map = {
                    "flashback": "graveyard",
                    "unearth": "graveyard",
                    "escape": "graveyard",
                    "embalm": "graveyard",
                    "eternalize": "graveyard",
                    "disturb": "graveyard",
                    "dredge": "graveyard",
                    "delve": "graveyard",
                    "retrace": "graveyard",
                    "exploit": "sacrifice",
                    "devour": "sacrifice",
                    "proliferate": "+1/+1 counters",
                    "modular": "+1/+1 counters",
                    "adapt": "+1/+1 counters",
                    "evolve": "+1/+1 counters",
                    "convoke": "tokens",
                    "populate": "tokens",
                    "constellation": "enchantment",
                    "bestow": "enchantment",
                    "affinity": "artifact",
                    "improvise": "artifact",
                    "metalcraft": "artifact",
                    "prowess": "spellslinger",
                    "magecraft": "spellslinger",
                    "storm": "spellslinger",
                }
                for kw in keywords:
                    if kw in keyword_theme_map:
                        theme_scores[keyword_theme_map[kw]] += 1.5

                # Type-line based detection
                if "artifact" in type_line and "creature" not in type_line:
                    theme_scores["artifact"] += 0.5
                if "enchantment" in type_line:
                    theme_scores["enchantment"] += 0.5

        # Return themes sorted by score, filtering out low-signal ones
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        # Only include themes with meaningful signal (score >= 0.75)
        return [t for t, s in sorted_themes if s >= 0.75][:7]

    async def _get_synergy_cards(
        self,
        themes: List[str],
        colors: List[str],
        limit: int = 30,
        format: str = "standard"
    ) -> List[Dict[str, Any]]:
        """Get cards that synergize with the given themes using pattern-based scoring."""
        if not themes:
            return []

        colors_upper = [c.upper() for c in colors]

        # Collect all search terms from theme patterns
        search_terms: Set[str] = set()
        for theme in themes:
            # Add the theme itself
            search_terms.add(theme)
            # Add key distinguishing words from patterns for this theme
            if theme in SYNERGY_PATTERNS:
                for pattern, weight in SYNERGY_PATTERNS[theme]:
                    if weight >= 0.75:
                        # Extract a searchable keyword from the pattern
                        # Use the longest literal word (no regex chars)
                        words = re.sub(r'[\\.*+?^${}()|[\]]', ' ', pattern).split()
                        for word in words:
                            if len(word) >= 4:
                                search_terms.add(word)

        if not search_terms:
            return []

        # Query cards from the format-specific materialized view
        # This enforces legality structurally and DISTINCT ON deduplicates printings
        view_name = get_format_view(format)
        term_list = list(search_terms)[:15]  # Cap to avoid overly broad queries
        or_clauses = []
        params: Dict[str, Any] = {}
        for i, term in enumerate(term_list):
            or_clauses.append(f"LOWER(c.oracle_text) LIKE :term_{i}")
            or_clauses.append(f"LOWER(c.type_line) LIKE :term_{i}")
            params[f"term_{i}"] = f"%{term.lower()}%"

        or_sql = " OR ".join(or_clauses)
        query_sql = f"""
            SELECT DISTINCT ON (c.name)
                c.name, c.mana_cost, c.type_line, c.oracle_text,
                c.colors, c.keywords
            FROM {view_name} c
            WHERE ({or_sql})
            ORDER BY c.name
            LIMIT 200
        """

        result = await self.db.execute(text(query_sql), params)
        rows = result.all()

        # Score and filter
        scored_cards = []
        for row in rows:
            card_colors = row.colors or []
            is_colorless = len(card_colors) == 0
            is_land = "land" in (row.type_line or "").lower()
            matches_colors = all(c in colors_upper for c in card_colors)

            if not (is_colorless or is_land or matches_colors):
                continue

            # Build a lightweight object for the scorer
            card_obj = type("Card", (), {
                "oracle_text": row.oracle_text,
                "type_line": row.type_line,
                "keywords": row.keywords or [],
            })()
            score = self._score_card_synergy(card_obj, themes)
            if score >= 1.5:
                scored_cards.append({
                    "name": row.name,
                    "mana_cost": row.mana_cost,
                    "type_line": row.type_line,
                    "oracle_text": row.oracle_text,
                    "synergy_score": score,
                })

        # Sort by synergy score descending
        scored_cards.sort(key=lambda x: x["synergy_score"], reverse=True)
        logger.info(
            f"Found {len(scored_cards)} synergy cards for themes {themes} "
            f"(top score: {scored_cards[0]['synergy_score']:.2f})" if scored_cards else
            f"Found 0 synergy cards for themes {themes}"
        )
        return scored_cards[:limit]

    def _score_card_synergy(self, card: Any, themes: List[str]) -> float:
        """Score a single card's mechanical synergy with the given themes.

        Returns a float score where higher = more synergistic.
        A score of 0 means no relevant synergy detected.
        """
        oracle = (card.oracle_text or "").lower()
        type_line = (card.type_line or "").lower()
        keywords = [k.lower() for k in (card.keywords or [])]

        total_score = 0.0

        for theme in themes:
            if theme not in SYNERGY_PATTERNS:
                # For themes without defined patterns (e.g., creature types),
                # use simple text matching
                if theme.lower() in oracle or theme.lower() in type_line:
                    total_score += 0.5
                continue

            theme_score = 0.0
            matched_patterns = 0

            for pattern, weight in SYNERGY_PATTERNS[theme]:
                if re.search(pattern, oracle):
                    theme_score += weight
                    matched_patterns += 1

            # Bonus for cards that match multiple patterns (multi-axis synergy)
            if matched_patterns >= 3:
                theme_score *= 1.3
            elif matched_patterns >= 2:
                theme_score *= 1.15

            total_score += theme_score

        # Small bonus for cards with relevant Scryfall keywords
        keyword_bonus_map = {
            "flashback": ["graveyard"],
            "unearth": ["graveyard"],
            "escape": ["graveyard"],
            "embalm": ["graveyard"],
            "eternalize": ["graveyard"],
            "disturb": ["graveyard"],
            "dredge": ["graveyard"],
            "delve": ["graveyard"],
            "exploit": ["sacrifice"],
            "populate": ["tokens"],
            "proliferate": ["+1/+1 counters"],
            "constellation": ["enchantment"],
            "affinity": ["artifact"],
            "prowess": ["spellslinger"],
            "magecraft": ["spellslinger"],
        }
        for kw in keywords:
            if kw in keyword_bonus_map:
                for bonus_theme in keyword_bonus_map[kw]:
                    if bonus_theme in themes:
                        total_score += 0.5

        return total_score

    async def _get_mechanical_synergy_cards(
        self,
        build_around_cards: List[str],
        colors: List[str],
        strategy: str = "",
        limit: int = 30,
        format: str = "standard",
        exclude_cards: Set[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find cards with strong mechanical synergy to the build-around cards.

        This is the primary improvement over pure co-occurrence: it detects
        synergy by analyzing card mechanics rather than just tournament data.

        Returns cards sorted by synergy_score descending.
        """
        exclude = exclude_cards or set()

        # Step 1: Detect themes from the build-around cards + strategy
        themes = await self._detect_card_themes(build_around_cards, strategy)
        if not themes:
            logger.info(f"No themes detected for {build_around_cards}")
            return []

        logger.info(f"Detected themes for {build_around_cards}: {themes}")

        # Step 2: Get synergy cards using pattern-based search
        synergy_cards = await self._get_synergy_cards(
            themes=themes,
            colors=colors,
            limit=limit * 2,  # Fetch extra to allow filtering
            format=format,
        )

        # Step 3: Filter out excluded cards and build-around cards themselves
        build_around_lower = {n.lower() for n in build_around_cards}
        exclude_lower = {n.lower() for n in exclude}
        filtered = [
            c for c in synergy_cards
            if c["name"].lower() not in build_around_lower
            and c["name"].lower() not in exclude_lower
        ]

        return filtered[:limit]
