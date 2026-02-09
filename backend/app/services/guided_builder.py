"""
Deck analysis service for the guided builder.

Provides real-time analysis of an in-progress deck: mana curve,
color distribution, role coverage gaps, and card suggestions.
"""

from typing import Optional, List, Dict, Any
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.services.card_service import CardService, get_format_view, FORMAT_LEGALITY_MAP

logger = logging.getLogger(__name__)

# Map user-facing role names (from Claude tool calls) to system role names in card_roles table
ROLE_MAP: Dict[str, List[str]] = {
    "threats": ["threat_cheap", "threat_midrange", "threat_finisher"],
    "creatures": ["threat_cheap", "threat_midrange", "threat_finisher"],
    "removal": ["removal_targeted", "removal_mass", "removal_artifact_enchantment"],
    "card advantage": ["card_draw", "card_selection"],
    "card draw": ["card_draw", "card_selection"],
    "counterspells": ["counterspell"],
    "protection": ["protection"],
    "ramp": ["ramp"],
    "burn": ["burn"],
    "recursion": ["recursion"],
    "finishers": ["threat_finisher"],
    "interaction": ["removal_targeted", "counterspell"],
    "discard": ["discard"],
    "lifegain": ["lifegain"],
    "graveyard hate": ["graveyard_hate"],
    "tutors": ["tutor"],
    "sacrifice outlets": ["recursion"],
    "board wipes": ["removal_mass"],
    "spot removal": ["removal_targeted"],
    "cheap threats": ["threat_cheap"],
    "big threats": ["threat_finisher"],
    "top end": ["threat_finisher"],
    "early threats": ["threat_cheap"],
}

# Max CMC constraints for roles that imply cheapness.
# Prevents expensive cards (e.g. 9-mana Rise of the Dark Realms) from
# appearing as "cheap threats" due to mis-tagged card_roles data.
ROLE_CMC_LIMITS: Dict[str, int] = {
    "cheap threats": 3,
    "early threats": 3,
}

# Role targets by archetype - rough guide for what a deck "wants"
ARCHETYPE_ROLE_TARGETS = {
    "aggro": {
        "creatures": (20, 28),
        "removal": (4, 8),
        "card_advantage": (0, 4),
        "lands": (20, 24),
    },
    "midrange": {
        "creatures": (14, 22),
        "removal": (6, 10),
        "card_advantage": (4, 8),
        "lands": (24, 26),
    },
    "control": {
        "creatures": (2, 8),
        "removal": (8, 14),
        "card_advantage": (6, 10),
        "counterspells": (4, 8),
        "lands": (25, 27),
    },
    "combo": {
        "combo_pieces": (8, 16),
        "card_advantage": (6, 10),
        "protection": (4, 8),
        "lands": (22, 26),
    },
}


class DeckAnalyzer:
    """Analyzes an in-progress deck and surfaces insights."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.card_service = CardService(db)

    async def analyze(
        self,
        main_deck: List[Dict[str, Any]],
        sideboard: List[Dict[str, Any]],
        format: str = "standard",
    ) -> Dict[str, Any]:
        """
        Analyze a deck and return stats + actionable insights.
        """
        # Basic counts
        main_count = sum(e.get("quantity", 0) for e in main_deck)
        sb_count = sum(e.get("quantity", 0) for e in sideboard)

        # Categorize cards
        creatures = []
        noncreature_spells = []
        lands = []
        for entry in main_deck:
            card_data = entry.get("card", {}) or {}
            type_line = (card_data.get("type_line") or "").lower()
            name = entry.get("card_name", "").lower()
            qty = entry.get("quantity", 0)

            if "land" in type_line or name in ["plains", "island", "swamp", "mountain", "forest"]:
                lands.append((entry, qty))
            elif "creature" in type_line:
                creatures.append((entry, qty))
            else:
                noncreature_spells.append((entry, qty))

        creature_count = sum(q for _, q in creatures)
        spell_count = sum(q for _, q in noncreature_spells)
        land_count = sum(q for _, q in lands)

        # Mana curve (nonland cards only)
        curve = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # 6 = 6+
        for entry in main_deck:
            card_data = entry.get("card", {}) or {}
            type_line = (card_data.get("type_line") or "").lower()
            if "land" in type_line:
                continue
            mana_cost = card_data.get("mana_cost", "")
            cmc = self._estimate_cmc(mana_cost)
            bucket = min(cmc, 6)
            curve[bucket] = curve.get(bucket, 0) + entry.get("quantity", 0)

        # Color distribution (from mana costs)
        colors_used: Dict[str, int] = {}
        for entry in main_deck:
            card_data = entry.get("card", {}) or {}
            mana_cost = card_data.get("mana_cost", "") or ""
            qty = entry.get("quantity", 0)
            for color in ["W", "U", "B", "R", "G"]:
                pips = mana_cost.count(f"{{{color}}}")
                if pips > 0:
                    colors_used[color] = colors_used.get(color, 0) + (pips * qty)

        # Identify issues
        issues = []
        suggestions = []

        target_main = 60 if format != "cedh" else 100
        target_sb = 15 if format != "cedh" else 0

        if main_count < target_main:
            issues.append(f"Need {target_main - main_count} more cards in main deck ({main_count}/{target_main})")
        elif main_count > target_main:
            issues.append(f"{main_count - target_main} cards over the {target_main}-card limit")

        if format != "cedh" and sb_count < target_sb and sb_count > 0:
            issues.append(f"Sideboard is {target_sb - sb_count} cards short ({sb_count}/{target_sb})")

        if land_count == 0 and main_count > 0:
            issues.append("No lands in deck")
            suggestions.append("Add lands to your mana base")
        elif main_count >= 30:
            land_ratio = land_count / main_count
            if land_ratio < 0.33:
                suggestions.append("Consider adding more lands - you may struggle to cast spells on curve")
            elif land_ratio > 0.47:
                suggestions.append("Land count is high - you might flood out. Consider cutting a few")

        # Curve analysis
        nonland_total = creature_count + spell_count
        if nonland_total > 20:
            low_end = curve.get(1, 0) + curve.get(2, 0)
            high_end = curve.get(5, 0) + curve.get(6, 0)
            if low_end < nonland_total * 0.3:
                suggestions.append("Your curve is top-heavy. More cheap spells would help you survive early turns")
            if high_end > nonland_total * 0.3:
                suggestions.append("Lots of expensive spells. Make sure you have enough ramp or early interaction to get there")

        return {
            "main_deck_count": main_count,
            "sideboard_count": sb_count,
            "target_main": target_main,
            "target_sideboard": target_sb,
            "creature_count": creature_count,
            "spell_count": spell_count,
            "land_count": land_count,
            "mana_curve": curve,
            "colors": colors_used,
            "issues": issues,
            "suggestions": suggestions,
        }

    async def suggest_cards(
        self,
        main_deck: List[Dict[str, Any]],
        colors: List[str],
        role: str,
        format: str = "standard",
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        """Suggest cards that fill a specific role for the deck."""
        existing_names = {e.get("card_name", "").lower() for e in main_deck}

        query = f"{role} cards for {''.join(colors)} deck"
        try:
            cards = await self.card_service.semantic_search(
                query=query,
                colors=colors if colors else None,
                format=format,
                limit=limit * 3,
            )
        except Exception:
            cards = await self.card_service.search(
                colors=colors if colors else None,
                standard_only=(format == "standard"),
                format=format,
                limit=limit * 3,
            )

        results = []
        for card in cards:
            if card.name.lower() in existing_names:
                continue
            if len(results) >= limit:
                break
            results.append({
                "card_name": card.name,
                "card_id": str(card.id),
                "mana_cost": card.mana_cost,
                "type_line": card.type_line,
                "image_uri": card.image_uri,
            })

        return results

    async def _get_meta_role_cards(
        self,
        strategy: str,
        colors: List[str],
        system_roles: List[str],
        format: str = "standard",
        limit: int = 8,
        exclude: set = None,
        max_cmc: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get cards for the given system roles, ranked by tournament frequency then efficiency.

        Primary source: card_roles table cross-referenced with tournament decklist frequency.
        Uses format-based legality (legalities JSONB) instead of is_standard_legal.
        """
        if not system_roles:
            return []

        exclude = exclude or set()
        color_list = [c.upper() for c in colors]
        color_placeholders = ", ".join([f"'{c}'" for c in color_list])
        legality_key = FORMAT_LEGALITY_MAP.get(format, "standard")
        view_name = get_format_view(format)

        # Build archetype filter for tournament decks
        strategy_terms = [strategy.lower()] if strategy else []
        strategy_map = {
            "graveyard": ["reanimator", "graveyard", "dredge"],
            "reanimate": ["reanimator"],
            "sacrifice": ["sacrifice", "aristocrats"],
            "tokens": ["tokens", "go-wide"],
            "ramp": ["ramp", "big mana"],
            "control": ["control"],
            "aggro": ["aggro", "red deck"],
            "burn": ["burn", "red deck", "aggro"],
            "tempo": ["tempo"],
            "midrange": ["midrange"],
            "combo": ["combo"],
        }
        for keyword, terms in strategy_map.items():
            if strategy and keyword in strategy.lower():
                strategy_terms.extend(terms)
        # Deduplicate
        strategy_terms = list(dict.fromkeys(strategy_terms))

        archetype_conditions = [f"LOWER(d.archetype) LIKE '%' || :strat_{i} || '%'" for i in range(len(strategy_terms))]
        archetype_filter = " OR ".join(archetype_conditions) if archetype_conditions else "TRUE"

        # Build role filter
        role_placeholders = ", ".join([f":role_{i}" for i in range(len(system_roles))])

        # Build exclude filter
        exclude_clause = ""
        if exclude:
            exclude_placeholders = ", ".join([f":excl_{i}" for i in range(len(exclude))])
            exclude_clause = f"AND LOWER(c.name) NOT IN ({exclude_placeholders})"

        # Build CMC ceiling filter
        cmc_clause = ""
        if max_cmc is not None:
            cmc_clause = "AND c.cmc <= :max_cmc"

        query_sql = f"""
            WITH tournament_freq AS (
                SELECT
                    card_entry->>'card_name' as card_name,
                    COUNT(DISTINCT d.id) as freq
                FROM decklists d
                JOIN events e ON d.event_id = e.id,
                     jsonb_array_elements(d.main_deck) as card_entry
                WHERE e.format = :format
                  AND ({archetype_filter})
                GROUP BY card_entry->>'card_name'
            ),
            role_cards AS (
                SELECT DISTINCT ON (c.name)
                    c.id,
                    c.name,
                    SPLIT_PART(c.name, ' // ', 1) as front_face_name,
                    c.mana_cost,
                    c.type_line,
                    c.oracle_text,
                    c.image_uri,
                    c.image_uri_small,
                    c.colors,
                    c.cmc,
                    cr.efficiency
                FROM {view_name} c
                JOIN card_roles cr ON c.id = cr.card_id
                WHERE cr.role IN ({role_placeholders})
                  {exclude_clause}
                  {cmc_clause}
                ORDER BY c.name, cr.efficiency DESC NULLS LAST
            )
            SELECT
                rc.id,
                rc.name,
                rc.mana_cost,
                rc.type_line,
                rc.oracle_text,
                rc.image_uri,
                rc.image_uri_small,
                rc.efficiency,
                rc.cmc,
                COALESCE(tf.freq, 0) as tournament_count
            FROM role_cards rc
            LEFT JOIN tournament_freq tf ON
                LOWER(tf.card_name) = LOWER(rc.name) OR
                LOWER(tf.card_name) = LOWER(rc.front_face_name)
            WHERE rc.colors = ARRAY[]::varchar[]
               OR rc.colors <@ ARRAY[{color_placeholders}]::varchar[]
            ORDER BY tournament_count DESC, rc.efficiency DESC NULLS LAST, rc.cmc ASC
            LIMIT :limit
        """

        # Build params
        params: Dict[str, Any] = {"format": format, "limit": limit}
        if max_cmc is not None:
            params["max_cmc"] = max_cmc
        for i, term in enumerate(strategy_terms):
            params[f"strat_{i}"] = term
        for i, role in enumerate(system_roles):
            params[f"role_{i}"] = role
        if exclude:
            for i, name in enumerate(sorted(exclude)):
                params[f"excl_{i}"] = name.lower()

        result = await self.db.execute(text(query_sql), params)
        rows = result.all()

        return [
            {
                "card_name": row.name,
                "card_id": str(row.id),
                "mana_cost": row.mana_cost,
                "type_line": row.type_line,
                "oracle_text": row.oracle_text,
                "image_uri": row.image_uri,
                "image_uri_small": row.image_uri_small,
                "tournament_count": row.tournament_count,
                "efficiency": row.efficiency,
            }
            for row in rows
        ]

    async def suggest_cards_for_strategy(
        self,
        strategy: str,
        colors: List[str],
        roles: List[str],
        existing_cards: List[str],
        format: str = "standard",
        cards_per_role: int = 8,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Suggest cards grouped by role for a given strategy.
        Returns {role: [card_data, ...]} with rich metadata.

        Meta-first: uses card_roles + tournament frequency as primary source,
        falls back to semantic search only when roles aren't in ROLE_MAP or
        the primary source doesn't return enough cards.
        """
        existing_lower = {n.lower() for n in existing_cards}
        # Track cards across ALL roles to prevent the same card appearing in
        # multiple groups (e.g. Rise of the Dark Realms in "threats", "payoffs",
        # "finishers", and "removal" simultaneously)
        global_seen: set = set(existing_lower)
        results: Dict[str, List[Dict[str, Any]]] = {}

        for role in roles:
            role_key = role.lower().strip()
            system_roles = ROLE_MAP.get(role_key)

            role_cards: List[Dict[str, Any]] = []

            if system_roles:
                # Primary path: card_roles + tournament frequency
                cmc_limit = ROLE_CMC_LIMITS.get(role_key)
                meta_cards = await self._get_meta_role_cards(
                    strategy=strategy,
                    colors=colors,
                    system_roles=system_roles,
                    format=format,
                    limit=cards_per_role * 3,  # fetch extra to compensate for dedup
                    max_cmc=cmc_limit,
                    exclude=global_seen,
                )
                for card in meta_cards:
                    if card["card_name"].lower() in global_seen:
                        continue
                    if len(role_cards) >= cards_per_role:
                        break
                    role_cards.append(card)
                    global_seen.add(card["card_name"].lower())

                logger.info(
                    f"Meta-first: role='{role}' -> system_roles={system_roles}, "
                    f"got {len(role_cards)} cards (strategy='{strategy}')"
                )

            # Fallback: semantic search if no ROLE_MAP entry or not enough cards
            if len(role_cards) < cards_per_role:
                remaining_needed = cards_per_role - len(role_cards)

                query = f"{role} {strategy} cards for {''.join(colors)} deck"
                try:
                    cards = await self.card_service.semantic_search(
                        query=query,
                        colors=colors if colors else None,
                        format=format,
                        limit=remaining_needed * 3,
                    )
                except Exception:
                    cards = await self.card_service.search(
                        colors=colors if colors else None,
                        standard_only=(format == "standard"),
                        format=format,
                        limit=remaining_needed * 3,
                    )

                for card in cards:
                    if card.name.lower() in global_seen:
                        continue
                    if len(role_cards) >= cards_per_role:
                        break
                    role_cards.append({
                        "card_name": card.name,
                        "card_id": str(card.id),
                        "mana_cost": card.mana_cost,
                        "type_line": card.type_line,
                        "oracle_text": card.oracle_text,
                        "image_uri": card.image_uri,
                        "image_uri_small": card.image_uri_small,
                    })
                    global_seen.add(card.name.lower())

                if not system_roles:
                    logger.info(f"Semantic fallback: role='{role}' not in ROLE_MAP, got {len(role_cards)} cards")

            if role_cards:
                results[role] = role_cards

        return results

    async def compute_mana_base(
        self,
        main_deck: List[Dict[str, Any]],
        colors: List[str],
        format: str = "standard",
        target_total: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute a mana base for the current deck.
        Returns a list of land entries with quantities.
        """
        from app.services.card_service import get_format_legality_condition
        from app.models.card import Card

        nonland_count = 0
        color_pips: Dict[str, int] = {}
        for entry in main_deck:
            card_data = entry.get("card", {}) or {}
            type_line = (card_data.get("type_line") or "").lower()
            if "land" in type_line:
                continue
            qty = entry.get("quantity", 0)
            nonland_count += qty
            mana_cost = card_data.get("mana_cost", "") or ""
            for color in ["W", "U", "B", "R", "G"]:
                pips = mana_cost.count(f"{{{color}}}")
                if pips > 0:
                    color_pips[color] = color_pips.get(color, 0) + (pips * qty)

        if format == "cedh":
            target = target_total or max(34, 99 - nonland_count)
        else:
            target = target_total or max(20, 60 - nonland_count)

        total_pips = sum(color_pips.values()) or 1
        lands: List[Dict[str, Any]] = []

        # Allocate dual/fetch lands first, then basics
        # For now, use basics proportional to color pips
        basic_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
        allocated = 0

        # Search for dual lands in the colors
        if len(colors) >= 2:
            dual_query = f"dual land {''.join(colors)}"
            try:
                dual_cards = await self.card_service.semantic_search(
                    query=dual_query,
                    format=format,
                    limit=20,
                )
            except Exception:
                dual_cards = []

            dual_count = min(len(dual_cards), target // 3)
            for card in dual_cards[:dual_count]:
                type_line = (card.type_line or "").lower()
                if "land" not in type_line:
                    continue
                lands.append({
                    "card_name": card.name,
                    "quantity": 1 if format == "cedh" else min(4, 2),
                    "mana_cost": card.mana_cost,
                    "type_line": card.type_line,
                    "image_uri": card.image_uri,
                })
                allocated += lands[-1]["quantity"]
                if allocated >= target // 3:
                    break

        # Fill remaining with basics
        remaining = target - allocated
        if remaining > 0 and colors:
            for color in colors:
                basic_name = basic_map.get(color)
                if not basic_name:
                    continue
                ratio = color_pips.get(color, 1) / total_pips
                qty = max(1, round(remaining * ratio))
                card = await self.card_service.get_by_name(basic_name, format=format)
                if card:
                    lands.append({
                        "card_name": card.name,
                        "quantity": qty,
                        "mana_cost": card.mana_cost,
                        "type_line": card.type_line,
                        "image_uri": card.image_uri,
                    })

        return lands

    def _estimate_cmc(self, mana_cost: str) -> int:
        """Estimate CMC from mana cost string like {2}{U}{U}."""
        if not mana_cost:
            return 0
        cmc = 0
        i = 0
        while i < len(mana_cost):
            if mana_cost[i] == '{':
                end = mana_cost.index('}', i)
                symbol = mana_cost[i + 1:end]
                if symbol.isdigit():
                    cmc += int(symbol)
                elif symbol in ('W', 'U', 'B', 'R', 'G'):
                    cmc += 1
                elif symbol == 'X':
                    pass  # X = 0 for curve purposes
                else:
                    cmc += 1  # hybrid, phyrexian, etc.
                i = end + 1
            else:
                i += 1
        return cmc
