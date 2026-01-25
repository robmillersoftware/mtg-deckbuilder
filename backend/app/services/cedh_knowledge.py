"""
cEDH (Competitive Elder Dragon Highlander) Knowledge Base

This module contains structured knowledge about the cEDH metagame,
including common win conditions, staples, and commander-specific packages.
"""

from typing import Dict, List, Any

# =============================================================================
# COMPETITIVE VIABILITY GUIDANCE
# =============================================================================

# Strategies that are NOT competitive in cEDH and should be avoided/redirected
NON_VIABLE_STRATEGIES = {
    "superfriends": {
        "reason": "Planeswalkers are too slow and vulnerable. Games end turns 3-5, not turn 10+.",
        "redirect_to": "stax_control",
        "note": "If running a planeswalker commander, build around efficient combos, not loyalty abilities.",
    },
    "battlecruiser": {
        "reason": "Big splashy spells and creatures are too slow. cEDH is about efficiency.",
        "redirect_to": "combo",
    },
    "tribal": {
        "reason": "Most tribal synergies are too slow and weak. Exceptions exist for hatebear-heavy builds.",
        "redirect_to": "midrange",
    },
    "voltron": {
        "reason": "Commander damage is too slow and easily disrupted. 21 damage takes too long.",
        "redirect_to": "combo",
    },
    "landfall": {
        "reason": "Land-based value engines are too slow for cEDH.",
        "redirect_to": "combo",
    },
    "group_hug": {
        "reason": "Giving opponents resources accelerates their wins, not yours.",
        "redirect_to": "stax",
    },
    "mill": {
        "reason": "Milling opponents is too slow. Self-mill for combo wins is fine.",
        "redirect_to": "combo",
    },
}

# Viable cEDH archetypes with descriptions
CEDH_ARCHETYPES = {
    "turbo": {
        "description": "Race to combo win as fast as possible, typically turns 2-4",
        "characteristics": ["Minimal interaction", "Maximum speed", "All-in on combo"],
        "land_count": "28-30",
        "ideal_for": "Fast combo commanders with built-in card advantage",
    },
    "midrange": {
        "description": "Balanced approach with good interaction and multiple combo lines",
        "characteristics": ["Moderate interaction", "Value engines", "Flexible gameplan"],
        "land_count": "30-32",
        "ideal_for": "4+ color commanders, value-oriented commanders",
    },
    "stax": {
        "description": "Slow the game down with hate pieces while assembling your win",
        "characteristics": ["Heavy disruption", "Tax effects", "Resource denial"],
        "land_count": "30-32",
        "ideal_for": "Commanders that break parity on stax pieces",
    },
    "control": {
        "description": "Heavy interaction to stop opponents, win through attrition",
        "characteristics": ["Counter-heavy", "Removal-heavy", "Patience"],
        "land_count": "32-34",
        "ideal_for": "Blue-heavy commanders with card advantage",
    },
}

# Card categories to AVOID in cEDH (these are "trap" cards that seem good but aren't)
CARDS_TO_AVOID = {
    "slow_planeswalkers": {
        "description": "Planeswalkers costing 4+ mana that don't immediately win or provide massive value",
        "examples_to_avoid": [
            "Most 5+ mana planeswalkers",
            "Planeswalkers that need to tick up to be useful",
            "Planeswalkers without game-winning ultimates",
        ],
        "exceptions": [
            "Teferi, Time Raveler (3 mana, immediate value, stops interaction)",
            "Narset, Parter of Veils (3 mana, shuts down opponents)",
            "Oko, Thief of Crowns (3 mana, immediate threat neutralization)",
        ],
    },
    "casual_value_engines": {
        "description": "Cards that generate value over many turns",
        "examples_to_avoid": [
            "Most equipment (too slow to equip)",
            "Most auras (2-for-1 risk)",
            "Cards that need to survive a turn cycle",
            "Pillow fort cards (don't advance your win)",
        ],
    },
    "overcosted_interaction": {
        "description": "Removal or counterspells costing 3+ mana",
        "examples_to_avoid": [
            "Most 4+ mana counterspells",
            "Most 3+ mana targeted removal",
            "Board wipes without additional value",
        ],
    },
    "win_more_cards": {
        "description": "Cards that are only good when you're already winning",
        "examples_to_avoid": [
            "Cards that double effects (usually too slow)",
            "Cards that need other cards to function",
            "Cute synergy pieces without standalone value",
        ],
    },
}

# Primary win conditions in cEDH
WIN_CONDITIONS = {
    "thoracle": {
        "name": "Thassa's Oracle Combo",
        "description": "The most common cEDH win condition. Exile your library then win with Thassa's Oracle trigger.",
        "required_cards": ["Thassa's Oracle"],
        "enablers": ["Demonic Consultation", "Tainted Pact"],
        "how_it_works": "Cast Demonic Consultation or Tainted Pact to exile your library, then cast Thassa's Oracle to win.",
    },
    "food_chain": {
        "name": "Food Chain Combo",
        "description": "Generate infinite mana by repeatedly exiling and casting creatures that can be cast from exile.",
        "required_cards": ["Food Chain"],
        "enablers": ["Misthollow Griffin", "Eternal Scourge"],
        "how_it_works": "Exile Misthollow Griffin or Eternal Scourge to Food Chain, generate mana, recast from exile, repeat for infinite creature mana.",
    },
    "underworld_breach": {
        "name": "Underworld Breach Combo",
        "description": "Loop spells from graveyard with Breach + Brain Freeze/Lion's Eye Diamond.",
        "required_cards": ["Underworld Breach"],
        "enablers": ["Brain Freeze", "Lion's Eye Diamond", "Wheel of Fortune"],
        "how_it_works": "Use LED + Breach + Brain Freeze to mill opponents, or loop wheels to draw entire deck.",
    },
    "ad_nauseam": {
        "name": "Ad Nauseam",
        "description": "Draw a massive portion of your deck at instant speed, then win with assembled combo.",
        "required_cards": ["Ad Nauseam"],
        "enablers": ["Angel's Grace", "Phyrexian Unlife"],
        "how_it_works": "Keep mana curve extremely low, Ad Nauseam draws 30+ cards, assemble Thoracle win.",
    },
}

# Staples that virtually every cEDH deck should run (by category)
CEDH_STAPLES = {
    "free_counterspells": {
        "description": "Counterspells that can be cast without paying mana - critical for protecting combos and stopping opponents",
        "cards": [
            "Force of Will",
            "Force of Negation",
            "Pact of Negation",
            "Fierce Guardianship",
            "Deflecting Swat",
            "Mindbreak Trap",
        ],
        "priority": "must_have",
    },
    "efficient_counterspells": {
        "description": "Low-cost counterspells for interaction",
        "cards": [
            "Swan Song",
            "Flusterstorm",
            "Mental Misstep",
            "Dispel",
            "Spell Pierce",
            "An Offer You Can't Refuse",
            "Counterspell",
            "Dovin's Veto",
            "Negate",
        ],
        "priority": "high",
    },
    "fast_mana": {
        "description": "Mana acceleration to deploy threats and combos faster",
        "cards": [
            "Sol Ring",
            "Mana Vault",
            "Chrome Mox",
            "Mox Diamond",
            "Mana Crypt",  # Note: Banned as of 2024
            "Jeweled Lotus",  # Note: Banned as of 2024
            "Lotus Petal",
            "Dark Ritual",
            "Cabal Ritual",
            "Culling the Weak",
        ],
        "priority": "must_have",
        "notes": "Mana Crypt and Jeweled Lotus were banned in 2024",
    },
    "tutors": {
        "description": "Search effects to find combo pieces consistently",
        "cards": [
            "Demonic Tutor",
            "Vampiric Tutor",
            "Imperial Seal",
            "Grim Tutor",
            "Diabolic Intent",
            "Worldly Tutor",
            "Enlightened Tutor",
            "Mystical Tutor",
            "Personal Tutor",
            "Gamble",
        ],
        "priority": "must_have",
    },
    "card_advantage": {
        "description": "Efficient card draw and selection",
        "cards": [
            "Rhystic Study",
            "Mystic Remora",
            "Sylvan Library",
            "Necropotence",
            "Ad Nauseam",
            "Brainstorm",
            "Ponder",
            "Preordain",
            "Gitaxian Probe",
        ],
        "priority": "high",
    },
    "interaction": {
        "description": "Efficient removal and disruption",
        "cards": [
            "Swords to Plowshares",
            "Path to Exile",
            "Abrupt Decay",
            "Assassin's Trophy",
            "Nature's Claim",
            "Chain of Vapor",
            "Cyclonic Rift",
            "Toxic Deluge",
        ],
        "priority": "high",
    },
    "mana_dorks": {
        "description": "One-mana creatures that produce mana",
        "cards": [
            "Birds of Paradise",
            "Llanowar Elves",
            "Elvish Mystic",
            "Fyndhorn Elves",
            "Deathrite Shaman",
            "Ignoble Hierarch",
            "Noble Hierarch",
            "Elves of Deep Shadow",
            "Avacyn's Pilgrim",
            "Boreal Druid",
            "Arbor Elf",
        ],
        "priority": "high",
    },
    "hatebears": {
        "description": "Creatures that disrupt opponents' strategies",
        "cards": [
            "Drannith Magistrate",
            "Opposition Agent",
            "Aven Mindcensor",
            "Collector Ouphe",
            "Dauthi Voidwalker",
            "Esper Sentinel",
            "Grand Abolisher",
            "Thalia, Guardian of Thraben",
            "Sanctum Prelate",
            "Leonin Arbiter",
            "Spirit of the Labyrinth",
            "Linvala, Keeper of Silence",
            "Hullbreacher",  # Note: Banned in regular Commander but legal in cEDH
        ],
        "priority": "high",
    },
    "utility_creatures": {
        "description": "Value creatures commonly played in cEDH",
        "cards": [
            "Archivist of Oghma",
            "Ranger-Captain of Eos",
            "Recruiter of the Guard",
            "Imperial Recruiter",
            "Spell Queller",
            "Gilded Drake",
            "Notion Thief",
            "Seedborn Muse",
            "Tendershoot Dryad",
            "Destiny Spinner",
            "Allosaurus Shepherd",
            "Vexing Shusher",
            "Grand Arbiter Augustin IV",
            "Lavinia, Azorius Renegade",
            "Kambal, Consul of Allocation",
        ],
        "priority": "medium",
    },
    "combo_creatures": {
        "description": "Creatures that enable or are part of combo wins",
        "cards": [
            "Thassa's Oracle",
            "Laboratory Maniac",
            "Jace, Wielder of Mysteries",
            "Walking Ballista",
            "Devoted Druid",
            "Vizier of Remedies",
            "Heliod, Sun-Crowned",
            "Kiki-Jiki, Mirror Breaker",
            "Felidar Guardian",
            "Phantasmal Image",
            "Phyrexian Metamorph",
        ],
        "priority": "high",
    },
}


# Complete fetch land data with color pairs
FETCH_LANDS = {
    "Polluted Delta": ["U", "B"],
    "Flooded Strand": ["W", "U"],
    "Bloodstained Mire": ["B", "R"],
    "Wooded Foothills": ["R", "G"],
    "Windswept Heath": ["W", "G"],
    "Marsh Flats": ["W", "B"],
    "Scalding Tarn": ["U", "R"],
    "Verdant Catacombs": ["B", "G"],
    "Arid Mesa": ["W", "R"],
    "Misty Rainforest": ["U", "G"],
    "Prismatic Vista": [],  # Fetches any basic, always include
}

# Original dual lands with color pairs
ORIGINAL_DUALS = {
    "Underground Sea": ["U", "B"],
    "Tropical Island": ["U", "G"],
    "Volcanic Island": ["U", "R"],
    "Tundra": ["W", "U"],
    "Bayou": ["B", "G"],
    "Badlands": ["B", "R"],
    "Savannah": ["W", "G"],
    "Scrubland": ["W", "B"],
    "Taiga": ["R", "G"],
    "Plateau": ["W", "R"],
}

# Shock lands with color pairs
SHOCK_LANDS = {
    "Watery Grave": ["U", "B"],
    "Breeding Pool": ["U", "G"],
    "Steam Vents": ["U", "R"],
    "Hallowed Fountain": ["W", "U"],
    "Overgrown Tomb": ["B", "G"],
    "Blood Crypt": ["B", "R"],
    "Temple Garden": ["W", "G"],
    "Godless Shrine": ["W", "B"],
    "Stomping Ground": ["R", "G"],
    "Sacred Foundry": ["W", "R"],
}

# Rainbow and utility lands (always include for 3+ colors)
RAINBOW_LANDS = [
    "Command Tower",
    "City of Brass",
    "Mana Confluence",
    "Exotic Orchard",
    "Forbidden Orchard",
    "Reflecting Pool",
    "Gemstone Caverns",
]

UTILITY_LANDS = [
    "Ancient Tomb",
    "Boseiju, Who Endures",
    "Otawara, Soaring City",
    "Urza's Saga",
]


def get_cedh_lands_for_colors(colors: List[str]) -> Dict[str, List[str]]:
    """Get the optimal cEDH mana base for a given color identity."""
    colors_upper = {c.upper() for c in colors}

    result = {
        "fetch_lands": [],
        "original_duals": [],
        "shock_lands": [],
        "rainbow_lands": [],
        "utility_lands": [],
    }

    # Get all on-color fetch lands
    for land, land_colors in FETCH_LANDS.items():
        if not land_colors or all(c in colors_upper for c in land_colors):
            result["fetch_lands"].append(land)

    # Get all on-color original duals
    for land, land_colors in ORIGINAL_DUALS.items():
        if all(c in colors_upper for c in land_colors):
            result["original_duals"].append(land)

    # Get all on-color shock lands
    for land, land_colors in SHOCK_LANDS.items():
        if all(c in colors_upper for c in land_colors):
            result["shock_lands"].append(land)

    # Rainbow lands for 3+ colors
    if len(colors_upper) >= 3:
        result["rainbow_lands"] = RAINBOW_LANDS.copy()
    elif len(colors_upper) >= 2:
        result["rainbow_lands"] = ["Command Tower", "City of Brass", "Mana Confluence"]

    # Utility lands (filter by color for channel lands)
    for land in UTILITY_LANDS:
        if land == "Ancient Tomb":
            result["utility_lands"].append(land)
        elif land == "Boseiju, Who Endures" and "G" in colors_upper:
            result["utility_lands"].append(land)
        elif land == "Otawara, Soaring City" and "U" in colors_upper:
            result["utility_lands"].append(land)
        elif land == "Urza's Saga":
            result["utility_lands"].append(land)

    return result

# Tainted Pact constraint
TAINTED_PACT_CONSTRAINT = """
If running Tainted Pact as a win condition, the deck must have NO duplicate card names.
This means:
- Only 1 of each basic land (1 Island, 1 Swamp, etc.)
- No cards with the same name
- Snow-covered basics count as different from regular basics
This severely constrains mana base construction but enables a 2-card win condition.
"""


def get_cedh_system_prompt(commander: str = None, colors: List[str] = None, strategy: str = None) -> str:
    """Generate a cEDH-specific system prompt for deck building."""

    # Check if the requested strategy is non-viable and needs redirection
    strategy_warning = ""
    if strategy:
        strategy_lower = strategy.lower()
        for non_viable, info in NON_VIABLE_STRATEGIES.items():
            if non_viable in strategy_lower:
                strategy_warning = f"""
*** STRATEGY WARNING ***
"{strategy}" is NOT a competitive cEDH strategy.
Reason: {info['reason']}
{info.get('note', '')}
Instead, build a competitive {info['redirect_to']} deck that can actually win in cEDH.
"""
                break

    # Note: Actual land count should be derived from tournament data
    # These are only fallback guidelines if no tournament data is available
    # Many cEDH decks run 28-30 lands regardless of color count
    num_colors = len(colors) if colors else 1

    prompt = f"""You are building a competitive cEDH (Competitive Elder Dragon Highlander) deck.
{strategy_warning}
## cEDH Format Rules
- Exactly 100 cards (99 + commander)
- Singleton: only 1 copy of each card except basic lands
- Commander's color identity restricts what cards you can play - ONLY use cards within the commander's colors
- No sideboard
- Mana Crypt and Jeweled Lotus are BANNED

## cEDH Deck Building Principles

### CRITICAL: What Makes cEDH Different from Casual Commander
cEDH games typically end on turns 3-5. Every card must be:
1. **Efficient** - Low mana cost relative to impact
2. **Immediately impactful** - No waiting for value over time
3. **Part of the gameplan** - Advances your win or stops opponents

### What to AVOID (These are TRAPS that seem good but lose games)
- **Slow planeswalkers** (4+ mana that don't immediately win) - Too slow, easily killed
- **"Value over time" cards** - Games don't last long enough
- **Pillow fort / defensive cards** - Don't advance your win condition
- **Synergy-dependent cards** - Must be good on their own
- **Casual "fun" cards** - This is competitive, not kitchen table
- **Most equipment and auras** - Too slow, 2-for-1 risk
- **Cards that need to untap** - You may not get another turn

### Win Conditions (REQUIRED - every deck needs these)
Include a primary combo win condition and ideally a backup. Common combos:
- Thassa's Oracle + Demonic Consultation/Tainted Pact (most common)
- Underworld Breach loops
- Food Chain (if running eligible creatures)
- Commander-specific combos

### Card Roles (fill ALL these categories based on your colors)
- **Fast Mana (10-15 cards)**: Mana rocks, rituals, and 1-mana dorks
- **Tutors (8-12 cards)**: Search effects to find combo pieces
- **Card Advantage (8-12 cards)**: Efficient draw (Rhystic Study, Mystic Remora, etc.)
- **Free/Cheap Counterspells (10-15 cards)**: Free counters + 1-2 mana counters
- **Interaction (5-8 cards)**: 1-2 mana targeted removal
- **Hatebears (3-6 cards)**: Creatures that disrupt opponents
- **Combo Pieces (5-8 cards)**: Your win condition and enablers

### Mana Base (typically 28-31 lands depending on deck speed and mana dork count)
- ALL on-color fetch lands (crucial for deck thinning and color fixing)
- ALL on-color original dual lands and shock lands
- Rainbow lands for 3+ color decks (Command Tower, City of Brass, Mana Confluence)
- Ancient Tomb (colorless but 2 mana)
- Very few basic lands (1-3 total) - only for fetch targets
- If running Tainted Pact: NO duplicate card names (use snow + regular basics)
- Fast decks with many mana dorks can run fewer lands; slower decks need more

### Anti-Synergy Awareness
- If your commander costs 4+ mana, DON'T run artifact hate that shuts off your own mana rocks
- If your deck uses graveyard, DON'T run symmetrical graveyard hate
- If your deck tutors, DON'T run anti-tutor effects
- High-CMC commanders need fast mana, not stax that stops it

### Final Checklist (VERIFY BEFORE OUTPUTTING)
- [ ] Exactly 99 cards (not 98, not 100)
- [ ] Appropriate land count (use tournament data if provided)
- [ ] Clear win condition present
- [ ] No cards over 4 mana without exceptional reason
- [ ] Every card has a purpose (if you can't explain it, cut it)
- [ ] No "cute" synergies that don't win the game

"""
    return prompt


def get_staples_for_colors(colors: List[str]) -> Dict[str, List[str]]:
    """Get cEDH staples filtered by color identity."""
    result = {}

    color_map = {
        "W": ["white"],
        "U": ["blue"],
        "B": ["black"],
        "R": ["red"],
        "G": ["green"],
    }

    for category, data in CEDH_STAPLES.items():
        # For now, return all staples - color filtering would require card color data
        result[category] = data["cards"]

    return result
