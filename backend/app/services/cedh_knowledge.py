"""
cEDH (Competitive Elder Dragon Highlander) Knowledge Base

This module contains structured knowledge about the cEDH metagame,
including common win conditions, staples, and commander-specific packages.
"""

from typing import Dict, List, Any

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


def get_cedh_system_prompt(commander: str = None, colors: List[str] = None) -> str:
    """Generate a cEDH-specific system prompt for deck building."""
    prompt = """You are building a competitive cEDH (Competitive Elder Dragon Highlander) deck.

## cEDH Format Rules
- Exactly 100 cards (99 + commander)
- Singleton: only 1 copy of each card except basic lands
- Commander's color identity restricts what cards you can play - ONLY use cards within the commander's colors
- No sideboard
- Mana Crypt and Jeweled Lotus are BANNED

## cEDH Deck Building Principles

### Win Conditions
Include a primary combo win condition and ideally a backup. Common combos include Thassa's Oracle lines, Food Chain loops, Underworld Breach combos, or commander-specific wins.

### Card Roles (fill these categories based on your colors)
- **Fast Mana**: Mana-positive rocks, rituals, and mana dorks to accelerate your gameplan
- **Tutors**: Search effects to find combo pieces consistently
- **Card Advantage**: Efficient draw and selection to maintain resources
- **Free/Cheap Counterspells**: Protection for your combos and disruption for opponents
- **Interaction**: Efficient removal for creatures, artifacts, and enchantments
- **Hatebears/Stax**: Creatures that disrupt opponents (e.g., prevent tutoring, tax spells, stop combos)
- **Utility Creatures**: Value creatures that provide card advantage or protect your strategy

### Mana Base
- Run all on-color fetch lands and dual lands (original duals or shocks)
- Include rainbow lands for fixing
- Keep basic land count very low (1-3 total)
- If running Tainted Pact, ensure no duplicate card names

### Anti-Synergy Awareness
Consider how your hate pieces interact with your own gameplan:
- If your deck relies heavily on artifact mana (especially for casting an expensive commander), avoid symmetrical artifact hate
- If your deck uses the graveyard, avoid symmetrical graveyard hate
- If your deck needs to tutor, avoid cards that prevent searching libraries
- High-CMC commanders that need fast mana should NOT run cards that shut off their own acceleration

### General Guidelines
- Keep mana curve extremely low (average CMC under 2.0 ideally)
- Prioritize efficiency - every card should have high impact for its cost
- Balance between proactive (advancing your plan) and reactive (stopping opponents)
- Consider your commander's mana cost when choosing acceleration and hate pieces

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
