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
        ],
        "priority": "high",
    },
}

# Commander-specific packages and strategies
COMMANDER_PACKAGES = {
    "atraxa_grand_unifier": {
        "commander": "Atraxa, Grand Unifier",
        "colors": ["W", "U", "B", "G"],
        "strategy": "Food Chain combo into looping Atraxa to draw entire deck, then win with Thassa's Oracle",
        "primary_win": "food_chain",
        "backup_win": "thoracle",
        "core_package": [
            "Food Chain",
            "Misthollow Griffin",
            "Eternal Scourge",
            "Thassa's Oracle",
            "Demonic Consultation",
            "Tainted Pact",
        ],
        "synergy_cards": [
            "Neoform",  # Find Food Chain creatures
            "Eldritch Evolution",
            "Finale of Devastation",
            "Green Sun's Zenith",
        ],
        "notes": "Atraxa's ETB draws 4+ cards, making her excellent for rebuilding after interaction. Loop her with Food Chain mana to draw the entire deck.",
    },
    "kinnan_bonder_prodigy": {
        "commander": "Kinnan, Bonder Prodigy",
        "colors": ["U", "G"],
        "strategy": "Generate infinite mana with Basalt Monolith, then use Kinnan to find Thassa's Oracle",
        "primary_win": "thoracle",
        "core_package": [
            "Basalt Monolith",
            "Thassa's Oracle",
            "Demonic Consultation",  # Not in colors, ignore
        ],
        "synergy_cards": [
            "Freed from the Real",
            "Pemmin's Aura",
        ],
    },
    "tymna_kraum": {
        "commander": ["Tymna the Weaver", "Kraum, Ludevic's Opus"],
        "colors": ["W", "U", "B", "R"],
        "strategy": "Aggressive card advantage from commanders, win with Thoracle or Breach lines",
        "primary_win": "thoracle",
        "backup_win": "underworld_breach",
        "core_package": [
            "Thassa's Oracle",
            "Demonic Consultation",
            "Tainted Pact",
            "Underworld Breach",
            "Brain Freeze",
            "Lion's Eye Diamond",
        ],
    },
    "najeela_blade_blossom": {
        "commander": "Najeela, the Blade-Blossom",
        "colors": ["W", "U", "B", "R", "G"],
        "strategy": "Infinite combat steps with Najeela + mana producers, or Thoracle backup",
        "primary_win": "najeela_combo",
        "backup_win": "thoracle",
        "core_package": [
            "Derevi, Empyrial Tactician",
            "Nature's Will",
            "Sword of Feast and Famine",
            "Bear Umbra",
            "Druids' Repository",
            "Thassa's Oracle",
            "Demonic Consultation",
        ],
    },
    "rogsi": {
        "commander": ["Rograkh, Son of Rohgahh", "Silas Renn, Seeker Adept"],
        "colors": ["U", "B", "R"],
        "strategy": "Turbo Ad Nauseam into Thoracle win",
        "primary_win": "ad_nauseam",
        "backup_win": "thoracle",
        "core_package": [
            "Ad Nauseam",
            "Thassa's Oracle",
            "Demonic Consultation",
            "Tainted Pact",
        ],
        "notes": "Keep average CMC extremely low (under 2.0) to maximize Ad Nauseam draws.",
    },
}

# Mana base guidelines for cEDH
MANA_BASE_GUIDELINES = {
    "fetch_lands": {
        "description": "Run all on-color fetch lands for mana fixing and deck thinning",
        "examples": [
            "Polluted Delta", "Flooded Strand", "Bloodstained Mire",
            "Wooded Foothills", "Windswept Heath", "Marsh Flats",
            "Scalding Tarn", "Verdant Catacombs", "Arid Mesa", "Misty Rainforest",
        ],
    },
    "original_duals": {
        "description": "Original dual lands are optimal but not required",
        "examples": ["Underground Sea", "Tropical Island", "Bayou", "Tundra"],
    },
    "shock_lands": {
        "description": "Budget-friendly alternatives to original duals",
        "examples": ["Watery Grave", "Breeding Pool", "Overgrown Tomb"],
    },
    "rainbow_lands": {
        "description": "Lands that produce any color - essential for 3+ color decks",
        "examples": [
            "Command Tower", "City of Brass", "Mana Confluence",
            "Exotic Orchard", "Forbidden Orchard", "Reflecting Pool",
            "Gemstone Caverns",
        ],
    },
    "utility_lands": {
        "description": "Lands with additional utility",
        "examples": [
            "Ancient Tomb", "Urza's Saga",
        ],
    },
    "basic_count": "Run 1-3 basics total, mainly for Path to Exile/Assassin's Trophy",
}

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
- Commander's color identity restricts what cards you can play
- No sideboard

## cEDH Meta Knowledge

### Primary Win Conditions
Most cEDH decks win with one of these:
1. **Thassa's Oracle + Demonic Consultation/Tainted Pact**: Exile library, Oracle trigger wins
2. **Food Chain + Misthollow Griffin/Eternal Scourge**: Infinite creature mana to loop commander or win
3. **Underworld Breach + Brain Freeze + LED**: Loop spells from graveyard
4. **Ad Nauseam**: Draw 30+ cards at instant speed, assemble combo

### Required Staples (include these)
**Free Counterspells** (CRITICAL - run all in your colors):
- Force of Will, Force of Negation, Pact of Negation
- Fierce Guardianship (if playing blue commander)
- Deflecting Swat (if playing red commander)
- Mindbreak Trap

**Efficient Counterspells**:
- Swan Song, Flusterstorm, Mental Misstep, Dispel, Spell Pierce
- An Offer You Can't Refuse, Counterspell, Dovin's Veto

**Fast Mana** (run all legal ones):
- Sol Ring, Mana Vault, Chrome Mox, Mox Diamond, Lotus Petal
- Dark Ritual, Cabal Ritual (if black)
- NOTE: Mana Crypt and Jeweled Lotus are BANNED

**Tutors** (run all in your colors):
- Demonic Tutor, Vampiric Tutor, Imperial Seal, Grim Tutor
- Worldly Tutor, Enlightened Tutor, Mystical Tutor, Gamble

**Card Advantage**:
- Rhystic Study, Mystic Remora, Sylvan Library, Necropotence
- Ad Nauseam, Brainstorm, Ponder, Preordain

**Interaction**:
- Swords to Plowshares, Path to Exile, Abrupt Decay
- Assassin's Trophy, Nature's Claim, Chain of Vapor, Cyclonic Rift

**Mana Dorks** (if green):
- Birds of Paradise, Llanowar Elves, Elvish Mystic
- Deathrite Shaman, Noble Hierarch, Ignoble Hierarch

### Mana Base Guidelines
- All on-color fetch lands
- Original duals or shock lands
- Rainbow lands: Command Tower, City of Brass, Mana Confluence, Exotic Orchard
- Ancient Tomb is auto-include
- Only 1-3 basic lands total
- If running Tainted Pact: NO duplicate card names (use snow basics + regular basics)

"""

    # Add commander-specific knowledge if available
    if commander:
        commander_key = commander.lower().replace(" ", "_").replace(",", "").replace("'", "")

        # Check for known commanders
        for key, package in COMMANDER_PACKAGES.items():
            cmd = package.get("commander", "")
            if isinstance(cmd, list):
                if any(commander.lower() in c.lower() for c in cmd):
                    prompt += f"\n### Commander-Specific: {commander}\n"
                    prompt += f"**Strategy**: {package['strategy']}\n"
                    prompt += f"**Core Combo Package**: {', '.join(package['core_package'])}\n"
                    if package.get('synergy_cards'):
                        prompt += f"**Synergy Cards**: {', '.join(package['synergy_cards'])}\n"
                    if package.get('notes'):
                        prompt += f"**Notes**: {package['notes']}\n"
                    break
            elif commander.lower() in cmd.lower():
                prompt += f"\n### Commander-Specific: {commander}\n"
                prompt += f"**Strategy**: {package['strategy']}\n"
                prompt += f"**Core Combo Package**: {', '.join(package['core_package'])}\n"
                if package.get('synergy_cards'):
                    prompt += f"**Synergy Cards**: {', '.join(package['synergy_cards'])}\n"
                if package.get('notes'):
                    prompt += f"**Notes**: {package['notes']}\n"
                break

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
