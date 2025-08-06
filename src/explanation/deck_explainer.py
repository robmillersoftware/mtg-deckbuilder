import json
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import openai
import anthropic
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeckExplainer:
    """Generate explanations for deck construction and card choices"""
    
    def __init__(self, card_database=None):
        self.card_database = card_database
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize LLM clients
        try:
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                openai.api_key = openai_key
                self.openai_client = openai
        except:
            pass
        
        try:
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if anthropic_key:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        except:
            pass
    
    def explain_deck(self, deck: Dict, meta_context: Dict = None) -> Dict:
        """Generate comprehensive deck explanation"""
        mainboard = deck.get('mainboard', [])
        sideboard = deck.get('sideboard', [])
        
        explanation = {
            'overview': self._generate_deck_overview(deck, meta_context),
            'card_explanations': self._explain_card_choices(mainboard),
            'mana_base_analysis': self._analyze_mana_base(mainboard),
            'curve_analysis': self._analyze_mana_curve(mainboard),
            'synergy_analysis': self._analyze_card_synergies(mainboard),
            'sideboard_guide': self._explain_sideboard(sideboard),
            'matchup_analysis': self._generate_matchup_analysis(deck, meta_context),
            'strengths_weaknesses': self._identify_strengths_weaknesses(deck)
        }
        
        return explanation
    
    def _generate_deck_overview(self, deck: Dict, meta_context: Dict = None) -> str:
        """Generate high-level deck overview"""
        mainboard = deck.get('mainboard', [])
        archetype = deck.get('archetype', 'Unknown')
        colors = deck.get('colors', [])
        
        # Analyze deck composition
        total_cards = sum(card['quantity'] for card in mainboard)
        card_types = self._categorize_cards(mainboard)
        
        # Build context for LLM
        context = f"""
        This is a {archetype} deck in Magic: The Gathering Standard format.
        Colors: {', '.join(colors) if colors else 'Colorless'}
        Total cards: {total_cards}
        
        Card type breakdown:
        - Creatures: {card_types['creatures']} cards
        - Spells: {card_types['spells']} cards  
        - Lands: {card_types['lands']} cards
        - Planeswalkers: {card_types['planeswalkers']} cards
        - Artifacts/Enchantments: {card_types['other']} cards
        
        Key cards in the deck:
        {self._format_key_cards(mainboard)}
        """
        
        if meta_context and 'deck_to_beat' in meta_context:
            context += f"\n\nCurrent meta leader: {meta_context['deck_to_beat'].get('name', 'Unknown')}"
        
        prompt = f"""
        {context}
        
        Please provide a 2-3 paragraph overview of this deck's strategy, gameplan, and position in the current Standard metagame. 
        Focus on:
        1. The deck's primary win condition and strategy
        2. How it plans to execute that strategy
        3. Its role in the current metagame (aggro/midrange/control/combo)
        """
        
        overview = self._query_llm(prompt)
        return overview if overview else f"This {archetype} deck focuses on {', '.join(colors)} colored strategies with {total_cards} total cards."
    
    def _explain_card_choices(self, mainboard: List[Dict]) -> Dict[str, str]:
        """Explain individual card choices"""
        explanations = {}
        
        # Group cards by role/type for efficient explanation
        card_groups = self._group_cards_by_role(mainboard)
        
        for group_name, cards in card_groups.items():
            if len(cards) <= 3:  # Explain individual cards in small groups
                for card in cards:
                    explanation = self._explain_single_card(card, group_name)
                    explanations[card['name']] = explanation
            else:  # Explain as a group for larger sets
                group_explanation = self._explain_card_group(cards, group_name)
                for card in cards:
                    explanations[card['name']] = group_explanation
        
        return explanations
    
    def _explain_single_card(self, card: Dict, role: str) -> str:
        """Explain why a specific card is included"""
        card_name = card['name']
        quantity = card['quantity']
        
        # Get card details if available
        card_details = ""
        if self.card_database:
            card_data = self.card_database.get_card_by_name(card_name)
            if card_data:
                card_details = f"Mana Cost: {card_data.get('mana_cost', 'Unknown')}, Type: {card_data.get('type_line', 'Unknown')}"
                if card_data.get('oracle_text'):
                    card_details += f", Effect: {card_data['oracle_text'][:100]}..."
        
        prompt = f"""
        Explain why {quantity}x {card_name} is included in this Magic: The Gathering deck.
        
        Card role: {role}
        {card_details}
        
        Provide a 1-2 sentence explanation focusing on:
        - What role this card serves in the deck
        - Why this specific card was chosen over alternatives
        - How the quantity ({quantity}) is justified
        
        Keep it concise and focused on gameplay impact.
        """
        
        explanation = self._query_llm(prompt)
        return explanation if explanation else f"{card_name} serves as {role} in the deck strategy."
    
    def _explain_card_group(self, cards: List[Dict], role: str) -> str:
        """Explain a group of similar cards"""
        card_names = [f"{card['quantity']}x {card['name']}" for card in cards]
        
        prompt = f"""
        Explain why these Magic: The Gathering cards are included in the deck:
        {', '.join(card_names)}
        
        These cards all serve the role: {role}
        
        Provide a 2-3 sentence explanation covering:
        - What collective role these cards serve
        - How they work together or complement each other
        - Why this particular selection was made
        
        Keep it focused on their strategic purpose.
        """
        
        explanation = self._query_llm(prompt)
        return explanation if explanation else f"These cards provide {role} for the deck's strategy."
    
    def _group_cards_by_role(self, mainboard: List[Dict]) -> Dict[str, List[Dict]]:
        """Group cards by their likely role in the deck"""
        groups = defaultdict(list)
        
        for card in mainboard:
            card_name = card['name'].lower()
            
            # Categorize by common patterns
            if any(word in card_name for word in ['plains', 'island', 'swamp', 'mountain', 'forest', 'land']):
                groups['lands'].append(card)
            elif any(word in card_name for word in ['bolt', 'shock', 'murder', 'destroy', 'exile']):
                groups['removal'].append(card)
            elif any(word in card_name for word in ['counterspell', 'negate', 'cancel']):
                groups['counterspells'].append(card)
            elif any(word in card_name for word in ['draw', 'divination', 'insight']):
                groups['card_draw'].append(card)
            elif any(word in card_name for word in ['dragon', 'angel', 'demon']):
                groups['finishers'].append(card)
            elif any(word in card_name for word in ['knight', 'soldier', 'warrior', 'creature']):
                groups['creatures'].append(card)
            else:
                groups['utility'].append(card)
        
        return dict(groups)
    
    def _analyze_mana_base(self, mainboard: List[Dict]) -> str:
        """Analyze and explain the mana base"""
        lands = [card for card in mainboard if 'land' in card['name'].lower()]
        total_lands = sum(card['quantity'] for card in lands)
        total_spells = sum(card['quantity'] for card in mainboard if 'land' not in card['name'].lower())
        
        land_ratio = total_lands / (total_lands + total_spells) if (total_lands + total_spells) > 0 else 0
        
        land_types = []
        for land in lands:
            land_types.extend([land['name']] * land['quantity'])
        
        land_breakdown = Counter(land_types)
        
        analysis = f"""
        Mana Base Analysis:
        - Total lands: {total_lands} ({land_ratio:.1%} of deck)
        - Land composition: {dict(land_breakdown)}
        
        The mana base appears {"consistent" if 22 <= total_lands <= 26 else "unusual"} 
        for this type of deck. {"Good land count for consistent draws." if 22 <= total_lands <= 26 else "Consider adjusting land count."}
        """
        
        return analysis
    
    def _analyze_mana_curve(self, mainboard: List[Dict]) -> str:
        """Analyze mana curve distribution"""
        curve = [0] * 8  # 0-7+ mana costs
        
        for card in mainboard:
            if 'land' not in card['name'].lower():
                # Estimate CMC (simplified)
                estimated_cmc = self._estimate_cmc(card['name'])
                cmc_slot = min(estimated_cmc, 7)
                curve[cmc_slot] += card['quantity']
        
        total_spells = sum(curve)
        curve_percentages = [count / total_spells * 100 if total_spells > 0 else 0 for count in curve]
        
        analysis = "Mana Curve Analysis:\n"
        for i, percentage in enumerate(curve_percentages):
            if percentage > 0:
                cmc_label = f"{i}" if i < 7 else "7+"
                analysis += f"  {cmc_label} mana: {curve[i]} cards ({percentage:.1f}%)\n"
        
        # Add strategic analysis
        if curve_percentages[1] + curve_percentages[2] > 50:
            analysis += "Low curve suggests aggressive strategy."
        elif curve_percentages[4] + curve_percentages[5] + curve_percentages[6] > 30:
            analysis += "High curve indicates late-game focused strategy."
        else:
            analysis += "Balanced curve supports midrange gameplay."
        
        return analysis
    
    def _estimate_cmc(self, card_name: str) -> int:
        """Estimate CMC from card name (simplified heuristic)"""
        name_lower = card_name.lower()
        
        if any(word in name_lower for word in ['bolt', 'shock', 'path']):
            return 1
        elif any(word in name_lower for word in ['bear', 'negate']):
            return 2  
        elif any(word in name_lower for word in ['murder', 'knight']):
            return 3
        elif any(word in name_lower for word in ['wrath', 'angel']):
            return 4
        elif any(word in name_lower for word in ['dragon', 'demon']):
            return 5
        else:
            return 3  # Default
    
    def _analyze_card_synergies(self, mainboard: List[Dict]) -> str:
        """Identify and explain card synergies"""
        # Look for common synergy patterns
        card_names = [card['name'].lower() for card in mainboard]
        synergies = []
        
        # Tribal synergies
        creature_types = ['knight', 'soldier', 'dragon', 'angel', 'demon', 'elemental']
        for creature_type in creature_types:
            type_cards = [name for name in card_names if creature_type in name]
            if len(type_cards) >= 3:
                synergies.append(f"Tribal {creature_type.title()} theme with {len(type_cards)} related cards")
        
        # Spell synergies
        if len([name for name in card_names if any(word in name for word in ['bolt', 'burn', 'shock'])]) >= 3:
            synergies.append("Burn spell synergy for direct damage strategy")
        
        if len([name for name in card_names if any(word in name for word in ['counter', 'negate'])]) >= 2:
            synergies.append("Counterspell package for control elements")
        
        if not synergies:
            synergies.append("Cards chosen for individual power level rather than specific synergies")
        
        return "Key Synergies:\n" + "\n".join(f"- {synergy}" for synergy in synergies)
    
    def _explain_sideboard(self, sideboard: List[Dict]) -> str:
        """Explain sideboard choices and when to bring them in"""
        if not sideboard:
            return "No sideboard provided."
        
        sb_categories = self._categorize_sideboard_cards(sideboard)
        
        explanation = "Sideboard Guide:\n\n"
        
        for category, cards in sb_categories.items():
            if cards:
                card_list = ", ".join([f"{card['quantity']}x {card['name']}" for card in cards])
                explanation += f"{category.title()}:\n"
                explanation += f"  Cards: {card_list}\n"
                explanation += f"  When to use: {self._get_sideboard_usage(category)}\n\n"
        
        return explanation
    
    def _categorize_sideboard_cards(self, sideboard: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize sideboard cards by purpose"""
        categories = defaultdict(list)
        
        for card in sideboard:
            name_lower = card['name'].lower()
            
            if any(word in name_lower for word in ['destroy', 'murder', 'exile']):
                categories['removal'].append(card)
            elif any(word in name_lower for word in ['counter', 'negate']):
                categories['counterspells'].append(card)
            elif any(word in name_lower for word in ['wrath', 'sweep', 'board']):
                categories['board_wipes'].append(card)
            elif any(word in name_lower for word in ['enchantment', 'artifact']):
                categories['hate'].append(card)
            else:
                categories['flex'].append(card)
        
        return dict(categories)
    
    def _get_sideboard_usage(self, category: str) -> str:
        """Get usage advice for sideboard category"""
        usage_map = {
            'removal': "Against creature-heavy decks and aggressive strategies",
            'counterspells': "Against combo decks and spell-based strategies", 
            'board_wipes': "Against go-wide token strategies and creature swarms",
            'hate': "Against specific problematic permanent types",
            'flex': "Situational cards for various matchups"
        }
        
        return usage_map.get(category, "Situational usage based on opponent's strategy")
    
    def _generate_matchup_analysis(self, deck: Dict, meta_context: Dict = None) -> str:
        """Generate matchup analysis against common archetypes"""
        if not meta_context or 'archetype_breakdown' not in meta_context:
            return "No meta context available for matchup analysis."
        
        archetype_breakdown = meta_context['archetype_breakdown']
        top_archetypes = sorted(archetype_breakdown.items(), key=lambda x: x[1]['percentage'], reverse=True)[:5]
        
        analysis = "Matchup Analysis:\n\n"
        
        for archetype, stats in top_archetypes:
            matchup_rating = self._analyze_single_matchup(deck, archetype, stats)
            analysis += f"{archetype} ({stats['percentage']:.1f}% of meta):\n"
            analysis += f"  {matchup_rating}\n\n"
        
        return analysis
    
    def _analyze_single_matchup(self, deck: Dict, opponent_archetype: str, opponent_stats: Dict) -> str:
        """Analyze matchup against specific archetype"""
        deck_archetype = deck.get('archetype', '').lower()
        opponent_lower = opponent_archetype.lower()
        
        # Simple matchup heuristics
        if 'aggro' in deck_archetype:
            if 'control' in opponent_lower:
                return "Favorable - Fast pressure before opponent stabilizes"
            elif 'aggro' in opponent_lower:
                return "Even - Mirror match depends on draws and curve"
            else:
                return "Slightly favorable - Speed advantage"
        
        elif 'control' in deck_archetype:
            if 'aggro' in opponent_lower:
                return "Challenging - Must survive early pressure"
            elif 'control' in opponent_lower:
                return "Grindy - Card advantage and win conditions matter"
            else:
                return "Favorable - Late game advantages"
        
        elif 'midrange' in deck_archetype:
            if 'aggro' in opponent_lower:
                return "Even - Depends on early interaction"
            elif 'control' in opponent_lower:
                return "Slightly favorable - Pressure with protection"
            else:
                return "Even - Power level contest"
        
        else:
            return "Even - Matchup depends on specific cards and draws"
    
    def _identify_strengths_weaknesses(self, deck: Dict) -> Dict[str, List[str]]:
        """Identify deck's main strengths and weaknesses"""
        mainboard = deck.get('mainboard', [])
        
        strengths = []
        weaknesses = []
        
        # Analyze card composition
        card_types = self._categorize_cards(mainboard)
        total_cards = sum(card['quantity'] for card in mainboard)
        
        # Strengths analysis
        if card_types['lands'] / total_cards >= 0.38:
            strengths.append("Consistent mana base reduces mana problems")
        
        if card_types['creatures'] >= 16:
            strengths.append("Strong creature presence for board pressure")
        
        if len(mainboard) <= 20:  # Low number of unique cards = high consistency
            strengths.append("Focused strategy with consistent draws")
        
        # Weaknesses analysis
        if card_types['lands'] / total_cards < 0.35:
            weaknesses.append("Low land count may cause mana problems")
        
        if card_types['spells'] < 8:
            weaknesses.append("Limited interaction with opponent's strategy")
        
        if len(deck.get('colors', [])) >= 3:
            weaknesses.append("Multi-color mana base may be inconsistent")
        
        # Default entries if none found
        if not strengths:
            strengths.append("Focused gameplan with synergistic card choices")
        if not weaknesses:
            weaknesses.append("May struggle against unexpected strategies")
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses
        }
    
    def _categorize_cards(self, mainboard: List[Dict]) -> Dict[str, int]:
        """Categorize cards by type"""
        categories = {
            'creatures': 0,
            'spells': 0, 
            'lands': 0,
            'planeswalkers': 0,
            'other': 0
        }
        
        for card in mainboard:
            name_lower = card['name'].lower()
            quantity = card['quantity']
            
            if any(word in name_lower for word in ['plains', 'island', 'swamp', 'mountain', 'forest', 'land']):
                categories['lands'] += quantity
            elif any(word in name_lower for word in ['dragon', 'angel', 'demon', 'knight', 'soldier', 'creature']):
                categories['creatures'] += quantity
            elif 'planeswalker' in name_lower:
                categories['planeswalkers'] += quantity
            else:
                categories['spells'] += quantity
        
        return categories
    
    def _format_key_cards(self, mainboard: List[Dict]) -> str:
        """Format key cards for display"""
        # Sort by quantity to highlight key cards
        sorted_cards = sorted(mainboard, key=lambda x: x['quantity'], reverse=True)
        key_cards = sorted_cards[:8]  # Top 8 most-played cards
        
        return '\n'.join([f"  {card['quantity']}x {card['name']}" for card in key_cards])
    
    def _query_llm(self, prompt: str) -> str:
        """Query available LLM for explanation"""
        try:
            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif self.openai_client:
                response = self.openai_client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                return response.choices[0].message.content
            
            else:
                logger.warning("No LLM client available for explanations")
                return ""
                
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return ""

if __name__ == "__main__":
    explainer = DeckExplainer()
    
    # Example deck
    test_deck = {
        'mainboard': [
            {'name': 'Lightning Bolt', 'quantity': 4},
            {'name': 'Mountain', 'quantity': 20},
            {'name': 'Red Dragon', 'quantity': 4},
            {'name': 'Shock', 'quantity': 4}
        ],
        'sideboard': [
            {'name': 'Counterspell', 'quantity': 3}
        ],
        'archetype': 'Aggro',
        'colors': ['R']
    }
    
    explanation = explainer.explain_deck(test_deck)
    print("Deck Overview:")
    print(explanation['overview'])