import random
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import json
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Card:
    name: str
    mana_cost: str
    cmc: int
    types: List[str]
    colors: List[str]
    power: Optional[int] = None
    toughness: Optional[int] = None
    oracle_text: str = ""
    keywords: List[str] = None

@dataclass
class GameState:
    turn: int = 1
    active_player: int = 0  # 0 or 1
    life_totals: List[int] = None
    hands: List[List[Card]] = None
    libraries: List[List[Card]] = None
    battlefields: List[List[Card]] = None
    graveyards: List[List[Card]] = None
    mana_available: List[int] = None
    lands_played: List[int] = None
    
    def __post_init__(self):
        if self.life_totals is None:
            self.life_totals = [20, 20]
        if self.hands is None:
            self.hands = [[], []]
        if self.libraries is None:
            self.libraries = [[], []]
        if self.battlefields is None:
            self.battlefields = [[], []]
        if self.graveyards is None:
            self.graveyards = [[], []]
        if self.mana_available is None:
            self.mana_available = [0, 0]
        if self.lands_played is None:
            self.lands_played = [0, 0]

class SimpleMTGSimulator:
    """Simplified MTG game simulator for deck evaluation"""
    
    def __init__(self):
        self.max_turns = 20
        self.hand_size_limit = 7
        
    def simulate_game(self, deck1: List[Dict], deck2: List[Dict]) -> Dict:
        """Simulate a single game between two decks"""
        # Convert deck lists to Card objects
        cards1 = self._build_deck(deck1)
        cards2 = self._build_deck(deck2)
        
        # Initialize game state
        game_state = self._initialize_game(cards1, cards2)
        
        # Play the game
        winner = self._play_game(game_state)
        
        return {
            'winner': winner,
            'turns': game_state.turn,
            'final_life': game_state.life_totals.copy(),
            'game_state': game_state
        }
    
    def simulate_matches(self, deck1: List[Dict], deck2: List[Dict], num_games: int = 100) -> Dict:
        """Simulate multiple games and return statistics"""
        results = []
        deck1_wins = 0
        deck2_wins = 0
        total_turns = 0
        
        for _ in range(num_games):
            result = self.simulate_game(deck1, deck2)
            results.append(result)
            
            if result['winner'] == 0:
                deck1_wins += 1
            elif result['winner'] == 1:
                deck2_wins += 1
            
            total_turns += result['turns']
        
        return {
            'total_games': num_games,
            'deck1_wins': deck1_wins,
            'deck2_wins': deck2_wins,
            'deck1_winrate': deck1_wins / num_games if num_games > 0 else 0,
            'deck2_winrate': deck2_wins / num_games if num_games > 0 else 0,
            'avg_game_length': total_turns / num_games if num_games > 0 else 0,
            'results': results
        }
    
    def _build_deck(self, decklist: List[Dict]) -> List[Card]:
        """Convert decklist to Card objects"""
        cards = []
        
        for entry in decklist:
            if entry.get('section', 'mainboard') != 'mainboard':
                continue  # Skip sideboard for now
                
            card_name = entry['name']
            quantity = entry['quantity']
            
            # Create simplified card object
            card = self._create_card_from_name(card_name)
            
            # Add multiple copies
            for _ in range(quantity):
                cards.append(card)
        
        return cards
    
    def _create_card_from_name(self, name: str) -> Card:
        """Create a Card object with estimated stats based on name"""
        # This is a simplified approach - in practice, use card database
        name_lower = name.lower()
        
        # Estimate card properties based on common patterns
        if 'plains' in name_lower or 'island' in name_lower or 'swamp' in name_lower or 'mountain' in name_lower or 'forest' in name_lower:
            return Card(
                name=name,
                mana_cost="",
                cmc=0,
                types=['Land'],
                colors=[],
                oracle_text=f"Tap: Add mana"
            )
        
        # Estimate creature stats
        power, toughness = self._estimate_creature_stats(name)
        cmc = self._estimate_cmc(name)
        colors = self._estimate_colors(name)
        types = self._estimate_types(name)
        
        return Card(
            name=name,
            mana_cost=self._estimate_mana_cost(cmc, colors),
            cmc=cmc,
            types=types,
            colors=colors,
            power=power,
            toughness=toughness,
            oracle_text="",
            keywords=[]
        )
    
    def _estimate_creature_stats(self, name: str) -> Tuple[Optional[int], Optional[int]]:
        """Estimate creature power/toughness"""
        name_lower = name.lower()
        
        # Common patterns
        if any(word in name_lower for word in ['dragon', 'demon', 'angel']):
            return (4, 4)  # Large creatures
        elif any(word in name_lower for word in ['knight', 'soldier', 'warrior']):
            return (2, 2)  # Medium creatures
        elif any(word in name_lower for word in ['token', 'spirit', 'thopter']):
            return (1, 1)  # Small creatures
        elif any(word in name_lower for word in ['wall', 'defender']):
            return (0, 4)  # Defensive creatures
        elif any(word in name_lower for word in ['creature', 'beast', 'elemental']):
            return (3, 3)  # Default creatures
        else:
            return (None, None)  # Not a creature
    
    def _estimate_cmc(self, name: str) -> int:
        """Estimate converted mana cost"""
        name_lower = name.lower()
        
        if any(word in name_lower for word in ['bolt', 'shock', 'path', 'push']):
            return 1
        elif any(word in name_lower for word in ['counterspell', 'negate', 'bear']):
            return 2
        elif any(word in name_lower for word in ['murder', 'cancel', 'knight']):
            return 3
        elif any(word in name_lower for word in ['wrath', 'sweeper', 'angel']):
            return 4
        elif any(word in name_lower for word in ['dragon', 'demon']):
            return 5
        else:
            return 3  # Default
    
    def _estimate_colors(self, name: str) -> List[str]:
        """Estimate card colors"""
        name_lower = name.lower()
        colors = []
        
        if any(word in name_lower for word in ['lightning', 'fire', 'red', 'mountain']):
            colors.append('R')
        if any(word in name_lower for word in ['island', 'blue', 'counter', 'draw']):
            colors.append('U')
        if any(word in name_lower for word in ['swamp', 'black', 'death', 'destroy']):
            colors.append('B')
        if any(word in name_lower for word in ['forest', 'green', 'ramp', 'growth']):
            colors.append('G')
        if any(word in name_lower for word in ['plains', 'white', 'angel', 'knight']):
            colors.append('W')
        
        return colors if colors else ['C']  # Colorless if no colors detected
    
    def _estimate_types(self, name: str) -> List[str]:
        """Estimate card types"""
        name_lower = name.lower()
        
        if any(word in name_lower for word in ['plains', 'island', 'swamp', 'mountain', 'forest']):
            return ['Land']
        elif any(word in name_lower for word in ['dragon', 'angel', 'demon', 'knight', 'soldier', 'warrior', 'beast', 'elemental']):
            return ['Creature']
        elif any(word in name_lower for word in ['bolt', 'shock', 'murder']):
            return ['Instant']
        elif any(word in name_lower for word in ['enchantment']):
            return ['Enchantment']
        elif any(word in name_lower for word in ['artifact']):
            return ['Artifact']
        else:
            return ['Instant']  # Default to instant
    
    def _estimate_mana_cost(self, cmc: int, colors: List[str]) -> str:
        """Estimate mana cost string"""
        if not colors or colors == ['C']:
            return str(cmc) if cmc > 0 else ""
        
        if len(colors) == 1:
            color = colors[0]
            if cmc == 1:
                return color
            else:
                generic = cmc - 1
                return f"{generic}{color}" if generic > 0 else color
        else:
            # Multi-color
            return "".join(colors)
    
    def _initialize_game(self, deck1: List[Card], deck2: List[Card]) -> GameState:
        """Initialize game state"""
        game_state = GameState()
        
        # Shuffle decks
        deck1_copy = deck1.copy()
        deck2_copy = deck2.copy()
        random.shuffle(deck1_copy)
        random.shuffle(deck2_copy)
        
        game_state.libraries = [deck1_copy, deck2_copy]
        
        # Draw opening hands
        for player in range(2):
            for _ in range(7):
                if game_state.libraries[player]:
                    card = game_state.libraries[player].pop(0)
                    game_state.hands[player].append(card)
        
        # Determine starting player (random)
        game_state.active_player = random.randint(0, 1)
        
        return game_state
    
    def _play_game(self, game_state: GameState) -> Optional[int]:
        """Play the game and return winner"""
        
        for turn in range(1, self.max_turns + 1):
            game_state.turn = turn
            
            # Each player takes a turn
            for player in range(2):
                game_state.active_player = player
                
                # Draw card (unless first turn for starting player)
                if not (turn == 1 and player == 0):
                    self._draw_card(game_state, player)
                
                # Reset mana and lands played
                game_state.mana_available[player] = 0
                game_state.lands_played[player] = 0
                
                # Add mana from lands
                lands_in_play = [card for card in game_state.battlefields[player] if 'Land' in card.types]
                game_state.mana_available[player] = len(lands_in_play)
                
                # Simple AI: play cards randomly
                self._simple_ai_turn(game_state, player)
                
                # Check win condition
                winner = self._check_win_condition(game_state)
                if winner is not None:
                    return winner
        
        # Game went to max turns - determine winner by life total
        if game_state.life_totals[0] > game_state.life_totals[1]:
            return 0
        elif game_state.life_totals[1] > game_state.life_totals[0]:
            return 1
        else:
            return None  # Draw
    
    def _draw_card(self, game_state: GameState, player: int):
        """Player draws a card"""
        if game_state.libraries[player]:
            card = game_state.libraries[player].pop(0)
            game_state.hands[player].append(card)
        else:
            # Empty library - player loses
            game_state.life_totals[player] = 0
    
    def _simple_ai_turn(self, game_state: GameState, player: int):
        """Simple AI takes a turn"""
        hand = game_state.hands[player]
        mana = game_state.mana_available[player]
        
        # Play lands first
        lands_in_hand = [card for card in hand if 'Land' in card.types]
        if lands_in_hand and game_state.lands_played[player] < 1:
            land = lands_in_hand[0]
            hand.remove(land)
            game_state.battlefields[player].append(land)
            game_state.lands_played[player] += 1
            game_state.mana_available[player] += 1
            mana = game_state.mana_available[player]
        
        # Play spells by mana cost (cheapest first)
        playable_spells = [card for card in hand if 'Land' not in card.types and card.cmc <= mana]
        playable_spells.sort(key=lambda x: x.cmc)
        
        for spell in playable_spells:
            if mana >= spell.cmc:
                hand.remove(spell)
                
                if 'Creature' in spell.types:
                    # Creature goes to battlefield
                    game_state.battlefields[player].append(spell)
                else:
                    # Spell goes to graveyard, might have effect
                    game_state.graveyards[player].append(spell)
                    self._resolve_spell(game_state, spell, player)
                
                mana -= spell.cmc
                game_state.mana_available[player] = mana
        
        # Attack with creatures (simplified)
        self._attack_phase(game_state, player)
        
        # Discard to hand size
        while len(game_state.hands[player]) > self.hand_size_limit:
            discarded = random.choice(game_state.hands[player])
            game_state.hands[player].remove(discarded)
            game_state.graveyards[player].append(discarded)
    
    def _resolve_spell(self, game_state: GameState, spell: Card, caster: int):
        """Resolve spell effects (simplified)"""
        opponent = 1 - caster
        
        # Simple damage spells
        if any(word in spell.name.lower() for word in ['bolt', 'shock', 'burn']):
            damage = 3 if 'bolt' in spell.name.lower() else 2
            game_state.life_totals[opponent] -= damage
        
        # Simple removal spells
        elif any(word in spell.name.lower() for word in ['murder', 'destroy', 'kill']):
            opponent_creatures = [card for card in game_state.battlefields[opponent] if 'Creature' in card.types]
            if opponent_creatures:
                target = random.choice(opponent_creatures)
                game_state.battlefields[opponent].remove(target)
                game_state.graveyards[opponent].append(target)
    
    def _attack_phase(self, game_state: GameState, attacker: int):
        """Simple attack phase"""
        defender = 1 - attacker
        
        attacking_creatures = [card for card in game_state.battlefields[attacker] if 'Creature' in card.types]
        
        for creature in attacking_creatures:
            if creature.power and creature.power > 0:
                # Simple: all damage goes to player (no blocking)
                game_state.life_totals[defender] -= creature.power
    
    def _check_win_condition(self, game_state: GameState) -> Optional[int]:
        """Check if anyone has won"""
        for player in range(2):
            if game_state.life_totals[player] <= 0:
                return 1 - player  # Opponent wins
        
        return None

class DeckEvaluator:
    """Higher-level deck evaluation using simulation"""
    
    def __init__(self):
        self.simulator = SimpleMTGSimulator()
    
    def evaluate_deck_vs_meta(self, deck: List[Dict], meta_decks: List[List[Dict]], num_games_per_matchup: int = 50) -> Dict:
        """Evaluate deck against current meta"""
        results = {}
        total_wins = 0
        total_games = 0
        
        for i, meta_deck in enumerate(meta_decks):
            matchup_result = self.simulator.simulate_matches(deck, meta_deck, num_games_per_matchup)
            
            results[f"meta_deck_{i}"] = matchup_result
            total_wins += matchup_result['deck1_wins']
            total_games += matchup_result['total_games']
        
        overall_winrate = total_wins / total_games if total_games > 0 else 0
        
        return {
            'overall_winrate': overall_winrate,
            'total_games': total_games,
            'matchup_results': results,
            'evaluation_summary': self._generate_evaluation_summary(results)
        }
    
    def _generate_evaluation_summary(self, results: Dict) -> str:
        """Generate human-readable evaluation summary"""
        summary_parts = []
        
        winrates = [matchup['deck1_winrate'] for matchup in results.values()]
        avg_winrate = np.mean(winrates) if winrates else 0
        
        summary_parts.append(f"Average winrate across meta: {avg_winrate:.1%}")
        
        # Find best and worst matchups
        if results:
            best_matchup = max(results.items(), key=lambda x: x[1]['deck1_winrate'])
            worst_matchup = min(results.items(), key=lambda x: x[1]['deck1_winrate'])
            
            summary_parts.append(f"Best matchup: {best_matchup[1]['deck1_winrate']:.1%}")
            summary_parts.append(f"Worst matchup: {worst_matchup[1]['deck1_winrate']:.1%}")
        
        return "; ".join(summary_parts)
    
    def evaluate_deck_consistency(self, deck: List[Dict], num_games: int = 100) -> Dict:
        """Evaluate deck's internal consistency"""
        # Simulate games against itself to test consistency
        self_matchup = self.simulator.simulate_matches(deck, deck, num_games)
        
        # Analyze opening hands
        opening_hand_stats = self._analyze_opening_hands(deck, num_samples=1000)
        
        return {
            'self_matchup': self_matchup,
            'opening_hands': opening_hand_stats,
            'consistency_score': self._calculate_consistency_score(opening_hand_stats)
        }
    
    def _analyze_opening_hands(self, deck: List[Dict], num_samples: int = 1000) -> Dict:
        """Analyze opening hand quality"""
        cards = self.simulator._build_deck(deck)
        
        land_counts = []
        curve_counts = []
        
        for _ in range(num_samples):
            # Simulate opening hand
            deck_copy = cards.copy()
            random.shuffle(deck_copy)
            opening_hand = deck_copy[:7]
            
            # Count lands
            lands = len([card for card in opening_hand if 'Land' in card.types])
            land_counts.append(lands)
            
            # Count playable spells by turn 3
            playable_early = len([card for card in opening_hand if 'Land' not in card.types and card.cmc <= 3])
            curve_counts.append(playable_early)
        
        return {
            'avg_lands': np.mean(land_counts),
            'land_variance': np.var(land_counts),
            'avg_early_plays': np.mean(curve_counts),
            'hands_with_2_3_lands': sum(1 for count in land_counts if 2 <= count <= 4) / len(land_counts),
            'hands_with_plays': sum(1 for count in curve_counts if count > 0) / len(curve_counts)
        }
    
    def _calculate_consistency_score(self, opening_hand_stats: Dict) -> float:
        """Calculate overall consistency score (0-1)"""
        land_score = opening_hand_stats['hands_with_2_3_lands']
        plays_score = opening_hand_stats['hands_with_plays']
        
        return (land_score + plays_score) / 2

if __name__ == "__main__":
    # Example usage
    evaluator = DeckEvaluator()
    
    # Example decks (simplified format)
    test_deck1 = [
        {'name': 'Lightning Bolt', 'quantity': 4},
        {'name': 'Mountain', 'quantity': 20},
        {'name': 'Red Dragon', 'quantity': 4}
    ]
    
    test_deck2 = [
        {'name': 'Counterspell', 'quantity': 4},
        {'name': 'Island', 'quantity': 20},
        {'name': 'Blue Angel', 'quantity': 4}
    ]
    
    # Simulate matchup
    result = evaluator.simulator.simulate_matches(test_deck1, test_deck2, 10)
    print(f"Deck 1 winrate: {result['deck1_winrate']:.1%}")
    
    # Evaluate consistency
    consistency = evaluator.evaluate_deck_consistency(test_deck1, 100)
    print(f"Consistency score: {consistency['consistency_score']:.2f}")