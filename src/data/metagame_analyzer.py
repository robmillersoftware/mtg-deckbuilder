import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetagameAnalyzer:
    def __init__(self, data_dir: str = "data/raw", card_database=None):
        self.data_dir = data_dir
        self.deck_data = []
        self.card_frequencies = Counter()
        self.archetype_stats = defaultdict(list)
        self.card_database = card_database
        
    def load_scraped_data(self, filename: Optional[str] = None) -> bool:
        """Load scraped MTGTop8 data"""
        try:
            if filename:
                filepath = os.path.join(self.data_dir, filename)
            else:
                # Find the latest file
                files = [f for f in os.listdir(self.data_dir) if f.startswith('mtgtop8_standard_')]
                if not files:
                    logger.error("No scraped data found")
                    return False
                
                latest_file = sorted(files)[-1]
                filepath = os.path.join(self.data_dir, latest_file)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.deck_data = data.get('decks', [])
            logger.info(f"Loaded {len(self.deck_data)} decks from {os.path.basename(filepath)}")
            
            # Process the data
            self._process_deck_data()
            return True
            
        except Exception as e:
            logger.error(f"Error loading scraped data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _process_deck_data(self):
        """Process deck data for analysis"""
        self.card_frequencies = Counter()
        self.archetype_stats = defaultdict(list)
        
        for deck in self.deck_data:
            # Count card frequencies
            for card in deck.get('mainboard', []):
                card_name = card['name']
                quantity = card['quantity']
                self.card_frequencies[card_name] += quantity
            
            # Classify archetype if not already set or if it's generic
            archetype = deck.get('archetype', 'Unknown')
            if not archetype or archetype == 'Unknown' or len(archetype) < 3:
                archetype = self._classify_deck_archetype(deck)
                deck['archetype'] = archetype
            
            # Clean up archetype names
            archetype = self._clean_archetype_name(archetype)
            self.archetype_stats[archetype].append(deck)
    
    def _classify_deck_archetype(self, deck: Dict) -> str:
        """Classify deck archetype based on actual card data"""
        mainboard = deck.get('mainboard', [])
        if not mainboard:
            return 'Unknown'
        
        # Count different card types using actual card data
        creatures = 0
        instants_sorceries = 0
        lands = 0
        artifacts = 0
        enchantments = 0
        planeswalkers = 0
        
        # Track mana curve and colors
        cmc_distribution = [0] * 8  # 0-7+ mana costs
        colors = set()
        
        for card_entry in mainboard:
            card_name = card_entry['name']
            quantity = card_entry['quantity']
            
            # Get actual card data
            if self.card_database:
                card_data = self.card_database.get_card_by_name(card_name)
                if card_data:
                    card_types = card_data.get('types', [])
                    cmc = card_data.get('cmc', 0)
                    card_colors = card_data.get('colors', [])
                    
                    # Count by actual type
                    if 'Land' in card_types:
                        lands += quantity
                    elif 'Creature' in card_types:
                        creatures += quantity
                    elif 'Instant' in card_types or 'Sorcery' in card_types:
                        instants_sorceries += quantity
                    elif 'Artifact' in card_types:
                        artifacts += quantity
                    elif 'Enchantment' in card_types:
                        enchantments += quantity
                    elif 'Planeswalker' in card_types:
                        planeswalkers += quantity
                    
                    # Track mana curve
                    cmc_slot = min(int(cmc), 7)
                    cmc_distribution[cmc_slot] += quantity
                    
                    # Track colors
                    colors.update(card_colors)
                else:
                    # Fallback for cards not in database
                    instants_sorceries += quantity
            else:
                # No database available, use simple heuristics
                instants_sorceries += quantity
        
        # Classify based on composition and curve
        total_nonland = creatures + instants_sorceries + artifacts + enchantments + planeswalkers
        
        if total_nonland == 0:
            return 'Unknown'
        
        creature_ratio = creatures / total_nonland
        spell_ratio = instants_sorceries / total_nonland
        
        # Determine strategy based on curve
        low_curve = sum(cmc_distribution[0:3])  # 0-2 mana
        total_curve = sum(cmc_distribution[1:])  # 1+ mana (exclude lands)
        
        if total_curve > 0:
            low_curve_ratio = low_curve / total_curve
        else:
            low_curve_ratio = 0
        
        # Color-based naming
        color_name = ""
        if len(colors) == 1:
            color_map = {'W': 'White', 'U': 'Blue', 'B': 'Black', 'R': 'Red', 'G': 'Green'}
            color_name = color_map.get(list(colors)[0], '')
        elif len(colors) == 2:
            guild_names = {
                ('W', 'U'): 'Azorius', ('U', 'B'): 'Dimir', ('B', 'R'): 'Rakdos',
                ('R', 'G'): 'Gruul', ('G', 'W'): 'Selesnya', ('W', 'B'): 'Orzhov',
                ('U', 'R'): 'Izzet', ('B', 'G'): 'Golgari', ('R', 'W'): 'Boros',
                ('G', 'U'): 'Simic'
            }
            sorted_colors = tuple(sorted(colors))
            color_name = guild_names.get(sorted_colors, f"{len(colors)}-Color")
        elif len(colors) >= 3:
            color_name = f"{len(colors)}-Color"
        
        # Strategy classification
        if creature_ratio > 0.6:
            # Creature-heavy
            if low_curve_ratio > 0.7:
                strategy = "Aggro"
            elif low_curve_ratio < 0.3:
                strategy = "Ramp"
            else:
                strategy = "Midrange"
        elif spell_ratio > 0.6:
            # Spell-heavy
            if low_curve_ratio > 0.7:
                strategy = "Tempo"
            elif low_curve_ratio < 0.3:
                strategy = "Control"
            else:
                strategy = "Midrange"
        else:
            # Balanced
            if low_curve_ratio > 0.6:
                strategy = "Midrange"
            else:
                strategy = "Control"
        
        # Combine color and strategy
        if color_name:
            return f"{color_name} {strategy}"
        else:
            return strategy
    
    def _clean_archetype_name(self, archetype: str) -> str:
        """Clean and normalize archetype names"""
        if not archetype or len(archetype) < 3:
            return 'Unknown'
        
        # Remove common prefixes/suffixes
        archetype = archetype.strip()
        
        # Common cleanup patterns
        replacements = {
            'mono red': 'Mono Red',
            'mono blue': 'Mono Blue', 
            'mono white': 'Mono White',
            'mono black': 'Mono Black',
            'mono green': 'Mono Green',
            'red aggro': 'Red Aggro',
            'blue control': 'Blue Control',
            'white aggro': 'White Aggro'
        }
        
        archetype_lower = archetype.lower()
        for old, new in replacements.items():
            if old in archetype_lower:
                return new
        
        # Capitalize first letters
        return ' '.join(word.capitalize() for word in archetype.split())
    
    def get_top_cards(self, limit: int = 50, exclude_lands: bool = True) -> List[Tuple[str, int]]:
        """Get most played cards in the format"""
        filtered_cards = []
        
        for card_name, count in self.card_frequencies.most_common():
            if exclude_lands and self._is_land_card(card_name):
                continue
            
            filtered_cards.append((card_name, count))
            
            if len(filtered_cards) >= limit:
                break
        
        return filtered_cards
    
    def _is_land_card(self, card_name: str) -> bool:
        """Check if a card is a land using card database"""
        if self.card_database:
            card_data = self.card_database.get_card_by_name(card_name)
            if card_data:
                return 'Land' in card_data.get('types', [])
        
        # Fallback to basic land names if no database
        basic_lands = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes'}
        return card_name in basic_lands
    
    def get_archetype_breakdown(self) -> Dict[str, Dict]:
        """Get detailed archetype statistics"""
        breakdown = {}
        
        total_decks = len(self.deck_data)
        
        for archetype, decks in self.archetype_stats.items():
            deck_count = len(decks)
            percentage = (deck_count / total_decks) * 100 if total_decks > 0 else 0
            
            # Calculate average deck composition
            all_cards = []
            for deck in decks:
                for card in deck.get('mainboard', []):
                    all_cards.extend([card['name']] * card['quantity'])
            
            card_counts = Counter(all_cards)
            avg_deck_size = len(all_cards) / len(decks) if decks else 0
            
            # Filter key cards to exclude lands
            filtered_key_cards = []
            for card_name, count in card_counts.most_common():
                if not self._is_land_card(card_name):
                    filtered_key_cards.append((card_name, count))
                if len(filtered_key_cards) >= 10:
                    break
            
            breakdown[archetype] = {
                'deck_count': deck_count,
                'percentage': round(percentage, 2),
                'avg_deck_size': round(avg_deck_size, 1),
                'key_cards': filtered_key_cards,
                'sample_decks': decks[:3]  # Store sample decks
            }
        
        return breakdown
    
    def analyze_mana_curves(self) -> Dict[str, List[float]]:
        """Analyze mana curves by archetype"""
        mana_curves = {}
        
        for archetype, decks in self.archetype_stats.items():
            curves = []
            
            for deck in decks:
                curve = [0] * 8  # 0-7+ mana costs
                
                for card in deck.get('mainboard', []):
                    # This would need card database integration to get CMC
                    # For now, we'll estimate based on common patterns
                    card_name = card['name']
                    quantity = card['quantity']
                    
                    # Simplified CMC estimation (would be better with card database)
                    estimated_cmc = self._estimate_cmc(card_name)
                    cmc_slot = min(estimated_cmc, 7)
                    curve[cmc_slot] += quantity
                
                # Normalize to percentages
                total_spells = sum(curve)
                if total_spells > 0:
                    curve = [count / total_spells for count in curve]
                
                curves.append(curve)
            
            # Average curve for archetype
            if curves:
                avg_curve = [sum(slot) / len(curves) for slot in zip(*curves)]
                mana_curves[archetype] = avg_curve
        
        return mana_curves
    
    def _estimate_cmc(self, card_name: str) -> int:
        """Rough CMC estimation based on card name patterns"""
        # This is a simplified heuristic - in practice, use the card database
        name_lower = card_name.lower()
        
        # Common patterns
        if any(word in name_lower for word in ['bolt', 'shock', 'path']):
            return 1
        elif any(word in name_lower for word in ['counterspell', 'cancel', 'negate']):
            return 2
        elif any(word in name_lower for word in ['murder', 'cancel']):
            return 3
        elif any(word in name_lower for word in ['wrath', 'sweeper']):
            return 4
        elif any(word in name_lower for word in ['dragon', 'angel', 'demon']):
            return 5
        else:
            return 3  # Default estimate
    
    def find_deck_clusters(self, n_clusters: int = 5) -> Dict:
        """Cluster decks by similarity"""
        if not self.deck_data:
            return {}
        
        try:
            # Convert decks to text representations
            deck_texts = []
            for deck in self.deck_data:
                cards = []
                for card in deck.get('mainboard', []):
                    cards.extend([card['name']] * card['quantity'])
                deck_texts.append(' '.join(cards))
            
            # Vectorize decks
            vectorizer = TfidfVectorizer(max_features=1000)
            deck_vectors = vectorizer.fit_transform(deck_texts)
            
            # Cluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(deck_vectors)
            
            # Organize results
            cluster_info = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                cluster_info[cluster_id].append(self.deck_data[i])
            
            # Analyze clusters
            cluster_analysis = {}
            for cluster_id, decks in cluster_info.items():
                # Find most common cards in cluster
                all_cards = []
                archetypes = []
                
                for deck in decks:
                    for card in deck.get('mainboard', []):
                        all_cards.extend([card['name']] * card['quantity'])
                    archetypes.append(deck.get('archetype', 'Unknown'))
                
                common_cards = Counter(all_cards).most_common(10)
                common_archetypes = Counter(archetypes).most_common(3)
                
                cluster_analysis[f"Cluster_{cluster_id}"] = {
                    'deck_count': len(decks),
                    'common_cards': common_cards,
                    'archetypes': common_archetypes,
                    'sample_decks': decks[:2]
                }
            
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"Error clustering decks: {e}")
            return {}
    
    def analyze_meta_trends(self, time_window_days: int = 30) -> Dict:
        """Analyze metagame trends over time"""
        trends = {
            'rising_cards': [],
            'falling_cards': [],
            'stable_archetypes': [],
            'emerging_archetypes': []
        }
        
        # This would need timestamp data from scraping
        # For now, return basic analysis
        
        # Find cards that appear in many different archetypes (versatile)
        card_archetype_map = defaultdict(set)
        
        for deck in self.deck_data:
            archetype = deck.get('archetype', 'Unknown')
            for card in deck.get('mainboard', []):
                card_archetype_map[card['name']].add(archetype)
        
        # Versatile cards (appear in 3+ archetypes)
        versatile_cards = [(card, len(archetypes)) for card, archetypes in card_archetype_map.items() if len(archetypes) >= 3]
        versatile_cards.sort(key=lambda x: x[1], reverse=True)
        
        trends['versatile_cards'] = versatile_cards[:10]
        
        return trends
    
    def get_sideboard_analysis(self) -> Dict[str, Dict]:
        """Analyze common sideboard cards by archetype"""
        sideboard_stats = {}
        
        for archetype, decks in self.archetype_stats.items():
            all_sideboard_cards = []
            
            for deck in decks:
                for card in deck.get('sideboard', []):
                    all_sideboard_cards.extend([card['name']] * card['quantity'])
            
            if all_sideboard_cards:
                common_sb_cards = Counter(all_sideboard_cards).most_common(10)
                sideboard_stats[archetype] = {
                    'total_sb_cards': len(all_sideboard_cards),
                    'common_cards': common_sb_cards,
                    'avg_sb_size': len(all_sideboard_cards) / len(decks)
                }
        
        return sideboard_stats
    
    def export_analysis(self, output_dir: str = "data/processed") -> str:
        """Export complete metagame analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        analysis = {
            'generated_at': datetime.now().isoformat(),
            'deck_count': len(self.deck_data),
            'top_cards': self.get_top_cards(30),
            'archetype_breakdown': self.get_archetype_breakdown(),
            'mana_curves': self.analyze_mana_curves(),
            'deck_clusters': self.find_deck_clusters(),
            'meta_trends': self.analyze_meta_trends(),
            'sideboard_analysis': self.get_sideboard_analysis()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metagame_analysis_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Exported metagame analysis to {filepath}")
        return filepath
    
    def get_deck_to_beat(self) -> Optional[Dict]:
        """Identify the most dominant deck in the format"""
        if not self.archetype_stats:
            return None
        
        # Find most popular successful archetype
        archetype_breakdown = self.get_archetype_breakdown()
        
        # Sort by percentage
        sorted_archetypes = sorted(
            archetype_breakdown.items(), 
            key=lambda x: x[1]['percentage'], 
            reverse=True
        )
        
        if sorted_archetypes:
            top_archetype = sorted_archetypes[0]
            return {
                'name': top_archetype[0],
                'stats': top_archetype[1]
            }
        
        return None

if __name__ == "__main__":
    analyzer = MetagameAnalyzer()
    analyzer.load_scraped_data()
    
    # Run analysis
    analysis_file = analyzer.export_analysis()
    print(f"Analysis exported to: {analysis_file}")
    
    # Show deck to beat
    deck_to_beat = analyzer.get_deck_to_beat()
    if deck_to_beat:
        print(f"\nDeck to Beat: {deck_to_beat['name']}")
        print(f"Percentage: {deck_to_beat['stats']['percentage']}%")