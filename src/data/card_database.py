import requests
import json
import os
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import aiohttp
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CardDatabase:
    def __init__(self, cards_dir: str = "data/cards"):
        self.cards_dir = cards_dir
        self.standard_cards = {}
        self.card_name_to_id = {}
        os.makedirs(cards_dir, exist_ok=True)
        
    def download_standard_cards(self) -> str:
        """Download Standard-legal cards from MTGJSON"""
        try:
            logger.info("Downloading Standard cards from MTGJSON...")
            url = "https://mtgjson.com/api/v5/Standard.json"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Save raw data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_filepath = os.path.join(self.cards_dir, f"standard_raw_{timestamp}.json")
            
            with open(raw_filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Process and normalize cards
            processed_cards = self._process_standard_cards(data)
            
            # Save processed data
            processed_filepath = os.path.join(self.cards_dir, f"standard_processed_{timestamp}.json")
            with open(processed_filepath, 'w') as f:
                json.dump(processed_cards, f, indent=2)
            
            # Update current standard cards
            self.standard_cards = processed_cards
            self._build_name_index()
            
            logger.info(f"Downloaded and processed {len(processed_cards)} Standard cards")
            return processed_filepath
            
        except Exception as e:
            logger.error(f"Error downloading Standard cards: {e}")
            return ""
    
    def _process_standard_cards(self, mtgjson_data: Dict) -> Dict[str, Dict]:
        """Process raw MTGJSON data into normalized card format"""
        processed_cards = {}
        
        try:
            sets_data = mtgjson_data.get('data', {})
            
            for set_code, set_info in sets_data.items():
                cards = set_info.get('cards', [])
                
                for card in cards:
                    # Skip non-playable cards
                    if not self._is_playable_card(card):
                        continue
                    
                    card_id = card.get('uuid', '')
                    if not card_id:
                        continue
                    
                    # Normalize card data
                    normalized_card = self._normalize_card(card, set_code)
                    processed_cards[card_id] = normalized_card
                    
        except Exception as e:
            logger.error(f"Error processing cards: {e}")
        
        return processed_cards
    
    def _is_playable_card(self, card: Dict) -> bool:
        """Check if a card is playable in constructed formats"""
        # Skip tokens, emblems, schemes, etc.
        skip_types = ['Token', 'Emblem', 'Scheme', 'Plane', 'Phenomenon', 'Vanguard']
        card_types = card.get('types', [])
        
        if any(skip_type in card_types for skip_type in skip_types):
            return False
        
        # Must have a name
        if not card.get('name'):
            return False
        
        return True
    
    def _normalize_card(self, card: Dict, set_code: str) -> Dict:
        """Normalize card data to standard format"""
        return {
            'id': card.get('uuid', ''),
            'name': card.get('name', ''),
            'mana_cost': card.get('manaCost', ''),
            'cmc': card.get('manaValue', 0),
            'colors': card.get('colors', []),
            'color_identity': card.get('colorIdentity', []),
            'type_line': card.get('type', ''),
            'types': card.get('types', []),
            'subtypes': card.get('subtypes', []),
            'supertypes': card.get('supertypes', []),
            'oracle_text': card.get('text', ''),
            'power': card.get('power'),
            'toughness': card.get('toughness'),
            'loyalty': card.get('loyalty'),
            'rarity': card.get('rarity', ''),
            'set_code': set_code,
            'set_name': card.get('setName', ''),
            'collector_number': card.get('number', ''),
            'legalities': card.get('legalities', {}),
            'keywords': card.get('keywords', []),
            'flavor_text': card.get('flavorText', ''),
            'artist': card.get('artist', ''),
            'layout': card.get('layout', ''),
            'is_reserved': card.get('isReserved', False),
            'edhrec_rank': card.get('edhrecRank'),
            'multiverse_ids': card.get('multiverseIds', []),
            'mtgo_id': card.get('mtgoId'),
            'arena_id': card.get('mtgArenaId')
        }
    
    def _build_name_index(self):
        """Build index for fast card name lookups"""
        self.card_name_to_id = {}
        for card_id, card in self.standard_cards.items():
            name = card['name'].lower()
            if name not in self.card_name_to_id:
                self.card_name_to_id[name] = []
            self.card_name_to_id[name].append(card_id)
    
    def get_card_by_name(self, name: str) -> Optional[Dict]:
        """Get card by name"""
        name_lower = name.lower()
        card_ids = self.card_name_to_id.get(name_lower, [])
        
        if card_ids:
            # Return the first match (could be improved to handle multiple printings)
            return self.standard_cards.get(card_ids[0])
        
        return None
    
    def get_cards_by_type(self, card_type: str) -> List[Dict]:
        """Get all cards of a specific type"""
        matching_cards = []
        for card in self.standard_cards.values():
            if card_type.lower() in [t.lower() for t in card['types']]:
                matching_cards.append(card)
        return matching_cards
    
    def get_cards_by_color(self, colors: List[str]) -> List[Dict]:
        """Get cards by color identity"""
        matching_cards = []
        for card in self.standard_cards.values():
            card_colors = set(card['color_identity'])
            target_colors = set(colors)
            
            if card_colors.issubset(target_colors):
                matching_cards.append(card)
        
        return matching_cards
    
    def search_cards(self, query: str, limit: int = 50) -> List[Dict]:
        """Search cards by name or text"""
        query_lower = query.lower()
        matching_cards = []
        
        for card in self.standard_cards.values():
            # Search in name and oracle text
            if (query_lower in card['name'].lower() or 
                query_lower in card['oracle_text'].lower()):
                matching_cards.append(card)
                
                if len(matching_cards) >= limit:
                    break
        
        return matching_cards
    
    def load_latest_standard_cards(self) -> bool:
        """Load the most recent Standard cards data"""
        try:
            # Find the latest processed file
            files = [f for f in os.listdir(self.cards_dir) if f.startswith('standard_processed_')]
            if not files:
                logger.warning("No processed Standard cards found. Run download_standard_cards() first.")
                return False
            
            latest_file = sorted(files)[-1]
            filepath = os.path.join(self.cards_dir, latest_file)
            
            with open(filepath, 'r') as f:
                self.standard_cards = json.load(f)
            
            self._build_name_index()
            logger.info(f"Loaded {len(self.standard_cards)} Standard cards from {latest_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Standard cards: {e}")
            return False
    
    def get_standard_legal_cards(self) -> List[Dict]:
        """Get all Standard-legal cards"""
        return list(self.standard_cards.values())
    
    def validate_deck_legality(self, decklist: List[Dict]) -> Dict:
        """Validate if a deck is Standard-legal"""
        issues = []
        card_counts = {}
        basic_lands = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes'}
        
        for entry in decklist:
            card_name = entry['name']
            quantity = entry['quantity']
            
            # Check if card exists and is Standard legal
            card = self.get_card_by_name(card_name)
            if not card:
                issues.append(f"Card '{card_name}' not found in Standard")
                continue
            
            # Count total copies
            if card_name in card_counts:
                card_counts[card_name] += quantity
            else:
                card_counts[card_name] = quantity
            
            # Check quantity limits (4-of rule, except basic lands)
            if card_name not in basic_lands and card_counts[card_name] > 4:
                issues.append(f"Too many copies of '{card_name}': {card_counts[card_name]} (max 4)")
        
        # Check deck size (60 minimum for constructed)
        total_cards = sum(entry['quantity'] for entry in decklist if entry.get('section', 'mainboard') == 'mainboard')
        if total_cards < 60:
            issues.append(f"Deck too small: {total_cards} cards (minimum 60)")
        
        return {
            'is_legal': len(issues) == 0,
            'issues': issues,
            'total_cards': total_cards,
            'unique_cards': len(card_counts)
        }

class ScryfallAPI:
    def __init__(self):
        self.base_url = "https://api.scryfall.com"
        self.session = requests.Session()
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    async def search_cards_async(self, query: str, limit: int = 175) -> List[Dict]:
        """Asynchronously search cards using Scryfall API"""
        try:
            url = f"{self.base_url}/cards/search"
            params = {
                'q': query,
                'format': 'json',
                'page': 1
            }
            
            all_cards = []
            
            async with aiohttp.ClientSession() as session:
                while len(all_cards) < limit:
                    async with session.get(url, params=params) as response:
                        if response.status != 200:
                            break
                            
                        data = await response.json()
                        cards = data.get('data', [])
                        all_cards.extend(cards)
                        
                        if not data.get('has_more', False):
                            break
                            
                        params['page'] += 1
                        await asyncio.sleep(self.rate_limit_delay)
            
            return all_cards[:limit]
            
        except Exception as e:
            logger.error(f"Error searching Scryfall: {e}")
            return []
    
    def get_card_by_name(self, name: str) -> Optional[Dict]:
        """Get card details by exact name"""
        try:
            url = f"{self.base_url}/cards/named"
            params = {'exact': name}
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching card '{name}' from Scryfall: {e}")
            return None

if __name__ == "__main__":
    db = CardDatabase()
    db.download_standard_cards()
    
    # Test search
    results = db.search_cards("lightning", limit=5)
    for card in results:
        print(f"{card['name']} - {card['mana_cost']} - {card['type_line']}")