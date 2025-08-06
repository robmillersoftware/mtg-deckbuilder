import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import openai
import anthropic
from collections import Counter
import random
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CardEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.card_embeddings = {}
        self.card_index = None
        self.card_list = []
        
    def generate_card_embeddings(self, cards: List[Dict]) -> None:
        """Generate embeddings for all cards"""
        logger.info(f"Generating embeddings for {len(cards)} cards...")
        
        card_descriptions = []
        self.card_list = []
        
        for card in cards:
            # Create rich description for embedding
            description = self._create_card_description(card)
            card_descriptions.append(description)
            self.card_list.append(card)
        
        # Generate embeddings
        embeddings = self.model.encode(card_descriptions)
        
        # Build FAISS index for fast similarity search
        dimension = embeddings.shape[1]
        self.card_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.card_index.add(embeddings.astype('float32'))
        
        logger.info("Card embeddings generated and indexed")
    
    def _create_card_description(self, card: Dict) -> str:
        """Create a rich text description for embedding"""
        parts = []
        
        # Name and type
        parts.append(f"Card name: {card['name']}")
        parts.append(f"Type: {card['type_line']}")
        
        # Mana cost and colors
        if card['mana_cost']:
            parts.append(f"Mana cost: {card['mana_cost']}")
        if card['colors']:
            parts.append(f"Colors: {', '.join(card['colors'])}")
        
        # Power/toughness for creatures
        if card['power'] and card['toughness']:
            parts.append(f"Power/Toughness: {card['power']}/{card['toughness']}")
        
        # Oracle text (main functionality)
        if card['oracle_text']:
            parts.append(f"Effect: {card['oracle_text']}")
        
        # Keywords for quick reference
        if card['keywords']:
            parts.append(f"Keywords: {', '.join(card['keywords'])}")
        
        # Rarity and set info
        parts.append(f"Rarity: {card['rarity']}")
        
        return " | ".join(parts)
    
    def find_similar_cards(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """Find cards similar to query description"""
        if not self.card_index:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.card_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.card_list):
                results.append((self.card_list[idx], float(score)))
        
        return results
    
    def find_cards_by_role(self, role: str, colors: List[str] = None, k: int = 5) -> List[Dict]:
        """Find cards that fulfill a specific role"""
        # Create role-based query
        role_queries = {
            'removal': 'destroy target creature or planeswalker, exile, removal spell',
            'counterspell': 'counter target spell, negate, cancel',
            'card_draw': 'draw cards, card advantage, refill hand',
            'ramp': 'add mana, search for lands, mana acceleration',
            'aggressive_creature': 'low cost creature with high power, haste, aggressive stats',
            'defensive_creature': 'high toughness, defender, blocks well',
            'finisher': 'high power creature, win condition, big threat',
            'utility': 'versatile effect, modal spell, flexible',
            'sweeper': 'destroy all creatures, board wipe, mass removal'
        }
        
        query = role_queries.get(role.lower(), role)
        
        # Add color restrictions to query if specified
        if colors:
            color_text = ' '.join(colors)
            query = f"{query} {color_text} mana"
        
        similar_cards = self.find_similar_cards(query, k * 2)  # Get more for filtering
        
        # Filter by colors if specified
        if colors:
            filtered_cards = []
            for card, score in similar_cards:
                card_colors = set(card.get('color_identity', []))
                target_colors = set(colors)
                
                if card_colors.issubset(target_colors):
                    filtered_cards.append(card)
                
                if len(filtered_cards) >= k:
                    break
            
            return filtered_cards
        else:
            return [card for card, score in similar_cards[:k]]

class LLMDeckGenerator:
    def __init__(self, card_embeddings: CardEmbeddings):
        self.card_embeddings = card_embeddings
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize clients if API keys are available
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
    
    def generate_deck(self, 
                     prompt: str,
                     colors: List[str] = None,
                     archetype: str = None,
                     meta_context: Dict = None,
                     must_include: List[str] = None) -> Dict:
        """Generate a deck using LLM with full card database"""
        
        # Let the LLM do the heavy lifting with all the context
        deck_data = self._generate_deck_with_llm(prompt, colors, archetype, meta_context)
        
        return deck_data
    
    def _generate_deck_with_llm(self, prompt: str, colors: List[str], archetype: str, meta_context: Dict) -> Dict:
        """Generate complete deck using LLM with full context"""
        
        try:
            # Get sample of relevant cards for context
            card_pool = self._get_relevant_cards_for_llm(prompt, colors)
            
            # Build comprehensive prompt
            llm_prompt = self._build_llm_deck_prompt(prompt, colors, archetype, meta_context, card_pool)
            
            # Query LLM for deck
            response = self._query_llm(llm_prompt)
            
            # Parse response into deck format
            deck_data = self._parse_llm_deck_response(response)
            
            return deck_data
            
        except Exception as e:
            # Return error deck instead of fallback
            return {
                'mainboard': [],
                'sideboard': [],
                'concept': {'strategy': f'Deck generation failed: {str(e)}'},
                'total_cards': 0,
                'colors': [],
                'archetype': 'Error',
                'error': str(e)
            }
    
    def _get_relevant_cards_for_llm(self, prompt: str, colors: List[str], max_cards: int = 200) -> List[Dict]:
        """Get relevant cards to include in LLM prompt"""
        relevant_cards = []
        
        if not hasattr(self.card_embeddings, 'card_list'):
            return relevant_cards
        
        # If colors specified, filter by color
        if colors:
            color_filtered = []
            for card in self.card_embeddings.card_list:
                card_colors = set(card.get('colors', []))
                target_colors = set(colors)
                # Include colorless cards and cards within color identity
                if not card_colors or card_colors.issubset(target_colors):
                    color_filtered.append(card)
            relevant_cards = color_filtered[:max_cards]
        else:
            # Use similarity search to find relevant cards
            similar_cards = self.card_embeddings.find_similar_cards(prompt, k=max_cards)
            relevant_cards = [card for card, score in similar_cards]
        
        return relevant_cards
    
    def _build_llm_deck_prompt(self, prompt: str, colors: List[str], archetype: str, meta_context: Dict, card_pool: List[Dict]) -> str:
        """Build comprehensive prompt for LLM deck generation"""
        
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("""You are an expert Magic: The Gathering deck builder. Generate a competitive Standard deck based on the request.""")
        
        # User request
        prompt_parts.append(f"REQUEST: {prompt}")
        
        # Meta context
        if meta_context:
            if meta_context.get('deck_to_beat'):
                deck_to_beat = meta_context['deck_to_beat']
                prompt_parts.append(f"CURRENT META LEADER: {deck_to_beat.get('name')} ({deck_to_beat.get('stats', {}).get('percentage', 0):.1f}%)")
            
            if meta_context.get('top_cards'):
                top_cards = [card for card, count in meta_context['top_cards'][:10]]
                prompt_parts.append(f"POPULAR CARDS: {', '.join(top_cards)}")
        
        # Color constraints
        if colors:
            prompt_parts.append(f"COLORS: {', '.join(colors)}")
        
        if archetype:
            prompt_parts.append(f"ARCHETYPE: {archetype}")
        
        # Available cards (sample)
        prompt_parts.append("\nAVAILABLE CARDS (sample):")
        for card in card_pool[:50]:  # Show first 50 cards
            prompt_parts.append(f"- {card['name']} ({card.get('mana_cost', '')}) - {card['type_line']}")
            if card.get('oracle_text'):
                prompt_parts.append(f"  {card['oracle_text'][:100]}{'...' if len(card.get('oracle_text', '')) > 100 else ''}")
        
        if len(card_pool) > 50:
            prompt_parts.append(f"... and {len(card_pool) - 50} more cards available")
        
        # Instructions
        prompt_parts.append("""
REQUIREMENTS:
1. Build a 60-card mainboard + 15-card sideboard
2. Follow 4-of rule (max 4 copies, except basic lands)
3. Use only Standard-legal cards from the available pool
4. Ensure mana curve and color requirements work
5. Include the specifically requested cards if mentioned
6. Make it competitive against the current meta

FORMAT YOUR RESPONSE AS JSON:
{
  "mainboard": [
    {"name": "Card Name", "quantity": 4},
    ...
  ],
  "sideboard": [
    {"name": "Card Name", "quantity": 2},
    ...
  ],
  "strategy": "Brief description of the deck's strategy",
  "colors": ["U", "B"],
  "archetype": "Control"
}
""")
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_deck_response(self, response: str) -> Dict:
        """Parse LLM response into deck format"""
        try:
            # Try to extract JSON from response
            import json
            import re
            
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                deck_data = json.loads(json_str)
                
                # Validate and normalize
                mainboard = deck_data.get('mainboard', [])
                sideboard = deck_data.get('sideboard', [])
                
                # Calculate totals
                total_main = sum(card.get('quantity', 0) for card in mainboard)
                total_side = sum(card.get('quantity', 0) for card in sideboard)
                
                return {
                    'mainboard': mainboard,
                    'sideboard': sideboard,
                    'concept': {'strategy': deck_data.get('strategy', 'LLM-generated deck')},
                    'total_cards': total_main,
                    'colors': deck_data.get('colors', []),
                    'archetype': deck_data.get('archetype', 'Custom')
                }
        
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response was: {response[:500]}")
        
        # Fallback to empty deck
        return {
            'mainboard': [],
            'sideboard': [],
            'concept': {'strategy': 'Failed to generate deck'},
            'total_cards': 0,
            'colors': [],
            'archetype': 'Unknown'
        }
    
    def _build_generation_context(self, prompt: str, colors: List[str], archetype: str, meta_context: Dict) -> str:
        """Build context for LLM generation"""
        context_parts = []
        
        context_parts.append(f"Generate a Magic: The Gathering Standard deck based on: {prompt}")
        
        if colors:
            context_parts.append(f"Deck colors: {', '.join(colors)}")
        
        if archetype:
            context_parts.append(f"Target archetype: {archetype}")
        
        # Add meta context
        if meta_context:
            if 'deck_to_beat' in meta_context:
                deck_to_beat = meta_context['deck_to_beat']
                context_parts.append(f"Current meta leader: {deck_to_beat.get('name', 'Unknown')}")
            
            if 'top_cards' in meta_context:
                top_cards = [card for card, count in meta_context['top_cards'][:10]]
                context_parts.append(f"Currently popular cards: {', '.join(top_cards)}")
        
        context_parts.append("The deck should be competitive, synergistic, and Standard-legal.")
        context_parts.append("Aim for 60 cards in the mainboard and 15 cards in the sideboard.")
        
        return "\n".join(context_parts)
    
    def _generate_deck_concept(self, context: str) -> Dict:
        """Generate high-level deck concept using LLM"""
        prompt = f"""
        {context}
        
        Please provide a deck concept including:
        1. Overall strategy (2-3 sentences)
        2. Key card roles needed:
           - Win conditions (2-3 types)
           - Removal/interaction (1-2 types)  
           - Card advantage/utility (1-2 types)
           - Mana curve considerations
        3. Suggested mana base approach
        
        Format as JSON with keys: strategy, win_conditions, removal, utility, mana_curve, mana_base
        """
        
        # Try Anthropic first, then OpenAI
        response = self._query_llm(prompt)
        
        try:
            # Parse JSON response
            concept = json.loads(response)
            return concept
        except:
            # Fallback to structured response
            return {
                'strategy': 'Midrange deck focusing on efficient threats and removal',
                'win_conditions': ['Efficient creatures', 'Planeswalkers'],
                'removal': ['Targeted removal', 'Some board control'],
                'utility': ['Card draw', 'Flexible spells'],
                'mana_curve': 'Curve out from 1-5 mana with focus on 2-4',
                'mana_base': 'Consistent 2-3 color mana base'
            }
    
    def _build_deck_iteratively(self, concept: Dict, colors: List[str]) -> Dict:
        """Build deck by finding cards for each role"""
        deck = {'mainboard': [], 'sideboard': []}
        
        # Target quantities for different roles
        role_targets = {
            'win_conditions': 12,
            'removal': 8,
            'utility': 8,
            'ramp': 4,
            'lands': 24
        }
        
        # Build each section
        for role, target_count in role_targets.items():
            if role == 'lands':
                lands = self._generate_mana_base(colors, target_count)
                deck['mainboard'].extend(lands)
            else:
                role_cards = self._find_role_cards(role, colors, target_count)
                deck['mainboard'].extend(role_cards)
        
        # Generate sideboard
        sideboard = self._generate_sideboard(colors, concept)
        deck['sideboard'] = sideboard
        
        return deck
    
    def _find_role_cards(self, role: str, colors: List[str], target_count: int) -> List[Dict]:
        """Find cards for a specific role"""
        # Map roles to embedding queries
        role_mapping = {
            'win_conditions': 'finisher',
            'removal': 'removal',
            'utility': 'card_draw',
            'ramp': 'ramp'
        }
        
        embedding_role = role_mapping.get(role, role)
        candidate_cards = self.card_embeddings.find_cards_by_role(embedding_role, colors, k=10)
        
        # Select diverse cards for the role
        selected_cards = []
        remaining_count = target_count
        
        for card in candidate_cards:
            if remaining_count <= 0:
                break
            
            # Determine quantity (1-4 copies)
            if 'Legendary' in card.get('type_line', ''):
                quantity = min(2, remaining_count)  # Max 2 legendaries
            else:
                quantity = min(4, remaining_count)  # Up to 4 copies
            
            selected_cards.append({
                'name': card['name'],
                'quantity': quantity
            })
            
            remaining_count -= quantity
        
        return selected_cards
    
    def _generate_mana_base(self, colors: List[str], land_count: int) -> List[Dict]:
        """Generate appropriate mana base"""
        lands = []
        
        if not colors:
            # Colorless/generic lands
            lands.append({'name': 'Wastes', 'quantity': land_count})
            return lands
        
        if len(colors) == 1:
            # Mono-color mana base
            basic_land_names = {
                'W': 'Plains', 'U': 'Island', 'B': 'Swamp', 
                'R': 'Mountain', 'G': 'Forest'
            }
            basic_name = basic_land_names.get(colors[0], 'Plains')
            
            lands.append({'name': basic_name, 'quantity': land_count - 4})
            lands.append({'name': 'Evolving Wilds', 'quantity': 4})  # Fetch lands
            
        elif len(colors) == 2:
            # Two-color mana base
            basic_land_names = {
                'W': 'Plains', 'U': 'Island', 'B': 'Swamp', 
                'R': 'Mountain', 'G': 'Forest'
            }
            
            # Dual lands
            lands.append({'name': 'Dual Land (Generic)', 'quantity': 8})
            
            # Basic lands
            for color in colors:
                basic_name = basic_land_names.get(color, 'Plains')
                lands.append({'name': basic_name, 'quantity': 6})
            
            # Utility
            lands.append({'name': 'Evolving Wilds', 'quantity': 4})
            
        else:
            # Three+ color mana base
            lands.append({'name': 'Triome/Tri-land', 'quantity': 4})
            lands.append({'name': 'Dual Land (Generic)', 'quantity': 8})
            lands.append({'name': 'Evolving Wilds', 'quantity': 4})
            
            # Remaining basics split evenly
            remaining = land_count - 16
            basics_per_color = remaining // len(colors)
            
            basic_land_names = {
                'W': 'Plains', 'U': 'Island', 'B': 'Swamp', 
                'R': 'Mountain', 'G': 'Forest'
            }
            
            for color in colors:
                if basics_per_color > 0:
                    basic_name = basic_land_names.get(color, 'Plains')
                    lands.append({'name': basic_name, 'quantity': basics_per_color})
        
        return lands
    
    def _generate_sideboard(self, colors: List[str], concept: Dict) -> List[Dict]:
        """Generate appropriate sideboard"""
        sideboard = []
        
        # Common sideboard roles
        sb_roles = ['removal', 'counterspell', 'utility', 'sweeper']
        
        for role in sb_roles:
            role_cards = self.card_embeddings.find_cards_by_role(role, colors, k=2)
            for card in role_cards[:1]:  # 1 card per role
                sideboard.append({'name': card['name'], 'quantity': 2})
        
        # Fill remaining slots
        remaining_slots = 15 - sum(card['quantity'] for card in sideboard)
        if remaining_slots > 0:
            # Add more removal/utility
            extra_cards = self.card_embeddings.find_cards_by_role('utility', colors, k=3)
            for card in extra_cards:
                if remaining_slots <= 0:
                    break
                quantity = min(2, remaining_slots)
                sideboard.append({'name': card['name'], 'quantity': quantity})
                remaining_slots -= quantity
        
        return sideboard
    
    def _validate_and_adjust_deck(self, deck: Dict) -> Dict:
        """Validate deck construction rules and adjust if needed"""
        mainboard = deck.get('mainboard', [])
        
        # Check total cards
        total_cards = sum(card['quantity'] for card in mainboard)
        
        if total_cards < 60:
            # Add more cards
            shortage = 60 - total_cards
            # Add to existing cards or find new ones
            if mainboard:
                # Distribute shortage among existing cards (up to 4-of limit)
                for card in mainboard:
                    if shortage <= 0:
                        break
                    current_qty = card['quantity']
                    if current_qty < 4:
                        add_qty = min(4 - current_qty, shortage)
                        card['quantity'] += add_qty
                        shortage -= add_qty
        
        elif total_cards > 60:
            # Remove excess cards
            excess = total_cards - 60
            # Remove from cards with highest quantities first
            mainboard.sort(key=lambda x: x['quantity'], reverse=True)
            
            for card in mainboard:
                if excess <= 0:
                    break
                reduce_by = min(card['quantity'] - 1, excess)  # Keep at least 1
                card['quantity'] -= reduce_by
                excess -= reduce_by
            
            # Remove cards with 0 quantity
            mainboard = [card for card in mainboard if card['quantity'] > 0]
            deck['mainboard'] = mainboard
        
        return deck
    
    def _query_llm(self, prompt: str) -> str:
        """Query available LLM"""
        try:
            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-opus-4-1-20250805",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif self.openai_client:
                from openai import OpenAI
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000
                )
                return response.choices[0].message.content
            
            else:
                logger.error("No LLM client available for deck generation")
                raise Exception("No LLM client configured. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
                
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise Exception(f"Deck generation failed: {e}")
    
    def _extract_card_names_from_prompt(self, prompt: str) -> List[str]:
        """Extract specific card names mentioned in the prompt"""
        potential_cards = []
        
        # Look for card names that might be in Standard
        # Split into potential card name chunks
        import re
        
        # Look for sequences of capitalized words (likely card names)
        potential_names = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*', prompt)
        
        # Also extract quoted names or specific patterns
        quoted_names = re.findall(r'"([^"]+)"', prompt)
        potential_names.extend(quoted_names)
        
        # Check against card database if available
        if hasattr(self.card_embeddings, 'card_list') and self.card_embeddings.card_list:
            for name in potential_names:
                # Try exact match first
                for card in self.card_embeddings.card_list:
                    card_name = card.get('name', '')
                    if name.lower() == card_name.lower():
                        if card_name not in potential_cards:
                            potential_cards.append(card_name)
                        break
                else:
                    # Try partial match if no exact match
                    for card in self.card_embeddings.card_list:
                        card_name = card.get('name', '')
                        if name.lower() in card_name.lower() and len(name) >= 4:  # Avoid short matches
                            if card_name not in potential_cards:
                                potential_cards.append(card_name)
                            break
        
        return potential_cards
    
    def _build_deck_with_requested_cards(self, deck_concept: Dict, colors: List[str], must_include: List[str], meta_context: Dict) -> Dict:
        """Build deck starting with requested cards"""
        deck = {'mainboard': [], 'sideboard': []}
        
        # Start with requested cards
        total_nonland = 0
        all_colors = set()
        
        for card_name in must_include:
            if hasattr(self.card_embeddings, 'card_list') and self.card_embeddings.card_list:
                # Find exact card match
                found_card = None
                for card in self.card_embeddings.card_list:
                    if card.get('name', '').lower() == card_name.lower():
                        found_card = card
                        break
                
                if found_card:
                    # Add the card with reasonable quantity
                    if 'Legendary' in found_card.get('type_line', ''):
                        quantity = 2  # 2 copies of legendary
                    elif found_card.get('cmc', 0) >= 6:
                        quantity = 1  # 1 copy of expensive cards
                    elif found_card.get('cmc', 0) >= 5:
                        quantity = 2  # 2 copies of costly cards
                    elif found_card.get('cmc', 0) >= 3:
                        quantity = 3  # 3 copies of mid-cost
                    else:
                        quantity = 4  # 4 copies of cheap cards
                    
                    deck['mainboard'].append({
                        'name': found_card['name'],
                        'quantity': quantity
                    })
                    total_nonland += quantity
                    
                    # Collect colors from included cards
                    card_colors = found_card.get('colors', [])
                    all_colors.update(card_colors)
        
        # Set colors based on requested cards if not specified
        if not colors and all_colors:
            colors = list(all_colors)
        
        # Fill out the rest with synergistic cards
        target_nonland = 36  # 60 - 24 lands
        remaining = target_nonland - total_nonland
        
        if remaining > 0 and hasattr(self.card_embeddings, 'card_list'):
            # Add synergistic cards based on strategy
            strategy_cards = self._find_strategy_cards(colors, remaining)
            deck['mainboard'].extend(strategy_cards)
        
        # Add lands
        lands = self._generate_better_mana_base(colors, 24)
        deck['mainboard'].extend(lands)
        
        # Simple sideboard
        deck['sideboard'] = self._generate_simple_sideboard(colors)
        
        return deck
    
    def _find_strategy_cards(self, colors: List[str], count_needed: int) -> List[Dict]:
        """Find strategic cards for the deck"""
        cards = []
        
        # Simple role-based card finding
        roles = ['finisher', 'removal', 'utility', 'aggressive_creature']
        cards_per_role = max(1, count_needed // len(roles))
        
        for role in roles:
            if len(cards) >= count_needed:
                break
                
            role_cards = self.card_embeddings.find_cards_by_role(role, colors, k=cards_per_role + 1)
            
            for card in role_cards:
                if len(cards) >= count_needed:
                    break
                    
                # Reasonable quantities
                if card.get('cmc', 0) <= 2:
                    quantity = 4
                elif card.get('cmc', 0) <= 4:
                    quantity = 3
                else:
                    quantity = 2
                
                cards.append({
                    'name': card['name'],
                    'quantity': quantity
                })
        
        return cards
    
    def _generate_better_mana_base(self, colors: List[str], land_count: int) -> List[Dict]:
        """Generate proper mana base"""
        lands = []
        
        if not colors or colors == ['C']:
            lands.append({'name': 'Wastes', 'quantity': land_count})
            return lands
        
        basic_names = {
            'W': 'Plains', 'U': 'Island', 'B': 'Swamp', 
            'R': 'Mountain', 'G': 'Forest'
        }
        
        if len(colors) == 1:
            # Mono color
            basic = basic_names.get(colors[0], 'Plains')
            lands.append({'name': basic, 'quantity': land_count})
            
        elif len(colors) == 2:
            # Two color
            for color in colors:
                basic = basic_names.get(color, 'Plains') 
                lands.append({'name': basic, 'quantity': land_count // 2})
                
        else:
            # Multi-color
            per_color = land_count // len(colors)
            for color in colors:
                basic = basic_names.get(color, 'Plains')
                lands.append({'name': basic, 'quantity': per_color})
        
        return lands
    
    def _generate_simple_sideboard(self, colors: List[str]) -> List[Dict]:
        """Generate basic sideboard"""
        sideboard = []
        
        # Add some basic sideboard cards
        if hasattr(self.card_embeddings, 'card_list') and self.card_embeddings.card_list:
            for role in ['removal', 'utility']:
                cards = self.card_embeddings.find_cards_by_role(role, colors, k=3)
                for card in cards[:2]:
                    sideboard.append({
                        'name': card['name'],
                        'quantity': 2
                    })
                    if len(sideboard) >= 7:
                        break
                if len(sideboard) >= 7:
                    break
        
        # Fill to 15 cards
        while sum(card['quantity'] for card in sideboard) < 15:
            sideboard.append({'name': 'Flexible Sideboard Card', 'quantity': 1})
        
        return sideboard

if __name__ == "__main__":
    # This would be used with actual card data
    embeddings = CardEmbeddings()
    generator = LLMDeckGenerator(embeddings)
    
    # Example usage
    deck = generator.generate_deck(
        prompt="Create an aggressive red deck that can beat control",
        colors=['R'],
        archetype="Aggro"
    )
    
    print(f"Generated deck with {deck['total_cards']} cards")
    for card in deck['mainboard'][:5]:
        print(f"  {card['quantity']}x {card['name']}")