#!/usr/bin/env python3
"""
Enhanced deck generator with local fine-tuned model integration
"""

import json
import logging
import numpy as np
import os
import torch
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import openai
import anthropic
from collections import Counter
import random

# Local model imports (only loaded if model exists)
try:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT not available. Install with: pip install peft transformers torch")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CardEmbeddings:
    """Card embedding system for similarity search - unchanged from original"""
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
            description = self._create_card_description(card)
            card_descriptions.append(description)
            self.card_list.append(card)
        
        # Generate embeddings
        embeddings = self.model.encode(card_descriptions)
        
        # Build FAISS index for fast similarity search
        dimension = embeddings.shape[1]
        self.card_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.card_index.add(embeddings.astype('float32'))
        
        logger.info("Card embeddings generated and indexed")
    
    def _create_card_description(self, card: Dict) -> str:
        """Create a rich text description for embedding"""
        parts = []
        
        parts.append(card.get('name', ''))
        parts.append(f"mana cost {card.get('mana_cost', '')}")
        
        if 'types' in card:
            parts.append(' '.join(card['types']))
        
        if 'text' in card:
            parts.append(card['text'])
        
        if 'power' in card and 'toughness' in card:
            parts.append(f"{card['power']}/{card['toughness']}")
        
        if 'colors' in card:
            parts.append(f"colors {' '.join(card['colors'])}")
        
        if 'keywords' in card:
            parts.append(' '.join(card['keywords']))
        
        return ' '.join(filter(None, parts))
    
    def find_similar_cards(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """Find cards similar to query description"""
        if not self.card_index:
            return []
        
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.card_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.card_list):
                results.append((self.card_list[idx], float(score)))
        
        return results

class LocalFineTunedDeckGenerator:
    """Enhanced deck generator using local fine-tuned model"""
    
    def __init__(self, card_embeddings: CardEmbeddings, local_model_path: str = "./mtg-deck-model"):
        self.card_embeddings = card_embeddings
        self.local_model_path = local_model_path
        
        # Model state
        self.local_model = None
        self.local_tokenizer = None
        self.model_loaded = False
        
        # Fallback API clients
        self.openai_client = None
        self.anthropic_client = None
        
        # Try to load local model first
        self._try_load_local_model()
        
        # Initialize API clients as fallback
        self._initialize_api_clients()
    
    def _try_load_local_model(self):
        """Attempt to load the local fine-tuned model"""
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not available - cannot load local model")
            return
        
        if not os.path.exists(self.local_model_path):
            logger.info(f"Local model not found at {self.local_model_path}")
            return
        
        try:
            logger.info(f"🤖 Loading fine-tuned model from {self.local_model_path}")
            
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            self.local_model = AutoPeftModelForCausalLM.from_pretrained(
                self.local_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.model_loaded = True
            logger.info("✅ Local fine-tuned model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load local model: {e}")
            logger.info("Will fall back to API-based generation")
    
    def _initialize_api_clients(self):
        """Initialize API clients for fallback"""
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
        """Generate a deck using the best available method"""
        
        if self.model_loaded:
            logger.info("🎯 Generating deck with fine-tuned model")
            return self._generate_deck_with_local_model(prompt, colors, archetype, meta_context)
        else:
            logger.info("🔄 Falling back to API-based generation")
            return self._generate_deck_with_api(prompt, colors, archetype, meta_context)
    
    def _generate_deck_with_local_model(self, prompt: str, colors: List[str], archetype: str, meta_context: Dict) -> Dict:
        """Generate deck using local fine-tuned model"""
        try:
            # Build context similar to training data format
            context_parts = []
            
            if archetype:
                context_parts.append(f"Archetype: {archetype}")
            
            if colors:
                context_parts.append(f"Colors: {', '.join(colors)}")
            
            # Add meta context
            if meta_context:
                if 'deck_to_beat' in meta_context and meta_context['deck_to_beat']:
                    deck_to_beat = meta_context['deck_to_beat']
                    deck_name = deck_to_beat.get('name', 'Unknown')
                    deck_pct = deck_to_beat.get('stats', {}).get('percentage', 0)
                    context_parts.append(f"Current meta leader: {deck_name} ({deck_pct:.1f}%)")
                
                if 'top_cards' in meta_context:
                    popular_cards = [card for card, count in meta_context['top_cards'][:5]]
                    context_parts.append(f"Popular cards in meta: {', '.join(popular_cards)}")
            
            # Get relevant cards for context
            relevant_cards = self._get_relevant_cards_for_context(prompt, colors, k=10)
            if relevant_cards:
                card_names = [card['name'] for card in relevant_cards]
                context_parts.append(f"Key available cards: {', '.join(card_names)}")
            
            context_str = " | ".join(context_parts)
            
            # Build prompt in training format
            formatted_prompt = f"Build a competitive {archetype or 'deck'} for Standard format.\\n\\nContext: {context_str}\\n\\nDeck:"
            
            # Generate with local model
            response = self._generate_with_local_model(formatted_prompt)
            
            # Parse the response
            deck_data = self._parse_local_model_response(response, archetype, colors)
            
            return deck_data
            
        except Exception as e:
            logger.error(f"Error generating deck with local model: {e}")
            # Fall back to API generation
            return self._generate_deck_with_api(prompt, colors, archetype, meta_context)
    
    def _generate_with_local_model(self, prompt: str, max_length: int = 800) -> str:
        """Generate text using the local fine-tuned model"""
        if not self.model_loaded:
            raise ValueError("Local model not loaded")
        
        # Format prompt with chat template used during training
        formatted_prompt = f"<|user|>\\n{prompt}\\n<|assistant|>\\n"
        
        # Tokenize
        inputs = self.local_tokenizer.encode(formatted_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Generate
        with torch.no_grad():
            outputs = self.local_model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.local_tokenizer.eos_token_id,
                eos_token_id=self.local_tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Decode
        generated_text = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<|assistant|>" in generated_text:
            response = generated_text.split("<|assistant|>")[-1].strip()
            if "<|end|>" in response:
                response = response.split("<|end|>")[0].strip()
            return response
        
        return generated_text
    
    def _parse_local_model_response(self, response: str, archetype: str, colors: List[str]) -> Dict:
        """Parse response from local model into deck format"""
        try:
            # The model should output in the format:
            # **Mainboard:**
            # 4x Card Name
            # ...
            # **Sideboard:**
            # 2x Card Name
            # ...
            
            mainboard = []
            sideboard = []
            current_section = None
            
            lines = response.split('\\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if '**Mainboard:**' in line or 'Mainboard:' in line:
                    current_section = 'mainboard'
                    continue
                elif '**Sideboard:**' in line or 'Sideboard:' in line:
                    current_section = 'sideboard'
                    continue
                
                # Parse card lines (format: "4x Card Name")
                if current_section:
                    # Match patterns like "4x Card Name" or "4 Card Name"
                    import re
                    match = re.match(r'^(\\d+)x?\\s+(.+)$', line)
                    if match:
                        quantity = int(match.group(1))
                        card_name = match.group(2).strip()
                        
                        # Validate card exists in our database
                        if self._validate_card_exists(card_name):
                            card_entry = {
                                'name': card_name,
                                'quantity': quantity
                            }
                            
                            if current_section == 'mainboard':
                                mainboard.append(card_entry)
                            else:
                                sideboard.append(card_entry)
            
            # Calculate totals
            total_main = sum(card['quantity'] for card in mainboard)
            total_side = sum(card['quantity'] for card in sideboard)
            
            logger.info(f"🎯 Generated deck: {total_main} mainboard + {total_side} sideboard")
            
            return {
                'mainboard': mainboard,
                'sideboard': sideboard,
                'concept': {'strategy': f'Fine-tuned {archetype or "deck"} generated from tournament data'},
                'total_cards': total_main,
                'colors': colors or [],
                'archetype': archetype or 'Custom',
                'generator': 'local_finetuned'
            }
            
        except Exception as e:
            logger.error(f"Error parsing local model response: {e}")
            logger.error(f"Response was: {response[:500]}")
            
            # Return empty deck with error info
            return {
                'mainboard': [],
                'sideboard': [],
                'concept': {'strategy': f'Failed to parse local model response: {str(e)}'},
                'total_cards': 0,
                'colors': colors or [],
                'archetype': archetype or 'Error',
                'generator': 'local_finetuned_error',
                'error': str(e)
            }
    
    def _validate_card_exists(self, card_name: str) -> bool:
        """Check if card exists in our card database"""
        if not hasattr(self.card_embeddings, 'card_list'):
            return True  # Can't validate, assume it's valid
        
        # Simple name matching (could be enhanced with fuzzy matching)
        for card in self.card_embeddings.card_list:
            if card.get('name', '').lower() == card_name.lower():
                return True
        
        return False
    
    def _get_relevant_cards_for_context(self, prompt: str, colors: List[str], k: int = 10) -> List[Dict]:
        """Get relevant cards to provide context for generation"""
        if not hasattr(self.card_embeddings, 'card_list'):
            return []
        
        # Use embeddings to find relevant cards
        similar_cards = self.card_embeddings.find_similar_cards(prompt, k * 2)
        
        # Filter by colors if specified
        if colors:
            color_filtered = []
            for card, score in similar_cards:
                card_colors = set(card.get('colors', []))
                target_colors = set(colors)
                if not card_colors or card_colors.issubset(target_colors):
                    color_filtered.append(card)
            return color_filtered[:k]
        
        return [card for card, score in similar_cards[:k]]
    
    def _generate_deck_with_api(self, prompt: str, colors: List[str], archetype: str, meta_context: Dict) -> Dict:
        """Fallback: Generate deck using API (Claude/OpenAI)"""
        try:
            # Get sample of relevant cards for context
            card_pool = self._get_relevant_cards_for_context(prompt, colors, k=50)
            
            # Build comprehensive prompt
            llm_prompt = self._build_api_deck_prompt(prompt, colors, archetype, meta_context, card_pool)
            
            # Query LLM for deck
            response = self._query_api_llm(llm_prompt)
            
            # Parse response into deck format
            deck_data = self._parse_api_deck_response(response, archetype, colors)
            
            return deck_data
            
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            return {
                'mainboard': [],
                'sideboard': [],
                'concept': {'strategy': f'Deck generation failed: {str(e)}'},
                'total_cards': 0,
                'colors': colors or [],
                'archetype': archetype or 'Error',
                'generator': 'api_error',
                'error': str(e)
            }
    
    def _build_api_deck_prompt(self, prompt: str, colors: List[str], archetype: str, meta_context: Dict, card_pool: List[Dict]) -> str:
        """Build prompt for API-based generation"""
        prompt_parts = []
        
        prompt_parts.append("You are an expert Magic: The Gathering deck builder.")
        prompt_parts.append(f"Build a competitive Standard deck: {prompt}")
        
        if colors:
            prompt_parts.append(f"Deck colors: {', '.join(colors)}")
        
        if archetype:
            prompt_parts.append(f"Target archetype: {archetype}")
        
        # Add meta context
        if meta_context:
            if 'deck_to_beat' in meta_context and meta_context['deck_to_beat']:
                deck_to_beat = meta_context['deck_to_beat']
                prompt_parts.append(f"Current meta leader: {deck_to_beat.get('name', 'Unknown')}")
            
            if 'top_cards' in meta_context:
                top_cards = [card for card, count in meta_context['top_cards'][:10]]
                prompt_parts.append(f"Popular cards in meta: {', '.join(top_cards)}")
        
        # Add available cards
        if card_pool:
            card_names = [card['name'] for card in card_pool[:30]]
            prompt_parts.append(f"Available cards include: {', '.join(card_names)}")
        
        prompt_parts.append("""
REQUIREMENTS:
1. Build a 60-card mainboard + 15-card sideboard
2. Follow 4-of rule (max 4 copies, except basic lands)
3. Use only Standard-legal cards from the available pool
4. Ensure mana curve and color requirements work
5. Make it competitive against the current meta

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
}""")
        
        return "\\n".join(prompt_parts)
    
    def _query_api_llm(self, prompt: str) -> str:
        """Query API LLM (Claude or OpenAI)"""
        # Try Anthropic first
        if self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
        
        # Try OpenAI as fallback
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
        
        raise Exception("No working API clients available")
    
    def _parse_api_deck_response(self, response: str, archetype: str, colors: List[str]) -> Dict:
        """Parse API response into deck format"""
        try:
            # Try to extract JSON from response
            import re
            
            json_match = re.search(r'\\{.*\\}', response, re.DOTALL)
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
                    'concept': {'strategy': deck_data.get('strategy', 'API-generated deck')},
                    'total_cards': total_main,
                    'colors': deck_data.get('colors', colors or []),
                    'archetype': deck_data.get('archetype', archetype or 'Custom'),
                    'generator': 'api'
                }
        
        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
            logger.error(f"Response was: {response[:500]}")
        
        # Fallback to empty deck
        return {
            'mainboard': [],
            'sideboard': [],
            'concept': {'strategy': 'Failed to generate deck'},
            'total_cards': 0,
            'colors': colors or [],
            'archetype': archetype or 'Unknown',
            'generator': 'api_error'
        }

# Backwards compatibility - use the new generator by default
LLMDeckGenerator = LocalFineTunedDeckGenerator