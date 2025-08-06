import gradio as gr
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
import sys
import asyncio
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.mtgtop8_scraper import MTGTop8Scraper
from data.card_database import CardDatabase
from data.metagame_analyzer import MetagameAnalyzer
from generation.deck_generator import CardEmbeddings, LLMDeckGenerator
from evaluation.simulation_engine import DeckEvaluator
from explanation.deck_explainer import DeckExplainer
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MTGDeckbuildingChatbot:
    """Main chatbot orchestrating all MTG deckbuilding components"""
    
    def __init__(self):
        self.config = Config()
        self.setup_components()
        self.conversation_history = []
        
    def setup_components(self):
        """Initialize all components"""
        logger.info("Initializing MTG Deckbuilding System...")
        
        # Create data directories
        os.makedirs(self.config.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(self.config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.config.CARDS_DATA_DIR, exist_ok=True)
        
        # Initialize components
        self.scraper = MTGTop8Scraper()
        self.card_db = CardDatabase(self.config.CARDS_DATA_DIR)
        self.meta_analyzer = MetagameAnalyzer(self.config.RAW_DATA_DIR, self.card_db)
        self.evaluator = DeckEvaluator()
        self.explainer = DeckExplainer(self.card_db)
        
        # Initialize embeddings and generator (will be lazy-loaded)
        self.card_embeddings = None
        self.deck_generator = None
        
        # Load existing data if available
        self._load_existing_data()
        
        logger.info("System initialized successfully!")
    
    def _load_existing_data(self):
        """Load existing card and meta data"""
        try:
            # Try to load card database
            if self.card_db.load_latest_standard_cards():
                logger.info("Loaded existing card database")
            else:
                logger.info("No card database found - downloading Standard cards...")
                try:
                    self.card_db.download_standard_cards()
                    logger.info("Successfully downloaded card database")
                except Exception as e:
                    logger.error(f"Failed to download cards: {e}")
            
            # Try to load meta analysis
            if self.meta_analyzer.load_scraped_data():
                logger.info("Loaded existing meta data")
            else:
                logger.info("No meta data found - will scrape when requested")
                
        except Exception as e:
            logger.warning(f"Error loading existing data: {e}")
    
    def _ensure_card_embeddings(self):
        """Lazy load card embeddings"""
        if self.card_embeddings is None:
            logger.info("Building card embeddings...")
            self.card_embeddings = CardEmbeddings()
            
            # Get standard cards
            if not self.card_db.standard_cards:
                self.card_db.download_standard_cards()
            
            standard_cards = list(self.card_db.standard_cards.values())
            self.card_embeddings.generate_card_embeddings(standard_cards)
            
            self.deck_generator = LLMDeckGenerator(self.card_embeddings)
            logger.info("Card embeddings ready!")
    
    def chat_response(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Main chat response function"""
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": message})
            
            # Determine intent and route to appropriate handler
            response = self._route_message(message)
            
            # Add to history
            history.append((message, response))
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return "", history
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            history.append((message, error_msg))
            return "", history
    
    def _route_message(self, message: str) -> str:
        """Route message to appropriate handler based on intent"""
        message_lower = message.lower()
        
        # Generate deck requests
        if any(phrase in message_lower for phrase in ['generate', 'create', 'build', 'suggest']):
            if any(phrase in message_lower for phrase in ['deck', 'decklist']):
                return self._handle_deck_generation(message)
        
        # Meta analysis requests
        if any(phrase in message_lower for phrase in ['meta', 'metagame', 'what\'s good', 'best deck']):
            return self._handle_meta_analysis(message)
        
        # Card search requests
        if any(phrase in message_lower for phrase in ['card', 'find', 'search']):
            return self._handle_card_search(message)
        
        # Deck evaluation requests
        if any(phrase in message_lower for phrase in ['evaluate', 'test', 'how good', 'analyze']):
            if 'deck' in message_lower:
                return self._handle_deck_evaluation(message)
        
        # Explanation requests
        if any(phrase in message_lower for phrase in ['explain', 'why', 'how', 'strategy']):
            return self._handle_explanation(message)
        
        # Data update requests
        if any(phrase in message_lower for phrase in ['update', 'refresh', 'scrape', 'download']):
            return self._handle_data_update(message)
        
        # Default: general help
        return self._handle_general_query(message)
    
    def _handle_deck_generation(self, message: str) -> str:
        """Handle deck generation requests"""
        try:
            self._ensure_card_embeddings()
            
            # Extract parameters from message
            colors = self._extract_colors(message)
            archetype = self._extract_archetype(message)
            
            # Get current meta context
            meta_context = self._get_meta_context()
            
            # Generate deck
            deck = self.deck_generator.generate_deck(
                prompt=message,
                colors=colors,
                archetype=archetype,
                meta_context=meta_context
            )
            
            # Format response
            response = self._format_deck_response(deck, message)
            
            # Add explanation if requested
            if any(word in message.lower() for word in ['explain', 'why', 'how']):
                explanation = self.explainer.explain_deck(deck, meta_context)
                response += f"\n\n**Deck Strategy:**\n{explanation['overview']}"
            
            return response
            
        except Exception as e:
            logger.error(f"Deck generation error: {e}")
            return f"I had trouble generating a deck. Error: {str(e)}\n\nTry being more specific about colors, archetype, or strategy you want."
    
    def _handle_meta_analysis(self, message: str) -> str:
        """Handle metagame analysis requests"""
        try:
            # Check if we have recent data
            if not self.meta_analyzer.deck_data:
                return "I need to analyze the current metagame first. Let me update the meta data...\n\n" + self._update_meta_data()
            
            # Get meta analysis
            archetype_breakdown = self.meta_analyzer.get_archetype_breakdown()
            deck_to_beat = self.meta_analyzer.get_deck_to_beat()
            top_cards = self.meta_analyzer.get_top_cards(10, exclude_lands=True)  # Filter out basic lands
            
            total_decks = len(self.meta_analyzer.deck_data)
            response = f"**Current Standard Metagame** (based on {total_decks} recent decks):\n\n"
            
            if deck_to_beat:
                response += f"**🏆 Deck to Beat:** {deck_to_beat['name']} ({deck_to_beat['stats']['percentage']:.1f}% of meta)\n\n"
            
            response += "**📊 Top Archetypes:**\n"
            sorted_archetypes = sorted(archetype_breakdown.items(), key=lambda x: x[1]['percentage'], reverse=True)
            
            if not sorted_archetypes:
                response += "- No clear archetype patterns detected\n"
            else:
                for i, (archetype, stats) in enumerate(sorted_archetypes[:5]):
                    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📈"
                    response += f"{emoji} **{archetype}**: {stats['percentage']:.1f}% ({stats['deck_count']} decks)\n"
                    
                    # Show key cards for top archetypes
                    if i < 2 and stats.get('key_cards'):
                        key_cards = [f"{card}" for card, count in stats['key_cards'][:3]]
                        response += f"   *Key cards: {', '.join(key_cards)}*\n"
                response += "\n"
            
            response += "**🔥 Most Played Non-Land Cards:**\n"
            if not top_cards:
                response += "- No clear card patterns detected\n"
            else:
                for card, count in top_cards:
                    # Calculate what percentage of decks play this card (not copies per deck)
                    decks_with_card = 0
                    for deck in self.meta_analyzer.deck_data:
                        for card_entry in deck.get('mainboard', []):
                            if card_entry['name'] == card:
                                decks_with_card += 1
                                break
                    
                    percentage = (decks_with_card / total_decks) * 100 if total_decks > 0 else 0
                    response += f"- **{card}**: {count} copies ({percentage:.0f}% of decks)\n"
            
            # Add meta insights
            response += f"\n**💡 Meta Insights:**\n"
            if len(sorted_archetypes) <= 2:
                response += "- Meta appears to be dominated by 1-2 strategies\n"
            elif len(sorted_archetypes) >= 5:
                response += "- Diverse meta with many viable strategies\n"
            else:
                response += "- Balanced meta with several competing archetypes\n"
            
            # Color analysis
            color_analysis = self._analyze_color_distribution()
            if color_analysis:
                response += f"- {color_analysis}\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Meta analysis error: {e}")
            return f"I had trouble analyzing the metagame. Error: {str(e)}"
    
    def _handle_card_search(self, message: str) -> str:
        """Handle card search requests"""
        try:
            # Extract search terms
            search_terms = self._extract_search_terms(message)
            
            if not search_terms:
                return "What cards would you like me to search for? Try something like 'find lightning cards' or 'search for counterspells'."
            
            # Search cards
            results = self.card_db.search_cards(' '.join(search_terms), limit=10)
            
            if not results:
                return f"I couldn't find any Standard-legal cards matching '{' '.join(search_terms)}'. Try different search terms."
            
            response = f"**Found {len(results)} Standard-legal cards:**\n\n"
            
            for card in results:
                response += f"**{card['name']}** - {card['mana_cost']} - {card['type_line']}\n"
                if card['oracle_text']:
                    response += f"  {card['oracle_text'][:100]}{'...' if len(card['oracle_text']) > 100 else ''}\n"
                response += "\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Card search error: {e}")
            return f"I had trouble searching for cards. Error: {str(e)}"
    
    def _handle_deck_evaluation(self, message: str) -> str:
        """Handle deck evaluation requests"""
        return "Deck evaluation requires you to provide a specific decklist. Please share your decklist and I can analyze its performance against the current meta."
    
    def _handle_explanation(self, message: str) -> str:
        """Handle explanation requests"""
        return "I can explain deck strategies, card choices, and matchups. Please share a specific deck or ask about a particular aspect of MTG strategy you'd like me to explain."
    
    def _handle_data_update(self, message: str) -> str:
        """Handle data update requests"""
        try:
            response = "Updating MTG data...\n\n"
            
            if 'card' in message.lower():
                response += self._update_card_data()
            elif 'meta' in message.lower():
                response += self._update_meta_data()
            else:
                # Update both
                response += self._update_card_data()
                response += "\n" + self._update_meta_data()
            
            return response
            
        except Exception as e:
            return f"Error updating data: {str(e)}"
    
    def _update_card_data(self) -> str:
        """Update card database"""
        try:
            filepath = self.card_db.download_standard_cards()
            return f"✅ Updated Standard card database ({len(self.card_db.standard_cards)} cards)"
        except Exception as e:
            return f"❌ Failed to update cards: {str(e)}"
    
    def _update_meta_data(self) -> str:
        """Update metagame data"""
        try:
            # Ensure we have card data for proper analysis
            if not self.card_db.standard_cards:
                logger.info("Downloading card data for meta analysis...")
                self.card_db.download_standard_cards()
            
            filepath = self.scraper.scrape_standard_meta(num_events=5)
            self.meta_analyzer.load_scraped_data()
            return f"✅ Updated metagame data ({len(self.meta_analyzer.deck_data)} decks analyzed)"
        except Exception as e:
            return f"❌ Failed to update meta: {str(e)}"
    
    def _handle_general_query(self, message: str) -> str:
        """Handle general queries and provide help"""
        return """I'm your AI MTG Deckbuilding assistant! I can help you with:

🔧 **Deck Generation**: "Generate a blue-white control deck" or "Create an aggressive red deck"

📊 **Meta Analysis**: "What's the current meta?" or "Show me the best decks"

🔍 **Card Search**: "Find lightning spells" or "Search for counterspells"

⚡ **Deck Evaluation**: Share a decklist and I'll analyze its performance

💡 **Strategy Explanation**: Ask about card choices, matchups, or deck strategies

📈 **Data Updates**: "Update meta data" or "Refresh card database"

What would you like to do?"""
    
    def _extract_colors(self, message: str) -> List[str]:
        """Extract MTG colors from message"""
        colors = []
        color_map = {
            'white': 'W', 'blue': 'U', 'black': 'B', 'red': 'R', 'green': 'G',
            'azorius': ['W', 'U'], 'dimir': ['U', 'B'], 'rakdos': ['B', 'R'],
            'gruul': ['R', 'G'], 'selesnya': ['G', 'W'], 'orzhov': ['W', 'B'],
            'izzet': ['U', 'R'], 'golgari': ['B', 'G'], 'boros': ['R', 'W'],
            'simic': ['G', 'U']
        }
        
        message_lower = message.lower()
        for color_name, color_code in color_map.items():
            if color_name in message_lower:
                if isinstance(color_code, list):
                    colors.extend(color_code)
                else:
                    colors.append(color_code)
        
        return list(set(colors))  # Remove duplicates
    
    def _analyze_color_distribution(self) -> str:
        """Analyze color distribution in current meta"""
        try:
            if not self.meta_analyzer.deck_data:
                return ""
            
            # Count color usage across all decks
            color_counts = {'Red': 0, 'Blue': 0, 'Black': 0, 'White': 0, 'Green': 0}
            
            for deck in self.meta_analyzer.deck_data:
                mainboard = deck.get('mainboard', [])
                deck_colors = set()
                
                for card in mainboard:
                    card_name = card['name'].lower()
                    # Simple color detection based on card names and common patterns
                    if any(word in card_name for word in ['mountain', 'red', 'bolt', 'shock', 'fire']):
                        deck_colors.add('Red')
                    if any(word in card_name for word in ['island', 'blue', 'counter', 'draw']):
                        deck_colors.add('Blue')
                    if any(word in card_name for word in ['swamp', 'black', 'death', 'murder']):
                        deck_colors.add('Black')
                    if any(word in card_name for word in ['plains', 'white', 'angel', 'heal']):
                        deck_colors.add('White')
                    if any(word in card_name for word in ['forest', 'green', 'ramp', 'growth']):
                        deck_colors.add('Green')
                
                # Count this deck's colors
                for color in deck_colors:
                    color_counts[color] += 1
            
            # Find most popular colors
            total_decks = len(self.meta_analyzer.deck_data)
            popular_colors = []
            
            for color, count in color_counts.items():
                if count > total_decks * 0.3:  # Appears in >30% of decks
                    percentage = (count / total_decks) * 100
                    popular_colors.append(f"{color} ({percentage:.0f}%)")
            
            if popular_colors:
                return f"Popular colors: {', '.join(popular_colors)}"
            else:
                return "Color distribution is fairly even"
                
        except Exception as e:
            logger.error(f"Error analyzing colors: {e}")
            return ""
    
    def _extract_archetype(self, message: str) -> Optional[str]:
        """Extract archetype from message"""
        message_lower = message.lower()
        
        archetypes = {
            'aggro': ['aggro', 'aggressive', 'fast', 'rush'],
            'control': ['control', 'counterspell', 'late game'],
            'midrange': ['midrange', 'midrange', 'balanced'],
            'combo': ['combo', 'synergy', 'engine'],
            'tempo': ['tempo', 'efficient']
        }
        
        for archetype, keywords in archetypes.items():
            if any(keyword in message_lower for keyword in keywords):
                return archetype.title()
        
        return None
    
    def _extract_search_terms(self, message: str) -> List[str]:
        """Extract search terms from message"""
        # Simple extraction - remove common words
        stop_words = {'find', 'search', 'for', 'cards', 'card', 'show', 'me', 'get', 'with', 'that', 'are'}
        words = message.lower().split()
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _get_meta_context(self) -> Dict:
        """Get current meta context for generation"""
        try:
            if not self.meta_analyzer.deck_data:
                return {}
            
            return {
                'deck_to_beat': self.meta_analyzer.get_deck_to_beat(),
                'top_cards': self.meta_analyzer.get_top_cards(15),
                'archetype_breakdown': self.meta_analyzer.get_archetype_breakdown()
            }
        except:
            return {}
    
    def _format_deck_response(self, deck: Dict, original_request: str) -> str:
        """Format deck generation response"""
        response = f"**Generated {deck.get('archetype', 'Custom')} Deck ({deck.get('total_cards', 0)} cards)**\n\n"
        
        if deck.get('colors'):
            response += f"**Colors:** {', '.join(deck['colors'])}\n\n"
        
        response += "**Mainboard:**\n"
        
        # Group cards by type for better presentation
        mainboard = deck.get('mainboard', [])
        creatures = []
        spells = []
        lands = []
        others = []
        
        for card in mainboard:
            name_lower = card['name'].lower()
            if any(word in name_lower for word in ['plains', 'island', 'swamp', 'mountain', 'forest', 'land']):
                lands.append(card)
            elif any(word in name_lower for word in ['dragon', 'angel', 'demon', 'knight', 'soldier', 'creature']):
                creatures.append(card)
            elif any(word in name_lower for word in ['bolt', 'shock', 'murder', 'counterspell']):
                spells.append(card)
            else:
                others.append(card)
        
        # Format each section
        if creatures:
            response += "\n*Creatures:*\n"
            for card in creatures:
                response += f"  {card['quantity']}x {card['name']}\n"
        
        if spells:
            response += "\n*Spells:*\n"
            for card in spells:
                response += f"  {card['quantity']}x {card['name']}\n"
        
        if others:
            response += "\n*Other:*\n"
            for card in others:
                response += f"  {card['quantity']}x {card['name']}\n"
        
        if lands:
            response += "\n*Lands:*\n"
            for card in lands:
                response += f"  {card['quantity']}x {card['name']}\n"
        
        # Add sideboard if present
        sideboard = deck.get('sideboard', [])
        if sideboard:
            response += "\n**Sideboard:**\n"
            for card in sideboard:
                response += f"  {card['quantity']}x {card['name']}\n"
        
        return response

def create_interface():
    """Create Gradio interface"""
    chatbot_instance = MTGDeckbuildingChatbot()
    
    with gr.Blocks(title="MTG Deckbuilding AI", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🧙‍♂️ MTG Deckbuilding AI Assistant")
        gr.Markdown("Your intelligent companion for Magic: The Gathering Standard deckbuilding, meta analysis, and strategy guidance.")
        
        chatbot = gr.Chatbot(
            height=500,
            show_label=False,
            avatar_images=(None, "🧙‍♂️")
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me to generate a deck, analyze the meta, or search for cards...",
                show_label=False,
                scale=4
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Example prompts
        gr.Markdown("### 💡 Try these examples:")
        example_buttons = gr.Row()
        with example_buttons:
            gr.Button("Generate a red aggro deck", size="sm").click(
                lambda: "Generate a red aggro deck", None, msg
            )
            gr.Button("What's the current meta?", size="sm").click(
                lambda: "What's the current meta?", None, msg
            )
            gr.Button("Find lightning spells", size="sm").click(
                lambda: "Find lightning spells", None, msg
            )
            gr.Button("Update meta data", size="sm").click(
                lambda: "Update meta data", None, msg
            )
        
        # Event handlers
        def handle_message(message, history):
            return chatbot_instance.chat_response(message, history)
        
        msg.submit(handle_message, [msg, chatbot], [msg, chatbot])
        send_btn.click(handle_message, [msg, chatbot], [msg, chatbot])
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("*Powered by AI and real MTGTop8 data. For competitive Standard deckbuilding.*")
    
    return interface

def main():
    """Main entry point"""
    print("Starting MTG Deckbuilding AI...")
    
    interface = create_interface()
    interface.launch(
        server_name=Config.HOST,
        server_port=Config.PORT,
        share=False,
        debug=Config.DEBUG
    )

if __name__ == "__main__":
    main()