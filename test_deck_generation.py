#!/usr/bin/env python3
"""
Test the improved deck generation system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.card_database import CardDatabase
from data.metagame_analyzer import MetagameAnalyzer
from generation.deck_generator import CardEmbeddings, LLMDeckGenerator

def test_deck_generation():
    print("Testing Improved Deck Generation...")
    
    # Initialize card database
    card_db = CardDatabase("data/cards")
    if not card_db.load_latest_standard_cards():
        print("Downloading card database...")
        card_db.download_standard_cards()
    
    print(f"✅ Card database loaded: {len(card_db.standard_cards)} cards")
    
    # Initialize meta analyzer for context
    analyzer = MetagameAnalyzer("data/raw", card_db)
    meta_context = {}
    
    if analyzer.load_scraped_data():
        print(f"✅ Meta data loaded: {len(analyzer.deck_data)} decks")
        
        # Get meta context
        deck_to_beat = analyzer.get_deck_to_beat()
        top_cards = analyzer.get_top_cards(20, exclude_lands=True)
        
        meta_context = {
            'deck_to_beat': deck_to_beat,
            'top_cards': top_cards
        }
        
        if deck_to_beat:
            print(f"📊 Meta leader: {deck_to_beat['name']} ({deck_to_beat['stats']['percentage']}%)")
    
    # Initialize embeddings and generator
    print("🔄 Initializing card embeddings...")
    embeddings = CardEmbeddings()
    
    # Convert card database format to list of cards
    card_list = list(card_db.standard_cards.values())
    embeddings.generate_card_embeddings(card_list)
    
    generator = LLMDeckGenerator(embeddings)
    
    # Test with your original request
    print("\n🎯 Testing deck generation with requested cards...")
    print("Request: Generate a deck using Tezzeret, Cruel Captain and Repurposing Bay which will be competitive against the current meta.")
    
    deck = generator.generate_deck(
        prompt="Generate a deck using Tezzeret, Cruel Captain and Repurposing Bay which will be competitive against the current meta.",
        meta_context=meta_context
    )
    
    # Check if generation failed
    if 'error' in deck:
        print(f"\n❌ Deck generation failed: {deck['error']}")
        return
    
    if deck['total_cards'] == 0:
        print(f"\n❌ No deck generated: {deck['concept'].get('strategy', 'Unknown error')}")
        return
    
    print(f"\n🃏 Generated Deck ({deck['total_cards']} cards):")
    print(f"Colors: {', '.join(deck['colors']) if deck['colors'] else 'Colorless'}")
    print(f"Archetype: {deck['archetype']}")
    
    print(f"\n📝 Strategy: {deck['concept'].get('strategy', 'No strategy provided')}")
    
    print("\n🔥 Mainboard:")
    for i, card in enumerate(deck['mainboard']):
        print(f"   {card['quantity']}x {card['name']}")
        if i >= 20:  # Show first 20 cards
            remaining = len(deck['mainboard']) - 21
            if remaining > 0:
                print(f"   ... and {remaining} more cards")
            break
    
    print(f"\n🛡️  Sideboard ({len(deck['sideboard'])} unique cards):")
    for card in deck['sideboard'][:10]:  # Show first 10 sideboard cards
        print(f"   {card['quantity']}x {card['name']}")
    
    # Check if requested cards are included
    print(f"\n✅ Requested card inclusion check:")
    mainboard_cards = [card['name'] for card in deck['mainboard']]
    
    requested_cards = ["Tezzeret", "Cruel Captain", "Repurposing Bay"]
    for requested in requested_cards:
        found = any(requested.lower() in card.lower() for card in mainboard_cards)
        print(f"   {requested}: {'✅ Found' if found else '❌ Missing'}")
        
        if found:
            # Show which card was matched
            matched = [card for card in mainboard_cards if requested.lower() in card.lower()][0]
            deck_card = next(card for card in deck['mainboard'] if card['name'] == matched)
            print(f"      → {deck_card['quantity']}x {matched}")

if __name__ == "__main__":
    test_deck_generation()