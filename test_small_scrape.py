#!/usr/bin/env python3
"""
Test scraping a few decks to verify 75-card format
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.mtgtop8_scraper import MTGTop8Scraper
from data.card_database import CardDatabase
from data.metagame_analyzer import MetagameAnalyzer

def test_small_batch():
    print("🧪 Testing small batch scrape...")
    
    scraper = MTGTop8Scraper()
    
    # Scrape just 2 events to test
    output_file = scraper.scrape_standard_meta(num_events=2, output_dir="data/test")
    
    if not output_file:
        print("❌ Failed to scrape")
        return
    
    # Load and analyze the scraped data
    card_db = CardDatabase("data/cards")
    card_db.load_latest_standard_cards()
    
    analyzer = MetagameAnalyzer("data/test", card_db)
    analyzer.load_scraped_data()
    
    print(f"✅ Scraped {len(analyzer.deck_data)} decks")
    
    # Check deck formats
    valid_count = 0
    total_count = len(analyzer.deck_data)
    
    for i, deck in enumerate(analyzer.deck_data):
        mainboard = deck.get('mainboard', [])
        sideboard = deck.get('sideboard', [])
        
        mainboard_count = sum(card['quantity'] for card in mainboard)
        sideboard_count = sum(card['quantity'] for card in sideboard)
        
        is_valid = mainboard_count == 60 and sideboard_count == 15
        if is_valid:
            valid_count += 1
        
        if i < 5:  # Show first 5 examples
            status = "✅" if is_valid else "❌"
            print(f"{status} Deck {i+1}: {mainboard_count}+{sideboard_count} = {mainboard_count + sideboard_count}")
    
    print(f"\n📊 Results: {valid_count}/{total_count} decks are valid 75-card format ({valid_count/total_count*100:.1f}%)")
    
    if valid_count > 0:
        print("🎉 Scraper is working! Can now generate proper training data.")
    else:
        print("😞 Still having issues with deck format.")

if __name__ == "__main__":
    test_small_batch()