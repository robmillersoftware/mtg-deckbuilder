#!/usr/bin/env python3
"""
Test improved meta analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.card_database import CardDatabase
from data.metagame_analyzer import MetagameAnalyzer

def test_meta_analysis():
    print("Testing Improved Meta Analysis...")
    
    # Initialize with card database
    card_db = CardDatabase("data/cards")
    if not card_db.load_latest_standard_cards():
        card_db.download_standard_cards()
    
    analyzer = MetagameAnalyzer("data/raw", card_db)
    
    # Load existing meta data
    if analyzer.load_scraped_data():
        print(f"✅ Loaded {len(analyzer.deck_data)} decks")
        
        # Test archetype breakdown
        breakdown = analyzer.get_archetype_breakdown()
        print(f"\n📊 Archetypes found: {len(breakdown)}")
        
        for archetype, stats in list(breakdown.items())[:5]:
            print(f"   {archetype}: {stats['percentage']:.1f}% ({stats['deck_count']} decks)")
        
        # Test filtered card analysis
        top_cards = analyzer.get_top_cards(10, exclude_lands=True)
        print(f"\n🔥 Top non-land cards:")
        
        for card, count in top_cards:
            print(f"   {card}: {count} copies")
        
        # Test land detection
        print(f"\n🏞️  Testing land detection:")
        test_cards = ["Island", "Mountain", "Spyglass Siren", "Riverpyre Verge"]
        for card_name in test_cards:
            is_land = analyzer._is_land_card(card_name)
            print(f"   {card_name}: {'Land' if is_land else 'Non-land'}")
        
    else:
        print("❌ No meta data found - need to scrape first")

if __name__ == "__main__":
    test_meta_analysis()