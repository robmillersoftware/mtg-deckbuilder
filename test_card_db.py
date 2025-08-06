#!/usr/bin/env python3
"""
Test card database functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.card_database import CardDatabase

def test_card_db():
    print("Testing Card Database...")
    
    db = CardDatabase("data/cards")
    
    # Try to load existing cards
    if db.load_latest_standard_cards():
        print(f"✅ Loaded {len(db.standard_cards)} Standard cards")
    else:
        print("No existing cards found, downloading...")
        db.download_standard_cards()
        print(f"✅ Downloaded {len(db.standard_cards)} Standard cards")
    
    # Test card lookup
    test_card = db.get_card_by_name("Lightning Bolt")
    if test_card:
        print(f"✅ Found Lightning Bolt: {test_card['type_line']}")
        print(f"   Types: {test_card['types']}")
        print(f"   Colors: {test_card['colors']}")
    else:
        print("❌ Lightning Bolt not found")
    
    # Test land detection
    land_card = db.get_card_by_name("Island")
    if land_card:
        is_land = 'Land' in land_card.get('types', [])
        print(f"✅ Island is a land: {is_land}")
    
    # Search for some cards
    creatures = db.get_cards_by_type("Creature")
    print(f"✅ Found {len(creatures)} creatures")
    
    if len(creatures) > 0:
        sample = creatures[0]
        print(f"   Sample creature: {sample['name']} - {sample['type_line']}")

if __name__ == "__main__":
    test_card_db()