#!/usr/bin/env python3
"""
Check if the requested cards exist in the database
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.card_database import CardDatabase

def check_cards():
    print("Checking for requested cards in database...")
    
    # Initialize card database
    card_db = CardDatabase("data/cards")
    if not card_db.load_latest_standard_cards():
        print("No card database found")
        return
    
    print(f"Database contains {len(card_db.standard_cards)} cards")
    
    # Search for the requested cards
    requested_cards = ["Tezzeret", "Cruel Captain", "Repurposing Bay"]
    
    for requested in requested_cards:
        print(f"\n🔍 Searching for '{requested}':")
        
        # Exact matches
        exact_matches = []
        partial_matches = []
        
        for card_id, card_data in card_db.standard_cards.items():
            card_name = card_data.get('name', '')
            
            if requested.lower() == card_name.lower():
                exact_matches.append(card_data)
            elif requested.lower() in card_name.lower():
                partial_matches.append(card_data)
        
        if exact_matches:
            print(f"   ✅ Exact matches:")
            for card in exact_matches:
                print(f"      - {card['name']} ({card['type_line']})")
        
        if partial_matches:
            print(f"   🔍 Partial matches:")
            for card in partial_matches[:5]:  # Show first 5
                print(f"      - {card['name']} ({card['type_line']})")
            if len(partial_matches) > 5:
                print(f"      ... and {len(partial_matches) - 5} more")
        
        if not exact_matches and not partial_matches:
            print(f"   ❌ No matches found")

if __name__ == "__main__":
    check_cards()