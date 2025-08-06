#!/usr/bin/env python3
"""
Debug why some decks are missing exactly 1 card
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.mtgtop8_scraper import MTGTop8Scraper

def debug_missing_cards():
    # Load the test data we just scraped
    with open('data/test/mtgtop8_standard_20250806_114204.json', 'r') as f:
        data = json.load(f)
    
    print("🔍 Debugging decks missing 1 card...")
    
    missing_one = []
    perfect = []
    
    for deck in data['decks']:
        mainboard = deck.get('mainboard', [])
        sideboard = deck.get('sideboard', [])
        
        mainboard_count = sum(card['quantity'] for card in mainboard)
        sideboard_count = sum(card['quantity'] for card in sideboard)
        
        if mainboard_count == 59 and sideboard_count == 15:
            missing_one.append(deck)
        elif mainboard_count == 60 and sideboard_count == 15:
            perfect.append(deck)
    
    print(f"Found {len(missing_one)} decks missing 1 card, {len(perfect)} perfect decks")
    
    if missing_one:
        print("\n🔍 Re-scraping a missing-1-card deck to see raw content...")
        
        # Re-scrape one of the problematic decks with full debug
        problem_deck = missing_one[0]
        deck_id = problem_deck['deck_id']
        event_id = problem_deck.get('event_id', '')
        
        print(f"Re-scraping deck {deck_id} from event {event_id}")
        
        scraper = MTGTop8Scraper()
        
        # Enable debug logging
        import logging
        logging.getLogger('data.mtgtop8_scraper').setLevel(logging.DEBUG)
        
        fresh_deck = scraper.scrape_deck(deck_id, event_id)
        
        if fresh_deck:
            fresh_main = sum(card['quantity'] for card in fresh_deck['mainboard'])
            fresh_side = sum(card['quantity'] for card in fresh_deck['sideboard'])
            print(f"\nFresh scrape: {fresh_main}+{fresh_side} = {fresh_main + fresh_side}")
            
            print("\nMainboard cards:")
            for card in fresh_deck['mainboard']:
                print(f"  {card['quantity']}x {card['name']}")
    
    if perfect:
        print(f"\n✅ Perfect deck example (deck {perfect[0]['deck_id']}):")
        for card in perfect[0]['mainboard']:
            print(f"  {card['quantity']}x {card['name']}")

if __name__ == "__main__":
    debug_missing_cards()