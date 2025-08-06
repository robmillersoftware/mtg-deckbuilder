#!/usr/bin/env python3
"""
Test script to debug MTGTop8 scraper
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.mtgtop8_scraper import MTGTop8Scraper

def test_scraper():
    print("Testing MTGTop8 Scraper...")
    
    scraper = MTGTop8Scraper()
    
    # Test getting events
    print("1. Getting Standard events...")
    events = scraper.get_standard_events(limit=2)
    print(f"Found {len(events)} events")
    
    for event in events:
        print(f"Event: {event['name']} (ID: {event['event_id']})")
        
        # Test getting decks from first event
        if event == events[0]:
            print(f"2. Getting decks from event {event['event_id']}...")
            decks = scraper.get_event_decks(event['event_id'])
            print(f"Found {len(decks)} decks")
            
            if decks:
                print("First deck:")
                deck = decks[0]
                print(f"  Player: {deck.get('player', 'Unknown')}")
                print(f"  Archetype: {deck.get('archetype', 'Unknown')}")
                print(f"  Mainboard cards: {len(deck.get('mainboard', []))}")
                print(f"  Sideboard cards: {len(deck.get('sideboard', []))}")
                
                # Show first few cards
                mainboard = deck.get('mainboard', [])
                if mainboard:
                    print("  Sample cards:")
                    for card in mainboard[:5]:
                        print(f"    {card['quantity']}x {card['name']}")
            break

if __name__ == "__main__":
    test_scraper()