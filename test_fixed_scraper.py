#!/usr/bin/env python3
"""
Test the fixed scraper on a known deck
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.mtgtop8_scraper import MTGTop8Scraper

def test_scraper():
    print("🧪 Testing fixed scraper...")
    
    scraper = MTGTop8Scraper()
    
    # Test on a known deck that should have 60+15 cards
    deck_data = scraper.scrape_deck("593031", "72076")
    
    if not deck_data:
        print("❌ Failed to scrape deck")
        return
    
    print(f"✅ Scraped deck: {deck_data['archetype']}")
    print(f"👤 Player: {deck_data['player']}")
    
    mainboard_total = sum(card['quantity'] for card in deck_data['mainboard'])
    sideboard_total = sum(card['quantity'] for card in deck_data['sideboard'])
    
    print(f"\n📊 Card counts:")
    print(f"   Mainboard: {mainboard_total} cards ({len(deck_data['mainboard'])} unique)")
    print(f"   Sideboard: {sideboard_total} cards ({len(deck_data['sideboard'])} unique)")
    print(f"   Total: {mainboard_total + sideboard_total} cards")
    
    print(f"\n🃏 All mainboard cards:")
    for card in deck_data['mainboard']:
        print(f"   {card['quantity']}x {card['name']}")
    
    print(f"\n🃏 All sideboard cards:")
    for card in deck_data['sideboard']:
        print(f"   {card['quantity']}x {card['name']}")
    
    # Check if it's valid 75-card format
    if mainboard_total == 60 and sideboard_total == 15:
        print("\n✅ VALID 75-card deck format!")
    else:
        print(f"\n❌ Invalid deck format: {mainboard_total}+{sideboard_total} != 60+15")
    
    return deck_data

if __name__ == "__main__":
    test_scraper()