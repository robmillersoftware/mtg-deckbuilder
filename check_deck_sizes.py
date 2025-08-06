#!/usr/bin/env python3
"""
Check actual deck sizes in our scraped data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.metagame_analyzer import MetagameAnalyzer
from data.card_database import CardDatabase

def check_deck_sizes():
    print("📊 Checking deck sizes in scraped data...")
    
    # Load data
    card_db = CardDatabase("data/cards")
    if not card_db.load_latest_standard_cards():
        print("No card database found")
        return
    
    analyzer = MetagameAnalyzer("data/raw", card_db)
    if not analyzer.load_scraped_data():
        print("No data found")
        return
    
    print(f"Analyzing {len(analyzer.deck_data)} decks...")
    
    # Count deck sizes
    size_counts = {}
    examples = {}
    
    for i, deck in enumerate(analyzer.deck_data):
        mainboard = deck.get('mainboard', [])
        sideboard = deck.get('sideboard', [])
        
        mainboard_count = sum(card['quantity'] for card in mainboard)
        sideboard_count = sum(card['quantity'] for card in sideboard)
        
        size_key = f"{mainboard_count}+{sideboard_count}"
        
        if size_key not in size_counts:
            size_counts[size_key] = 0
            examples[size_key] = {
                'deck': deck,
                'mainboard_cards': len(mainboard),
                'sideboard_cards': len(sideboard)
            }
        
        size_counts[size_key] += 1
        
        # Show first few examples
        if i < 10:
            print(f"Deck {i+1}: {mainboard_count} mainboard + {sideboard_count} sideboard = {mainboard_count + sideboard_count} total")
            print(f"   Mainboard unique cards: {len(mainboard)}")
            print(f"   Sideboard unique cards: {len(sideboard)}")
            if mainboard:
                print(f"   Sample mainboard: {mainboard[0]['name']} x{mainboard[0]['quantity']}")
            print()
    
    print(f"\n📈 Deck size distribution:")
    print("="*50)
    
    for size_key, count in sorted(size_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(analyzer.deck_data)) * 100
        print(f"{size_key:>10} cards: {count:>3} decks ({percentage:>5.1f}%)")
        
        # Show example for common sizes
        if count >= 5:
            example = examples[size_key]
            deck = example['deck']
            print(f"           Example: {deck.get('player_name', 'Unknown')} - {example['mainboard_cards']} unique mainboard, {example['sideboard_cards']} unique sideboard")
    
    print("="*50)
    
    # Find closest to 75 cards
    closest_75 = []
    for size_key, count in size_counts.items():
        main_sb = size_key.split('+')
        total = int(main_sb[0]) + int(main_sb[1])
        if 70 <= total <= 80:  # Close to 75
            closest_75.append((size_key, count, total))
    
    if closest_75:
        print(f"\n🎯 Decks closest to 75 cards:")
        for size_key, count, total in sorted(closest_75, key=lambda x: abs(75 - x[2])):
            print(f"   {size_key} ({total} total): {count} decks")
    
    # Check for exactly 60+15
    exact_75 = size_counts.get("60+15", 0)
    print(f"\n✅ Exactly 60+15: {exact_75} decks")
    
    return size_counts

if __name__ == "__main__":
    check_deck_sizes()