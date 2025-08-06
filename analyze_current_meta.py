#!/usr/bin/env python3
"""
Analyze the current meta data to see all archetypes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.card_database import CardDatabase
from data.metagame_analyzer import MetagameAnalyzer

def analyze_current_meta():
    print("📈 Analyzing current meta data for all archetypes...")
    
    # Load card database for proper analysis
    card_db = CardDatabase("data/cards")
    if not card_db.load_latest_standard_cards():
        print("⚠️ No card database - downloading...")
        card_db.download_standard_cards()
    
    # Initialize analyzer with card database
    analyzer = MetagameAnalyzer("data/raw", card_db)
    
    # Load the current data
    if not analyzer.load_scraped_data():
        print("❌ Failed to load scraped data")
        return
    
    print(f"✅ Loaded {len(analyzer.deck_data)} total decks")
    
    # Get comprehensive archetype breakdown
    breakdown = analyzer.get_archetype_breakdown()
    
    print(f"\n🎯 Found {len(breakdown)} unique archetypes:")
    print("="*80)
    
    # Sort by meta percentage (descending)
    sorted_archetypes = sorted(
        breakdown.items(),
        key=lambda x: x[1]['percentage'],
        reverse=True
    )
    
    total_percentage = 0
    for i, (archetype, stats) in enumerate(sorted_archetypes, 1):
        print(f"{i:2d}. {stats['percentage']:5.1f}% | {stats['deck_count']:3d} decks | {archetype}")
        total_percentage += stats['percentage']
        
        # Show a few key cards for context
        key_cards = [card for card, count in stats['key_cards'][:4]]
        if key_cards:
            print(f"     └─ Key cards: {', '.join(key_cards)}")
        
        # Show a sample deck source if available
        if stats['sample_decks']:
            sample = stats['sample_decks'][0]
            event_name = sample.get('event_name', 'Unknown Event')
            player_name = sample.get('player_name', 'Unknown Player')
            print(f"     └─ Example: {event_name} by {player_name}")
        print()
    
    print("="*80)
    print(f"Total coverage: {total_percentage:.1f}%")
    
    # Show detailed statistics
    print(f"\n📊 Detailed Meta Statistics:")
    print(f"• Total unique archetypes: {len(breakdown)}")
    print(f"• Total decks analyzed: {len(analyzer.deck_data)}")
    print(f"• Most popular archetype: {sorted_archetypes[0][0]} ({sorted_archetypes[0][1]['percentage']:.1f}%)")
    
    # Count tier distribution
    tier_1 = len([a for a in sorted_archetypes if a[1]['percentage'] >= 10])
    tier_2 = len([a for a in sorted_archetypes if 5 <= a[1]['percentage'] < 10])
    tier_3 = len([a for a in sorted_archetypes if 2 <= a[1]['percentage'] < 5])
    tier_4 = len([a for a in sorted_archetypes if 1 <= a[1]['percentage'] < 2])
    tier_5 = len([a for a in sorted_archetypes if a[1]['percentage'] < 1])
    
    print(f"• Tier 1 archetypes (≥10%): {tier_1}")
    print(f"• Tier 2 archetypes (5-9.9%): {tier_2}")
    print(f"• Tier 3 archetypes (2-4.9%): {tier_3}")
    print(f"• Tier 4 archetypes (1-1.9%): {tier_4}")
    print(f"• Tier 5 archetypes (<1%): {tier_5}")
    
    # Show diversity metrics
    total_decks = len(analyzer.deck_data)
    print(f"\n🌈 Meta Diversity:")
    print(f"• Archetype diversity index: {len(breakdown)/total_decks:.3f}")
    print(f"• Top 3 archetypes represent: {sum(a[1]['percentage'] for a in sorted_archetypes[:3]):.1f}% of meta")
    print(f"• Top 5 archetypes represent: {sum(a[1]['percentage'] for a in sorted_archetypes[:5]):.1f}% of meta")
    
    return len(breakdown), sorted_archetypes

if __name__ == "__main__":
    count, archetypes = analyze_current_meta()