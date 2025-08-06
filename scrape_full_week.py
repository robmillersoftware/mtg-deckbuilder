#!/usr/bin/env python3
"""
Scrape all Standard events from MTGTop8 for the past week (8/1/2025-8/6/2025)
"""

import sys
import os
import json
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.mtgtop8_scraper import MTGTop8Scraper
from data.card_database import CardDatabase
from data.metagame_analyzer import MetagameAnalyzer

def scrape_full_week():
    print("🔍 Scraping ALL Standard events from MTGTop8 (8/1/2025-8/6/2025)...")
    
    # Initialize scraper
    scraper = MTGTop8Scraper()
    
    # Scrape much more comprehensive data
    print("📊 Scraping Standard events (this will take a while)...")
    output_file = scraper.scrape_standard_meta(
        num_events=50  # Increase from default to get more comprehensive data
    )
    
    if not output_file:
        print("❌ Failed to scrape data")
        return
    
    # Analyze the scraped data
    print("\n📈 Analyzing comprehensive meta data...")
    
    # Load card database for proper analysis
    card_db = CardDatabase("data/cards")
    if not card_db.load_latest_standard_cards():
        print("⚠️ No card database - downloading...")
        card_db.download_standard_cards()
    
    # Initialize analyzer with card database
    analyzer = MetagameAnalyzer("data/raw", card_db)
    
    # Load the newly scraped data
    if not analyzer.load_scraped_data():
        print("❌ Failed to load scraped data")
        return
    
    print(f"✅ Loaded {len(analyzer.deck_data)} total decks")
    
    # Get comprehensive archetype breakdown
    breakdown = analyzer.get_archetype_breakdown()
    
    print(f"\n🎯 Found {len(breakdown)} unique archetypes:")
    print("="*60)
    
    # Sort by meta percentage (descending)
    sorted_archetypes = sorted(
        breakdown.items(),
        key=lambda x: x[1]['percentage'],
        reverse=True
    )
    
    total_percentage = 0
    for archetype, stats in sorted_archetypes:
        print(f"{stats['percentage']:5.1f}% | {stats['deck_count']:3d} decks | {archetype}")
        total_percentage += stats['percentage']
        
        # Show a few key cards for context
        key_cards = [card for card, count in stats['key_cards'][:3]]
        if key_cards:
            print(f"       └─ Key cards: {', '.join(key_cards)}")
        print()
    
    print("="*60)
    print(f"Total coverage: {total_percentage:.1f}%")
    
    # Show some interesting stats
    print(f"\n📊 Meta Statistics:")
    print(f"• Total unique archetypes: {len(breakdown)}")
    print(f"• Total decks analyzed: {len(analyzer.deck_data)}")
    print(f"• Most popular archetype: {sorted_archetypes[0][0]} ({sorted_archetypes[0][1]['percentage']:.1f}%)")
    
    # Count tier distribution
    tier_1 = len([a for a in sorted_archetypes if a[1]['percentage'] >= 10])
    tier_2 = len([a for a in sorted_archetypes if 5 <= a[1]['percentage'] < 10])
    tier_3 = len([a for a in sorted_archetypes if 2 <= a[1]['percentage'] < 5])
    tier_4 = len([a for a in sorted_archetypes if a[1]['percentage'] < 2])
    
    print(f"• Tier 1 archetypes (≥10%): {tier_1}")
    print(f"• Tier 2 archetypes (5-9.9%): {tier_2}")
    print(f"• Tier 3 archetypes (2-4.9%): {tier_3}")
    print(f"• Tier 4 archetypes (<2%): {tier_4}")
    
    # Export comprehensive analysis
    analysis_file = analyzer.export_analysis("data/processed")
    print(f"\n💾 Exported full analysis to: {analysis_file}")
    
    return len(breakdown), analyzer

if __name__ == "__main__":
    archetype_count, analyzer = scrape_full_week()