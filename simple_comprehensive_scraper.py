#!/usr/bin/env python3
"""
Simple comprehensive scraper using the working approach but with more pages
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.mtgtop8_scraper import MTGTop8Scraper

def scrape_comprehensive():
    print("🔍 Scraping comprehensive Standard meta data...")
    
    # Use our working scraper but get more events
    scraper = MTGTop8Scraper()
    
    # Scrape a lot more events to get full archetype diversity
    print("📊 Scraping 100+ Standard events (this will take several minutes)...")
    output_file = scraper.scrape_standard_meta(
        num_events=100  # Much larger sample
    )
    
    if not output_file:
        print("❌ Failed to scrape data")
        return None
    
    print(f"✅ Scraping complete! Data saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    scrape_comprehensive()