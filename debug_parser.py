#!/usr/bin/env python3
"""
Debug the deck parsing specifically
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.mtgtop8_scraper import MTGTop8Scraper
import requests
from bs4 import BeautifulSoup

def debug_parsing():
    print("Debugging deck parsing...")
    
    # Get a specific deck page
    url = "https://www.mtgtop8.com/?d=746506"
    response = requests.get(url)
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Print raw text to see structure
        page_text = soup.get_text()
        lines = page_text.split('\n')
        
        print("Looking for deck sections in text...")
        for i, line in enumerate(lines):
            line = line.strip()
            if any(keyword in line.upper() for keyword in ['LANDS', 'CREATURES', 'SPELLS', 'SIDEBOARD']):
                print(f"Line {i}: {repr(line)}")
                # Show next few lines
                for j in range(1, 6):
                    if i + j < len(lines):
                        next_line = lines[i + j].strip()
                        if next_line:
                            print(f"  +{j}: {repr(next_line)}")
                print()
        
        # Now test the actual parsing
        print("Testing parser...")
        scraper = MTGTop8Scraper()
        deck_data = {
            'deck_id': '746506',
            'mainboard': [],
            'sideboard': []
        }
        
        scraper._parse_decklist(soup, deck_data)
        
        print(f"Parsed mainboard cards: {len(deck_data['mainboard'])}")
        print(f"Parsed sideboard cards: {len(deck_data['sideboard'])}")
        
        if deck_data['mainboard']:
            print("First few mainboard cards:")
            for card in deck_data['mainboard'][:5]:
                print(f"  {card['quantity']}x {card['name']}")

if __name__ == "__main__":
    debug_parsing()