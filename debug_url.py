#!/usr/bin/env python3
"""
Debug URL formats
"""

import requests
from bs4 import BeautifulSoup

def test_urls():
    print("Testing different URL formats...")
    
    deck_id = "746506"
    
    # Test different URL formats
    urls_to_test = [
        f"https://www.mtgtop8.com/?d={deck_id}",
        f"https://www.mtgtop8.com/deck?d={deck_id}",
        f"https://www.mtgtop8.com/event?e=72076&d={deck_id}&f=ST"
    ]
    
    for url in urls_to_test:
        print(f"\nTesting URL: {url}")
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        print(f"Final URL after redirects: {response.url}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for deck content indicators
            text = soup.get_text().lower()
            
            indicators = ['mainboard', 'sideboard', 'creatures', 'lands', 'spells']
            found_indicators = [ind for ind in indicators if ind in text]
            
            print(f"Deck content indicators found: {found_indicators}")
            
            # Look for card patterns
            lines = soup.get_text().split('\n')
            card_lines = []
            for line in lines:
                line = line.strip()
                # Look for lines that start with numbers (potential card quantities)
                if line and line[0].isdigit() and len(line.split()) > 1:
                    card_lines.append(line)
            
            print(f"Potential card lines found: {len(card_lines)}")
            if card_lines:
                print("Sample lines:")
                for line in card_lines[:5]:
                    print(f"  {repr(line)}")

if __name__ == "__main__":
    test_urls()