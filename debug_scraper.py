#!/usr/bin/env python3
"""
Debug MTGTop8 scraper to understand HTML structure
"""

import requests
from bs4 import BeautifulSoup
import re

def debug_deck_page():
    url = "https://www.mtgtop8.com/event?e=72076&d=593031&f=ST"
    
    print(f"🔍 Debugging deck page: {url}")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    response = session.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Look for deck content in various ways
    print("\n📄 Raw text content (first 1000 chars):")
    text = soup.get_text()
    print(text[:1000])
    
    print("\n🔍 Looking for table elements...")
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables")
    
    for i, table in enumerate(tables[:3]):  # Check first 3 tables
        print(f"\nTable {i+1} content (first 500 chars):")
        print(table.get_text()[:500])
    
    print("\n🔍 Looking for div elements with card info...")
    divs = soup.find_all('div')
    for i, div in enumerate(divs):
        div_text = div.get_text().strip()
        if any(word in div_text.lower() for word in ['creatures', 'lands', 'spells', 'sideboard']):
            print(f"\nDiv {i}: {div_text[:200]}")
    
    print("\n🔍 Looking for specific patterns...")
    # Look for common patterns
    if 'SIDEBOARD' in text:
        print("✅ Found SIDEBOARD marker")
        parts = text.split('SIDEBOARD')
        print(f"Mainboard section (last 300 chars): ...{parts[0][-300:]}")
        if len(parts) > 1:
            print(f"Sideboard section (first 300 chars): {parts[1][:300]}")
    else:
        print("❌ No SIDEBOARD marker found")
    
    # Look for card quantity patterns
    card_pattern = r'(\d+)\s+([A-Za-z][^0-9\n]*?)(?=\s*\d+\s+[A-Z]|\s*$)'
    matches = re.findall(card_pattern, text)
    print(f"\n🃏 Found {len(matches)} potential card entries")
    for i, (qty, name) in enumerate(matches[:10]):
        print(f"  {qty}x {name.strip()}")

if __name__ == "__main__":
    debug_deck_page()