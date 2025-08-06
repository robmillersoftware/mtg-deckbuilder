#!/usr/bin/env python3
"""
Comprehensive MTGTop8 scraper for all Standard events from 8/1/2025-8/6/2025
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from datetime import datetime
from typing import Dict, List

class ComprehensiveScraper:
    def __init__(self):
        self.base_url = "https://mtgtop8.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; MTG-Research-Bot/1.0)'
        })
    
    def scrape_all_standard_events(self, start_date="2025-08-01", end_date="2025-08-06"):
        """Scrape ALL Standard events in the date range"""
        print(f"🔍 Scraping ALL Standard events from {start_date} to {end_date}")
        
        all_decks = []
        page = 1
        
        while True:
            print(f"📄 Scraping page {page}...")
            
            # MTGTop8 event search URL for Standard
            url = f"{self.base_url}/format?f=ST&meta=51&a=&page={page}"
            
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find event links on this page
                event_links = soup.find_all('a', href=re.compile(r'event\\?e=\\d+'))
                
                if not event_links:
                    print(f"✅ No more events found on page {page}. Stopping.")
                    break
                
                print(f"   Found {len(event_links)} events on page {page}")
                
                # Process each event
                for link in event_links:
                    event_url = link.get('href')
                    if event_url.startswith('/'):
                        event_url = self.base_url + event_url
                    
                    # Extract event ID
                    event_match = re.search(r'event\\?e=(\\d+)', event_url)
                    if not event_match:
                        continue
                    
                    event_id = event_match.group(1)
                    
                    # Get event details and check if in date range
                    event_decks = self.scrape_event_decks(event_id, start_date, end_date)
                    if event_decks:
                        all_decks.extend(event_decks)
                        print(f"   ✅ Event {event_id}: {len(event_decks)} decks")
                    
                    # Be nice to the server
                    time.sleep(1.0)
                
                page += 1
                
                # Safety limit
                if page > 20:  # Prevent infinite loops
                    print("⚠️ Hit safety limit of 20 pages")
                    break
                    
            except Exception as e:
                print(f"❌ Error on page {page}: {e}")
                break
        
        print(f"🎯 Total decks scraped: {len(all_decks)}")
        
        # Save the comprehensive data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mtgtop8_comprehensive_{timestamp}.json"
        filepath = os.path.join("data/raw", filename)
        
        os.makedirs("data/raw", exist_ok=True)
        
        data = {
            'scrape_date': datetime.now().isoformat(),
            'date_range': f"{start_date} to {end_date}",
            'total_decks': len(all_decks),
            'decks': all_decks
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved comprehensive data to: {filepath}")
        return filepath
    
    def scrape_event_decks(self, event_id: str, start_date: str, end_date: str) -> List[Dict]:
        """Scrape all decks from a single event"""
        try:
            event_url = f"{self.base_url}/event?e={event_id}"
            response = self.session.get(event_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get event info
            event_info = self.parse_event_info(soup)
            
            # Check if event is in our date range (basic check)
            # For now, we'll scrape all events and filter later
            
            # Find deck links
            deck_links = soup.find_all('a', href=re.compile(r'\\?e=\\d+&d=\\d+&f=ST'))
            
            if not deck_links:
                return []
            
            decks = []
            for deck_link in deck_links:
                deck_url = deck_link.get('href')
                if deck_url.startswith('?'):
                    deck_url = self.base_url + '/event' + deck_url
                elif deck_url.startswith('/'):
                    deck_url = self.base_url + deck_url
                
                # Extract deck ID
                deck_match = re.search(r'd=(\\d+)', deck_url)
                if deck_match:
                    deck_id = deck_match.group(1)
                    deck_data = self.scrape_single_deck(deck_id, event_id, event_info)
                    if deck_data:
                        decks.append(deck_data)
            
            return decks
            
        except Exception as e:
            print(f"   ⚠️ Error scraping event {event_id}: {e}")
            return []
    
    def parse_event_info(self, soup: BeautifulSoup) -> Dict:
        """Parse event information from event page"""
        event_info = {
            'event_name': 'Unknown Event',
            'event_date': 'Unknown',
            'location': 'Unknown'
        }
        
        try:
            # Try to find event name
            title_elem = soup.find('title')
            if title_elem:
                title_text = title_elem.get_text(strip=True)
                if 'Standard' in title_text:
                    event_info['event_name'] = title_text.replace(' - MTGTop8', '')
            
            # Try to find date and location (this is tricky with MTGTop8's layout)
            text_content = soup.get_text()
            
            # Look for date patterns
            date_patterns = [
                r'(\\d{1,2}/\\d{1,2}/\\d{4})',
                r'(\\d{4}-\\d{2}-\\d{2})'
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, text_content)
                if date_match:
                    event_info['event_date'] = date_match.group(1)
                    break
                    
        except Exception as e:
            print(f"   ⚠️ Error parsing event info: {e}")
        
        return event_info
    
    def scrape_single_deck(self, deck_id: str, event_id: str, event_info: Dict) -> Dict:
        """Scrape a single deck"""
        try:
            deck_url = f"{self.base_url}/event?e={event_id}&d={deck_id}&f=ST"
            response = self.session.get(deck_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse deck data
            deck_data = {
                'deck_id': deck_id,
                'event_id': event_id,
                'event_name': event_info.get('event_name', 'Unknown'),
                'event_date': event_info.get('event_date', 'Unknown'),
                'mainboard': [],
                'sideboard': [],
                'player_name': 'Unknown',
                'archetype': 'Unknown'
            }
            
            # Parse the deck list from page text
            page_text = soup.get_text()
            
            # Try to find player name
            player_match = re.search(r'by ([^\\n]+)', page_text, re.IGNORECASE)
            if player_match:
                deck_data['player_name'] = player_match.group(1).strip()
            
            # Parse deck sections
            current_section = None
            lines = page_text.split('\\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if 'LANDS' in line.upper():
                    current_section = 'mainboard'
                elif 'CREATURES' in line.upper():
                    current_section = 'mainboard'
                elif 'SPELLS' in line.upper():
                    current_section = 'mainboard'
                elif 'ARTIFACTS' in line.upper():
                    current_section = 'mainboard'
                elif 'ENCHANTMENTS' in line.upper():
                    current_section = 'mainboard'
                elif 'PLANESWALKERS' in line.upper():
                    current_section = 'mainboard'
                elif 'INSTANTS' in line.upper():
                    current_section = 'mainboard'
                elif 'SORCERIES' in line.upper():
                    current_section = 'mainboard'
                elif 'SIDEBOARD' in line.upper():
                    current_section = 'sideboard'
                
                # Parse card lines (format: "4 Card Name")
                card_match = re.match(r'^(\\d+)\\s+(.+)', line)
                if card_match and current_section:
                    quantity = int(card_match.group(1))
                    card_name = card_match.group(2).strip()
                    
                    # Skip obvious non-cards
                    if len(card_name) < 2 or card_name.isdigit():
                        continue
                    
                    deck_data[current_section].append({
                        'name': card_name,
                        'quantity': quantity
                    })
            
            return deck_data
            
        except Exception as e:
            print(f"   ⚠️ Error scraping deck {deck_id}: {e}")
            return None

def main():
    scraper = ComprehensiveScraper()
    filepath = scraper.scrape_all_standard_events()
    print(f"\\n✅ Comprehensive scraping complete!")
    print(f"📁 Data saved to: {filepath}")

if __name__ == "__main__":
    main()