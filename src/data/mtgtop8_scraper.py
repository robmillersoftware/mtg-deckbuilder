import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from typing import Dict, List, Optional
import re
from urllib.parse import urljoin, urlparse
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MTGTop8Scraper:
    def __init__(self, base_url: str = "https://www.mtgtop8.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_standard_events(self, limit: int = 50) -> List[Dict]:
        """Fetch recent Standard events from MTGTop8"""
        events = []
        try:
            # Standard format page
            url = f"{self.base_url}/format?f=ST"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find event links (both old and new formats)
            event_links = soup.find_all('a', href=re.compile(r'(event\?e=\d+|\?.*e=\d+)'))
            
            for link in event_links[:limit]:
                event_url = urljoin(self.base_url, link['href'])
                event_id = re.search(r'e=(\d+)', link['href']).group(1)
                
                event_data = {
                    'event_id': event_id,
                    'event_url': event_url,
                    'name': link.text.strip(),
                    'scraped_at': datetime.now().isoformat()
                }
                
                events.append(event_data)
                
            logger.info(f"Found {len(events)} Standard events")
            return events
            
        except Exception as e:
            logger.error(f"Error fetching events: {e}")
            return []
    
    def get_event_decks(self, event_id: str) -> List[Dict]:
        """Get all decklists from a specific event"""
        decks = []
        try:
            url = f"{self.base_url}/event?e={event_id}"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find deck links (updated pattern for new format)
            deck_links = soup.find_all('a', href=re.compile(r'\?.*d=\d+'))
            
            logger.info(f"Found {len(deck_links)} potential deck links in event {event_id}")
            
            for link in deck_links:
                try:
                    deck_url = urljoin(self.base_url, link['href'])
                    deck_match = re.search(r'd=(\d+)', link['href'])
                    if not deck_match:
                        continue
                        
                    deck_id = deck_match.group(1)
                    logger.debug(f"Attempting to scrape deck {deck_id}")
                    
                    deck_data = self.scrape_deck(deck_id, event_id)
                    if deck_data:
                        deck_data['event_id'] = event_id
                        decks.append(deck_data)
                        logger.debug(f"Successfully scraped deck {deck_id}")
                    else:
                        logger.debug(f"Failed to scrape deck {deck_id}")
                        
                    # Rate limiting
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error processing deck link {link.get('href', '')}: {e}")
                    continue
                
            logger.info(f"Scraped {len(decks)} decks from event {event_id}")
            return decks
            
        except Exception as e:
            logger.error(f"Error fetching decks for event {event_id}: {e}")
            return []
    
    def scrape_deck(self, deck_id: str, event_id: str = None) -> Optional[Dict]:
        """Scrape a specific deck by ID"""
        try:
            # Use event-based URL format which works better
            if event_id:
                url = f"{self.base_url}/event?e={event_id}&d={deck_id}&f=ST"
            else:
                # Fallback formats
                url = f"{self.base_url}/?d={deck_id}"
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            deck_data = {
                'deck_id': deck_id,
                'url': url,
                'mainboard': [],
                'sideboard': [],
                'player': '',
                'archetype': '',
                'placing': '',
                'scraped_at': datetime.now().isoformat()
            }
            
            # Get player name and archetype from page content
            page_text = soup.get_text()
            
            # Extract deck name/archetype (usually appears near the top)
            lines = page_text.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Look for player name patterns
                if not deck_data['player'] and ('by' in line.lower() or 'pilot' in line.lower()):
                    # Try to extract player name
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and len(next_line) < 50:  # Reasonable length for a name
                            deck_data['player'] = next_line
                
                # Look for deck archetype/name (often the title or first major text)
                if not deck_data['archetype'] and line and len(line) > 5 and len(line) < 100:
                    # Skip common headers and navigation
                    skip_patterns = ['mtg top', 'format', 'event', 'home', 'search', 'login', 'register', 'tix', 'price', 'total cards']
                    if not any(pattern in line.lower() for pattern in skip_patterns):
                        # This might be the deck name
                        if any(word in line.lower() for word in ['aggro', 'control', 'midrange', 'combo', 'ramp', 'burn', 'tempo']):
                            deck_data['archetype'] = line
                            break
                        # Or if it contains color names or strategy terms
                        elif any(word in line.lower() for word in ['red', 'blue', 'white', 'black', 'green', 'mono', 'azorius', 'dimir', 'rakdos', 'gruul', 'selesnya']):
                            deck_data['archetype'] = line
                            break
                        # Or if it's a reasonable deck name format
                        elif ' ' in line and not line.isupper() and not line.isdigit():
                            deck_data['archetype'] = line
                            break
            
            # Parse mainboard and sideboard
            self._parse_decklist(soup, deck_data)
            
            return deck_data
            
        except Exception as e:
            logger.error(f"Error scraping deck {deck_id}: {e}")
            return None
    
    def _parse_decklist(self, soup: BeautifulSoup, deck_data: Dict):
        """Parse mainboard and sideboard from deck page"""
        try:
            # Find the specific div that contains just the deck content
            deck_text = ""
            
            # Look for the specific pattern we found in debugging
            # The deck content is in a div that starts with "18 LANDS4 Great Furnace"
            divs = soup.find_all('div')
            for div in divs:
                div_text = div.get_text().strip()
                # Look for div that starts with card sections and includes SIDEBOARD
                # Pattern: starts with number + LANDS, contains CREATURES, contains SIDEBOARD
                if (re.match(r'^\d+\s+LANDS', div_text) and 
                    'CREATURES' in div_text and 
                    'SIDEBOARD' in div_text):
                    deck_text = div_text
                    logger.debug(f"Found deck div with content: {div_text[:100]}...")
                    break
            
            # If still not found, try more specific pattern matching
            if not deck_text:
                for div in divs:
                    div_text = div.get_text().strip()
                    # Look for pattern that includes both mainboard sections and sideboard
                    if ('LANDS' in div_text and 'CREATURES' in div_text and 
                        'SIDEBOARD' in div_text and len(div_text) < 1000):  # Not too long
                        deck_text = div_text
                        logger.debug(f"Found deck div (fallback): {div_text[:100]}...")
                        break
            
            if not deck_text:
                logger.warning("Could not find deck content div")
                return
            
            # Split into mainboard and sideboard
            if 'SIDEBOARD' not in deck_text:
                logger.warning("No SIDEBOARD section found")
                return
            
            parts = deck_text.split('SIDEBOARD')
            mainboard_text = parts[0]
            sideboard_text = parts[1] if len(parts) > 1 else ""
            
            # Parse mainboard
            self._parse_deck_section(mainboard_text, deck_data, 'mainboard')
            
            # Parse sideboard
            self._parse_deck_section(sideboard_text, deck_data, 'sideboard')
            
            logger.debug(f"Parsed {len(deck_data['mainboard'])} mainboard, {len(deck_data['sideboard'])} sideboard cards")
            
        except Exception as e:
            logger.error(f"Error parsing decklist: {e}")
    
    def _parse_deck_section(self, section_text: str, deck_data: Dict, section_name: str):
        """Parse a deck section (mainboard or sideboard)"""
        try:
            logger.debug(f"Parsing {section_name} section: {section_text[:200]}...")
            
            # Remove section headers but keep the card data
            clean_text = section_text
            
            # Remove section headers like "18 LANDS", "14 CREATURES", etc.
            clean_text = re.sub(r'\d+\s+(LANDS|CREATURES|SPELLS|ARTIFACTS|ENCHANTMENTS|PLANESWALKERS|INSTANTS.*?|OTHER.*?)(?=\d+\s+[A-Z])', '', clean_text, flags=re.IGNORECASE)
            
            logger.debug(f"After removing headers: {clean_text[:200]}...")
            
            # Find all card entries with pattern: number followed by card name
            # Cards are concatenated like: "4 Great Furnace 14 Mountain 2 Goblin Blast-Runner"
            # Handle special characters: hyphens, apostrophes, commas, slashes (double-faced cards)
            # We need to match: digit(s) + space + card name (until next digit+space or end)
            card_pattern = r'(\d+)\s+([A-Za-z][A-Za-z\-\'\,\s\/]+?)(?=\s+\d+\s+[A-Za-z]|\s*$)'
            
            matches = re.findall(card_pattern, clean_text)
            logger.debug(f"Found {len(matches)} potential matches")
            
            for quantity_str, card_name in matches:
                try:
                    quantity = int(quantity_str)
                    card_name = card_name.strip()
                    
                    # Skip very short names, section headers, or invalid names
                    skip_patterns = ['LANDS', 'CREATURES', 'SPELLS', 'ARTIFACTS', 'ENCHANTMENTS', 'PLANESWALKERS']
                    if (len(card_name) < 3 or 
                        card_name.upper() in skip_patterns or
                        any(pattern in card_name.upper() for pattern in ['INSTANTS', 'OTHER', 'SORC'])):
                        logger.debug(f"Skipping {card_name} (invalid)")
                        continue
                    
                    card_entry = {
                        'quantity': quantity,
                        'name': card_name,
                        'section': section_name
                    }
                    
                    deck_data[section_name].append(card_entry)
                    logger.debug(f"Added {quantity}x {card_name} to {section_name}")
                    
                except ValueError:
                    logger.debug(f"ValueError parsing {quantity_str}: {card_name}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing {section_name} section: {e}")
    
    def scrape_standard_meta(self, num_events: int = 10, output_dir: str = "data/raw", parallel: bool = True) -> str:
        """Scrape recent Standard metagame data"""
        import concurrent.futures
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get recent events
        events = self.get_standard_events(limit=num_events)
        all_decks = []
        
        if parallel and len(events) > 1:
            logger.info(f"Scraping {len(events)} events in parallel...")
            
            # Use ThreadPoolExecutor for parallel scraping
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all events for parallel processing
                future_to_event = {
                    executor.submit(self._scrape_event_with_rate_limit, event): event
                    for event in events
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_event):
                    event = future_to_event[future]
                    try:
                        decks = future.result()
                        all_decks.extend(decks)
                        logger.info(f"✅ Completed event: {event['name']} ({len(decks)} decks)")
                    except Exception as e:
                        logger.error(f"❌ Failed event: {event['name']} - {e}")
        else:
            # Sequential scraping (original method)
            for event in events:
                logger.info(f"Scraping event: {event['name']}")
                decks = self.get_event_decks(event['event_id'])
                all_decks.extend(decks)
                
                # Rate limiting between events
                time.sleep(2)
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mtgtop8_standard_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        data = {
            'scraped_at': datetime.now().isoformat(),
            'events_count': len(events),
            'decks_count': len(all_decks),
            'events': events,
            'decks': all_decks
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(all_decks)} decks from {len(events)} events to {filepath}")
        return filepath
    
    def _scrape_event_with_rate_limit(self, event: Dict) -> List[Dict]:
        """Scrape a single event with rate limiting"""
        decks = self.get_event_decks(event['event_id'])
        
        # Small delay to be respectful to the server
        time.sleep(1)
        return decks

if __name__ == "__main__":
    scraper = MTGTop8Scraper()
    scraper.scrape_standard_meta(num_events=5)