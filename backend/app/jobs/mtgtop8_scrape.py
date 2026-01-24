"""
mtgtop8 Weekly Scrape Job
Schedule: Sunday 6:00 AM UTC

Scrapes tournament decklists and meta data from mtgtop8.com.
Updates meta snapshots and card co-occurrence data.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from sqlalchemy.dialects.postgresql import insert

from app.db.session import async_session_factory
from app.models.meta import Event, Decklist, MetaSnapshot, CardCooccurrence
from app.models.card import Card
from app.core.config import settings

logger = logging.getLogger(__name__)

MTGTOP8_BASE_URL = "https://www.mtgtop8.com"
STANDARD_FORMAT_ID = "ST"
CEDH_FORMAT_ID = "cEDH"
REQUEST_DELAY = 1.0  # Be nice to the server

# Mapping of internal format names to mtgtop8 format IDs
FORMAT_CONFIG = {
    "standard": {"mtgtop8_id": "ST", "name": "Standard"},
    "cedh": {"mtgtop8_id": "cEDH", "name": "cEDH"},
}


async def fetch_page(client: httpx.AsyncClient, url: str) -> str:
    """Fetch a page with rate limiting."""
    await asyncio.sleep(REQUEST_DELAY)
    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    return response.text


async def scrape_recent_events(
    client: httpx.AsyncClient,
    format_id: str = STANDARD_FORMAT_ID,
    format_name: str = "standard",
    days: int = 14,
) -> List[Dict[str, Any]]:
    """Scrape recent events from mtgtop8 for a specific format."""
    events = []

    # Scrape the format events page
    url = f"{MTGTOP8_BASE_URL}/format?f={format_id}"
    html = await fetch_page(client, url)
    soup = BeautifulSoup(html, "html.parser")

    # Find event links
    event_links = soup.select("a[href*='event?e=']")

    for link in event_links[:50]:  # Limit to recent events
        try:
            href = link.get("href", "")
            event_id_match = re.search(r"e=(\d+)", href)
            if not event_id_match:
                continue

            event_id = event_id_match.group(1)
            event_name = link.get_text(strip=True)

            # Get parent row for date and player count
            parent = link.find_parent("tr")
            if parent:
                cells = parent.find_all("td")
                date_text = cells[-1].get_text(strip=True) if cells else ""
                player_count = None

                # Try to parse date
                event_date = None
                try:
                    # mtgtop8 uses format like "21/01/24"
                    event_date = datetime.strptime(date_text, "%d/%m/%y").date()
                except ValueError:
                    pass

                # Skip events without valid dates
                if not event_date:
                    continue

                # Check if within date range
                cutoff = datetime.now().date() - timedelta(days=days)
                if event_date < cutoff:
                    continue

                events.append({
                    "mtgtop8_id": event_id,
                    "name": event_name,
                    "date": event_date,
                    "format": format_name,
                    "url": f"{MTGTOP8_BASE_URL}/event?e={event_id}",
                })

        except Exception as e:
            logger.warning(f"Error parsing event link: {e}")
            continue

    logger.info(f"Found {len(events)} recent {format_name} events")
    return events


async def scrape_event_decklists(client: httpx.AsyncClient, event_id: str) -> List[Dict[str, Any]]:
    """Scrape all decklists from an event."""
    url = f"{MTGTOP8_BASE_URL}/event?e={event_id}"
    html = await fetch_page(client, url)
    soup = BeautifulSoup(html, "html.parser")

    # Find decklist links
    deck_links = soup.select("a[href*='&d=']")

    # First pass: collect all deck IDs and find best archetype for each
    deck_info = {}  # deck_id -> {archetype, placement}

    for link in deck_links:
        try:
            href = link.get("href", "")
            deck_id_match = re.search(r"d=(\d+)", href)
            if not deck_id_match:
                continue

            deck_id = deck_id_match.group(1)
            archetype = link.get_text(strip=True)

            # Skip navigation arrows, empty text, and other non-archetype text
            if not archetype or archetype in ["→", "←", ""] or len(archetype) < 3:
                continue

            # Skip if it looks like a navigation element (contains "deck" in lowercase)
            if "decks" in archetype.lower():
                continue

            # Get placement from parent row
            parent = link.find_parent("tr")
            placement = None
            if parent:
                cells = parent.find_all("td")
                if cells:
                    place_text = cells[0].get_text(strip=True)
                    if place_text.isdigit():
                        placement = int(place_text)

            # Keep the archetype with the longest name for each deck
            if deck_id not in deck_info or len(archetype) > len(deck_info[deck_id].get("archetype", "")):
                deck_info[deck_id] = {
                    "archetype": archetype,
                    "placement": placement,
                }

        except Exception as e:
            logger.warning(f"Error parsing deck link: {e}")
            continue

    # Build decklists from collected info
    decklists = []
    for deck_id, info in list(deck_info.items())[:32]:  # Top 32 max
        decklists.append({
            "mtgtop8_deck_id": deck_id,
            "archetype": info["archetype"],
            "placement": info["placement"],
            "url": f"{MTGTOP8_BASE_URL}/event?e={event_id}&d={deck_id}",
        })

    return decklists


async def scrape_decklist_cards(client: httpx.AsyncClient, event_id: str, deck_id: str) -> Dict[str, Any]:
    """Scrape the actual card list from a decklist page."""
    from bs4 import NavigableString

    url = f"{MTGTOP8_BASE_URL}/event?e={event_id}&d={deck_id}"
    html = await fetch_page(client, url)
    soup = BeautifulSoup(html, "html.parser")

    main_deck = []
    sideboard = []

    # Find all card entries - they're in divs with class "deck_line"
    # Main deck cards have id starting with "md", sideboard with "sb"
    card_divs = soup.select("div.deck_line")

    for div in card_divs:
        try:
            div_id = div.get("id", "")

            # Determine if main deck or sideboard based on id prefix
            if div_id.startswith("md"):
                target_list = main_deck
            elif div_id.startswith("sb"):
                target_list = sideboard
            else:
                continue

            # Get the card name from the span with class L14
            card_span = div.select_one("span.L14")
            if not card_span:
                continue

            card_name = card_span.get_text(strip=True)
            if not card_name:
                continue

            # Get quantity from text node before the span
            # The structure is: <div>4 <span>Card Name</span></div>
            quantity = 1  # Default
            for child in div.children:
                if isinstance(child, NavigableString):
                    text = child.strip()
                    if text.isdigit():
                        quantity = int(text)
                        break
                elif child == card_span:
                    break

            # Handle split cards (e.g., "Fire // Ice") - keep the full name
            # Handle double-faced cards - keep the full name as shown
            # No special processing needed, mtgtop8 shows them correctly

            target_list.append({
                "card_name": card_name,
                "quantity": quantity,
            })

        except Exception as e:
            logger.warning(f"Error parsing card div: {e}")
            continue

    return {
        "main_deck": main_deck,
        "sideboard": sideboard,
    }


async def calculate_meta_percentages(
    db: AsyncSession,
    format: str = "standard",
    days: int = 14,
) -> Dict[str, Dict[str, Any]]:
    """Calculate meta percentages for archetypes."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    # Get all recent decklists with their archetypes
    result = await db.execute(
        select(Decklist.archetype, func.count(Decklist.id).label("count"))
        .join(Event, Decklist.event_id == Event.id)
        .where(Event.format == format)
        .where(Event.date >= cutoff.date())
        .group_by(Decklist.archetype)
    )
    rows = result.all()

    total = sum(row.count for row in rows)
    if total == 0:
        return {}

    archetypes = {}
    for row in rows:
        archetype = row.archetype or "Unknown"
        archetypes[archetype] = {
            "count": row.count,
            "percentage": (row.count / total) * 100,
        }

    return archetypes


async def calculate_cooccurrence(
    db: AsyncSession,
    format: str = "standard",
    days: int = 14,
) -> List[Tuple[str, str, int]]:
    """Calculate card co-occurrence from recent decklists."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    # Get all recent decklists
    result = await db.execute(
        select(Decklist)
        .join(Event, Decklist.event_id == Event.id)
        .where(Event.format == format)
        .where(Event.date >= cutoff.date())
    )
    decklists = result.scalars().all()

    # Count co-occurrences
    cooccurrence = defaultdict(int)

    for decklist in decklists:
        cards = [e.get("card_name") for e in (decklist.main_deck or [])]
        cards = [c for c in cards if c]

        # Count pairs (sorted to avoid duplicates)
        for i, card1 in enumerate(cards):
            for card2 in cards[i + 1 :]:
                pair = tuple(sorted([card1, card2]))
                cooccurrence[pair] += 1

    # Convert to list of tuples
    result = []
    for (card1, card2), count in cooccurrence.items():
        if count >= 3:  # Minimum threshold
            result.append((card1, card2, count))

    return result


async def update_meta_snapshots(
    db: AsyncSession,
    archetypes: Dict[str, Dict[str, Any]],
    format: str = "standard",
) -> int:
    """Update meta snapshot records."""
    # Delete old snapshots for this format (keep only latest)
    await db.execute(
        delete(MetaSnapshot).where(MetaSnapshot.format == format)
    )

    # Get key cards for each archetype
    for archetype, data in archetypes.items():
        # Get most common cards in this archetype
        result = await db.execute(
            select(Decklist)
            .join(Event, Decklist.event_id == Event.id)
            .where(Decklist.archetype == archetype)
            .where(Event.format == format)
            .limit(10)
        )
        decklists = result.scalars().all()

        # Count card occurrences
        card_counts = defaultdict(int)
        for decklist in decklists:
            for entry in decklist.main_deck or []:
                card_name = entry.get("card_name")
                if card_name:
                    card_counts[card_name] += entry.get("quantity", 1)

        # Get top cards
        key_cards = sorted(card_counts.items(), key=lambda x: -x[1])[:10]
        key_card_names = [c[0] for c in key_cards]

        snapshot = MetaSnapshot(
            format=format,
            archetype=archetype,
            meta_percentage=data["percentage"],
            sample_size=data["count"],
            key_cards=key_card_names,
            snapshot_date=datetime.utcnow().date(),
        )
        db.add(snapshot)

    await db.commit()
    return len(archetypes)


async def update_cooccurrence_data(
    db: AsyncSession,
    cooccurrence: List[Tuple[str, str, int]],
    format: str = "standard",
) -> int:
    """Update card co-occurrence records."""
    # Get card IDs
    card_names = set()
    for card1, card2, _ in cooccurrence:
        card_names.add(card1)
        card_names.add(card2)

    result = await db.execute(
        select(Card.id, Card.name).where(Card.name.in_(card_names))
    )
    card_map = {row.name: row.id for row in result.all()}

    # Delete old co-occurrence data
    await db.execute(
        delete(CardCooccurrence).where(CardCooccurrence.format == format)
    )

    # Insert new data
    records = []
    for card1, card2, count in cooccurrence:
        card1_id = card_map.get(card1)
        card2_id = card_map.get(card2)

        records.append({
            "card_a": card1,
            "card_b": card2,
            "card1_id": card1_id,
            "card2_id": card2_id,
            "cooccurrence_count": count,
            "format": format,
        })

    if records:
        # Use simple inserts since we cleared the table
        for record in records:
            cooc = CardCooccurrence(**record)
            db.add(cooc)

    await db.commit()
    return len(records)


async def scrape_mtgtop8(formats: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Main scrape function - downloads tournament data and updates meta snapshots.

    Args:
        formats: List of format names to scrape (e.g., ["standard", "cedh"]).
                 If None, scrapes all configured formats.

    Returns:
        Dict with scrape statistics
    """
    if formats is None:
        formats = list(FORMAT_CONFIG.keys())

    start_time = datetime.utcnow()
    stats = {
        "started_at": start_time.isoformat(),
        "formats_scraped": formats,
        "events_scraped": 0,
        "decklists_scraped": 0,
        "archetypes_updated": 0,
        "cooccurrences_updated": 0,
        "errors": [],
    }

    async with async_session_factory() as db:
        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "Spellbook-MTG-Deckbuilder/1.0 (Educational Project)"
                },
            ) as client:
                # Scrape recent events for each format
                all_events = []
                for format_name in formats:
                    if format_name not in FORMAT_CONFIG:
                        logger.warning(f"Unknown format: {format_name}, skipping")
                        continue

                    config = FORMAT_CONFIG[format_name]
                    events = await scrape_recent_events(
                        client,
                        format_id=config["mtgtop8_id"],
                        format_name=format_name,
                        days=14,
                    )
                    all_events.extend(events)

                stats["events_scraped"] = len(all_events)
                events = all_events

                for event_data in events:
                    try:
                        # Check if event already exists (use first() to handle duplicates)
                        result = await db.execute(
                            select(Event).where(
                                Event.mtgtop8_id == event_data["mtgtop8_id"]
                            ).limit(1)
                        )
                        existing = result.scalars().first()

                        if existing:
                            event = existing
                        else:
                            # Create event record
                            event = Event(
                                mtgtop8_id=event_data["mtgtop8_id"],
                                name=event_data["name"],
                                date=event_data["date"],
                                format=event_data["format"],
                                source_url=event_data["url"],
                            )
                            db.add(event)
                            await db.flush()

                        # Scrape decklists
                        decklists = await scrape_event_decklists(
                            client, event_data["mtgtop8_id"]
                        )
                        logger.info(f"Found {len(decklists)} decklists for event {event_data['mtgtop8_id']}")

                        for deck_data in decklists:
                            try:
                                # Check if decklist exists (use first() to handle duplicates)
                                result = await db.execute(
                                    select(Decklist).where(
                                        Decklist.mtgtop8_deck_id == deck_data["mtgtop8_deck_id"]
                                    ).limit(1)
                                )
                                if result.scalars().first():
                                    continue

                                # Scrape cards
                                cards = await scrape_decklist_cards(
                                    client,
                                    event_data["mtgtop8_id"],
                                    deck_data["mtgtop8_deck_id"],
                                )

                                # Create decklist record
                                decklist = Decklist(
                                    event_id=event.id,
                                    mtgtop8_deck_id=deck_data["mtgtop8_deck_id"],
                                    archetype=deck_data["archetype"],
                                    placement=deck_data["placement"],
                                    main_deck=cards["main_deck"],
                                    sideboard=cards["sideboard"],
                                    source_url=deck_data["url"],
                                )
                                db.add(decklist)
                                stats["decklists_scraped"] += 1

                            except Exception as e:
                                logger.warning(f"Error scraping decklist: {e}")
                                stats["errors"].append(f"Decklist error: {e}")

                        await db.commit()

                    except Exception as e:
                        logger.warning(f"Error scraping event {event_data.get('mtgtop8_id')}: {e}")
                        stats["errors"].append(f"Event error: {e}")
                        await db.rollback()  # Rollback to recover from error

            # Calculate and update meta percentages for each format
            total_archetypes = 0
            total_cooccurrences = 0
            for format_name in formats:
                archetypes = await calculate_meta_percentages(db, format=format_name)
                total_archetypes += await update_meta_snapshots(db, archetypes, format=format_name)

                cooccurrence = await calculate_cooccurrence(db, format=format_name)
                total_cooccurrences += await update_cooccurrence_data(db, cooccurrence, format=format_name)

            stats["archetypes_updated"] = total_archetypes
            stats["cooccurrences_updated"] = total_cooccurrences

        except Exception as e:
            logger.error(f"mtgtop8 scrape failed: {e}")
            stats["errors"].append(str(e))
            raise

    # Compute archetype templates from the updated decklists
    try:
        from app.jobs.compute_archetype_templates import compute_archetype_templates
        template_stats = await compute_archetype_templates()
        stats["archetype_templates_updated"] = template_stats.get("templates_saved", 0)
        logger.info(f"Updated {stats['archetype_templates_updated']} archetype templates")
    except Exception as e:
        logger.warning(f"Failed to compute archetype templates: {e}")
        stats["errors"].append(f"Archetype templates error: {e}")

    stats["completed_at"] = datetime.utcnow().isoformat()
    return stats


if __name__ == "__main__":
    # Allow running directly for testing
    asyncio.run(scrape_mtgtop8())
