"""
Scryfall Daily Sync Job
Schedule: 2:00 AM UTC daily

Downloads bulk card data from Scryfall API and updates the card database.
Also computes and updates card embeddings for semantic search.
"""

import asyncio
import logging
import gzip
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from io import BytesIO

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.db.session import async_session_factory
from app.models.card import Card
from app.core.config import settings

logger = logging.getLogger(__name__)

SCRYFALL_BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
STANDARD_LEGAL_SETS = set()  # Will be populated from Scryfall


async def get_bulk_data_url() -> Optional[str]:
    """Get the URL for the default cards bulk data file."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(SCRYFALL_BULK_DATA_URL)
        response.raise_for_status()
        data = response.json()

        for item in data.get("data", []):
            if item.get("type") == "default_cards":
                return item.get("download_uri")

    return None


async def fetch_standard_sets() -> set:
    """Fetch the list of sets legal in Standard."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get("https://api.scryfall.com/sets")
        response.raise_for_status()
        data = response.json()

        standard_sets = set()
        for set_data in data.get("data", []):
            # Check if set is Standard legal
            if set_data.get("set_type") in ["expansion", "core"]:
                # Check release date - Standard is roughly last 2 years
                release_date = set_data.get("released_at")
                if release_date:
                    release = datetime.fromisoformat(release_date)
                    cutoff = datetime.now() - timedelta(days=730)
                    if release >= cutoff:
                        standard_sets.add(set_data.get("code", "").lower())

        return standard_sets


async def download_bulk_cards(url: str) -> List[Dict[str, Any]]:
    """Download and parse the bulk card data."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        logger.info(f"Downloading bulk card data from {url}")
        response = await client.get(url)
        response.raise_for_status()

        # The file may be gzipped
        content = response.content
        if url.endswith(".gz"):
            content = gzip.decompress(content)

        cards = json.loads(content)
        logger.info(f"Downloaded {len(cards)} cards")
        return cards


def extract_card_data(card: Dict[str, Any], standard_sets: set) -> Optional[Dict[str, Any]]:
    """Extract relevant card data from Scryfall format."""
    # Skip tokens, emblems, etc.
    if card.get("layout") in ["token", "double_faced_token", "emblem", "art_series"]:
        return None

    # Skip digital-only cards unless they're also in paper
    if card.get("digital") and not card.get("released_at"):
        return None

    scryfall_id = card.get("id")
    if not scryfall_id:
        return None

    # Get image URIs
    image_uris = card.get("image_uris", {})
    if not image_uris and card.get("card_faces"):
        # Double-faced card - use front face
        image_uris = card["card_faces"][0].get("image_uris", {})

    # Determine Standard legality
    legalities = card.get("legalities", {})
    is_standard = legalities.get("standard") == "legal"

    # Get prices
    prices = card.get("prices", {})

    return {
        "scryfall_id": scryfall_id,
        "oracle_id": card.get("oracle_id"),
        "name": card.get("name"),
        "mana_cost": card.get("mana_cost"),
        "cmc": card.get("cmc"),
        "type_line": card.get("type_line"),
        "oracle_text": card.get("oracle_text"),
        "power": card.get("power"),
        "toughness": card.get("toughness"),
        "colors": card.get("colors") or [],
        "color_identity": card.get("color_identity") or [],
        "keywords": card.get("keywords") or [],
        "set_code": card.get("set"),
        "set_name": card.get("set_name"),
        "collector_number": card.get("collector_number"),
        "rarity": card.get("rarity"),
        "image_uri": image_uris.get("normal"),
        "image_uri_small": image_uris.get("small"),
        "image_uri_art_crop": image_uris.get("art_crop"),
        "price_usd": float(prices.get("usd") or 0),
        "price_usd_foil": float(prices.get("usd_foil") or 0) if prices.get("usd_foil") else None,
        "is_standard_legal": is_standard,
        "legalities": legalities,
        "scryfall_uri": card.get("scryfall_uri"),
        "updated_at": datetime.utcnow(),
    }


async def upsert_cards(db: AsyncSession, cards_data: List[Dict[str, Any]]) -> int:
    """Upsert cards into the database."""
    if not cards_data:
        return 0

    # Use PostgreSQL upsert (ON CONFLICT)
    stmt = insert(Card).values(cards_data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["scryfall_id"],
        set_={
            "name": stmt.excluded.name,
            "mana_cost": stmt.excluded.mana_cost,
            "cmc": stmt.excluded.cmc,
            "type_line": stmt.excluded.type_line,
            "oracle_text": stmt.excluded.oracle_text,
            "power": stmt.excluded.power,
            "toughness": stmt.excluded.toughness,
            "colors": stmt.excluded.colors,
            "color_identity": stmt.excluded.color_identity,
            "keywords": stmt.excluded.keywords,
            "set_code": stmt.excluded.set_code,
            "set_name": stmt.excluded.set_name,
            "collector_number": stmt.excluded.collector_number,
            "rarity": stmt.excluded.rarity,
            "image_uri": stmt.excluded.image_uri,
            "image_uri_small": stmt.excluded.image_uri_small,
            "image_uri_art_crop": stmt.excluded.image_uri_art_crop,
            "price_usd": stmt.excluded.price_usd,
            "price_usd_foil": stmt.excluded.price_usd_foil,
            "is_standard_legal": stmt.excluded.is_standard_legal,
            "legalities": stmt.excluded.legalities,
            "scryfall_uri": stmt.excluded.scryfall_uri,
            "updated_at": stmt.excluded.updated_at,
        },
    )

    await db.execute(stmt)
    return len(cards_data)


async def compute_card_embeddings(db: AsyncSession, batch_size: int = 50) -> int:
    """
    Compute embeddings for cards that don't have them.
    Uses OpenAI embeddings API for semantic search capability.
    """
    from app.services.embedding_service import get_embedding_service

    embedding_service = get_embedding_service()

    # Find Standard-legal cards without embeddings (prioritize these)
    result = await db.execute(
        select(Card)
        .where(Card.embedding.is_(None))
        .where(Card.is_standard_legal == True)
        .limit(batch_size)
    )
    cards = list(result.scalars().all())

    if not cards:
        logger.info("All Standard cards have embeddings")
        return 0

    logger.info(f"Computing embeddings for {len(cards)} cards")

    # Build text for each card
    texts = []
    for card in cards:
        text = embedding_service._build_card_text(
            name=card.name,
            type_line=card.type_line,
            oracle_text=card.oracle_text,
            keywords=card.keywords,
        )
        texts.append(text)

    # Get embeddings in batch
    embeddings = await embedding_service.get_embeddings_batch(texts, batch_size=batch_size)

    updated = 0
    for card, embedding in zip(cards, embeddings):
        if embedding:
            card.embedding = embedding
            updated += 1

    await db.commit()
    logger.info(f"Updated embeddings for {updated} cards")
    return updated


async def sync_scryfall_cards() -> Dict[str, Any]:
    """
    Main sync function - downloads and updates all card data from Scryfall.

    Returns:
        Dict with sync statistics
    """
    start_time = datetime.utcnow()  # Use naive datetime for DB compatibility
    stats = {
        "started_at": start_time.isoformat(),
        "cards_processed": 0,
        "cards_updated": 0,
        "errors": [],
    }

    async with async_session_factory() as db:
        try:
            # Get bulk data URL
            bulk_url = await get_bulk_data_url()
            if not bulk_url:
                raise Exception("Could not find bulk data URL")

            # Fetch Standard-legal sets
            standard_sets = await fetch_standard_sets()
            logger.info(f"Found {len(standard_sets)} Standard-legal sets")

            # Download cards
            all_cards = await download_bulk_cards(bulk_url)
            stats["cards_processed"] = len(all_cards)

            # Process in batches
            batch_size = 1000
            total_updated = 0

            for i in range(0, len(all_cards), batch_size):
                batch = all_cards[i : i + batch_size]
                cards_data = []

                for card in batch:
                    card_data = extract_card_data(card, standard_sets)
                    if card_data:
                        cards_data.append(card_data)

                if cards_data:
                    updated = await upsert_cards(db, cards_data)
                    total_updated += updated

                await db.commit()
                logger.info(f"Processed batch {i // batch_size + 1}, total updated: {total_updated}")

            stats["cards_updated"] = total_updated

            # Update embeddings (limited batch)
            embeddings_updated = await compute_card_embeddings(db)
            stats["embeddings_updated"] = embeddings_updated

        except Exception as e:
            logger.error(f"Scryfall sync failed: {e}")
            stats["errors"].append(str(e))
            raise

    stats["completed_at"] = datetime.utcnow().isoformat()
    return stats


if __name__ == "__main__":
    # Allow running directly for testing
    asyncio.run(sync_scryfall_cards())
