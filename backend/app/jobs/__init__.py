"""
Scheduled jobs for data synchronization.
"""

from app.jobs.scryfall_sync import sync_scryfall_cards
from app.jobs.mtgtop8_scrape import scrape_mtgtop8
from app.jobs.classify_card_roles import classify_all_cards, get_classification_stats
from app.jobs.compute_archetype_templates import (
    compute_archetype_templates,
    get_archetype_template,
    get_all_archetype_templates,
)

__all__ = [
    "sync_scryfall_cards",
    "scrape_mtgtop8",
    "classify_all_cards",
    "get_classification_stats",
    "compute_archetype_templates",
    "get_archetype_template",
    "get_all_archetype_templates",
]
