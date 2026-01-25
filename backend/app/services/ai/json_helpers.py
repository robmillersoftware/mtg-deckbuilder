"""JSON repair and extraction utilities for AI responses."""

import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def repair_json(json_str: str) -> str:
    """Attempt to repair common JSON formatting issues."""
    repaired = json_str

    # Fix trailing commas before ] or }
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

    # Fix missing commas between array elements
    repaired = re.sub(r'"\s*\n\s*\{', '",\n{', repaired)
    repaired = re.sub(r'}\s*\n\s*\{', '},\n{', repaired)

    # Fix single quotes used instead of double quotes
    repaired = re.sub(r"'(\w+)':", r'"\1":', repaired)

    return repaired


def extract_deck_from_malformed_json(json_str: str) -> Optional[Dict[str, Any]]:
    """Last resort: extract deck data from malformed JSON using regex."""
    try:
        deck_data = {
            "name": "Generated Deck",
            "strategy_summary": "",
            "main_deck": [],
            "sideboard": []
        }

        # Try to extract card entries using regex
        # Pattern: "card_name": "Card Name", "quantity": N
        card_pattern = r'"card_name"\s*:\s*"([^"]+)"\s*,\s*"quantity"\s*:\s*(\d+)'
        matches = re.findall(card_pattern, json_str)

        if matches:
            for card_name, quantity in matches:
                deck_data["main_deck"].append({
                    "card_name": card_name,
                    "quantity": int(quantity)
                })
            logger.info(f"Extracted {len(matches)} cards from malformed JSON")
            return deck_data

        # Alternative pattern: "Card Name": N or N Card Name
        alt_pattern = r'(\d+)\s*[x×]?\s*([A-Z][^",\n]+)'
        matches = re.findall(alt_pattern, json_str)

        if matches:
            for quantity, card_name in matches:
                deck_data["main_deck"].append({
                    "card_name": card_name.strip(),
                    "quantity": int(quantity)
                })
            logger.info(f"Extracted {len(matches)} cards using alt pattern")
            return deck_data

        return None

    except Exception as e:
        logger.error(f"Failed to extract deck from malformed JSON: {e}")
        return None
