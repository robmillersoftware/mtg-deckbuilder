"""Scryfall bulk-data parsing: legacy JSON array and current gzipped JSONL."""

import gzip
import json

from app.jobs.scryfall_sync import parse_bulk_cards, pick_bulk_download_uri

CARDS = [{"name": "Murder"}, {"name": "Consider"}]


def test_parses_gzipped_jsonl():
    body = gzip.compress("\n".join(json.dumps(c) for c in CARDS).encode())
    assert parse_bulk_cards(body) == CARDS


def test_parses_plain_json_array():
    assert parse_bulk_cards(json.dumps(CARDS).encode()) == CARDS


def test_parses_already_decompressed_jsonl_with_blank_lines():
    body = ("\n".join(json.dumps(c) for c in CARDS) + "\n\n").encode()
    assert parse_bulk_cards(body) == CARDS


def test_picks_jsonl_uri_when_download_uri_missing():
    item = {"type": "default_cards", "jsonl_download_uri": "https://x/default.jsonl.gz"}
    assert pick_bulk_download_uri(item) == "https://x/default.jsonl.gz"


def test_prefers_download_uri_when_present():
    item = {"download_uri": "https://x/default.json", "jsonl_download_uri": "https://x/d.jsonl.gz"}
    assert pick_bulk_download_uri(item) == "https://x/default.json"
