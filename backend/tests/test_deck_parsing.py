"""parse_deck_request with Jev: shape, own-vs-opponent colors, wanted cards, fallback."""

import pytest

from app.services.ai import deck_parsing
from tests.jev_fake import FakeJev

SHEOLDRED = "Sheoldred, the Apocalypse"


@pytest.fixture
def names(monkeypatch):
    found = []

    async def extract(prompt, db, format="standard"):
        return list(found)
    monkeypatch.setattr(deck_parsing, "extract_card_names_from_prompt", extract)

    async def fallback(prompt, db):
        return {"fallback": True}
    monkeypatch.setattr(deck_parsing, "fallback_parse", fallback)
    return found


async def test_shape_colors_and_wanted_cards(names):
    names.extend(["Monastery Swiftspear", SHEOLDRED])
    client = FakeJev(lambda state, qs: {"archetype": "aggro", "color:R": 0.9, "colors_specified": 0.9,
                                        "wants:0": 0.9, "wants:1": 0.1})
    prompt = f"Mono-red aggro with Monastery Swiftspear that beats {SHEOLDRED}"
    out = await deck_parsing.parse_deck_request(prompt, db=None, client=client)
    assert out == {"archetype": "aggro", "colors": ["R"], "colors_specified": True,
                   "strategy": prompt, "specific_cards": ["Monastery Swiftspear"]}
    state, qs, _ = client.calls[0]
    assert state == {"request": prompt, "cards": ["Monastery Swiftspear", SHEOLDRED]}
    assert set(qs["archetype"].criteria) == {"aggro", "control", "midrange", "combo", "tempo"}


async def test_opponent_color_is_not_specified(names):
    client = FakeJev(lambda state, qs: {"archetype": "control", "color:R": 0.1, "colors_specified": 0.2})
    out = await deck_parsing.parse_deck_request("beat mono-red", db=None, client=client)
    assert out["colors"] == [] and out["colors_specified"] is False


async def test_unstated_colors_are_dropped(names):
    client = FakeJev(lambda state, qs: {"color:W": 0.8, "color:U": 0.7, "colors_specified": 0.1})
    out = await deck_parsing.parse_deck_request("A deck that beats mono-red", db=None, client=client)
    assert out["colors"] == [] and out["colors_specified"] is False


async def test_colors_specified_needs_a_color(names):
    client = FakeJev(lambda state, qs: {"colors_specified": 0.9})
    out = await deck_parsing.parse_deck_request("aggro deck", db=None, client=client)
    assert out["colors"] == [] and out["colors_specified"] is False


async def test_failure_falls_back(names):
    client = FakeJev(fail=lambda state: True)
    assert await deck_parsing.parse_deck_request("x", db=None, client=client) == {"fallback": True}


async def test_partial_answer_falls_back(names):
    client = FakeJev(lambda state, qs: {"archetype": "aggro"}, fill=False)
    assert await deck_parsing.parse_deck_request("x", db=None, client=client) == {"fallback": True}


async def test_no_key_falls_back_without_lookup(names, monkeypatch):
    async def boom(prompt, db, format="standard"):
        raise AssertionError("card lookup should not run without Jev")
    monkeypatch.setattr(deck_parsing, "extract_card_names_from_prompt", boom)
    assert await deck_parsing.parse_deck_request("x", db=None) == {"fallback": True}
