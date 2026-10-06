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

    async def fallback(prompt, db, names=None):
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


class CountingDb:
    """Mock db: every execute() counts; `match` decides whether a lookup finds a card."""

    def __init__(self, match):
        self.match, self.calls = match, 0

    async def execute(self, query):
        self.calls += 1
        n, hit = self.calls, self.match
        return type("R", (), {"scalar_one_or_none": lambda self: f"Card {n}" if hit else None})()


LONG_PROMPT = " ".join(f"Word{i}" for i in range(200))


async def test_long_prompt_bounds_queries_and_names(monkeypatch):
    async def fallback(prompt, db, names=None):
        return {"fallback": True}
    monkeypatch.setattr(deck_parsing, "fallback_parse", fallback)
    db = CountingDb(match=True)
    client = FakeJev()
    await deck_parsing.parse_deck_request(LONG_PROMPT, db=db, client=client)
    state, qs, _ = client.calls[0]
    assert len(state["cards"]) == deck_parsing.MAX_CARD_NAMES == 10
    assert len([q for q in qs if q.startswith("wants:")]) == 10
    assert db.calls <= 4 * deck_parsing.MAX_PROMPT_WORDS

    db = CountingDb(match=False)  # no matches: still bounded by the word cap
    await deck_parsing.parse_deck_request(LONG_PROMPT, db=db, client=FakeJev())
    assert db.calls <= 4 * deck_parsing.MAX_PROMPT_WORDS


async def test_failure_looks_up_names_once(monkeypatch):
    db = CountingDb(match=True)
    out = await deck_parsing.parse_deck_request("Tezzeret rocks", db=db,
                                                client=FakeJev(fail=lambda state: True))
    assert out["specific_cards"] and out["archetype"]
    first = db.calls
    assert first > 0
    # fallback must reuse the names: a second full lookup would double the count
    db2 = CountingDb(match=True)
    await deck_parsing.extract_card_names_from_prompt("Tezzeret rocks", db2)
    assert first == db2.calls
