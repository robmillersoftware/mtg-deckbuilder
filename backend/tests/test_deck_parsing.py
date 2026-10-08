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
    """Fake session for extract_card_names_from_prompt: every phrase matches a card of the same name."""

    def __init__(self, match):
        self.match, self.calls, self.queries = match, 0, []

    async def execute(self, query):
        self.calls += 1
        self.queries.append(query)
        params = query.compile().params
        phrases = next(v for v in params.values() if isinstance(v, list))
        rows = [(p.title(), p, p, p) for p in phrases] if self.match else []
        return type("R", (), {"all": lambda self: rows})()


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
    assert db.calls == 1  # one query for all phrases

    db = CountingDb(match=False)
    await deck_parsing.parse_deck_request(LONG_PROMPT, db=db, client=FakeJev())
    assert db.calls == 1


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


async def test_guild_name_adds_its_colors_only_when_wanted(names):
    answers = {"color:R": 0.6, "color:W": 0.3, "colors_specified": 0.9}
    client = FakeJev(lambda state, qs: answers)
    out = await deck_parsing.parse_deck_request("Build me a Boros aggro deck", db=None, client=client)
    assert out["colors"] == ["W", "R"] and out["colors_specified"] is True
    answers = {"color:R": 0.2, "color:W": 0.1, "colors_specified": 0.9}
    out = await deck_parsing.parse_deck_request("how do I beat Boros?", db=None, client=client)
    assert out["colors"] == [] and out["colors_specified"] is False


class RowsDb:
    """Returns fixed (name, lower(name), before-comma, front-face) rows."""

    def __init__(self, rows):
        self.rows, self.queries = rows, []

    async def execute(self, query):
        self.queries.append(query)
        rows = self.rows
        return type("R", (), {"all": lambda _: rows})()


def _row(name):
    low = name.lower()
    return (name, low, low.split(", ")[0], low.split(" // ")[0])


async def test_name_lookup_never_uses_substring_matching():
    db = RowsDb([])
    await deck_parsing.extract_card_names_from_prompt("Build me a mono-red aggro deck for Standard", db)
    sql = str(db.queries[0].compile(compile_kwargs={"literal_binds": True})).lower()
    assert " like " not in sql
    assert "split_part" in sql


async def test_name_lookup_matches_whole_name_comma_head_and_front_face():
    rows = [_row("Atraxa, Grand Unifier"), _row("Fable of the Mirror-Breaker // Reflection of Kiki-Jiki"),
            _row("Lightning Bolt")]
    db = RowsDb(rows)
    names = await deck_parsing.extract_card_names_from_prompt(
        "Deck around Atraxa with Lightning Bolt and Fable of the Mirror-Breaker", db)
    assert names == ["Atraxa, Grand Unifier", "Lightning Bolt",
                     "Fable of the Mirror-Breaker // Reflection of Kiki-Jiki"]


async def test_exact_name_preferred_over_comma_head():
    rows = [_row("Chandra, Flameshaper"), _row("Chandra")]
    names = await deck_parsing.extract_card_names_from_prompt("Play Chandra", RowsDb(rows))
    assert names == ["Chandra"]


def test_candidate_phrases_lowercased_deduped_and_capped():
    phrases = deck_parsing._candidate_phrases("Lightning Bolt, Lightning Bolt! " + " ".join(["x"] * 100))
    assert phrases[0] == "lightning"
    assert "lightning bolt" in phrases
    assert len(phrases) == len(set(phrases))
    assert all(p == p.lower() for p in phrases)


async def test_word_inside_a_matched_longer_name_is_not_a_separate_card():
    rows = [_row("Lightning Strike"), _row("Lightning, Army of One"), _row("Shock")]
    names = await deck_parsing.extract_card_names_from_prompt(
        "Red deck with Lightning Strike and Shock", RowsDb(rows))
    assert names == ["Lightning Strike", "Shock"]
