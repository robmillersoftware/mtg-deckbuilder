"""semantic_search: shortlist merge (vector, full text, popular), Jev re-rank, fallbacks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services import card_service as cs
from app.services.card_service import CardService
from tests.jev_fake import FakeJev

REAL_SCORE_SEARCH = getattr(cs, "score_search", None)  # the svc fixture replaces cs.score_search


def card(name):
    return SimpleNamespace(name=name, mana_cost="{R}", type_line="Instant", oracle_text="Deal 2.")


@pytest.fixture
def svc(monkeypatch):
    s = CardService.__new__(CardService)
    s.db = MagicMock(rollback=AsyncMock())
    s.vec, s.fts, s.pop, s.calls = [], [], [], []
    s.embedding = None
    emb = SimpleNamespace(get_query_embedding=AsyncMock(side_effect=lambda q: s.embedding))
    monkeypatch.setattr(cs, "get_embedding_service", lambda: emb)

    async def vector(embedding, **kw):
        s.calls.append("vec")
        if isinstance(s.vec, Exception):
            raise s.vec
        return [SimpleNamespace(name=n) for n in s.vec]

    async def fts(query, format=None, standard_only=True, colors=None, limit=60):
        s.calls.append("fts")
        return list(s.fts)

    async def popular(format=None, standard_only=True, colors=None, limit=60):
        s.calls.append("pop")
        return list(s.pop)

    async def by_names(names):
        return {n.lower(): card(n) for n in names}

    s._vector_search, s.text_search_names, s.popular_card_names = vector, fts, popular
    s.get_cards_by_names = by_names
    s.scores = None

    async def score(query, cards, client=None):
        s.scored = [c.name for c in cards]
        return s.scores
    monkeypatch.setattr(cs, "score_search", score)
    return s


def names(cards):
    return [c.name for c in cards]


async def test_shortlist_merges_in_priority_order_and_dedupes(svc):
    svc.embedding = [0.1]
    svc.vec, svc.fts, svc.pop = ["A", "B"], ["B", "C"], ["C", "D"]
    assert names(await svc.semantic_search("burn", limit=10)) == ["A", "B", "C", "D"]


async def test_without_openai_skips_vector_search(svc):
    svc.vec, svc.fts = ["X"], ["A"]
    assert names(await svc.semantic_search("burn")) == ["A"]
    assert "vec" not in svc.calls


async def test_vector_failure_rolls_back_and_continues(svc):
    svc.embedding = [0.1]
    svc.vec, svc.fts = RuntimeError("no pgvector"), ["A"]
    assert names(await svc.semantic_search("burn")) == ["A"]
    svc.db.rollback.assert_awaited_once()


async def test_shortlist_capped_skips_lower_sources(svc, monkeypatch):
    monkeypatch.setattr(cs, "SHORTLIST_SIZE", 3)
    svc.fts, svc.pop = ["A", "B", "C", "D", "E"], ["F"]
    await svc.semantic_search("burn", limit=10)
    assert svc.scored == ["A", "B", "C"]
    assert "pop" not in svc.calls


async def test_rerank_orders_by_probability_and_cuts(svc):
    svc.fts = ["A", "B", "C"]
    svc.scores = {"A": 0.2, "B": 0.9, "C": 0.6}
    assert names(await svc.semantic_search("burn")) == ["B", "C"]


async def test_limit_applies_after_rerank(svc):
    svc.fts = ["A", "B", "C"]
    svc.scores = {"A": 0.6, "B": 0.9, "C": 0.7}
    assert names(await svc.semantic_search("burn", limit=1)) == ["B"]


async def test_jev_unavailable_returns_shortlist_order(svc):
    svc.fts, svc.pop = ["A", "B"], ["C"]
    svc.scores = None
    assert names(await svc.semantic_search("burn", limit=2)) == ["A", "B"]


async def test_empty_shortlist_makes_no_jev_call(svc, monkeypatch):
    client = FakeJev()

    async def score(query, cards):
        return await REAL_SCORE_SEARCH(query, cards, client=client)
    monkeypatch.setattr(cs, "score_search", score)
    assert await svc.semantic_search("?!") == []
    assert client.calls == []


@pytest.mark.parametrize("scores,expected", [(None, ["Front", "Other"]), ({"Front": 0.9, "Other": 0.6}, ["Front", "Other"])])
async def test_double_faced_names_resolve_once(svc, scores, expected):
    svc.fts = ["Front", "Front // Back", "Other"]
    front = card("Front")

    async def by_names(names):
        return {n.lower(): front if n.startswith("Front") else card(n) for n in names}
    svc.get_cards_by_names = by_names
    svc.scores = scores
    assert names(await svc.semantic_search("burn", limit=2)) == expected
    assert svc.scored == ["Front", "Other"]


class TestScoreSearch:
    async def test_one_noul_per_card(self):
        client = FakeJev(lambda state, qs: {"match": 0.7 if state["card"]["name"] == "A" else 0.1})
        assert await cs.score_search("burn", [card("A"), card("B")], client=client) == {"A": 0.7, "B": 0.1}
        assert len(client.calls) == 2
        state, qs, _ = client.calls[0]
        assert state["query"] == "burn" and set(qs) == {"match"}

    async def test_any_failure_returns_none(self):
        client = FakeJev(fail=lambda state: state["card"]["name"] == "B")
        assert await cs.score_search("burn", [card("A"), card("B")], client=client) is None

    async def test_no_key_returns_none(self):
        assert await cs.score_search("burn", [card("A")]) is None
