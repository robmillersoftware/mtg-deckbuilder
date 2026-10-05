"""suggest_cards_for_strategy re-ranks a larger pool by Jev fit, or falls back unchanged."""

from unittest.mock import MagicMock

import pytest

from app.services import guided_builder
from app.services.deck_fit import DeckIdentity, FitScore
from app.services.guided_builder import DeckAnalyzer


def cd(name):
    return {"card_name": name, "type_line": "Creature", "oracle_text": "", "mana_cost": "{B}"}


@pytest.fixture
def analyzer(monkeypatch):
    a = DeckAnalyzer.__new__(DeckAnalyzer)
    a.db = MagicMock()
    pools = {}

    async def collect(strategy, colors, roles, existing_cards, format, cards_per_role):
        pools["size"] = cards_per_role
        return {"threats": [cd(f"T{i}") for i in range(cards_per_role)]}

    async def freq(names, format="standard"):
        return {}

    async def load(db, names):
        return []

    a._collect_role_candidates = collect
    a._rank_cards_by_tournament_frequency = freq
    monkeypatch.setattr(guided_builder, "load_payloads", load)
    return a, pools


async def test_without_identity_behaves_as_before(analyzer):
    a, pools = analyzer
    out = await a.suggest_cards_for_strategy("s", [], ["threats"], [], cards_per_role=2)
    assert pools["size"] == 2
    assert [c["card_name"] for c in out["threats"]] == ["T0", "T1"]
    assert "fit" not in out["threats"][0]


async def test_with_identity_reranks_triple_pool(analyzer, monkeypatch):
    a, pools = analyzer

    async def score(identity, keys, cands, client=None):
        return {c["name"]: FitScore(plan_fit=int(c["name"][1:]) / 10, synergy=None, anti_synergy=0.0)
                for c in cands}
    monkeypatch.setattr(guided_builder, "score_fit", score)
    out = await a.suggest_cards_for_strategy("s", [], ["threats"], [], cards_per_role=2,
                                             identity=DeckIdentity(tags=["tokens"]))
    assert pools["size"] == 6
    assert [c["card_name"] for c in out["threats"]] == ["T5", "T4"]
    assert out["threats"][0]["fit"] == {"plan_fit": 0.5, "synergy": None}


async def test_fit_unavailable_falls_back_to_retrieval_order(analyzer, monkeypatch):
    a, pools = analyzer

    async def score(identity, keys, cands, client=None):
        return {}
    monkeypatch.setattr(guided_builder, "score_fit", score)
    out = await a.suggest_cards_for_strategy("s", [], ["threats"], [], cards_per_role=2,
                                             identity=DeckIdentity(tags=["tokens"]))
    assert [c["card_name"] for c in out["threats"]] == ["T0", "T1"]
    assert "fit" not in out["threats"][0]


async def test_role_emptied_by_anti_synergy_is_dropped(analyzer, monkeypatch):
    a, _ = analyzer

    async def score(identity, keys, cands, client=None):
        return {c["name"]: FitScore(plan_fit=1.0, synergy=None, anti_synergy=0.9) for c in cands}
    monkeypatch.setattr(guided_builder, "score_fit", score)
    out = await a.suggest_cards_for_strategy("s", [], ["threats"], [], cards_per_role=2,
                                             identity=DeckIdentity(tags=["tokens"]))
    assert out == {}
