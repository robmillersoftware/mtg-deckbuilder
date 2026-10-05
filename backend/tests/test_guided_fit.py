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
        pools.setdefault("calls", []).append(list(roles))
        if "lists" in pools:
            return {r: [cd(n) for n in pools["lists"][r]] for r in roles}
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


async def test_overlapping_roles_dedupe_after_ranking_without_starving(analyzer, monkeypatch):
    a, pools = analyzer
    pools["lists"] = {"r1": ["A", "B", "C", "D"], "r2": ["A", "B", "E", "F"]}

    async def score(identity, keys, cands, client=None):
        s = {"A": .9, "B": .8, "C": .7, "D": .1, "E": .6, "F": .5}
        return {c["name"]: FitScore(plan_fit=s[c["name"]], synergy=None, anti_synergy=0.0)
                for c in cands}
    monkeypatch.setattr(guided_builder, "score_fit", score)
    out = await a.suggest_cards_for_strategy("s", [], ["r1", "r2"], [], cards_per_role=2,
                                             identity=DeckIdentity(tags=["x"]))
    assert pools["calls"] == [["r1"], ["r2"]]
    assert [c["card_name"] for c in out["r1"]] == ["A", "B"]
    assert [c["card_name"] for c in out["r2"]] == ["E", "F"]


async def test_fallback_with_overlapping_roles_has_no_duplicates(analyzer, monkeypatch):
    a, pools = analyzer
    pools["lists"] = {"r1": ["A", "B", "C"], "r2": ["a", "B", "D", "E"]}

    async def score(identity, keys, cands, client=None):
        return {}
    monkeypatch.setattr(guided_builder, "score_fit", score)
    out = await a.suggest_cards_for_strategy("s", [], ["r1", "r2"], [], cards_per_role=2,
                                             identity=DeckIdentity(tags=["x"]))
    assert [c["card_name"] for c in out["r1"]] == ["A", "B"]
    assert [c["card_name"] for c in out["r2"]] == ["D", "E"]


# --- ChatService._fit_identity caching ---

from app.models.conversation import Conversation
from app.services import deck_fit
from app.services.chat_service import ChatService


@pytest.fixture
def fitenv(monkeypatch):
    svc = ChatService.__new__(ChatService)
    svc.db = MagicMock()
    state = {"n": 0, "land": 0}

    async def load(db, names):
        return [{"name": n, "type_line": "Creature", "oracle_text": "", "mana_cost": ""}
                for n in names]

    async def infer(cards, request_text, overrides=None, client=None):
        state["n"] += 1
        return DeckIdentity(tags=["t"], request_text=request_text)
    monkeypatch.setattr(deck_fit, "load_payloads", load)
    monkeypatch.setattr(deck_fit, "infer_identity", infer)
    return svc, Conversation(), state


def deck(n):
    return [{"card_name": f"C{i}"} for i in range(n)]


async def test_fit_identity_cache_hit_same_bucket_and_text(fitenv):
    svc, conv, st = fitenv
    await svc._fit_identity(conv, deck(2), "tokens")
    await svc._fit_identity(conv, deck(3), "tokens")
    assert st["n"] == 1


async def test_fit_identity_bucket_crossing_reinfers(fitenv):
    svc, conv, st = fitenv
    await svc._fit_identity(conv, deck(2), "tokens")
    await svc._fit_identity(conv, deck(deck_fit.RE_INFER_AT[0]), "tokens")
    assert st["n"] == 2


async def test_fit_identity_new_text_reinfers_empty_text_reuses(fitenv):
    svc, conv, st = fitenv
    await svc._fit_identity(conv, deck(2), "tokens")
    await svc._fit_identity(conv, deck(2), "aggro")
    assert st["n"] == 2
    ident = await svc._fit_identity(conv, deck(2), "")
    assert st["n"] == 2 and ident.request_text == "aggro"


async def test_fit_identity_malformed_cache_reinfers(fitenv):
    svc, conv, st = fitenv
    conv.update_context(fit_identity={"bucket": 0, "request_text": "tokens",
                                      "identity": {"tags": 5, "bogus": object}})
    ident = await svc._fit_identity(conv, deck(2), "tokens")
    assert st["n"] == 1 and ident is not None
