"""Tests for deck_fit: identity overrides, ranking, flagging, and Jev calls (stubbed)."""

import asyncio
from types import SimpleNamespace

import pytest

from app.services import deck_fit
from app.services.deck_fit import DeckIdentity, FitScore, IdentityOverrides


def fs(plan, syn=None, anti=0.0):
    return FitScore(plan_fit=plan, synergy=syn, anti_synergy=anti)


class TestCardPayload:
    def test_from_dict_and_object(self):
        d = deck_fit.card_payload({"card_name": "Murder", "mana_cost": "{1}{B}{B}",
                                   "type_line": "Instant", "oracle_text": "Destroy target creature."})
        o = deck_fit.card_payload(SimpleNamespace(name="Murder", mana_cost="{1}{B}{B}",
                                                  type_line="Instant", oracle_text="Destroy target creature."))
        assert d == o == {"name": "Murder", "mana_cost": "{1}{B}{B}",
                          "type_line": "Instant", "oracle_text": "Destroy target creature."}

    def test_truncates_long_oracle_text(self):
        p = deck_fit.card_payload({"card_name": "Wall", "oracle_text": "x" * 5000})
        assert len(p["oracle_text"]) == deck_fit.ORACLE_CHAR_LIMIT

    def test_none_fields_become_empty(self):
        p = deck_fit.card_payload({"card_name": "Blank"})
        assert p["type_line"] == "" and p["oracle_text"] == "" and p["mana_cost"] == ""

    def test_is_land(self):
        assert deck_fit.is_land({"type_line": "Basic Land — Swamp"})
        assert not deck_fit.is_land({"type_line": "Artifact Creature — Golem"})


class TestApplyOverrides:
    def test_tags_and_pins(self):
        ident = DeckIdentity(
            tags=["graveyard", "control"], key_cards=["A", "B"],
            overrides=IdentityOverrides(tags_on=["sacrifice"], tags_off=["control"],
                                        pinned=["C"], unpinned=["A"]),
        )
        out = deck_fit.apply_overrides(ident, ["A", "B", "C"])
        assert out.tags == ["graveyard", "sacrifice"]
        assert out.key_cards == ["B", "C"]

    def test_ignores_unknown_tag_and_pinned_card_not_in_deck(self):
        ident = DeckIdentity(tags=[], key_cards=[],
                             overrides=IdentityOverrides(tags_on=["not_a_tag"], pinned=["Gone"]))
        out = deck_fit.apply_overrides(ident, ["A"])
        assert out.tags == [] and out.key_cards == []

    def test_no_duplicates(self):
        ident = DeckIdentity(tags=["tokens"], key_cards=["A"],
                             overrides=IdentityOverrides(tags_on=["tokens"], pinned=["A"]))
        out = deck_fit.apply_overrides(ident, ["A"])
        assert out.tags == ["tokens"] and out.key_cards == ["A"]


class TestRank:
    def test_drops_anti_synergy_and_orders_by_total(self):
        fit = {"Good": fs(1.0, 1.0), "Meh": fs(0.3, 0.3), "Anti": fs(1.0, 1.0, anti=0.9)}
        assert deck_fit.rank(["Meh", "Anti", "Good"], fit, {}) == ["Good", "Meh"]

    def test_meta_frequency_breaks_fit_ties(self):
        fit = {"A": fs(0.5, 0.5), "B": fs(0.5, 0.5)}
        assert deck_fit.rank(["A", "B"], fit, {"b": 10, "a": 1}) == ["B", "A"]

    def test_missing_synergy_moves_weight_to_plan_fit(self):
        fit = {"A": fs(0.9), "B": fs(0.5)}
        assert deck_fit.rank(["B", "A"], fit, {}) == ["A", "B"]

    def test_ties_keep_input_order(self):
        fit = {"A": fs(0.5, 0.5), "B": fs(0.5, 0.5)}
        assert deck_fit.rank(["B", "A"], fit, {}) == ["B", "A"]


class TestFlagLowFit:
    def test_flags_low_average_and_anti_synergy_worst_first(self):
        fit = {"Fine": fs(0.8, 0.7), "Low": fs(0.1, 0.2), "Anti": fs(0.9, 0.9, anti=0.8),
               "Lowest": fs(0.0, 0.0)}
        assert deck_fit.flag_low_fit(fit) == ["Lowest", "Low", "Anti"]

    def test_limit(self):
        fit = {f"C{i}": fs(0.0, 0.0) for i in range(9)}
        assert len(deck_fit.flag_low_fit(fit)) == 5

    def test_empty(self):
        assert deck_fit.flag_low_fit({}) == []


def test_bucket():
    assert [deck_fit.bucket(n) for n in (0, 5, 6, 14, 15, 29, 30, 60)] == [0, 0, 1, 1, 2, 2, 3, 3]


class FakeClient:
    """Stands in for AsyncTypeSafeClient. `answer(state, questions)` -> (nouls, scores) dicts of floats."""

    def __init__(self, answer=None, fail_on=None):
        self.calls = []
        self.answer = answer or (lambda state, qs: (
            {k: 0.0 for k, q in qs.items() if type(q).__name__ == "Noul"},
            {k: 0.0 for k, q in qs.items() if type(q).__name__ == "Score"},
        ))
        self.fail_on = fail_on

    async def system_one(self, state, questions):
        self.calls.append((state, questions))
        if self.fail_on and self.fail_on(state):
            from typesafe_sdk import TypeSafeAPITimeoutError
            raise TypeSafeAPITimeoutError(2.0)
        nouls, scores = self.answer(state, questions)
        return SimpleNamespace(
            nouls={k: SimpleNamespace(noul=v) for k, v in nouls.items()},
            scores={k: SimpleNamespace(score=v) for k, v in scores.items()},
        )


def card(name, type_line="Creature — Zombie", text=""):
    return {"name": name, "mana_cost": "{B}", "type_line": type_line, "oracle_text": text}


class TestInferIdentity:
    async def test_no_cards_no_request_returns_none_without_calling(self):
        client = FakeClient()
        assert await deck_fit.infer_identity([], None, client=client) is None
        assert client.calls == []

    async def test_request_only_infers_tags_no_key_cards(self):
        def answer(state, qs):
            return ({k: (0.9 if k == "tag:graveyard" else 0.1) for k in qs}, {})
        ident = await deck_fit.infer_identity([], "self-mill zombies", client=FakeClient(answer))
        assert ident.tags == ["graveyard"]
        assert ident.key_cards == []
        assert ident.request_text == "self-mill zombies"

    async def test_key_cards_top5_by_score_lands_excluded(self):
        cards = [card(f"C{i}") for i in range(8)] + [card("Swamp", "Basic Land — Swamp")]

        def answer(state, qs):
            nouls = {k: 0.0 for k in qs if k.startswith("tag:")}
            scores = {k: float(k.split(":")[1]) for k in qs if k.startswith("key:")}
            return nouls, scores
        client = FakeClient(answer)
        ident = await deck_fit.infer_identity(cards, None, client=client)
        assert ident.key_cards == ["C7", "C6", "C5", "C4", "C3"]
        state, _ = client.calls[0]
        assert all(c["name"] != "Swamp" for c in state["deck"]["cards"])

    async def test_fewer_than_six_nonland_cards_skips_key_questions(self):
        client = FakeClient()
        await deck_fit.infer_identity([card(f"C{i}") for i in range(5)], "x", client=client)
        _, qs = client.calls[0]
        assert not any(k.startswith("key:") for k in qs)

    async def test_overrides_applied(self):
        ident = await deck_fit.infer_identity(
            [card("A")], "x", IdentityOverrides(tags_on=["tokens"]), client=FakeClient())
        assert ident.tags == ["tokens"]

    async def test_failure_returns_none(self):
        ident = await deck_fit.infer_identity([card("A")], "x", client=FakeClient(fail_on=lambda s: True))
        assert ident is None

    async def test_response_parse_error_returns_none(self):
        def bad_answer(state, qs):
            # Omit tag:graveyard key, causing KeyError during parsing
            return {k: 0.0 for k in qs if k.startswith("tag:") and k != "tag:graveyard"}, {}
        ident = await deck_fit.infer_identity([card("A")], "x", client=FakeClient(bad_answer))
        assert ident is None


class TestScoreFit:
    async def test_scores_normalized_and_deduped(self):
        def answer(state, qs):
            return {"anti_synergy": 0.2}, {"plan_fit": 3.0, "synergy": 1.5}
        client = FakeClient(answer)
        ident = DeckIdentity(tags=["graveyard"], key_cards=["K"])
        fit = await deck_fit.score_fit(ident, [card("K")], [card("A"), card("A"), card("B")], client=client)
        assert set(fit) == {"A", "B"}
        assert len(client.calls) == 2
        assert fit["A"] == FitScore(plan_fit=1.0, synergy=0.5, anti_synergy=0.2)

    async def test_no_key_cards_skips_synergy(self):
        client = FakeClient()
        fit = await deck_fit.score_fit(DeckIdentity(tags=["tokens"]), [], [card("A")], client=client)
        _, qs = client.calls[0]
        assert "synergy" not in qs
        assert fit["A"].synergy is None

    async def test_any_failure_returns_empty(self):
        client = FakeClient(fail_on=lambda s: s["candidate"]["name"] == "B")
        fit = await deck_fit.score_fit(DeckIdentity(), [], [card("A"), card("B")], client=client)
        assert fit == {}

    async def test_no_api_key_returns_empty(self, monkeypatch):
        monkeypatch.setattr(deck_fit.settings, "TYPESAFE_API_KEY", None)
        assert await deck_fit.score_fit(DeckIdentity(), [], [card("A")]) == {}

    async def test_cancels_pending_tasks_on_first_failure(self):
        class CancellationTrackingClient:
            def __init__(self):
                self.calls = []
                self.cancelled = []

            async def system_one(self, state, questions):
                self.calls.append((state, questions))
                try:
                    if state["candidate"]["name"] == "A":
                        raise ValueError("A failed")
                    # B waits forever, allowing A to fail and cancel it
                    await asyncio.sleep(float('inf'))
                except asyncio.CancelledError:
                    self.cancelled.append(state["candidate"]["name"])
                    raise

        client = CancellationTrackingClient()
        fit = await asyncio.wait_for(
            deck_fit.score_fit(DeckIdentity(), [], [card("A"), card("B")], client=client),
            timeout=1.0,
        )
        assert fit == {}
        assert "B" in client.cancelled  # B was cancelled, not left hanging

    async def test_propagates_caller_cancellation(self):
        class SlowClient:
            async def system_one(self, state, questions):
                # All requests wait forever
                await asyncio.sleep(float('inf'))

        client = SlowClient()
        task = asyncio.create_task(
            deck_fit.score_fit(DeckIdentity(), [], [card("A")], client=client)
        )
        await asyncio.sleep(0.01)  # Let task start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestReviewDeck:
    async def test_lands_only_deck_returns_empty(self, monkeypatch):
        async def fake_load(db, names):
            return [card("Swamp", "Basic Land — Swamp")]
        monkeypatch.setattr(deck_fit, "load_payloads", fake_load)
        identity, fit = await deck_fit.review_deck(None, [{"card_name": "Swamp"}], None, client=FakeClient())
        assert identity is None and fit == {}

    async def test_scores_nonland_cards_with_key_payloads(self, monkeypatch):
        cards = [card(f"C{i}") for i in range(6)] + [card("Swamp", "Basic Land — Swamp")]

        async def fake_load(db, names):
            return cards
        monkeypatch.setattr(deck_fit, "load_payloads", fake_load)

        def answer(state, qs):
            nouls = {k: 0.0 for k in qs if k.startswith("tag:") or k == "anti_synergy"}
            scores = {k: float(k.split(":")[1]) if k.startswith("key:") else 1.0
                      for k in qs if k.startswith("key:") or k in ("plan_fit", "synergy")}
            return nouls, scores
        client = FakeClient(answer)
        identity, fit = await deck_fit.review_deck(
            None, [{"card_name": c["name"]} for c in cards], "x", client=client)
        assert identity.key_cards == ["C5", "C4", "C3", "C2", "C1"]
        assert set(fit) == {f"C{i}" for i in range(6)}
        fit_state, _ = client.calls[1]
        assert [k["name"] for k in fit_state["deck"]["key_cards"]] == ["C5", "C4", "C3", "C2", "C1"]


def test_deck_fit_response_schema():
    from app.schemas.deck import DeckFitResponse
    r = DeckFitResponse(identity=None, cards={}, flagged=[], available_tags=list(deck_fit.THEME_TAGS))
    assert r.model_dump()["available_tags"][0] == "graveyard"


def test_generate_response_has_fit_flagged_default():
    from app.schemas.deck import DeckGenerateResponse
    assert DeckGenerateResponse.model_fields["fit_flagged"].default_factory() == []
