"""Role tagging with Jev: thresholds, efficiency mapping, stored confidence, batch failure."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.dialects import postgresql

from app.jobs import classify_card_roles as job
from app.models.card import CARD_ROLES
from tests.jev_fake import FakeJev


def card(name, **kw):
    base = dict(name=name, mana_cost="{1}{B}{B}", cmc=3.0, type_line="Instant",
                oracle_text="Destroy target creature.", power=None, toughness=None)
    return SimpleNamespace(**{**base, **kw})


async def test_one_request_per_card_with_a_noul_and_score_per_role():
    client = FakeJev()
    await job.classify_cards_batch([card("Murder"), card("Shock")], client=client)
    assert len(client.calls) == 2
    state, questions, kwargs = client.calls[0]
    assert state == {"card": {"name": "Murder", "mana_cost": "{1}{B}{B}", "mana_value": 3.0,
                              "type_line": "Instant", "oracle_text": "Destroy target creature."}}
    assert set(questions) == {f"is:{r}" for r in CARD_ROLES} | {f"eff:{r}" for r in CARD_ROLES}
    assert kwargs == {"timeout": job.TAG_REQUEST_TIMEOUT}


async def test_power_toughness_included_for_creatures():
    client = FakeJev()
    await job.classify_cards_batch(
        [card("Bear", type_line="Creature — Bear", power="2", toughness="2")], client=client)
    assert client.calls[0][0]["card"]["power_toughness"] == "2/2"


async def test_threshold_efficiency_and_confidence():
    def answer(state, questions):
        return {"is:removal_targeted": 0.834, "eff:removal_targeted": 3.6,
                "is:burn": 0.5, "eff:burn": 0.0,
                "is:card_draw": 0.49, "eff:card_draw": 4.0}
    out = await job.classify_cards_batch([card("Murder")], client=FakeJev(answer))
    assert out == [{"name": "Murder", "roles": [
        {"role": "removal_targeted", "efficiency": 5, "confidence": 0.83},
        {"role": "burn", "efficiency": 1, "confidence": 0.5},
    ]}]


async def test_any_failure_skips_the_batch():
    client = FakeJev(fail=lambda state: state["card"]["name"] == "Shock")
    assert await job.classify_cards_batch([card("Murder"), card("Shock")], client=client) == []


async def test_partial_answer_skips_the_batch():
    assert await job.classify_cards_batch([card("Murder")], client=FakeJev(fill=False)) == []


async def test_no_key_skips_the_batch():
    assert await job.classify_cards_batch([card("Murder")]) == []


async def test_save_stores_noul_as_confidence_and_no_reasoning():
    ids = MagicMock()
    ids.all.return_value = [("id-1", "Murder")]
    db = MagicMock(execute=AsyncMock(side_effect=[ids, None]), commit=AsyncMock())
    saved = await job.save_card_roles(db, [card("Murder")], [
        {"name": "Murder", "roles": [{"role": "removal_targeted", "efficiency": 5, "confidence": 0.83}]}])
    assert saved == 1
    params = db.execute.call_args_list[1][0][0].compile(dialect=postgresql.dialect()).params
    assert params["confidence"] == 0.83
    assert params["efficiency"] == 5
    assert params["reasoning"] is None


async def test_skipped_batch_is_reported_in_stats(monkeypatch):
    class Session:
        async def __aenter__(self):
            return MagicMock(execute=AsyncMock(return_value=MagicMock(scalar=lambda: 2)))

        async def __aexit__(self, *exc):
            return False

    async def pending(db, limit):
        return [card("A"), card("B")]

    async def none(cards):
        return []
    monkeypatch.setattr(job, "async_session_factory", Session)
    monkeypatch.setattr(job, "get_unclassified_cards", pending)
    monkeypatch.setattr(job, "classify_cards_batch", none)
    stats = await job.classify_all_cards()
    assert stats["errors"] == ["batch 1: skipped (2 cards)"]
