"""chat_routing: Jev questions, tool-argument assembly, and failure fallbacks."""

from app.services import chat_routing as cr
from app.services.chat_service import TOOLS
from tests.jev_fake import FakeJev

SHEOLDRED = "Sheoldred, the Apocalypse"
NO_DECK = {"has_deck": False, "unique_nonland_cards": 0, "colors": []}
BLACK_DECK = {"has_deck": True, "unique_nonland_cards": 12, "colors": ["B"]}


async def route(message, answers, summary=NO_DECK, strategy="", cards=(), meta=(), fill=True,
                history=None):
    client = FakeJev(lambda state, qs: answers, fill=fill)
    r = await cr.route(message=message, history=history or [{"role": "user", "content": message}],
                       summary=summary, strategy=strategy, card_names=list(cards),
                       meta_archetypes=list(meta), tools=TOOLS, client=client)
    return r, client


def test_deck_summary_colors_from_pips_else_fallback():
    deck = {"main_deck": [
        {"card_name": "Shock", "card": {"mana_cost": "{R}", "type_line": "Instant"}},
        {"card_name": "Mountain", "card": {"mana_cost": "", "type_line": "Basic Land — Mountain"}},
        {"card_name": "Murder", "card": {"mana_cost": "{1}{B}{B}", "type_line": "Instant"}},
    ]}
    assert cr.deck_summary(deck, ["G"]) == {"has_deck": True, "unique_nonland_cards": 2, "colors": ["B", "R"]}
    assert cr.deck_summary(None, ["G"]) == {"has_deck": False, "unique_nonland_cards": 0, "colors": ["G"]}


def test_choice_options_dedupe():
    qs = cr.build_questions(TOOLS, [SHEOLDRED], ["protection", "surveil"],
                            ["Izzet Prowess", "izzet prowess", "Mono-Red Aggro", ""])
    assert list(qs["opponent"].criteria) == ["Izzet Prowess", "Mono-Red Aggro", "none"]
    pkg = qs["package_role"].criteria
    assert list(pkg).count("protection") == 1
    assert pkg["surveil"] == "Cards with the surveil mechanic"
    assert set(qs["action"].criteria) == {t["name"] for t in TOOLS} | {cr.REPLY}
    assert "wants:0" in qs and "wants:1" not in qs
    assert all(f"role:{r}" in qs for r in cr.CORE_ROLES)


def test_opponent_choice_always_has_none():
    qs = cr.build_questions(TOOLS, [], [], [])
    assert list(qs["opponent"].criteria) == ["none"]


async def test_action_confidence_and_probabilities():
    answers = {"action": {"choice": "suggest_core", "confidence": 0.8,
                          "probabilities": {"suggest_core": 0.9, "reply": 0.1}}}
    r, _ = await route("Build me a mono-red aggro deck", answers)
    assert (r.action, r.confidence) == ("suggest_core", 0.8)
    assert r.probabilities == {"suggest_core": 0.9, "reply": 0.1}
    assert r.top_tool() == "suggest_core"


async def test_top_tool_never_returns_reply():
    r = cr.Route("reply", 0.9, {"reply": 0.7, "analyze_meta": 0.2, "suggest_core": 0.1},
                 {t["name"]: {} for t in TOOLS})
    assert r.top_tool() == "analyze_meta"


async def test_colors_from_nouls_else_deck_colors_and_opponent():
    r, _ = await route("beat mono-red", {"color:R": 0.1, "opponent": "Mono-Red Aggro"},
                       summary=BLACK_DECK, meta=["Mono-Red Aggro"])
    assert r.inputs["suggest_core"]["colors"] == ["B"]
    assert r.inputs["get_matchup_info"] == {"opponent_deck": "Mono-Red Aggro"}
    r, _ = await route("red aggro", {"color:R": 0.9}, summary=BLACK_DECK)
    assert r.inputs["suggest_core"]["colors"] == ["R"]


async def test_roles_package_role_and_count():
    r, _ = await route("give me 8 removal spells", {"role:removal": 0.9, "package_role": "removal"})
    assert r.inputs["suggest_core"]["roles"] == ["removal"]
    assert r.inputs["suggest_package"]["role"] == "removal"
    assert r.inputs["suggest_package"]["count"] == 8
    r, _ = await route("more stuff", {})
    assert r.inputs["suggest_core"]["roles"] == cr.DEFAULT_ROLES
    assert r.inputs["suggest_package"]["count"] == cr.DEFAULT_COUNT
    r, _ = await route("give me 100 cards", {})
    assert r.inputs["suggest_package"]["count"] == cr.MAX_COUNT


async def test_unwanted_card_is_not_built_around():
    msg = f"Beat {SHEOLDRED} with Monastery Swiftspear aggro"
    r, client = await route(msg, {"wants:0": 0.1, "wants:1": 0.9},
                            cards=[SHEOLDRED, "Monastery Swiftspear"])
    assert r.inputs["generate_full_deck"]["specific_cards"] == ["Monastery Swiftspear"]
    assert "Sheoldred" not in r.inputs["suggest_core"]["strategy"]
    assert "Apocalypse" not in r.inputs["suggest_core"]["strategy"]
    assert "Monastery Swiftspear" in r.inputs["suggest_core"]["strategy"]
    assert r.inputs["modify_deck"] == {"modification": msg}
    assert client.calls[0][0]["cards"] == [SHEOLDRED, "Monastery Swiftspear"]


async def test_strategy_joins_persisted_and_is_capped():
    r, _ = await route("more burn", {}, strategy="Mono-red aggro")
    assert r.inputs["suggest_core"]["strategy"] == "Mono-red aggro; more burn"
    r, _ = await route("more burn", {}, strategy="x" * 400)
    assert len(r.inputs["suggest_core"]["strategy"]) == cr.STRATEGY_CHARS


async def test_archetype_none_is_omitted():
    r, _ = await route("just build it", {"archetype": "none"})
    assert "archetype" not in r.inputs["generate_full_deck"]
    assert r.inputs["analyze_meta"] == {"focus": ""}
    r, _ = await route("just build it", {"archetype": "tempo"})
    assert r.inputs["generate_full_deck"]["archetype"] == "tempo"


async def test_state_carries_last_six_turns_before_this_message():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}" + "x" * 600}
               for i in range(9)] + [{"role": "user", "content": "now"}]
    _, client = await route("now", {}, history=history, strategy="s", meta=["A"])
    state = client.calls[0][0]
    assert [t["content"][:2] for t in state["recent_turns"]] == ["m3", "m4", "m5", "m6", "m7", "m8"]
    assert all(len(t["content"]) == cr.TURN_CHARS for t in state["recent_turns"])
    assert state["message"] == "now" and state["strategy"] == "s" and state["meta_archetypes"] == ["A"]


async def test_mechanics_from_message_become_package_options():
    _, client = await route("cards with surveil", {})
    state, qs, _ = client.calls[0]
    assert state["mechanics"] == ["surveil"] and "surveil" in qs["package_role"].criteria


async def test_partial_answer_falls_back():
    r, _ = await route("hi", {"action": "reply"}, fill=False)
    assert r is None


async def test_failure_and_missing_key_fall_back():
    client = FakeJev(fail=lambda state: True)
    assert await cr.route(message="hi", history=[], summary=NO_DECK, strategy="", card_names=[],
                          meta_archetypes=[], tools=TOOLS, client=client) is None
    assert await cr.route(message="hi", history=[], summary=NO_DECK, strategy="", card_names=[],
                          meta_archetypes=[], tools=TOOLS) is None


async def test_guild_name_adds_its_colors_only_when_wanted():
    r, _ = await route("Build me a Boros aggro deck", {"color:R": 0.6, "color:W": 0.3})
    assert r.inputs["suggest_core"]["colors"] == ["W", "R"]
    r, _ = await route("how do I beat Boros?", {"color:R": 0.2, "color:W": 0.1})
    assert r.inputs["suggest_core"]["colors"] == []
