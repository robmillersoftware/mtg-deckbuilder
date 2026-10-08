"""
Run labeled chat messages through real Jev routing, and deck requests through
Jev parsing, and report action/argument accuracy plus a ROUTE_CONFIDENCE sweep.

Usage (needs TYPESAFE_API_KEY in .env):  cd backend && python3 scripts/eval_routing.py
"""

import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import chat_routing  # noqa: E402
from app.services.ai import deck_parsing  # noqa: E402
from app.services.chat_service import TOOLS  # noqa: E402

META = ["Mono-Red Aggro", "Dimir Midrange", "Izzet Prowess", "Azorius Control"]
NO_DECK = {"has_deck": False, "unique_nonland_cards": 0, "colors": []}
RED_DECK = {"has_deck": True, "unique_nonland_cards": 14, "colors": ["R"]}
UB_DECK = {"has_deck": True, "unique_nonland_cards": 20, "colors": ["U", "B"]}
FULL_DECK = {"has_deck": True, "unique_nonland_cards": 36, "colors": ["R"]}
SHEOLDRED = "Sheoldred, the Apocalypse"

# (message, deck summary, persisted strategy, resolved card names, expected action, expected args)
# Args are checked on the expected action's input; lists compare as sets. Special keys:
#   _not_color: a color that must not be in suggest_core's colors
#   _has_role:  a role that must be in suggest_core's roles
#   _wants:     generate_full_deck's specific_cards, as a set
CASES = [
    ("Build me a mono-red aggro deck", NO_DECK, "", [], "suggest_core", {"colors": ["R"]}),
    ("beat mono-red", NO_DECK, "", [], "get_matchup_info",
     {"opponent_deck": "Mono-Red Aggro", "_not_color": "R"}),
    ("more removal", RED_DECK, "Mono-red aggro", [], "suggest_package", {"role": "removal"}),
    ("just build it", NO_DECK, "Mono-red aggro", [], "generate_full_deck", {}),
    ("what's good right now?", NO_DECK, "", [], "analyze_meta", {}),
    ("swap Shock for Lightning Strike", RED_DECK, "Mono-red aggro", ["Shock", "Lightning Strike"],
     "modify_deck", {"modification": "swap Shock for Lightning Strike"}),
    ("What does trample do?", RED_DECK, "Mono-red aggro", [], "reply", {}),
    ("How do I beat Dimir Midrange?", RED_DECK, "Mono-red aggro", [], "get_matchup_info",
     {"opponent_deck": "Dimir Midrange"}),
    ("Give me 8 cards with surveil", UB_DECK, "UB self-mill", [], "suggest_package",
     {"role": "surveil", "count": 8}),
    ("I need more card draw", UB_DECK, "UB control", [], "suggest_package", {"role": "card draw"}),
    ("Let's add the mana base", FULL_DECK, "Mono-red aggro", [], "finalize_mana_base", {"colors": ["R"]}),
    (f"Build around {SHEOLDRED}", NO_DECK, "", [SHEOLDRED], "suggest_core", {"_wants": [SHEOLDRED]}),
    (f"My opponent keeps casting {SHEOLDRED}, how do I deal with it?", RED_DECK, "Mono-red aggro",
     [SHEOLDRED], "get_matchup_info", {"_wants": []}),
    ("Make a Dimir control deck", NO_DECK, "", [], "suggest_core", {"colors": ["U", "B"]}),
    ("Skip the suggestions and give me a whole Boros deck", NO_DECK, "", [], "generate_full_deck",
     {"colors": ["R", "W"]}),
    ("Is Lightning Strike better than Shock?", RED_DECK, "Mono-red aggro",
     ["Lightning Strike", "Shock"], "reply", {}),
    ("Cut the four Shocks", RED_DECK, "Mono-red aggro", ["Shock"], "modify_deck", {}),
    ("What's the best aggro deck in the format?", NO_DECK, "", [], "analyze_meta", {}),
    ("What sideboard cards help against Izzet Prowess?", RED_DECK, "Mono-red aggro", [],
     "get_matchup_info", {"opponent_deck": "Izzet Prowess"}),
    ("Show me some counterspells for my deck", UB_DECK, "UB control", [], "suggest_package",
     {"role": "counterspells"}),
    ("I want to play green ramp", NO_DECK, "", [], "suggest_core", {"colors": ["G"], "_has_role": "ramp"}),
    ("Thanks!", RED_DECK, "Mono-red aggro", [], "reply", {}),
    ("Add lands", FULL_DECK, "Mono-red aggro", [], "finalize_mana_base", {}),
    ("Give me some cheap threats", RED_DECK, "Mono-red aggro", [], "suggest_package",
     {"role": "cheap threats"}),
    ("What archetypes are popular?", NO_DECK, "", [], "analyze_meta", {}),
]
# The spec's must-route cases: these may fall below the threshold (and go to the
# LLM) but must never be confidently wrong.
REQUIRED = {"beat mono-red", "more removal", "just build it", "what's good right now?",
            "swap Shock for Lightning Strike", "What does trample do?"}

# (prompt, resolved card names, expected fields). _not_color: a color that must not be picked.
PARSE_CASES = [
    ("Build a mono-red aggro deck", [],
     {"archetype": "aggro", "colors": ["R"], "colors_specified": True, "specific_cards": []}),
    ("A deck that beats mono-red", [], {"colors_specified": False, "_not_color": "R"}),
    (f"Dimir control with {SHEOLDRED}", [SHEOLDRED],
     {"archetype": "control", "colors": ["U", "B"], "colors_specified": True,
      "specific_cards": [SHEOLDRED]}),
    (f"Something to beat {SHEOLDRED} decks", [SHEOLDRED], {"specific_cards": []}),
    ("aggro deck", [], {"archetype": "aggro", "colors_specified": False}),
]


def same(a, b):
    return set(a) == set(b) if isinstance(b, list) else a == b


def args_ok(r: chat_routing.Route, action: str, expected: dict) -> bool:
    got = r.inputs.get(action, {})
    for key, want in expected.items():
        if key == "_not_color":
            if want in r.inputs["suggest_core"]["colors"]:
                return False
        elif key == "_has_role":
            if want not in r.inputs["suggest_core"]["roles"]:
                return False
        elif key == "_wants":
            if not same(r.inputs["generate_full_deck"]["specific_cards"], want):
                return False
        elif not same(got.get(key), want):
            return False
    return True


async def eval_routing() -> bool:
    rows = []
    times = []
    failed = 0
    print(f"{'message':46} {'expected':18} {'got':18} conf")
    for msg, deck, strategy, cards, want, expected in CASES:
        t0 = time.perf_counter()
        r = await chat_routing.route(
            message=msg, history=[{"role": "user", "content": msg}], summary=deck,
            strategy=strategy, card_names=cards, meta_archetypes=META, tools=TOOLS)
        dt = time.perf_counter() - t0
        times.append(dt)
        if r is None:
            failed += 1
            print(f"{msg[:46]:46} ERROR (no key, Jev unavailable, or timed out) {dt:.2f}s")
            continue
        ok = r.action == want
        ok_args = ok and args_ok(r, want, expected)
        rows.append((msg, r, ok, ok_args))
        verdict = ("ok" if ok_args else "args-MISS") if ok else "MISS"
        print(f"{msg[:46]:46} {want:18} {r.action:18} {r.confidence:.2f} {verdict}")

    print(f"\nlatency s: min {min(times):.2f} median {statistics.median(times):.2f} max {max(times):.2f}; "
          f"{failed} of {len(times)} routing calls returned None (timeout/error)")
    if failed:
        return False
    n = len(rows)
    right = [row for row in rows if row[2]]
    print(f"\naction accuracy {len(right)}/{n}; argument accuracy {sum(row[3] for row in right)}/{len(right)}")
    for t in (0.4, 0.5, 0.6, 0.7, 0.8):
        confident = [row for row in rows if row[1].confidence >= t]
        good = sum(row[3] for row in confident)
        print(f"threshold {t:.1f}: {len(confident)}/{n} routed without the LLM, {good} fully correct")

    t = chat_routing.ROUTE_CONFIDENCE
    confident = [row for row in rows if row[1].confidence >= t]
    wrong_required = [row[0] for row in confident if row[0] in REQUIRED and not row[3]]
    for msg in wrong_required:
        print(f"REQUIRED CONFIDENTLY WRONG  {msg}")
    precision = sum(row[3] for row in confident) / max(len(confident), 1)
    print(f"at ROUTE_CONFIDENCE={t}: {precision:.0%} of confident routes fully correct")
    return precision >= 0.9 and not wrong_required


async def eval_parsing() -> bool:
    agree = 0
    print(f"\n{'prompt':46} result")
    for prompt, names, want in PARSE_CASES:
        async def extract(p, db, format="standard", _names=names):
            return list(_names)
        deck_parsing.extract_card_names_from_prompt = extract
        got = await deck_parsing.parse_deck_request(prompt, db=None)
        if "budget" in got:
            print(f"{prompt[:46]:46} FALLBACK (no key or Jev unavailable)")
            continue
        ok = all(want["_not_color"] not in got["colors"] if k == "_not_color" else same(got[k], v)
                 for k, v in want.items())
        agree += ok
        print(f"{prompt[:46]:46} {'ok' if ok else 'MISS'}  {got}")
    print(f"parsing: {agree}/{len(PARSE_CASES)}")
    return agree >= 4


async def main() -> None:
    routing_ok = await eval_routing()
    parsing_ok = await eval_parsing()
    print("PASS" if routing_ok and parsing_ok else
          "FAIL (need >= 90% of confident routes correct, no required case confidently wrong, parsing >= 4/5)")


if __name__ == "__main__":
    asyncio.run(main())
