"""
Build four Standard decks with real Jev assembly and check each one: legal,
60 main and 15 sideboard, every card played in the last 14
days or requested, nonbasic lands within the deck colors, at most 4 copies.
Brews print the relatives Jev chose and every nonland card from outside the
relatives' lists (a format top-up). Prints the lists for human review. Not run in CI.

Usage (inside the backend container, which has the DB and keys):
  docker compose exec -T backend python scripts/eval_assembly.py
"""

import asyncio
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from app.db.session import async_session_factory  # noqa: E402
from app.services import deck_fill  # noqa: E402
from app.services.ai.deck_parsing import parse_deck_request  # noqa: E402
from app.services.deck_plan import RECENT, archetype_keys  # noqa: E402
from app.services.deck_validator import BASIC_LANDS  # noqa: E402

# (request, must have a reference archetype). Brews print their relatives; none is
# not a failure (the brew then uses the LLM plan).
CASES = [
    ("Build me a Boros aggro deck", True),
    ("mono-red aggro", False),
    ("a red-green aggro deck", False),
    ("A black-red midrange deck built around Sephiroth, Fabled SOLDIER", False),
]

CARD_SQL = text("""
    SELECT c.name, bool_or(c.legalities->>'standard' = 'legal') AS legal,
           MAX(c.type_line) AS type_line, MAX(c.color_identity) AS identity
    FROM cards c WHERE c.name = ANY(CAST(:names AS varchar[])) GROUP BY c.name
""")
PLAYED_SQL = text(f"""
    SELECT DISTINCT lower(split_part(x->>'card_name', ' // ', 1))
    FROM decklists d JOIN events e ON e.id = d.event_id
    CROSS JOIN LATERAL jsonb_array_elements(d.main_deck || d.sideboard) AS x
    WHERE {RECENT}
""")
RELATIVES_SQL = text(f"""
    SELECT DISTINCT lower(split_part(x->>'card_name', ' // ', 1))
    FROM decklists d JOIN events e ON e.id = d.event_id
    CROSS JOIN LATERAL jsonb_array_elements(d.main_deck || d.sideboard) AS x
    WHERE {RECENT} AND lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))
""")


async def top_ups(db, deck) -> list:
    """Nonland cards from outside the relatives' lists: format top-ups."""
    if not deck["relatives"]:
        return []
    params = {"format": "standard", "archetypes": archetype_keys(deck["relatives"])}
    theirs = {r[0] for r in (await db.execute(RELATIVES_SQL, params)).all()}
    names = [e["card_name"] for e in deck["main_deck"] + deck["sideboard"]]
    cards = {r.name: r for r in (await db.execute(CARD_SQL, {"names": names})).all()}
    return [n for n in names if n.split(" // ")[0].lower() not in theirs
            and n in cards and "Land" not in cards[n].type_line.split(" // ")[0]]


async def check(db, deck, requested, need_reference) -> list:
    main = Counter({e["card_name"]: e["quantity"] for e in deck["main_deck"]})
    side = Counter({e["card_name"]: e["quantity"] for e in deck["sideboard"]})
    names = sorted(set(main) | set(side))
    cards = {r.name: r for r in (await db.execute(CARD_SQL, {"names": names})).all()}
    played = {r[0] for r in (await db.execute(PLAYED_SQL, {"format": "standard"})).all()}
    asked = {n.lower() for n in requested}
    problems = []
    if need_reference and not deck["reference"]:
        problems.append("no reference archetype")
    if sum(main.values()) != 60:
        problems.append(f"main is {sum(main.values())}")
    if sum(side.values()) != 15:
        problems.append(f"sideboard is {sum(side.values())}")
    for n in names:
        row = cards.get(n)
        if row is None or not row.legal:
            problems.append(f"not legal: {n}")
            continue
        if n not in BASIC_LANDS and n.lower() not in asked \
                and n.split(" // ")[0].lower() not in played:
            problems.append(f"not played or requested: {n}")
        if "Land" in row.type_line.split(" // ")[0] and n not in BASIC_LANDS \
                and not set(row.identity or []) <= set(deck["colors"]):
            problems.append(f"off-color land: {n} {row.identity}")
        if n not in BASIC_LANDS and main[n] + side[n] > 4:
            problems.append(f"{main[n] + side[n]} copies: {n}")
    return problems


async def main() -> None:
    failed = 0
    async with async_session_factory() as db:
        for request, need_reference in CASES:
            parsed = await parse_deck_request(request, db)
            start = time.time()
            try:
                deck = await deck_fill.assemble(db, request, parsed["colors"], parsed["specific_cards"],
                                                "standard", True, parsed["archetype"])
            except Exception as e:  # in the app this falls back to the LLM path
                failed += 1
                print(f"\n=== {request}\nERROR (Jev assembly raised): {e!r}")
                continue
            took = time.time() - start
            problems = await check(db, deck, parsed["specific_cards"], need_reference)
            failed += bool(problems)
            print(f"\n=== {request}  ({took:.1f}s)")
            print(f"reference={deck['reference']} relatives={deck['relatives']} colors={deck['colors']} "
                  f"requested={parsed['specific_cards']}")
            if deck["relatives"]:
                print(f"format top-ups: {await top_ups(db, deck) or 'none'}")
            print(f"{deck['name']}: {deck['strategy_summary']}")
            for e in deck["main_deck"]:
                print(f"  {e['quantity']} {e['card_name']}")
            if deck["sideboard"]:
                print("  Sideboard:")
                for e in deck["sideboard"]:
                    print(f"  {e['quantity']} {e['card_name']}")
            print("OK" if not problems else "PROBLEMS: " + "; ".join(problems))
    print("\nPASS" if not failed else f"\nFAIL ({failed} of {len(CASES)} decks)")


if __name__ == "__main__":
    asyncio.run(main())
