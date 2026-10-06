"""
Run hand-labeled cards through real Jev role tagging and report per-role
precision and recall.

Usage (needs TYPESAFE_API_KEY in .env):  cd backend && python3 scripts/eval_roles.py
"""

import asyncio
import os
import sys
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.jobs import classify_card_roles as job  # noqa: E402


def c(name, cost, cmc, type_line, text, pt=None):
    power, toughness = pt.split("/") if pt else (None, None)
    return SimpleNamespace(name=name, mana_cost=cost, cmc=cmc, type_line=type_line,
                           oracle_text=text, power=power, toughness=toughness)


# (card, expected roles)
CASES = [
    (c("Basilisk Collar", "{1}", 1, "Artifact — Equipment",
       "Equipped creature has deathtouch and lifelink.\nEquip {2}"), set()),
    (c("Murder", "{1}{B}{B}", 3, "Instant", "Destroy target creature."), {"removal_targeted"}),
    (c("Consider", "{U}", 1, "Instant", "Surveil 1.\nDraw a card."), {"card_selection", "card_draw"}),
    (c("Forest", "", 0, "Basic Land — Forest", "({T}: Add {G}.)"), {"land_basic"}),
    (c("Hallowed Fountain", "", 0, "Land — Plains Island",
       "({T}: Add {W} or {U}.)\nAs Hallowed Fountain enters the battlefield, you may pay 2 life. "
       "If you don't, it enters the battlefield tapped."), {"land_fixing_untapped"}),
    (c("Monastery Swiftspear", "{R}", 1, "Creature — Human Monk", "Haste\nProwess", "1/2"),
     {"threat_cheap"}),
    (c("Lightning Bolt", "{R}", 1, "Instant", "Lightning Bolt deals 3 damage to any target."),
     {"burn", "removal_targeted"}),
    (c("Wrath of God", "{2}{W}{W}", 4, "Sorcery", "Destroy all creatures. They can't be regenerated."),
     {"removal_mass"}),
    (c("Counterspell", "{U}{U}", 2, "Instant", "Counter target spell."), {"counterspell"}),
    (c("Negate", "{1}{U}", 2, "Instant", "Counter target noncreature spell."), {"counterspell"}),
    (c("Divination", "{2}{U}", 3, "Sorcery", "Draw two cards."), {"card_draw"}),
    (c("Opt", "{U}", 1, "Instant", "Scry 1.\nDraw a card."), {"card_selection", "card_draw"}),
    (c("Llanowar Elves", "{G}", 1, "Creature — Elf Druid", "{T}: Add {G}.", "1/1"), {"ramp"}),
    (c("Cultivate", "{2}{G}", 3, "Sorcery",
       "Search your library for up to two basic land cards, reveal those cards, put one onto the "
       "battlefield tapped and the other into your hand, then shuffle."), {"ramp"}),
    (c("Thoughtseize", "{B}", 1, "Sorcery",
       "Target player reveals their hand. You choose a nonland card from it. That player discards "
       "that card. You lose 2 life."), {"discard"}),
    (c("Disenchant", "{1}{W}", 2, "Instant", "Destroy target artifact or enchantment."),
     {"removal_artifact_enchantment"}),
    (c("Rest in Peace", "{1}{W}", 2, "Enchantment",
       "When Rest in Peace enters the battlefield, exile all graveyards.\nIf a card or token would be "
       "put into a graveyard from anywhere, exile it instead."), {"graveyard_hate"}),
    (c("Raise Dead", "{B}", 1, "Sorcery", "Return target creature card from your graveyard to your hand."),
     {"recursion"}),
    (c("Demonic Tutor", "{1}{B}", 2, "Sorcery",
       "Search your library for a card, put that card into your hand, then shuffle."), {"tutor"}),
    (c("Heroic Intervention", "{1}{G}", 2, "Instant",
       "Permanents you control gain hexproof and indestructible until end of turn."), {"protection"}),
    (c("Revitalize", "{1}{W}", 2, "Instant", "You gain 3 life.\nDraw a card."), {"lifegain", "card_draw"}),
    (c("Shivan Dragon", "{4}{R}{R}", 6, "Creature — Dragon",
       "Flying\n{R}: Shivan Dragon gets +1/+0 until end of turn.", "5/5"), {"threat_finisher"}),
    (c("Watchwolf", "{G}{W}", 2, "Creature — Wolf", "", "3/3"), {"threat_cheap"}),
    (c("Ravenous Chupacabra", "{2}{B}{B}", 4, "Creature — Beast Horror",
       "When Ravenous Chupacabra enters the battlefield, destroy target creature an opponent controls.",
       "2/2"), {"threat_midrange", "removal_targeted"}),
    (c("Sheoldred, the Apocalypse", "{2}{B}{B}", 4, "Legendary Creature — Phyrexian Praetor",
       "Deathtouch\nWhenever you draw a card, you gain 2 life.\nWhenever an opponent draws a card, "
       "they lose 2 life.", "4/5"), {"threat_midrange"}),
    (c("Lava Spike", "{R}", 1, "Sorcery", "Lava Spike deals 3 damage to target player or planeswalker."),
     {"burn"}),
    (c("Mutavault", "", 0, "Land",
       "{T}: Add {C}.\n{1}: Mutavault becomes a 2/2 creature with all creature types until end of turn. "
       "It's still a land."), {"land_creature"}),
    (c("Field of Ruin", "", 0, "Land",
       "{T}: Add {C}.\n{2}, {T}, Sacrifice Field of Ruin: Destroy target nonbasic land an opponent "
       "controls. Each player searches their library for a basic land card, puts it onto the "
       "battlefield, then shuffles."), {"land_utility"}),
    (c("Dismal Backwater", "", 0, "Land",
       "Dismal Backwater enters the battlefield tapped.\nWhen Dismal Backwater enters the battlefield, "
       "you gain 1 life.\n{T}: Add {U} or {B}."), {"land_fixing_tapped"}),
    (c("Ornithopter", "{0}", 0, "Artifact Creature — Thopter", "Flying", "0/2"), set()),
]

# Spec-required checks: (card name, predicate on the assigned role set, description)
REQUIRED = [
    ("Basilisk Collar", lambda got: "removal_targeted" not in got, "not removal_targeted"),
    ("Murder", lambda got: "removal_targeted" in got, "removal_targeted"),
    ("Consider", lambda got: "card_selection" in got, "card_selection"),
    ("Forest", lambda got: "land_basic" in got, "land_basic"),
    ("Hallowed Fountain", lambda got: any(r.startswith("land_fixing_") for r in got), "land_fixing_*"),
    ("Monastery Swiftspear", lambda got: "threat_cheap" in got, "threat_cheap"),
]


async def main() -> None:
    out = await job.classify_cards_batch([card for card, _ in CASES])
    if not out:
        print("ERROR: no results (no TYPESAFE_API_KEY, or Jev unavailable)")
        return
    got_by_name = {r["name"]: {x["role"] for x in r["roles"]} for r in out}
    tp, fp, fn = Counter(), Counter(), Counter()
    print(f"{'card':28} {'expected':45} got")
    for card, want in CASES:
        got = got_by_name[card.name]
        for r in got & want:
            tp[r] += 1
        for r in got - want:
            fp[r] += 1
        for r in want - got:
            fn[r] += 1
        mark = "" if got == want else "  <-"
        print(f"{card.name:28} {','.join(sorted(want)) or '-':45} {','.join(sorted(got)) or '-'}{mark}")

    print(f"\n{'role':30} precision  recall")
    for role in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[role] / (tp[role] + fp[role]) if tp[role] + fp[role] else float("nan")
        r = tp[role] / (tp[role] + fn[role]) if tp[role] + fn[role] else float("nan")
        print(f"{role:30} {p:9.2f}  {r:6.2f}")
    micro_p = sum(tp.values()) / max(sum(tp.values()) + sum(fp.values()), 1)
    micro_r = sum(tp.values()) / max(sum(tp.values()) + sum(fn.values()), 1)
    print(f"\nmicro precision {micro_p:.2f}  micro recall {micro_r:.2f}")

    failed = [f"{name}: {desc}" for name, ok, desc in REQUIRED if not ok(got_by_name[name])]
    for f in failed:
        print(f"REQUIRED FAILED  {f}")
    passed = not failed and micro_p >= 0.8 and micro_r >= 0.7
    print("PASS" if passed else "FAIL (need all required checks, precision >= 0.80, recall >= 0.70)")


if __name__ == "__main__":
    asyncio.run(main())
