"""
Run hand-labeled deck/card cases through real Jev and report agreement.

Usage (needs TYPESAFE_API_KEY in .env):  cd backend && python3 scripts/eval_fit.py

Labels:
  good - should fit: average fit >= 0.5 and anti_synergy < ANTI_SYNERGY_CUTOFF
  bad  - shouldn't fit: average fit < LOW_FIT_CUTOFF
  anti - works against the plan: anti_synergy >= ANTI_SYNERGY_CUTOFF
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import deck_fit  # noqa: E402
from app.services.deck_fit import DeckIdentity  # noqa: E402


def c(name, type_line, text, cost=""):
    return {"name": name, "mana_cost": cost, "type_line": type_line, "oracle_text": text}


GRAVEYARD = (
    DeckIdentity(tags=["graveyard"], key_cards=["Doom Whisperer", "Gisa's Bidding"],
                 request_text="Self-mill zombies that recur from the graveyard"),
    [c("Doom Whisperer", "Creature — Nightmare Demon", "Flying, trample\nPay 2 life: Surveil 2.", "{3}{B}{B}"),
     c("Gisa's Bidding", "Sorcery", "Create two 2/2 black Zombie creature tokens.\nMadness {2}{B}", "{2}{B}")],
)
TOKENS = (
    DeckIdentity(tags=["tokens", "aggro"], key_cards=["Intangible Virtue"],
                 request_text="Go-wide white tokens"),
    [c("Intangible Virtue", "Enchantment", "Creature tokens you control get +1/+1 and have vigilance.", "{1}{W}")],
)
CONTROL = (
    DeckIdentity(tags=["control"], key_cards=[], request_text="Blue-black draw-go control"),
    [],
)

CASES = [
    (GRAVEYARD, c("Stitcher's Supplier", "Creature — Zombie",
                  "When Stitcher's Supplier enters the battlefield or dies, mill three cards.", "{B}"), "good"),
    (GRAVEYARD, c("Rest in Peace", "Enchantment",
                  "When Rest in Peace enters the battlefield, exile all graveyards.\nIf a card or token would be put into a graveyard from anywhere, exile it instead.", "{1}{W}"), "anti"),
    (GRAVEYARD, c("Lazotep Reaver", "Creature — Zombie Beast",
                  "When Lazotep Reaver enters the battlefield, amass 1.", "{1}{B}"), "good"),
    (GRAVEYARD, c("Healing Salve", "Instant", "You gain 3 life.", "{W}"), "bad"),
    (GRAVEYARD, c("Cling to Dust", "Instant",
                  "Exile target card from a graveyard. If it was a creature card, you gain 3 life. Otherwise, scry 1.", "{B}"), "bad"),
    (TOKENS, c("Raise the Alarm", "Instant", "Create two 1/1 white Soldier creature tokens.", "{1}{W}"), "good"),
    (TOKENS, c("Anointed Procession", "Enchantment",
               "If an effect would create one or more tokens under your control, it creates twice that many of those tokens instead.", "{3}"), "good"),
    (TOKENS, c("Massacre Wurm", "Creature — Phyrexian Wurm",
               "When Massacre Wurm enters the battlefield, creatures your opponents control get -2/-2 until end of turn.\nWhenever a creature an opponent controls dies, that player loses 2 life.", "{3}{B}{B}{B}"), "bad"),
    (TOKENS, c("Wrath of God", "Sorcery", "Destroy all creatures. They can't be regenerated.", "{2}{W}{W}"), "anti"),
    (TOKENS, c("Divination", "Sorcery", "Draw two cards.", "{2}{U}"), "bad"),
    (CONTROL, c("Counterspell", "Instant", "Counter target spell.", "{U}{U}"), "good"),
    (CONTROL, c("Memory Lapse", "Instant",
                "Counter target spell. If that spell is countered this way, put it on top of its owner's library instead of into that player's graveyard.", "{1}{U}"), "good"),
    (CONTROL, c("Grizzly Bears", "Creature — Bear", "", "{1}{G}"), "bad"),
    (CONTROL, c("Goblin Guide", "Creature — Goblin Scout",
                "Haste\nWhenever Goblin Guide attacks, defending player reveals the top card of their library. If it's a land card, that player puts it into their hand.", "{R}"), "bad"),
    (CONTROL, c("Fact or Fiction", "Instant",
                "Reveal the top five cards of your library. An opponent separates those cards into two piles. Put one pile into your hand and the other into your graveyard.", "{3}{U}"), "good"),
]


def verdict(f: deck_fit.FitScore) -> str:
    avg = f.plan_fit if f.synergy is None else (f.plan_fit + f.synergy) / 2
    if f.anti_synergy >= deck_fit.ANTI_SYNERGY_CUTOFF:
        return "anti"
    if avg < deck_fit.LOW_FIT_CUTOFF:
        return "bad"
    return "good" if avg >= 0.5 else "unsure"


async def main() -> None:
    agree = 0
    print(f"{'card':24} {'label':5} {'got':6} plan   syn    anti")
    for (identity, keys), cand, label in CASES:
        fit = await deck_fit.score_fit(identity, keys, [cand])
        if not fit:
            print(f"{cand['name']:24} {label:5} ERROR (no key or Jev unavailable)")
            continue
        f = fit[cand["name"]]
        got = verdict(f)
        agree += got == label
        syn = "  -  " if f.synergy is None else f"{f.synergy:.2f}"
        print(f"{cand['name']:24} {label:5} {got:6} {f.plan_fit:.2f}   {syn}   {f.anti_synergy:.2f}")
    print(f"\nagreement: {agree}/{len(CASES)}")


if __name__ == "__main__":
    asyncio.run(main())
