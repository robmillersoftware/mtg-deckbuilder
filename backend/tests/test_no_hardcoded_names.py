"""The metagame changes every week, so deck assembly code names no card, archetype
or land: everything comes from recent decklists. Basic land names are rules of
the game, not the metagame, and are allowed."""

import ast
import re
from pathlib import Path

from app.services import deck_fill as df
from tests import test_deck_fill as tdf
from tests import test_deck_plan as tdp

PRODUCTION = [Path(df.__file__), Path(df.__file__).with_name("deck_plan.py")]

FIXTURE_NAMES = (
    {r.name for r in tdp.LISTS + tdp.RELATIVE_LISTS} | {r.name for r in tdp.CANDIDATE_ROWS}
    | {r.archetype for r in tdp.CANDIDATE_ROWS} | {a for a, _ in tdp.ARCHETYPES}
    | {c.name for c in tdf.CARDS + tdf.MAIN_POOL + tdf.LANDS} | set(tdf.LAND_TEXTS)
    | {tdf.reference_plan().reference, "Rakdos Aggro", "UW Control", "Candy Trail"}
) - set(df.BASICS.values())


def literals(path: Path) -> list:
    """Every string literal in the file, f-string parts and docstrings included."""
    return [n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def named(text: str) -> set:
    return {n for n in FIXTURE_NAMES if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text, re.IGNORECASE)}


def test_fixture_names_are_collected():
    assert {"Hired Claw", "Boros Aggro", "Sacred Foundry", "Rakdos Aggro"} <= FIXTURE_NAMES
    assert "Mountain" not in FIXTURE_NAMES
    assert named('x = "Boros aggro lists"') == {"Boros Aggro"}  # the check itself works


def test_assembly_code_names_no_card_archetype_or_land():
    found = {f"{path.name}: {name}" for path in PRODUCTION for lit in literals(path) for name in named(lit)}
    assert not found
