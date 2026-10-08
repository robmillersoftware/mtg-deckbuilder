"""Forge (github.com/Card-Forge/forge) as the game engine: write decks, run headless
AI-vs-AI games in parallel JVMs, parse the game logs. Only parse_games knows Forge's
log format."""

import asyncio
import functools
import logging
import os
import re
import tempfile
import zipfile
from asyncio.subprocess import PIPE, STDOUT
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

TESTED, OPPONENT = "Tested", "Opponent"  # .dck deck names; Forge's log names players after them
SIDES = (TESTED, OPPONENT)
GAMES_PER_PROCESS = 10  # games per JVM; a JVM takes about 4 s to start
GAME_TIMEOUT_S = 60  # Forge calls a slower game a draw
JVM_ARGS = ["-Xmx1g", "-Djava.awt.headless=true", "-Dio.netty.tryReflectionSetAccessible=true",
            "-Dfile.encoding=UTF-8"]


class ForgeError(RuntimeError):
    """Forge is missing, crashed, or took too long."""


def jar() -> Optional[Path]:
    home = Path(settings.FORGE_HOME)
    found = sorted(home.glob("forge-gui-desktop-*-jar-with-dependencies.jar")) if home.is_dir() else []
    return found[-1] if found else None


def available() -> bool:
    return jar() is not None


@functools.lru_cache(maxsize=1)
def card_names() -> Set[str]:
    """Every card name Forge can play, lowercased, each face of a multi-face card."""
    names: Set[str] = set()
    with zipfile.ZipFile(Path(settings.FORGE_HOME) / "res" / "cardsfolder" / "cardsfolder.zip") as z:
        for info in z.infolist():
            if info.filename.endswith(".txt"):
                for line in z.read(info).decode("utf-8", "replace").splitlines():
                    if line.startswith("Name:"):
                        names.add(line[len("Name:"):].strip().lower())
    return names


_FACES = re.compile(r"\s+//?\s+")


def forge_name(name: str, known: Set[str]) -> Optional[str]:
    """Our card name as Forge spells it: the full name, or a split or double-faced
    card's front face; None when Forge lacks the card."""
    for candidate in (name.strip(), _FACES.split(name.strip())[0]):
        if candidate.lower() in known:
            return candidate
    return None


def missing_cards(main: Dict[str, int], known: Set[str]) -> List[str]:
    return sorted(n for n, q in main.items() if q > 0 and forge_name(n, known) is None)


def write_deck(path: Path, deck_name: str, main: Dict[str, int], known: Set[str]) -> None:
    """A Forge .dck of the main deck; cards Forge lacks are left out (see missing_cards)."""
    lines = ["[metadata]", f"Name={deck_name}", "[Main]"]
    for name, qty in main.items():
        spelled = forge_name(name, known)
        if qty > 0 and spelled:
            lines.append(f"{qty} {spelled}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class GameRecord:
    winner: Optional[str]  # TESTED, OPPONENT, or None for a draw
    turns: int  # rounds, as Forge reports them
    own_turns: Dict[str, int]  # turns each side took
    mulligans: Dict[str, int]  # cards each side mulliganed away
    casts: Dict[str, List[Tuple[int, str]]]  # spells cast: (caster's own turn, card)
    lands: Dict[str, List[Tuple[int, str]]]  # lands played: (own turn, card)
    log: List[str] = field(default_factory=list)  # condensed: turns, lands, casts, attacks, life


_SIDE = r"Ai\(\d\)-(?P<side>Tested|Opponent)"
_TURN = re.compile(rf"^Turn: Turn (?P<n>\d+) \({_SIDE}\)")
_CAST = re.compile(rf"^Add To Stack: {_SIDE} cast (?P<card>.+?)(?: targeting \[.*)?$")
_LAND = re.compile(rf"^Land: {_SIDE} played (?P<card>.+?) \(\d+\)$")
_MULLIGAN = re.compile(rf"^Mulligan: {_SIDE} has mulliganed down to (?P<n>\d+) cards")
_ATTACK = re.compile(rf"^Combat: {_SIDE} assigned (?P<what>.+?) to attack")
_LIFE = re.compile(rf"^Life: Life: {_SIDE} (?P<before>-?\d+) > (?P<after>-?\d+)")
_ROUNDS = re.compile(r"^Game Outcome: Turn (?P<n>\d+)")
_RESULT = re.compile(r"^Game Result: Game \d+ ended in (?:a Draw|\d+ ms\. Ai\(\d\)-(?P<side>Tested|Opponent) has won)")


def _blank() -> Dict:
    return {"turns": 0, "own_turns": {s: 0 for s in SIDES}, "mulligans": {s: 0 for s in SIDES},
            "casts": {s: [] for s in SIDES}, "lands": {s: [] for s in SIDES}, "log": []}


def parse_games(text: str) -> List[GameRecord]:
    """The games in a `sim` run's output, in order."""
    games, g = [], _blank()
    for raw in text.splitlines():
        line = raw.rstrip()
        if m := _TURN.match(line):
            g["own_turns"][m["side"]] += 1
            g["log"].append(f"Turn {m['n']}: {m['side']}")
        elif m := _LAND.match(line):
            g["lands"][m["side"]].append((g["own_turns"][m["side"]], m["card"]))
            g["log"].append(f"{m['side']} plays {m['card']}")
        elif m := _CAST.match(line):
            g["casts"][m["side"]].append((g["own_turns"][m["side"]], m["card"]))
            g["log"].append(f"{m['side']} casts {m['card']}")
        elif m := _MULLIGAN.match(line):
            g["mulligans"][m["side"]] = max(g["mulligans"][m["side"]], 7 - int(m["n"]))
        elif m := _ATTACK.match(line):
            attack_str = re.sub(r' \(\d+\)', '', m['what'])
            g["log"].append(f"{m['side']} attacks with {attack_str}")
        elif m := _LIFE.match(line):
            g["log"].append(f"{m['side']} life {m['before']} → {m['after']}")
        elif m := _ROUNDS.match(line):
            g["turns"] = int(m["n"])
        elif m := _RESULT.match(line):
            games.append(GameRecord(winner=m["side"], **g))
            g = _blank()
    return games


@dataclass
class Matchup:
    key: str  # the caller's name for this opponent
    opponent: Dict[str, int]  # its main deck


def workers() -> int:
    return settings.FORGE_WORKERS or max(1, min(6, (os.cpu_count() or 4) - 2))


def command(deck_dir: Path, games: int, seed: int) -> List[str]:
    path = jar()
    if path is None:
        raise ForgeError(f"Forge isn't installed in {settings.FORGE_HOME}")
    return ["java", *JVM_ARGS, "-jar", str(path), "sim", "-D", str(deck_dir), "-d", "tested.dck", "opponent.dck",
            "-n", str(games), "-f", "constructed", "-s", str(seed), "-c", str(GAME_TIMEOUT_S)]


async def _run(cmd: List[str], timeout: float) -> str:
    home = settings.FORGE_HOME if Path(settings.FORGE_HOME).is_dir() else None  # Forge reads res/ from its home
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=home, stdout=PIPE, stderr=STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ForgeError(f"Forge took longer than {timeout:.0f} s")
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    text = out.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise ForgeError(f"Forge exited with code {proc.returncode}: {text[-500:].strip()}")
    return text  # ponytail: always return text; tail is checked by caller if needed


OnProgress = Optional[Callable[[str, List[GameRecord]], Awaitable[None]]]


async def play(tested: Dict[str, int], matchups: Sequence[Matchup], games: int, seed: int,
               on_progress: OnProgress = None) -> Dict[str, List[GameRecord]]:
    """`games` games of `tested` against each matchup, split into GAMES_PER_PROCESS
    chunks on up to workers() concurrent JVMs. Chunk i of every matchup uses seed + i,
    so two decks played with the same seed see paired shuffles as far as Forge's RNG
    allows. `on_progress(key, records)` runs after each chunk; if it raises, the other
    chunks are cancelled and the exception propagates."""
    known = card_names()
    results: Dict[str, List[GameRecord]] = {m.key: [] for m in matchups}
    limit = asyncio.Semaphore(workers())
    with tempfile.TemporaryDirectory(prefix="forge-") as tmp:
        jobs = []
        for i, matchup in enumerate(matchups):
            deck_dir = Path(tmp) / str(i)
            deck_dir.mkdir()
            write_deck(deck_dir / "tested.dck", TESTED, tested, known)
            write_deck(deck_dir / "opponent.dck", OPPONENT, matchup.opponent, known)
            for chunk, start in enumerate(range(0, games, GAMES_PER_PROCESS)):
                jobs.append((matchup.key, deck_dir, min(GAMES_PER_PROCESS, games - start), seed + chunk))

        async def run(key: str, deck_dir: Path, n: int, chunk_seed: int) -> None:
            async with limit:
                output = await _run(command(deck_dir, n, chunk_seed), n * GAME_TIMEOUT_S + 120)
                records = parse_games(output)
            if not records:
                raise ForgeError(f"Forge played no games: {output[-500:].strip()}")
            results[key].extend(records)
            if on_progress:
                await on_progress(key, records)

        try:
            async with asyncio.TaskGroup() as group:
                for job in jobs:
                    group.create_task(run(*job))
        except ExceptionGroup as eg:
            raise eg.exceptions[0] from None
    return results
