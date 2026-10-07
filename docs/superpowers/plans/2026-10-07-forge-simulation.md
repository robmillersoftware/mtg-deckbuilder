# Forge Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LLM-narrated simulator with Forge, a real rules engine: test any deck against the meta, and playtest every deck built from chat by swapping cards and keeping the swaps that win more, with live progress and an actionable report.

**Architecture:** A new `sim-worker` container (Python + Java + Forge) consumes an RQ queue `spellbook_sim`. `forge.py` writes `.dck` files, runs Forge's headless `sim` mode in parallel JVMs and parses the logs into `GameRecord`s. `sim_stats.py` turns records into win rates (Wilson intervals, meta-share weighting) and card/mana evidence. A run (`simulation_runs`, replaced) is either a `test` (play a deck against a gauntlet) or a `build` (`deck_search.playtest` hill-climbs swaps). Progress is written to the run row and polled by the frontend every 2 s.

**Tech Stack:** FastAPI, SQLAlchemy async, Alembic, RQ/Redis, Forge 2.0.16 daily snapshot (Java 17+), Jev (`typesafe_sdk`), React 18 + TypeScript + @tanstack/react-query v5 + zustand.

**Spec:** `docs/superpowers/specs/2026-10-06-forge-simulation-design.md`

## Global Constraints

- No hardcoded card, archetype or land names in production code. Names come from decklists, rules text and Forge's card list. (Test fixtures may name cards.)
- 60-card constructed only (`deck_fill.SIXTY_CARD_FORMATS`); Commander, cEDH and multiplayer are out of scope.
- Best-of-one, no sideboarding; sideboards are never simulated.
- Gauntlet: top 5 archetypes by meta share, each its most recent best-placing list from the last 14 days (`deck_plan.RECENT`).
- Win rates are reported with 95% Wilson intervals; draws count half a win.
- Favored above 55%, unfavored below 45%, even otherwise.
- A Forge failure never blocks a build: the user keeps the assembled deck, told why it is untested.
- Every report carries the one-line limits sentence verbatim: "Played by Forge's AI, which plays straightforwardly; decks that rely on tricky play may do better in real games."
- Build search: 20 screening games per matchup, 50 confirmation games, at most 6 rounds, at most 6 candidates per round, 6-minute budget.
- Never cut: a requested card; a card Forge cannot play; a card in more than half of the reference's or relatives' recent lists; a synergy-slot card when the cut leaves the slot under 8 copies.
- Do not recreate or restart existing containers. Bring up only the new service with `docker compose up -d --no-deps sim-worker`. Never use `docker compose run` (it recreates Redis on a port another project uses).
- Commits end with the session's attribution lines (the controller supplies them).

## Review Focus

1. **A deck containing a card Forge lacks.** Expected: the card is left out of the `.dck`, named in the report under "not simulated", announced as a progress event, and never chosen as a cut. Tests: Task 2 (`write_deck` skips it), Task 5 (`run_test` event and report), Task 7 (`run_build` protects it).
2. **Forge crashes, exits non-zero or hangs.** Expected: `ForgeError`, the run is `failed` with a plain message, and the chat build keeps the untested deck. Tests: Task 2 (non-zero exit), Task 5 (`execute` marks the run failed with the message).
3. **The user presses Stop mid-run.** Expected: a test reports the games played so far with status `stopped`; a build keeps the best deck so far. Tests: Task 5 (`run_test` stop), Task 6 (`playtest` stop during baseline and between rounds).
4. **The sim worker is not running.** Expected: `POST /api/simulations` answers 503 with a plain sentence; a chat build returns the untested deck with a note instead of a run that sits queued forever. Tests: Task 5 (API 503), Task 7 (`start_build` note).
5. **Double-faced and split cards.** Expected: Forge logs a DFC by its front face; card stats map it back to the deck's full name, so it is neither "never cast" nor a phantom card. Tests: Task 3 (`card_stats` with a `//` name), Task 2 (`forge_name` with `//` and `/`).

---

### Task 1: Forge runtime: the sim-worker container and its queue

**Files:**
- Create: `backend/Dockerfile.sim`
- Create: `scripts/update_forge.sh`
- Create: `backend/tests/fixtures/forge/three_games.log`, `backend/tests/fixtures/forge/tested.dck`, `backend/tests/fixtures/forge/opponent.dck` (copied from the spike)
- Create: `backend/tests/test_sim_queue.py`
- Modify: `docker-compose.yml` (new `sim-worker` service)
- Modify: `backend/app/core/config.py` (Forge settings)
- Modify: `backend/app/core/queue.py` (`sim_queue`, `sim_worker_running`)
- Modify: `.env.example`

**Interfaces:**
- Produces: `settings.FORGE_HOME: str = "/opt/forge"`, `settings.FORGE_WORKERS: int = 0`, `settings.FORGE_ENABLED: bool = True`; `app.core.queue.SIM_QUEUE = "spellbook_sim"`, `app.core.queue.sim_queue: rq.Queue`, `app.core.queue.sim_worker_running() -> bool`.

- [ ] **Step 1: Copy the spike fixtures into the repo**

```bash
cd /Users/robmiller/Projects/mtg-deckbuilder
SPIKE=/private/tmp/claude-501/-Users-robmiller-Projects-mtg-deckbuilder/b8847f7b-d5b1-408b-9720-62bba18fb355/scratchpad/forge/fx
mkdir -p backend/tests/fixtures/forge
cp $SPIKE/three_games.log $SPIKE/tested.dck $SPIKE/opponent.dck backend/tests/fixtures/forge/
wc -l backend/tests/fixtures/forge/three_games.log   # expect 987
```

If the spike directory is gone, regenerate the fixtures in Step 8 (inside the container) with the same commands and decks, then adjust no assertions: Task 2's expected values come from this exact log, so regenerate only if missing and re-derive the values with Task 2's parser.

- [ ] **Step 2: Write the failing queue test**

Create `backend/tests/test_sim_queue.py`:

```python
"""The simulation queue has its own worker; the API needs to know whether it is up."""

from types import SimpleNamespace

from app.core import queue as q


def test_sim_worker_running_only_counts_sim_queue_workers(monkeypatch):
    monkeypatch.setattr(q.Worker, "all", lambda connection: [SimpleNamespace(queue_names=lambda: ["spellbook_jobs"])])
    assert not q.sim_worker_running()
    monkeypatch.setattr(q.Worker, "all", lambda connection: [SimpleNamespace(queue_names=lambda: ["spellbook_sim"])])
    assert q.sim_worker_running()


def test_sim_worker_running_is_false_when_redis_fails(monkeypatch):
    def down(connection):
        raise ConnectionError("redis down")
    monkeypatch.setattr(q.Worker, "all", down)
    assert not q.sim_worker_running()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_sim_queue.py -q`
Expected: FAIL (`AttributeError: module 'app.core.queue' has no attribute 'Worker'` or `sim_worker_running`).

- [ ] **Step 4: Add the settings and the queue**

In `backend/app/core/config.py`, after the `ANSWER_MODEL` setting, add:

```python
    # Forge game simulation (the sim-worker container)
    FORGE_HOME: str = "/opt/forge"
    FORGE_WORKERS: int = 0  # concurrent Forge JVMs; 0 picks CPU count - 2, at most 6
    FORGE_ENABLED: bool = True  # playtest decks built from chat when the sim worker is running
```

In `backend/app/core/queue.py`, change `from rq import Queue` to `from rq import Queue, Worker`, and after `high_priority_queue` add:

```python
# Forge simulations run on their own worker: the sim-worker container has Java and Forge
SIM_QUEUE = "spellbook_sim"
sim_queue = Queue(SIM_QUEUE, connection=redis_conn)


def sim_worker_running() -> bool:
    """Whether any worker listens on the simulation queue."""
    try:
        return any(SIM_QUEUE in w.queue_names() for w in Worker.all(connection=redis_conn))
    except Exception:
        return False
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd backend && python3 -m pytest tests/test_sim_queue.py -q`
Expected: PASS (2 passed). Then the full suite: `python3 -m pytest -q` → all pass.

- [ ] **Step 6: Add the image, the service and the update script**

Create `backend/Dockerfile.sim`:

```dockerfile
# Simulation worker: the backend plus Java and Forge (github.com/Card-Forge/forge)
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    default-jre-headless \
    curl \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# A Forge release tarball; scripts/update_forge.sh sets the newest daily snapshot in .env
ARG FORGE_SNAPSHOT_URL
RUN test -n "$FORGE_SNAPSHOT_URL" || (echo "Set FORGE_SNAPSHOT_URL: run scripts/update_forge.sh" && exit 1)
RUN mkdir -p /opt/forge && curl -fsSL "$FORGE_SNAPSHOT_URL" | tar xj -C /opt/forge

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "rq_worker.py", "--queues", "spellbook_sim"]
```

In `docker-compose.yml`, add after the `rq-worker` service (same indentation). It has no `platform:` pin on purpose: Java under amd64 emulation on Apple Silicon is several times slower.

```yaml
  # Forge simulation worker: plays AI-vs-AI games for deck tests and build playtests
  sim-worker:
    build:
      context: ./backend
      dockerfile: Dockerfile.sim
      args:
        FORGE_SNAPSHOT_URL: ${FORGE_SNAPSHOT_URL:-}
    container_name: spellbook-sim-worker
    environment:
      DATABASE_URL: postgresql+asyncpg://spellbook:spellbook@db:5432/spellbook
      DATABASE_URL_SYNC: postgresql://spellbook:spellbook@db:5432/spellbook
      REDIS_URL: redis://redis:6379/0
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:-}
      LLM_MODEL: ${LLM_MODEL:-deepseek/deepseek-v4-flash}
      TYPESAFE_API_KEY: ${TYPESAFE_API_KEY:-}
      FORGE_HOME: /opt/forge
      FORGE_WORKERS: ${FORGE_WORKERS:-0}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./backend:/app
    command: python rq_worker.py --queues spellbook_sim
    restart: unless-stopped
```

Create `scripts/update_forge.sh` and `chmod +x` it:

```bash
#!/usr/bin/env bash
# Point the sim worker at Forge's newest daily snapshot (new sets get card scripts
# there first), then rebuild and restart only the sim worker.
set -euo pipefail
cd "$(dirname "$0")/.."

url=$(curl -fsSL https://api.github.com/repos/Card-Forge/forge/releases/tags/daily-snapshots |
  python3 -c 'import json,sys; print(next(a["browser_download_url"] for a in json.load(sys.stdin)["assets"] if a["name"].endswith(".tar.bz2")))')
echo "Forge snapshot: $url"

touch .env
if grep -q '^FORGE_SNAPSHOT_URL=' .env; then
  sed -i.bak "s#^FORGE_SNAPSHOT_URL=.*#FORGE_SNAPSHOT_URL=$url#" .env && rm -f .env.bak
else
  echo "FORGE_SNAPSHOT_URL=$url" >> .env
fi

docker compose build sim-worker
docker compose up -d --no-deps sim-worker
```

In `.env.example`, after the `ANSWER_MODEL` line, add:

```
# Forge simulation worker (set by scripts/update_forge.sh)
FORGE_SNAPSHOT_URL=
FORGE_WORKERS=0
```

- [ ] **Step 7: Build and start the sim worker**

```bash
cd /Users/robmiller/Projects/mtg-deckbuilder
env -u OPENAI_API_KEY ./scripts/update_forge.sh
docker ps --filter name=spellbook-sim-worker --format '{{.Names}} {{.Status}}'
docker logs spellbook-sim-worker 2>&1 | grep -i "listening on queues"
docker exec spellbook-sim-worker java -version 2>&1 | head -1
docker exec spellbook-sim-worker uname -m
```

Expected: the container is `Up`; the log shows `Listening on queues: ['spellbook_sim']`; Java 17 or newer; `uname -m` is `aarch64` on this Mac (native, no emulation).

- [ ] **Step 8: Measure speed inside the container**

```bash
docker exec spellbook-sim-worker sh -c '
cd /opt/forge
J=$(ls forge-gui-desktop-*-jar-with-dependencies.jar)
start=$(date +%s)
for i in 1 2 3 4; do
  java -Xmx1g -Djava.awt.headless=true -Dio.netty.tryReflectionSetAccessible=true -Dfile.encoding=UTF-8 \
    -jar $J sim -D /app/tests/fixtures/forge -d tested.dck opponent.dck -n 10 -f constructed -q -s $i -c 60 \
    > /tmp/smoke$i.log 2>&1 &
done
wait
echo "40 games in $(( $(date +%s) - start ))s"
grep -h "Match Result" /tmp/smoke*.log | tail -4
nproc; free -m | head -2'
```

Expected: 4 `Match Result` lines summing to 10 games each; 40 games in roughly 30-60 s. Record the time, `nproc` and memory in your report. If 4 JVMs exhaust memory (OOM kills, exit 137), set `FORGE_WORKERS` in `.env` to what fits and say so in the report.

- [ ] **Step 9: Commit**

```bash
git add backend/Dockerfile.sim scripts/update_forge.sh docker-compose.yml .env.example \
  backend/app/core/config.py backend/app/core/queue.py backend/tests/test_sim_queue.py \
  backend/tests/fixtures/forge
git commit -m "Add the Forge sim worker and its queue"
```

---

### Task 2: Forge engine: decks, games, logs

**Files:**
- Create: `backend/app/services/forge.py`
- Test: `backend/tests/test_forge.py`

**Interfaces:**
- Consumes: `settings.FORGE_HOME`, `settings.FORGE_WORKERS` (Task 1); fixtures in `backend/tests/fixtures/forge/` (Task 1).
- Produces (later tasks rely on these exact names):
  - `TESTED = "Tested"`, `OPPONENT = "Opponent"`, `SIDES`
  - `class ForgeError(RuntimeError)`
  - `available() -> bool`, `card_names() -> Set[str]` (lru-cached; lowercased names of every face)
  - `forge_name(name: str, known: Set[str]) -> Optional[str]`, `missing_cards(main: Dict[str, int], known) -> List[str]`, `write_deck(path, deck_name, main, known) -> None`
  - `@dataclass GameRecord(winner: Optional[str], turns: int, own_turns: Dict[str, int], mulligans: Dict[str, int], casts: Dict[str, List[Tuple[int, str]]], lands: Dict[str, List[Tuple[int, str]]], log: List[str])`; `turns` is Forge's round count; casts/lands carry the caster's own turn number
  - `parse_games(text: str) -> List[GameRecord]`
  - `@dataclass Matchup(key: str, opponent: Dict[str, int])`
  - `async play(tested: Dict[str, int], matchups: Sequence[Matchup], games: int, seed: int, on_progress=None) -> Dict[str, List[GameRecord]]`; `on_progress(key, records)` is awaited after each process; an exception it raises cancels the remaining processes and propagates.
  - `command(deck_dir, games, seed) -> List[str]` (patched in tests)

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_forge.py`:

```python
"""Forge engine: card names, .dck files, log parsing, and running games in parallel."""

import sys
import zipfile
from pathlib import Path

import pytest

from app.core.config import settings
from app.services import forge

FIXTURES = Path(__file__).parent / "fixtures" / "forge"
LOG = (FIXTURES / "three_games.log").read_text()
KNOWN = {"forest", "sazh's chocobo", "esper origins", "summon: esper maduin", "unholy annex", "ritual chamber"}


class TestParseGames:
    def test_each_game_from_a_real_log(self):
        games = forge.parse_games(LOG)
        assert [g.winner for g in games] == ["Tested", "Opponent", "Opponent"]
        assert [g.turns for g in games] == [8, 6, 6]
        assert games[0].own_turns == {"Tested": 8, "Opponent": 7}
        assert games[0].casts["Tested"][:3] == [(1, "Sazh's Chocobo"), (2, "Shared Roots"), (3, "Esper Origins")]
        assert games[0].lands["Tested"][:3] == [(1, "Forest"), (2, "Ba Sing Se"), (3, "Escape Tunnel")]
        assert games[0].log[:3] == ["Turn 1: Tested", "Tested plays Forest", "Tested casts Sazh's Chocobo"]
        assert "Opponent casts Burst Lightning" in games[0].log  # targets are dropped
        assert "Tested attacks with Sazh's Chocobo" in games[0].log
        assert "Opponent life 20 → 18" in games[0].log

    def test_mulligans_draws_and_negative_life(self):
        text = "\n".join([
            "Mulligan: Ai(1)-Tested has mulliganed down to 6 cards.",
            "Mulligan: Ai(1)-Tested has mulliganed down to 5 cards.",
            "Life: Life: Ai(2)-Opponent 2 > -1",
            "Game Outcome: Turn 30",
            "Stopping slow match as draw",
            "Game Result: Game 1 ended in a Draw! Took 60000 ms.",
        ])
        (game,) = forge.parse_games(text)
        assert game.winner is None and game.turns == 30
        assert game.mulligans == {"Tested": 2, "Opponent": 0}
        assert game.log == ["Opponent life 2 → -1"]

    def test_no_result_line_means_no_game(self):
        assert forge.parse_games("Turn: Turn 1 (Ai(1)-Tested)\n") == []


class TestNames:
    def test_forge_name_full_or_front_face(self):
        assert forge.forge_name("Forest", KNOWN) == "Forest"
        assert forge.forge_name("Esper Origins // Summon: Esper Maduin", KNOWN) == "Esper Origins"
        assert forge.forge_name("Unholy Annex / Ritual Chamber", KNOWN) == "Unholy Annex"
        assert forge.forge_name("Made Up Card", KNOWN) is None

    def test_write_deck_leaves_out_unknown_cards(self, tmp_path):
        path = tmp_path / "t.dck"
        forge.write_deck(path, "Tested", {"Forest": 20, "Esper Origins // Summon: Esper Maduin": 4,
                                          "Made Up Card": 4, "Sazh's Chocobo": 0}, KNOWN)
        assert path.read_text() == "[metadata]\nName=Tested\n[Main]\n20 Forest\n4 Esper Origins\n"
        assert forge.missing_cards({"Forest": 1, "Made Up Card": 4}, KNOWN) == ["Made Up Card"]

    def test_card_names_reads_every_face(self, tmp_path, monkeypatch):
        folder = tmp_path / "res" / "cardsfolder"
        folder.mkdir(parents=True)
        with zipfile.ZipFile(folder / "cardsfolder.zip", "w") as z:
            z.writestr("e/esper_origins.txt", "Name:Esper Origins\nManaCost:1 U\nALTERNATE\nName:Summon: Esper Maduin\n")
            z.writestr("f/forest.txt", "Name:Forest\r\nTypes:Basic Land Forest\r\n")
        monkeypatch.setattr(settings, "FORGE_HOME", str(tmp_path))
        forge.card_names.cache_clear()
        try:
            assert forge.card_names() == {"esper origins", "summon: esper maduin", "forest"}
        finally:
            forge.card_names.cache_clear()

    def test_command_needs_forge_installed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "FORGE_HOME", str(tmp_path))
        assert not forge.available()
        with pytest.raises(forge.ForgeError, match="isn't installed"):
            forge.command(tmp_path, 1, 1)


def replay_fixture(calls):
    def command(deck_dir, games, seed):
        tested = (Path(deck_dir) / "tested.dck").read_text().splitlines()
        calls.append((Path(deck_dir).name, games, seed, tested[1]))
        return [sys.executable, "-c", f"import sys; sys.stdout.write(open({str(FIXTURES / 'three_games.log')!r}).read())"]
    return command


class TestPlay:
    async def test_splits_games_into_processes_with_consecutive_seeds(self, monkeypatch):
        calls, seen = [], []
        monkeypatch.setattr(forge, "command", replay_fixture(calls))
        monkeypatch.setattr(forge, "card_names", lambda: KNOWN)
        monkeypatch.setattr(forge, "GAMES_PER_PROCESS", 10)

        async def on_progress(key, records):
            seen.append((key, len(records)))

        out = await forge.play({"Forest": 60}, [forge.Matchup("A", {"Forest": 60}), forge.Matchup("B", {"Forest": 60})],
                               games=25, seed=7, on_progress=on_progress)
        assert sorted((d, n, s) for d, n, s, _ in calls) == [
            ("0", 10, 7), ("0", 10, 8), ("0", 5, 9), ("1", 10, 7), ("1", 10, 8), ("1", 5, 9)]
        assert {name for *_, name in calls} == {"Name=Tested"}
        assert len(out["A"]) == len(out["B"]) == 9  # the fixture holds 3 games per process
        assert sorted(seen) == [("A", 3)] * 3 + [("B", 3)] * 3

    async def test_a_failed_process_raises_forge_error(self, monkeypatch):
        monkeypatch.setattr(forge, "command", lambda d, g, s: [sys.executable, "-c", "print('boom'); raise SystemExit(3)"])
        monkeypatch.setattr(forge, "card_names", lambda: KNOWN)
        with pytest.raises(forge.ForgeError, match="boom"):
            await forge.play({"Forest": 60}, [forge.Matchup("A", {"Forest": 60})], games=2, seed=1)

    async def test_an_error_in_on_progress_stops_the_rest(self, monkeypatch):
        calls = []
        monkeypatch.setattr(forge, "command", replay_fixture(calls))
        monkeypatch.setattr(forge, "card_names", lambda: KNOWN)
        monkeypatch.setattr(forge, "GAMES_PER_PROCESS", 1)
        monkeypatch.setattr(forge, "workers", lambda: 1)

        class Stop(Exception):
            pass

        async def stop(key, records):
            raise Stop()

        with pytest.raises(Stop):
            await forge.play({"Forest": 60}, [forge.Matchup("A", {"Forest": 60})], games=5, seed=1, on_progress=stop)
        assert len(calls) < 5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_forge.py -q`
Expected: FAIL with `ImportError: cannot import name 'forge'`.

- [ ] **Step 3: Write the engine**

Create `backend/app/services/forge.py`:

```python
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
            g["log"].append(f"{m['side']} attacks with {re.sub(r' \(\d+\)', '', m['what'])}")
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
    return text


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
                records = parse_games(await _run(command(deck_dir, n, chunk_seed), n * GAME_TIMEOUT_S + 120))
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_forge.py -q` → all pass. Then `python3 -m pytest -q` → all pass.

- [ ] **Step 5: Check against real Forge in the sim worker**

```bash
docker exec -w /app spellbook-sim-worker python -c "
import asyncio, time
from app.services import forge
from pathlib import Path
deck = lambda p: {l.split(' ', 1)[1]: int(l.split(' ', 1)[0]) for l in Path(p).read_text().splitlines()
                  if l[:1].isdigit() and '[Sideboard]' not in l}
main = Path('tests/fixtures/forge/tested.dck').read_text().split('[Sideboard]')[0]
opp = Path('tests/fixtures/forge/opponent.dck').read_text().split('[Sideboard]')[0]
parse = lambda t: {l.split(' ', 1)[1]: int(l.split(' ', 1)[0]) for l in t.splitlines() if l[:1].isdigit()}
print('missing:', forge.missing_cards(parse(main), forge.card_names()))
t = time.time()
out = asyncio.run(forge.play(parse(main), [forge.Matchup('opp', parse(opp))], games=20, seed=1))
rs = out['opp']
print(len(rs), 'games in', round(time.time() - t), 's; tested won', sum(r.winner == 'Tested' for r in rs))
print(rs[0].casts['Tested'][:5])
"
```

Expected: `missing: []`, `20 games in` N s (record N), a plausible win count, and real card names in the casts.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/forge.py backend/tests/test_forge.py
git commit -m "Run Forge games in parallel and parse their logs"
```

---

### Task 3: Win rates, card evidence and mana evidence

**Files:**
- Create: `backend/app/services/sim_stats.py`
- Test: `backend/tests/test_sim_stats.py`

**Interfaces:**
- Consumes: `forge.GameRecord`, `forge.TESTED`, `forge.OPPONENT` (Task 2).
- Produces:
  - `wilson(score: float, n: int) -> Tuple[float, float]`, `label(rate: float) -> str`
  - `@dataclass MatchupStats(opponent: str, share: float, wins: int, losses: int, draws: int, avg_turns: float)` with properties `games`, `win_rate` and `as_dict() -> dict` (keys: opponent, share, wins, losses, draws, games, win_rate, lo, hi, label, avg_turns)
  - `matchup_stats(opponent: str, share: float, records: Sequence[GameRecord]) -> MatchupStats`
  - `@dataclass Overall(win_rate, lo, hi, se, games)` with `as_dict()`; `overall(matchups: Sequence[MatchupStats]) -> Overall` (share-weighted)
  - `@dataclass CardStat(name, copies, games_cast, cast_share, win_rate_when_cast: Optional[float], median_turn: Optional[float])` with `as_dict()`; `card_stats(records, main: Dict[str, int], lands: Set[str]) -> List[CardStat]` (nonland cards only, in deck order)
  - `mana_stats(records) -> Dict[str, float]` (mulligan_rate, screw_rate, flood_rate); `mana_advice(mana) -> List[str]`
  - `pct(x: float) -> str` ("53%")

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_sim_stats.py`:

```python
"""Win rates with uncertainty, and the card and mana evidence behind them."""

import pytest

from app.services.forge import OPPONENT, TESTED, GameRecord
from app.services import sim_stats as ss


def game(winner=TESTED, casts=(), lands=5, own=8, mulligans=0, turns=8):
    return GameRecord(winner=winner, turns=turns, own_turns={TESTED: own, OPPONENT: own},
                      mulligans={TESTED: mulligans, OPPONENT: 0},
                      casts={TESTED: list(casts), OPPONENT: []},
                      lands={TESTED: [(t, "Forest") for t in range(1, lands + 1)], OPPONENT: []})


class TestWinRates:
    def test_wilson_interval(self):
        lo, hi = ss.wilson(10, 20)
        assert lo == pytest.approx(0.299, abs=0.001) and hi == pytest.approx(0.701, abs=0.001)
        assert ss.wilson(0, 0) == (0.0, 1.0)

    def test_draws_count_half(self):
        m = ss.matchup_stats("A", 10.0, [game(TESTED), game(OPPONENT), game(None, turns=30)])
        assert (m.wins, m.losses, m.draws, m.games) == (1, 1, 1, 3)
        assert m.win_rate == pytest.approx(0.5)
        assert m.avg_turns == pytest.approx(46 / 3)
        d = m.as_dict()
        assert d["label"] == "even" and d["opponent"] == "A" and d["games"] == 3 and d["lo"] < 0.5 < d["hi"]

    def test_labels(self):
        assert ss.label(0.56) == "favored" and ss.label(0.55) == "even" and ss.label(0.44) == "unfavored"

    def test_overall_weights_by_meta_share(self):
        a = ss.matchup_stats("A", 30.0, [game(TESTED)] * 8 + [game(OPPONENT)] * 2)
        b = ss.matchup_stats("B", 10.0, [game(TESTED)] * 2 + [game(OPPONENT)] * 8)
        o = ss.overall([a, b])
        assert o.win_rate == pytest.approx(0.75 * 0.8 + 0.25 * 0.2)
        assert o.games == 20 and o.lo < o.win_rate < o.hi and o.se > 0

    def test_overall_with_no_games(self):
        o = ss.overall([ss.matchup_stats("A", 10.0, [])])
        assert (o.win_rate, o.games) == (0.0, 0)


class TestCardStats:
    def test_win_rate_when_cast_maps_front_faces_back(self):
        main = {"Forest": 20, "Bear": 4, "Esper Origins // Summon: Esper Maduin": 4, "Never Cast": 4}
        records = [game(TESTED, [(2, "Bear"), (3, "Esper Origins"), (4, "Bear")]),
                   game(OPPONENT, [(2, "Bear")]),
                   game(TESTED, [(5, "Esper Origins")])]
        stats = {c.name: c for c in ss.card_stats(records, main, lands={"Forest"})}
        assert set(stats) == {"Bear", "Esper Origins // Summon: Esper Maduin", "Never Cast"}  # no lands
        bear = stats["Bear"]
        assert (bear.games_cast, bear.win_rate_when_cast, bear.median_turn) == (2, 0.5, 2)
        assert bear.cast_share == pytest.approx(2 / 3)
        assert stats["Esper Origins // Summon: Esper Maduin"].win_rate_when_cast == 1.0
        assert stats["Never Cast"].games_cast == 0 and stats["Never Cast"].win_rate_when_cast is None


class TestMana:
    def test_screw_flood_and_mulligans(self):
        screwed = game(OPPONENT, lands=2, own=6)
        flooded = game(OPPONENT, lands=8, own=8, casts=[(2, "Bear")])
        short_loss = game(OPPONENT, lands=2, own=3)  # lost before its 5th turn: not counted as screw
        records = [screwed, flooded, short_loss, game(TESTED, mulligans=1)]
        m = ss.mana_stats(records)
        assert m == {"mulligan_rate": 0.25, "screw_rate": 0.25, "flood_rate": 0.25}
        advice = ss.mana_advice(m)
        assert any("3 or fewer lands" in a for a in advice) and any("one fewer land" in a for a in advice)
        assert ss.mana_advice({"mulligan_rate": 0.0, "screw_rate": 0.0, "flood_rate": 0.0}) == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_sim_stats.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the module**

Create `backend/app/services/sim_stats.py`:

```python
"""Win rates with uncertainty (Wilson intervals, meta-share weighting), and the
per-card and mana evidence from Forge games that explains them."""

import math
import re
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from app.services.forge import OPPONENT, TESTED, GameRecord

Z = 1.96  # 95% intervals
FAVORED, UNFAVORED = 0.55, 0.45
SCREW_TURN, SCREW_LANDS = 5, 3  # lost after reaching own turn 5 with 3 or fewer lands played
FLOOD_TURN, FLOOD_LANDS, FLOOD_SPELLS = 7, 7, 4  # lost with 7+ lands and 4 or fewer spells by own turn 7
ADVICE_RATE = 0.15
MULLIGAN_ADVICE_RATE = 0.25


def pct(x: float) -> str:
    return f"{round(100 * x)}%"


def wilson(score: float, n: int) -> Tuple[float, float]:
    """95% interval for a win rate of score/n (draws count half)."""
    if n == 0:
        return (0.0, 1.0)
    p = score / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def label(rate: float) -> str:
    return "favored" if rate > FAVORED else "unfavored" if rate < UNFAVORED else "even"


@dataclass
class MatchupStats:
    opponent: str
    share: float
    wins: int
    losses: int
    draws: int
    avg_turns: float

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def as_dict(self) -> Dict:
        lo, hi = wilson(self.wins + 0.5 * self.draws, self.games)
        return {**asdict(self), "games": self.games, "win_rate": self.win_rate, "lo": lo, "hi": hi,
                "label": label(self.win_rate)}


def matchup_stats(opponent: str, share: float, records: Sequence[GameRecord]) -> MatchupStats:
    return MatchupStats(
        opponent=opponent, share=share,
        wins=sum(r.winner == TESTED for r in records),
        losses=sum(r.winner == OPPONENT for r in records),
        draws=sum(r.winner is None for r in records),
        avg_turns=statistics.fmean(r.turns for r in records) if records else 0.0)


@dataclass
class Overall:
    win_rate: float
    lo: float
    hi: float
    se: float
    games: int

    def as_dict(self) -> Dict:
        return {"win_rate": self.win_rate, "lo": self.lo, "hi": self.hi, "games": self.games}


def overall(matchups: Sequence[MatchupStats]) -> Overall:
    """Win rate against the field: matchups weighted by meta share."""
    played = [m for m in matchups if m.games]
    total = sum(m.share for m in played)
    if not played or total <= 0:
        return Overall(0.0, 0.0, 1.0, 0.5, 0)
    rate = sum(m.share / total * m.win_rate for m in played)
    se = math.sqrt(sum((m.share / total) ** 2 * m.win_rate * (1 - m.win_rate) / m.games for m in played))
    return Overall(rate, max(0.0, rate - Z * se), min(1.0, rate + Z * se), se, sum(m.games for m in played))


@dataclass
class CardStat:
    name: str
    copies: int
    games_cast: int
    cast_share: float  # share of games in which it was cast
    win_rate_when_cast: Optional[float]
    median_turn: Optional[float]  # own turn it was first cast, median

    def as_dict(self) -> Dict:
        return asdict(self)


_FACES = re.compile(r"\s+//?\s+")


def card_stats(records: Sequence[GameRecord], main: Dict[str, int], lands: Set[str]) -> List[CardStat]:
    """Per nonland card of `main`: how often it was cast and how those games went.
    Forge logs a split or double-faced card by its front face."""
    back: Dict[str, str] = {}
    for name in main:
        back[name.lower()] = name
        back.setdefault(_FACES.split(name)[0].lower(), name)
    seen: Dict[str, List[Tuple[Optional[str], int]]] = {n: [] for n in main if n not in lands}
    for record in records:
        first: Dict[str, int] = {}
        for turn, card in record.casts[TESTED]:
            name = back.get(card.lower())
            if name in seen and name not in first:
                first[name] = turn
        for name, turn in first.items():
            seen[name].append((record.winner, turn))
    stats = []
    for name, games in seen.items():
        score = sum(1.0 if w == TESTED else 0.5 if w is None else 0.0 for w, _ in games)
        stats.append(CardStat(
            name=name, copies=main[name], games_cast=len(games),
            cast_share=len(games) / len(records) if records else 0.0,
            win_rate_when_cast=score / len(games) if games else None,
            median_turn=statistics.median(t for _, t in games) if games else None))
    return stats


def _by(entries: Sequence[Tuple[int, str]], turn: int) -> int:
    return sum(1 for t, _ in entries if t <= turn)


def mana_stats(records: Sequence[GameRecord]) -> Dict[str, float]:
    n = len(records) or 1
    screw = flood = 0
    for r in records:
        if r.winner != OPPONENT:
            continue
        if r.own_turns[TESTED] >= SCREW_TURN and _by(r.lands[TESTED], SCREW_TURN) <= SCREW_LANDS:
            screw += 1
        elif (r.own_turns[TESTED] >= FLOOD_TURN and _by(r.lands[TESTED], FLOOD_TURN) >= FLOOD_LANDS
              and _by(r.casts[TESTED], FLOOD_TURN) <= FLOOD_SPELLS):
            flood += 1
    return {"mulligan_rate": sum(r.mulligans[TESTED] > 0 for r in records) / n,
            "screw_rate": screw / n, "flood_rate": flood / n}


def mana_advice(mana: Dict[str, float]) -> List[str]:
    advice = []
    if mana["screw_rate"] > ADVICE_RATE:
        advice.append(f"Lost {pct(mana['screw_rate'])} of games stuck on 3 or fewer lands by turn 5: "
                      "consider one more land.")
    if mana["flood_rate"] > ADVICE_RATE:
        advice.append(f"Lost {pct(mana['flood_rate'])} of games drawing mostly lands: consider one fewer land.")
    if mana["mulligan_rate"] > MULLIGAN_ADVICE_RATE:
        advice.append(f"Mulliganed {pct(mana['mulligan_rate'])} of games: the mana or the curve may be awkward.")
    return advice
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_sim_stats.py -q` → pass; `python3 -m pytest -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/sim_stats.py backend/tests/test_sim_stats.py
git commit -m "Win rates with intervals, card and mana evidence from sim games"
```

---

### Task 4: Replace the old simulator's data with Forge runs and the gauntlet

**Files:**
- Delete: `backend/app/services/game_simulator.py`, `backend/app/api/routes/simulation.py`, `backend/app/schemas/simulation.py`
- Rewrite: `backend/app/models/simulation.py`
- Create: `backend/alembic/versions/017_forge_simulation_runs.py`
- Create: `backend/app/services/gauntlet.py`
- Modify: `backend/app/main.py` (drop the old router and the stale-run cleanup)
- Test: `backend/tests/test_gauntlet.py`

**Interfaces:**
- Consumes: `deck_plan.RECENT`, `app.models.meta.MetaSnapshot`.
- Produces:
  - `SimulationRun` model: `id, user_id (nullable), conversation_id (nullable), kind ('test'|'build'), status ('queued'|'running'|'completed'|'failed'|'stopped'), format, deck (JSONB {name, main_deck, sideboard}), opponents (JSONB list or null), games_per_matchup, options (JSONB), progress (JSONB), report (JSONB), final_deck (JSONB), stop_requested (bool), error (text), created_at, updated_at`.
  - `gauntlet.GAUNTLET_SIZE = 5`; `@dataclass Opponent(archetype: str, share: float, main: Dict[str, int])`; `async gauntlet(db, format: str, archetypes: Optional[Sequence[str]] = None) -> List[Opponent]` (meta: top 5 by share, weighted by share; chosen archetypes: weight 1.0 each; archetypes without a recent list are skipped).

- [ ] **Step 1: Find every user of the old simulator**

Run: `cd backend && grep -rn "game_simulator\|GameSimulator\|schemas.simulation\|routes import simulation\|cleanup_stale_simulations" app tests | grep -v __pycache__`
Expected: `app/main.py` (import list, `cleanup_stale_simulations`, `include_router`), `app/api/routes/simulation.py`, `app/services/game_simulator.py`, `app/models/__init__.py` (model import, which stays). If anything else appears, remove that use too and say so in the report.

- [ ] **Step 2: Write the failing gauntlet tests**

Create `backend/tests/test_gauntlet.py`:

```python
"""The gauntlet: the top meta decks as recent real lists."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.services import gauntlet as g

SNAPSHOTS = [SimpleNamespace(archetype="Dimir Aggro", meta_percentage=10.9),
             SimpleNamespace(archetype="Boros Dragons ", meta_percentage=10.3),
             SimpleNamespace(archetype="No Lists", meta_percentage=9.0)]


def fake_db(lists):
    """execute(): first the snapshot query, then one list query per archetype tried."""
    snap = MagicMock()
    snap.scalars.return_value.all.return_value = SNAPSHOTS

    def result(row):
        r = MagicMock()
        r.first.return_value = None if row is None else SimpleNamespace(main_deck=row)
        return r
    return MagicMock(execute=AsyncMock(side_effect=[snap] + [result(x) for x in lists]))


async def test_meta_gauntlet_uses_shares_and_skips_archetypes_without_lists(monkeypatch):
    monkeypatch.setattr(g, "GAUNTLET_SIZE", 2)
    db = fake_db([[{"card_name": "Island", "quantity": 24}], [{"card_name": "Mountain", "quantity": "22"}]])
    got = await g.gauntlet(db, "standard")
    assert got == [g.Opponent("Dimir Aggro", 10.9, {"Island": 24}), g.Opponent("Boros Dragons", 10.3, {"Mountain": 22})]
    list_sql, params = db.execute.call_args_list[1].args
    assert params == {"format": "standard", "archetype": "Dimir Aggro"}
    assert "lower(trim(d.archetype)) = lower(trim(:archetype))" in str(list_sql)


async def test_chosen_archetypes_weigh_equally_and_skip_missing():
    db = fake_db([None, [{"card_name": "Forest", "quantity": 20}]])
    got = await g.gauntlet(db, "standard", ["No Lists", "Mono Green"])
    assert got == [g.Opponent("Mono Green", 1.0, {"Forest": 20})]
```

- [ ] **Step 3: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_gauntlet.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 4: Write the gauntlet**

Create `backend/app/services/gauntlet.py`:

```python
"""The opponents a deck is tested against: the top meta archetypes (or chosen ones),
each as its most recent best-placing list from the recent window."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meta import MetaSnapshot
from app.services.deck_plan import RECENT

GAUNTLET_SIZE = 5

LIST_SQL = text(f"""
    SELECT d.main_deck
    FROM decklists d JOIN events e ON e.id = d.event_id
    WHERE {RECENT} AND lower(trim(d.archetype)) = lower(trim(:archetype))
    ORDER BY e.date DESC, d.placement NULLS LAST
    LIMIT 1
""")


@dataclass
class Opponent:
    archetype: str
    share: float  # weight in the overall win rate
    main: Dict[str, int]


async def gauntlet(db: AsyncSession, format: str, archetypes: Optional[Sequence[str]] = None) -> List[Opponent]:
    """With `archetypes`, those (equal weight); otherwise the top GAUNTLET_SIZE by meta
    share (weighted by share). Archetypes without a recent list are skipped."""
    snapshots = (await db.execute(
        select(MetaSnapshot).where(MetaSnapshot.format == format)
        .order_by(MetaSnapshot.meta_percentage.desc()))).scalars().all()
    shares = {s.archetype.strip().lower(): float(s.meta_percentage or 0) for s in snapshots}
    names = [a.strip() for a in archetypes] if archetypes else [s.archetype.strip() for s in snapshots]
    size = len(names) if archetypes else GAUNTLET_SIZE
    out: List[Opponent] = []
    for name in names:
        if len(out) >= size:
            break
        row = (await db.execute(LIST_SQL, {"format": format, "archetype": name})).first()
        if row is None:
            continue
        main = {e["card_name"]: int(e["quantity"]) for e in row.main_deck}
        out.append(Opponent(name, 1.0 if archetypes else shares.get(name.lower(), 0.0), main))
    return out
```

- [ ] **Step 5: Run the gauntlet tests**

Run: `cd backend && python3 -m pytest tests/test_gauntlet.py -q` → PASS.

- [ ] **Step 6: Replace the model, add the migration, remove the old simulator**

Overwrite `backend/app/models/simulation.py`:

```python
"""A Forge simulation run: testing a deck against the meta, or playtesting a build."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.db.session import Base


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)  # anonymous allowed
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True)
    kind = Column(String(10), nullable=False)  # test | build
    status = Column(String(20), nullable=False, default="queued")  # queued, running, completed, failed, stopped
    format = Column(String(50), nullable=False, default="standard")
    deck = Column(JSONB, nullable=False)  # {name, main_deck: [{card_name, quantity}], sideboard}
    opponents = Column(JSONB, nullable=True)  # chosen archetype names; null = the meta gauntlet
    games_per_matchup = Column(Integer, nullable=False)
    options = Column(JSONB, nullable=True)  # build: requested, archetypes, synergy, colors
    progress = Column(JSONB, nullable=True)  # stage, games, live matchups, events, current deck
    report = Column(JSONB, nullable=True)
    final_deck = Column(JSONB, nullable=True)  # build: the playtested deck
    stop_requested = Column(Boolean, nullable=False, default=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (Index("idx_simulation_runs_user", "user_id", "created_at"),)
```

Create `backend/alembic/versions/017_forge_simulation_runs.py`:

```python
"""Replace the LLM-narrated simulation_runs with Forge simulation runs (the old table holds no data)."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("simulation_runs")
    op.create_table(
        "simulation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("kind", sa.String(10), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("format", sa.String(50), nullable=False, server_default="standard"),
        sa.Column("deck", postgresql.JSONB, nullable=False),
        sa.Column("opponents", postgresql.JSONB, nullable=True),
        sa.Column("games_per_matchup", sa.Integer, nullable=False),
        sa.Column("options", postgresql.JSONB, nullable=True),
        sa.Column("progress", postgresql.JSONB, nullable=True),
        sa.Column("report", postgresql.JSONB, nullable=True),
        sa.Column("final_deck", postgresql.JSONB, nullable=True),
        sa.Column("stop_requested", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_simulation_runs_user", "simulation_runs", ["user_id", "created_at"])


def downgrade() -> None:
    # The LLM simulator's table is not restored; its code is gone.
    op.drop_index("idx_simulation_runs_user", table_name="simulation_runs")
    op.drop_table("simulation_runs")
```

Delete the old simulator: `git rm backend/app/services/game_simulator.py backend/app/api/routes/simulation.py backend/app/schemas/simulation.py`.

In `backend/app/main.py`: remove `simulation,` from the routes import list, the `await cleanup_stale_simulations()` call and the comment above it, the whole `cleanup_stale_simulations` function, and the `app.include_router(simulation.router, ...)` line. (Task 5 adds the new router.)

- [ ] **Step 7: Migrate and run everything**

```bash
cd /Users/robmiller/Projects/mtg-deckbuilder
docker exec spellbook-backend alembic upgrade head
docker exec spellbook-db psql -U spellbook -d spellbook -c "\d simulation_runs" | head -25
cd backend && python3 -m pytest -q
docker logs spellbook-backend --since 2m 2>&1 | grep -iE "error|traceback" | tail -5
```

Expected: upgrade to `017` succeeds; the table has the new columns; all tests pass; no import errors after the backend reloads.

- [ ] **Step 8: Commit**

```bash
git add -A backend/app backend/alembic/versions/017_forge_simulation_runs.py backend/tests/test_gauntlet.py
git commit -m "Replace the LLM simulator's runs with Forge runs; add the gauntlet"
```

---

### Task 5: Test a deck: the job, the report and the API

**Files:**
- Create: `backend/app/services/sim_report.py`
- Create: `backend/app/services/sim_runs.py`
- Create: `backend/app/jobs/sim_tasks.py`
- Create: `backend/app/api/routes/simulation.py` (new)
- Modify: `backend/app/main.py` (mount the new router at `/api/simulations`)
- Test: `backend/tests/test_sim_report.py`, `backend/tests/test_sim_runs.py`, `backend/tests/test_simulation_api.py`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces:
  - `sim_report.LIMITS` (the limits sentence), `sim_report.build_report(matchups, records, main, lands, missing, baseline=None, changes=(), stopped=None) -> dict` with keys `overall, baseline, matchups, changes, cards{strongest, weakest, too_few}, mana{mulligan_rate, screw_rate, flood_rate, advice}, games{win, loss}, not_simulated{cards, sideboard}, limits, stopped`.
  - `sim_runs`: `MIN_GAMES = 10`, `MAX_GAMES = 200`, `TEST_GAMES = 50`; `main_of(entries) -> Dict[str, int]`; `entries_of(main) -> List[dict]`; `async land_names(db, names) -> Set[str]`; `enqueue(run_id)`; `queue_position(run_id) -> Optional[int]`; `class SimError(ValueError)` (message shown to users); `class Progress(db, run, opponents, games_planned)` with async `save, stage(text), event(text, kind="info"), games(key, records, tally=True), set_matchups(stats), deck(main), stop_requested() -> bool`; `async run_test(db, run)`; `async execute(run_id: UUID)`.
  - `deck_search.Stopped` is created in Task 6; this task defines `class Stopped(Exception)` in `sim_runs` and Task 6 moves it (see Task 6 Step 3). Raised by `Progress.games` when a stop was requested.
  - `app.jobs.sim_tasks.run_simulation(run_id: str)` (RQ entry point).
  - API under `/api/simulations`: `POST ""`, `GET ""`, `GET "/archetypes"`, `GET "/{id}"`, `POST "/{id}/stop"`; response model `SimulationResponse`.

- [ ] **Step 1: Write the failing report test**

Create `backend/tests/test_sim_report.py`:

```python
"""The report a user reads after a test or a build."""

from app.services.forge import OPPONENT, TESTED, GameRecord
from app.services.sim_report import LIMITS, build_report
from app.services.sim_stats import matchup_stats


def game(winner, casts, turns, log):
    return GameRecord(winner=winner, turns=turns, own_turns={TESTED: turns, OPPONENT: turns},
                      mulligans={TESTED: 0, OPPONENT: 0}, casts={TESTED: casts, OPPONENT: []},
                      lands={TESTED: [(t, "Forest") for t in range(1, 6)], OPPONENT: []}, log=log)


def test_report_sections():
    wins = [game(TESTED, [(2, "Good")], 6 + i % 3, [f"win {i}"]) for i in range(6)]
    losses = [game(OPPONENT, [(2, "Bad")], 9, [f"loss {i}"]) for i in range(6)]
    records = wins + losses
    stats = [matchup_stats("A", 10.0, records)]
    main = {"Forest": 24, "Good": 4, "Bad": 4, "Rare": 4}
    report = build_report(stats, records, main, {"Forest"}, ["Unknown Card"],
                          changes=[{"cut": "Bad", "add": "Good"}], stopped="no_improvement")
    assert report["overall"]["win_rate"] == 0.5 and report["overall"]["games"] == 12
    assert report["baseline"] is None
    assert report["matchups"][0]["opponent"] == "A" and report["matchups"][0]["label"] == "even"
    assert [c["name"] for c in report["cards"]["strongest"]] == ["Good"]
    assert [c["name"] for c in report["cards"]["weakest"]] == ["Bad"]
    assert report["cards"]["too_few"] == ["Rare"]
    assert report["games"]["win"] == ["win 1"]  # the win closest to the median win length (7 turns)
    assert report["games"]["loss"] == ["loss 0"]
    assert report["not_simulated"] == {"cards": ["Unknown Card"], "sideboard": True}
    assert report["limits"] == LIMITS
    assert report["changes"] == [{"cut": "Bad", "add": "Good"}] and report["stopped"] == "no_improvement"
    assert set(report["mana"]) == {"mulligan_rate", "screw_rate", "flood_rate", "advice"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_sim_report.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the report builder**

Create `backend/app/services/sim_report.py`:

```python
"""The report shown after a deck test or a build playtest."""

import statistics
from typing import Dict, List, Optional, Sequence, Set

from app.services.forge import TESTED, GameRecord
from app.services.sim_stats import MatchupStats, card_stats, mana_advice, mana_stats, overall

LIMITS = ("Played by Forge's AI, which plays straightforwardly; decks that rely on tricky play "
          "may do better in real games.")
MIN_CASTS = 5  # a card needs this many games cast to be ranked
SHOWN_CARDS = 5
LOG_LINES = 300


def representative(records: Sequence[GameRecord], won: bool) -> Optional[List[str]]:
    """The condensed log of the win (or loss) closest to the median length."""
    games = [r for r in records if r.winner is not None and (r.winner == TESTED) == won]
    if not games:
        return None
    mid = statistics.median(r.turns for r in games)
    return min(games, key=lambda r: abs(r.turns - mid)).log[:LOG_LINES]


def build_report(matchups: Sequence[MatchupStats], records: Sequence[GameRecord], main: Dict[str, int],
                 lands: Set[str], missing: Sequence[str], baseline: Optional[Sequence[MatchupStats]] = None,
                 changes: Sequence[Dict] = (), stopped: Optional[str] = None) -> Dict:
    cards = card_stats(records, main, lands)
    judged = sorted((c for c in cards if c.games_cast >= MIN_CASTS),
                    key=lambda c: c.win_rate_when_cast, reverse=True)
    half = (len(judged) + 1) // 2  # the top half can be "strongest", the bottom half "weakest"
    strongest = judged[:min(half, SHOWN_CARDS)]
    weakest = list(reversed(judged[half:]))[:SHOWN_CARDS]
    mana = mana_stats(records)
    return {
        "overall": overall(matchups).as_dict(),
        "baseline": overall(baseline).as_dict() if baseline else None,
        "matchups": [m.as_dict() for m in matchups],
        "changes": list(changes),
        "cards": {"strongest": [c.as_dict() for c in strongest], "weakest": [c.as_dict() for c in weakest],
                  "too_few": sorted(c.name for c in cards if c.games_cast < MIN_CASTS)},
        "mana": {**mana, "advice": mana_advice(mana)},
        "games": {"win": representative(records, True), "loss": representative(records, False)},
        "not_simulated": {"cards": list(missing), "sideboard": True},
        "limits": LIMITS,
        "stopped": stopped,
    }
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd backend && python3 -m pytest tests/test_sim_report.py -q` → PASS.

- [ ] **Step 5: Write the failing job tests**

Create `backend/tests/test_sim_runs.py`:

```python
"""Running a deck test: progress, missing cards, stop, and failures."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.models.simulation import SimulationRun
from app.services import forge, sim_runs
from app.services.forge import OPPONENT, TESTED, GameRecord
from app.services.gauntlet import Opponent

OPPONENTS = [Opponent("A", 20.0, {"Island": 60}), Opponent("B", 10.0, {"Swamp": 60})]


def game(winner):
    return GameRecord(winner=winner, turns=7, own_turns={TESTED: 7, OPPONENT: 7}, mulligans={TESTED: 0, OPPONENT: 0},
                      casts={TESTED: [(2, "Bear")], OPPONENT: []}, lands={TESTED: [(1, "Forest")], OPPONENT: []},
                      log=["Turn 1: Tested"])


def make_run(**overrides):
    fields = dict(id=uuid4(), kind="test", status="running", format="standard", games_per_matchup=4,
                  stop_requested=False, deck={"name": "Mine", "main_deck": [
                      {"card_name": "Forest", "quantity": 20}, {"card_name": "Bear", "quantity": 4},
                      {"card_name": "Unknown Card", "quantity": 4}]})
    return SimulationRun(**{**fields, **overrides})


def fake_db():
    return MagicMock(commit=AsyncMock(), refresh=AsyncMock(), execute=AsyncMock())


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(sim_runs, "gauntlet", AsyncMock(return_value=OPPONENTS))
    monkeypatch.setattr(sim_runs, "land_names", AsyncMock(return_value={"Forest"}))
    monkeypatch.setattr(forge, "card_names", lambda: {"forest", "bear"})

    async def play(tested, matchups, games, seed, on_progress=None):
        assert "Unknown Card" in tested  # write_deck leaves it out; the job doesn't strip it
        out = {}
        for m in matchups:
            records = [game(TESTED), game(OPPONENT)] * (games // 2)
            out[m.key] = records
            if on_progress:
                await on_progress(m.key, records)
        return out
    monkeypatch.setattr(forge, "play", play)


async def test_run_test_reports_and_tracks_progress(wired):
    db, run = fake_db(), make_run()
    await sim_runs.run_test(db, run)
    assert run.status == "completed"
    assert run.report["overall"]["games"] == 8 and run.report["not_simulated"]["cards"] == ["Unknown Card"]
    assert [m["opponent"] for m in run.report["matchups"]] == ["A", "B"]
    p = run.progress
    assert p["games_done"] == p["games_planned"] == 8 and p["stage"] == "Done"
    assert p["matchups"][0] == {"opponent": "A", "share": 20.0, "wins": 2, "losses": 2, "draws": 0}
    assert any("Unknown Card" in e["text"] and "left out" in e["text"] for e in p["events"])


async def test_stop_keeps_the_games_played(wired, monkeypatch):
    db, run = fake_db(), make_run()
    calls = {"n": 0}

    async def refresh(obj, attrs=None):
        calls["n"] += 1
        obj.stop_requested = calls["n"] >= 1  # stop after the first batch
    db.refresh = refresh
    await sim_runs.run_test(db, run)
    assert run.status == "stopped" and run.report["stopped"] == "user"
    assert run.report["overall"]["games"] == 4  # only matchup A finished


async def test_no_opponents_is_a_user_facing_error(wired, monkeypatch):
    monkeypatch.setattr(sim_runs, "gauntlet", AsyncMock(return_value=[]))
    with pytest.raises(sim_runs.SimError, match="no recent decklists"):
        await sim_runs.run_test(fake_db(), make_run())


async def test_execute_marks_failures_in_plain_words(monkeypatch):
    run = make_run(status="queued")
    db = fake_db()
    db.get = AsyncMock(return_value=run)
    db.rollback = AsyncMock()
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=db)
    session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(sim_runs, "async_session_factory", lambda: session)
    monkeypatch.setattr(forge, "available", lambda: True)
    monkeypatch.setattr(sim_runs, "run_test", AsyncMock(side_effect=forge.ForgeError("Forge exited with code 1")))
    await sim_runs.execute(run.id)
    assert run.status == "failed"
    assert run.error == "Couldn't finish playtesting: Forge exited with code 1"

    monkeypatch.setattr(forge, "available", lambda: False)
    run.status = "queued"
    await sim_runs.execute(run.id)
    assert run.status == "failed" and "isn't installed" in run.error


def test_deck_entry_conversions():
    assert sim_runs.main_of([{"card_name": "A", "quantity": "2"}, {"card_name": "A", "quantity": 1}]) == {"A": 3}
    assert sim_runs.entries_of({"A": 3, "B": 0}) == [{"card_name": "A", "quantity": 3}]
```

- [ ] **Step 6: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_sim_runs.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 7: Write the runs module and the RQ entry point**

Create `backend/app/services/sim_runs.py`:

```python
"""Simulation runs: queueing, live progress, and the 'test a deck' job. Builds
(sim_build.run_build) share Progress and execute()."""

import copy
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.queue import sim_queue
from app.db.session import async_session_factory
from app.models.simulation import SimulationRun
from app.services import forge
from app.services.forge import GameRecord
from app.services.gauntlet import Opponent, gauntlet
from app.services.sim_report import build_report
from app.services.sim_stats import MatchupStats, matchup_stats

logger = logging.getLogger(__name__)

MIN_GAMES, MAX_GAMES, TEST_GAMES = 10, 200, 50
MAX_EVENTS = 60
JOB_TIMEOUT_S = 1800


class SimError(ValueError):
    """A reason the run can't go on, worded for the user."""


class Stopped(Exception):
    """The user asked to stop."""


def main_of(entries: Sequence[Dict]) -> Dict[str, int]:
    main: Dict[str, int] = {}
    for e in entries or []:
        main[e["card_name"]] = main.get(e["card_name"], 0) + int(e["quantity"])
    return main


def entries_of(main: Dict[str, int]) -> List[Dict]:
    return [{"card_name": n, "quantity": q} for n, q in main.items() if q > 0]


LANDS_SQL = text("""
    SELECT DISTINCT name FROM cards
    WHERE name = ANY(CAST(:names AS varchar[]))
      AND split_part(coalesce(type_line, ''), ' // ', 1) LIKE '%Land%'
""")


async def land_names(db: AsyncSession, names: Sequence[str]) -> Set[str]:
    return {r.name for r in (await db.execute(LANDS_SQL, {"names": list(names)})).all()}


def enqueue(run_id: UUID) -> None:
    sim_queue.enqueue("app.jobs.sim_tasks.run_simulation", str(run_id), job_id=str(run_id),
                      job_timeout=JOB_TIMEOUT_S)


def queue_position(run_id: UUID) -> Optional[int]:
    ids = sim_queue.job_ids
    return ids.index(str(run_id)) + 1 if str(run_id) in ids else None


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


class Progress:
    """The live view of a run, written to run.progress as it changes."""

    def __init__(self, db: AsyncSession, run: SimulationRun, opponents: Sequence[Opponent], games_planned: int):
        self.db, self.run = db, run
        self.data = {
            "stage": "Starting", "games_done": 0, "games_planned": games_planned, "started_at": _now(),
            "matchups": [{"opponent": o.archetype, "share": o.share, "wins": 0, "losses": 0, "draws": 0}
                         for o in opponents],
            "events": [], "deck": None,
        }

    async def save(self) -> None:
        self.run.progress = copy.deepcopy(self.data)  # a new object, so SQLAlchemy sees the change
        self.run.updated_at = datetime.utcnow()
        await self.db.commit()

    async def stage(self, text_: str) -> None:
        self.data["stage"] = text_
        await self.save()

    async def event(self, text_: str, kind: str = "info") -> None:
        self.data["events"] = (self.data["events"] + [{"at": _now(), "text": text_, "kind": kind}])[-MAX_EVENTS:]
        await self.save()

    async def games(self, key: str, records: Sequence[GameRecord], tally: bool = True) -> None:
        """Count finished games (and, for the deck being reported, tally them); raise
        Stopped when the user asked to stop."""
        self.data["games_done"] += len(records)
        if tally:
            row = next(m for m in self.data["matchups"] if m["opponent"] == key)
            for r in records:
                field = "wins" if r.winner == forge.TESTED else "losses" if r.winner == forge.OPPONENT else "draws"
                row[field] += 1
        await self.save()
        if await self.stop_requested():
            raise Stopped()

    async def set_matchups(self, stats: Sequence[MatchupStats]) -> None:
        self.data["matchups"] = [{"opponent": m.opponent, "share": m.share, "wins": m.wins, "losses": m.losses,
                                  "draws": m.draws} for m in stats]
        await self.save()

    async def deck(self, main: Dict[str, int]) -> None:
        self.data["deck"] = entries_of(main)
        await self.save()

    async def stop_requested(self) -> bool:
        await self.db.refresh(self.run, ["stop_requested"])
        return bool(self.run.stop_requested)


async def run_test(db: AsyncSession, run: SimulationRun) -> None:
    opponents = await gauntlet(db, run.format, run.opponents)
    if not opponents:
        raise SimError("There are no recent decklists for these opponents to play against.")
    main = main_of(run.deck["main_deck"])
    known = forge.card_names()
    missing = forge.missing_cards(main, known)
    lands = await land_names(db, list(main))
    progress = Progress(db, run, opponents, run.games_per_matchup * len(opponents))
    await progress.stage(f"Playing {run.games_per_matchup} games against each of {len(opponents)} decks")
    for name in missing:
        await progress.event(f"Forge can't play [[{name}]] yet, so it was left out of testing.")

    played: Dict[str, List[GameRecord]] = {o.archetype: [] for o in opponents}

    async def on_games(key: str, records: List[GameRecord]) -> None:
        played[key].extend(records)
        await progress.games(key, records)

    stopped = None
    try:
        await forge.play(main, [forge.Matchup(o.archetype, o.main) for o in opponents], run.games_per_matchup,
                         random.randrange(1 << 30), on_games)
    except Stopped:
        stopped = "user"
    stats = [matchup_stats(o.archetype, o.share, played[o.archetype]) for o in opponents]
    run.report = build_report(stats, [r for rs in played.values() for r in rs], main, lands, missing,
                              stopped=stopped)
    run.status = "stopped" if stopped else "completed"
    progress.data["stage"] = "Stopped" if stopped else "Done"
    await progress.save()


async def execute(run_id: UUID) -> None:
    """Run a queued simulation (the sim worker's job)."""
    async with async_session_factory() as db:
        run = await db.get(SimulationRun, run_id)
        if run is None:
            return
        if run.stop_requested:
            run.status = "stopped"
            await db.commit()
            return
        run.status = "running"
        await db.commit()
        try:
            if not forge.available():
                raise forge.ForgeError("the simulator isn't installed on the worker")
            if run.kind == "build":
                from app.services.sim_build import run_build
                await run_build(db, run)
            else:
                await run_test(db, run)
        except Exception as e:
            logger.exception(f"[SIM] run {run_id} failed")
            await db.rollback()
            run = await db.get(SimulationRun, run_id)
            run.status = "failed"
            run.error = (f"Couldn't finish playtesting: {e}" if isinstance(e, (forge.ForgeError, SimError))
                         else "Something went wrong while playtesting.")
            await db.commit()
```

Create `backend/app/jobs/sim_tasks.py`:

```python
"""RQ entry point for Forge simulation runs (consumed by the sim-worker)."""

from uuid import UUID

from app.jobs.rq_tasks import _run_async


def run_simulation(run_id: str) -> None:
    from app.services.sim_runs import execute
    _run_async(execute(UUID(run_id)))
```

- [ ] **Step 8: Run the job tests**

Run: `cd backend && python3 -m pytest tests/test_sim_runs.py tests/test_sim_report.py -q` → PASS. If `test_stop_keeps_the_games_played` sees 8 games, the stop check isn't cancelling the second matchup: `forge.play` in the test fake runs sequentially, so the first `on_progress` must raise; fix `Progress.games`, not the test.

- [ ] **Step 9: Write the failing API tests**

Create `backend/tests/test_simulation_api.py`:

```python
"""The simulations API: creating a test, reading it, stopping it."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.routes import simulation as api
from app.models.simulation import SimulationRun

DECK = {"name": "Mine", "main_deck": [{"card_name": "Forest", "quantity": 60}], "sideboard": []}
ALICE, BOB = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
NOW = datetime(2026, 10, 7)


def saved_run(**fields):
    return SimulationRun(**{"id": uuid4(), "kind": "test", "format": "standard", "deck": DECK,
                            "games_per_matchup": 50, "stop_requested": False, "created_at": NOW,
                            "updated_at": NOW, **fields})


def db_returning(obj=None):
    db = MagicMock(add=MagicMock(), commit=AsyncMock(), refresh=AsyncMock(), get=AsyncMock(return_value=obj))
    return db


async def test_create_needs_the_sim_worker(monkeypatch):
    monkeypatch.setattr(api, "sim_worker_running", lambda: False)
    with pytest.raises(HTTPException) as e:
        await api.create_simulation(api.SimulationCreate(deck=DECK), db_returning(), None)
    assert e.value.status_code == 503 and "isn't running" in e.value.detail


async def test_create_queues_a_test(monkeypatch):
    monkeypatch.setattr(api, "sim_worker_running", lambda: True)
    queued = []
    monkeypatch.setattr(api, "enqueue", queued.append)
    monkeypatch.setattr(api, "queue_position", lambda run_id: 1)
    db = db_returning()
    resp = await api.create_simulation(api.SimulationCreate(deck=DECK, opponents=["Dimir Aggro"], games=20), db, ALICE)
    run = db.add.call_args.args[0]
    assert (run.kind, run.user_id, run.opponents, run.games_per_matchup) == ("test", ALICE.id, ["Dimir Aggro"], 20)
    assert queued == [run.id] and resp.status == "queued" and resp.queue_position == 1


async def test_create_rejects_bad_input(monkeypatch):
    monkeypatch.setattr(api, "sim_worker_running", lambda: True)
    for body in (api.SimulationCreate(), api.SimulationCreate(deck={"main_deck": []}),
                 api.SimulationCreate(deck=DECK, format="cedh")):
        with pytest.raises(HTTPException) as e:
            await api.create_simulation(body, db_returning(), None)
        assert e.value.status_code == 400


@pytest.mark.parametrize("owner,caller,ok", [(None, None, True), (ALICE.id, ALICE, True),
                                             (ALICE.id, None, False), (ALICE.id, BOB, False)])
async def test_get_access(owner, caller, ok, monkeypatch):
    monkeypatch.setattr(api, "queue_position", lambda run_id: None)
    run = saved_run(user_id=owner, status="running")
    if ok:
        assert (await api.get_simulation(run.id, db_returning(run), caller)).id == run.id
    else:
        with pytest.raises(HTTPException) as e:
            await api.get_simulation(run.id, db_returning(run), caller)
        assert e.value.status_code == 404


async def test_stop_a_queued_run_stops_it_now(monkeypatch):
    monkeypatch.setattr(api, "queue_position", lambda run_id: None)
    removed = []
    monkeypatch.setattr(api.sim_queue, "remove", removed.append)
    run = saved_run(user_id=None, status="queued")
    resp = await api.stop_simulation(run.id, db_returning(run), None)
    assert run.stop_requested and resp.status == "stopped" and removed == [str(run.id)]
```

- [ ] **Step 10: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_simulation_api.py -q`
Expected: FAIL (`ImportError` for `app.api.routes.simulation`).

- [ ] **Step 11: Write the API and mount it**

Create `backend/app/api/routes/simulation.py`:

```python
"""Forge simulations: test a deck against the meta, read live progress and the report, stop a run."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.auth import get_current_user
from app.core.queue import sim_queue, sim_worker_running
from app.db.session import get_db
from app.models.deck import Deck
from app.models.meta import MetaSnapshot
from app.models.simulation import SimulationRun
from app.models.user import User
from app.services.deck_fill import SIXTY_CARD_FORMATS
from app.services.sim_runs import MAX_GAMES, MIN_GAMES, TEST_GAMES, enqueue, queue_position

router = APIRouter()

NOT_RUNNING = "The simulator isn't running right now, so this deck can't be tested. Try again in a few minutes."


class SimulationCreate(BaseModel):
    deck_id: Optional[UUID] = None
    deck: Optional[Dict[str, Any]] = None  # {name, main_deck: [{card_name, quantity}], sideboard}
    format: str = "standard"
    opponents: Optional[List[str]] = None  # archetype names; none = the top meta decks
    games: int = Field(TEST_GAMES, ge=MIN_GAMES, le=MAX_GAMES)


class SimulationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    kind: str
    status: str
    format: str
    deck: Dict[str, Any]
    opponents: Optional[List[str]] = None
    games_per_matchup: int
    progress: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    final_deck: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stop_requested: bool = False
    queue_position: Optional[int] = None
    created_at: datetime
    updated_at: datetime


def to_response(run: SimulationRun) -> SimulationResponse:
    resp = SimulationResponse.model_validate(run)
    resp.queue_position = queue_position(run.id) if run.status == "queued" else None
    return resp


def visible(run: Optional[SimulationRun], user: Optional[User]) -> bool:
    return run is not None and (run.user_id is None or (user is not None and run.user_id == user.id))


async def _deck_for(body: SimulationCreate, db: AsyncSession, user: Optional[User]) -> Dict[str, Any]:
    if body.deck_id:
        deck = await db.get(Deck, body.deck_id)
        if deck is None or (deck.visibility == "private" and (user is None or deck.owner_id != user.id)):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "That deck wasn't found.")
        return {"name": deck.name, "main_deck": deck.main_deck or [], "sideboard": deck.sideboard or []}
    if body.deck and body.deck.get("main_deck"):
        return {"name": body.deck.get("name") or "Untitled deck", "main_deck": body.deck["main_deck"],
                "sideboard": body.deck.get("sideboard") or []}
    raise HTTPException(status.HTTP_400_BAD_REQUEST, "Choose a deck with a main deck to test.")


@router.post("", response_model=SimulationResponse)
async def create_simulation(body: SimulationCreate, db: AsyncSession = Depends(get_db),
                            current_user: Optional[User] = Depends(get_current_user)):
    if body.format not in SIXTY_CARD_FORMATS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Simulation supports 60-card formats only.")
    deck = await _deck_for(body, db, current_user)
    if not sim_worker_running():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, NOT_RUNNING)
    run = SimulationRun(id=uuid4(), user_id=current_user.id if current_user else None, kind="test", status="queued",
                        format=body.format, deck=deck, opponents=body.opponents or None,
                        games_per_matchup=body.games, stop_requested=False,
                        created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.add(run)
    await db.commit()
    enqueue(run.id)
    return to_response(run)


@router.get("", response_model=List[SimulationResponse])
async def list_simulations(db: AsyncSession = Depends(get_db),
                           current_user: Optional[User] = Depends(get_current_user)):
    if current_user is None:
        return []
    runs = (await db.execute(select(SimulationRun).where(SimulationRun.user_id == current_user.id)
                             .order_by(SimulationRun.created_at.desc()).limit(20))).scalars().all()
    return [to_response(r) for r in runs]


@router.get("/archetypes", response_model=List[str])
async def list_archetypes(format: str = "standard", db: AsyncSession = Depends(get_db)):
    """Meta archetypes to choose as opponents, by share."""
    snaps = (await db.execute(select(MetaSnapshot).where(MetaSnapshot.format == format)
                              .order_by(MetaSnapshot.meta_percentage.desc()))).scalars().all()
    return [s.archetype.strip() for s in snaps]


@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(simulation_id: UUID, db: AsyncSession = Depends(get_db),
                         current_user: Optional[User] = Depends(get_current_user)):
    run = await db.get(SimulationRun, simulation_id)
    if not visible(run, current_user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Simulation not found")
    return to_response(run)


@router.post("/{simulation_id}/stop", response_model=SimulationResponse)
async def stop_simulation(simulation_id: UUID, db: AsyncSession = Depends(get_db),
                          current_user: Optional[User] = Depends(get_current_user)):
    """Stop after the current games; a queued run stops right away."""
    run = await db.get(SimulationRun, simulation_id)
    if not visible(run, current_user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Simulation not found")
    run.stop_requested = True
    if run.status == "queued":
        sim_queue.remove(str(run.id))
        run.status = "stopped"
    await db.commit()
    return to_response(run)
```

Check that `Deck.visibility` stores the string `"private"` (`grep -n "PRIVATE" backend/app/models/deck.py`); if the enum value differs, compare against `DeckVisibility.PRIVATE.value` instead.

In `backend/app/main.py`, import the new module alongside the other routes (`simulation,` back in the import list) and add:

```python
app.include_router(simulation.router, prefix="/api/simulations", tags=["Simulations"])
```

- [ ] **Step 12: Run everything**

Run: `cd backend && python3 -m pytest -q` → all pass. Then check the API is mounted live:

```bash
curl -s localhost:8000/api/simulations/archetypes?format=standard | head -c 200; echo
curl -s -X POST localhost:8000/api/simulations -H 'Content-Type: application/json' \
  -d '{"deck":{"name":"t","main_deck":[{"card_name":"Forest","quantity":60}]},"games":10}' | head -c 300; echo
```

Expected: a JSON list of archetype names; the POST returns a run with `"status":"queued"` (the sim worker is up from Task 1). Within a minute, `curl -s localhost:8000/api/simulations/<id>` shows `"status":"completed"` and a report (60 Forests lose every game: overall win rate 0). Paste both outputs in your report.

- [ ] **Step 13: Commit**

```bash
git add backend/app/services/sim_report.py backend/app/services/sim_runs.py backend/app/jobs/sim_tasks.py \
  backend/app/api/routes/simulation.py backend/app/main.py \
  backend/tests/test_sim_report.py backend/tests/test_sim_runs.py backend/tests/test_simulation_api.py
git commit -m "Test a deck against the meta with Forge: job, report and API"
```

---

### Task 6: Playtest search

**Files:**
- Create: `backend/app/services/deck_search.py`
- Modify: `backend/app/services/sim_runs.py` (import `Stopped` from `deck_search` instead of defining it)
- Test: `backend/tests/test_deck_search.py`

**Interfaces:**
- Consumes: `sim_stats` (Task 3), `forge.GameRecord` (Task 2), `gauntlet.Opponent` (Task 4), the `Progress` interface (Task 5): `stage, event, set_matchups, deck, stop_requested`.
- Produces:
  - `class Stopped(Exception)` (moved here; `sim_runs` imports it)
  - `@dataclass SearchConfig(screen_games=20, confirm_games=50, max_rounds=6, candidates=6, budget_s=360.0, min_casts=5, stuck_rate=0.15, problem_rate=0.15)`
  - `@dataclass Swap(cut: str, add: str, copies: int, reason: str)` with `apply(main) -> Dict[str, int]`
  - `@dataclass SearchResult(main, baseline: List[MatchupStats], final: List[MatchupStats], records: Dict[str, List[GameRecord]], changes: List[Dict], stopped: str)`
  - `land_swap(main, records, cards, protect, cfg) -> Optional[Swap]`
  - `async playtest(seed_main, opponents, lands, evaluate, find_add, protect, progress, cfg=SearchConfig(), clock=time.monotonic, rng_seed=0) -> SearchResult` where `evaluate(main, games, seed) -> Dict[archetype, records]`, `find_add(main, cut, losing_to: List[str]) -> Optional[str]`, `protect(main, cut) -> bool`.
  - `stopped` values: `"no_improvement"`, `"no_candidates"`, `"max_rounds"`, `"budget"`, `"user"`, `"reverted"`.
  - Each change: `{"cut", "add", "copies", "before", "after", "best_matchup": {"opponent", "before", "after"}}`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_deck_search.py`:

```python
"""Playtest search: cut what underperforms, keep swaps that clearly win more, confirm."""

import pytest

from app.services import deck_search as ds
from app.services.forge import OPPONENT, TESTED, GameRecord
from app.services.gauntlet import Opponent

OPPONENTS = [Opponent("A", 20.0, {}), Opponent("B", 10.0, {})]
LANDS = {"Forest"}
EFFECT = {"Good": 0.06, "Dud": -0.06}  # win-rate change per copy
SEED = {"Forest": 24, "Dud": 4, "Meh": 4, "Filler": 28}


def rate_of(main):
    return min(0.95, max(0.05, 0.5 + sum(EFFECT.get(n, 0) * q for n, q in main.items())))


def fake_evaluate(calls, flip_on_confirm=False):
    """Deterministic games: Dud is cast only in losses, Good only in wins."""
    async def evaluate(main, games, seed):
        calls.append((dict(main), games, seed))
        rate = rate_of(main)
        if flip_on_confirm and games == ds.SearchConfig().confirm_games:
            rate = 1 - rate
        out = {}
        for opp in OPPONENTS:
            wins = round(games * rate)
            records = []
            for i in range(games):
                won = i < wins
                casts = [(2, n) for n in main if n not in LANDS and (n != "Dud" or not won) and (n != "Good" or won)]
                records.append(GameRecord(
                    winner=TESTED if won else OPPONENT, turns=8, own_turns={TESTED: 8, OPPONENT: 8},
                    mulligans={TESTED: 0, OPPONENT: 0}, casts={TESTED: casts, OPPONENT: []},
                    lands={TESTED: [(t, "Forest") for t in range(1, 6)], OPPONENT: []}))
            out[opp.archetype] = records
        return out
    return evaluate


async def find_add(main, cut, losing_to):
    return "Good" if cut == "Dud" else "Meh2"


class FakeProgress:
    def __init__(self, stop_after=None):
        self.events, self.stages, self.decks, self.matchups = [], [], [], []
        self.stop_after, self.checks = stop_after, 0

    async def stage(self, text):
        self.stages.append(text)

    async def event(self, text, kind="info"):
        self.events.append((kind, text))

    async def set_matchups(self, stats):
        self.matchups.append(stats)

    async def deck(self, main):
        self.decks.append(dict(main))

    async def stop_requested(self):
        self.checks += 1
        return self.stop_after is not None and self.checks > self.stop_after


async def test_keeps_a_clearly_better_swap_and_confirms_it():
    calls, progress = [], FakeProgress()
    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate(calls), find_add,
                               lambda main, cut: False, progress, rng_seed=5)
    assert result.main == {"Forest": 24, "Meh": 4, "Filler": 28, "Good": 4}
    assert [(c["cut"], c["add"], c["copies"]) for c in result.changes] == [("Dud", "Good", 4)]
    change = result.changes[0]
    assert change["before"] == pytest.approx(0.25) and change["after"] == pytest.approx(0.75)
    assert result.stopped == "no_improvement"
    screen = {seed for _, games, seed in calls if games == 20}
    confirm = [(m, seed) for m, games, seed in calls if games == 50]
    assert screen == {5} and [m for m, _ in confirm] == [SEED, result.main] and len({s for _, s in confirm}) == 1
    assert any(kind == "kept" and "Dud" in text and "Good" in text for kind, text in progress.events)
    assert any(kind == "tried" and "Dud won 0% of the games it was cast in" in text for kind, text in progress.events)
    assert progress.decks[-1] == result.main


async def test_protected_cards_are_never_cut():
    calls, progress = [], FakeProgress()
    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate(calls), find_add,
                               lambda main, cut: cut == "Dud", progress)
    assert result.main == SEED and result.changes == []
    assert all(c[0]["Dud"] == 4 for c in calls)


async def test_changes_that_fail_confirmation_are_reverted():
    calls, progress = [], FakeProgress()
    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate(calls, flip_on_confirm=True), find_add,
                               lambda main, cut: False, progress)
    assert result.main == SEED and result.changes == [] and result.stopped == "reverted"
    assert progress.decks[-1] == SEED


async def test_budget_and_user_stop():
    times = iter([0.0, 1000.0, 1000.0])
    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate([]), find_add, lambda m, c: False,
                               FakeProgress(), clock=lambda: next(times))
    assert result.stopped == "budget" and result.main == SEED

    result = await ds.playtest(SEED, OPPONENTS, LANDS, fake_evaluate([]), find_add, lambda m, c: False,
                               FakeProgress(stop_after=0))
    assert result.stopped == "user" and result.main == SEED


async def test_stop_during_the_baseline_returns_the_seed():
    async def evaluate(main, games, seed):
        raise ds.Stopped()
    result = await ds.playtest(SEED, OPPONENTS, LANDS, evaluate, find_add, lambda m, c: False, FakeProgress())
    assert result.main == SEED and result.stopped == "user" and result.final == []


def test_swap_apply_respects_existing_copies():
    assert ds.Swap("A", "B", 2, "").apply({"A": 2, "B": 1}) == {"B": 3}


def test_land_swap_for_screw():
    screwed = GameRecord(winner=OPPONENT, turns=6, own_turns={TESTED: 6, OPPONENT: 6},
                         mulligans={TESTED: 0, OPPONENT: 0}, casts={TESTED: [(2, "Weak")], OPPONENT: []},
                         lands={TESTED: [(1, "Forest"), (2, "Forest")], OPPONENT: []})
    records = [screwed] * 10
    from app.services.sim_stats import card_stats
    main = {"Forest": 22, "Weak": 4}
    cards = card_stats(records, main, {"Forest"})
    swap = ds.land_swap(main, records, cards, lambda m, c: False, ds.SearchConfig())
    assert (swap.cut, swap.add, swap.copies) == ("Weak", "Forest", 1)
    assert "short on lands" in swap.reason
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_deck_search.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the search and move `Stopped`**

Create `backend/app/services/deck_search.py`:

```python
"""Playtest search: play the deck against the gauntlet, swap out what underperforms,
keep a swap only when it clearly wins more, and confirm the result on fresh games."""

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Set

from app.services.deck_fill import BASICS
from app.services.forge import GameRecord
from app.services.gauntlet import Opponent
from app.services.sim_stats import CardStat, MatchupStats, card_stats, mana_stats, matchup_stats, overall, pct

BASIC_NAMES = set(BASICS.values())
MAX_COPIES = 4
CONFIRM_SEED_OFFSET = 1000

Evaluate = Callable[[Dict[str, int], int, int], Awaitable[Dict[str, List[GameRecord]]]]
FindAdd = Callable[[Dict[str, int], str, List[str]], Awaitable[Optional[str]]]
Protect = Callable[[Dict[str, int], str], bool]


class Stopped(Exception):
    """The user asked to stop."""


@dataclass
class SearchConfig:
    screen_games: int = 20  # per matchup, for the baseline and each candidate
    confirm_games: int = 50  # per matchup, for the final check
    max_rounds: int = 6
    candidates: int = 6  # swaps tried per round
    budget_s: float = 360.0
    min_casts: int = 5  # games cast before a card's win rate is trusted
    stuck_rate: float = 0.15  # cast in fewer games than this: stuck in hand or uncastable
    problem_rate: float = 0.15  # screw or flood rate that earns a land-count candidate


@dataclass
class Swap:
    cut: str
    add: str
    copies: int
    reason: str

    def apply(self, main: Dict[str, int]) -> Dict[str, int]:
        new = dict(main)
        new[self.cut] = new.get(self.cut, 0) - self.copies
        if new[self.cut] <= 0:
            del new[self.cut]
        new[self.add] = new.get(self.add, 0) + self.copies
        return new


@dataclass
class SearchResult:
    main: Dict[str, int]
    baseline: List[MatchupStats]
    final: List[MatchupStats]
    records: Dict[str, List[GameRecord]]
    changes: List[Dict] = field(default_factory=list)
    stopped: str = "max_rounds"


def _flat(records: Dict[str, List[GameRecord]]) -> List[GameRecord]:
    return [r for rs in records.values() for r in rs]


def land_swap(main: Dict[str, int], records: Sequence[GameRecord], cards: Sequence[CardStat],
              protect: Protect, cfg: SearchConfig) -> Optional[Swap]:
    """One land more (screw) or fewer (flood), traded with a spell."""
    mana = mana_stats(records)
    basics = [n for n in main if n in BASIC_NAMES]
    judged = [c for c in cards if c.games_cast >= cfg.min_casts]
    if not basics or not judged:
        return None
    basic = max(basics, key=main.get)
    lands = sum(q for n, q in main.items() if n in BASIC_NAMES)
    if mana["screw_rate"] > cfg.problem_rate:
        weakest = min((c for c in judged if not protect(main, c.name)), key=lambda c: c.win_rate_when_cast,
                      default=None)
        if weakest:
            return Swap(weakest.name, basic, 1, f"Lost {pct(mana['screw_rate'])} of games short on lands: "
                                                f"trying one more {basic} instead of a {weakest.name}")
    if mana["flood_rate"] > cfg.problem_rate and lands > 0:
        best = max((c for c in judged if main[c.name] < MAX_COPIES), key=lambda c: c.win_rate_when_cast,
                   default=None)
        if best:
            return Swap(basic, best.name, 1, f"Lost {pct(mana['flood_rate'])} of games drawing mostly lands: "
                                             f"trying another {best.name} instead of a {basic}")
    return None


async def _propose(main: Dict[str, int], records: Sequence[GameRecord], stats: Sequence[MatchupStats],
                   lands: Set[str], find_add: FindAdd, protect: Protect, cfg: SearchConfig,
                   tried: Set) -> List[Swap]:
    cards = card_stats(records, main, lands)
    judged = sorted((c for c in cards if c.games_cast >= cfg.min_casts), key=lambda c: c.win_rate_when_cast)
    weakest = judged[:3]
    stuck = [c for c in cards if c.cast_share < cfg.stuck_rate and c not in weakest]
    losing = [m.opponent for m in sorted(stats, key=lambda m: m.win_rate)[:2]]
    swaps: List[Swap] = []
    for card in weakest + stuck:
        if len(swaps) >= cfg.candidates - 1:  # leave room for a land candidate
            break
        if protect(main, card.name):
            continue
        add = await find_add(main, card.name, losing)
        if not add or (card.name, add) in tried:
            continue
        copies = min(main[card.name], MAX_COPIES - main.get(add, 0))
        if copies <= 0:
            continue
        why = (f"{card.name} won {pct(card.win_rate_when_cast)} of the games it was cast in" if card in weakest
               else f"{card.name} was cast in only {pct(card.cast_share)} of games")
        swaps.append(Swap(card.name, add, copies, f"{why}: trying {add} instead"))
        tried.add((card.name, add))
    land = land_swap(main, records, cards, protect, cfg)
    if land and (land.cut, land.add) not in tried:
        swaps.append(land)
        tried.add((land.cut, land.add))
    return swaps[:cfg.candidates]


def _change(swap: Swap, before: Sequence[MatchupStats], after: Sequence[MatchupStats],
            before_rate: float, after_rate: float) -> Dict:
    pairs = [(b, a) for b in before for a in after if a.opponent == b.opponent]
    b, a = max(pairs, key=lambda p: p[1].win_rate - p[0].win_rate)
    return {"cut": swap.cut, "add": swap.add, "copies": swap.copies, "before": before_rate, "after": after_rate,
            "best_matchup": {"opponent": a.opponent, "before": b.win_rate, "after": a.win_rate}}


async def playtest(seed_main: Dict[str, int], opponents: Sequence[Opponent], lands: Set[str],
                   evaluate: Evaluate, find_add: FindAdd, protect: Protect, progress,
                   cfg: SearchConfig = SearchConfig(), clock: Callable[[], float] = time.monotonic,
                   rng_seed: int = 0) -> SearchResult:
    start = clock()

    def stats_of(records: Dict[str, List[GameRecord]]) -> List[MatchupStats]:
        return [matchup_stats(o.archetype, o.share, records.get(o.archetype, [])) for o in opponents]

    await progress.stage("Baseline: playing the first draft against the meta")
    try:
        current_records = await evaluate(seed_main, cfg.screen_games, rng_seed)
    except Stopped:
        return SearchResult(dict(seed_main), [], [], {}, [], "user")
    current, current_main = stats_of(current_records), dict(seed_main)
    first = overall(current)
    rate, se = first.win_rate, first.se
    await progress.set_matchups(current)
    await progress.event(f"First draft: {pct(rate)} against the meta")

    changes: List[Dict] = []
    stopped, tried = "max_rounds", set()
    try:
        for round_no in range(1, cfg.max_rounds + 1):
            if clock() - start > cfg.budget_s:
                stopped = "budget"
                await progress.event("Out of time: keeping the best deck so far")
                break
            if await progress.stop_requested():
                raise Stopped()
            swaps = await _propose(current_main, _flat(current_records), current, lands, find_add, protect, cfg, tried)
            if not swaps:
                stopped = "no_candidates"
                await progress.event("No more swaps worth trying")
                break
            await progress.stage(f"Round {round_no} of {cfg.max_rounds}: trying {len(swaps)} swaps")
            for swap in swaps:
                await progress.event(swap.reason, "tried")
            best = None
            for swap in swaps:
                records = await evaluate(swap.apply(current_main), cfg.screen_games, rng_seed)
                stats = stats_of(records)
                candidate_rate = overall(stats).win_rate
                if best is None or candidate_rate > best[1]:
                    best = (swap, candidate_rate, records, stats)
            swap, candidate_rate, records, stats = best
            if candidate_rate - rate > se:
                changes.append(_change(swap, current, stats, rate, candidate_rate))
                await progress.event(f"Kept −{swap.copies} {swap.cut} +{swap.copies} {swap.add}: "
                                     f"{pct(rate)} → {pct(candidate_rate)} against the meta", "kept")
                current_main, current_records, current = swap.apply(current_main), records, stats
                rate, se = candidate_rate, overall(stats).se
                await progress.set_matchups(current)
                await progress.deck(current_main)
            else:
                stopped = "no_improvement"
                await progress.event(f"Tried {len(swaps)} swaps; none was clearly better")
                break
    except Stopped:
        stopped = "user"
        await progress.event("Stopped: keeping the best deck so far")

    if not changes or stopped == "user":
        return SearchResult(current_main, current, current, current_records, changes, stopped)

    await progress.stage("Confirming: replaying the first draft and the final deck on new games")
    try:
        confirm_seed = rng_seed + CONFIRM_SEED_OFFSET
        seed_records = await evaluate(seed_main, cfg.confirm_games, confirm_seed)
        final_records = await evaluate(current_main, cfg.confirm_games, confirm_seed)
    except Stopped:
        return SearchResult(current_main, current, current, current_records, changes, "user")
    baseline, final = stats_of(seed_records), stats_of(final_records)
    if overall(final).win_rate <= overall(baseline).win_rate:
        await progress.event("The changes didn't hold up over more games, so the first draft stands")
        await progress.set_matchups(baseline)
        await progress.deck(seed_main)
        return SearchResult(dict(seed_main), baseline, baseline, seed_records, [], "reverted")
    await progress.set_matchups(final)
    await progress.event(f"Confirmed: {pct(overall(baseline).win_rate)} → {pct(overall(final).win_rate)} "
                         "against the meta", "kept")
    return SearchResult(current_main, baseline, final, final_records, changes, stopped)
```

In `backend/app/services/sim_runs.py`, delete the `class Stopped` definition and add `from app.services.deck_search import Stopped` to the imports (Task 5's tests and code keep working).

- [ ] **Step 4: Run the tests**

Run: `cd backend && python3 -m pytest tests/test_deck_search.py tests/test_sim_runs.py -q` → PASS; `python3 -m pytest -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/deck_search.py backend/app/services/sim_runs.py backend/tests/test_deck_search.py
git commit -m "Playtest search: swap underperformers, keep clear wins, confirm"
```

---

### Task 7: Playtest every chat build

**Files:**
- Create: `backend/app/services/sim_build.py`
- Modify: `backend/app/services/deck_fill.py` (`assemble` returns `synergy` and `requested`)
- Modify: `backend/app/schemas/deck.py` (`DeckGenerateResponse.assembly`)
- Modify: `backend/app/services/deck_generator.py` (fill `assembly`)
- Modify: `backend/app/schemas/conversation.py` (`ChatResponse.simulation_id`)
- Modify: `backend/app/services/chat_service.py` (`_handle_generate_full_deck` starts the playtest)
- Test: `backend/tests/test_sim_build.py`; modify `backend/tests/test_deck_fill.py`, `backend/tests/test_chat_fit_handlers.py`

**Interfaces:**
- Consumes: Tasks 2-6; `deck_fill.requested_rows`, `card_text`, `slot_pool`, `fill_slot`, `SYNERGY_COPIES`, `SIXTY_CARD_FORMATS`, `MAX_COPIES`, `colors_of`; `deck_plan.archetype_keys`, `is_land`, `Slot`, `RECENT`; `jev.session`.
- Produces:
  - `assemble(...)` result gains `"synergy": {"type_contains": str, "min": 8} | None` and `"requested": List[str]`.
  - `DeckGenerateResponse.assembly: Optional[Dict[str, Any]]` = `{reference, relatives, colors, synergy, requested}` when assembly built the deck, else `None`.
  - `ChatResponse.simulation_id: Optional[UUID]`.
  - `sim_build`: `BUILD_MINUTES = 6`; `async staples(db, archetypes, format, names) -> Set[str]`; `make_protect(protected: Set[str], synergy: Optional[Dict], rows: Dict[str, Any]) -> Protect`; `async replacement(db, client, main, cut, losing, colors, format, known, rows) -> Optional[str]`; `async run_build(db, run)`; `async start_build(db, deck, assembly, conversation_id, user_id, format) -> Tuple[Optional[SimulationRun], Optional[str]]`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_sim_build.py`:

```python
"""Playtesting chat builds: starting the run, protection rules, replacements, the job."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.config import settings
from app.services import sim_build as sb

DECK = {"name": "Brew", "main_deck": [{"card_name": "Forest", "quantity": 60}], "sideboard": []}
ASSEMBLY = {"reference": None, "relatives": ["Boros Dwarves", "Jeskai Control"], "colors": ["W", "U", "R"],
            "synergy": {"type_contains": "Artifact", "min": 8}, "requested": ["Weapons Manufacturing"]}


class TestStartBuild:
    async def test_queues_a_build_with_its_options(self, monkeypatch):
        monkeypatch.setattr(sb, "sim_worker_running", lambda: True)
        queued = []
        monkeypatch.setattr(sb, "enqueue", queued.append)
        db = MagicMock(add=MagicMock(), commit=AsyncMock())
        conv, user = uuid4(), uuid4()
        run, note = await sb.start_build(db, DECK, ASSEMBLY, conv, user, "standard")
        assert note is None and queued == [run.id]
        assert (run.kind, run.status, run.conversation_id, run.user_id) == ("build", "queued", conv, user)
        assert run.options == {"requested": ["Weapons Manufacturing"], "archetypes": ["Boros Dwarves", "Jeskai Control"],
                               "synergy": {"type_contains": "Artifact", "min": 8}, "colors": ["W", "U", "R"]}

    async def test_reference_decks_protect_the_reference_lists(self, monkeypatch):
        monkeypatch.setattr(sb, "sim_worker_running", lambda: True)
        monkeypatch.setattr(sb, "enqueue", lambda run_id: None)
        run, _ = await sb.start_build(MagicMock(add=MagicMock(), commit=AsyncMock()), DECK,
                                      {**ASSEMBLY, "reference": "Boros Aggro"}, None, None, "standard")
        assert run.options["archetypes"] == ["Boros Aggro"]

    async def test_explains_when_it_cannot_playtest(self, monkeypatch):
        monkeypatch.setattr(sb, "sim_worker_running", lambda: False)
        run, note = await sb.start_build(MagicMock(), DECK, ASSEMBLY, None, None, "standard")
        assert run is None and "simulator isn't running" in note
        for args in ((DECK, None, None, None, "standard"), (DECK, ASSEMBLY, None, None, "cedh")):
            assert await sb.start_build(MagicMock(), *args) == (None, None)
        monkeypatch.setattr(settings, "FORGE_ENABLED", False)
        assert await sb.start_build(MagicMock(), DECK, ASSEMBLY, None, None, "standard") == (None, None)


def test_protect_rules():
    rows = {"Relic": SimpleNamespace(type_line="Artifact"), "Gadget": SimpleNamespace(type_line="Artifact Creature"),
            "Bolt": SimpleNamespace(type_line="Instant")}
    protect = sb.make_protect({"Engine"}, {"type_contains": "Artifact", "min": 8}, rows)
    main = {"Engine": 4, "Relic": 4, "Gadget": 5, "Bolt": 4}
    assert protect(main, "Engine")  # requested
    assert protect(main, "Relic")  # would leave 5 artifacts, under 8
    assert not protect({**main, "Gadget": 8}, "Relic")  # leaves 8
    assert not protect(main, "Bolt")
    assert not sb.make_protect(set(), None, rows)(main, "Relic")


async def test_staples_map_front_faces_back_to_deck_names():
    result = MagicMock()
    result.all.return_value = [SimpleNamespace(k="esper origins"), SimpleNamespace(k="forest")]
    db = MagicMock(execute=AsyncMock(return_value=result))
    got = await sb.staples(db, ["Jeskai Control "], "standard", ["Esper Origins // Summon: Esper Maduin", "Bolt"])
    assert got == {"Esper Origins // Summon: Esper Maduin"}
    sql, params = db.execute.call_args.args
    assert params == {"format": "standard", "archetypes": ["jeskai control"]}
    assert "HAVING COUNT(DISTINCT lists.id) * 2 > (SELECT COUNT(*) FROM lists)" in str(sql)
    assert await sb.staples(db, [], "standard", ["Bolt"]) == set()


async def test_replacement_picks_from_a_slot_like_the_cut_card(monkeypatch):
    cut = SimpleNamespace(name="Dud", type_line="Instant", cmc=2, roles=["removal_targeted"], mana_cost="{1}{R}",
                          oracle_text="Deal 1 damage.")
    pool = [SimpleNamespace(name="Unscripted"), SimpleNamespace(name="Shock")]
    seen = {}

    async def slot_pool(db, slot, colors, format, chosen, archetypes=()):
        seen["slot"], seen["chosen"] = slot, chosen
        return pool

    async def fill_slot(client, slot, pool_, state, format, copies_for, question=None, allow_none=True):
        seen["pool"], seen["state"] = [r.name for r in pool_], state
        return [("Shock", 4)]
    monkeypatch.setattr(sb, "slot_pool", slot_pool)
    monkeypatch.setattr(sb, "fill_slot", fill_slot)
    got = await sb.replacement(None, object(), {"Dud": 3, "Forest": 22}, "Dud", ["Dimir Aggro"], ["R"], "standard",
                               {"shock", "dud", "forest"}, {"Dud": cut})
    assert got == "Shock"
    slot = seen["slot"]
    assert (slot.role, slot.cmc_min, slot.cmc_max, slot.copies) == ("removal_targeted", 1, 3, 3)
    assert seen["pool"] == ["Shock"]  # Forge can't play the other
    assert seen["state"]["deck"]["losing_to"] == ["Dimir Aggro"] and "Dud" in seen["state"]["deck"]["cut"]
    assert await sb.replacement(None, None, {"Dud": 3}, "Dud", [], ["R"], "standard", set(), {"Dud": cut}) is None
```

In `backend/tests/test_deck_fill.py`, in `test_build_around_gets_support_colors_and_a_synergy_slot`, append:

```python
        assert deck["synergy"] == {"type_contains": "Artifact", "min": 8}
        assert deck["requested"] == ["Engine"]
```

and in `test_reference_deck_is_60_and_15`, append:

```python
        assert deck["synergy"] is None and deck["requested"] == []
```

In `backend/tests/test_chat_fit_handlers.py`, append:

```python
async def test_generate_handler_starts_a_playtest(monkeypatch):
    from app.services import chat_service
    cid = uuid4()
    result = SimpleNamespace(deck=_deck(), conversation_id=cid, strategy_summary="s", fit_flagged=[],
                             assembly={"reference": None, "relatives": [], "colors": ["R"], "synergy": None,
                                       "requested": []})
    s = _service(generate=AsyncMock(return_value=result))
    sim_id = uuid4()
    start = AsyncMock(return_value=(SimpleNamespace(id=sim_id), None))
    monkeypatch.setattr(chat_service, "start_build", start)
    resp = await s._handle_generate_full_deck({"colors": ["R"]}, SimpleNamespace(id=cid), None)
    assert resp.simulation_id == sim_id and "Playtesting against the top 5 decks" in resp.response
    deck_arg, assembly_arg = start.await_args.args[1:3]
    assert deck_arg["name"] == "D" and assembly_arg == result.assembly

    start.return_value = (None, "Couldn't playtest this deck (the simulator isn't running), so this is the untested build.")
    resp = await s._handle_generate_full_deck({"colors": ["R"]}, SimpleNamespace(id=cid), None)
    assert resp.simulation_id is None and "simulator isn't running" in resp.response
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && python3 -m pytest tests/test_sim_build.py tests/test_deck_fill.py tests/test_chat_fit_handlers.py -q`
Expected: FAIL (`ImportError` for `sim_build`; `KeyError: 'synergy'`; `start_build` missing in `chat_service`).

- [ ] **Step 3: Return the assembly details**

In `backend/app/services/deck_fill.py`, at the end of `assemble`, replace the return:

```python
    return {"name": name, "strategy_summary": summary, "main_deck": main.entries(),
            "sideboard": side.entries(), "reference": reference, "relatives": relatives, "colors": deck_colors,
            "synergy": ({"type_contains": synergy.type_contains, "min": SYNERGY_COPIES[0]}
                        if synergy and synergy.type_contains else None),
            "requested": list(specific_cards or [])}
```

In `backend/app/schemas/deck.py`, add to `DeckGenerateResponse` (and `Dict`/`Any` to its typing import if missing):

```python
    assembly: Optional[Dict[str, Any]] = None  # Jev assembly details, for playtesting; None on the LLM path
```

In `backend/app/services/deck_generator.py`, right after the `deck_fill.assemble(...)` call succeeds, keep the details (`assembly = None` before the `try`):

```python
        assembly = None
        try:
            # SAVEPOINT: a SQL error inside assemble must not abort the session the LLM path reuses
            async with self.db.begin_nested():
                deck_data = await deck_fill.assemble(
                    ...unchanged arguments...
                )
            assembly = {k: deck_data.get(k) for k in ("reference", "relatives", "colors", "synergy", "requested")}
        except Exception as e:
```

and pass `assembly=assembly,` to the `DeckGenerateResponse(...)` at the end of `generate`.

- [ ] **Step 4: Write `sim_build`**

Create `backend/app/services/sim_build.py`:

```python
"""Playtesting a deck built from chat: queue the run, then (on the sim worker) improve
the deck by playing it with deck_search.playtest."""

import logging
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.queue import sim_worker_running
from app.models.simulation import SimulationRun
from app.services import forge, jev
from app.services.deck_fill import (
    MAX_COPIES, SIXTY_CARD_FORMATS, card_text, colors_of, fill_slot, requested_rows, slot_pool,
)
from app.services.deck_plan import RECENT, Slot, archetype_keys, is_land
from app.services.deck_search import BASIC_NAMES, Protect, SearchConfig, playtest
from app.services.gauntlet import GAUNTLET_SIZE, gauntlet
from app.services.sim_report import build_report
from app.services.sim_runs import Progress, SimError, enqueue, entries_of, main_of

logger = logging.getLogger(__name__)

BUILD_MINUTES = 6
NOT_RUNNING_NOTE = "Couldn't playtest this deck (the simulator isn't running), so this is the untested build."
REPLACE_QUESTION = ("Which card should replace the card in `deck.cut`? It underperformed when this deck was "
                    "playtested; prefer a card that helps against the decks in `deck.losing_to`.")
_FACES = re.compile(r"\s+//?\s+")


async def start_build(db: AsyncSession, deck: Dict[str, Any], assembly: Optional[Dict[str, Any]],
                      conversation_id: Optional[UUID], user_id: Optional[UUID],
                      format: str) -> Tuple[Optional[SimulationRun], Optional[str]]:
    """Queue a playtest of a deck Jev assembly built. Returns the run, or a note for
    the user when the simulator can't take it."""
    if not settings.FORGE_ENABLED or format not in SIXTY_CARD_FORMATS or not assembly:
        return None, None
    if not sim_worker_running():
        return None, NOT_RUNNING_NOTE
    archetypes = [assembly["reference"]] if assembly.get("reference") else list(assembly.get("relatives") or [])
    run = SimulationRun(
        id=uuid4(), user_id=user_id, conversation_id=conversation_id, kind="build", status="queued", format=format,
        deck=deck, games_per_matchup=SearchConfig().screen_games, stop_requested=False,
        options={"requested": list(assembly.get("requested") or []), "archetypes": archetypes,
                 "synergy": assembly.get("synergy"), "colors": list(assembly.get("colors") or [])},
        created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.add(run)
    await db.commit()
    enqueue(run.id)
    return run, None


STAPLES_SQL = text(f"""
    WITH lists AS (
      SELECT d.id, d.main_deck
      FROM decklists d JOIN events e ON e.id = d.event_id
      WHERE {RECENT} AND lower(trim(d.archetype)) = ANY(CAST(:archetypes AS varchar[]))
    )
    SELECT lower(split_part(x->>'card_name', ' // ', 1)) AS k
    FROM lists CROSS JOIN LATERAL jsonb_array_elements(lists.main_deck) AS x
    GROUP BY 1
    HAVING COUNT(DISTINCT lists.id) * 2 > (SELECT COUNT(*) FROM lists)
""")


async def staples(db: AsyncSession, archetypes: Sequence[str], format: str, names: Sequence[str]) -> Set[str]:
    """The deck's cards that more than half of the archetypes' recent lists play."""
    if not archetypes:
        return set()
    rows = (await db.execute(STAPLES_SQL, {"format": format, "archetypes": archetype_keys(archetypes)})).all()
    keys = {r.k for r in rows}
    return {n for n in names if _FACES.split(n)[0].lower() in keys}


def make_protect(protected: Set[str], synergy: Optional[Dict[str, Any]], rows: Dict[str, Any]) -> Protect:
    """Never cut a protected card, or a synergy card when the cut leaves fewer than
    synergy["min"] copies of that type."""
    word = synergy["type_contains"].lower() if synergy else None

    def of_type(name: str) -> bool:
        return word is not None and name in rows and word in (rows[name].type_line or "").lower()

    def protect(main: Dict[str, int], cut: str) -> bool:
        if cut in protected:
            return True
        if of_type(cut):
            left = sum(q for n, q in main.items() if of_type(n)) - main[cut]
            return left < synergy["min"]
        return False
    return protect


async def replacement(db: AsyncSession, client, main: Dict[str, int], cut: str, losing: List[str],
                      colors: List[str], format: str, known: Set[str], rows: Dict[str, Any]) -> Optional[str]:
    """Jev's pick, from played cards like `cut` (role, mana value within one), for its place."""
    if client is None:
        return None
    if cut not in rows:
        rows.update({r.name: r for r in await requested_rows(db, [cut], format)})
    row = rows.get(cut)
    if row is None:
        return None
    role = next((r for r in (row.roles or []) if not r.startswith("land_")), None) or (
        "creature" if "Creature" in (row.type_line or "") else "noncreature")
    cmc = int(row.cmc or 0)
    slot = Slot(role, max(0, cmc - 1), cmc + 1, main[cut], f"a replacement for {row.name}: {row.type_line}")
    pool = [r for r in await slot_pool(db, slot, colors, format, list(main)) if forge.forge_name(r.name, known)]
    if not pool:
        return None
    state = {"deck": {"cards": [f"{q}x {n}" for n, q in main.items()], "cut": card_text(row),
                      "losing_to": list(losing)}}
    picks = await fill_slot(client, slot, pool, state, format, lambda name, rank: MAX_COPIES,
                            question=REPLACE_QUESTION)
    return picks[0][0] if picks else None


async def run_build(db: AsyncSession, run: SimulationRun) -> None:
    opts = run.options or {}
    opponents = await gauntlet(db, run.format)
    if not opponents:
        raise SimError("There are no recent decklists to playtest against.")
    seed_main = main_of(run.deck["main_deck"])
    known = forge.card_names()
    missing = forge.missing_cards(seed_main, known)
    rows = {r.name: r for r in await requested_rows(db, list(seed_main), run.format)}
    lands = {n for n, r in rows.items() if is_land(r.type_line)} | BASIC_NAMES
    protected = (set(opts.get("requested") or []) | set(missing)
                 | await staples(db, opts.get("archetypes") or [], run.format, list(seed_main)))
    protect = make_protect(protected, opts.get("synergy"), rows)
    colors = list(opts.get("colors") or colors_of(list(rows.values())))
    cfg = SearchConfig()
    planned = len(opponents) * (cfg.screen_games * (1 + cfg.max_rounds * cfg.candidates) + 2 * cfg.confirm_games)
    progress = Progress(db, run, opponents, planned)
    await progress.deck(seed_main)
    for name in missing:
        await progress.event(f"Forge can't play [[{name}]] yet, so it was left out of testing and kept in your list.")
    matchups = [forge.Matchup(o.archetype, o.main) for o in opponents]

    async def evaluate(main: Dict[str, int], games: int, seed: int):
        async def counted(key, records):
            await progress.games(key, records, tally=False)
        return await forge.play(main, matchups, games, seed, counted)

    async with jev.session(None) as client:
        async def find_add(main: Dict[str, int], cut: str, losing: List[str]) -> Optional[str]:
            try:
                return await replacement(db, client, main, cut, losing, colors, run.format, known, rows)
            except Exception as e:  # a failed pick skips this swap, not the run
                logger.warning(f"[SIM] replacement for {cut} failed: {e}")
                return None
        result = await playtest(seed_main, opponents, lands, evaluate, find_add, protect, progress, cfg,
                                rng_seed=random.randrange(1 << 30))

    run.final_deck = {**run.deck, "main_deck": entries_of(result.main)}
    run.report = build_report(result.final, [r for rs in result.records.values() for r in rs], result.main, lands,
                              missing, baseline=result.baseline, changes=result.changes, stopped=result.stopped)
    run.status = "stopped" if result.stopped == "user" else "completed"
    progress.data["stage"] = "Stopped" if result.stopped == "user" else "Done"
    progress.data["deck"] = entries_of(result.main)
    await progress.save()
```

- [ ] **Step 5: Start the playtest from chat**

In `backend/app/schemas/conversation.py`, add to `ChatResponse`:

```python
    simulation_id: Optional[UUID] = None  # a build's playtest; poll /api/simulations/{id}
```

In `backend/app/services/chat_service.py`, add `from app.services.sim_build import BUILD_MINUTES, start_build` and `from app.services.gauntlet import GAUNTLET_SIZE` to the imports. In `_handle_generate_full_deck`, after `response += result.strategy_summary or ""`, add:

```python
        def entries(es):
            return [e if isinstance(e, dict) else e.model_dump() for e in es or []]

        sim, note = await start_build(
            self.db, {"name": result.deck.name, "main_deck": entries(result.deck.main_deck),
                      "sideboard": entries(result.deck.sideboard)},
            getattr(result, "assembly", None), result.conversation_id, user_id, format)
        if sim:
            response += (f"\n\nPlaytesting against the top {GAUNTLET_SIZE} decks, about {BUILD_MINUTES} minutes. "
                         "The list updates as it improves.")
        elif note:
            response += f"\n\n{note}"
```

and pass `simulation_id=sim.id if sim else None,` to the `ChatResponse(...)` this handler returns.

- [ ] **Step 6: Run the tests**

Run: `cd backend && python3 -m pytest -q` → all pass (including the guard test `test_no_hardcoded_names.py`).

- [ ] **Step 7: Live check**

```bash
S=/private/tmp/claude-501/-Users-robmiller-Projects-mtg-deckbuilder/b8847f7b-d5b1-408b-9720-62bba18fb355/scratchpad
curl -s -X POST localhost:8000/api/conversations/chat -H 'Content-Type: application/json' \
  -d '{"message":"Build me a mono-red aggro deck","mode":"build","format":"standard"}' > $S/b.json
python3 -c "import json; d=json.load(open('$S/b.json')); print(d['simulation_id']); print(d['response'][-160:])"
```

Then poll every 30 s until the status is `completed`, `stopped` or `failed` (at most 10 minutes), printing stage, games, the last 3 events, and at the end the report's overall/baseline and changes:

```bash
ID=<simulation_id>
for i in $(seq 1 20); do
  curl -s localhost:8000/api/simulations/$ID > $S/run.json
  python3 -c "
import json; r=json.load(open('$S/run.json')); p=r.get('progress') or {}
print(r['status'], p.get('stage'), p.get('games_done'), '/', p.get('games_planned'))
for e in (p.get('events') or [])[-3:]: print('  ', e['kind'], e['text'])
if r['status'] in ('completed','stopped','failed'):
    rep=r.get('report') or {}; print('overall', rep.get('overall'), 'baseline', rep.get('baseline'))
    print('changes', rep.get('changes')); print('error', r.get('error'))"
  grep -q '"status":"\(completed\|stopped\|failed\)"' $S/run.json && break
  sleep 30
done
docker logs spellbook-sim-worker --since 15m 2>&1 | grep -iE "error|traceback" | tail -5
```

Expected: the chat reply ends with the "Playtesting against the top 5 decks…" line; the run goes `queued` → `running` → `completed` within about 6-8 minutes; events read as plain sentences; the report has `overall`, `baseline` and zero or more `changes`; no tracebacks. Paste the full poll output in your report.

- [ ] **Step 8: Commit**

```bash
git add backend/app/services/sim_build.py backend/app/services/deck_fill.py backend/app/schemas/deck.py \
  backend/app/services/deck_generator.py backend/app/schemas/conversation.py backend/app/services/chat_service.py \
  backend/tests/test_sim_build.py backend/tests/test_deck_fill.py backend/tests/test_chat_fit_handlers.py
git commit -m "Playtest every chat build with Forge"
```

---

### Task 8: Frontend: live progress, the report, the new Simulate page

**Files:**
- Modify: `frontend/src/types/index.ts` (remove the old simulation types from `// Simulation types` through `SimulationRunListResponse`; add the new ones; `ChatResponse.simulation_id`)
- Modify: `frontend/src/services/api.ts` (replace `simulationApi`)
- Create: `frontend/src/hooks/useSimulation.ts`
- Create: `frontend/src/components/SimulationProgress.tsx`
- Create: `frontend/src/components/SimulationReport.tsx`
- Rewrite: `frontend/src/pages/Simulation.tsx`
- Modify: `frontend/src/store/conversation.ts` (`simulationId`)
- Modify: `frontend/src/hooks/useChat.ts` (store `simulation_id`; clear it on a new conversation)
- Modify: `frontend/src/components/DeckList.tsx` (`highlighted` prop)
- Modify: `frontend/src/pages/Home.tsx` (playtest panel under the deck; deck updates in place)

**Interfaces:**
- Consumes: the API from Task 5 (`/api/simulations`) and `ChatResponse.simulation_id` from Task 7.
- Produces: `SimulationRun`, `SimProgress`, `SimReport` types; `simulationApi.{create, get, list, stop, archetypes}`; `useSimulationRun(id)`; `<SimulationProgress run onStop />`; `<SimulationReport report kind />`.

- [ ] **Step 1: Types and API**

In `frontend/src/types/index.ts`, delete everything from the `// Simulation types` comment through the end of `SimulationRunListResponse` (the old `TurnAction`, `GameResult`, `KeyCardAnalysis`, `DeckRecommendation`, `MatchupAnalysisResult`, `SimulationRun`, `SimulationRunListResponse`), add `simulation_id?: string;` to `ChatResponse`, and add:

```ts
// Forge simulation
export interface SimMatchup {
  opponent: string;
  share: number;
  wins: number;
  losses: number;
  draws: number;
  games?: number;
  win_rate?: number;
  lo?: number;
  hi?: number;
  label?: 'favored' | 'even' | 'unfavored';
  avg_turns?: number;
}

export interface SimEvent {
  at: string;
  text: string;
  kind: 'info' | 'tried' | 'kept';
}

export interface SimProgress {
  stage: string;
  games_done: number;
  games_planned: number;
  started_at: string;
  matchups: SimMatchup[];
  events: SimEvent[];
  deck?: DeckEntry[] | null;
}

export interface SimRate {
  win_rate: number;
  lo: number;
  hi: number;
  games: number;
}

export interface SimCardStat {
  name: string;
  copies: number;
  games_cast: number;
  cast_share: number;
  win_rate_when_cast: number | null;
  median_turn: number | null;
}

export interface SimChange {
  cut: string;
  add: string;
  copies: number;
  before: number;
  after: number;
  best_matchup: { opponent: string; before: number; after: number };
}

export interface SimReport {
  overall: SimRate;
  baseline: SimRate | null;
  matchups: SimMatchup[];
  changes: SimChange[];
  cards: { strongest: SimCardStat[]; weakest: SimCardStat[]; too_few: string[] };
  mana: { mulligan_rate: number; screw_rate: number; flood_rate: number; advice: string[] };
  games: { win: string[] | null; loss: string[] | null };
  not_simulated: { cards: string[]; sideboard: boolean };
  limits: string;
  stopped: string | null;
}

export interface SimulationRun {
  id: string;
  kind: 'test' | 'build';
  status: 'queued' | 'running' | 'completed' | 'failed' | 'stopped';
  format: string;
  deck: { name?: string; main_deck: DeckEntry[]; sideboard?: DeckEntry[] };
  opponents?: string[] | null;
  games_per_matchup: number;
  progress?: SimProgress | null;
  report?: SimReport | null;
  final_deck?: { name?: string; main_deck: DeckEntry[]; sideboard?: DeckEntry[] } | null;
  error?: string | null;
  queue_position?: number | null;
  created_at: string;
  updated_at: string;
}
```

In `frontend/src/services/api.ts`, replace the whole `simulationApi` object (from `// Simulation API` to its closing `};`) with the following, and add `SimulationRun` to the type imports from `@/types`:

```ts
// Forge simulations
export const simulationApi = {
  create: (body: {
    deck_id?: string;
    deck?: { name?: string; main_deck: DeckEntry[]; sideboard?: DeckEntry[] };
    format: string;
    opponents?: string[];
    games?: number;
  }) => api.post<SimulationRun>('/simulations', body),

  get: (id: string) => api.get<SimulationRun>(`/simulations/${id}`),

  list: () => api.get<SimulationRun[]>('/simulations'),

  stop: (id: string) => api.post<SimulationRun>(`/simulations/${id}/stop`),

  archetypes: (format: string) => api.get<string[]>('/simulations/archetypes', { params: { format } }),
};
```

(Import `DeckEntry` too if `api.ts` does not already.)

- [ ] **Step 2: The polling hook**

Create `frontend/src/hooks/useSimulation.ts`:

```ts
import { useQuery } from '@tanstack/react-query';
import { simulationApi } from '@/services/api';
import { SimulationRun } from '@/types';

const ACTIVE = new Set<SimulationRun['status']>(['queued', 'running']);

export const isActive = (run?: SimulationRun | null) => !!run && ACTIVE.has(run.status);

// Polls every 2 s while the run is queued or running, then stops.
export function useSimulationRun(id: string | null | undefined) {
  return useQuery({
    queryKey: ['simulation', id],
    queryFn: async () => (await simulationApi.get(id!)).data,
    enabled: !!id,
    refetchInterval: (query) => (query.state.data && !isActive(query.state.data) ? false : 2000),
  });
}
```

- [ ] **Step 3: The progress panel**

Create `frontend/src/components/SimulationProgress.tsx`:

```tsx
import clsx from 'clsx';
import { SimulationRun } from '@/types';

const pct = (x: number) => `${Math.round(100 * x)}%`;

function timeLeft(run: SimulationRun): string | null {
  const p = run.progress;
  if (!p || !p.games_done || p.games_done >= p.games_planned) return null;
  const elapsed = (Date.now() - new Date(p.started_at).getTime()) / 1000;
  const left = (elapsed / p.games_done) * (p.games_planned - p.games_done);
  const minutes = Math.ceil(left / 60);
  return minutes <= 1 ? 'under a minute left' : `up to ${minutes} min left`;
}

interface Props {
  run: SimulationRun;
  onStop?: () => void;
  stopping?: boolean;
}

export function SimulationProgress({ run, onStop, stopping }: Props) {
  const p = run.progress;
  if (run.status === 'queued') {
    return (
      <div className="bg-gray-900 rounded-lg p-4 text-sm text-gray-300">
        Waiting for the simulator
        {run.queue_position ? ` (${run.queue_position - 1} ahead of you)` : ''}…
      </div>
    );
  }
  const done = p ? Math.min(1, p.games_done / Math.max(1, p.games_planned)) : 0;
  const left = timeLeft(run);
  return (
    <div className="bg-gray-900 rounded-lg p-4 space-y-3 text-sm">
      <div className="flex items-center justify-between gap-2">
        <div className="text-white font-medium">{p?.stage ?? 'Starting'}</div>
        {onStop && (
          <button
            onClick={onStop}
            disabled={stopping || run.stop_requested}
            className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-200 disabled:opacity-50"
          >
            {stopping ? 'Stopping…' : 'Stop'}
          </button>
        )}
      </div>
      <div>
        <div className="h-2 rounded bg-gray-800 overflow-hidden">
          <div className="h-2 bg-primary-500 transition-all" style={{ width: `${Math.round(100 * done)}%` }} />
        </div>
        <div className="mt-1 text-xs text-gray-400">
          {p?.games_done ?? 0} games played{left ? ` · ${left}` : ''}
        </div>
      </div>
      {p && p.matchups.length > 0 && (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 text-left">
              <th className="font-normal">Opponent</th>
              <th className="font-normal text-right">Meta</th>
              <th className="font-normal text-right">W–L</th>
              <th className="font-normal text-right">Win</th>
            </tr>
          </thead>
          <tbody>
            {p.matchups.map((m) => {
              const games = m.wins + m.losses + m.draws;
              const rate = games ? (m.wins + m.draws / 2) / games : null;
              return (
                <tr key={m.opponent} className="text-gray-200">
                  <td className="py-0.5">{m.opponent}</td>
                  <td className="text-right text-gray-400">{m.share > 1 ? `${m.share.toFixed(1)}%` : ''}</td>
                  <td className="text-right">{m.wins}–{m.losses}{m.draws ? `–${m.draws}` : ''}</td>
                  <td className={clsx('text-right', rate !== null && (rate > 0.55 ? 'text-green-400' : rate < 0.45 ? 'text-red-400' : 'text-gray-200'))}>
                    {rate === null ? '—' : pct(rate)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {p && p.events.length > 0 && (
        <ul className="space-y-1 text-xs max-h-40 overflow-y-auto">
          {[...p.events].reverse().slice(0, 8).map((e, i) => (
            <li key={i} className={clsx(e.kind === 'kept' ? 'text-green-300' : e.kind === 'tried' ? 'text-gray-400' : 'text-gray-300')}>
              {e.text.replace(/\[\[([^\]]+)\]\]/g, '$1')}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

Add `stop_requested?: boolean;` to the `SimulationRun` type (the API already returns it; the panel disables Stop once requested).

- [ ] **Step 4: The report**

Create `frontend/src/components/SimulationReport.tsx`:

```tsx
import { useState } from 'react';
import clsx from 'clsx';
import { CardTooltip } from '@/components/CardTooltip';
import { SimReport } from '@/types';

const pct = (x: number) => `${Math.round(100 * x)}%`;
const LABEL_STYLE = { favored: 'text-green-400', even: 'text-gray-300', unfavored: 'text-red-400' } as const;
const STOPPED: Record<string, string> = {
  user: 'Stopped early; this is the best deck found so far.',
  budget: 'The search ran out of time; this is the best deck found.',
  reverted: "The swaps didn't hold up over more games, so the first draft stands.",
};

interface Props {
  report: SimReport;
  kind: 'test' | 'build';
  compact?: boolean;
}

export function SimulationReport({ report, kind, compact }: Props) {
  const [game, setGame] = useState<'win' | 'loss' | null>(null);
  const { overall, baseline } = report;
  return (
    <div className="bg-gray-900 rounded-lg p-4 space-y-4 text-sm text-gray-200">
      <div>
        <div className="text-2xl font-semibold text-white">{pct(overall.win_rate)}</div>
        <div className="text-gray-400">
          against the top decks, likely {pct(overall.lo)}–{pct(overall.hi)} ({overall.games} games)
        </div>
        {kind === 'build' && baseline && (
          <div className="text-gray-400 mt-1">First draft: {pct(baseline.win_rate)}</div>
        )}
        {report.stopped && STOPPED[report.stopped] && (
          <div className="text-amber-300 mt-1">{STOPPED[report.stopped]}</div>
        )}
      </div>

      <section>
        <h3 className="text-white font-medium mb-1">Matchups</h3>
        <table className="w-full text-xs">
          <tbody>
            {report.matchups.map((m) => (
              <tr key={m.opponent}>
                <td className="py-0.5">{m.opponent}</td>
                <td className="text-right">{m.win_rate !== undefined ? pct(m.win_rate) : '—'}</td>
                <td className="text-right text-gray-500">
                  {m.lo !== undefined && m.hi !== undefined ? `${pct(m.lo)}–${pct(m.hi)}` : ''}
                </td>
                <td className={clsx('text-right', m.label && LABEL_STYLE[m.label])}>{m.label}</td>
                {!compact && <td className="text-right text-gray-500">{m.avg_turns ? `ends turn ${m.avg_turns.toFixed(1)}` : ''}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {report.changes.length > 0 && (
        <section>
          <h3 className="text-white font-medium mb-1">What changed</h3>
          <ul className="space-y-1 text-xs">
            {report.changes.map((c, i) => (
              <li key={i}>
                −{c.copies} <CardTooltip cardName={c.cut}>{c.cut}</CardTooltip> +{c.copies}{' '}
                <CardTooltip cardName={c.add}>{c.add}</CardTooltip>: {pct(c.before)} → {pct(c.after)} overall; vs{' '}
                {c.best_matchup.opponent} {pct(c.best_matchup.before)} → {pct(c.best_matchup.after)}
              </li>
            ))}
          </ul>
        </section>
      )}

      {!compact && (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          {(['strongest', 'weakest'] as const).map((k) => (
            <div key={k}>
              <h3 className="text-white font-medium mb-1 text-sm">{k === 'strongest' ? 'Pulling weight' : 'Underperforming'}</h3>
              <ul className="space-y-0.5">
                {report.cards[k].map((c) => (
                  <li key={c.name}>
                    <CardTooltip cardName={c.name}>{c.name}</CardTooltip>{' '}
                    <span className="text-gray-400">
                      won {pct(c.win_rate_when_cast ?? 0)} when cast · cast in {c.games_cast} games
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
          {report.cards.too_few.length > 0 && (
            <p className="text-gray-500 md:col-span-2">
              Cast too rarely to judge: {report.cards.too_few.join(', ')}
            </p>
          )}
        </section>
      )}

      <section className="text-xs">
        <h3 className="text-white font-medium mb-1 text-sm">Mana</h3>
        <p className="text-gray-400">
          Mulligans {pct(report.mana.mulligan_rate)} · stuck on lands {pct(report.mana.screw_rate)} · flooded{' '}
          {pct(report.mana.flood_rate)}
        </p>
        {report.mana.advice.map((a) => (
          <p key={a} className="text-amber-300">{a}</p>
        ))}
      </section>

      {!compact && (report.games.win || report.games.loss) && (
        <section className="text-xs">
          <div className="flex gap-2 mb-1">
            <h3 className="text-white font-medium text-sm mr-2">Watch a game</h3>
            {report.games.win && (
              <button className="underline text-gray-300" onClick={() => setGame(game === 'win' ? null : 'win')}>a win</button>
            )}
            {report.games.loss && (
              <button className="underline text-gray-300" onClick={() => setGame(game === 'loss' ? null : 'loss')}>a loss</button>
            )}
          </div>
          {game && (
            <ol className="max-h-64 overflow-y-auto bg-gray-950 rounded p-2 space-y-0.5 font-mono">
              {(report.games[game] ?? []).map((line, i) => (
                <li key={i} className={line.startsWith('Turn ') ? 'text-white mt-1' : 'text-gray-400'}>
                  {line.replace(/^Tested/, 'You').replace(/^Opponent/, 'Opponent')}
                </li>
              ))}
            </ol>
          )}
        </section>
      )}

      <footer className="text-xs text-gray-500 space-y-0.5">
        <p>{report.limits}</p>
        {report.not_simulated.cards.length > 0 && (
          <p>Not simulated (Forge can't play them yet): {report.not_simulated.cards.join(', ')}.</p>
        )}
        <p>The sideboard isn't tested: games are best-of-one.</p>
      </footer>
    </div>
  );
}
```

- [ ] **Step 5: The Simulate page**

Overwrite `frontend/src/pages/Simulation.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { decksApi, simulationApi } from '@/services/api';
import { useAuth } from '@/hooks/useAuth';
import { isActive, useSimulationRun } from '@/hooks/useSimulation';
import { SimulationProgress } from '@/components/SimulationProgress';
import { SimulationReport } from '@/components/SimulationReport';
import { Deck, SimulationRun } from '@/types';

const GAME_OPTIONS = [20, 50, 100, 200];

export function SimulationPage() {
  const { isAuthenticated } = useAuth();
  const [params, setParams] = useSearchParams();
  const [decks, setDecks] = useState<Deck[]>([]);
  const [deckId, setDeckId] = useState(params.get('deck') ?? '');
  const [archetypes, setArchetypes] = useState<string[]>([]);
  const [chosen, setChosen] = useState<string[]>([]);
  const [games, setGames] = useState(50);
  const [runs, setRuns] = useState<SimulationRun[]>([]);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const runId = params.get('run');
  const { data: run } = useSimulationRun(runId);

  useEffect(() => {
    if (isAuthenticated) {
      decksApi.list(100, 0).then((r) => setDecks(r.data.items ?? r.data ?? [])).catch(() => setDecks([]));
      simulationApi.list().then((r) => setRuns(r.data)).catch(() => setRuns([]));
    }
    simulationApi.archetypes('standard').then((r) => setArchetypes(r.data.slice(0, 12))).catch(() => setArchetypes([]));
  }, [isAuthenticated, run?.status]);

  const start = async () => {
    const deck = decks.find((d) => d.id === deckId);
    if (!deck) return;
    setStarting(true);
    try {
      const { data } = await simulationApi.create({
        deck_id: deck.id, format: deck.format, games, opponents: chosen.length ? chosen : undefined,
      });
      setParams({ run: data.id });
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? "Couldn't start the test");
    } finally {
      setStarting(false);
    }
  };

  const stop = async () => {
    if (!run) return;
    setStopping(true);
    try {
      await simulationApi.stop(run.id);
    } finally {
      setStopping(false);
    }
  };

  const toggle = (a: string) => setChosen((c) => (c.includes(a) ? c.filter((x) => x !== a) : [...c, a]));

  return (
    <div className="max-w-5xl mx-auto p-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="space-y-4">
        <div className="bg-gray-900 rounded-lg p-4 space-y-3 text-sm">
          <h1 className="text-lg font-semibold text-white">Test a deck</h1>
          <p className="text-gray-400">
            Plays your deck against real lists from the current meta with Forge, a Magic rules engine.
          </p>
          <label className="block">
            <span className="text-gray-300">Deck</span>
            <select value={deckId} onChange={(e) => setDeckId(e.target.value)}
                    className="mt-1 w-full bg-gray-800 text-white rounded p-2">
              <option value="">Choose a saved deck…</option>
              {decks.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </label>
          <div>
            <span className="text-gray-300">Opponents</span>
            <p className="text-xs text-gray-500">None chosen: the top 5 decks, weighted by meta share.</p>
            <div className="mt-1 flex flex-wrap gap-1">
              {archetypes.map((a) => (
                <button key={a} onClick={() => toggle(a)}
                        className={`text-xs px-2 py-1 rounded ${chosen.includes(a) ? 'bg-primary-600 text-white' : 'bg-gray-800 text-gray-300'}`}>
                  {a}
                </button>
              ))}
            </div>
          </div>
          <label className="block">
            <span className="text-gray-300">Games per opponent</span>
            <select value={games} onChange={(e) => setGames(Number(e.target.value))}
                    className="mt-1 w-full bg-gray-800 text-white rounded p-2">
              {GAME_OPTIONS.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          </label>
          <button onClick={start} disabled={!deckId || starting}
                  className="w-full py-2 rounded bg-primary-600 hover:bg-primary-500 text-white disabled:opacity-50">
            {starting ? 'Starting…' : 'Run test'}
          </button>
          {!isAuthenticated && <p className="text-xs text-gray-500">Sign in to test your saved decks.</p>}
        </div>
        {runs.length > 0 && (
          <div className="bg-gray-900 rounded-lg p-4 text-sm">
            <h2 className="text-white font-medium mb-2">Recent runs</h2>
            <ul className="space-y-1">
              {runs.map((r) => (
                <li key={r.id}>
                  <button onClick={() => setParams({ run: r.id })} className="text-left w-full text-gray-300 hover:text-white">
                    {r.deck.name ?? 'Deck'} · {r.kind === 'build' ? 'build playtest' : 'test'} · {r.status}
                    {r.report ? ` · ${Math.round(100 * r.report.overall.win_rate)}%` : ''}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div className="lg:col-span-2 space-y-4">
        {!run && <div className="bg-gray-900 rounded-lg p-8 text-center text-gray-400">Choose a deck and run a test.</div>}
        {run && isActive(run) && <SimulationProgress run={run} onStop={stop} stopping={stopping} />}
        {run && run.status === 'failed' && (
          <div className="bg-gray-900 rounded-lg p-4 text-red-300 text-sm">{run.error ?? 'The test failed.'}</div>
        )}
        {run?.report && <SimulationReport report={run.report} kind={run.kind} />}
      </div>
    </div>
  );
}
```

Check `decksApi.list`'s response shape: `grep -n "items" frontend/src/pages/Decks.tsx | head -3`. If decks come back as `{ items: [...] }` keep `r.data.items`; if as a bare list, the `?? r.data` fallback covers it.

- [ ] **Step 6: The playtest under the Build page's deck**

In `frontend/src/store/conversation.ts`: add `simulationId: string | null;` and `setSimulationId: (id: string | null) => void;` to the state interface; `simulationId: null,` to the initial state; `setSimulationId: (simulationId) => set({ simulationId }),` to the actions; `simulationId: null,` inside `reset`; and `simulationId: state.simulationId,` to `partialize`.

In `frontend/src/hooks/useChat.ts`: take `setSimulationId` from `useConversationStore()`; next to `if (data.deck) { setCurrentDeck(data.deck); }` add `if (data.deck) { setSimulationId(data.simulation_id ?? null); }`; and wherever a new conversation clears the deck (`setCurrentDeck(null)`), also call `setSimulationId(null)`. Add `setSimulationId` to the `useCallback` dependency arrays that use it.

In `frontend/src/components/DeckList.tsx`: add `highlighted?: string[];` to `DeckListProps`, destructure it, build `const highlightedSet = new Set((highlighted ?? []).map(normalizeCardName));`, pass `highlighted={highlightedSet.has(normalizeCardName(entry.card_name))}` wherever `flagged={...}` is passed to `CardEntry`, add `highlighted?: boolean;` to `CardEntryProps` and the `CardEntry` parameters, and change the card name span to:

```tsx
<span className={clsx('text-sm', highlighted ? 'text-green-300' : 'text-white')}>{entry.card_name}</span>
```

In `frontend/src/pages/Home.tsx`, add the imports:

```tsx
import { useEffect, useRef } from 'react';  // merge with the existing react import
import { simulationApi } from '@/services/api';  // merge with the existing api import
import { isActive, useSimulationRun } from '@/hooks/useSimulation';
import { SimulationProgress } from '@/components/SimulationProgress';
import { SimulationReport } from '@/components/SimulationReport';
```

Inside the component, after the existing store hooks:

```tsx
  const { simulationId } = useConversationStore();
  const { data: playtest } = useSimulationRun(simulationId);
  const [added, setAdded] = useState<string[]>([]);
  const lastDeck = useRef<string>('');

  // A kept swap changes the playtest's deck: show it in place and highlight what came in.
  useEffect(() => {
    const entries = playtest?.progress?.deck;
    if (!entries || !currentDeck) return;
    const key = JSON.stringify(entries);
    if (key === lastDeck.current) return;
    const before = new Set((currentDeck.main_deck || []).map((e) => e.card_name));
    if (lastDeck.current) setAdded(entries.filter((e) => !before.has(e.card_name)).map((e) => e.card_name));
    lastDeck.current = key;
    setCurrentDeck({ ...currentDeck, main_deck: entries });
  }, [playtest?.progress?.deck]);
```

(Use the deck store's existing `setCurrentDeck`; if `Home.tsx` doesn't already take it from `useDeckStore()`, add it.) Pass `highlighted={added}` to the `DeckList`, and right after `{isAuthenticated && (<DeckActions deck={currentDeck} />)}` add:

```tsx
            {playtest && isActive(playtest) && (
              <SimulationProgress run={playtest} onStop={() => simulationApi.stop(playtest.id)} />
            )}
            {playtest?.status === 'failed' && (
              <div className="bg-gray-900 rounded-lg p-3 text-xs text-red-300">{playtest.error}</div>
            )}
            {playtest?.report && (
              <>
                <SimulationReport report={playtest.report} kind="build" compact />
                <a href={`/simulate?run=${playtest.id}`} className="text-xs text-primary-400 hover:underline">
                  Full playtest report
                </a>
              </>
            )}
```

- [ ] **Step 7: Type-check and look at it**

Run: `cd frontend && ./node_modules/.bin/tsc --noEmit -p . && echo tsc-ok`
Expected: `tsc-ok`. Fix any type errors in the files above (not by loosening types to `any` beyond the existing `catch (e: any)`).

Then in the browser (http://localhost:3000, the frontend container hot-reloads):
1. On Build, send "Build me a mono-red aggro deck". The deck appears; under it the progress panel shows the stage, a filling bar with time left, the matchup table and plain-sentence events; Stop is enabled.
2. When a swap is kept, the deck list updates and the new card shows in green.
3. At the end the compact report shows the headline, matchups, what changed and mana; "Full playtest report" opens `/simulate?run=…` with the full report including "Watch a game".
4. On `/simulate`, pick a saved deck, run a 20-game test against the top 5, watch progress, then press Stop on a second run and see the partial report with "Stopped early".

Take a screenshot of the Build page during a playtest and of the full report, and attach both to your report.

- [ ] **Step 8: Commit**

```bash
git add frontend/src
git commit -m "Show live playtest progress and reports; new Simulate page"
```

---

### Task 9: Live eval of playtested builds

**Files:**
- Create: `backend/scripts/eval_playtest.py`

**Interfaces:**
- Consumes: chat (`POST /api/conversations/chat` with `mode: "build"`), `GET /api/simulations/{id}`.
- Produces: a script that builds the three decks from review through the chat, waits for each playtest, prints seed vs final with the report, and checks the invariants. Not run in CI.

- [ ] **Step 1: Write the script**

Create `backend/scripts/eval_playtest.py`:

```python
"""
Build three decks through the Build page's chat, wait for each Forge playtest, and
check: the final deck is 60 cards, no requested card was cut, every kept swap is in
the report, and the confirmed win rate is not below the first draft's. Prints the
lists and reports for human review. Needs the stack and the sim worker running.

    python scripts/eval_playtest.py [--base http://localhost:8000]
"""

import argparse
import json
import sys
import time
import urllib.request

CASES = [
    ("Build me a mono-red aggro deck", []),
    ("I want to build a deck around Weapons Manufacturing", ["Weapons Manufacturing"]),
    ("I want a deck that can beat the current meta", []),
]
TIMEOUT_S = 15 * 60


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def counts(entries):
    out = {}
    for e in entries:
        out[e["card_name"]] = out.get(e["card_name"], 0) + int(e["quantity"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    base = parser.parse_args().base + "/api"
    failures = []
    for ask, requested in CASES:
        print(f"\n=== {ask}")
        chat = call(base, "/conversations/chat", {"message": ask, "mode": "build", "format": "standard"})
        sim_id = chat.get("simulation_id")
        if not chat.get("deck"):
            failures.append(f"{ask}: no deck ({chat.get('response', '')[:120]})")
            continue
        if not sim_id:
            print("not playtested:", chat["response"][-200:])
            failures.append(f"{ask}: no playtest started")
            continue
        start = time.time()
        while True:
            run = call(base, f"/simulations/{sim_id}")
            if run["status"] in ("completed", "stopped", "failed") or time.time() - start > TIMEOUT_S:
                break
            time.sleep(15)
        print(f"status={run['status']} after {time.time() - start:.0f}s error={run.get('error')}")
        if run["status"] != "completed":
            failures.append(f"{ask}: playtest {run['status']}")
            continue
        seed, final = counts(run["deck"]["main_deck"]), counts(run["final_deck"]["main_deck"])
        report = run["report"]
        print("first draft:", "; ".join(f"{q} {n}" for n, q in seed.items()))
        print("final:      ", "; ".join(f"{q} {n}" for n, q in final.items()))
        print("overall", report["overall"], "baseline", report["baseline"], "stopped", report["stopped"])
        for c in report["changes"]:
            print(f"  -{c['copies']} {c['cut']} +{c['copies']} {c['add']}: {c['before']:.2f} -> {c['after']:.2f}")
        for e in (run["progress"] or {}).get("events", [])[-6:]:
            print("  event:", e["text"])
        checks = {
            "60 cards": sum(final.values()) == 60,
            "requested kept": all(final.get(r, 0) >= seed.get(r, 0) for r in requested),
            "changes explain the diff": {c["add"] for c in report["changes"]} >= {n for n in final if n not in seed},
            "not worse than the draft": report["baseline"] is None
                                        or report["overall"]["win_rate"] >= report["baseline"]["win_rate"],
        }
        for name, ok in checks.items():
            print(f"  {'OK ' if ok else 'FAIL'} {name}")
            if not ok:
                failures.append(f"{ask}: {name}")
    print("\nPASS" if not failures else "\nFAIL:\n" + "\n".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd /Users/robmiller/Projects/mtg-deckbuilder/backend && python3 scripts/eval_playtest.py`
Expected: three playtests complete (about 6-8 minutes each), every check `OK`, final line `PASS`. Paste the full output into your report. A `FAIL` on "not worse than the draft" or "changes explain the diff" is a search bug: debug it (systematic-debugging) with a failing unit test in `test_deck_search.py` before fixing.

- [ ] **Step 3: Commit**

```bash
git add backend/scripts/eval_playtest.py
git commit -m "Live eval for playtested builds"
```
