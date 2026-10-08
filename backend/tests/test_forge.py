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
        by_dir_seed = sorted(((d, s, n) for d, n, s, _ in calls), key=lambda x: (x[0], x[1]))
        expected = [("0", 7, 10), ("0", 8, 10), ("0", 9, 5), ("1", 7, 10), ("1", 8, 10), ("1", 9, 5)]
        assert by_dir_seed == expected
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

    async def test_raises_when_forge_produces_no_games(self, monkeypatch):
        monkeypatch.setattr(forge, "command", lambda d, g, s: [sys.executable, "-c", "print('Simulation mode\\nAi(1) vs Ai(2)'); print('Turn: Turn 1')"])
        monkeypatch.setattr(forge, "card_names", lambda: KNOWN)
        with pytest.raises(forge.ForgeError, match="played no games"):
            await forge.play({"Forest": 60}, [forge.Matchup("A", {"Forest": 60})], games=1, seed=1)

    async def test_run_timeout_raises_forge_error(self):
        import time
        start = time.time()
        with pytest.raises(forge.ForgeError, match="took longer"):
            await forge._run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
        elapsed = time.time() - start
        assert elapsed < 10, f"Timeout took {elapsed}s, should cancel quickly"
