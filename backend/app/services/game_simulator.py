"""
Game Simulator Service

Uses LLM to simulate Magic: The Gathering games between two decks.
The LLM plays both sides, tracking game state and making decisions.
"""

from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime
import logging
import json
import random

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.core.config import settings
from app.models.deck import Deck
from app.models.meta import Decklist, Event
from app.models.card import Card
from app.models.simulation import SimulationRun
from app.services.card_service import CardService
from app.schemas.simulation import (
    DeckInput,
    GameResult,
    MatchupAnalysisResult,
    SimulationRequest,
)

logger = logging.getLogger(__name__)


# Maximum turns before declaring a draw
MAX_TURNS = 20

# System prompt for game simulation
GAME_SYSTEM_PROMPT = """You are a Magic: The Gathering game simulator. You will play out a game between two decks, making optimal decisions for both players.

RULES:
1. Follow official MTG rules strictly
2. Make the best possible play for whichever player's turn it is
3. Track life totals, cards in hand, battlefield, graveyard
4. Consider mana efficiency, tempo, and card advantage
5. Be realistic about combat math and sequencing

OUTPUT FORMAT:
Return JSON with this structure:
{
    "turns": [
        {
            "turn_number": 1,
            "active_player": "player1",
            "life_totals": {"player1": 20, "player2": 20},
            "actions": [
                "Drew [Card Name]",
                "Played Mountain",
                "Cast Goblin Guide, attacked for 2"
            ],
            "board_state": {
                "player1": {"lands": [...], "creatures": [...], "other": [...]},
                "player2": {"lands": [...], "creatures": [...], "other": [...]}
            },
            "hands": {
                "player1": 6,
                "player2": 7
            }
        }
    ],
    "winner": "player1",
    "win_condition": "damage",
    "final_life": {"player1": 12, "player2": 0},
    "total_turns": 7,
    "key_moments": [
        "Turn 3: Player 2's removal spell on Goblin Guide prevented early pressure",
        "Turn 6: Player 1 topdecked Lightning Strike for lethal"
    ],
    "mvp_cards": {
        "player1": ["Goblin Guide", "Lightning Strike"],
        "player2": ["Fatal Push"]
    }
}

IMPORTANT:
- Simulate the full game from opening hands to conclusion
- Both players mulligan optimally (mulligan poor hands, keep good ones)
- Games should end within 15 turns typically
- If game is clearly decided, you can end early with "concede" as win condition
- Be concise in actions but clear about what happens"""


# System prompt for multiplayer Commander games
MULTIPLAYER_SYSTEM_PROMPT = """You are a Magic: The Gathering Commander game simulator. You will play out a multiplayer game (3-4 players), making optimal decisions for all players.

COMMANDER RULES:
1. Starting life total: 40 per player
2. Commander damage: 21 damage from a single commander eliminates a player
3. Command zone: Commanders can be cast from command zone with +2 mana each time
4. Free-for-all: Players can attack/target any opponent
5. Politics matter: Consider threat assessment, temporary alliances, and optimal targeting

MULTIPLAYER DYNAMICS:
- Players should avoid becoming the "archenemy" too early
- Threat assessment: Focus removal on the biggest threats
- Board wipes become more valuable
- Card advantage and resilience matter more than raw speed

OUTPUT FORMAT:
Return JSON with this structure:
{
    "turns": [
        {
            "turn_number": 1,
            "active_player": "player1",
            "life_totals": {"player1": 40, "player2": 40, "player3": 40, "player4": 40},
            "commander_damage": {"player1": {}, "player2": {}, "player3": {}, "player4": {}},
            "actions": ["Drew [Card]", "Played Sol Ring", "Cast Commander"],
            "eliminations": []
        }
    ],
    "winner": "player1",
    "elimination_order": ["player3", "player4", "player2"],
    "win_condition": "damage",
    "final_life": {"player1": 23, "player2": 0, "player3": 0, "player4": 0},
    "total_turns": 12,
    "key_moments": [
        "Turn 5: Player 2 cast board wipe, resetting the game",
        "Turn 8: Player 1's commander dealt lethal commander damage to Player 3"
    ],
    "mvp_cards": {
        "player1": ["Sol Ring", "Commander"],
        "player2": ["Wrath of God"],
        "player3": ["Rhystic Study"],
        "player4": ["Mana Crypt"]
    }
}

IMPORTANT:
- Players are eliminated when reaching 0 life or 21 commander damage from one source
- Game ends when one player remains
- Be concise but track all eliminations and key plays
- Commander games typically last 10-15 turns"""


class GameSimulator:
    """
    Service for simulating MTG games between decks using LLM.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.card_service = CardService(db)

    async def simulate_match(
        self,
        your_deck: DeckInput,
        opponent_deck: DeckInput,
        num_games: int = 5,
        include_sideboard_games: bool = True,
        format: str = "standard",
        on_progress: Optional[Any] = None,  # Callback(games_completed, current_turn)
        opponent_decks: Optional[List[DeckInput]] = None,  # For multiplayer
        num_players: int = 2,
    ) -> MatchupAnalysisResult:
        """
        Simulate a match (multiple games) between decks.

        Args:
            your_deck: Your deck configuration
            opponent_deck: Opponent deck configuration (for 2-player)
            num_games: Number of games to simulate
            include_sideboard_games: Whether games 2+ use sideboards
            on_progress: Optional callback for progress updates
            opponent_decks: List of opponent decks (for multiplayer, up to 3)
            num_players: Number of players (2-4)

        Returns:
            MatchupAnalysisResult with statistics and insights
        """
        # Resolve deck data
        your_deck_data = await self._resolve_deck(your_deck)
        if not your_deck_data:
            raise ValueError("Could not resolve your deck data")

        # Handle multiplayer vs 2-player
        is_multiplayer = num_players > 2
        opponent_deck_datas = []

        if is_multiplayer and opponent_decks:
            for opp_deck in opponent_decks:
                opp_data = await self._resolve_deck(opp_deck)
                if opp_data:
                    opponent_deck_datas.append(opp_data)
            if not opponent_deck_datas:
                raise ValueError("Could not resolve any opponent deck data")
        else:
            opponent_deck_data = await self._resolve_deck(opponent_deck)
            if not opponent_deck_data:
                raise ValueError("Could not resolve opponent deck data")
            opponent_deck_datas = [opponent_deck_data]

        # Enrich decks with card oracle text
        your_deck_enriched = await self._enrich_deck(your_deck_data)
        opponent_decks_enriched = [await self._enrich_deck(d) for d in opponent_deck_datas]

        games: List[GameResult] = []
        your_wins = 0
        first_place_count = 0
        total_turns = 0
        placement_sum = 0

        for game_num in range(1, num_games + 1):
            # Games 2+ can use sideboards (only for 2-player)
            use_sideboard = include_sideboard_games and game_num > 1 and not is_multiplayer

            if is_multiplayer:
                game_result = await self._simulate_multiplayer_game(
                    your_deck=your_deck_enriched,
                    opponent_decks=opponent_decks_enriched,
                    game_number=game_num,
                    num_players=num_players,
                )
                # In multiplayer, track placement
                if game_result.winner == "you":
                    first_place_count += 1
                    placement_sum += 1
                elif game_result.elimination_order:
                    # Find your placement (elimination_order is who got eliminated, in order)
                    # If you're not in elimination_order, you won (1st place)
                    # Otherwise, your placement is num_players - index_in_elimination
                    try:
                        elim_index = game_result.elimination_order.index("you")
                        placement_sum += num_players - elim_index
                    except ValueError:
                        # Not in elimination order = winner
                        placement_sum += 1
                        first_place_count += 1
                else:
                    placement_sum += 1 if game_result.winner == "you" else num_players
            else:
                game_result = await self._simulate_single_game(
                    your_deck=your_deck_enriched,
                    opponent_deck=opponent_decks_enriched[0],
                    game_number=game_num,
                    use_sideboard=use_sideboard,
                    on_the_play=(game_num % 2 == 1),
                )
                if game_result.winner == "you":
                    your_wins += 1

            games.append(game_result)
            total_turns += game_result.turns

            if on_progress:
                await on_progress(game_num, None)

        # Calculate statistics
        avg_game_length = total_turns / num_games

        if is_multiplayer:
            win_rate = first_place_count / num_games
            placement_avg = placement_sum / num_games
        else:
            win_rate = your_wins / num_games
            placement_avg = None

        # Determine matchup assessment
        if is_multiplayer:
            # For multiplayer, favorable means avg placement < 2
            if placement_avg and placement_avg < 2:
                assessment = "favored"
            elif placement_avg and placement_avg > 2.5:
                assessment = "unfavored"
            else:
                assessment = "even"
        else:
            if win_rate >= 0.6:
                assessment = "favored"
            elif win_rate <= 0.4:
                assessment = "unfavored"
            else:
                assessment = "even"

        # Aggregate key cards across games
        your_key_cards = self._aggregate_key_cards(games, "your_key_cards")
        opponent_key_cards = self._aggregate_key_cards(games, "opponent_key_cards")

        # Generate strategic analysis
        analysis = await self._generate_matchup_analysis(
            your_deck=your_deck_enriched,
            opponent_deck=opponent_decks_enriched[0],  # Use first opponent for analysis
            games=games,
            win_rate=win_rate,
            is_multiplayer=is_multiplayer,
        )

        # Build opponent name(s)
        if is_multiplayer:
            opponent_names = [d.get("name", "Opponent") for d in opponent_deck_datas]
            combined_name = " vs ".join(opponent_names)
        else:
            combined_name = opponent_deck_datas[0].get("name", "Opponent Deck")
            opponent_names = None

        return MatchupAnalysisResult(
            your_deck_name=your_deck_data.get("name", "Your Deck"),
            opponent_deck_name=combined_name,
            opponent_deck_names=opponent_names,
            num_players=num_players,
            games_played=num_games,
            your_wins=your_wins if not is_multiplayer else first_place_count,
            opponent_wins=num_games - your_wins if not is_multiplayer else 0,
            first_place_count=first_place_count if is_multiplayer else None,
            your_placement_avg=placement_avg,
            win_rate=win_rate,
            average_game_length=avg_game_length,
            matchup_assessment=assessment,
            key_cards_for_you=your_key_cards,
            key_cards_against_you=opponent_key_cards,
            sideboard_guide=analysis.get("sideboard_guide", {"in": [], "out": []}),
            strategic_advice=analysis.get("strategic_advice", []),
            mulligan_advice=analysis.get("mulligan_advice", "Keep hands with a good curve and interaction."),
            deck_recommendations=analysis.get("deck_recommendations", []),
            games=games,
        )

    async def simulate_vs_archetype(
        self,
        deck_id: UUID,
        opponent_archetype: str,
        num_games: int = 5,
    ) -> MatchupAnalysisResult:
        """
        Simulate games against a meta archetype.
        Fetches a representative decklist from tournament data.
        """
        # Get user's deck
        result = await self.db.execute(
            select(Deck).where(Deck.id == deck_id)
        )
        deck = result.scalar_one_or_none()
        if not deck:
            raise ValueError(f"Deck {deck_id} not found")

        # Get format from user's deck
        deck_format = deck.format or "standard"

        # Find a tournament decklist for the archetype in the same format
        opponent_decklist = await self._get_archetype_decklist(opponent_archetype, deck_format)
        if not opponent_decklist:
            raise ValueError(f"No {deck_format} decklists found for archetype: {opponent_archetype}")

        your_deck = DeckInput(
            deck_id=deck_id,
            name=deck.name,
        )

        opponent_deck = DeckInput(
            main_deck=opponent_decklist["main_deck"],
            sideboard=opponent_decklist.get("sideboard", []),
            name=f"{opponent_archetype} (Tournament)",
        )

        return await self.simulate_match(
            your_deck=your_deck,
            opponent_deck=opponent_deck,
            num_games=num_games,
            format=deck_format,
        )

    async def _resolve_deck(self, deck_input: DeckInput) -> Optional[Dict[str, Any]]:
        """Resolve a DeckInput to full deck data."""
        if deck_input.deck_id:
            result = await self.db.execute(
                select(Deck).where(Deck.id == deck_input.deck_id)
            )
            deck = result.scalar_one_or_none()
            if deck:
                return {
                    "name": deck.name,
                    "main_deck": deck.main_deck or [],
                    "sideboard": deck.sideboard or [],
                }
            return None
        else:
            return {
                "name": deck_input.name or "Unknown Deck",
                "main_deck": deck_input.main_deck or [],
                "sideboard": deck_input.sideboard or [],
            }

    async def _enrich_deck(self, deck_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add oracle text and card details to deck."""
        enriched = deck_data.copy()
        enriched["cards"] = {}

        # Collect all card names
        card_names = set()
        for entry in deck_data.get("main_deck", []):
            card_names.add(entry.get("card_name", ""))
        for entry in deck_data.get("sideboard", []):
            card_names.add(entry.get("card_name", ""))

        # Fetch card details
        for name in card_names:
            if not name:
                continue
            result = await self.db.execute(
                select(Card).where(Card.name == name).limit(1)
            )
            card = result.scalar_one_or_none()
            if card:
                enriched["cards"][name] = {
                    "name": card.name,
                    "mana_cost": card.mana_cost,
                    "type_line": card.type_line,
                    "oracle_text": card.oracle_text,
                    "power": card.power,
                    "toughness": card.toughness,
                }

        return enriched

    async def _simulate_single_game(
        self,
        your_deck: Dict[str, Any],
        opponent_deck: Dict[str, Any],
        game_number: int,
        use_sideboard: bool = False,
        on_the_play: bool = True,
    ) -> GameResult:
        """Simulate a single game using the LLM."""

        if not settings.ANTHROPIC_API_KEY:
            # Return mock result if no API key
            return self._mock_game_result(game_number)

        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        # Build the game prompt
        prompt = self._build_game_prompt(
            your_deck=your_deck,
            opponent_deck=opponent_deck,
            game_number=game_number,
            use_sideboard=use_sideboard,
            on_the_play=on_the_play,
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=GAME_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse the response
            response_text = response.content[0].text

            # Extract JSON from response
            game_data = self._parse_game_response(response_text)

            return GameResult(
                game_number=game_number,
                winner="you" if game_data.get("winner") == "player1" else "opponent",
                turns=game_data.get("total_turns", 10),
                your_life=game_data.get("final_life", {}).get("player1", 0),
                opponent_life=game_data.get("final_life", {}).get("player2", 0),
                win_condition=game_data.get("win_condition", "damage"),
                key_moments=game_data.get("key_moments", []),
                your_key_cards=game_data.get("mvp_cards", {}).get("player1", []),
                opponent_key_cards=game_data.get("mvp_cards", {}).get("player2", []),
                sideboard_in=game_data.get("sideboard_in") if use_sideboard else None,
                sideboard_out=game_data.get("sideboard_out") if use_sideboard else None,
            )

        except Exception as e:
            logger.error(f"Error simulating game: {e}")
            return self._mock_game_result(game_number)

    async def _simulate_multiplayer_game(
        self,
        your_deck: Dict[str, Any],
        opponent_decks: List[Dict[str, Any]],
        game_number: int,
        num_players: int = 4,
    ) -> GameResult:
        """Simulate a multiplayer Commander game using the LLM."""

        if not settings.ANTHROPIC_API_KEY:
            return self._mock_multiplayer_result(game_number, num_players)

        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        # Build the multiplayer game prompt
        prompt = self._build_multiplayer_prompt(
            your_deck=your_deck,
            opponent_decks=opponent_decks,
            game_number=game_number,
            num_players=num_players,
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=6000,  # Multiplayer games need more tokens
                system=MULTIPLAYER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text
            game_data = self._parse_game_response(response_text)

            # Map winner to our naming convention
            winner = game_data.get("winner", "player1")
            if winner == "player1":
                winner_str = "you"
            else:
                # player2 = opponent_1, player3 = opponent_2, etc.
                player_num = int(winner.replace("player", ""))
                winner_str = f"opponent_{player_num - 1}"

            # Build life totals dict with our naming
            raw_life = game_data.get("final_life", {})
            life_totals = {}
            for key, val in raw_life.items():
                if key == "player1":
                    life_totals["you"] = val
                else:
                    player_num = int(key.replace("player", ""))
                    life_totals[f"opponent_{player_num - 1}"] = val

            # Map elimination order
            raw_elim = game_data.get("elimination_order", [])
            elimination_order = []
            for p in raw_elim:
                if p == "player1":
                    elimination_order.append("you")
                else:
                    player_num = int(p.replace("player", ""))
                    elimination_order.append(f"opponent_{player_num - 1}")

            # Collect MVP cards per player
            raw_mvp = game_data.get("mvp_cards", {})
            your_key = raw_mvp.get("player1", [])
            opponent_key_cards: List[str] = []
            opponent_key_by_player: Dict[str, List[str]] = {}
            for key, cards in raw_mvp.items():
                if key != "player1":
                    opponent_key_cards.extend(cards)
                    player_num = int(key.replace("player", ""))
                    opponent_key_by_player[f"opponent_{player_num - 1}"] = cards

            return GameResult(
                game_number=game_number,
                winner=winner_str,
                turns=game_data.get("total_turns", 12),
                your_life=life_totals.get("you", 0),
                opponent_life=0,  # Not meaningful for multiplayer
                life_totals=life_totals,
                elimination_order=elimination_order if elimination_order else None,
                win_condition=game_data.get("win_condition", "damage"),
                key_moments=game_data.get("key_moments", []),
                your_key_cards=your_key,
                opponent_key_cards=opponent_key_cards,
                opponent_key_cards_by_player=opponent_key_by_player if opponent_key_by_player else None,
            )

        except Exception as e:
            logger.error(f"Error simulating multiplayer game: {e}")
            return self._mock_multiplayer_result(game_number, num_players)

    def _build_multiplayer_prompt(
        self,
        your_deck: Dict[str, Any],
        opponent_decks: List[Dict[str, Any]],
        game_number: int,
        num_players: int,
    ) -> str:
        """Build the prompt for simulating a multiplayer Commander game."""

        your_list = self._format_decklist_with_text(your_deck)

        prompt = f"""Simulate Game {game_number} of a {num_players}-player Commander game.

PLAYER 1 (You) - {your_deck.get('name', 'Your Deck')}
Decklist:
{your_list}

"""
        for i, opp_deck in enumerate(opponent_decks):
            opp_list = self._format_decklist_with_text(opp_deck)
            prompt += f"""PLAYER {i + 2} (Opponent {i + 1}) - {opp_deck.get('name', f'Opponent {i + 1}')}
Decklist:
{opp_list}

"""

        prompt += """
COMMANDER GAME RULES:
- Starting life: 40 per player
- Commander damage tracked separately (21 from one commander = elimination)
- Free-for-all: Attack any opponent, target anyone
- Consider politics and threat assessment

Simulate this Commander game from start to finish:
1. Each player draws 7, mulligans optimally (free mulligan in Commander)
2. Play proceeds clockwise (Player 1 -> Player 2 -> Player 3 -> Player 4)
3. Players make politically-aware decisions
4. Track eliminations as players are knocked out
5. Game ends when one player remains

Return the complete game as JSON with elimination_order showing who died in what order."""

        return prompt

    def _mock_multiplayer_result(self, game_number: int, num_players: int) -> GameResult:
        """Generate a mock multiplayer game result for testing without API."""
        players = ["you"] + [f"opponent_{i}" for i in range(1, num_players)]
        random.shuffle(players)
        winner = players[0]
        elimination_order = players[1:]  # Everyone else got eliminated

        life_totals = {winner: random.randint(5, 30)}
        for p in elimination_order:
            life_totals[p] = 0

        return GameResult(
            game_number=game_number,
            winner=winner,
            turns=random.randint(8, 16),
            your_life=life_totals.get("you", 0),
            opponent_life=0,
            life_totals=life_totals,
            elimination_order=elimination_order,
            win_condition="damage",
            key_moments=["[Mock multiplayer game - no API key configured]"],
            your_key_cards=[],
            opponent_key_cards=[],
        )

    def _build_game_prompt(
        self,
        your_deck: Dict[str, Any],
        opponent_deck: Dict[str, Any],
        game_number: int,
        use_sideboard: bool,
        on_the_play: bool,
    ) -> str:
        """Build the prompt for simulating a game."""

        # Format decklists with card text
        your_list = self._format_decklist_with_text(your_deck)
        opponent_list = self._format_decklist_with_text(opponent_deck)

        play_draw = "on the play" if on_the_play else "on the draw"

        prompt = f"""Simulate Game {game_number} of a Magic: The Gathering match.

PLAYER 1 (You) - {your_deck.get('name', 'Your Deck')} - {play_draw}
Decklist:
{your_list}

PLAYER 2 (Opponent) - {opponent_deck.get('name', 'Opponent Deck')}
Decklist:
{opponent_list}
"""

        if use_sideboard and game_number > 1:
            your_sb = self._format_sideboard(your_deck)
            opp_sb = self._format_sideboard(opponent_deck)
            prompt += f"""
SIDEBOARDING (Game {game_number}):
Player 1 Sideboard: {your_sb}
Player 2 Sideboard: {opp_sb}

First, determine optimal sideboard swaps for both players, then simulate the post-board game.
Include "sideboard_in" and "sideboard_out" arrays for player 1 in your response.
"""

        prompt += """
Simulate this game from start to finish. Both players:
1. Shuffle and draw 7 cards
2. Mulligan if hand is bad (land ratio, curve, matchup-specific keeps)
3. Play optimally given the cards drawn
4. Consider sequencing, holding interaction, bluffing, etc.

Return the complete game as JSON."""

        return prompt

    def _format_decklist_with_text(self, deck: Dict[str, Any]) -> str:
        """Format a decklist with card oracle text for the LLM."""
        lines = []
        cards = deck.get("cards", {})

        for entry in deck.get("main_deck", []):
            name = entry.get("card_name", "Unknown")
            qty = entry.get("quantity", 1)
            card_info = cards.get(name, {})

            if card_info:
                mana = card_info.get("mana_cost") or ""
                type_line = card_info.get("type_line") or ""
                oracle = card_info.get("oracle_text") or ""
                pt = ""
                if card_info.get("power") and card_info.get("toughness"):
                    pt = f" [{card_info['power']}/{card_info['toughness']}]"

                # Truncate oracle text if too long
                if oracle and len(oracle) > 100:
                    oracle = oracle[:100] + "..."

                lines.append(f"{qty}x {name} {mana} - {type_line}{pt}")
                if oracle:
                    lines.append(f"   {oracle}")
            else:
                lines.append(f"{qty}x {name}")

        return "\n".join(lines)

    def _format_sideboard(self, deck: Dict[str, Any]) -> str:
        """Format sideboard as a simple list."""
        entries = []
        for entry in deck.get("sideboard", []):
            name = entry.get("card_name", "Unknown")
            qty = entry.get("quantity", 1)
            entries.append(f"{qty}x {name}")
        return ", ".join(entries) if entries else "No sideboard"

    def _parse_game_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM's game response into structured data."""
        from app.services.ai.json_helpers import repair_json

        # Try to extract JSON from the response
        try:
            # Look for JSON block
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "{" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            else:
                json_str = response_text

            # Try direct parse first
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try repairing common JSON issues
                repaired = repair_json(json_str)
                return json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse game JSON: {e}")
            # Return a default result
            return {
                "winner": "player1" if random.random() > 0.5 else "player2",
                "win_condition": "damage",
                "total_turns": 8,
                "final_life": {"player1": 5, "player2": 0},
                "key_moments": ["Game simulation parsing failed, result randomized"],
                "mvp_cards": {"player1": [], "player2": []},
            }

    def _mock_game_result(self, game_number: int) -> GameResult:
        """Generate a mock game result for testing without API."""
        winner = "you" if random.random() > 0.5 else "opponent"
        return GameResult(
            game_number=game_number,
            winner=winner,
            turns=random.randint(5, 12),
            your_life=random.randint(0, 15) if winner == "you" else 0,
            opponent_life=0 if winner == "you" else random.randint(0, 15),
            win_condition="damage",
            key_moments=["[Mock game - no API key configured]"],
            your_key_cards=[],
            opponent_key_cards=[],
        )

    def _aggregate_key_cards(
        self,
        games: List[GameResult],
        field: str,
    ) -> List[Dict[str, Any]]:
        """Aggregate key cards across multiple games."""
        card_counts: Dict[str, int] = {}

        for game in games:
            cards = getattr(game, field, [])
            for card in cards:
                card_counts[card] = card_counts.get(card, 0) + 1

        # Sort by frequency
        sorted_cards = sorted(card_counts.items(), key=lambda x: -x[1])

        return [
            {
                "card": card,
                "importance": count / len(games),
                "reason": f"Key card in {count}/{len(games)} games",
            }
            for card, count in sorted_cards[:10]
        ]

    async def _generate_matchup_analysis(
        self,
        your_deck: Dict[str, Any],
        opponent_deck: Dict[str, Any],
        games: List[GameResult],
        win_rate: float,
        is_multiplayer: bool = False,
    ) -> Dict[str, Any]:
        """Generate strategic analysis and sideboard guide."""

        if not settings.ANTHROPIC_API_KEY:
            return {
                "sideboard_guide": {"in": [], "out": []},
                "strategic_advice": ["Configure API key for detailed analysis"],
                "mulligan_advice": "Keep balanced hands with lands and spells.",
                "deck_recommendations": [],
            }

        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        # Summarize game results
        game_summaries = []
        for g in games:
            result_str = 'Won' if g.winner == 'you' else 'Lost'
            key_cards = ', '.join(g.your_key_cards[:3]) if g.your_key_cards else 'none'
            opp_cards = ', '.join(g.opponent_key_cards[:3]) if g.opponent_key_cards else 'none'
            game_summaries.append(
                f"Game {g.game_number}: {result_str} in {g.turns} turns. "
                f"Your key cards: {key_cards}. Opponent's key cards: {opp_cards}."
            )

        # Get full decklist for better recommendations
        main_deck_cards = [e.get('card_name', '') for e in your_deck.get('main_deck', [])]
        sideboard_cards = [e.get('card_name', '') for e in your_deck.get('sideboard', [])]

        prompt = f"""Analyze this Magic: The Gathering matchup and provide strategic advice AND deck improvement recommendations.

YOUR DECK: {your_deck.get('name')}
Full main deck: {', '.join(main_deck_cards)}
Sideboard: {', '.join(sideboard_cards) if sideboard_cards else 'None'}

OPPONENT DECK: {opponent_deck.get('name')}
Main cards: {', '.join(e.get('card_name', '') for e in opponent_deck.get('main_deck', [])[:20])}

SIMULATION RESULTS ({len(games)} games, {win_rate:.0%} win rate):
{chr(10).join(game_summaries)}

Based on the simulation results, provide:
1. Sideboard guide (which cards to bring in and take out for this matchup)
2. 3-5 strategic tips for playing this matchup
3. Mulligan advice (what hands to keep/ship)
4. **Deck improvement recommendations** - specific suggestions for cards to add, remove, or adjust quantities to improve this matchup. Consider:
   - Cards that consistently underperformed
   - Types of effects the deck lacks (removal, card draw, threats, etc.)
   - Cards the opponent played that were hard to answer
   - Mana curve or color issues observed

Return as JSON:
{{
    "sideboard_guide": {{"in": ["card1", "card2"], "out": ["card3", "card4"]}},
    "strategic_advice": ["tip1", "tip2", "tip3"],
    "mulligan_advice": "Keep hands that...",
    "deck_recommendations": [
        {{
            "category": "add_cards",
            "priority": "high",
            "suggestion": "Add 2-3 more removal spells to handle early threats",
            "cards_mentioned": ["Fatal Push", "Go for the Throat"],
            "reasoning": "Opponent's early creatures went unanswered in multiple games"
        }},
        {{
            "category": "remove_cards",
            "priority": "medium",
            "suggestion": "Consider cutting slow cards that don't impact this matchup",
            "cards_mentioned": ["Card Name"],
            "reasoning": "This card was too slow against the aggressive opponent"
        }}
    ]
}}

Categories for recommendations: "add_cards", "remove_cards", "adjust_quantities", "sideboard", "strategy"
Priorities: "high", "medium", "low"
Provide 2-4 specific, actionable recommendations."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,  # Increased for recommendations
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Parse JSON response
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "{" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            else:
                json_str = response_text

            return json.loads(json_str)

        except Exception as e:
            logger.error(f"Error generating matchup analysis: {e}")
            return {
                "sideboard_guide": {"in": [], "out": []},
                "strategic_advice": ["Analysis generation failed"],
                "deck_recommendations": [],
                "mulligan_advice": "Keep balanced hands.",
            }

    async def _get_archetype_decklist(
        self, archetype: str, format: str = "standard"
    ) -> Optional[Dict[str, Any]]:
        """Get a representative decklist for a meta archetype in the specified format."""
        # Search for recent tournament decklists with this archetype in the same format
        result = await self.db.execute(
            select(Decklist)
            .join(Event)
            .where(
                Decklist.archetype.ilike(f"%{archetype}%"),
                Event.format == format,
            )
            .order_by(Event.date.desc())
            .limit(1)
        )
        decklist = result.scalar_one_or_none()

        if decklist:
            return {
                "name": f"{decklist.archetype} ({decklist.player_name or 'Tournament'})",
                "main_deck": decklist.main_deck or [],
                "sideboard": decklist.sideboard or [],
            }

        return None

    async def get_meta_archetypes(self, format: str = "standard") -> List[str]:
        """Get list of available meta archetypes to simulate against for a specific format."""
        result = await self.db.execute(
            select(Decklist.archetype)
            .join(Event)
            .distinct()
            .where(
                Decklist.archetype.isnot(None),
                Event.format == format,
            )
            .order_by(Decklist.archetype)
        )
        return [row[0] for row in result.all() if row[0]]

    # =========================================================================
    # Persistent Simulation Run Methods
    # =========================================================================

    async def create_simulation_run(
        self,
        user_id: UUID,
        deck_id: UUID,
        opponent_archetype: Optional[str] = None,
        num_games: int = 5,
        include_sideboard_games: bool = True,
        opponent_archetypes: Optional[List[str]] = None,
        num_players: int = 2,
    ) -> SimulationRun:
        """
        Create a new simulation run record.
        Returns the created run so it can be executed in the background.

        For multiplayer (num_players > 2), use opponent_archetypes list.
        For 2-player, use opponent_archetype.
        """
        # Get user's deck
        result = await self.db.execute(
            select(Deck).where(Deck.id == deck_id)
        )
        deck = result.scalar_one_or_none()
        if not deck:
            raise ValueError(f"Deck {deck_id} not found")

        deck_format = deck.format or "standard"
        is_multiplayer = num_players > 2

        # Create snapshot of your deck
        your_deck_snapshot = {
            "main_deck": deck.main_deck or [],
            "sideboard": deck.sideboard or [],
        }

        # Handle opponents based on player count
        if is_multiplayer:
            if not opponent_archetypes or len(opponent_archetypes) < num_players - 1:
                raise ValueError(f"Need {num_players - 1} opponent archetypes for {num_players}-player game")

            opponent_deck_names_list = []
            opponent_archetypes_list = []
            opponent_snapshots_list = []

            for arch in opponent_archetypes[:num_players - 1]:
                decklist = await self._get_archetype_decklist(arch, deck_format)
                if not decklist:
                    raise ValueError(f"No {deck_format} decklists found for archetype: {arch}")
                opponent_deck_names_list.append(decklist.get("name", arch))
                opponent_archetypes_list.append(arch)
                opponent_snapshots_list.append({
                    "main_deck": decklist.get("main_deck", []),
                    "sideboard": decklist.get("sideboard", []),
                })

            # Create the simulation run record for multiplayer
            sim_run = SimulationRun(
                user_id=user_id,
                status="pending",
                your_deck_id=deck_id,
                your_deck_name=deck.name,
                your_deck_snapshot=your_deck_snapshot,
                # For backward compatibility, set first opponent as primary
                opponent_deck_name=" vs ".join(opponent_deck_names_list),
                opponent_archetype=opponent_archetypes_list[0],
                opponent_deck_snapshot=opponent_snapshots_list[0],
                # Multiplayer fields
                num_players=num_players,
                opponent_deck_names=opponent_deck_names_list,
                opponent_archetypes=opponent_archetypes_list,
                opponent_deck_snapshots=opponent_snapshots_list,
                format=deck_format,
                num_games=num_games,
                include_sideboard_games=0,  # No sideboarding in multiplayer
                games_completed=0,
            )
        else:
            # 2-player game
            if not opponent_archetype:
                raise ValueError("opponent_archetype required for 2-player game")

            opponent_decklist = await self._get_archetype_decklist(opponent_archetype, deck_format)
            if not opponent_decklist:
                raise ValueError(f"No {deck_format} decklists found for archetype: {opponent_archetype}")

            opponent_deck_snapshot = {
                "main_deck": opponent_decklist.get("main_deck", []),
                "sideboard": opponent_decklist.get("sideboard", []),
            }

            sim_run = SimulationRun(
                user_id=user_id,
                status="pending",
                your_deck_id=deck_id,
                your_deck_name=deck.name,
                your_deck_snapshot=your_deck_snapshot,
                opponent_deck_name=opponent_decklist.get("name", opponent_archetype),
                opponent_archetype=opponent_archetype,
                opponent_deck_snapshot=opponent_deck_snapshot,
                num_players=2,
                format=deck_format,
                num_games=num_games,
                include_sideboard_games=1 if include_sideboard_games else 0,
                games_completed=0,
            )

        self.db.add(sim_run)
        await self.db.commit()
        await self.db.refresh(sim_run)

        return sim_run

    async def execute_simulation_run(self, simulation_id: UUID) -> SimulationRun:
        """
        Execute a simulation run and update the record with results.
        This is designed to be called from a background job.
        """
        # Get the simulation run
        result = await self.db.execute(
            select(SimulationRun).where(SimulationRun.id == simulation_id)
        )
        sim_run = result.scalar_one_or_none()
        if not sim_run:
            raise ValueError(f"Simulation run {simulation_id} not found")

        # Mark as running
        sim_run.status = "running"
        sim_run.started_at = datetime.utcnow()
        await self.db.commit()

        try:
            # Build deck inputs from snapshots
            your_deck = DeckInput(
                main_deck=sim_run.your_deck_snapshot.get("main_deck", []),
                sideboard=sim_run.your_deck_snapshot.get("sideboard", []),
                name=sim_run.your_deck_name,
            )

            is_multiplayer = sim_run.num_players > 2
            opponent_decks = None

            if is_multiplayer and sim_run.opponent_deck_snapshots:
                # Build opponent deck list for multiplayer
                opponent_decks = []
                for i, snapshot in enumerate(sim_run.opponent_deck_snapshots):
                    name = sim_run.opponent_deck_names[i] if sim_run.opponent_deck_names else f"Opponent {i + 1}"
                    opponent_decks.append(DeckInput(
                        main_deck=snapshot.get("main_deck", []),
                        sideboard=snapshot.get("sideboard", []),
                        name=name,
                    ))
                # Use first opponent as the "primary" for backward compat
                opponent_deck = opponent_decks[0]
            else:
                opponent_deck = DeckInput(
                    main_deck=sim_run.opponent_deck_snapshot.get("main_deck", []),
                    sideboard=sim_run.opponent_deck_snapshot.get("sideboard", []),
                    name=sim_run.opponent_deck_name,
                )

            # Run the simulation
            match_result = await self.simulate_match(
                your_deck=your_deck,
                opponent_deck=opponent_deck,
                num_games=sim_run.num_games,
                include_sideboard_games=bool(sim_run.include_sideboard_games),
                format=sim_run.format,
                on_progress=lambda completed, turn: self._update_progress(
                    sim_run, completed, turn
                ),
                opponent_decks=opponent_decks,
                num_players=sim_run.num_players,
            )

            # Update with results
            sim_run.status = "completed"
            sim_run.completed_at = datetime.utcnow()
            sim_run.your_wins = match_result.your_wins
            sim_run.opponent_wins = match_result.opponent_wins
            # Multiplayer-specific results
            if is_multiplayer:
                sim_run.first_place_count = match_result.first_place_count
                sim_run.your_placement_avg = match_result.your_placement_avg
            sim_run.win_rate = match_result.win_rate
            sim_run.average_game_length = match_result.average_game_length
            sim_run.matchup_assessment = match_result.matchup_assessment
            sim_run.games = [g.model_dump() for g in match_result.games]
            sim_run.key_cards_for_you = match_result.key_cards_for_you
            sim_run.key_cards_against_you = match_result.key_cards_against_you
            sim_run.sideboard_guide = match_result.sideboard_guide
            sim_run.strategic_advice = match_result.strategic_advice
            sim_run.mulligan_advice = match_result.mulligan_advice
            # Store deck recommendations (convert to dicts if they're Pydantic models)
            if match_result.deck_recommendations:
                sim_run.deck_recommendations = [
                    r.model_dump() if hasattr(r, 'model_dump') else r
                    for r in match_result.deck_recommendations
                ]
            sim_run.games_completed = sim_run.num_games

            await self.db.commit()
            await self.db.refresh(sim_run)

        except Exception as e:
            logger.error(f"Simulation run {simulation_id} failed: {e}")
            sim_run.status = "failed"
            sim_run.error_message = str(e)[:1000]
            sim_run.completed_at = datetime.utcnow()
            await self.db.commit()
            raise

        return sim_run

    async def _update_progress(
        self, sim_run: SimulationRun, games_completed: int, current_turn: Optional[int]
    ):
        """Update simulation progress (called during simulation)."""
        sim_run.games_completed = games_completed
        sim_run.current_game_turn = current_turn
        await self.db.commit()

    async def get_simulation_run(self, simulation_id: UUID, user_id: UUID) -> Optional[SimulationRun]:
        """Get a simulation run by ID (user must own it)."""
        result = await self.db.execute(
            select(SimulationRun).where(
                SimulationRun.id == simulation_id,
                SimulationRun.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_simulation_runs(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> Tuple[List[SimulationRun], int]:
        """List simulation runs for a user."""
        query = select(SimulationRun).where(SimulationRun.user_id == user_id)

        if status:
            query = query.where(SimulationRun.status == status)

        # Count total
        from sqlalchemy import func
        count_query = select(func.count()).select_from(
            query.subquery()
        )
        count_result = await self.db.execute(count_query)
        total = count_result.scalar() or 0

        # Get paginated results
        query = query.order_by(desc(SimulationRun.created_at)).offset(offset).limit(limit)
        result = await self.db.execute(query)
        runs = result.scalars().all()

        return list(runs), total

    async def delete_simulation_run(self, simulation_id: UUID, user_id: UUID) -> bool:
        """Delete a simulation run (user must own it)."""
        result = await self.db.execute(
            select(SimulationRun).where(
                SimulationRun.id == simulation_id,
                SimulationRun.user_id == user_id,
            )
        )
        sim_run = result.scalar_one_or_none()
        if not sim_run:
            return False

        await self.db.delete(sim_run)
        await self.db.commit()
        return True

    async def cleanup_stale_runs(self, timeout_minutes: int = 30) -> int:
        """
        Mark stale 'running' simulations as failed.
        Called on app startup to handle orphaned runs from crashes/restarts.

        Args:
            timeout_minutes: How long a run can be 'running' before considered stale

        Returns:
            Number of runs marked as failed
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(minutes=timeout_minutes)

        # Find running simulations that started more than timeout_minutes ago
        result = await self.db.execute(
            select(SimulationRun).where(
                SimulationRun.status == "running",
                SimulationRun.started_at < cutoff,
            )
        )
        stale_runs = result.scalars().all()

        count = 0
        for run in stale_runs:
            run.status = "failed"
            run.error_message = f"Simulation timed out or server restarted (was running for >{timeout_minutes} minutes)"
            run.completed_at = datetime.utcnow()
            count += 1
            logger.warning(f"Marked stale simulation {run.id} as failed")

        if count > 0:
            await self.db.commit()

        return count

    async def retry_simulation_run(
        self, simulation_id: UUID, user_id: UUID
    ) -> Optional[SimulationRun]:
        """
        Retry a failed simulation run by resetting its status to pending.

        Args:
            simulation_id: ID of the simulation to retry
            user_id: User ID (must own the simulation)

        Returns:
            Updated SimulationRun or None if not found/not retryable
        """
        result = await self.db.execute(
            select(SimulationRun).where(
                SimulationRun.id == simulation_id,
                SimulationRun.user_id == user_id,
            )
        )
        sim_run = result.scalar_one_or_none()

        if not sim_run:
            return None

        # Only allow retrying failed runs
        if sim_run.status != "failed":
            return None

        # Reset for retry
        sim_run.status = "pending"
        sim_run.error_message = None
        sim_run.started_at = None
        sim_run.completed_at = None
        sim_run.games_completed = 0
        sim_run.current_game_turn = None
        # Clear previous partial results
        sim_run.games = None
        sim_run.your_wins = None
        sim_run.opponent_wins = None
        sim_run.win_rate = None
        sim_run.average_game_length = None
        sim_run.matchup_assessment = None
        sim_run.key_cards_for_you = None
        sim_run.key_cards_against_you = None
        sim_run.sideboard_guide = None
        sim_run.strategic_advice = None
        sim_run.mulligan_advice = None
        sim_run.deck_recommendations = None

        await self.db.commit()
        await self.db.refresh(sim_run)

        return sim_run
