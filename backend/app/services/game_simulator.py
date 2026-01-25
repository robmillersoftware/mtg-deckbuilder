"""
Game Simulator Service

Uses LLM to simulate Magic: The Gathering games between two decks.
The LLM plays both sides, tracking game state and making decisions.
"""

from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID, uuid4
import logging
import json
import random

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.models.deck import Deck
from app.models.meta import Decklist, Event
from app.models.card import Card
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
    ) -> MatchupAnalysisResult:
        """
        Simulate a match (multiple games) between two decks.

        Args:
            your_deck: Your deck configuration
            opponent_deck: Opponent deck configuration
            num_games: Number of games to simulate
            include_sideboard_games: Whether games 2+ use sideboards

        Returns:
            MatchupAnalysisResult with statistics and insights
        """
        # Resolve deck data
        your_deck_data = await self._resolve_deck(your_deck)
        opponent_deck_data = await self._resolve_deck(opponent_deck)

        if not your_deck_data or not opponent_deck_data:
            raise ValueError("Could not resolve deck data")

        # Enrich decks with card oracle text
        your_deck_enriched = await self._enrich_deck(your_deck_data)
        opponent_deck_enriched = await self._enrich_deck(opponent_deck_data)

        games: List[GameResult] = []
        your_wins = 0
        total_turns = 0

        for game_num in range(1, num_games + 1):
            # Games 2+ can use sideboards
            use_sideboard = include_sideboard_games and game_num > 1

            game_result = await self._simulate_single_game(
                your_deck=your_deck_enriched,
                opponent_deck=opponent_deck_enriched,
                game_number=game_num,
                use_sideboard=use_sideboard,
                on_the_play=(game_num % 2 == 1),  # Alternate who goes first
            )

            games.append(game_result)
            if game_result.winner == "you":
                your_wins += 1
            total_turns += game_result.turns

        # Calculate statistics
        win_rate = your_wins / num_games
        avg_game_length = total_turns / num_games

        # Determine matchup assessment
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
            opponent_deck=opponent_deck_enriched,
            games=games,
            win_rate=win_rate,
        )

        return MatchupAnalysisResult(
            your_deck_name=your_deck_data.get("name", "Your Deck"),
            opponent_deck_name=opponent_deck_data.get("name", "Opponent Deck"),
            games_played=num_games,
            your_wins=your_wins,
            opponent_wins=num_games - your_wins,
            win_rate=win_rate,
            average_game_length=avg_game_length,
            matchup_assessment=assessment,
            key_cards_for_you=your_key_cards,
            key_cards_against_you=opponent_key_cards,
            sideboard_guide=analysis.get("sideboard_guide", {"in": [], "out": []}),
            strategic_advice=analysis.get("strategic_advice", []),
            mulligan_advice=analysis.get("mulligan_advice", "Keep hands with a good curve and interaction."),
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
                mana = card_info.get("mana_cost", "")
                type_line = card_info.get("type_line", "")
                oracle = card_info.get("oracle_text", "")
                pt = ""
                if card_info.get("power") and card_info.get("toughness"):
                    pt = f" [{card_info['power']}/{card_info['toughness']}]"

                # Truncate oracle text if too long
                if len(oracle) > 100:
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

            return json.loads(json_str)
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
    ) -> Dict[str, Any]:
        """Generate strategic analysis and sideboard guide."""

        if not settings.ANTHROPIC_API_KEY:
            return {
                "sideboard_guide": {"in": [], "out": []},
                "strategic_advice": ["Configure API key for detailed analysis"],
                "mulligan_advice": "Keep balanced hands with lands and spells.",
            }

        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        # Summarize game results
        game_summaries = []
        for g in games:
            game_summaries.append(
                f"Game {g.game_number}: {'Won' if g.winner == 'you' else 'Lost'} "
                f"in {g.turns} turns. Key cards: {', '.join(g.your_key_cards[:3])}"
            )

        prompt = f"""Analyze this Magic: The Gathering matchup and provide strategic advice.

YOUR DECK: {your_deck.get('name')}
Main cards: {', '.join(e.get('card_name', '') for e in your_deck.get('main_deck', [])[:15])}
Sideboard: {', '.join(e.get('card_name', '') for e in your_deck.get('sideboard', [])[:10])}

OPPONENT DECK: {opponent_deck.get('name')}
Main cards: {', '.join(e.get('card_name', '') for e in opponent_deck.get('main_deck', [])[:15])}

SIMULATION RESULTS ({len(games)} games, {win_rate:.0%} win rate):
{chr(10).join(game_summaries)}

Provide:
1. Sideboard guide (which cards to bring in and take out)
2. 3-5 strategic tips for this matchup
3. Mulligan advice (what hands to keep/ship)

Return as JSON:
{{
    "sideboard_guide": {{"in": ["card1", "card2"], "out": ["card3", "card4"]}},
    "strategic_advice": ["tip1", "tip2", "tip3"],
    "mulligan_advice": "Keep hands that..."
}}"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
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
