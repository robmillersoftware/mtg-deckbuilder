import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { decksApi, simulationApi } from '@/services/api';
import { Deck, MatchupAnalysisResult, GameResult } from '@/types';
import toast from 'react-hot-toast';

export function SimulationPage() {
  const [searchParams] = useSearchParams();
  const initialDeckId = searchParams.get('deck');

  const [decks, setDecks] = useState<Deck[]>([]);
  const [selectedDeckId, setSelectedDeckId] = useState<string>(initialDeckId || '');
  const [archetypes, setArchetypes] = useState<string[]>([]);
  const [selectedArchetype, setSelectedArchetype] = useState<string>('');
  const [numGames, setNumGames] = useState<number>(5);
  const [includeSideboard, setIncludeSideboard] = useState<boolean>(true);

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<MatchupAnalysisResult | null>(null);
  const [expandedGame, setExpandedGame] = useState<number | null>(null);

  useEffect(() => {
    loadDecks();
    loadArchetypes();
  }, []);

  const loadDecks = async () => {
    try {
      const response = await decksApi.list(100);
      setDecks(response.data.items || response.data);
    } catch (error) {
      console.error('Failed to load decks:', error);
    }
  };

  const loadArchetypes = async () => {
    try {
      const response = await simulationApi.getAvailableArchetypes();
      setArchetypes(response.data);
      if (response.data.length > 0 && !selectedArchetype) {
        setSelectedArchetype(response.data[0]);
      }
    } catch (error) {
      console.error('Failed to load archetypes:', error);
    }
  };

  const runSimulation = async () => {
    if (!selectedDeckId) {
      toast.error('Please select a deck');
      return;
    }
    if (!selectedArchetype) {
      toast.error('Please select an opponent archetype');
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const response = await simulationApi.simulateVsArchetype({
        deck_id: selectedDeckId,
        opponent_archetype: selectedArchetype,
        num_games: numGames,
      });
      setResult(response.data);
      toast.success('Simulation complete!');
    } catch (error: any) {
      console.error('Simulation failed:', error);
      toast.error(error.response?.data?.detail || 'Simulation failed');
    } finally {
      setIsLoading(false);
    }
  };

  const getAssessmentColor = (assessment: string) => {
    switch (assessment) {
      case 'favored': return 'text-green-400';
      case 'unfavored': return 'text-red-400';
      default: return 'text-yellow-400';
    }
  };

  const getWinRateColor = (winRate: number) => {
    if (winRate >= 0.6) return 'text-green-400';
    if (winRate <= 0.4) return 'text-red-400';
    return 'text-yellow-400';
  };

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-6">Game Simulator</h1>
      <p className="text-gray-400 mb-8">
        Simulate games between your deck and meta archetypes to find optimal configurations and learn matchups.
      </p>

      {/* Configuration */}
      <div className="bg-gray-800 rounded-lg p-6 mb-8">
        <h2 className="text-lg font-semibold text-white mb-4">Configuration</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Deck Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Your Deck
            </label>
            <select
              value={selectedDeckId}
              onChange={(e) => setSelectedDeckId(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">Select a deck...</option>
              {decks.map((deck) => (
                <option key={deck.id} value={deck.id}>
                  {deck.name}
                </option>
              ))}
            </select>
          </div>

          {/* Archetype Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Opponent Archetype
            </label>
            <select
              value={selectedArchetype}
              onChange={(e) => setSelectedArchetype(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">Select archetype...</option>
              {archetypes.map((arch) => (
                <option key={arch} value={arch}>
                  {arch}
                </option>
              ))}
            </select>
          </div>

          {/* Number of Games */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Number of Games
            </label>
            <select
              value={numGames}
              onChange={(e) => setNumGames(parseInt(e.target.value))}
              className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {[3, 5, 7, 10].map((n) => (
                <option key={n} value={n}>
                  {n} games
                </option>
              ))}
            </select>
          </div>

          {/* Sideboard Toggle */}
          <div className="flex items-end">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={includeSideboard}
                onChange={(e) => setIncludeSideboard(e.target.checked)}
                className="w-4 h-4 text-indigo-600 bg-gray-700 border-gray-600 rounded focus:ring-indigo-500"
              />
              <span className="ml-2 text-sm text-gray-300">Include sideboard games</span>
            </label>
          </div>
        </div>

        <button
          onClick={runSimulation}
          disabled={isLoading || !selectedDeckId || !selectedArchetype}
          className="mt-6 px-6 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Simulating...
            </>
          ) : (
            'Run Simulation'
          )}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Results Summary</h2>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-sm text-gray-400">Win Rate</div>
                <div className={`text-2xl font-bold ${getWinRateColor(result.win_rate)}`}>
                  {(result.win_rate * 100).toFixed(0)}%
                </div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-sm text-gray-400">Record</div>
                <div className="text-2xl font-bold text-white">
                  {result.your_wins}-{result.opponent_wins}
                </div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-sm text-gray-400">Assessment</div>
                <div className={`text-2xl font-bold capitalize ${getAssessmentColor(result.matchup_assessment)}`}>
                  {result.matchup_assessment}
                </div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-sm text-gray-400">Avg Game Length</div>
                <div className="text-2xl font-bold text-white">
                  {result.average_game_length.toFixed(1)} turns
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Your Key Cards */}
              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-2">Your Key Cards</h3>
                <div className="space-y-2">
                  {result.key_cards_for_you.slice(0, 5).map((card, i) => (
                    <div key={i} className="flex justify-between items-center bg-gray-700 rounded px-3 py-2">
                      <span className="text-white">{card.card}</span>
                      <span className="text-gray-400 text-sm">
                        {(card.importance * 100).toFixed(0)}% of games
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Opponent's Key Cards */}
              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-2">Opponent's Key Cards</h3>
                <div className="space-y-2">
                  {result.key_cards_against_you.slice(0, 5).map((card, i) => (
                    <div key={i} className="flex justify-between items-center bg-gray-700 rounded px-3 py-2">
                      <span className="text-white">{card.card}</span>
                      <span className="text-gray-400 text-sm">
                        {(card.importance * 100).toFixed(0)}% of games
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Sideboard Guide */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Sideboard Guide</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-sm font-medium text-green-400 mb-2">Bring In</h3>
                <div className="bg-gray-700 rounded p-3">
                  {result.sideboard_guide.in.length > 0 ? (
                    <ul className="space-y-1">
                      {result.sideboard_guide.in.map((card, i) => (
                        <li key={i} className="text-white">+ {card}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-gray-400">No sideboard changes recommended</p>
                  )}
                </div>
              </div>

              <div>
                <h3 className="text-sm font-medium text-red-400 mb-2">Take Out</h3>
                <div className="bg-gray-700 rounded p-3">
                  {result.sideboard_guide.out.length > 0 ? (
                    <ul className="space-y-1">
                      {result.sideboard_guide.out.map((card, i) => (
                        <li key={i} className="text-white">- {card}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-gray-400">No sideboard changes recommended</p>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Strategic Advice */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Strategic Advice</h2>

            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-400 mb-2">Mulligan Strategy</h3>
              <p className="text-white bg-gray-700 rounded p-3">{result.mulligan_advice}</p>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-2">Key Tips</h3>
              <ul className="space-y-2">
                {result.strategic_advice.map((tip, i) => (
                  <li key={i} className="flex items-start">
                    <span className="text-indigo-400 mr-2">*</span>
                    <span className="text-white">{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Individual Games */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Game Details</h2>

            <div className="space-y-3">
              {result.games.map((game) => (
                <div key={game.game_number} className="bg-gray-700 rounded-lg overflow-hidden">
                  <button
                    onClick={() => setExpandedGame(expandedGame === game.game_number ? null : game.game_number)}
                    className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-600 transition-colors"
                  >
                    <div className="flex items-center space-x-4">
                      <span className="text-gray-400">Game {game.game_number}</span>
                      <span className={game.winner === 'you' ? 'text-green-400 font-medium' : 'text-red-400 font-medium'}>
                        {game.winner === 'you' ? 'WIN' : 'LOSS'}
                      </span>
                      <span className="text-gray-400">
                        {game.turns} turns - {game.win_condition}
                      </span>
                    </div>
                    <svg
                      className={`w-5 h-5 text-gray-400 transform transition-transform ${expandedGame === game.game_number ? 'rotate-180' : ''}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {expandedGame === game.game_number && (
                    <div className="px-4 py-3 border-t border-gray-600">
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                          <span className="text-gray-400 text-sm">Your Life:</span>
                          <span className="text-white ml-2">{game.your_life}</span>
                        </div>
                        <div>
                          <span className="text-gray-400 text-sm">Opponent Life:</span>
                          <span className="text-white ml-2">{game.opponent_life}</span>
                        </div>
                      </div>

                      {game.key_moments.length > 0 && (
                        <div className="mb-4">
                          <h4 className="text-sm font-medium text-gray-400 mb-2">Key Moments</h4>
                          <ul className="space-y-1">
                            {game.key_moments.map((moment, i) => (
                              <li key={i} className="text-white text-sm">{moment}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {game.sideboard_in && game.sideboard_in.length > 0 && (
                        <div className="text-sm">
                          <span className="text-green-400">In: </span>
                          <span className="text-white">{game.sideboard_in.join(', ')}</span>
                          <span className="text-red-400 ml-4">Out: </span>
                          <span className="text-white">{game.sideboard_out?.join(', ')}</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
