import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { decksApi, simulationApi } from '@/services/api';
import { Deck, SimulationRun, GameResult } from '@/types';
import toast from 'react-hot-toast';

export function SimulationPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialDeckId = searchParams.get('deck');
  const viewRunId = searchParams.get('run');

  // Deck and configuration state
  const [decks, setDecks] = useState<Deck[]>([]);
  const [selectedDeckId, setSelectedDeckId] = useState<string>(initialDeckId || '');
  const [selectedFormat, setSelectedFormat] = useState<string>('standard');
  const [archetypes, setArchetypes] = useState<string[]>([]);
  const [selectedArchetype, setSelectedArchetype] = useState<string>('');
  const [numGames, setNumGames] = useState<number>(5);

  // Simulation runs state
  const [runs, setRuns] = useState<SimulationRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<SimulationRun | null>(null);
  const [isLoadingRuns, setIsLoadingRuns] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [isLoadingArchetypes, setIsLoadingArchetypes] = useState(false);

  // UI state
  const [expandedGame, setExpandedGame] = useState<number | null>(null);

  // Load decks and runs on mount
  useEffect(() => {
    loadDecks();
    loadRuns();
  }, []);

  // Load archetypes when format changes
  useEffect(() => {
    loadArchetypes(selectedFormat);
  }, [selectedFormat]);

  // Update format when deck selection changes
  useEffect(() => {
    if (selectedDeckId) {
      const deck = decks.find((d) => d.id === selectedDeckId);
      if (deck?.format) {
        setSelectedFormat(deck.format);
      }
    }
  }, [selectedDeckId, decks]);

  // Load specific run from URL
  useEffect(() => {
    if (viewRunId) {
      loadRun(viewRunId);
    }
  }, [viewRunId]);

  // Poll for running simulations
  useEffect(() => {
    const runningRuns = runs.filter((r) => r.status === 'running' || r.status === 'pending');
    if (runningRuns.length === 0) return;

    const interval = setInterval(() => {
      loadRuns();
      // Also refresh the selected run if it's still running
      if (selectedRun && (selectedRun.status === 'running' || selectedRun.status === 'pending')) {
        loadRun(selectedRun.id);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [runs, selectedRun]);

  const loadDecks = async () => {
    try {
      const response = await decksApi.list(100);
      const loadedDecks = response.data.items || response.data;
      setDecks(loadedDecks);

      if (initialDeckId) {
        const deck = loadedDecks.find((d: Deck) => d.id === initialDeckId);
        if (deck?.format) {
          setSelectedFormat(deck.format);
        }
      }
    } catch (error) {
      console.error('Failed to load decks:', error);
    }
  };

  const loadRuns = async () => {
    try {
      const response = await simulationApi.listRuns({ limit: 50 });
      setRuns(response.data.items || []);
    } catch (error) {
      console.error('Failed to load simulation runs:', error);
    } finally {
      setIsLoadingRuns(false);
    }
  };

  const loadRun = async (runId: string) => {
    try {
      const response = await simulationApi.getRun(runId);
      setSelectedRun(response.data);
    } catch (error) {
      console.error('Failed to load simulation run:', error);
    }
  };

  const loadArchetypes = async (format: string) => {
    setIsLoadingArchetypes(true);
    try {
      const response = await simulationApi.getAvailableArchetypes(format);
      setArchetypes(response.data);
      if (response.data.length > 0 && !response.data.includes(selectedArchetype)) {
        setSelectedArchetype(response.data[0]);
      } else if (response.data.length === 0) {
        setSelectedArchetype('');
      }
    } catch (error) {
      console.error('Failed to load archetypes:', error);
      setArchetypes([]);
    } finally {
      setIsLoadingArchetypes(false);
    }
  };

  const startSimulation = async () => {
    if (!selectedDeckId) {
      toast.error('Please select a deck');
      return;
    }
    if (!selectedArchetype) {
      toast.error('Please select an opponent archetype');
      return;
    }

    setIsCreating(true);

    try {
      const response = await simulationApi.createRun({
        deck_id: selectedDeckId,
        opponent_archetype: selectedArchetype,
        num_games: numGames,
      });
      toast.success('Simulation started!');
      setSelectedRun(response.data);
      setSearchParams({ run: response.data.id });
      loadRuns();
    } catch (error: any) {
      console.error('Failed to start simulation:', error);
      toast.error(error.response?.data?.detail || 'Failed to start simulation');
    } finally {
      setIsCreating(false);
    }
  };

  const deleteRun = async (runId: string) => {
    try {
      await simulationApi.deleteRun(runId);
      toast.success('Simulation deleted');
      if (selectedRun?.id === runId) {
        setSelectedRun(null);
        setSearchParams({});
      }
      loadRuns();
    } catch (error) {
      toast.error('Failed to delete simulation');
    }
  };

  const selectRun = (run: SimulationRun) => {
    setSelectedRun(run);
    setSearchParams({ run: run.id });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400';
      case 'running': return 'text-yellow-400';
      case 'pending': return 'text-blue-400';
      case 'failed': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getAssessmentColor = (assessment?: string) => {
    switch (assessment) {
      case 'favored': return 'text-green-400';
      case 'unfavored': return 'text-red-400';
      default: return 'text-yellow-400';
    }
  };

  const getWinRateColor = (winRate?: number) => {
    if (!winRate) return 'text-gray-400';
    if (winRate >= 0.6) return 'text-green-400';
    if (winRate <= 0.4) return 'text-red-400';
    return 'text-yellow-400';
  };

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return '';
    return new Date(dateStr).toLocaleString();
  };

  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-2">Game Simulator</h1>
      <p className="text-gray-400 mb-6">
        Simulate games between your deck and meta archetypes. Simulations run in the background - you can navigate away and come back.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Configuration + Run List */}
        <div className="space-y-6">
          {/* New Simulation */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-white mb-4">New Simulation</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Your Deck</label>
                <select
                  value={selectedDeckId}
                  onChange={(e) => setSelectedDeckId(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="">Select a deck...</option>
                  {decks.map((deck) => (
                    <option key={deck.id} value={deck.id}>
                      {deck.name} ({deck.format || 'standard'})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Opponent ({selectedFormat})
                </label>
                <select
                  value={selectedArchetype}
                  onChange={(e) => setSelectedArchetype(e.target.value)}
                  disabled={isLoadingArchetypes}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
                >
                  {isLoadingArchetypes ? (
                    <option value="">Loading...</option>
                  ) : archetypes.length === 0 ? (
                    <option value="">No archetypes available</option>
                  ) : (
                    <>
                      <option value="">Select archetype...</option>
                      {archetypes.map((arch) => (
                        <option key={arch} value={arch}>{arch}</option>
                      ))}
                    </>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Games</label>
                <select
                  value={numGames}
                  onChange={(e) => setNumGames(parseInt(e.target.value))}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {[3, 5, 7, 10].map((n) => (
                    <option key={n} value={n}>{n} games</option>
                  ))}
                </select>
              </div>

              <button
                onClick={startSimulation}
                disabled={isCreating || !selectedDeckId || !selectedArchetype}
                className="w-full px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              >
                {isCreating ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Starting...
                  </>
                ) : (
                  'Start Simulation'
                )}
              </button>
            </div>
          </div>

          {/* Simulation History */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-white mb-4">History</h2>

            {isLoadingRuns ? (
              <p className="text-gray-400 text-sm">Loading...</p>
            ) : runs.length === 0 ? (
              <p className="text-gray-400 text-sm">No simulations yet</p>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {runs.map((run) => (
                  <div
                    key={run.id}
                    onClick={() => selectRun(run)}
                    className={`p-3 rounded-lg cursor-pointer transition-colors ${
                      selectedRun?.id === run.id ? 'bg-indigo-900/50 border border-indigo-500' : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-white text-sm font-medium truncate">{run.your_deck_name}</span>
                      <span className={`text-xs font-medium ${getStatusColor(run.status)}`}>
                        {run.status === 'running' && `${run.games_completed}/${run.num_games}`}
                        {run.status === 'completed' && run.win_rate !== undefined && `${(run.win_rate * 100).toFixed(0)}%`}
                        {run.status === 'pending' && 'Pending'}
                        {run.status === 'failed' && 'Failed'}
                      </span>
                    </div>
                    <div className="text-gray-400 text-xs">
                      vs {run.opponent_archetype || run.opponent_deck_name}
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-gray-500 text-xs">{formatDate(run.created_at)}</span>
                      <button
                        onClick={(e) => { e.stopPropagation(); deleteRun(run.id); }}
                        className="text-gray-500 hover:text-red-400 text-xs"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="lg:col-span-2">
          {!selectedRun ? (
            <div className="bg-gray-800 rounded-lg p-8 text-center">
              <p className="text-gray-400">Select a simulation from the history or start a new one</p>
            </div>
          ) : selectedRun.status === 'pending' || selectedRun.status === 'running' ? (
            <div className="bg-gray-800 rounded-lg p-8">
              <div className="text-center">
                <svg className="animate-spin mx-auto h-8 w-8 text-indigo-500 mb-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <h3 className="text-white font-semibold mb-2">
                  {selectedRun.status === 'pending' ? 'Starting simulation...' : 'Simulating games...'}
                </h3>
                <p className="text-gray-400">
                  {selectedRun.your_deck_name} vs {selectedRun.opponent_archetype}
                </p>
                <p className="text-gray-500 mt-2">
                  {selectedRun.games_completed} / {selectedRun.num_games} games completed
                </p>
                <div className="w-full bg-gray-700 rounded-full h-2 mt-4">
                  <div
                    className="bg-indigo-500 h-2 rounded-full transition-all"
                    style={{ width: `${(selectedRun.games_completed / selectedRun.num_games) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ) : selectedRun.status === 'failed' ? (
            <div className="bg-gray-800 rounded-lg p-8">
              <div className="text-center">
                <div className="text-red-400 text-4xl mb-4">!</div>
                <h3 className="text-white font-semibold mb-2">Simulation Failed</h3>
                <p className="text-gray-400">{selectedRun.error_message || 'An error occurred'}</p>
              </div>
            </div>
          ) : (
            /* Completed Results */
            <div className="space-y-6">
              {/* Summary */}
              <div className="bg-gray-800 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-white">
                    {selectedRun.your_deck_name} vs {selectedRun.opponent_archetype}
                  </h2>
                  <span className="text-xs text-gray-500">{formatDate(selectedRun.completed_at)}</span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Win Rate</div>
                    <div className={`text-2xl font-bold ${getWinRateColor(selectedRun.win_rate)}`}>
                      {selectedRun.win_rate !== undefined ? `${(selectedRun.win_rate * 100).toFixed(0)}%` : '-'}
                    </div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Record</div>
                    <div className="text-2xl font-bold text-white">
                      {selectedRun.your_wins ?? 0}-{selectedRun.opponent_wins ?? 0}
                    </div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Assessment</div>
                    <div className={`text-2xl font-bold capitalize ${getAssessmentColor(selectedRun.matchup_assessment)}`}>
                      {selectedRun.matchup_assessment || '-'}
                    </div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Avg Length</div>
                    <div className="text-2xl font-bold text-white">
                      {selectedRun.average_game_length?.toFixed(1) ?? '-'} turns
                    </div>
                  </div>
                </div>

                {/* Key Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-sm font-medium text-gray-400 mb-2">Your Key Cards</h3>
                    <div className="space-y-2">
                      {(selectedRun.key_cards_for_you || []).slice(0, 5).map((card, i) => (
                        <div key={i} className="flex justify-between items-center bg-gray-700 rounded px-3 py-2">
                          <span className="text-white text-sm">{card.card}</span>
                          <span className="text-gray-400 text-xs">
                            {(card.importance * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-gray-400 mb-2">Opponent's Key Cards</h3>
                    <div className="space-y-2">
                      {(selectedRun.key_cards_against_you || []).slice(0, 5).map((card, i) => (
                        <div key={i} className="flex justify-between items-center bg-gray-700 rounded px-3 py-2">
                          <span className="text-white text-sm">{card.card}</span>
                          <span className="text-gray-400 text-xs">
                            {(card.importance * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Sideboard Guide */}
              {selectedRun.sideboard_guide && (
                <div className="bg-gray-800 rounded-lg p-6">
                  <h2 className="text-lg font-semibold text-white mb-4">Sideboard Guide</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-sm font-medium text-green-400 mb-2">Bring In</h3>
                      <div className="bg-gray-700 rounded p-3">
                        {selectedRun.sideboard_guide.in?.length > 0 ? (
                          <ul className="space-y-1">
                            {selectedRun.sideboard_guide.in.map((card, i) => (
                              <li key={i} className="text-white text-sm">+ {card}</li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-gray-400 text-sm">No changes</p>
                        )}
                      </div>
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-red-400 mb-2">Take Out</h3>
                      <div className="bg-gray-700 rounded p-3">
                        {selectedRun.sideboard_guide.out?.length > 0 ? (
                          <ul className="space-y-1">
                            {selectedRun.sideboard_guide.out.map((card, i) => (
                              <li key={i} className="text-white text-sm">- {card}</li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-gray-400 text-sm">No changes</p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Strategic Advice */}
              {(selectedRun.strategic_advice || selectedRun.mulligan_advice) && (
                <div className="bg-gray-800 rounded-lg p-6">
                  <h2 className="text-lg font-semibold text-white mb-4">Strategic Advice</h2>
                  {selectedRun.mulligan_advice && (
                    <div className="mb-4">
                      <h3 className="text-sm font-medium text-gray-400 mb-2">Mulligan Strategy</h3>
                      <p className="text-white bg-gray-700 rounded p-3 text-sm">{selectedRun.mulligan_advice}</p>
                    </div>
                  )}
                  {selectedRun.strategic_advice && selectedRun.strategic_advice.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium text-gray-400 mb-2">Key Tips</h3>
                      <ul className="space-y-2">
                        {selectedRun.strategic_advice.map((tip, i) => (
                          <li key={i} className="flex items-start">
                            <span className="text-indigo-400 mr-2">*</span>
                            <span className="text-white text-sm">{tip}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Game Details */}
              {selectedRun.games && selectedRun.games.length > 0 && (
                <div className="bg-gray-800 rounded-lg p-6">
                  <h2 className="text-lg font-semibold text-white mb-4">Game Details</h2>
                  <div className="space-y-3">
                    {selectedRun.games.map((game) => (
                      <div key={game.game_number} className="bg-gray-700 rounded-lg overflow-hidden">
                        <button
                          onClick={() => setExpandedGame(expandedGame === game.game_number ? null : game.game_number)}
                          className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-600 transition-colors"
                        >
                          <div className="flex items-center space-x-4">
                            <span className="text-gray-400 text-sm">Game {game.game_number}</span>
                            <span className={`font-medium text-sm ${game.winner === 'you' ? 'text-green-400' : 'text-red-400'}`}>
                              {game.winner === 'you' ? 'WIN' : 'LOSS'}
                            </span>
                            <span className="text-gray-400 text-sm">
                              {game.turns} turns
                            </span>
                          </div>
                          <svg
                            className={`w-5 h-5 text-gray-400 transform transition-transform ${expandedGame === game.game_number ? 'rotate-180' : ''}`}
                            fill="none" viewBox="0 0 24 24" stroke="currentColor"
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
                            {game.key_moments && game.key_moments.length > 0 && (
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
                                {game.sideboard_out && game.sideboard_out.length > 0 && (
                                  <>
                                    <span className="text-red-400 ml-4">Out: </span>
                                    <span className="text-white">{game.sideboard_out.join(', ')}</span>
                                  </>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
