import { useState } from 'react';
import { Deck, SideboardMatrixResponse, MatchupSideboardPlan } from '@/types';
import { decksApi } from '@/services/api';
import { CardTooltip } from './CardTooltip';
import toast from 'react-hot-toast';
import clsx from 'clsx';

interface SideboardMatrixProps {
  deck: Partial<Deck>;
  onClose: () => void;
}

export function SideboardMatrix({ deck, onClose }: SideboardMatrixProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [matrix, setMatrix] = useState<SideboardMatrixResponse | null>(null);
  const [selectedMatchup, setSelectedMatchup] = useState<MatchupSideboardPlan | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!deck.id) {
      toast.error('Save the deck first to generate sideboard guide');
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const response = await decksApi.getSideboardMatrix(deck.id);
      setMatrix(response.data);
      if (response.data.matchups?.length > 0) {
        setSelectedMatchup(response.data.matchups[0]);
      }
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } };
      const message = axiosError.response?.data?.detail || 'Failed to generate sideboard guide';
      setError(message);
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-xl w-full max-w-5xl max-h-[90vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-700 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white">Sideboard Guide</h2>
            <p className="text-sm text-gray-400 mt-1">
              {deck.name} {deck.archetype && `(${deck.archetype})`}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors p-2"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {!matrix && !isLoading && (
            <div className="flex-1 flex flex-col items-center justify-center p-8">
              <div className="text-center max-w-md">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary-900/50 flex items-center justify-center">
                  <svg className="w-8 h-8 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Generate Your Sideboard Guide
                </h3>
                <p className="text-gray-400 mb-6">
                  Get AI-powered sideboard recommendations for every matchup in the current meta.
                  Learn what to bring in, take out, and how to approach each game.
                </p>
                {error && (
                  <p className="text-red-400 text-sm mb-4">{error}</p>
                )}
                <button
                  onClick={handleGenerate}
                  disabled={!deck.id || !deck.sideboard?.length}
                  className={clsx(
                    'px-6 py-3 rounded-lg font-medium transition-colors',
                    deck.id && deck.sideboard?.length
                      ? 'bg-primary-600 hover:bg-primary-700 text-white'
                      : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  )}
                >
                  Generate Sideboard Guide
                </button>
                {!deck.id && (
                  <p className="text-yellow-500 text-xs mt-2">Save the deck first</p>
                )}
                {deck.id && !deck.sideboard?.length && (
                  <p className="text-yellow-500 text-xs mt-2">Add sideboard cards first</p>
                )}
              </div>
            </div>
          )}

          {isLoading && (
            <div className="flex-1 flex flex-col items-center justify-center p-8">
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-500 border-t-transparent mb-4"></div>
              <p className="text-gray-400">Analyzing matchups and generating sideboard plans...</p>
              <p className="text-gray-500 text-sm mt-2">This may take a moment</p>
            </div>
          )}

          {matrix && !isLoading && (
            <div className="flex-1 flex overflow-hidden">
              {/* Matchup List */}
              <div className="w-64 border-r border-gray-700 overflow-y-auto">
                <div className="p-3 border-b border-gray-700">
                  <h3 className="text-sm font-medium text-gray-400">Matchups</h3>
                </div>
                <div className="divide-y divide-gray-800">
                  {matrix.matchups.map((matchup, index) => (
                    <button
                      key={index}
                      onClick={() => setSelectedMatchup(matchup)}
                      className={clsx(
                        'w-full text-left px-4 py-3 transition-colors',
                        selectedMatchup === matchup
                          ? 'bg-primary-900/50 border-l-2 border-primary-500'
                          : 'hover:bg-gray-800'
                      )}
                    >
                      <div className="font-medium text-white text-sm">{matchup.matchup}</div>
                      <div className="text-xs text-gray-400 mt-1 line-clamp-1">
                        {matchup.matchup_description}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Selected Matchup Detail */}
              {selectedMatchup && (
                <div className="flex-1 overflow-y-auto p-6">
                  <div className="mb-6">
                    <h3 className="text-xl font-bold text-white mb-2">{selectedMatchup.matchup}</h3>
                    <p className="text-gray-400">{selectedMatchup.matchup_description}</p>
                  </div>

                  {/* Cards In/Out Grid */}
                  <div className="grid md:grid-cols-2 gap-6 mb-6">
                    {/* Cards In */}
                    <div className="bg-green-900/20 border border-green-800/50 rounded-lg p-4">
                      <h4 className="font-semibold text-green-400 flex items-center gap-2 mb-3">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        Cards In ({selectedMatchup.cards_in.reduce((sum, c) => sum + c.quantity, 0)})
                      </h4>
                      <div className="space-y-2">
                        {selectedMatchup.cards_in.map((card, idx) => (
                          <div key={idx} className="flex items-start gap-2">
                            <span className="text-green-400 font-medium w-4 text-right">
                              {card.quantity}
                            </span>
                            <div className="flex-1">
                              <CardTooltip cardName={card.card_name}>
                                <span className="text-white hover:text-green-300 cursor-help">
                                  {card.card_name}
                                </span>
                              </CardTooltip>
                              <p className="text-xs text-gray-400 mt-0.5">{card.reasoning}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Cards Out */}
                    <div className="bg-red-900/20 border border-red-800/50 rounded-lg p-4">
                      <h4 className="font-semibold text-red-400 flex items-center gap-2 mb-3">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                        </svg>
                        Cards Out ({selectedMatchup.cards_out.reduce((sum, c) => sum + c.quantity, 0)})
                      </h4>
                      <div className="space-y-2">
                        {selectedMatchup.cards_out.map((card, idx) => (
                          <div key={idx} className="flex items-start gap-2">
                            <span className="text-red-400 font-medium w-4 text-right">
                              {card.quantity}
                            </span>
                            <div className="flex-1">
                              <CardTooltip cardName={card.card_name}>
                                <span className="text-white hover:text-red-300 cursor-help">
                                  {card.card_name}
                                </span>
                              </CardTooltip>
                              <p className="text-xs text-gray-400 mt-0.5">{card.reasoning}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Strategy Notes */}
                  <div className="bg-gray-800 rounded-lg p-4 mb-6">
                    <h4 className="font-semibold text-white mb-2">Strategy Notes</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">{selectedMatchup.strategy_notes}</p>
                  </div>

                  {/* Key Cards */}
                  <div className="grid md:grid-cols-2 gap-4">
                    {selectedMatchup.key_cards_to_find.length > 0 && (
                      <div className="bg-blue-900/20 border border-blue-800/50 rounded-lg p-4">
                        <h4 className="font-semibold text-blue-400 text-sm mb-2">Key Cards to Find</h4>
                        <div className="flex flex-wrap gap-1">
                          {selectedMatchup.key_cards_to_find.map((card, idx) => (
                            <CardTooltip key={idx} cardName={card}>
                              <span className="inline-block px-2 py-1 bg-blue-900/50 text-blue-200 text-xs rounded cursor-help hover:bg-blue-800/50">
                                {card}
                              </span>
                            </CardTooltip>
                          ))}
                        </div>
                      </div>
                    )}

                    {selectedMatchup.cards_to_play_around.length > 0 && (
                      <div className="bg-yellow-900/20 border border-yellow-800/50 rounded-lg p-4">
                        <h4 className="font-semibold text-yellow-400 text-sm mb-2">Play Around</h4>
                        <div className="flex flex-wrap gap-1">
                          {selectedMatchup.cards_to_play_around.map((card, idx) => (
                            <CardTooltip key={idx} cardName={card}>
                              <span className="inline-block px-2 py-1 bg-yellow-900/50 text-yellow-200 text-xs rounded cursor-help hover:bg-yellow-800/50">
                                {card}
                              </span>
                            </CardTooltip>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* General Notes Footer */}
          {matrix?.general_sideboard_notes && (
            <div className="px-6 py-4 border-t border-gray-700 bg-gray-800/50">
              <h4 className="font-semibold text-white text-sm mb-1">General Sideboard Notes</h4>
              <p className="text-gray-400 text-sm">{matrix.general_sideboard_notes}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
