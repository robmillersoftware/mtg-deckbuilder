import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { decksApi } from '@/services/api';
import { Deck } from '@/types';
import { DeckList } from '@/components/DeckList';
import toast from 'react-hot-toast';

export function SharedDeckViewPage() {
  const { shareToken } = useParams<{ shareToken: string }>();
  const [deck, setDeck] = useState<Deck | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (shareToken) {
      loadDeck(shareToken);
    }
  }, [shareToken]);

  const loadDeck = async (token: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await decksApi.getByShareToken(token);
      setDeck(response.data);
    } catch (err) {
      console.error('Failed to load shared deck:', err);
      setError('Deck not found or link has expired');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopyToArena = async () => {
    if (!deck) return;

    const lines: string[] = [];

    // Main deck
    for (const entry of deck.main_deck) {
      lines.push(`${entry.quantity} ${entry.card_name}`);
    }

    // Sideboard
    if (deck.sideboard?.length) {
      lines.push('');
      for (const entry of deck.sideboard) {
        lines.push(`${entry.quantity} ${entry.card_name}`);
      }
    }

    const exportText = lines.join('\n');

    try {
      await navigator.clipboard.writeText(exportText);
      toast.success('Deck copied to clipboard (Arena format)');
    } catch {
      toast.error('Failed to copy deck');
    }
  };

  const handleCopyToMTGO = async () => {
    if (!deck) return;

    const lines: string[] = [];

    // Main deck
    for (const entry of deck.main_deck) {
      lines.push(`${entry.quantity} ${entry.card_name}`);
    }

    // Sideboard
    if (deck.sideboard?.length) {
      lines.push('');
      for (const entry of deck.sideboard) {
        lines.push(`SB: ${entry.quantity} ${entry.card_name}`);
      }
    }

    const exportText = lines.join('\n');

    try {
      await navigator.clipboard.writeText(exportText);
      toast.success('Deck copied to clipboard (MTGO format)');
    } catch {
      toast.error('Failed to copy deck');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading shared deck...</div>
      </div>
    );
  }

  if (error || !deck) {
    return (
      <div className="max-w-md mx-auto mt-16 text-center">
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-8">
          <h1 className="text-xl font-bold text-red-300 mb-2">Deck Not Found</h1>
          <p className="text-gray-400">
            {error || 'This deck link may have expired or been removed.'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-2">
          <span className="px-2 py-1 bg-purple-600/30 text-purple-300 text-xs rounded">
            Shared Deck
          </span>
        </div>
        <h1 className="text-2xl font-bold text-white">{deck.name}</h1>
        <p className="text-gray-400 mt-1">
          {deck.archetype || 'Standard'} • {deck.format}
        </p>
      </div>

      {/* Strategy */}
      {deck.strategy_summary && (
        <div className="bg-gray-900 rounded-lg p-4 mb-6">
          <h2 className="text-lg font-semibold text-white mb-2">Strategy</h2>
          <p className="text-gray-300">{deck.strategy_summary}</p>
        </div>
      )}

      {/* Validation Status */}
      {!deck.is_validated && deck.validation_errors && deck.validation_errors.length > 0 && (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 mb-6">
          <h3 className="text-yellow-300 font-medium mb-2">Validation Issues</h3>
          <ul className="list-disc list-inside text-sm text-yellow-200">
            {deck.validation_errors.map((error, index) => (
              <li key={index}>{error.message}</li>
            ))}
          </ul>
        </div>
      )}

      {deck.is_validated && (
        <div className="bg-green-900/30 border border-green-700 rounded-lg p-3 mb-6">
          <span className="text-green-300 font-medium">Deck is valid for Standard</span>
        </div>
      )}

      {/* Deck List */}
      <div className="grid gap-6 md:grid-cols-2">
        <DeckList
          mainDeck={deck.main_deck}
          sideboard={deck.sideboard}
          commander={deck.commander}
          format={deck.format}
          title="Deck List"
        />

        {/* Stats & Actions */}
        <div className="space-y-4">
          {/* Statistics */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Statistics</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Main Deck</span>
                <span className="text-white">
                  {deck.main_deck.reduce((sum, e) => sum + e.quantity, 0)} cards
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Sideboard</span>
                <span className="text-white">
                  {deck.sideboard.reduce((sum, e) => sum + e.quantity, 0)} cards
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Unique Cards</span>
                <span className="text-white">
                  {deck.main_deck.length + deck.sideboard.length}
                </span>
              </div>
            </div>
          </div>

          {/* Export Actions (read-only view, no edit) */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Export</h3>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleCopyToArena}
                className="px-4 py-2 rounded-lg font-medium bg-blue-600 hover:bg-blue-700 text-white transition-colors"
              >
                Copy to Arena
              </button>
              <button
                onClick={handleCopyToMTGO}
                className="px-4 py-2 rounded-lg font-medium bg-blue-600 hover:bg-blue-700 text-white transition-colors"
              >
                Copy to MTGO
              </button>
            </div>
          </div>

          {/* Matchup Notes */}
          {deck.matchup_notes && Object.keys(deck.matchup_notes).length > 0 && (
            <div className="bg-gray-900 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-white mb-4">Matchup Notes</h3>
              <div className="space-y-3">
                {Object.entries(deck.matchup_notes).map(([matchup, note]) => (
                  <div key={matchup}>
                    <h4 className="text-sm font-medium text-gray-300">{matchup}</h4>
                    <p className="text-sm text-gray-400">{note}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
