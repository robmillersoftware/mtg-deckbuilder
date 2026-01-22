import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { decksApi } from '@/services/api';
import { Deck } from '@/types';
import { DeckList } from '@/components/DeckList';
import { DeckActions } from '@/components/DeckActions';
import { useDeckStore } from '@/store/deck';
import toast from 'react-hot-toast';

export function DeckViewPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [deck, setDeck] = useState<Deck | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isChangingVisibility, setIsChangingVisibility] = useState(false);
  const { setCurrentDeck } = useDeckStore();

  useEffect(() => {
    if (id) {
      loadDeck(id);
    }
  }, [id]);

  const loadDeck = async (deckId: string) => {
    setIsLoading(true);
    try {
      const response = await decksApi.getById(deckId);
      setDeck(response.data);
    } catch (error) {
      console.error('Failed to load deck:', error);
      navigate('/decks');
    } finally {
      setIsLoading(false);
    }
  };

  const handleEdit = () => {
    if (deck) {
      setCurrentDeck(deck);
      navigate('/');
    }
  };

  const handleVisibilityChange = async (newVisibility: 'private' | 'public' | 'unlisted') => {
    if (!deck?.id) return;

    setIsChangingVisibility(true);
    try {
      await decksApi.toggleVisibility(deck.id, newVisibility);
      setDeck((prev) => prev ? { ...prev, visibility: newVisibility } : null);
      toast.success(`Deck visibility changed to ${newVisibility}`);
    } catch (error) {
      console.error('Failed to change visibility:', error);
      toast.error('Failed to change visibility');
    } finally {
      setIsChangingVisibility(false);
    }
  };

  const copyShareLink = async () => {
    if (!deck?.share_token) {
      toast.error('Save the deck as public first to get a share link');
      return;
    }
    const shareUrl = `${window.location.origin}/deck/shared/${deck.share_token}`;
    try {
      await navigator.clipboard.writeText(shareUrl);
      toast.success('Share link copied to clipboard');
    } catch {
      toast.error('Failed to copy share link');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading deck...</div>
      </div>
    );
  }

  if (!deck) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-400">Deck not found</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">{deck.name}</h1>
          <p className="text-gray-400 mt-1">
            {deck.archetype || 'Standard'} • {deck.format}
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleEdit}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
          >
            Edit
          </button>
        </div>
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

      {/* Deck List */}
      <div className="grid gap-6 md:grid-cols-2">
        <DeckList
          mainDeck={deck.main_deck}
          sideboard={deck.sideboard}
          title="Deck List"
        />

        {/* Stats & Actions */}
        <div className="space-y-4">
          {/* Mana Curve (simplified) */}
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

          {/* Actions */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Actions</h3>
            <DeckActions deck={deck} />
          </div>

          {/* Visibility Settings */}
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Visibility</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Current:</span>
                <span className="text-white text-sm capitalize">{deck.visibility || 'private'}</span>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => handleVisibilityChange('private')}
                  disabled={isChangingVisibility || deck.visibility === 'private'}
                  className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    deck.visibility === 'private'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                  } disabled:opacity-50`}
                >
                  Private
                </button>
                <button
                  onClick={() => handleVisibilityChange('unlisted')}
                  disabled={isChangingVisibility || deck.visibility === 'unlisted'}
                  className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    deck.visibility === 'unlisted'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                  } disabled:opacity-50`}
                >
                  Unlisted
                </button>
                <button
                  onClick={() => handleVisibilityChange('public')}
                  disabled={isChangingVisibility || deck.visibility === 'public'}
                  className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    deck.visibility === 'public'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                  } disabled:opacity-50`}
                >
                  Public
                </button>
              </div>
              {deck.visibility !== 'private' && deck.share_token && (
                <button
                  onClick={copyShareLink}
                  className="w-full mt-2 px-4 py-2 text-sm bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
                >
                  Copy Share Link
                </button>
              )}
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
