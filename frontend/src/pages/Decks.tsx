import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { decksApi } from '@/services/api';
import { Deck } from '@/types';
import clsx from 'clsx';

export function DecksPage() {
  const [decks, setDecks] = useState<Deck[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadDecks();
  }, []);

  const loadDecks = async () => {
    setIsLoading(true);
    try {
      const response = await decksApi.list(50, 0);
      setDecks(response.data.items || []);
    } catch (error) {
      console.error('Failed to load decks:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this deck?')) return;

    try {
      await decksApi.delete(id);
      setDecks((prev) => prev.filter((d) => d.id !== id));
    } catch (error) {
      console.error('Failed to delete deck:', error);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading decks...</div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">My Decks</h1>
        <Link
          to="/"
          className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
        >
          Build New Deck
        </Link>
      </div>

      {decks.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-400 text-lg mb-4">You haven't created any decks yet</p>
          <Link
            to="/"
            className="text-primary-400 hover:text-primary-300"
          >
            Start building your first deck
          </Link>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {decks.map((deck) => (
            <div
              key={deck.id}
              className="bg-gray-900 rounded-lg overflow-hidden hover:bg-gray-850 transition-colors"
            >
              <div className="p-4">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-white">{deck.name}</h3>
                    <p className="text-sm text-gray-400 mt-1">
                      {deck.archetype || 'Standard'} • {deck.format}
                    </p>
                  </div>
                  <span
                    className={clsx(
                      'px-2 py-1 text-xs rounded',
                      deck.is_validated
                        ? 'bg-green-900 text-green-300'
                        : 'bg-yellow-900 text-yellow-300'
                    )}
                  >
                    {deck.is_validated ? 'Valid' : 'Invalid'}
                  </span>
                </div>

                {deck.strategy_summary && (
                  <p className="text-sm text-gray-400 mt-3 line-clamp-2">
                    {deck.strategy_summary}
                  </p>
                )}

                <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-800">
                  <div className="text-xs text-gray-500">
                    {(deck.main_deck || []).reduce((sum, e) => sum + e.quantity, 0)} cards • {formatDate(deck.updated_at)}
                  </div>
                  <div className="flex items-center space-x-2">
                    <Link
                      to={`/deck/${deck.id}`}
                      className="text-sm text-primary-400 hover:text-primary-300"
                    >
                      View
                    </Link>
                    <button
                      onClick={() => handleDelete(deck.id)}
                      className="text-sm text-red-400 hover:text-red-300"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
