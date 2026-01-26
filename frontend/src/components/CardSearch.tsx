import { useState, useCallback, useRef, useEffect } from 'react';
import { useCardSearch } from '@/hooks/useCards';
import { CardTooltip } from './CardTooltip';
import { Card } from '@/types';
import clsx from 'clsx';

interface CardSearchProps {
  onAddCard: (cardName: string, target: 'main' | 'sideboard') => void;
  existingCards?: Set<string>;
}

export function CardSearch({ onAddCard, existingCards = new Set() }: CardSearchProps) {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const { cards, isLoading, search } = useCardSearch();
  const inputRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  // Debounced search
  const handleSearch = useCallback((value: string) => {
    setQuery(value);

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    if (value.trim().length >= 2) {
      debounceRef.current = setTimeout(() => {
        search({ q: value.trim() });
      }, 300);
    }
  }, [search]);

  // Close on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('.card-search-container')) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleAddCard = (cardName: string, target: 'main' | 'sideboard') => {
    onAddCard(cardName, target);
    // Keep search open for adding more cards
  };

  const hasResults = cards.length > 0 && query.trim().length >= 2;

  return (
    <div className="card-search-container relative">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => handleSearch(e.target.value)}
            onFocus={() => setIsOpen(true)}
            placeholder="Search cards to add..."
            className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
          />
          {isLoading && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
            </div>
          )}
        </div>
      </div>

      {/* Search Results Dropdown */}
      {isOpen && hasResults && (
        <div className="absolute z-50 mt-1 w-full bg-gray-800 border border-gray-600 rounded-lg shadow-xl max-h-80 overflow-y-auto">
          {cards.map((card: Card) => {
            const isInDeck = existingCards.has(card.name);

            return (
              <div
                key={card.id}
                className={clsx(
                  'flex items-center justify-between px-3 py-2 hover:bg-gray-700 border-b border-gray-700 last:border-b-0',
                  isInDeck && 'bg-gray-750'
                )}
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <CardTooltip cardName={card.name}>
                    <span className="text-white text-sm truncate cursor-help">
                      {card.name}
                    </span>
                  </CardTooltip>
                  {card.mana_cost && (
                    <span className="text-gray-400 text-xs whitespace-nowrap">
                      {card.mana_cost}
                    </span>
                  )}
                  {isInDeck && (
                    <span className="text-purple-400 text-xs">(in deck)</span>
                  )}
                </div>

                <div className="flex items-center gap-1 ml-2">
                  <button
                    onClick={() => handleAddCard(card.name, 'main')}
                    className="px-2 py-1 text-xs bg-purple-600 hover:bg-purple-500 text-white rounded transition-colors"
                    title="Add to main deck"
                  >
                    +Main
                  </button>
                  <button
                    onClick={() => handleAddCard(card.name, 'sideboard')}
                    className="px-2 py-1 text-xs bg-gray-600 hover:bg-gray-500 text-white rounded transition-colors"
                    title="Add to sideboard"
                  >
                    +Side
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* No results message */}
      {isOpen && query.trim().length >= 2 && !isLoading && cards.length === 0 && (
        <div className="absolute z-50 mt-1 w-full bg-gray-800 border border-gray-600 rounded-lg shadow-xl p-4 text-center text-gray-400 text-sm">
          No cards found for "{query}"
        </div>
      )}
    </div>
  );
}
