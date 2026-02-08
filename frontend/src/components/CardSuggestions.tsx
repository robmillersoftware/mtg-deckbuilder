import { useState } from 'react';
import { CardSuggestionGroup, CardSuggestionItem } from '@/types';
import { useDeckStore } from '@/store/deck';
import { CardTooltip } from '@/components/CardTooltip';
import clsx from 'clsx';

interface CardSuggestionsProps {
  groups: CardSuggestionGroup[];
}

export function CardSuggestions({ groups }: CardSuggestionsProps) {
  return (
    <div className="space-y-4">
      {groups.map((group, i) => (
        <SuggestionGroup key={`${group.role}-${i}`} group={group} />
      ))}
    </div>
  );
}

function SuggestionGroup({ group }: { group: CardSuggestionGroup }) {
  const { addCard } = useDeckStore();
  const [addedCards, setAddedCards] = useState<Record<string, number>>({});

  const handleAddCard = (card: CardSuggestionItem) => {
    addCard(
      {
        card_name: card.card_name,
        quantity: card.quantity,
        card: {
          mana_cost: card.mana_cost,
          type_line: card.type_line,
          image_uri: card.image_uri,
        },
      } as any,
      'main'
    );
    setAddedCards((prev) => ({
      ...prev,
      [card.card_name]: (prev[card.card_name] || 0) + card.quantity,
    }));
  };

  const handleAddAll = () => {
    for (const card of group.cards) {
      if (!addedCards[card.card_name]) {
        handleAddCard(card);
      }
    }
  };

  const allAdded = group.cards.every((c) => addedCards[c.card_name]);

  return (
    <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-3">
      {/* Group Header */}
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-primary-400">{group.group_name}</h4>
        {group.is_batch && (
          <button
            onClick={handleAddAll}
            disabled={allAdded}
            className={clsx(
              'px-3 py-1 text-xs font-medium rounded-full transition-colors',
              allAdded
                ? 'bg-green-900/30 text-green-400 cursor-default'
                : 'bg-primary-600 hover:bg-primary-700 text-white'
            )}
          >
            {allAdded ? 'All Added' : 'Add All'}
          </button>
        )}
      </div>

      {/* Cards */}
      <div className="space-y-2">
        {group.cards.map((card) => (
          <SuggestionCard
            key={card.card_name}
            card={card}
            isAdded={!!addedCards[card.card_name]}
            addedQty={addedCards[card.card_name] || 0}
            onAdd={() => handleAddCard(card)}
          />
        ))}
      </div>
    </div>
  );
}

function SuggestionCard({
  card,
  isAdded,
  addedQty,
  onAdd,
}: {
  card: CardSuggestionItem;
  isAdded: boolean;
  addedQty: number;
  onAdd: () => void;
}) {
  return (
    <div className="flex items-center gap-2 sm:gap-3 group">
      {/* Card image preview (small) - hidden on mobile to prevent overflow */}
      <div className="relative w-10 h-14 flex-shrink-0 rounded overflow-hidden bg-gray-700 cursor-pointer hidden sm:block">
        {card.image_uri ? (
          <img
            src={card.image_uri}
            alt={card.card_name}
            className="w-full h-full object-cover"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-[8px] text-gray-500 text-center px-0.5">
            {card.card_name.slice(0, 8)}
          </div>
        )}
      </div>

      {/* Card info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <CardTooltip cardName={card.card_name}>
            <span className="text-sm font-medium text-gray-200 hover:text-white transition-colors">
              {card.card_name}
            </span>
          </CardTooltip>
          {card.mana_cost && (
            <span className="text-xs text-gray-500 flex-shrink-0">{card.mana_cost}</span>
          )}
        </div>
        {card.type_line && (
          <p className="text-xs text-gray-500 truncate">{card.type_line}</p>
        )}
        {card.reasoning && (
          <p className="text-xs text-gray-400 mt-0.5">{card.reasoning}</p>
        )}
      </div>

      {/* Quantity + Add button */}
      <div className="flex items-center gap-1.5 sm:gap-2 flex-shrink-0">
        <span className="text-xs text-gray-500">{card.quantity}x</span>
        <button
          onClick={onAdd}
          disabled={isAdded}
          className={clsx(
            'px-2 sm:px-2.5 py-1 text-xs font-medium rounded transition-colors',
            isAdded
              ? 'bg-green-900/30 text-green-400 cursor-default'
              : 'bg-gray-700 hover:bg-primary-600 text-gray-300 hover:text-white'
          )}
        >
          {isAdded ? `Added (${addedQty})` : '+ Add'}
        </button>
      </div>
    </div>
  );
}
