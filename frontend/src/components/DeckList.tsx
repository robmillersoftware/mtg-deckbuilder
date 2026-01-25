import { useMemo } from 'react';
import { DeckEntry, Card } from '@/types';
import { CardTooltip } from './CardTooltip';
import { CardSearch } from './CardSearch';
import clsx from 'clsx';

interface DeckListProps {
  mainDeck: DeckEntry[];
  sideboard: DeckEntry[];
  commander?: DeckEntry;
  format?: string;
  title?: string;
  cardExplanations?: Record<string, string>;
  onCardClick?: (cardName: string) => void;
  onQuantityChange?: (cardName: string, quantity: number, target: 'main' | 'sideboard') => void;
  onAddCard?: (cardName: string, target: 'main' | 'sideboard') => void;
  editable?: boolean;
  className?: string;
}

export function DeckList({
  mainDeck,
  sideboard,
  commander,
  format,
  title,
  cardExplanations,
  onCardClick,
  onQuantityChange,
  onAddCard,
  editable = false,
  className,
}: DeckListProps) {
  // Check if this is a commander format
  const isCommanderFormat = format === 'commander' || format === 'cedh';
  // Group cards by type
  const groupedMain = useMemo(() => groupCardsByType(mainDeck), [mainDeck]);

  const mainCount = useMemo(
    () => mainDeck.reduce((sum, e) => sum + e.quantity, 0),
    [mainDeck]
  );

  const sideboardCount = useMemo(
    () => sideboard.reduce((sum, e) => sum + e.quantity, 0),
    [sideboard]
  );

  // Track existing card names for search highlighting
  const existingCards = useMemo(() => {
    const names = new Set<string>();
    mainDeck.forEach((e) => names.add(e.card_name));
    sideboard.forEach((e) => names.add(e.card_name));
    return names;
  }, [mainDeck, sideboard]);

  return (
    <div className={clsx('bg-gray-900 rounded-lg', className)}>
      {title && (
        <div className="px-4 py-3 border-b border-gray-700">
          <h2 className="text-lg font-semibold text-white">{title}</h2>
        </div>
      )}

      <div className="p-4 space-y-4">
        {/* Card Search - shown when editable */}
        {editable && onAddCard && (
          <div className="pb-3 border-b border-gray-700">
            <CardSearch onAddCard={onAddCard} existingCards={existingCards} />
          </div>
        )}
        {/* Commander - shown for commander/cEDH formats */}
        {isCommanderFormat && commander?.card_name && (
          <div className="pb-3 border-b border-gray-700">
            <h3 className="text-sm font-medium text-purple-400 mb-2">
              Commander
            </h3>
            <div className="space-y-0.5">
              <CardEntry
                entry={{ ...commander, quantity: 1 }}
                target="main"
                explanation={cardExplanations?.[commander.card_name]}
                onClick={onCardClick}
                editable={false}
              />
            </div>
          </div>
        )}

        {/* Main Deck */}
        <div>
          <h3 className="text-sm font-medium text-gray-400 mb-2">
            Main Deck ({mainCount})
          </h3>

          {Object.entries(groupedMain).map(([type, cards]) => (
            <div key={type} className="mb-3">
              <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">
                {type} ({cards.reduce((sum, c) => sum + c.quantity, 0)})
              </h4>
              <div className="space-y-0.5">
                {cards.map((entry) => (
                  <CardEntry
                    key={entry.card_name}
                    entry={entry}
                    target="main"
                    explanation={cardExplanations?.[entry.card_name]}
                    onClick={onCardClick}
                    onQuantityChange={onQuantityChange}
                    editable={editable}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Sideboard */}
        {sideboard.length > 0 && (
          <div className="pt-4 border-t border-gray-700">
            <h3 className="text-sm font-medium text-gray-400 mb-2">
              Sideboard ({sideboardCount})
            </h3>
            <div className="space-y-0.5">
              {sideboard.map((entry) => (
                <CardEntry
                  key={entry.card_name}
                  entry={entry}
                  target="sideboard"
                  explanation={cardExplanations?.[entry.card_name]}
                  onClick={onCardClick}
                  onQuantityChange={onQuantityChange}
                  editable={editable}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="px-4 py-3 border-t border-gray-700 text-xs text-gray-500">
        <div className="flex justify-between">
          {isCommanderFormat && commander?.card_name ? (
            <>
              <span>Commander: 1</span>
              <span>Main: {mainCount} cards</span>
            </>
          ) : (
            <>
              <span>Main: {mainCount} cards</span>
              <span>Sideboard: {sideboardCount} cards</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

interface CardEntryProps {
  entry: DeckEntry;
  target: 'main' | 'sideboard';
  explanation?: string;
  onClick?: (cardName: string) => void;
  onQuantityChange?: (cardName: string, quantity: number, target: 'main' | 'sideboard') => void;
  editable?: boolean;
}

function CardEntry({ entry, target, explanation, onClick, onQuantityChange, editable }: CardEntryProps) {
  const handleClick = () => {
    onClick?.(entry.card_name);
  };

  const handleDecrease = (e: React.MouseEvent) => {
    e.stopPropagation();
    onQuantityChange?.(entry.card_name, entry.quantity - 1, target);
  };

  const handleIncrease = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (entry.quantity < 4) {
      onQuantityChange?.(entry.card_name, entry.quantity + 1, target);
    }
  };

  return (
    <div
      className={clsx(
        'flex items-center justify-between py-1 px-2 rounded hover:bg-gray-800 transition-colors',
        onClick && 'cursor-pointer'
      )}
      onClick={handleClick}
    >
      <div className="flex items-center space-x-2">
        <span className="text-gray-400 w-4 text-right text-sm">
          {entry.quantity}
        </span>
        <CardTooltip cardName={entry.card_name} explanation={explanation}>
          <span className="text-white text-sm">{entry.card_name}</span>
        </CardTooltip>
      </div>

      {editable && (
        <div className="flex items-center space-x-1">
          <button
            onClick={handleDecrease}
            className="w-5 h-5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 text-xs flex items-center justify-center"
          >
            -
          </button>
          <button
            onClick={handleIncrease}
            disabled={entry.quantity >= 4}
            className={clsx(
              'w-5 h-5 rounded text-xs flex items-center justify-center',
              entry.quantity >= 4
                ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
            )}
          >
            +
          </button>
        </div>
      )}
    </div>
  );
}

// Known basic land names and common land patterns
const BASIC_LANDS = ['plains', 'island', 'swamp', 'mountain', 'forest', 'wastes'];
const LAND_KEYWORDS = [
  'land', 'pool', 'gate', 'sanctum', 'passage', 'canal', 'vents', 'falls',
  'tomb', 'crypt', 'grove', 'garden', 'fountain', 'shrine', 'temple',
  'spire', 'citadel', 'fortress', 'castle', 'manor', 'cave', 'wilds',
  'verge', 'triome', 'pathway', 'channel', 'heights', 'depths', 'delta',
  'strand', 'flats', 'mesa', 'tarn', 'mire', 'heath', 'rainforest',
  'catacomb', 'foundry', 'stream', 'copse', 'hollow', 'reach'
];

// Check if a card name looks like a land
function isLikelyLand(cardName: string): boolean {
  // For DFCs (double-faced cards), only check the front face name
  // e.g., "Ojer Axonil, Deepest Might // Temple of Power" -> only check "Ojer Axonil, Deepest Might"
  const frontFace = cardName.includes(' // ') ? cardName.split(' // ')[0] : cardName;
  const lower = frontFace.toLowerCase();

  // Check basic lands
  if (BASIC_LANDS.includes(lower)) return true;

  // Split into words and check if any word matches a land keyword
  // This prevents false positives like "Scavenging" matching "cave" or "Negate" matching "gate"
  const words = lower.split(/\s+/);
  for (const word of words) {
    if (LAND_KEYWORDS.includes(word)) return true;
  }

  return false;
}

// Helper to group cards by type
function groupCardsByType(cards: DeckEntry[]): Record<string, DeckEntry[]> {
  const groups: Record<string, DeckEntry[]> = {
    Creatures: [],
    Planeswalkers: [],
    Instants: [],
    Sorceries: [],
    Artifacts: [],
    Enchantments: [],
    Lands: [],
    Other: [],
  };

  for (const entry of cards) {
    const card = entry.card as Card | undefined;
    const typeLine = card?.type_line?.toLowerCase() || '';
    const cardName = entry.card_name;

    // If we have type_line from card data, use it
    if (typeLine) {
      if (typeLine.includes('creature')) {
        groups.Creatures.push(entry);
      } else if (typeLine.includes('planeswalker')) {
        groups.Planeswalkers.push(entry);
      } else if (typeLine.includes('instant')) {
        groups.Instants.push(entry);
      } else if (typeLine.includes('sorcery')) {
        groups.Sorceries.push(entry);
      } else if (typeLine.includes('artifact')) {
        groups.Artifacts.push(entry);
      } else if (typeLine.includes('enchantment')) {
        groups.Enchantments.push(entry);
      } else if (typeLine.includes('land')) {
        groups.Lands.push(entry);
      } else {
        groups.Other.push(entry);
      }
    } else {
      // Fallback: check if card name looks like a land
      if (isLikelyLand(cardName)) {
        groups.Lands.push(entry);
      } else {
        groups.Other.push(entry);
      }
    }
  }

  // Remove empty groups
  return Object.fromEntries(
    Object.entries(groups).filter(([, cards]) => cards.length > 0)
  );
}
