import { ChatWindow } from '@/components/ChatWindow';
import { DeckList } from '@/components/DeckList';
import { DeckActions } from '@/components/DeckActions';
import { ConversationList } from '@/components/ConversationList';
import { useDeckStore } from '@/store/deck';
import { useAuth } from '@/hooks/useAuth';

export function HomePage() {
  const { currentDeck, updateCardQuantity } = useDeckStore();
  const { isAuthenticated } = useAuth();

  const handleQuantityChange = (
    cardName: string,
    quantity: number,
    target: 'main' | 'sideboard'
  ) => {
    updateCardQuantity(cardName, quantity, target);
  };

  return (
    <div className="flex gap-6 h-[calc(100vh-180px)]">
      {/* Left Sidebar - History */}
      <div className="w-64 flex-shrink-0 hidden lg:block">
        <ConversationList />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 min-w-0">
        <ChatWindow className="h-full rounded-lg" />
      </div>

      {/* Right Sidebar - Deck */}
      <div className="w-80 flex-shrink-0 hidden md:flex flex-col gap-4">
        {currentDeck ? (
          <>
            <DeckList
              mainDeck={currentDeck.main_deck || []}
              sideboard={currentDeck.sideboard || []}
              title={currentDeck.name || 'Current Deck'}
              onQuantityChange={handleQuantityChange}
              editable
              className="flex-1 overflow-y-auto"
            />
            {isAuthenticated && (
              <DeckActions deck={currentDeck} />
            )}
          </>
        ) : (
          <div className="bg-gray-900 rounded-lg p-4 text-center text-gray-400">
            <p className="text-lg mb-2">No deck yet</p>
            <p className="text-sm">
              Tell Spellbook what kind of deck you want to build
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
