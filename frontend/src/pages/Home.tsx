import { useState } from 'react';
import { ChatWindow } from '@/components/ChatWindow';
import { DeckList } from '@/components/DeckList';
import { DeckActions } from '@/components/DeckActions';
import { ConversationList } from '@/components/ConversationList';
import { useDeckStore } from '@/store/deck';
import { useAuth } from '@/hooks/useAuth';
import clsx from 'clsx';

export function HomePage() {
  const { currentDeck, updateCardQuantity } = useDeckStore();
  const { isAuthenticated } = useAuth();
  const [mobileTab, setMobileTab] = useState<'chat' | 'deck'>('chat');

  const handleQuantityChange = (
    cardName: string,
    quantity: number,
    target: 'main' | 'sideboard'
  ) => {
    updateCardQuantity(cardName, quantity, target);
  };

  const deckCardCount = (currentDeck?.main_deck || []).reduce((sum, e) => sum + e.quantity, 0);

  return (
    <div className="flex flex-col md:flex-row gap-4 md:gap-6 h-[calc(100vh-180px)]">
      {/* Mobile Tab Switcher */}
      <div className="md:hidden flex bg-gray-900 rounded-lg p-1">
        <button
          onClick={() => setMobileTab('chat')}
          className={clsx(
            'flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors',
            mobileTab === 'chat'
              ? 'bg-primary-600 text-white'
              : 'text-gray-400 hover:text-white'
          )}
        >
          Chat
        </button>
        <button
          onClick={() => setMobileTab('deck')}
          className={clsx(
            'flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors relative',
            mobileTab === 'deck'
              ? 'bg-primary-600 text-white'
              : 'text-gray-400 hover:text-white'
          )}
        >
          Deck
          {deckCardCount > 0 && (
            <span className="absolute -top-1 -right-1 bg-green-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
              {deckCardCount}
            </span>
          )}
        </button>
      </div>

      {/* Left Sidebar - History (desktop only) */}
      <div className="w-64 flex-shrink-0 hidden lg:block">
        <ConversationList />
      </div>

      {/* Main Chat Area - visible on desktop, or mobile when chat tab selected */}
      <div className={clsx(
        'flex-1 min-w-0',
        mobileTab !== 'chat' && 'hidden md:block'
      )}>
        <ChatWindow className="h-full rounded-lg" />
      </div>

      {/* Right Sidebar - Deck - visible on desktop, or mobile when deck tab selected */}
      <div className={clsx(
        'md:w-80 flex-shrink-0 flex flex-col gap-4',
        mobileTab !== 'deck' && 'hidden md:flex'
      )}>
        {currentDeck ? (
          <>
            <DeckList
              mainDeck={currentDeck.main_deck || []}
              sideboard={currentDeck.sideboard || []}
              title={currentDeck.name || 'Current Deck'}
              cardExplanations={currentDeck.card_explanations}
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
