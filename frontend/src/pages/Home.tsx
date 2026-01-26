import { useState } from 'react';
import { ChatWindow } from '@/components/ChatWindow';
import { DeckList } from '@/components/DeckList';
import { DeckActions } from '@/components/DeckActions';
import { ConversationList } from '@/components/ConversationList';
import { useDeckStore } from '@/store/deck';
import { useAuth } from '@/hooks/useAuth';
import clsx from 'clsx';

export function HomePage() {
  const { currentDeck, updateCardQuantity, addCard } = useDeckStore();
  const { isAuthenticated } = useAuth();
  const [mobileTab, setMobileTab] = useState<'chat' | 'deck'>('chat');

  const handleQuantityChange = (
    cardName: string,
    quantity: number,
    target: 'main' | 'sideboard'
  ) => {
    updateCardQuantity(cardName, quantity, target);
  };

  const handleAddCard = (cardName: string, target: 'main' | 'sideboard') => {
    addCard({ card_name: cardName, quantity: 1 }, target);
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
              commander={currentDeck.commander}
              format={currentDeck.format}
              title={currentDeck.name || 'Current Deck'}
              cardExplanations={currentDeck.card_explanations}
              onQuantityChange={handleQuantityChange}
              onAddCard={handleAddCard}
              editable
              className="flex-1 overflow-y-auto"
            />
            {isAuthenticated && (
              <DeckActions deck={currentDeck} />
            )}
          </>
        ) : (
          <div className="bg-gray-900 rounded-lg p-4 text-center text-gray-400">
            <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-gray-800 flex items-center justify-center">
              <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </div>
            <p className="text-lg mb-2 text-white">No deck yet</p>
            <p className="text-sm">
              Tell Spellbook what kind of deck you want to build
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
