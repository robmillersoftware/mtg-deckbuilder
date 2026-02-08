import { useState, useRef, useEffect, useMemo } from 'react';
import { useChat } from '@/hooks/useChat';
import { useDeckStore } from '@/store/deck';
import { useAuth } from '@/hooks/useAuth';
import { guidedBuildApi } from '@/services/api';
import { DeckList } from '@/components/DeckList';
import { DeckActions } from '@/components/DeckActions';
import { Message } from '@/types';
import ReactMarkdown, { Components } from 'react-markdown';
import { CardTooltip } from '@/components/CardTooltip';
import clsx from 'clsx';

const FORMAT_OPTIONS = [
  { value: 'standard', label: 'Standard' },
  { value: 'historic', label: 'Historic' },
  { value: 'modern', label: 'Modern' },
  { value: 'legacy', label: 'Legacy' },
  { value: 'cedh', label: 'cEDH' },
];

const STARTER_PROMPTS = [
  { label: 'Build around a card', prompt: "I want to build around ", incomplete: true },
  { label: "What's good right now?", prompt: "What decks are performing well in the current meta?" },
  { label: 'Aggressive deck', prompt: "Build me a fast aggressive deck that can win quickly" },
  { label: 'Control deck', prompt: "Build me a control deck with lots of answers and card draw" },
  { label: 'Pick for me', prompt: "Build me the best deck for the current meta. Surprise me." },
  { label: 'Combo deck', prompt: "Build me a combo deck with a powerful win condition" },
];

const PROGRESS_MESSAGES = [
  'Thinking about your request...',
  'Searching the card database...',
  'Evaluating synergies...',
  'Balancing the mana curve...',
  'Tuning the list...',
  'Almost there...',
];

const COLOR_INFO: Record<string, { label: string; bg: string }> = {
  W: { label: 'White', bg: 'bg-amber-100' },
  U: { label: 'Blue', bg: 'bg-blue-500' },
  B: { label: 'Black', bg: 'bg-gray-800' },
  R: { label: 'Red', bg: 'bg-red-500' },
  G: { label: 'Green', bg: 'bg-green-500' },
};

interface DeckStats {
  main_deck_count: number;
  sideboard_count: number;
  target_main: number;
  target_sideboard: number;
  creature_count: number;
  spell_count: number;
  land_count: number;
  mana_curve: Record<string, number>;
  colors: Record<string, number>;
  issues: string[];
  suggestions: string[];
}

export function GuidedBuilderPage() {
  const { isAuthenticated } = useAuth();
  const { currentDeck, updateCardQuantity, addCard } = useDeckStore();
  const {
    messages,
    isLoading,
    suggestions,
    format,
    setFormat,
    sendMessage,
    startNewConversation,
  } = useChat();

  const [input, setInput] = useState('');
  const [progressIdx, setProgressIdx] = useState(0);
  const [mobileTab, setMobileTab] = useState<'chat' | 'deck'>('chat');
  const [deckStats, setDeckStats] = useState<DeckStats | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Progress message rotation
  useEffect(() => {
    if (!isLoading) { setProgressIdx(0); return; }
    const interval = setInterval(() => {
      setProgressIdx((prev) => (prev + 1) % PROGRESS_MESSAGES.length);
    }, 2500);
    return () => clearInterval(interval);
  }, [isLoading]);

  // Fetch deck stats when deck changes
  useEffect(() => {
    if (!currentDeck?.main_deck?.length) {
      setDeckStats(null);
      return;
    }
    const fetchStats = async () => {
      try {
        const res = await guidedBuildApi.analyze(
          currentDeck.main_deck || [],
          currentDeck.sideboard || [],
          currentDeck.format || 'standard',
        );
        setDeckStats(res.data);
      } catch {
        // Stats are nice-to-have, don't block on failure
      }
    };
    fetchStats();
  }, [currentDeck?.main_deck, currentDeck?.sideboard]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      sendMessage(input.trim());
      setInput('');
    }
  };

  const handleStarterClick = (prompt: string, incomplete?: boolean) => {
    if (incomplete) {
      setInput(prompt);
    } else {
      sendMessage(prompt);
    }
  };

  const deckCardCount = (currentDeck?.main_deck || []).reduce(
    (sum, e) => sum + e.quantity, 0
  );

  const hasMessages = messages.length > 0;

  return (
    <div className="flex flex-col md:flex-row gap-4 md:gap-6 h-[calc(100vh-180px)]">
      {/* Mobile Tab Switcher */}
      <div className="md:hidden flex bg-gray-900 rounded-lg p-1">
        <button
          onClick={() => setMobileTab('chat')}
          className={clsx(
            'flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors',
            mobileTab === 'chat' ? 'bg-primary-600 text-white' : 'text-gray-400 hover:text-white'
          )}
        >
          Chat
        </button>
        <button
          onClick={() => setMobileTab('deck')}
          className={clsx(
            'flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors relative',
            mobileTab === 'deck' ? 'bg-primary-600 text-white' : 'text-gray-400 hover:text-white'
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

      {/* Chat Panel */}
      <div className={clsx(
        'flex-1 min-w-0 flex flex-col bg-gray-900 rounded-lg',
        mobileTab !== 'chat' && 'hidden md:flex'
      )}>
        {/* Chat Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <h2 className="text-lg font-semibold text-white">Guided Build</h2>
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value)}
              className="px-2 py-1 text-xs font-medium rounded bg-gray-800 text-primary-400 border border-primary-600/30 focus:outline-none focus:border-primary-500 cursor-pointer"
            >
              {FORMAT_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          <button
            onClick={startNewConversation}
            className="text-sm text-gray-400 hover:text-white transition-colors"
          >
            Start Over
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {!hasMessages ? (
            <div className="flex flex-col items-center justify-center h-full text-center px-4">
              <h3 className="text-2xl font-bold text-white mb-2">
                What do you want to build?
              </h3>
              <p className="text-gray-400 mb-8 max-w-md">
                Tell me anything -- a card you love, colors you want to play,
                a strategy that sounds fun, or just let me pick something good.
                There's no wrong starting point.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
                {STARTER_PROMPTS.map((sp) => (
                  <button
                    key={sp.label}
                    onClick={() => handleStarterClick(sp.prompt, sp.incomplete)}
                    className="text-left px-4 py-3 rounded-lg bg-gray-800 hover:bg-gray-700 border border-gray-700 hover:border-gray-600 transition-colors"
                  >
                    <span className="text-sm text-gray-200">{sp.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message, index) => (
              <MessageBubble key={index} message={message} />
            ))
          )}

          {isLoading && (
            <div className="flex items-start space-x-3 p-3 bg-gray-800/50 rounded-lg">
              <div className="flex space-x-1 pt-1">
                <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '100ms' }} />
                <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '200ms' }} />
              </div>
              <span className="text-sm text-primary-400 font-medium">
                {PROGRESS_MESSAGES[progressIdx]}
              </span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Quick Actions (when deck exists) */}
        {currentDeck && !isLoading && (
          <div className="px-4 py-2 border-t border-gray-700">
            <div className="flex flex-wrap gap-2">
              {suggestions.length > 0
                ? suggestions.map((s) => (
                    <button
                      key={s}
                      onClick={() => sendMessage(s)}
                      className="px-3 py-1 text-xs rounded-full bg-gray-800 hover:bg-gray-700 text-gray-300 transition-colors"
                    >
                      {s}
                    </button>
                  ))
                : ['Add more removal', 'Optimize the mana base', 'Build the sideboard', 'Make it faster'].map((s) => (
                    <button
                      key={s}
                      onClick={() => sendMessage(s)}
                      className="px-3 py-1 text-xs rounded-full bg-gray-800 hover:bg-gray-700 text-gray-300 transition-colors"
                    >
                      {s}
                    </button>
                  ))
              }
            </div>
          </div>
        )}

        {/* Input */}
        <form onSubmit={handleSubmit} className="p-4 border-t border-gray-700">
          <div className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={hasMessages ? 'Tell me what to change...' : 'Describe what you want to build...'}
              className="flex-1 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-primary-500"
              disabled={isLoading}
              autoFocus
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className={clsx(
                'px-4 py-2 rounded-lg font-medium transition-colors',
                input.trim() && !isLoading
                  ? 'bg-primary-600 hover:bg-primary-700 text-white'
                  : 'bg-gray-700 text-gray-400 cursor-not-allowed'
              )}
            >
              Send
            </button>
          </div>
        </form>
      </div>

      {/* Deck Panel */}
      <div className={clsx(
        'md:w-96 flex-shrink-0 flex flex-col gap-4 overflow-y-auto',
        mobileTab !== 'deck' && 'hidden md:flex'
      )}>
        {currentDeck ? (
          <>
            {/* Live Stats */}
            {deckStats && <DeckStatsPanel stats={deckStats} />}

            {/* Deck List */}
            <DeckList
              mainDeck={currentDeck.main_deck || []}
              sideboard={currentDeck.sideboard || []}
              commander={currentDeck.commander}
              format={currentDeck.format}
              title={currentDeck.name || 'Current Deck'}
              cardExplanations={currentDeck.card_explanations}
              onQuantityChange={(cardName, qty, target) => updateCardQuantity(cardName, qty, target)}
              onAddCard={(cardName, target) => addCard({ card_name: cardName, quantity: 1 }, target)}
              editable
              className="flex-1 overflow-y-auto"
            />

            {isAuthenticated && <DeckActions deck={currentDeck} />}
          </>
        ) : (
          <div className="bg-gray-900 rounded-lg p-6 text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-800 flex items-center justify-center">
              <svg className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </div>
            <p className="text-lg text-white mb-2">Your deck will appear here</p>
            <p className="text-sm text-gray-500">
              As you chat, the AI will build your deck and you'll see it take shape in real time.
              Mana curve, color breakdown, and card counts update live.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// --- Deck Stats Panel ---

function DeckStatsPanel({ stats }: { stats: DeckStats }) {
  const curveMax = Math.max(...Object.values(stats.mana_curve), 1);
  const totalPips = Object.values(stats.colors).reduce((a, b) => a + b, 0) || 1;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 space-y-4">
      {/* Card counts */}
      <div className="flex justify-between text-sm">
        <div>
          <span className="text-gray-400">Main: </span>
          <span className={clsx(
            'font-medium',
            stats.main_deck_count === stats.target_main ? 'text-green-400' :
            stats.main_deck_count > stats.target_main ? 'text-red-400' : 'text-yellow-400'
          )}>
            {stats.main_deck_count}/{stats.target_main}
          </span>
        </div>
        <div>
          <span className="text-gray-400">Sideboard: </span>
          <span className={clsx(
            'font-medium',
            stats.sideboard_count === stats.target_sideboard ? 'text-green-400' :
            stats.sideboard_count > stats.target_sideboard ? 'text-red-400' : 'text-yellow-400'
          )}>
            {stats.sideboard_count}/{stats.target_sideboard}
          </span>
        </div>
      </div>

      {/* Type breakdown */}
      <div className="flex gap-4 text-xs text-gray-400">
        <span>Creatures: {stats.creature_count}</span>
        <span>Spells: {stats.spell_count}</span>
        <span>Lands: {stats.land_count}</span>
      </div>

      {/* Mana curve */}
      <div>
        <p className="text-xs text-gray-500 mb-1">Mana Curve</p>
        <div className="flex items-end gap-1 h-12">
          {['0', '1', '2', '3', '4', '5', '6'].map((cmc) => {
            const count = stats.mana_curve[cmc] || 0;
            const height = count > 0 ? Math.max((count / curveMax) * 100, 8) : 0;
            return (
              <div key={cmc} className="flex-1 flex flex-col items-center">
                <div
                  className="w-full bg-primary-500/60 rounded-t transition-all"
                  style={{ height: `${height}%` }}
                  title={`${cmc === '6' ? '6+' : cmc} CMC: ${count} cards`}
                />
                <span className="text-[10px] text-gray-500 mt-0.5">
                  {cmc === '6' ? '6+' : cmc}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Color distribution */}
      {Object.keys(stats.colors).length > 0 && (
        <div>
          <p className="text-xs text-gray-500 mb-1">Colors</p>
          <div className="flex gap-1 h-3 rounded-full overflow-hidden">
            {Object.entries(stats.colors).map(([color, count]) => (
              <div
                key={color}
                className={clsx(COLOR_INFO[color]?.bg || 'bg-gray-600', 'transition-all')}
                style={{ width: `${(count / totalPips) * 100}%` }}
                title={`${COLOR_INFO[color]?.label || color}: ${count} pips`}
              />
            ))}
          </div>
          <div className="flex gap-2 mt-1">
            {Object.entries(stats.colors).map(([color, count]) => (
              <span key={color} className="text-[10px] text-gray-500">
                {COLOR_INFO[color]?.label || color}: {count}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Issues & Suggestions */}
      {stats.issues.length > 0 && (
        <div className="space-y-1">
          {stats.issues.map((issue, i) => (
            <p key={i} className="text-xs text-red-400">{issue}</p>
          ))}
        </div>
      )}
      {stats.suggestions.length > 0 && (
        <div className="space-y-1">
          {stats.suggestions.map((s, i) => (
            <p key={i} className="text-xs text-yellow-400/80">{s}</p>
          ))}
        </div>
      )}
    </div>
  );
}

// --- Message Bubble ---

const cardLinePattern = /^(\d+)x?\s+(.+)$/;

function parseCardNames(text: string): React.ReactNode {
  const bracketPattern = /\[\[([^\]]+)\]\]/g;
  const deckListMatch = text.match(cardLinePattern);
  if (deckListMatch) {
    const quantity = deckListMatch[1];
    const cardName = deckListMatch[2].trim();
    return (
      <>
        {quantity}{' '}
        <CardTooltip cardName={cardName}>{cardName}</CardTooltip>
      </>
    );
  }
  const parts = text.split(bracketPattern);
  if (parts.length === 1) return text;
  return parts.map((part, index) => {
    if (index % 2 === 1) {
      return <CardTooltip key={index} cardName={part}>{part}</CardTooltip>;
    }
    return part;
  });
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';

  const markdownComponents: Components = useMemo(() => ({
    p: ({ children }) => <p>{children}</p>,
    li: ({ children }) => {
      if (typeof children === 'string') return <li>{parseCardNames(children)}</li>;
      if (Array.isArray(children)) {
        return <li>{children.map((c) => typeof c === 'string' ? parseCardNames(c) : c)}</li>;
      }
      return <li>{children}</li>;
    },
    strong: ({ children }) => {
      if (typeof children === 'string') {
        return <strong><CardTooltip cardName={children}>{children}</CardTooltip></strong>;
      }
      return <strong>{children}</strong>;
    },
  }), []);

  return (
    <div className={clsx('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div className={clsx(
        'max-w-[85%] rounded-lg px-4 py-2',
        isUser ? 'bg-primary-600 text-white' : 'bg-gray-800 text-gray-200'
      )}>
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown components={markdownComponents}>{message.content}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
