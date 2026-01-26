import { useState, useRef, useEffect, useMemo } from 'react';
import { useChat } from '@/hooks/useChat';
import { Message } from '@/types';
import ReactMarkdown, { Components } from 'react-markdown';
import clsx from 'clsx';
import { CardTooltip } from './CardTooltip';

// Progress messages that rotate during deck generation
const PROGRESS_MESSAGES = [
  'Analyzing your request...',
  'Searching for synergies...',
  'Building the mana base...',
  'Selecting sideboard cards...',
  'Optimizing the curve...',
  'Finding key interactions...',
  'Validating deck...',
];

interface ChatWindowProps {
  className?: string;
}

// Format options
const FORMAT_OPTIONS = [
  { value: 'standard', label: 'Standard' },
  { value: 'historic', label: 'Historic' },
  { value: 'modern', label: 'Modern' },
  { value: 'legacy', label: 'Legacy' },
  { value: 'cedh', label: 'cEDH' },
];

export function ChatWindow({ className }: ChatWindowProps) {
  const [input, setInput] = useState('');
  const [progressMessageIndex, setProgressMessageIndex] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const {
    messages,
    isLoading,
    suggestions,
    format,
    setFormat,
    sendMessage,
    startNewConversation,
  } = useChat();

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Rotate progress messages during loading
  useEffect(() => {
    if (!isLoading) {
      setProgressMessageIndex(0);
      return;
    }

    const interval = setInterval(() => {
      setProgressMessageIndex((prev) => (prev + 1) % PROGRESS_MESSAGES.length);
    }, 2500);

    return () => clearInterval(interval);
  }, [isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      sendMessage(input.trim());
      setInput('');
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    sendMessage(suggestion);
  };

  return (
    <div className={clsx('flex flex-col h-full bg-gray-900', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
        <div className="flex items-center space-x-3">
          <h2 className="text-lg font-semibold text-white">Chat</h2>
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value)}
            className="px-2 py-1 text-xs font-medium rounded bg-gray-800 text-primary-400 border border-primary-600/30 focus:outline-none focus:border-primary-500 cursor-pointer"
          >
            {FORMAT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <button
          onClick={startNewConversation}
          className="text-sm text-gray-400 hover:text-white transition-colors"
        >
          New Conversation
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            <p className="text-lg mb-2">Welcome to Spellbook!</p>
            <p className="text-sm">
              Tell me what kind of MTG deck you want to build.
            </p>
            <div className="mt-4 space-y-2">
              <p className="text-xs text-gray-500">Try asking:</p>
              {[
                'Build me a mono-red aggro deck',
                'Create a blue-white control deck',
                'I want a deck with lots of creatures',
              ].map((example) => (
                <button
                  key={example}
                  onClick={() => handleSuggestionClick(example)}
                  className="block w-full text-left px-3 py-2 rounded bg-gray-800 hover:bg-gray-700 text-sm text-gray-300 transition-colors"
                >
                  "{example}"
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
            <div className="flex-1">
              <span className="text-sm text-primary-400 font-medium transition-opacity duration-300">
                {PROGRESS_MESSAGES[progressMessageIndex]}
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggestions */}
      {suggestions.length > 0 && !isLoading && (
        <div className="px-4 py-2 border-t border-gray-700">
          <div className="flex flex-wrap gap-2">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => handleSuggestionClick(suggestion)}
                className="px-3 py-1 text-sm rounded-full bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
              >
                {suggestion}
              </button>
            ))}
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
            placeholder="Describe the deck you want to build..."
            className="flex-1 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-primary-500"
            disabled={isLoading}
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
  );
}

interface MessageBubbleProps {
  message: Message;
}

// Pattern to match deck list lines: "4 Card Name" or "4x Card Name"
const cardLinePattern = /^(\d+)x?\s+(.+)$/;

// Helper to parse card names from text and wrap them in tooltips
function parseCardNames(text: string): React.ReactNode {
  // Match card references like "[[Card Name]]" or deck list format "4 Card Name"
  const bracketPattern = /\[\[([^\]]+)\]\]/g;

  // Check if this is a deck list line
  const deckListMatch = text.match(cardLinePattern);
  if (deckListMatch) {
    const quantity = deckListMatch[1];
    const cardName = deckListMatch[2].trim();
    return (
      <>
        {quantity}{' '}
        <CardTooltip cardName={cardName}>
          {cardName}
        </CardTooltip>
      </>
    );
  }

  // Check for [[Card Name]] syntax
  const parts = text.split(bracketPattern);
  if (parts.length === 1) {
    return text;
  }

  return parts.map((part, index) => {
    // Odd indices are card names (captured groups)
    if (index % 2 === 1) {
      return (
        <CardTooltip key={index} cardName={part}>
          {part}
        </CardTooltip>
      );
    }
    return part;
  });
}

function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  // Custom markdown components to handle card names
  const markdownComponents: Components = useMemo(() => ({
    // Override paragraph rendering to detect card names
    p: ({ children }) => {
      return <p>{children}</p>;
    },
    // Override list item to detect deck list entries
    li: ({ children }) => {
      // Check if children is a string that looks like a deck entry
      if (typeof children === 'string') {
        const parsed = parseCardNames(children);
        return <li>{parsed}</li>;
      }
      // If children contain text nodes, try to parse them
      if (Array.isArray(children)) {
        const processedChildren = children.map((child) => {
          if (typeof child === 'string') {
            return parseCardNames(child);
          }
          return child;
        });
        return <li>{processedChildren}</li>;
      }
      return <li>{children}</li>;
    },
    // Handle strong/bold text that might be card names
    strong: ({ children }) => {
      if (typeof children === 'string') {
        // Bold card names should have tooltips
        return (
          <strong>
            <CardTooltip cardName={children}>
              {children}
            </CardTooltip>
          </strong>
        );
      }
      return <strong>{children}</strong>;
    },
  }), []);

  return (
    <div
      className={clsx(
        'flex',
        isUser ? 'justify-end' : 'justify-start'
      )}
    >
      <div
        className={clsx(
          'max-w-[80%] rounded-lg px-4 py-2',
          isUser
            ? 'bg-primary-600 text-white'
            : 'bg-gray-800 text-gray-200'
        )}
      >
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
