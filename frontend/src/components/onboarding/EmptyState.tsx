import { Link } from 'react-router-dom';
import clsx from 'clsx';

type EmptyStateVariant = 'decks' | 'conversations' | 'search' | 'generic';

interface EmptyStateProps {
  variant?: EmptyStateVariant;
  title?: string;
  description?: string;
  actionText?: string;
  actionLink?: string;
  onAction?: () => void;
  showTips?: boolean;
  className?: string;
}

const VARIANT_CONFIG: Record<EmptyStateVariant, {
  icon: React.ReactNode;
  title: string;
  description: string;
  actionText: string;
  actionLink: string;
  tips: string[];
}> = {
  decks: {
    icon: (
      <svg className="w-16 h-16 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
      </svg>
    ),
    title: 'No decks yet',
    description: 'Your deck collection is empty. Build your first deck using AI or import an existing one.',
    actionText: 'Build Your First Deck',
    actionLink: '/',
    tips: [
      'Try describing a playstyle: "aggressive red deck" or "control with counterspells"',
      'Specify a format like Standard, Modern, or cEDH for better results',
      'You can also import existing decklists from text or MTG Arena',
    ],
  },
  conversations: {
    icon: (
      <svg className="w-16 h-16 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
      </svg>
    ),
    title: 'No conversations yet',
    description: 'Start a conversation with Spellbook to build decks and get recommendations.',
    actionText: 'Start a Conversation',
    actionLink: '/',
    tips: [
      'Ask for deck suggestions based on your favorite colors or playstyle',
      'Request improvements to your current deck',
      'Get advice on sideboard choices for specific matchups',
    ],
  },
  search: {
    icon: (
      <svg className="w-16 h-16 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
    title: 'No results found',
    description: 'We couldn\'t find anything matching your search. Try adjusting your filters or search terms.',
    actionText: 'Clear Filters',
    actionLink: '#',
    tips: [
      'Try using broader search terms',
      'Check for typos in card or deck names',
      'Remove some filters to see more results',
    ],
  },
  generic: {
    icon: (
      <svg className="w-16 h-16 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
      </svg>
    ),
    title: 'Nothing here yet',
    description: 'This section is empty. Get started by taking an action.',
    actionText: 'Get Started',
    actionLink: '/',
    tips: [],
  },
};

export function EmptyState({
  variant = 'generic',
  title,
  description,
  actionText,
  actionLink,
  onAction,
  showTips = true,
  className,
}: EmptyStateProps) {
  const config = VARIANT_CONFIG[variant];

  const displayTitle = title || config.title;
  const displayDescription = description || config.description;
  const displayActionText = actionText || config.actionText;
  const displayActionLink = actionLink || config.actionLink;
  const tips = config.tips;

  const ActionButton = () => (
    <button
      onClick={onAction}
      className="inline-flex items-center px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
    >
      {displayActionText}
      <svg className="ml-2 w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
      </svg>
    </button>
  );

  return (
    <div className={clsx('text-center py-12 px-4', className)}>
      {/* Icon */}
      <div className="flex justify-center mb-6">
        <div className="p-4 bg-gray-800/50 rounded-full">
          {config.icon}
        </div>
      </div>

      {/* Title */}
      <h3 className="text-xl font-semibold text-white mb-2">{displayTitle}</h3>

      {/* Description */}
      <p className="text-gray-400 max-w-md mx-auto mb-6">{displayDescription}</p>

      {/* Action */}
      {onAction ? (
        <ActionButton />
      ) : (
        <Link
          to={displayActionLink}
          className="inline-flex items-center px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
        >
          {displayActionText}
          <svg className="ml-2 w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </Link>
      )}

      {/* Tips */}
      {showTips && tips.length > 0 && (
        <div className="mt-8 max-w-lg mx-auto">
          <div className="bg-gray-800/50 rounded-lg p-4 text-left">
            <div className="flex items-center gap-2 mb-3">
              <svg className="w-5 h-5 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              <span className="text-sm font-medium text-white">Tips to get started</span>
            </div>
            <ul className="space-y-2">
              {tips.map((tip, index) => (
                <li key={index} className="flex items-start gap-2 text-sm text-gray-400">
                  <span className="text-primary-400 mt-0.5">•</span>
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
