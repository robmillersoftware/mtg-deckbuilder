import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useOnboardingStore } from '@/store/onboarding';
import { useAuth } from '@/hooks/useAuth';
import clsx from 'clsx';

export function OnboardingChecklist() {
  const { isAuthenticated } = useAuth();
  const { checklistItems, hasCompletedTour, startTour } = useOnboardingStore();
  const [isExpanded, setIsExpanded] = useState(true);
  const [isDismissed, setIsDismissed] = useState(false);

  const completedCount = checklistItems.filter((item) => item.completed).length;
  const totalCount = checklistItems.length;
  const progressPercent = (completedCount / totalCount) * 100;
  const allCompleted = completedCount === totalCount;

  // Don't show if dismissed or all items completed
  if (isDismissed || allCompleted) return null;

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
      {/* Header */}
      <div
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-850 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-primary-900/50 flex items-center justify-center">
            <svg className="w-5 h-5 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
          </div>
          <div>
            <h3 className="text-white font-semibold">Getting Started</h3>
            <p className="text-sm text-gray-400">{completedCount} of {totalCount} complete</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {/* Progress ring */}
          <div className="relative w-10 h-10">
            <svg className="w-10 h-10 transform -rotate-90">
              <circle
                cx="20"
                cy="20"
                r="16"
                stroke="currentColor"
                strokeWidth="3"
                fill="none"
                className="text-gray-700"
              />
              <circle
                cx="20"
                cy="20"
                r="16"
                stroke="currentColor"
                strokeWidth="3"
                fill="none"
                strokeDasharray={`${progressPercent} 100`}
                strokeLinecap="round"
                className="text-primary-500 transition-all duration-500"
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
              {Math.round(progressPercent)}%
            </span>
          </div>
          <svg
            className={clsx(
              'w-5 h-5 text-gray-400 transition-transform',
              isExpanded && 'rotate-180'
            )}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-3">
          {/* Tour prompt if not completed */}
          {!hasCompletedTour && (
            <button
              onClick={startTour}
              className="w-full flex items-center gap-3 p-3 bg-primary-900/30 border border-primary-800/50 rounded-lg hover:bg-primary-900/50 transition-colors text-left"
            >
              <div className="w-8 h-8 rounded-full bg-primary-600 flex items-center justify-center flex-shrink-0">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="flex-1">
                <p className="text-white font-medium text-sm">Take the tour</p>
                <p className="text-primary-300 text-xs">Learn how Spellbook works</p>
              </div>
              <svg className="w-5 h-5 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          )}

          {/* Checklist items */}
          {checklistItems.map((item) => (
            <ChecklistItem
              key={item.id}
              item={item}
              disabled={item.id !== 'create-deck' && item.id !== 'explore-meta' && !isAuthenticated}
            />
          ))}

          {/* Dismiss button */}
          <button
            onClick={() => setIsDismissed(true)}
            className="w-full text-center text-sm text-gray-500 hover:text-gray-400 py-2 transition-colors"
          >
            Dismiss checklist
          </button>
        </div>
      )}
    </div>
  );
}

interface ChecklistItemProps {
  item: {
    id: string;
    title: string;
    description: string;
    completed: boolean;
    link?: string;
  };
  disabled?: boolean;
}

function ChecklistItem({ item, disabled }: ChecklistItemProps) {
  const content = (
    <div
      className={clsx(
        'flex items-center gap-3 p-3 rounded-lg transition-colors',
        item.completed
          ? 'bg-gray-800/50'
          : disabled
          ? 'bg-gray-800/30 opacity-50 cursor-not-allowed'
          : 'bg-gray-800 hover:bg-gray-750 cursor-pointer'
      )}
    >
      {/* Checkbox */}
      <div
        className={clsx(
          'w-6 h-6 rounded-full border-2 flex items-center justify-center flex-shrink-0 transition-colors',
          item.completed
            ? 'bg-green-600 border-green-600'
            : 'border-gray-600'
        )}
      >
        {item.completed && (
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        )}
      </div>

      {/* Text */}
      <div className="flex-1 min-w-0">
        <p className={clsx(
          'text-sm font-medium',
          item.completed ? 'text-gray-500 line-through' : 'text-white'
        )}>
          {item.title}
        </p>
        <p className="text-xs text-gray-500 truncate">{item.description}</p>
      </div>

      {/* Arrow */}
      {!item.completed && !disabled && (
        <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      )}

      {/* Lock icon for disabled items */}
      {disabled && !item.completed && (
        <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
      )}
    </div>
  );

  if (item.link && !item.completed && !disabled) {
    return <Link to={item.link}>{content}</Link>;
  }

  return content;
}
