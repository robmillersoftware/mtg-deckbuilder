import { useEffect, useState, useCallback } from 'react';
import { useOnboardingStore } from '@/store/onboarding';
import { useAuth } from '@/hooks/useAuth';
import clsx from 'clsx';

interface TourStep {
  target: string;
  title: string;
  content: string;
  placement: 'top' | 'bottom' | 'left' | 'right';
  authRequired?: boolean;
}

const TOUR_STEPS: TourStep[] = [
  {
    target: '[data-tour="build"]',
    title: 'Build Decks',
    content: 'Start here! Describe the deck you want and our AI will create it for you. Try "Build me a mono-red aggro deck".',
    placement: 'bottom',
  },
  {
    target: '[data-tour="my-decks"]',
    title: 'My Decks',
    content: 'All your saved decks appear here. Edit, export, or share them with friends.',
    placement: 'bottom',
    authRequired: true,
  },
  {
    target: '[data-tour="simulate"]',
    title: 'Game Simulator',
    content: 'Test your deck against meta archetypes! AI simulates full games and provides matchup analysis, sideboard guides, and strategic advice.',
    placement: 'bottom',
    authRequired: true,
  },
  {
    target: '[data-tour="import"]',
    title: 'Import Decks',
    content: 'Already have a deck? Paste a decklist or import from MTG Arena to bring it into Spellbook.',
    placement: 'bottom',
    authRequired: true,
  },
  {
    target: '[data-tour="meta"]',
    title: 'Meta Analysis',
    content: 'Explore the competitive landscape. See top decks, popular archetypes, and format statistics.',
    placement: 'bottom',
  },
  {
    target: '[data-tour="format-selector"]',
    title: 'Choose Your Format',
    content: 'Select your preferred format before building. We support Standard, Historic, Modern, Legacy, and cEDH.',
    placement: 'bottom',
  },
];

export function FeatureTour() {
  const { isAuthenticated } = useAuth();
  const { isTourActive, currentTourStep, nextTourStep, prevTourStep, endTour } = useOnboardingStore();
  const [tooltipPosition, setTooltipPosition] = useState<{ top: number; left: number } | null>(null);
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);

  // Filter steps based on auth status
  const availableSteps = TOUR_STEPS.filter(
    (step) => !step.authRequired || isAuthenticated
  );

  const currentStep = availableSteps[currentTourStep];
  const isLastStep = currentTourStep >= availableSteps.length - 1;
  const isFirstStep = currentTourStep === 0;

  const calculatePosition = useCallback(() => {
    if (!currentStep) return;

    const targetElement = document.querySelector(currentStep.target);
    if (!targetElement) {
      // Element not found, try to skip to next valid step or end tour
      if (!isLastStep) {
        nextTourStep();
      } else {
        endTour();
      }
      return;
    }

    const rect = targetElement.getBoundingClientRect();
    setTargetRect(rect);

    const tooltipWidth = 320;
    const tooltipHeight = 180;
    const spacing = 12;

    let top = 0;
    let left = 0;

    switch (currentStep.placement) {
      case 'bottom':
        top = rect.bottom + spacing;
        left = rect.left + rect.width / 2 - tooltipWidth / 2;
        break;
      case 'top':
        top = rect.top - tooltipHeight - spacing;
        left = rect.left + rect.width / 2 - tooltipWidth / 2;
        break;
      case 'left':
        top = rect.top + rect.height / 2 - tooltipHeight / 2;
        left = rect.left - tooltipWidth - spacing;
        break;
      case 'right':
        top = rect.top + rect.height / 2 - tooltipHeight / 2;
        left = rect.right + spacing;
        break;
    }

    // Keep tooltip within viewport
    left = Math.max(16, Math.min(left, window.innerWidth - tooltipWidth - 16));
    top = Math.max(16, Math.min(top, window.innerHeight - tooltipHeight - 16));

    setTooltipPosition({ top, left });
  }, [currentStep, isLastStep, nextTourStep, endTour]);

  useEffect(() => {
    if (isTourActive) {
      calculatePosition();
      window.addEventListener('resize', calculatePosition);
      window.addEventListener('scroll', calculatePosition);
      return () => {
        window.removeEventListener('resize', calculatePosition);
        window.removeEventListener('scroll', calculatePosition);
      };
    }
  }, [isTourActive, currentTourStep, calculatePosition]);

  // Handle keyboard navigation
  useEffect(() => {
    if (!isTourActive) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        endTour();
      } else if (e.key === 'ArrowRight' || e.key === 'Enter') {
        if (isLastStep) {
          endTour();
        } else {
          nextTourStep();
        }
      } else if (e.key === 'ArrowLeft') {
        if (!isFirstStep) {
          prevTourStep();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isTourActive, isLastStep, isFirstStep, nextTourStep, prevTourStep, endTour]);

  if (!isTourActive || !currentStep || !tooltipPosition) return null;

  return (
    <>
      {/* Overlay */}
      <div className="fixed inset-0 z-40 pointer-events-none">
        {/* Spotlight effect around target */}
        {targetRect && (
          <div
            className="absolute transition-all duration-300"
            style={{
              top: targetRect.top - 4,
              left: targetRect.left - 4,
              width: targetRect.width + 8,
              height: targetRect.height + 8,
              boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.75)',
              borderRadius: '8px',
            }}
          />
        )}
      </div>

      {/* Target highlight ring */}
      {targetRect && (
        <div
          className="fixed z-40 pointer-events-none rounded-lg ring-2 ring-primary-500 ring-offset-2 ring-offset-transparent animate-pulse"
          style={{
            top: targetRect.top - 4,
            left: targetRect.left - 4,
            width: targetRect.width + 8,
            height: targetRect.height + 8,
          }}
        />
      )}

      {/* Tooltip */}
      <div
        className="fixed z-50 w-80 bg-gray-900 rounded-xl shadow-2xl border border-gray-700 overflow-hidden animate-in fade-in slide-in-from-bottom-2 duration-200"
        style={{
          top: tooltipPosition.top,
          left: tooltipPosition.left,
        }}
      >
        {/* Progress bar */}
        <div className="h-1 bg-gray-800">
          <div
            className="h-full bg-primary-500 transition-all duration-300"
            style={{ width: `${((currentTourStep + 1) / availableSteps.length) * 100}%` }}
          />
        </div>

        <div className="p-4">
          {/* Step indicator */}
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-500">
              Step {currentTourStep + 1} of {availableSteps.length}
            </span>
            <button
              onClick={endTour}
              className="text-gray-500 hover:text-gray-300 transition-colors"
              aria-label="Close tour"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Content */}
          <h3 className="text-white font-semibold text-lg mb-2">{currentStep.title}</h3>
          <p className="text-gray-400 text-sm mb-4">{currentStep.content}</p>

          {/* Navigation */}
          <div className="flex items-center justify-between">
            <button
              onClick={prevTourStep}
              disabled={isFirstStep}
              className={clsx(
                'px-3 py-1.5 text-sm font-medium rounded-lg transition-colors',
                isFirstStep
                  ? 'text-gray-600 cursor-not-allowed'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              )}
            >
              Back
            </button>
            <div className="flex gap-1">
              {availableSteps.map((_, index) => (
                <div
                  key={index}
                  className={clsx(
                    'w-2 h-2 rounded-full transition-colors',
                    index === currentTourStep ? 'bg-primary-500' : 'bg-gray-700'
                  )}
                />
              ))}
            </div>
            <button
              onClick={isLastStep ? endTour : nextTourStep}
              className="px-3 py-1.5 text-sm font-medium bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
            >
              {isLastStep ? 'Get Started' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
