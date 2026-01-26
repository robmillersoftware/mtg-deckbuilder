import { useEffect, useState } from 'react';
import { useOnboardingStore } from '@/store/onboarding';
import { useAuth } from '@/hooks/useAuth';

interface WelcomeModalProps {
  onStartTour?: () => void;
}

export function WelcomeModal({ onStartTour }: WelcomeModalProps) {
  const { isAuthenticated } = useAuth();
  const { hasSeenWelcome, setHasSeenWelcome, startTour } = useOnboardingStore();
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Show welcome modal for first-time visitors (both authenticated and anonymous)
    if (!hasSeenWelcome) {
      // Small delay for smoother appearance
      const timer = setTimeout(() => setIsVisible(true), 500);
      return () => clearTimeout(timer);
    }
  }, [hasSeenWelcome]);

  const handleClose = () => {
    setIsVisible(false);
    setHasSeenWelcome(true);
  };

  const handleStartTour = () => {
    setIsVisible(false);
    setHasSeenWelcome(true);
    startTour();
    onStartTour?.();
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div className="bg-gray-900 rounded-2xl max-w-lg w-full shadow-2xl border border-gray-700 overflow-hidden animate-in fade-in zoom-in-95 duration-300">
        {/* Header with gradient */}
        <div className="bg-gradient-to-r from-primary-600 to-purple-600 p-6 text-center">
          <span className="text-5xl mb-4 block">🔮</span>
          <h2 className="text-2xl font-bold text-white">Welcome to Spellbook</h2>
          <p className="text-primary-100 mt-2">Your AI-powered MTG deck builder</p>
        </div>

        {/* Features */}
        <div className="p-6 space-y-4">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-lg bg-primary-900/50 flex items-center justify-center flex-shrink-0">
              <svg className="w-5 h-5 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <div>
              <h3 className="text-white font-semibold">Build with AI</h3>
              <p className="text-gray-400 text-sm">Describe the deck you want in natural language and let AI craft it for you</p>
            </div>
          </div>

          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-lg bg-primary-900/50 flex items-center justify-center flex-shrink-0">
              <svg className="w-5 h-5 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <h3 className="text-white font-semibold">Explore the Meta</h3>
              <p className="text-gray-400 text-sm">Browse top-performing decks and discover winning strategies</p>
            </div>
          </div>

          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-lg bg-primary-900/50 flex items-center justify-center flex-shrink-0">
              <svg className="w-5 h-5 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h3 className="text-white font-semibold">Simulate Draws</h3>
              <p className="text-gray-400 text-sm">Test opening hands and practice mulligan decisions</p>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="p-6 pt-2 flex flex-col sm:flex-row gap-3">
          <button
            onClick={handleStartTour}
            className="flex-1 px-4 py-3 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
          >
            Take a Quick Tour
          </button>
          <button
            onClick={handleClose}
            className="flex-1 px-4 py-3 bg-gray-800 hover:bg-gray-700 text-gray-300 font-medium rounded-lg transition-colors"
          >
            {isAuthenticated ? 'Start Building' : 'Explore First'}
          </button>
        </div>

        {/* Sign up prompt for anonymous users */}
        {!isAuthenticated && (
          <div className="px-6 pb-6 pt-0">
            <p className="text-center text-sm text-gray-500">
              <a href="/register" className="text-primary-400 hover:text-primary-300">Create an account</a>
              {' '}to save your decks and access all features
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
