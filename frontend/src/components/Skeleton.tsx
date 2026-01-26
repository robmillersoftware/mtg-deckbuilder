import clsx from 'clsx';

interface SkeletonProps {
  className?: string;
}

/**
 * Base skeleton element with animation
 */
export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={clsx(
        'animate-pulse bg-gray-700 rounded',
        className
      )}
    />
  );
}

interface SkeletonTextProps {
  lines?: number;
  className?: string;
}

/**
 * Skeleton for text content
 */
export function SkeletonText({ lines = 3, className }: SkeletonTextProps) {
  return (
    <div className={clsx('space-y-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={clsx(
            'h-4',
            // Make last line shorter for natural look
            i === lines - 1 ? 'w-2/3' : 'w-full'
          )}
        />
      ))}
    </div>
  );
}

/**
 * Skeleton for a card-like element
 */
export function SkeletonCard({ className }: SkeletonProps) {
  return (
    <div className={clsx('bg-gray-800 rounded-lg p-4 space-y-3', className)}>
      <Skeleton className="h-5 w-1/3" />
      <SkeletonText lines={2} />
    </div>
  );
}

/**
 * Skeleton for deck list entries
 */
export function SkeletonDeckList({ className }: SkeletonProps) {
  return (
    <div className={clsx('space-y-2', className)}>
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="flex items-center space-x-3">
          <Skeleton className="h-4 w-6" />
          <Skeleton className="h-4 flex-1" />
        </div>
      ))}
    </div>
  );
}

/**
 * Skeleton for simulation result summary
 */
export function SkeletonSimulationResult({ className }: SkeletonProps) {
  return (
    <div className={clsx('space-y-6', className)}>
      {/* Summary header */}
      <div className="bg-gray-800 rounded-lg p-6 space-y-4">
        <div className="flex justify-between items-center">
          <Skeleton className="h-6 w-1/2" />
          <Skeleton className="h-4 w-24" />
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="bg-gray-700 rounded-lg p-4 space-y-2">
              <Skeleton className="h-3 w-16" />
              <Skeleton className="h-8 w-12" />
            </div>
          ))}
        </div>

        {/* Key cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Array.from({ length: 2 }).map((_, i) => (
            <div key={i} className="space-y-2">
              <Skeleton className="h-4 w-24" />
              {Array.from({ length: 4 }).map((_, j) => (
                <div key={j} className="flex justify-between items-center bg-gray-700 rounded px-3 py-2">
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-3 w-8" />
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Game details skeleton */}
      <div className="bg-gray-800 rounded-lg p-6 space-y-3">
        <Skeleton className="h-5 w-32" />
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="bg-gray-700 rounded-lg px-4 py-3 flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Skeleton className="h-4 w-16" />
              <Skeleton className="h-4 w-12" />
              <Skeleton className="h-4 w-16" />
            </div>
            <Skeleton className="h-5 w-5" />
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Animated simulation loading state with card dealing effect
 */
export function SimulationLoadingState({
  gamesCompleted,
  totalGames,
  deckName,
  opponentName,
}: {
  gamesCompleted: number;
  totalGames: number;
  deckName: string;
  opponentName: string;
}) {
  const progress = (gamesCompleted / totalGames) * 100;

  return (
    <div className="bg-gray-800 rounded-lg p-8">
      <div className="text-center">
        {/* Animated cards */}
        <div className="flex justify-center items-center space-x-4 mb-6">
          <div className="relative">
            {/* Stack of cards with animation */}
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="absolute w-12 h-16 bg-gradient-to-br from-indigo-600 to-indigo-800 rounded-lg border border-indigo-400/30"
                style={{
                  transform: `translateX(${i * 4}px) translateY(${i * 2}px) rotate(${i * 3}deg)`,
                  animation: `cardShuffle 1.5s ease-in-out infinite`,
                  animationDelay: `${i * 0.2}s`,
                  zIndex: 3 - i,
                }}
              />
            ))}
            {/* Visible card placeholder for positioning */}
            <div className="w-12 h-16 opacity-0" />
          </div>

          <div className="text-2xl text-gray-400 animate-pulse">vs</div>

          <div className="relative">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="absolute w-12 h-16 bg-gradient-to-br from-red-600 to-red-800 rounded-lg border border-red-400/30"
                style={{
                  transform: `translateX(${-i * 4}px) translateY(${i * 2}px) rotate(${-i * 3}deg)`,
                  animation: `cardShuffle 1.5s ease-in-out infinite`,
                  animationDelay: `${i * 0.2 + 0.1}s`,
                  zIndex: 3 - i,
                }}
              />
            ))}
            <div className="w-12 h-16 opacity-0" />
          </div>
        </div>

        <h3 className="text-white font-semibold mb-2">Simulating Games...</h3>
        <p className="text-gray-400 text-sm mb-1">{deckName}</p>
        <p className="text-gray-500 text-sm mb-4">vs {opponentName}</p>

        {/* Progress */}
        <div className="max-w-xs mx-auto">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-gray-400">Game {gamesCompleted + 1} of {totalGames}</span>
            <span className="text-indigo-400 font-medium">{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
            <div
              className="bg-indigo-500 h-2.5 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      {/* CSS for card animation */}
      <style>{`
        @keyframes cardShuffle {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-8px);
          }
        }
      `}</style>
    </div>
  );
}
