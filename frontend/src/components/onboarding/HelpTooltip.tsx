import { useState, useRef, useEffect } from 'react';
import clsx from 'clsx';

interface HelpTooltipProps {
  content: string;
  title?: string;
  placement?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
  iconClassName?: string;
}

export function HelpTooltip({
  content,
  title,
  placement = 'top',
  className,
  iconClassName,
}: HelpTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isVisible && triggerRef.current && tooltipRef.current) {
      const triggerRect = triggerRef.current.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();
      const spacing = 8;

      let top = 0;
      let left = 0;

      switch (placement) {
        case 'top':
          top = triggerRect.top - tooltipRect.height - spacing;
          left = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
          break;
        case 'bottom':
          top = triggerRect.bottom + spacing;
          left = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
          break;
        case 'left':
          top = triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2;
          left = triggerRect.left - tooltipRect.width - spacing;
          break;
        case 'right':
          top = triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2;
          left = triggerRect.right + spacing;
          break;
      }

      // Keep within viewport
      left = Math.max(8, Math.min(left, window.innerWidth - tooltipRect.width - 8));
      top = Math.max(8, Math.min(top, window.innerHeight - tooltipRect.height - 8));

      setPosition({ top, left });
    }
  }, [isVisible, placement]);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className={clsx(
          'inline-flex items-center justify-center w-5 h-5 rounded-full text-gray-500 hover:text-gray-300 hover:bg-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-gray-900',
          className
        )}
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        aria-label="Help"
      >
        <svg className={clsx('w-4 h-4', iconClassName)} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </button>

      {isVisible && (
        <div
          ref={tooltipRef}
          className="fixed z-50 max-w-xs bg-gray-800 rounded-lg shadow-xl border border-gray-700 p-3 animate-in fade-in zoom-in-95 duration-150"
          style={position ? { top: position.top, left: position.left } : { visibility: 'hidden' }}
        >
          {title && (
            <p className="text-white font-medium text-sm mb-1">{title}</p>
          )}
          <p className="text-gray-400 text-sm leading-relaxed">{content}</p>
          {/* Arrow indicator based on placement */}
          <div
            className={clsx(
              'absolute w-2 h-2 bg-gray-800 border-gray-700 transform rotate-45',
              placement === 'top' && 'bottom-[-5px] left-1/2 -translate-x-1/2 border-r border-b',
              placement === 'bottom' && 'top-[-5px] left-1/2 -translate-x-1/2 border-l border-t',
              placement === 'left' && 'right-[-5px] top-1/2 -translate-y-1/2 border-t border-r',
              placement === 'right' && 'left-[-5px] top-1/2 -translate-y-1/2 border-b border-l'
            )}
          />
        </div>
      )}
    </>
  );
}
