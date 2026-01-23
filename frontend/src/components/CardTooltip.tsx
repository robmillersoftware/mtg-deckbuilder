import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { useCardByName } from '@/hooks/useCards';
import clsx from 'clsx';

interface CardTooltipProps {
  cardName: string;
  children: React.ReactNode;
  className?: string;
  explanation?: string;
}

export function CardTooltip({ cardName, children, className, explanation }: CardTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const triggerRef = useRef<HTMLSpanElement>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>();

  // For DFCs, use only the front face name for API lookup
  const lookupName = cardName.includes(' // ') ? cardName.split(' // ')[0] : cardName;
  const { data: card, isLoading } = useCardByName(isVisible ? lookupName : undefined);

  const handleMouseEnter = (e: React.MouseEvent) => {
    const rect = (e.target as HTMLElement).getBoundingClientRect();

    // Position tooltip to the right of the element
    let x = rect.right + 10;
    let y = rect.top;

    // Check if tooltip would go off screen
    const tooltipWidth = 250;
    const tooltipHeight = 350;

    if (x + tooltipWidth > window.innerWidth) {
      x = rect.left - tooltipWidth - 10;
    }

    if (y + tooltipHeight > window.innerHeight) {
      y = window.innerHeight - tooltipHeight - 10;
    }

    setPosition({ x, y });

    // Delay showing tooltip (200ms per spec requirement)
    timeoutRef.current = setTimeout(() => {
      setIsVisible(true);
    }, 200);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <>
      <span
        ref={triggerRef}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        className={clsx('cursor-help underline decoration-dotted', className)}
      >
        {children}
      </span>

      {isVisible &&
        createPortal(
          <div
            className="fixed z-50 pointer-events-none"
            style={{ left: position.x, top: position.y }}
          >
            <CardPreview card={card} isLoading={isLoading} explanation={explanation} />
          </div>,
          document.body
        )}
    </>
  );
}

interface CardPreviewProps {
  card?: {
    name: string;
    mana_cost?: string;
    type_line?: string;
    oracle_text?: string;
    power?: string;
    toughness?: string;
    image_uri?: string;
    image_uri_small?: string;
    rarity?: string;
    set_name?: string;
    price_usd?: number;
  };
  isLoading: boolean;
  explanation?: string;
}

function CardPreview({ card, isLoading, explanation }: CardPreviewProps) {
  if (isLoading) {
    return (
      <div className="w-[250px] h-[350px] bg-gray-800 rounded-lg animate-pulse" />
    );
  }

  if (!card) {
    return (
      <div className="w-[250px] p-4 bg-gray-800 rounded-lg text-gray-400 text-sm">
        Card not found
      </div>
    );
  }

  return (
    <div className="w-[250px] bg-gray-800 rounded-lg overflow-hidden shadow-xl border border-gray-700">
      {card.image_uri ? (
        <img
          src={card.image_uri}
          alt={card.name}
          className="w-full"
          loading="lazy"
        />
      ) : (
        <div className="p-4">
          <h3 className="font-bold text-white">{card.name}</h3>
          {card.mana_cost && (
            <p className="text-sm text-gray-400 mt-1">{card.mana_cost}</p>
          )}
          {card.type_line && (
            <p className="text-sm text-gray-300 mt-2">{card.type_line}</p>
          )}
          {card.oracle_text && (
            <p className="text-sm text-gray-400 mt-2 whitespace-pre-line">
              {card.oracle_text}
            </p>
          )}
          {card.power && card.toughness && (
            <p className="text-sm text-gray-300 mt-2">
              {card.power}/{card.toughness}
            </p>
          )}
        </div>
      )}

      {/* Context-aware explanation */}
      {explanation && (
        <div className="px-3 py-2 bg-primary-900/50 border-t border-primary-700">
          <p className="text-xs text-primary-200 italic leading-relaxed">
            {explanation}
          </p>
        </div>
      )}

      {/* Footer with price */}
      <div className="px-3 py-2 bg-gray-900 text-xs text-gray-400 flex justify-between">
        <span>{card.set_name}</span>
        {card.price_usd !== undefined && card.price_usd > 0 && (
          <span>${card.price_usd.toFixed(2)}</span>
        )}
      </div>
    </div>
  );
}

// HOC for wrapping card names with tooltips
export function withCardTooltip(text: string): React.ReactNode {
  // Pattern to match card names (simplified - in production would use card database)
  const cardNamePattern = /\[\[([^\]]+)\]\]/g;

  const parts = text.split(cardNamePattern);

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
