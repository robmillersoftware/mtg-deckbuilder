import clsx from 'clsx';
import type { DeckFitResponse, IdentityOverrides } from '../types';

interface DeckFitPanelProps {
  result: DeckFitResponse;
  onOverridesChange: (overrides: IdentityOverrides) => void;
  isSaving: boolean;
}

const without = (xs: string[], x: string) => xs.filter((v) => v !== x);
const withOnce = (xs: string[], x: string) => (xs.includes(x) ? xs : [...xs, x]);

export default function DeckFitPanel({ result, onOverridesChange, isSaving }: DeckFitPanelProps) {
  const { identity, cards, flagged, available_tags } = result;
  if (!identity) {
    return (
      <div className="bg-gray-900 rounded-lg p-4 text-sm text-gray-400">
        Fit check unavailable for this deck.
      </div>
    );
  }
  const o = identity.overrides;

  const toggleTag = (tag: string) =>
    onOverridesChange(
      identity.tags.includes(tag)
        ? { ...o, tags_off: withOnce(o.tags_off, tag), tags_on: without(o.tags_on, tag) }
        : { ...o, tags_on: withOnce(o.tags_on, tag), tags_off: without(o.tags_off, tag) }
    );
  const unpin = (name: string) =>
    onOverridesChange({ ...o, unpinned: withOnce(o.unpinned, name), pinned: without(o.pinned, name) });
  const pin = (name: string) =>
    onOverridesChange({ ...o, pinned: withOnce(o.pinned, name), unpinned: without(o.unpinned, name) });

  const pinnable = Object.keys(cards).filter((n) => !identity.key_cards.includes(n)).sort();

  return (
    <div className={clsx('bg-gray-900 rounded-lg p-4 space-y-4', isSaving && 'opacity-60')}>
      <h3 className="text-lg font-semibold text-white">Deck Fit</h3>

      <div>
        <h4 className="text-xs text-gray-500 uppercase mb-1">Themes</h4>
        <div className="flex flex-wrap gap-1">
          {available_tags.map((tag) => (
            <button
              key={tag}
              disabled={isSaving}
              onClick={() => toggleTag(tag)}
              className={clsx(
                'text-xs px-2 py-0.5 rounded',
                identity.tags.includes(tag) ? 'bg-indigo-700 text-white' : 'bg-gray-800 text-gray-500'
              )}
            >
              {tag}
            </button>
          ))}
        </div>
      </div>

      <div>
        <h4 className="text-xs text-gray-500 uppercase mb-1">Key cards</h4>
        <ul className="space-y-0.5">
          {identity.key_cards.map((name) => (
            <li key={name} className="flex justify-between text-sm text-white">
              {name}
              <button disabled={isSaving} onClick={() => unpin(name)} className="text-gray-500 hover:text-red-400">
                unpin
              </button>
            </li>
          ))}
        </ul>
        {pinnable.length > 0 && (
          <select
            disabled={isSaving}
            value=""
            onChange={(e) => e.target.value && pin(e.target.value)}
            className="mt-2 w-full bg-gray-800 text-sm text-gray-300 rounded px-2 py-1"
          >
            <option value="">Pin a key card…</option>
            {pinnable.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        )}
      </div>

      <div>
        <h4 className="text-xs text-gray-500 uppercase mb-1">Weakest fits</h4>
        {flagged.length === 0 ? (
          <p className="text-sm text-gray-400">Every card fits the deck's themes.</p>
        ) : (
          <ul className="space-y-0.5">
            {flagged.map((name) => (
              <li key={name} className="text-sm text-amber-400">
                {name}{' '}
                <span className="text-gray-500">
                  plan {Math.round(cards[name].plan_fit * 100)}%
                  {cards[name].synergy !== null && ` · synergy ${Math.round((cards[name].synergy ?? 0) * 100)}%`}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
