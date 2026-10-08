import clsx from 'clsx';
import { SimulationRun } from '@/types';

const pct = (x: number) => `${Math.round(100 * x)}%`;

function timeLeft(run: SimulationRun): string | null {
  const p = run.progress;
  if (!p || !p.games_done || p.games_done >= p.games_planned) return null;
  const elapsed = (Date.now() - new Date(p.started_at).getTime()) / 1000;
  const left = (elapsed / p.games_done) * (p.games_planned - p.games_done);
  const minutes = Math.ceil(left / 60);
  return minutes <= 1 ? 'under a minute left' : `up to ${minutes} min left`;
}

interface Props {
  run: SimulationRun;
  onStop?: () => void;
  stopping?: boolean;
}

export function SimulationProgress({ run, onStop, stopping }: Props) {
  const p = run.progress;
  if (run.status === 'queued') {
    return (
      <div className="bg-gray-900 rounded-lg p-4 text-sm text-gray-300">
        Waiting for the simulator
        {run.queue_position ? ` (${run.queue_position - 1} ahead of you)` : ''}…
      </div>
    );
  }
  const done = p ? Math.min(1, p.games_done / Math.max(1, p.games_planned)) : 0;
  const left = timeLeft(run);
  return (
    <div className="bg-gray-900 rounded-lg p-4 space-y-3 text-sm">
      <div className="flex items-center justify-between gap-2">
        <div className="text-white font-medium">{p?.stage ?? 'Starting'}</div>
        {onStop && (
          <button
            onClick={onStop}
            disabled={stopping || run.stop_requested}
            className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-200 disabled:opacity-50"
          >
            {stopping ? 'Stopping…' : 'Stop'}
          </button>
        )}
      </div>
      <div>
        <div className="h-2 rounded bg-gray-800 overflow-hidden">
          <div className="h-2 bg-primary-500 transition-all" style={{ width: `${Math.round(100 * done)}%` }} />
        </div>
        <div className="mt-1 text-xs text-gray-400">
          {p?.games_done ?? 0} games played{left ? ` · ${left}` : ''}
        </div>
      </div>
      {p && p.matchups.length > 0 && (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 text-left">
              <th className="font-normal">Opponent</th>
              <th className="font-normal text-right">Meta</th>
              <th className="font-normal text-right">W–L</th>
              <th className="font-normal text-right">Win</th>
            </tr>
          </thead>
          <tbody>
            {p.matchups.map((m) => {
              const games = m.wins + m.losses + m.draws;
              const rate = games ? (m.wins + m.draws / 2) / games : null;
              return (
                <tr key={m.opponent} className="text-gray-200">
                  <td className="py-0.5">{m.opponent}</td>
                  <td className="text-right text-gray-400">{m.share > 1 ? `${m.share.toFixed(1)}%` : ''}</td>
                  <td className="text-right">{m.wins}–{m.losses}{m.draws ? `–${m.draws}` : ''}</td>
                  <td className={clsx('text-right', rate !== null && (rate > 0.55 ? 'text-green-400' : rate < 0.45 ? 'text-red-400' : 'text-gray-200'))}>
                    {rate === null ? '—' : pct(rate)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {p && p.events.length > 0 && (
        <ul className="space-y-1 text-xs max-h-40 overflow-y-auto">
          {[...p.events].reverse().slice(0, 8).map((e, i) => (
            <li key={i} className={clsx(e.kind === 'kept' ? 'text-green-300' : e.kind === 'tried' ? 'text-gray-400' : 'text-gray-300')}>
              {e.text.replace(/\[\[([^\]]+)\]\]/g, '$1')}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
