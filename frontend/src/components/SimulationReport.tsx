import { useState } from 'react';
import clsx from 'clsx';
import { CardTooltip } from '@/components/CardTooltip';
import { SimReport } from '@/types';

const pct = (x: number) => `${Math.round(100 * x)}%`;
const LABEL_STYLE = { favored: 'text-green-400', even: 'text-gray-300', unfavored: 'text-red-400' } as const;
const STOPPED: Record<string, string> = {
  user: 'Stopped early; this is the best deck found so far.',
  budget: 'The search ran out of time; this is the best deck found.',
  reverted: "The swaps didn't hold up over more games, so the first draft stands.",
};

interface Props {
  report: SimReport;
  kind: 'test' | 'build';
  compact?: boolean;
}

export function SimulationReport({ report, kind, compact }: Props) {
  const [game, setGame] = useState<'win' | 'loss' | null>(null);
  const { overall, baseline } = report;
  return (
    <div className="bg-gray-900 rounded-lg p-4 space-y-4 text-sm text-gray-200">
      <div>
        <div className="text-2xl font-semibold text-white">{pct(overall.win_rate)}</div>
        <div className="text-gray-400">
          against the top decks, likely {pct(overall.lo)}–{pct(overall.hi)} ({overall.games} games)
        </div>
        {kind === 'build' && baseline && (
          <div className="text-gray-400 mt-1">First draft: {pct(baseline.win_rate)}</div>
        )}
        {report.stopped && STOPPED[report.stopped] && (
          <div className="text-amber-300 mt-1">{STOPPED[report.stopped]}</div>
        )}
      </div>

      <section>
        <h3 className="text-white font-medium mb-1">Matchups</h3>
        <table className="w-full text-xs">
          <tbody>
            {report.matchups.map((m) => (
              <tr key={m.opponent}>
                <td className="py-0.5">{m.opponent}</td>
                <td className="text-right">{m.win_rate !== undefined ? pct(m.win_rate) : '—'}</td>
                <td className="text-right text-gray-500">
                  {m.lo !== undefined && m.hi !== undefined ? `${pct(m.lo)}–${pct(m.hi)}` : ''}
                </td>
                <td className={clsx('text-right', m.label && LABEL_STYLE[m.label])}>{m.label}</td>
                {!compact && <td className="text-right text-gray-500">{m.avg_turns ? `ends turn ${m.avg_turns.toFixed(1)}` : ''}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {report.changes.length > 0 && (
        <section>
          <h3 className="text-white font-medium mb-1">What changed</h3>
          <ul className="space-y-1 text-xs">
            {report.changes.map((c, i) => (
              <li key={i}>
                −{c.copies} <CardTooltip cardName={c.cut}>{c.cut}</CardTooltip> +{c.copies}{' '}
                <CardTooltip cardName={c.add}>{c.add}</CardTooltip>: {pct(c.before)} → {pct(c.after)} overall
                {c.best_matchup &&
                  `; vs ${c.best_matchup.opponent} ${pct(c.best_matchup.before)} → ${pct(c.best_matchup.after)}`}
              </li>
            ))}
          </ul>
        </section>
      )}

      {!compact && (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          {(['strongest', 'weakest'] as const).map((k) => (
            <div key={k}>
              <h3 className="text-white font-medium mb-1 text-sm">{k === 'strongest' ? 'Pulling weight' : 'Underperforming'}</h3>
              <ul className="space-y-0.5">
                {report.cards[k].map((c) => (
                  <li key={c.name}>
                    <CardTooltip cardName={c.name}>{c.name}</CardTooltip>{' '}
                    <span className="text-gray-400">
                      won {c.win_rate_when_cast === null ? '—' : pct(c.win_rate_when_cast)} when cast · cast in {c.games_cast} games
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
          {report.cards.too_few.length > 0 && (
            <p className="text-gray-500 md:col-span-2">
              Cast too rarely to judge: {report.cards.too_few.join(', ')}
            </p>
          )}
        </section>
      )}

      <section className="text-xs">
        <h3 className="text-white font-medium mb-1 text-sm">Mana</h3>
        <p className="text-gray-400">
          Mulligans {pct(report.mana.mulligan_rate)} · stuck on lands {pct(report.mana.screw_rate)} · flooded{' '}
          {pct(report.mana.flood_rate)}
        </p>
        {report.mana.advice.map((a) => (
          <p key={a} className="text-amber-300">{a}</p>
        ))}
      </section>

      {!compact && (report.games.win || report.games.loss) && (
        <section className="text-xs">
          <div className="flex gap-2 mb-1">
            <h3 className="text-white font-medium text-sm mr-2">Watch a game</h3>
            {report.games.win && (
              <button className="underline text-gray-300" onClick={() => setGame(game === 'win' ? null : 'win')}>a win</button>
            )}
            {report.games.loss && (
              <button className="underline text-gray-300" onClick={() => setGame(game === 'loss' ? null : 'loss')}>a loss</button>
            )}
          </div>
          {game && (
            <ol className="max-h-64 overflow-y-auto bg-gray-950 rounded p-2 space-y-0.5 font-mono">
              {(report.games[game] ?? []).map((line, i) => (
                <li key={i} className={line.startsWith('Turn ') ? 'text-white mt-1' : 'text-gray-400'}>
                  {line.replace(/^Tested/, 'You').replace(/^Opponent/, 'Opponent')}
                </li>
              ))}
            </ol>
          )}
        </section>
      )}

      <footer className="text-xs text-gray-500 space-y-0.5">
        <p>{report.limits}</p>
        {report.not_simulated.cards.length > 0 && (
          <p>Not simulated (Forge can't play them yet): {report.not_simulated.cards.join(', ')}.</p>
        )}
        <p>The sideboard isn't tested: games are best-of-one.</p>
      </footer>
    </div>
  );
}
