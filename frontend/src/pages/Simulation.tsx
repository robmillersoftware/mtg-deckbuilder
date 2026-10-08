import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { decksApi, simulationApi } from '@/services/api';
import { useAuth } from '@/hooks/useAuth';
import { useDeckStore } from '@/store/deck';
import { isActive, useSimulationRun } from '@/hooks/useSimulation';
import { SimulationProgress } from '@/components/SimulationProgress';
import { SimulationReport } from '@/components/SimulationReport';
import { Deck, SimulationRun } from '@/types';

const GAME_OPTIONS = [20, 50, 100];
const MAX_OPPONENTS = 10; // the API's limit
const CURRENT = 'current'; // the deck in the current conversation

export function SimulationPage() {
  const { isAuthenticated } = useAuth();
  const [params, setParams] = useSearchParams();
  const [decks, setDecks] = useState<Deck[]>([]);
  const currentDeck = useDeckStore((s) => s.currentDeck);
  const hasCurrent = (currentDeck?.main_deck?.length ?? 0) > 0;
  const [deckId, setDeckId] = useState(params.get('deck') ?? (hasCurrent ? CURRENT : ''));
  const [archetypes, setArchetypes] = useState<string[]>([]);
  const [chosen, setChosen] = useState<string[]>([]);
  const [games, setGames] = useState(50);
  const [runs, setRuns] = useState<SimulationRun[]>([]);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const runId = params.get('run');
  const { data: run } = useSimulationRun(runId);

  useEffect(() => {
    if (isAuthenticated) {
      decksApi.list(100, 0).then((r) => setDecks(Array.isArray(r.data) ? r.data : (r.data?.items ?? []))).catch(() => setDecks([]));
      simulationApi.list().then((r) => setRuns(r.data)).catch(() => setRuns([]));
    }
  }, [isAuthenticated, run?.status]);

  const format = (deckId === CURRENT ? currentDeck?.format : decks.find((d) => d.id === deckId)?.format) ?? 'standard';
  useEffect(() => {
    setChosen([]);
    simulationApi.archetypes(format).then((r) => setArchetypes(r.data.slice(0, 12))).catch(() => setArchetypes([]));
  }, [format]);

  const start = async () => {
    const saved = decks.find((d) => d.id === deckId);
    const current = deckId === CURRENT && hasCurrent ? currentDeck : null;
    if (!saved && !current) return;
    setStarting(true);
    try {
      const { data } = await simulationApi.create({
        ...(saved
          ? { deck_id: saved.id }
          : {
              deck: {
                name: current?.name || 'Current deck',
                main_deck: (current?.main_deck ?? []).map((e) => ({ card_name: e.card_name, quantity: e.quantity })),
              },
            }),
        format, games, opponents: chosen.length ? chosen : undefined,
      });
      setParams({ run: data.id });
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? "Couldn't start the test");
    } finally {
      setStarting(false);
    }
  };

  const stop = async () => {
    if (!run) return;
    setStopping(true);
    try {
      await simulationApi.stop(run.id);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? "Couldn't stop the test");
    } finally {
      setStopping(false);
    }
  };

  const toggle = (a: string) => setChosen((c) => (c.includes(a) ? c.filter((x) => x !== a) : [...c, a]));

  return (
    <div className="max-w-5xl mx-auto p-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="space-y-4">
        <div className="bg-gray-900 rounded-lg p-4 space-y-3 text-sm">
          <h1 className="text-lg font-semibold text-white">Test a deck</h1>
          <p className="text-gray-400">
            Plays your deck against real lists from the current meta with Forge, a Magic rules engine.
          </p>
          <label className="block">
            <span className="text-gray-300">Deck</span>
            <select value={deckId} onChange={(e) => setDeckId(e.target.value)}
                    className="mt-1 w-full bg-gray-800 text-white rounded p-2">
              <option value="">Choose a deck…</option>
              {hasCurrent && <option value={CURRENT}>Current deck{currentDeck?.name ? ` (${currentDeck.name})` : ''}</option>}
              {decks.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </label>
          <div>
            <span className="text-gray-300">Opponents</span>
            <p className="text-xs text-gray-500">
              None chosen: the top 5 decks, weighted by meta share. Choose up to {MAX_OPPONENTS}.
            </p>
            <div className="mt-1 flex flex-wrap gap-1">
              {archetypes.map((a) => (
                <button key={a} onClick={() => toggle(a)} aria-pressed={chosen.includes(a)}
                        disabled={!chosen.includes(a) && chosen.length >= MAX_OPPONENTS}
                        className={`text-xs px-2 py-1 rounded disabled:opacity-40 ${chosen.includes(a) ? 'bg-primary-600 text-white' : 'bg-gray-800 text-gray-300'}`}>
                  {a}
                </button>
              ))}
            </div>
          </div>
          <label className="block">
            <span className="text-gray-300">Games per opponent</span>
            <select value={games} onChange={(e) => setGames(Number(e.target.value))}
                    className="mt-1 w-full bg-gray-800 text-white rounded p-2">
              {GAME_OPTIONS.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          </label>
          <button onClick={start} disabled={!deckId || starting}
                  className="w-full py-2 rounded bg-primary-600 hover:bg-primary-500 text-white disabled:opacity-50">
            {starting ? 'Starting…' : 'Run test'}
          </button>
          {!isAuthenticated && <p className="text-xs text-gray-500">Sign in to test your saved decks.</p>}
        </div>
        {runs.length > 0 && (
          <div className="bg-gray-900 rounded-lg p-4 text-sm">
            <h2 className="text-white font-medium mb-2">Recent runs</h2>
            <ul className="space-y-1">
              {runs.map((r) => (
                <li key={r.id}>
                  <button onClick={() => setParams({ run: r.id })} className="text-left w-full text-gray-300 hover:text-white">
                    {r.deck.name ?? 'Deck'} · {r.kind === 'build' ? 'build playtest' : 'test'} · {r.status}
                    {r.report ? ` · ${Math.round(100 * r.report.overall.win_rate)}%` : ''}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div className="lg:col-span-2 space-y-4">
        {!run && <div className="bg-gray-900 rounded-lg p-8 text-center text-gray-400">Choose a deck and run a test.</div>}
        {run && isActive(run) && <SimulationProgress run={run} onStop={stop} stopping={stopping} />}
        {run && run.status === 'failed' && (
          <div className="bg-gray-900 rounded-lg p-4 text-red-300 text-sm">{run.error ?? 'The test failed.'}</div>
        )}
        {run && run.status === 'stopped' && !run.report && (
          <div className="bg-gray-900 rounded-lg p-4 text-gray-300 text-sm">Stopped before any games finished.</div>
        )}
        {run?.report && <SimulationReport report={run.report} kind={run.kind} chosenOpponents={!!run.opponents?.length} />}
      </div>
    </div>
  );
}
