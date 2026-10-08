import { useQuery } from '@tanstack/react-query';
import { simulationApi } from '@/services/api';
import { SimulationRun } from '@/types';

const ACTIVE = new Set<SimulationRun['status']>(['queued', 'running']);

export const isActive = (run?: SimulationRun | null) => !!run && ACTIVE.has(run.status);

// Polls every 2 s while the run is queued or running, then stops.
export function useSimulationRun(id: string | null | undefined) {
  return useQuery({
    queryKey: ['simulation', id],
    queryFn: async () => (await simulationApi.get(id!)).data,
    enabled: !!id,
    refetchInterval: (query) => (query.state.data && !isActive(query.state.data) ? false : 2000),
  });
}
