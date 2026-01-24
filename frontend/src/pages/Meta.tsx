import { useEffect, useState } from 'react';
import { metaApi, usersApi } from '@/services/api';
import { MetaArchetype } from '@/types';
import { CardTooltip } from '@/components/CardTooltip';
import clsx from 'clsx';

const FORMAT_DISPLAY_NAMES: Record<string, string> = {
  standard: 'Standard',
  historic: 'Historic',
  modern: 'Modern',
  legacy: 'Legacy',
  cedh: 'cEDH',
};

export function MetaPage() {
  const [archetypes, setArchetypes] = useState<MetaArchetype[]>([]);
  const [selectedArchetype, setSelectedArchetype] = useState<MetaArchetype | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [format, setFormat] = useState<string>('standard');

  useEffect(() => {
    loadPreferencesAndMeta();
  }, []);

  const loadPreferencesAndMeta = async () => {
    setIsLoading(true);
    try {
      // Try to get user's preferred format
      let userFormat = 'standard';
      try {
        const prefsResponse = await usersApi.getPreferences();
        userFormat = prefsResponse.data.default_format || 'standard';
      } catch {
        // User not logged in or preferences unavailable, use default
      }
      setFormat(userFormat);

      const response = await metaApi.getDashboard(userFormat);
      setArchetypes(response.data.archetypes || []);
      setLastUpdated(response.data.last_updated || null);
    } catch (error) {
      console.error('Failed to load meta:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading meta data...</div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">{FORMAT_DISPLAY_NAMES[format] || format} Metagame</h1>
        <p className="text-gray-400 mt-1">
          Current competitive meta breakdown based on tournament results
        </p>
      </div>

      {archetypes.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-400">
            No meta data available yet. Check back after tournament data is synced.
          </p>
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Archetype List */}
          <div className="lg:col-span-2">
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-800">
                <h2 className="text-lg font-semibold text-white">Top Archetypes</h2>
              </div>

              <div className="divide-y divide-gray-800">
                {archetypes.map((archetype) => (
                  <button
                    key={archetype.name}
                    onClick={() => setSelectedArchetype(archetype)}
                    className={clsx(
                      'w-full px-4 py-4 flex items-center justify-between hover:bg-gray-800 transition-colors text-left',
                      selectedArchetype?.name === archetype.name && 'bg-gray-800'
                    )}
                  >
                    <div className="flex items-center space-x-4">
                      <div className="w-16">
                        <div
                          className="h-2 rounded-full bg-primary-600"
                          style={{ width: `${Math.min(archetype.meta_percentage * 3, 100)}%` }}
                        />
                      </div>
                      <div>
                        <span className="text-white font-medium">{archetype.name}</span>
                        <span className="text-gray-500 text-sm ml-2">
                          ({archetype.sample_size} decks)
                        </span>
                      </div>
                    </div>
                    <span className="text-primary-400 font-medium">
                      {archetype.meta_percentage.toFixed(1)}%
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Selected Archetype Details */}
          <div>
            {selectedArchetype ? (
              <div className="bg-gray-900 rounded-lg p-4">
                <h2 className="text-lg font-semibold text-white mb-4">
                  {selectedArchetype.name}
                </h2>

                <div className="space-y-4">
                  <div>
                    <span className="text-sm text-gray-400">Meta Share</span>
                    <p className="text-2xl font-bold text-primary-400">
                      {selectedArchetype.meta_percentage.toFixed(1)}%
                    </p>
                  </div>

                  <div>
                    <span className="text-sm text-gray-400">Sample Size</span>
                    <p className="text-lg text-white">
                      {selectedArchetype.sample_size} decks
                    </p>
                  </div>

                  {selectedArchetype.avg_finish > 0 && (
                    <div>
                      <span className="text-sm text-gray-400">Avg Finish</span>
                      <p className="text-lg text-white">
                        {selectedArchetype.avg_finish.toFixed(1)}
                      </p>
                    </div>
                  )}

                  {selectedArchetype.key_cards && selectedArchetype.key_cards.length > 0 && (
                    <div>
                      <span className="text-sm text-gray-400 block mb-2">Key Cards</span>
                      <div className="flex flex-wrap gap-2">
                        {selectedArchetype.key_cards.map((card) => (
                          <CardTooltip key={card} cardName={card}>
                            <span className="px-2 py-1 bg-gray-800 rounded text-sm text-gray-300 cursor-pointer hover:bg-gray-700 transition-colors">
                              {card}
                            </span>
                          </CardTooltip>
                        ))}
                      </div>
                    </div>
                  )}

                  {lastUpdated && (
                    <div className="pt-4 border-t border-gray-800">
                      <span className="text-xs text-gray-500">
                        Last updated: {new Date(lastUpdated).toLocaleDateString()}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="bg-gray-900 rounded-lg p-4 text-center text-gray-400">
                <p>Select an archetype to see details</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
