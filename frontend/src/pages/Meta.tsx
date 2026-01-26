import { useEffect, useState } from 'react';
import { metaApi, usersApi } from '@/services/api';
import { MetaArchetype, MetaTrendsResponse, MetaHealthResponse, ArchetypeTrend } from '@/types';
import { CardTooltip } from '@/components/CardTooltip';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import clsx from 'clsx';

const FORMAT_DISPLAY_NAMES: Record<string, string> = {
  standard: 'Standard',
  historic: 'Historic',
  modern: 'Modern',
  legacy: 'Legacy',
  cedh: 'cEDH',
};

const AVAILABLE_FORMATS = ['standard', 'historic', 'modern', 'legacy', 'cedh'];

// cEDH uses commanders instead of archetypes
const isCommanderFormat = (format: string) => format === 'cedh';

// Health rating colors
const HEALTH_COLORS: Record<string, string> = {
  Healthy: 'text-green-400',
  Moderate: 'text-yellow-400',
  Concentrated: 'text-orange-400',
  Unhealthy: 'text-red-400',
  Unknown: 'text-gray-400',
};

const HEALTH_BG_COLORS: Record<string, string> = {
  Healthy: 'bg-green-500',
  Moderate: 'bg-yellow-500',
  Concentrated: 'bg-orange-500',
  Unhealthy: 'bg-red-500',
  Unknown: 'bg-gray-500',
};

interface HistoryDataPoint {
  date: string;
  percentage: number;
}

export function MetaPage() {
  const [archetypes, setArchetypes] = useState<MetaArchetype[]>([]);
  const [selectedArchetype, setSelectedArchetype] = useState<MetaArchetype | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [format, setFormat] = useState<string>('standard');

  // New state for trends and health
  const [trends, setTrends] = useState<MetaTrendsResponse | null>(null);
  const [health, setHealth] = useState<MetaHealthResponse | null>(null);
  const [trendsLoading, setTrendsLoading] = useState(false);

  // History chart data
  const [historyData, setHistoryData] = useState<HistoryDataPoint[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  useEffect(() => {
    loadPreferencesAndMeta();
  }, []);

  // Load history when archetype is selected
  useEffect(() => {
    if (selectedArchetype) {
      loadArchetypeHistory(selectedArchetype.name);
    } else {
      setHistoryData([]);
    }
  }, [selectedArchetype, format]);

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
      await loadMetaData(userFormat);
    } catch (error) {
      console.error('Failed to load meta:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadMetaData = async (selectedFormat: string) => {
    setIsLoading(true);
    setTrendsLoading(true);

    try {
      // Load all data in parallel
      const [dashboardRes, trendsRes, healthRes] = await Promise.allSettled([
        metaApi.getDashboard(selectedFormat),
        metaApi.getTrends(selectedFormat, 7),
        metaApi.getHealth(selectedFormat),
      ]);

      if (dashboardRes.status === 'fulfilled') {
        setArchetypes(dashboardRes.value.data.archetypes || []);
        setLastUpdated(dashboardRes.value.data.last_updated || null);
      }

      if (trendsRes.status === 'fulfilled') {
        setTrends(trendsRes.value.data);
      } else {
        setTrends(null);
      }

      if (healthRes.status === 'fulfilled') {
        setHealth(healthRes.value.data);
      } else {
        setHealth(null);
      }
    } catch (error) {
      console.error('Failed to load meta data:', error);
    } finally {
      setIsLoading(false);
      setTrendsLoading(false);
    }
  };

  const loadArchetypeHistory = async (archetype: string) => {
    setHistoryLoading(true);
    try {
      const response = await metaApi.getHistory(archetype, format, 10);
      const data = response.data
        .map((snapshot: { snapshot_date: string; meta_percentage: number }) => ({
          date: new Date(snapshot.snapshot_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          percentage: Number(snapshot.meta_percentage) || 0,
        }))
        .reverse(); // Oldest first for the chart
      setHistoryData(data);
    } catch (error) {
      console.error('Failed to load archetype history:', error);
      setHistoryData([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleFormatChange = (newFormat: string) => {
    setFormat(newFormat);
    setSelectedArchetype(null);
    loadMetaData(newFormat);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading meta data...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Format Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">{FORMAT_DISPLAY_NAMES[format] || format} Metagame</h1>
          <p className="text-gray-400 mt-1">
            Current competitive meta breakdown based on tournament results
          </p>
        </div>

        {/* Format Selector - wraps on mobile */}
        <div className="flex flex-wrap gap-2">
          {AVAILABLE_FORMATS.map((f) => (
            <button
              key={f}
              onClick={() => handleFormatChange(f)}
              className={clsx(
                'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors whitespace-nowrap',
                format === f
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              )}
            >
              {FORMAT_DISPLAY_NAMES[f]}
            </button>
          ))}
        </div>
      </div>

      {/* Meta Health Card */}
      {health && (
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold text-white">Format Health</h2>
            <span className={clsx('font-semibold', HEALTH_COLORS[health.health_rating])}>
              {health.health_rating}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
            <div>
              <div className="text-sm text-gray-400">Diversity Score</div>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={clsx('h-full rounded-full', HEALTH_BG_COLORS[health.health_rating])}
                    style={{ width: `${health.diversity_score}%` }}
                  />
                </div>
                <span className="text-white font-medium">{health.diversity_score}</span>
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Top Deck Share</div>
              <div className="text-white font-medium">{health.top_deck_share}%</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Top 3 Share</div>
              <div className="text-white font-medium">{health.top_3_share}%</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Total Archetypes</div>
              <div className="text-white font-medium">{health.total_archetypes}</div>
            </div>
          </div>

          <p className="text-gray-300 text-sm">{health.assessment}</p>
        </div>
      )}

      {archetypes.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-400">
            No meta data available yet. Check back after tournament data is synced.
          </p>
        </div>
      ) : (
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Left Column: Archetype List + Trends */}
          <div className="flex-1 lg:flex-[2] space-y-6 order-1">
            {/* Archetype/Commander List - Scrollable */}
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-800">
                <h2 className="text-lg font-semibold text-white">
                  {isCommanderFormat(format) ? 'Top Commanders' : 'Top Archetypes'}
                </h2>
              </div>

              <div className="divide-y divide-gray-800 max-h-[400px] overflow-y-auto">
                {archetypes.map((archetype, index) => (
                  <button
                    key={archetype.name}
                    onClick={() => setSelectedArchetype(archetype)}
                    className={clsx(
                      'w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800 transition-colors text-left',
                      selectedArchetype?.name === archetype.name && 'bg-gray-800'
                    )}
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-gray-500 text-sm w-6">#{index + 1}</span>
                      <div className="w-16 hidden sm:block">
                        <div
                          className="h-2 rounded-full bg-primary-600"
                          style={{ width: `${Math.min(archetype.meta_percentage * 3, 100)}%` }}
                        />
                      </div>
                      <div>
                        <span className="text-white font-medium">{archetype.name}</span>
                        <span className="text-gray-500 text-sm ml-2 hidden sm:inline">
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

            {/* Trends Section */}
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-800">
                <h2 className="text-lg font-semibold text-white">Meta Trends</h2>
                {trends && (
                  <p className="text-sm text-gray-400 mt-1">
                    Comparing to {new Date(trends.comparison_date).toLocaleDateString()}
                  </p>
                )}
              </div>

              {trendsLoading ? (
                <div className="p-4 text-center text-gray-400">Loading trends...</div>
              ) : trends ? (
                <div className="p-4 space-y-6">
                  {/* Rising Decks */}
                  {trends.rising.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium text-green-400 mb-3 flex items-center gap-2">
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                        Rising
                      </h3>
                      <div className="space-y-2">
                        {trends.rising.map((trend) => (
                          <TrendCard key={trend.name} trend={trend} isRising={true} />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Falling Decks */}
                  {trends.falling.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium text-red-400 mb-3 flex items-center gap-2">
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        Falling
                      </h3>
                      <div className="space-y-2">
                        {trends.falling.map((trend) => (
                          <TrendCard key={trend.name} trend={trend} isRising={false} />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* New Archetypes */}
                  {trends.new_archetypes.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium text-blue-400 mb-3 flex items-center gap-2">
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z" />
                        </svg>
                        New to Meta
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {trends.new_archetypes.map((arch) => (
                          <span
                            key={arch.name}
                            className="px-3 py-1 bg-blue-900/30 text-blue-300 rounded-full text-sm"
                          >
                            {arch.name} ({arch.meta_percentage.toFixed(1)}%)
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Disappeared Archetypes */}
                  {trends.disappeared.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 000 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                        </svg>
                        Fallen Off
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {trends.disappeared.map((name) => (
                          <span
                            key={name}
                            className="px-3 py-1 bg-gray-800 text-gray-400 rounded-full text-sm line-through"
                          >
                            {name}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* No Trends */}
                  {trends.rising.length === 0 &&
                    trends.falling.length === 0 &&
                    trends.new_archetypes.length === 0 &&
                    trends.disappeared.length === 0 && (
                      <p className="text-gray-400 text-center py-4">
                        No significant changes in the meta this week.
                      </p>
                    )}
                </div>
              ) : (
                <div className="p-4 text-center text-gray-400">
                  Not enough historical data to show trends.
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Selected Archetype Details */}
          <div className="lg:flex-1 order-2">
            {selectedArchetype ? (
              <div className="bg-gray-900 rounded-lg p-4 lg:sticky lg:top-4">
                <h2 className="text-lg font-semibold text-white mb-4">
                  {selectedArchetype.name}
                </h2>

                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
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
                  </div>

                  {selectedArchetype.avg_finish > 0 && (
                    <div>
                      <span className="text-sm text-gray-400">Avg Finish</span>
                      <p className="text-lg text-white">
                        {selectedArchetype.avg_finish.toFixed(1)}
                      </p>
                    </div>
                  )}

                  {/* History Chart */}
                  <div>
                    <span className="text-sm text-gray-400 block mb-2">Meta Share Over Time</span>
                    {historyLoading ? (
                      <div className="h-32 flex items-center justify-center text-gray-500">
                        Loading...
                      </div>
                    ) : historyData.length > 1 ? (
                      <div className="h-32 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={historyData} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
                            <XAxis
                              dataKey="date"
                              tick={{ fill: '#9CA3AF', fontSize: 10 }}
                              axisLine={{ stroke: '#374151' }}
                              tickLine={{ stroke: '#374151' }}
                            />
                            <YAxis
                              tick={{ fill: '#9CA3AF', fontSize: 10 }}
                              axisLine={{ stroke: '#374151' }}
                              tickLine={{ stroke: '#374151' }}
                              domain={['dataMin - 2', 'dataMax + 2']}
                              tickFormatter={(value) => `${value}%`}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: '#1F2937',
                                border: '1px solid #374151',
                                borderRadius: '0.5rem',
                                color: '#F3F4F6',
                              }}
                              formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Meta Share']}
                            />
                            <Line
                              type="monotone"
                              dataKey="percentage"
                              stroke="#3B82F6"
                              strokeWidth={2}
                              dot={{ fill: '#3B82F6', strokeWidth: 0, r: 3 }}
                              activeDot={{ fill: '#60A5FA', strokeWidth: 0, r: 5 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="h-32 flex items-center justify-center text-gray-500 text-sm">
                        Not enough data for chart
                      </div>
                    )}
                  </div>

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
                <p>Select {isCommanderFormat(format) ? 'a commander' : 'an archetype'} to see details</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Trend Card Component
function TrendCard({ trend, isRising }: { trend: ArchetypeTrend; isRising: boolean }) {
  return (
    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
      <div>
        <span className="text-white font-medium">{trend.name}</span>
        <div className="text-sm text-gray-400">
          {trend.previous_percentage.toFixed(1)}% → {trend.current_percentage.toFixed(1)}%
        </div>
      </div>
      <div className={clsx(
        'text-sm font-medium px-2 py-1 rounded',
        isRising ? 'text-green-400 bg-green-900/30' : 'text-red-400 bg-red-900/30'
      )}>
        {isRising ? '+' : ''}{trend.change.toFixed(1)}%
      </div>
    </div>
  );
}
