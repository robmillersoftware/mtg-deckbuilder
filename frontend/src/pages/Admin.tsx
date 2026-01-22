import { useState, useEffect } from 'react';
import { useAuth } from '@/hooks/useAuth';
import toast from 'react-hot-toast';
import api from '@/services/api';

interface JobMetrics {
  total_runs_30d: number;
  success_count_30d: number;
  failure_count_30d: number;
  success_rate_30d: number;
  last_success: string | null;
}

interface JobRun {
  id: string;
  job_name: string;
  run_id: string;
  status: string;
  started_at: string | null;
  ended_at: string | null;
  duration_seconds: number | null;
  records_processed: number | null;
  records_inserted: number | null;
  records_updated: number | null;
  error_message: string | null;
  attempt_number: number | null;
}

interface JobsHealth {
  [key: string]: {
    status: string;
    last_success_timestamp: string | null;
    last_failure_timestamp: string | null;
    next_scheduled_run: string | null;
  };
}

export function AdminPage() {
  const { user } = useAuth();
  const [metrics, setMetrics] = useState<Record<string, JobMetrics>>({});
  const [history, setHistory] = useState<JobRun[]>([]);
  const [health, setHealth] = useState<JobsHealth>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isTriggering, setIsTriggering] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setIsLoading(true);
    try {
      const [metricsRes, historyRes, healthRes] = await Promise.all([
        api.get('/admin/dashboard/jobs'),
        api.get('/admin/jobs/history', { params: { limit: 50 } }),
        api.get('/health/jobs'),
      ]);
      setMetrics(metricsRes.data);
      setHistory(historyRes.data.jobs || []);
      setHealth(healthRes.data);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setIsLoading(false);
    }
  };

  const triggerJob = async (jobName: string) => {
    setIsTriggering(jobName);
    try {
      const response = await api.post(`/admin/jobs/${jobName}/run`);
      toast.success(`Job ${jobName} triggered. Run ID: ${response.data.run_id}`);
      // Refresh data after a short delay
      setTimeout(loadDashboardData, 2000);
    } catch (error) {
      console.error('Failed to trigger job:', error);
      toast.error('Failed to trigger job');
    } finally {
      setIsTriggering(null);
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  const formatDuration = (seconds: number | null) => {
    if (seconds === null) return '-';
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}m ${secs}s`;
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'success':
      case 'completed':
      case 'healthy':
        return 'text-green-400 bg-green-900/30';
      case 'failed':
      case 'unhealthy':
        return 'text-red-400 bg-red-900/30';
      case 'running':
      case 'pending':
        return 'text-yellow-400 bg-yellow-900/30';
      default:
        return 'text-gray-400 bg-gray-900/30';
    }
  };

  if (!user?.is_superuser) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-white mb-2">Access Denied</h2>
          <p className="text-gray-400">You need admin privileges to access this page.</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">Operations Dashboard</h1>
        <button
          onClick={loadDashboardData}
          className="px-4 py-2 text-sm bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Health Overview */}
      <div className="grid gap-6 md:grid-cols-2 mb-8">
        {Object.entries(health).map(([jobName, jobHealth]) => (
          <div
            key={jobName}
            className="bg-gray-900 rounded-lg p-6 border border-gray-800"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white capitalize">
                {jobName.replace('_', ' ')}
              </h3>
              <span
                className={`px-2 py-1 text-xs rounded ${getStatusColor(jobHealth.status)}`}
              >
                {jobHealth.status}
              </span>
            </div>

            <div className="space-y-2 text-sm mb-4">
              <div className="flex justify-between">
                <span className="text-gray-400">Last Success:</span>
                <span className="text-gray-300">
                  {formatDate(jobHealth.last_success_timestamp)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Last Failure:</span>
                <span className="text-gray-300">
                  {formatDate(jobHealth.last_failure_timestamp)}
                </span>
              </div>
              {jobHealth.next_scheduled_run && (
                <div className="flex justify-between">
                  <span className="text-gray-400">Next Run:</span>
                  <span className="text-gray-300">
                    {formatDate(jobHealth.next_scheduled_run)}
                  </span>
                </div>
              )}
            </div>

            <button
              onClick={() => triggerJob(jobName)}
              disabled={isTriggering === jobName}
              className="w-full px-4 py-2 text-sm bg-primary-600 hover:bg-primary-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
            >
              {isTriggering === jobName ? 'Triggering...' : 'Run Now'}
            </button>
          </div>
        ))}
      </div>

      {/* Metrics Cards */}
      <h2 className="text-xl font-semibold text-white mb-4">30-Day Metrics</h2>
      <div className="grid gap-6 md:grid-cols-2 mb-8">
        {Object.entries(metrics).map(([jobName, jobMetrics]) => (
          <div
            key={jobName}
            className="bg-gray-900 rounded-lg p-6 border border-gray-800"
          >
            <h3 className="text-lg font-semibold text-white capitalize mb-4">
              {jobName.replace('_', ' ')}
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold text-white">
                  {jobMetrics.total_runs_30d}
                </div>
                <div className="text-sm text-gray-400">Total Runs</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold text-green-400">
                  {jobMetrics.success_rate_30d.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-400">Success Rate</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold text-green-400">
                  {jobMetrics.success_count_30d}
                </div>
                <div className="text-sm text-gray-400">Successes</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold text-red-400">
                  {jobMetrics.failure_count_30d}
                </div>
                <div className="text-sm text-gray-400">Failures</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Job History */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white">Execution History</h2>
        <select
          value={selectedJob || ''}
          onChange={(e) => setSelectedJob(e.target.value || null)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white"
        >
          <option value="">All Jobs</option>
          <option value="scryfall_sync">Scryfall Sync</option>
          <option value="mtgtop8_scrape">mtgtop8 Scrape</option>
        </select>
      </div>

      <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-800">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">
                  Job
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">
                  Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">
                  Started
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">
                  Duration
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">
                  Records
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">
                  Attempt
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {history
                .filter((run) => !selectedJob || run.job_name === selectedJob)
                .map((run) => (
                  <tr key={run.id} className="hover:bg-gray-850">
                    <td className="px-4 py-3 text-sm text-white capitalize">
                      {run.job_name.replace('_', ' ')}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`px-2 py-1 text-xs rounded ${getStatusColor(run.status)}`}
                      >
                        {run.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {formatDate(run.started_at)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {formatDuration(run.duration_seconds)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {run.records_processed ?? '-'}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300">
                      {run.attempt_number ?? 1}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>

        {history.length === 0 && (
          <div className="p-8 text-center text-gray-400">
            No job execution history found.
          </div>
        )}
      </div>
    </div>
  );
}
