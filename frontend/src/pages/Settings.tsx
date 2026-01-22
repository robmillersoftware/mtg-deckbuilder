import { useState, useEffect } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { usersApi } from '@/services/api';
import toast from 'react-hot-toast';

interface Preferences {
  language: string;
  theme: string;
  default_format: string;
}

export function SettingsPage() {
  const { user } = useAuth();
  const [preferences, setPreferences] = useState<Preferences>({
    language: 'en',
    theme: 'dark',
    default_format: 'standard',
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = async () => {
    try {
      const response = await usersApi.getPreferences();
      setPreferences(response.data);
    } catch (error) {
      console.error('Failed to load preferences:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = async (key: keyof Preferences, value: string) => {
    const newPreferences = { ...preferences, [key]: value };
    setPreferences(newPreferences);
    setIsSaving(true);

    try {
      await usersApi.updatePreferences({ [key]: value });
      toast.success('Preference saved');

      // Apply theme immediately
      if (key === 'theme') {
        document.documentElement.classList.toggle('dark', value === 'dark');
        document.documentElement.classList.toggle('light', value === 'light');
      }
    } catch (error) {
      console.error('Failed to save preference:', error);
      toast.error('Failed to save preference');
      // Revert on error
      setPreferences(preferences);
    } finally {
      setIsSaving(false);
    }
  };

  if (!user) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Please log in to view settings</div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading settings...</div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-6">Settings</h1>

      <div className="space-y-6">
        {/* Language */}
        <div className="bg-gray-900 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Language</h2>
          <p className="text-sm text-gray-400 mb-4">
            Choose your preferred language for the interface
          </p>
          <select
            value={preferences.language}
            onChange={(e) => handleChange('language', e.target.value)}
            disabled={isSaving}
            className="w-full md:w-64 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-primary-500 disabled:opacity-50"
          >
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
            <option value="de">German</option>
            <option value="pt">Portuguese</option>
            <option value="ja">Japanese</option>
          </select>
        </div>

        {/* Theme */}
        <div className="bg-gray-900 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Theme</h2>
          <p className="text-sm text-gray-400 mb-4">
            Choose between light and dark mode
          </p>
          <div className="flex gap-4">
            <button
              onClick={() => handleChange('theme', 'light')}
              disabled={isSaving}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg border transition-colors ${
                preferences.theme === 'light'
                  ? 'border-primary-500 bg-primary-900/20'
                  : 'border-gray-700 hover:border-gray-600'
              } disabled:opacity-50`}
            >
              <div className="w-8 h-8 rounded-full bg-white border border-gray-300 flex items-center justify-center">
                <svg className="w-5 h-5 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <span className="text-white">Light</span>
            </button>

            <button
              onClick={() => handleChange('theme', 'dark')}
              disabled={isSaving}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg border transition-colors ${
                preferences.theme === 'dark'
                  ? 'border-primary-500 bg-primary-900/20'
                  : 'border-gray-700 hover:border-gray-600'
              } disabled:opacity-50`}
            >
              <div className="w-8 h-8 rounded-full bg-gray-800 border border-gray-600 flex items-center justify-center">
                <svg className="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                </svg>
              </div>
              <span className="text-white">Dark</span>
            </button>
          </div>
        </div>

        {/* Default Format */}
        <div className="bg-gray-900 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Default Deck Format</h2>
          <p className="text-sm text-gray-400 mb-4">
            Choose the default format for new decks
          </p>
          <select
            value={preferences.default_format}
            onChange={(e) => handleChange('default_format', e.target.value)}
            disabled={isSaving}
            className="w-full md:w-64 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-primary-500 disabled:opacity-50"
          >
            <option value="standard">Standard</option>
            <option value="historic">Historic</option>
            <option value="modern">Modern</option>
            <option value="legacy">Legacy</option>
          </select>
        </div>
      </div>
    </div>
  );
}
