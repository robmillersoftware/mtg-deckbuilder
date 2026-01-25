import { useState, useEffect } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { usersApi } from '@/services/api';
import { usePreferencesStore } from '@/store/preferences';
import toast from 'react-hot-toast';

interface Preferences {
  language: string;
  default_format: string;
}

export function SettingsPage() {
  const { user } = useAuth();
  const { setPreferences: setStorePreferences } = usePreferencesStore();
  const [preferences, setPreferences] = useState<Preferences>({
    language: 'en',
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
      // Update global preferences store
      setStorePreferences(response.data);
    } catch (error) {
      console.error('Failed to load preferences:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = async (key: keyof Preferences, value: string) => {
    const newPreferences = { ...preferences, [key]: value };
    setPreferences(newPreferences);
    // Update global store immediately for responsive UI
    setStorePreferences(newPreferences);
    setIsSaving(true);

    try {
      await usersApi.updatePreferences({ [key]: value });
      toast.success('Preference saved');
    } catch (error) {
      console.error('Failed to save preference:', error);
      toast.error('Failed to save preference');
      // Revert on error
      setPreferences(preferences);
      setStorePreferences(preferences);
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
            <option value="cedh">cEDH</option>
          </select>
        </div>
      </div>
    </div>
  );
}
