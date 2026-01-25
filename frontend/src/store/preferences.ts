import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface PreferencesState {
  defaultFormat: string;
  language: string;
  setDefaultFormat: (format: string) => void;
  setLanguage: (language: string) => void;
  setPreferences: (prefs: { default_format?: string; language?: string }) => void;
}

export const usePreferencesStore = create<PreferencesState>()(
  persist(
    (set) => ({
      defaultFormat: 'standard',
      language: 'en',
      setDefaultFormat: (format) => set({ defaultFormat: format }),
      setLanguage: (language) => set({ language }),
      setPreferences: (prefs) =>
        set({
          defaultFormat: prefs.default_format || 'standard',
          language: prefs.language || 'en',
        }),
    }),
    {
      name: 'spellbook-preferences',
    }
  )
);
