import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface OnboardingChecklistItem {
  id: string;
  title: string;
  description: string;
  completed: boolean;
  link?: string;
}

interface OnboardingState {
  // Welcome modal state
  hasSeenWelcome: boolean;
  setHasSeenWelcome: (seen: boolean) => void;

  // Feature tour state
  hasCompletedTour: boolean;
  currentTourStep: number;
  isTourActive: boolean;
  setHasCompletedTour: (completed: boolean) => void;
  setCurrentTourStep: (step: number) => void;
  setTourActive: (active: boolean) => void;
  startTour: () => void;
  nextTourStep: () => void;
  prevTourStep: () => void;
  endTour: () => void;

  // Checklist state
  checklistItems: OnboardingChecklistItem[];
  completeChecklistItem: (id: string) => void;
  resetChecklist: () => void;

  // Global reset
  resetOnboarding: () => void;
}

const DEFAULT_CHECKLIST_ITEMS: OnboardingChecklistItem[] = [
  {
    id: 'create-deck',
    title: 'Create your first deck',
    description: 'Use AI to generate a custom MTG deck',
    completed: false,
    link: '/',
  },
  {
    id: 'explore-meta',
    title: 'Explore the meta',
    description: 'See top-performing decks and strategies',
    completed: false,
    link: '/meta',
  },
  {
    id: 'try-simulator',
    title: 'Test your draws',
    description: 'Simulate opening hands and mulligan decisions',
    completed: false,
    link: '/simulate',
  },
  {
    id: 'import-deck',
    title: 'Import a deck',
    description: 'Bring in an existing deck from text or Arena',
    completed: false,
    link: '/import',
  },
];

export const useOnboardingStore = create<OnboardingState>()(
  persist(
    (set, get) => ({
      // Welcome modal
      hasSeenWelcome: false,
      setHasSeenWelcome: (seen) => set({ hasSeenWelcome: seen }),

      // Feature tour
      hasCompletedTour: false,
      currentTourStep: 0,
      isTourActive: false,
      setHasCompletedTour: (completed) => set({ hasCompletedTour: completed }),
      setCurrentTourStep: (step) => set({ currentTourStep: step }),
      setTourActive: (active) => set({ isTourActive: active }),

      startTour: () => set({ isTourActive: true, currentTourStep: 0 }),
      nextTourStep: () => {
        const { currentTourStep } = get();
        set({ currentTourStep: currentTourStep + 1 });
      },
      prevTourStep: () => {
        const { currentTourStep } = get();
        if (currentTourStep > 0) {
          set({ currentTourStep: currentTourStep - 1 });
        }
      },
      endTour: () => set({ isTourActive: false, hasCompletedTour: true }),

      // Checklist
      checklistItems: DEFAULT_CHECKLIST_ITEMS,
      completeChecklistItem: (id) =>
        set((state) => ({
          checklistItems: state.checklistItems.map((item) =>
            item.id === id ? { ...item, completed: true } : item
          ),
        })),
      resetChecklist: () => set({ checklistItems: DEFAULT_CHECKLIST_ITEMS }),

      // Global reset
      resetOnboarding: () =>
        set({
          hasSeenWelcome: false,
          hasCompletedTour: false,
          currentTourStep: 0,
          isTourActive: false,
          checklistItems: DEFAULT_CHECKLIST_ITEMS,
        }),
    }),
    {
      name: 'spellbook-onboarding',
    }
  )
);
