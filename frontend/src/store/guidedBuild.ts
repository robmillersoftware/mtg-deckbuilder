import { create } from 'zustand';

export type GuidedBuildStep =
  | 'strategy'
  | 'colors'
  | 'core'
  | 'support'
  | 'mana_base'
  | 'sideboard'
  | 'review';

export interface ArchetypeOption {
  name: string;
  description: string;
  playstyle: string;
  meta_percentage?: number;
  example_cards: string[];
}

export interface ColorOption {
  colors: string[];
  name: string;
  description: string;
  strengths: string[];
  weaknesses: string[];
}

export interface CardRecommendation {
  card_name: string;
  card_id?: string;
  quantity: number;
  role: string;
  reasoning: string;
  image_uri?: string;
  mana_cost?: string;
  type_line?: string;
}

export interface CardSlotGroup {
  slot_name: string;
  description: string;
  target_count: number;
  recommendations: CardRecommendation[];
}

export interface LandRecommendation {
  card_name: string;
  card_id?: string;
  quantity: number;
  category: string;
  reasoning: string;
  image_uri?: string;
}

export interface SideboardRecommendation {
  card_name: string;
  card_id?: string;
  quantity: number;
  target_matchups: string[];
  reasoning: string;
  image_uri?: string;
}

export interface StepResponse {
  session_id: string;
  current_step: GuidedBuildStep;
  step_index: number;
  total_steps: number;
  step_title: string;
  step_description: string;
  data: Record<string, any>;
  ai_message: string;
}

export interface CompleteResponse {
  session_id: string;
  deck_id?: string;
  deck_name: string;
  main_deck: Record<string, any>[];
  sideboard: Record<string, any>[];
  strategy_summary: string;
  archetype: string;
  colors: string[];
  format: string;
  is_valid: boolean;
  validation_errors: string[];
}

interface GuidedBuildState {
  sessionId: string | null;
  currentStep: GuidedBuildStep | null;
  stepIndex: number;
  totalSteps: number;
  stepTitle: string;
  stepDescription: string;
  stepData: Record<string, any>;
  aiMessage: string;
  isLoading: boolean;
  error: string | null;

  // User selections accumulated across steps
  selectedArchetype: string | null;
  selectedColors: string[];
  selectedCards: Record<string, any>[];
  selectedLands: Record<string, any>[];
  selectedSideboard: Record<string, any>[];
  format: string;

  // Completed deck
  completedDeck: CompleteResponse | null;

  // Actions
  setStepResponse: (response: StepResponse) => void;
  setSelectedArchetype: (archetype: string) => void;
  setSelectedColors: (colors: string[]) => void;
  setSelectedCards: (cards: Record<string, any>[]) => void;
  setSelectedLands: (lands: Record<string, any>[]) => void;
  setSelectedSideboard: (cards: Record<string, any>[]) => void;
  setFormat: (format: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setCompletedDeck: (deck: CompleteResponse | null) => void;
  reset: () => void;
}

const initialState = {
  sessionId: null,
  currentStep: null as GuidedBuildStep | null,
  stepIndex: 0,
  totalSteps: 7,
  stepTitle: '',
  stepDescription: '',
  stepData: {},
  aiMessage: '',
  isLoading: false,
  error: null as string | null,
  selectedArchetype: null as string | null,
  selectedColors: [] as string[],
  selectedCards: [] as Record<string, any>[],
  selectedLands: [] as Record<string, any>[],
  selectedSideboard: [] as Record<string, any>[],
  format: 'standard',
  completedDeck: null as CompleteResponse | null,
};

export const useGuidedBuildStore = create<GuidedBuildState>((set) => ({
  ...initialState,

  setStepResponse: (response) =>
    set({
      sessionId: response.session_id,
      currentStep: response.current_step,
      stepIndex: response.step_index,
      totalSteps: response.total_steps,
      stepTitle: response.step_title,
      stepDescription: response.step_description,
      stepData: response.data,
      aiMessage: response.ai_message,
      error: null,
    }),

  setSelectedArchetype: (archetype) => set({ selectedArchetype: archetype }),
  setSelectedColors: (colors) => set({ selectedColors: colors }),
  setSelectedCards: (cards) => set({ selectedCards: cards }),
  setSelectedLands: (lands) => set({ selectedLands: lands }),
  setSelectedSideboard: (cards) => set({ selectedSideboard: cards }),
  setFormat: (format) => set({ format }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  setCompletedDeck: (completedDeck) => set({ completedDeck }),
  reset: () => set(initialState),
}));
