import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Deck, DeckEntry, ChangeLogEntry } from '@/types';

// Normalize commander to always be a DeckEntry (handles legacy string format)
function normalizeCommander(commander: unknown): DeckEntry | undefined {
  if (!commander) return undefined;

  // If it's already a DeckEntry object with card_name
  if (typeof commander === 'object' && commander !== null && 'card_name' in commander) {
    return commander as DeckEntry;
  }

  // If it's a string (legacy format), convert to DeckEntry
  if (typeof commander === 'string') {
    return {
      card_name: commander,
      quantity: 1,
    };
  }

  return undefined;
}

interface DeckState {
  currentDeck: Partial<Deck> | null;
  changeLog: ChangeLogEntry[];
  isDirty: boolean;

  setCurrentDeck: (deck: Partial<Deck> | null) => void;
  updateMainDeck: (mainDeck: DeckEntry[]) => void;
  updateSideboard: (sideboard: DeckEntry[]) => void;
  addCard: (card: DeckEntry, target: 'main' | 'sideboard') => void;
  removeCard: (cardName: string, target: 'main' | 'sideboard') => void;
  updateCardQuantity: (cardName: string, quantity: number, target: 'main' | 'sideboard') => void;
  addChangeLog: (entry: ChangeLogEntry) => void;
  clearChangeLog: () => void;
  setDirty: (dirty: boolean) => void;
  reset: () => void;
}

export const useDeckStore = create<DeckState>()(
  persist(
    (set, _get) => ({
      currentDeck: null,
      changeLog: [],
      isDirty: false,

      setCurrentDeck: (deck) => set({
        currentDeck: deck ? {
          ...deck,
          commander: normalizeCommander(deck.commander),
        } : null,
        isDirty: false
      }),

      updateMainDeck: (mainDeck) =>
        set((state) => ({
          currentDeck: state.currentDeck ? { ...state.currentDeck, main_deck: mainDeck } : null,
          isDirty: true,
        })),

      updateSideboard: (sideboard) =>
        set((state) => ({
          currentDeck: state.currentDeck ? { ...state.currentDeck, sideboard } : null,
          isDirty: true,
        })),

      addCard: (card, target) =>
        set((state) => {
          // Initialize a new deck if one doesn't exist yet (e.g. when adding
          // cards from AI suggestions before a full deck has been generated)
          const deck = state.currentDeck || { main_deck: [], sideboard: [] };

          const list = target === 'main'
            ? [...(deck.main_deck || [])]
            : [...(deck.sideboard || [])];

          const existingIndex = list.findIndex((e) => e.card_name === card.card_name);

          if (existingIndex >= 0) {
            list[existingIndex] = {
              ...list[existingIndex],
              quantity: list[existingIndex].quantity + card.quantity,
            };
          } else {
            list.push(card);
          }

          return {
            currentDeck: {
              ...deck,
              [target === 'main' ? 'main_deck' : 'sideboard']: list,
            },
            isDirty: true,
          };
        }),

      removeCard: (cardName, target) =>
        set((state) => {
          if (!state.currentDeck) return state;

          const list = target === 'main'
            ? (state.currentDeck.main_deck || []).filter((e) => e.card_name !== cardName)
            : (state.currentDeck.sideboard || []).filter((e) => e.card_name !== cardName);

          return {
            currentDeck: {
              ...state.currentDeck,
              [target === 'main' ? 'main_deck' : 'sideboard']: list,
            },
            isDirty: true,
          };
        }),

      updateCardQuantity: (cardName, quantity, target) =>
        set((state) => {
          if (!state.currentDeck) return state;

          const list = target === 'main'
            ? [...(state.currentDeck.main_deck || [])]
            : [...(state.currentDeck.sideboard || [])];

          const index = list.findIndex((e) => e.card_name === cardName);

          if (index >= 0) {
            if (quantity <= 0) {
              list.splice(index, 1);
            } else {
              list[index] = { ...list[index], quantity };
            }
          }

          return {
            currentDeck: {
              ...state.currentDeck,
              [target === 'main' ? 'main_deck' : 'sideboard']: list,
            },
            isDirty: true,
          };
        }),

      addChangeLog: (entry) =>
        set((state) => ({
          changeLog: [...state.changeLog, entry],
        })),

      clearChangeLog: () => set({ changeLog: [] }),

      setDirty: (isDirty) => set({ isDirty }),

      reset: () =>
        set({
          currentDeck: null,
          changeLog: [],
          isDirty: false,
        }),
    }),
    {
      name: 'spellbook-deck',
      partialize: (state) => ({
        currentDeck: state.currentDeck,
      }),
    }
  )
);
