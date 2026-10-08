import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Message, Conversation, CardSuggestionGroup } from '@/types';

export type ConversationMode = 'build' | 'guided' | null;

interface ConversationState {
  currentConversation: Conversation | null;
  conversations: Conversation[];
  isLoading: boolean;
  currentFormat: string; // Format for current conversation
  cardSuggestions: CardSuggestionGroup[] | null;
  lastConversationId: string | null;
  conversationIds: string[]; // conversations this browser started, for signed-out history
  conversationMode: ConversationMode;
  simulationId: string | null;
  simulationConversationId: string | null; // the conversation the playtest belongs to

  setCurrentConversation: (conversation: Conversation | null) => void;
  setConversations: (conversations: Conversation[]) => void;
  addMessage: (message: Message) => void;
  setLoading: (loading: boolean) => void;
  setFormat: (format: string) => void;
  setCardSuggestions: (suggestions: CardSuggestionGroup[] | null) => void;
  setConversationMode: (mode: ConversationMode) => void;
  setSimulationId: (id: string | null, conversationId?: string | null) => void;
  clearSimulationUnless: (conversationId: string) => void;
  reset: () => void;
}

export const useConversationStore = create<ConversationState>()(
  persist(
    (set) => ({
      currentConversation: null,
      conversations: [],
      isLoading: false,
      currentFormat: 'standard',
      cardSuggestions: null,
      lastConversationId: null,
      conversationIds: [],
      conversationMode: null,
      simulationId: null,
      simulationConversationId: null,

      setCurrentConversation: (conversation) => {
        // When loading a conversation, also set its format from current_deck if available
        const format = conversation?.current_deck?.format || 'standard';
        set((state) => ({
          currentConversation: conversation,
          currentFormat: format,
          lastConversationId: conversation?.id || null,
          conversationIds:
            conversation?.id && !state.conversationIds.includes(conversation.id)
              ? [conversation.id, ...state.conversationIds]
              : state.conversationIds,
        }));
      },

      setConversations: (conversations) => set({ conversations }),

      addMessage: (message) =>
        set((state) => {
          if (!state.currentConversation) {
            return {
              currentConversation: {
                id: '',
                messages: [message],
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
              },
            };
          }

          return {
            currentConversation: {
              ...state.currentConversation,
              messages: [...state.currentConversation.messages, message],
              updated_at: new Date().toISOString(),
            },
          };
        }),

      setLoading: (isLoading) => set({ isLoading }),

      setFormat: (format) => set({ currentFormat: format }),

      setCardSuggestions: (cardSuggestions) => set({ cardSuggestions }),

      setConversationMode: (conversationMode) => set({ conversationMode }),

      setSimulationId: (simulationId, conversationId = null) =>
        set({ simulationId, simulationConversationId: simulationId ? conversationId : null }),

      // Drop the playtest unless it belongs to the conversation being loaded.
      clearSimulationUnless: (conversationId) =>
        set((state) =>
          state.simulationConversationId === conversationId
            ? {}
            : { simulationId: null, simulationConversationId: null }
        ),

      reset: () =>
        set({
          currentConversation: null,
          conversations: [],
          isLoading: false,
          currentFormat: 'standard',
          cardSuggestions: null,
          lastConversationId: null,
          conversationMode: null,
          simulationId: null,
          simulationConversationId: null,
        }),
    }),
    {
      name: 'spellbook-conversation',
      partialize: (state) => ({
        currentConversation: state.currentConversation,
        currentFormat: state.currentFormat,
        cardSuggestions: state.cardSuggestions,
        lastConversationId: state.lastConversationId,
        conversationIds: state.conversationIds,
        conversationMode: state.conversationMode,
        simulationId: state.simulationId,
        simulationConversationId: state.simulationConversationId,
      }),
    }
  )
);
