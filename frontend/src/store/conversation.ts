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
  conversationMode: ConversationMode;

  setCurrentConversation: (conversation: Conversation | null) => void;
  setConversations: (conversations: Conversation[]) => void;
  addMessage: (message: Message) => void;
  setLoading: (loading: boolean) => void;
  setFormat: (format: string) => void;
  setCardSuggestions: (suggestions: CardSuggestionGroup[] | null) => void;
  setConversationMode: (mode: ConversationMode) => void;
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
      conversationMode: null,

      setCurrentConversation: (conversation) => {
        // When loading a conversation, also set its format from current_deck if available
        const format = conversation?.current_deck?.format || 'standard';
        set({
          currentConversation: conversation,
          currentFormat: format,
          lastConversationId: conversation?.id || null,
        });
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

      reset: () =>
        set({
          currentConversation: null,
          conversations: [],
          isLoading: false,
          currentFormat: 'standard',
          cardSuggestions: null,
          lastConversationId: null,
          conversationMode: null,
        }),
    }),
    {
      name: 'spellbook-conversation',
      partialize: (state) => ({
        currentConversation: state.currentConversation,
        currentFormat: state.currentFormat,
        cardSuggestions: state.cardSuggestions,
        lastConversationId: state.lastConversationId,
        conversationMode: state.conversationMode,
      }),
    }
  )
);
