import { create } from 'zustand';
import { Message, Conversation } from '@/types';

interface ConversationState {
  currentConversation: Conversation | null;
  conversations: Conversation[];
  isLoading: boolean;
  currentFormat: string; // Format for current conversation

  setCurrentConversation: (conversation: Conversation | null) => void;
  setConversations: (conversations: Conversation[]) => void;
  addMessage: (message: Message) => void;
  setLoading: (loading: boolean) => void;
  setFormat: (format: string) => void;
  reset: () => void;
}

export const useConversationStore = create<ConversationState>((set) => ({
  currentConversation: null,
  conversations: [],
  isLoading: false,
  currentFormat: 'standard',

  setCurrentConversation: (conversation) => {
    // When loading a conversation, also set its format from current_deck if available
    const format = conversation?.current_deck?.format || 'standard';
    set({ currentConversation: conversation, currentFormat: format });
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

  reset: () =>
    set({
      currentConversation: null,
      conversations: [],
      isLoading: false,
      currentFormat: 'standard',
    }),
}));
