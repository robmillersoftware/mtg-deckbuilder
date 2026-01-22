import { create } from 'zustand';
import { Message, Conversation } from '@/types';

interface ConversationState {
  currentConversation: Conversation | null;
  conversations: Conversation[];
  isLoading: boolean;

  setCurrentConversation: (conversation: Conversation | null) => void;
  setConversations: (conversations: Conversation[]) => void;
  addMessage: (message: Message) => void;
  setLoading: (loading: boolean) => void;
  reset: () => void;
}

export const useConversationStore = create<ConversationState>((set) => ({
  currentConversation: null,
  conversations: [],
  isLoading: false,

  setCurrentConversation: (conversation) => set({ currentConversation: conversation }),

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

  reset: () =>
    set({
      currentConversation: null,
      conversations: [],
      isLoading: false,
    }),
}));
