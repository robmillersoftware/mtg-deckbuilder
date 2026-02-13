import { useState, useCallback, useEffect } from 'react';
import { useConversationStore, ConversationMode } from '@/store/conversation';
import { useDeckStore } from '@/store/deck';
import { usePreferencesStore } from '@/store/preferences';
import { conversationsApi } from '@/services/api';
import { Message, ChatResponse } from '@/types';
import toast from 'react-hot-toast';

export function useChat(mode?: ConversationMode) {
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);

  const {
    currentConversation,
    setCurrentConversation,
    addMessage,
    currentFormat,
    cardSuggestions,
    setCardSuggestions,
    conversationMode,
    setConversationMode,
  } = useConversationStore();

  const { setCurrentDeck } = useDeckStore();

  // When a mode is specified, ensure the conversation belongs to this mode.
  // If the persisted conversation was from a different mode, clear it.
  useEffect(() => {
    if (!mode) return;
    if (conversationMode && conversationMode !== mode) {
      // Different mode — start fresh
      setCurrentConversation(null);
      setCurrentDeck(null);
      setSuggestions([]);
      setCardSuggestions(null);
      const defaultFormat = usePreferencesStore.getState().defaultFormat;
      useConversationStore.getState().setFormat(defaultFormat);
    }
    setConversationMode(mode);
  }, [mode]); // eslint-disable-line react-hooks/exhaustive-deps

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return;

    setIsLoading(true);

    // Add user message optimistically
    const userMessage: Message = {
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    addMessage(userMessage);

    try {
      // Get the current format from conversation store (per-conversation format)
      const format = useConversationStore.getState().currentFormat;

      // Get current deck state to sync with backend
      const currentDeck = useDeckStore.getState().currentDeck;

      const response = await conversationsApi.sendMessage(
        content,
        currentConversation?.id,
        format,
        currentDeck || undefined
      );

      const data: ChatResponse = response.data;

      // Update conversation ID if new - get current state from store to avoid stale closure
      if (data.conversation_id) {
        const currentState = useConversationStore.getState();
        const existingMessages = currentState.currentConversation?.messages || [];

        // Only update if the ID changed or we don't have a conversation yet
        if (!currentState.currentConversation || currentState.currentConversation.id !== data.conversation_id) {
          setCurrentConversation({
            id: data.conversation_id,
            messages: existingMessages,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          });
        }
      }

      // Add assistant message
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString(),
      };
      addMessage(assistantMessage);

      // Update deck if included in response (full deck generation / modification)
      if (data.deck) {
        setCurrentDeck(data.deck);
      }

      // Update card suggestions if included (persisted in conversation store)
      if (data.card_suggestions) {
        setCardSuggestions(data.card_suggestions);
      }

      // Update suggestions
      if (data.suggestions) {
        setSuggestions(data.suggestions);
      }

    } catch (error) {
      console.error('Chat error:', error);
      toast.error('Failed to send message. Please try again.');

      // Add error message
      addMessage({
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date().toISOString(),
      });
    } finally {
      setIsLoading(false);
    }
  }, [currentConversation, addMessage, setCurrentConversation, setCurrentDeck, setCardSuggestions, isLoading]);

  const explainCard = useCallback(async (cardName: string) => {
    if (isLoading) return;

    setIsLoading(true);

    try {
      const response = await conversationsApi.explainCard(
        cardName,
        currentConversation?.id
      );

      const data = response.data;

      // Format explanation as message
      const explanation = `**${data.card_name}**\n\n**Role:** ${data.role}\n\n${data.explanation}\n\n**Synergies:** ${data.synergies?.join(', ') || 'N/A'}\n\n**Alternatives:** ${data.alternatives?.join(', ') || 'N/A'}`;

      addMessage({
        role: 'user',
        content: `Explain ${cardName}`,
        timestamp: new Date().toISOString(),
      });

      addMessage({
        role: 'assistant',
        content: explanation,
        timestamp: new Date().toISOString(),
      });

    } catch (error) {
      console.error('Explain card error:', error);
      toast.error('Failed to explain card');
    } finally {
      setIsLoading(false);
    }
  }, [currentConversation, addMessage, isLoading]);

  const startNewConversation = useCallback(() => {
    setCurrentConversation(null);
    setCurrentDeck(null);
    setSuggestions([]);
    setCardSuggestions(null);
    // Reset format to user's default preference
    const defaultFormat = usePreferencesStore.getState().defaultFormat;
    useConversationStore.getState().setFormat(defaultFormat);
  }, [setCurrentConversation, setCurrentDeck, setCardSuggestions]);

  const setFormat = useCallback((format: string) => {
    useConversationStore.getState().setFormat(format);
  }, []);

  return {
    messages: currentConversation?.messages || [],
    conversationId: currentConversation?.id,
    isLoading,
    suggestions,
    cardSuggestions,
    format: currentFormat,
    setFormat,
    sendMessage,
    explainCard,
    startNewConversation,
  };
}
