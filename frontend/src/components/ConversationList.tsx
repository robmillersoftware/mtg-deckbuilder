import { useEffect, useState } from 'react';
import { conversationsApi } from '@/services/api';
import { useConversationStore } from '@/store/conversation';
import { Conversation } from '@/types';
import clsx from 'clsx';

interface ConversationListProps {
  className?: string;
}

export function ConversationList({ className }: ConversationListProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const { currentConversation, setCurrentConversation } = useConversationStore();

  useEffect(() => {
    loadConversations();
  }, []);

  const loadConversations = async () => {
    setIsLoading(true);
    try {
      const response = await conversationsApi.list(20, 0);
      setConversations(response.data || []);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelect = async (conversation: Conversation) => {
    try {
      const response = await conversationsApi.getById(conversation.id);
      setCurrentConversation(response.data);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await conversationsApi.delete(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (currentConversation?.id === id) {
        setCurrentConversation(null);
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return date.toLocaleDateString([], { weekday: 'short' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  return (
    <div className={clsx('bg-gray-900 rounded-lg', className)}>
      <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">History</h2>
        <button
          onClick={() => setCurrentConversation(null)}
          className="text-sm text-primary-400 hover:text-primary-300"
        >
          New
        </button>
      </div>

      <div className="overflow-y-auto max-h-96">
        {isLoading ? (
          <div className="p-4 text-center text-gray-400">Loading...</div>
        ) : conversations.length === 0 ? (
          <div className="p-4 text-center text-gray-400">No conversations yet</div>
        ) : (
          <div className="divide-y divide-gray-800">
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                onClick={() => handleSelect(conversation)}
                className={clsx(
                  'px-4 py-3 cursor-pointer hover:bg-gray-800 transition-colors',
                  currentConversation?.id === conversation.id && 'bg-gray-800'
                )}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-white truncate">
                      {conversation.summary || 'New Conversation'}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      {formatDate(conversation.updated_at)}
                    </p>
                  </div>
                  <button
                    onClick={(e) => handleDelete(e, conversation.id)}
                    className="ml-2 p-1 text-gray-500 hover:text-red-400 transition-colors"
                    title="Delete"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
