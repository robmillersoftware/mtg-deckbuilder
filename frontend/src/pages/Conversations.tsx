import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { conversationsApi } from '@/services/api';
import { Conversation } from '@/types';
import { useAuth } from '@/hooks/useAuth';
import toast from 'react-hot-toast';
import { EmptyState } from '@/components/onboarding';

export function ConversationsPage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);

  useEffect(() => {
    if (user) {
      loadConversations();
    }
  }, [user]);

  const loadConversations = async () => {
    setIsLoading(true);
    try {
      const response = await conversationsApi.list(50, 0);
      setConversations(response.data || []);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this conversation?')) return;

    try {
      await conversationsApi.delete(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (selectedConversation?.id === id) {
        setSelectedConversation(null);
      }
      toast.success('Conversation deleted');
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      toast.error('Failed to delete conversation');
    }
  };

  const handleContinue = (conversationId: string) => {
    // Navigate to home with conversation context
    navigate(`/?conversation=${conversationId}`);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return date.toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
      });
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return date.toLocaleDateString(undefined, { weekday: 'long' });
    } else {
      return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
      });
    }
  };

  if (!user) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Please log in to view conversations</div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading conversations...</div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">Conversation History</h1>
        <Link
          to="/"
          className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
        >
          New Conversation
        </Link>
      </div>

      {conversations.length === 0 ? (
        <EmptyState variant="conversations" />
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Conversation List */}
          <div className="space-y-3">
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                onClick={() => setSelectedConversation(conversation)}
                className={`bg-gray-900 rounded-lg p-4 cursor-pointer transition-colors ${
                  selectedConversation?.id === conversation.id
                    ? 'ring-2 ring-primary-500'
                    : 'hover:bg-gray-850'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-white font-medium truncate">
                      {conversation.summary || 'Untitled Conversation'}
                    </h3>
                    <p className="text-sm text-gray-400 mt-1">
                      {conversation.messages?.length || 0} messages
                    </p>
                  </div>
                  <div className="flex items-center gap-2 ml-4">
                    <span className="text-xs text-gray-500">
                      {formatDate(conversation.updated_at)}
                    </span>
                    <button
                      onClick={(e) => handleDelete(conversation.id, e)}
                      className="p-1 text-gray-500 hover:text-red-400 transition-colors"
                      title="Delete"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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

                {/* Show deck info if present */}
                {conversation.current_deck && (
                  <div className="mt-3 pt-3 border-t border-gray-800">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 text-xs bg-primary-900/50 text-primary-300 rounded">
                        Deck
                      </span>
                      <span className="text-sm text-gray-300">
                        {conversation.current_deck.name || 'Unnamed Deck'}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Conversation Detail */}
          {selectedConversation ? (
            <div className="bg-gray-900 rounded-lg p-6 sticky top-6 max-h-[calc(100vh-200px)] overflow-y-auto">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white">
                  {selectedConversation.summary || 'Conversation Detail'}
                </h2>
                <button
                  onClick={() => handleContinue(selectedConversation.id)}
                  className="px-3 py-1 text-sm bg-primary-600 hover:bg-primary-700 text-white rounded transition-colors"
                >
                  Continue
                </button>
              </div>

              <div className="space-y-4">
                {selectedConversation.messages?.map((message, index) => (
                  <div
                    key={index}
                    className={`p-3 rounded-lg ${
                      message.role === 'user'
                        ? 'bg-gray-800 ml-8'
                        : 'bg-gray-850 mr-8'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xs font-medium text-gray-400">
                        {message.role === 'user' ? 'You' : 'Spellbook'}
                      </span>
                      {message.timestamp && (
                        <span className="text-xs text-gray-600">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-300 whitespace-pre-wrap">
                      {message.content}
                    </p>
                  </div>
                ))}

                {(!selectedConversation.messages || selectedConversation.messages.length === 0) && (
                  <p className="text-gray-500 text-center py-8">No messages in this conversation</p>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-gray-900 rounded-lg p-6 flex items-center justify-center text-gray-500">
              Select a conversation to view details
            </div>
          )}
        </div>
      )}
    </div>
  );
}
