import axios, { AxiosError, AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import { useAuthStore } from '@/store/auth';

const API_BASE_URL = '/api';

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = useAuthStore.getState().accessToken;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor to handle token refresh
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

    // Skip token refresh for auth endpoints to avoid loops
    const isAuthEndpoint = originalRequest?.url?.includes('/auth/');

    // If 401 and we haven't already retried, and it's not an auth endpoint
    if (error.response?.status === 401 && !originalRequest._retry && !isAuthEndpoint) {
      originalRequest._retry = true;

      const refreshToken = useAuthStore.getState().refreshToken;
      if (refreshToken) {
        try {
          const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
            refresh_token: refreshToken,
          });

          const { access_token, refresh_token } = response.data;
          useAuthStore.getState().setTokens(access_token, refresh_token);

          // Retry original request with new token
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return api(originalRequest);
        } catch (refreshError) {
          // Refresh failed, logout user
          useAuthStore.getState().logout();
          // Only redirect if not already on auth pages
          const currentPath = window.location.pathname;
          if (!currentPath.startsWith('/login') && !currentPath.startsWith('/register')) {
            window.location.href = '/login';
          }
          return Promise.reject(refreshError);
        }
      }
    }

    return Promise.reject(error);
  }
);

export default api;

// Auth API
export const authApi = {
  login: (email: string, password: string) =>
    api.post('/auth/login', { email, password }),

  register: (email: string, password: string) =>
    api.post('/auth/register', { email, password }),

  verifyEmail: (token: string) =>
    api.post(`/auth/verify/${token}`),

  forgotPassword: (email: string) =>
    api.post('/auth/password-reset/request', { email }),

  resetPassword: (token: string, newPassword: string, newPasswordConfirm: string) =>
    api.post('/auth/password-reset/confirm', {
      token,
      new_password: newPassword,
      new_password_confirm: newPasswordConfirm,
    }),

  refresh: (refreshToken: string) =>
    api.post('/auth/refresh', { refresh_token: refreshToken }),
};

// Users API
export const usersApi = {
  getMe: () => api.get('/users/me'),

  updateProfile: (data: { display_name?: string; avatar_url?: string }) =>
    api.patch('/users/me', data),

  uploadAvatar: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/users/me/avatar', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  getPreferences: () => api.get('/users/me/preferences'),

  updatePreferences: (preferences: { language?: string; default_format?: string }) =>
    api.patch('/users/me/preferences', preferences),

  changePassword: (currentPassword: string, newPassword: string) =>
    api.post('/users/me/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    }),

  requestEmailChange: (newEmail: string) =>
    api.post('/users/me/email-change', { new_email: newEmail }),

  confirmEmailChange: (token: string) =>
    api.post('/users/me/email-change/confirm', { token }),
};

// Cards API
export const cardsApi = {
  search: (params: {
    q?: string;
    colors?: string[];
    card_type?: string;
    cmc_min?: number;
    cmc_max?: number;
    standard_only?: boolean;
    limit?: number;
    offset?: number;
  }) => api.get('/cards/search', { params }),

  getById: (id: string) => api.get(`/cards/${id}`),

  getByName: (name: string) => api.get(`/cards/by-name/${encodeURIComponent(name)}`),

  semanticSearch: (query: string, limit?: number) =>
    api.post('/cards/semantic-search', { query, limit }),

  getCandidates: (params: {
    colors: string[];
    archetype: string;
    slot_type: string;
    current_cards?: string[];
    limit?: number;
  }) => api.post('/cards/candidates', params),
};

// Decks API
export const decksApi = {
  list: (limit?: number, offset?: number) =>
    api.get('/decks', { params: { limit, offset } }),

  getById: (id: string) => api.get(`/decks/${id}`),

  getByShareToken: (token: string) => api.get(`/decks/public/${token}`),

  create: (deck: {
    name: string;
    description?: string;
    format?: string;
    archetype?: string;
    main_deck: { card_name: string; quantity: number }[];
    sideboard?: { card_name: string; quantity: number }[];
    visibility?: string;
  }) => api.post('/decks', deck),

  update: (id: string, deck: Partial<{
    name: string;
    description: string;
    main_deck: { card_name: string; quantity: number }[];
    sideboard: { card_name: string; quantity: number }[];
    visibility: string;
  }>) => api.patch(`/decks/${id}`, deck),

  delete: (id: string) => api.delete(`/decks/${id}`),

  validate: (mainDeck: { card_name: string; quantity: number }[], sideboard?: { card_name: string; quantity: number }[]) =>
    api.post('/decks/validate', { main_deck: mainDeck, sideboard }),

  import: (text: string, format?: string) =>
    api.post('/decks/import', { decklist_text: text, format }),

  export: (id: string, format?: string) =>
    api.get(`/decks/${id}/export`, { params: { format } }),

  generate: (prompt: string, conversationId?: string) =>
    api.post('/decks/generate', {
      prompt,
      conversation_id: conversationId,
    }),

  iterate: (modification: string, conversationId?: string, deckId?: string) =>
    api.post('/decks/iterate', {
      modification,
      conversation_id: conversationId,
      deck_id: deckId,
    }),

  toggleVisibility: (id: string, visibility: string) =>
    api.patch(`/decks/${id}`, { visibility }),

  getSideboardMatrix: (id: string) =>
    api.post(`/decks/${id}/sideboard-matrix`),
};

// Conversations API
export const conversationsApi = {
  list: (limit?: number, offset?: number) =>
    api.get('/conversations', { params: { limit, offset } }),

  getById: (id: string) => api.get(`/conversations/${id}`),

  create: () => api.post('/conversations'),

  sendMessage: (message: string, conversationId?: string, format?: string) =>
    api.post('/conversations/chat', {
      message,
      conversation_id: conversationId,
      format: format || 'standard',
    }),

  explainCard: (cardName: string, conversationId?: string) =>
    api.post('/conversations/explain-card', {
      card_name: cardName,
      conversation_id: conversationId,
    }),

  delete: (id: string) => api.delete(`/conversations/${id}`),
};

// Meta API
export const metaApi = {
  getDashboard: (format?: string) =>
    api.get('/meta', { params: { format } }),

  getArchetype: (archetype: string, format?: string) =>
    api.get(`/meta/archetypes/${encodeURIComponent(archetype)}`, { params: { format } }),

  getCooccurrence: (cardName: string, format?: string, limit?: number) =>
    api.get(`/meta/cooccurrence/${encodeURIComponent(cardName)}`, { params: { format, limit } }),

  getHistory: (archetype: string, format?: string, limit?: number) =>
    api.get('/meta/history', { params: { archetype, format, limit } }),

  getTrends: (format?: string, daysBack?: number) =>
    api.get('/meta/trends', { params: { format, days_back: daysBack } }),

  getHealth: (format?: string) =>
    api.get('/meta/health', { params: { format } }),
};

// Simulation API
export const simulationApi = {
  // Legacy synchronous endpoints
  runSimulation: (request: {
    your_deck: { deck_id?: string; main_deck?: { card_name: string; quantity: number }[]; sideboard?: { card_name: string; quantity: number }[]; name?: string };
    opponent_deck: { deck_id?: string; main_deck?: { card_name: string; quantity: number }[]; sideboard?: { card_name: string; quantity: number }[]; name?: string };
    num_games?: number;
    include_sideboard_games?: boolean;
    format?: string;
  }) => api.post('/simulation', request),

  simulateVsArchetype: (request: {
    deck_id: string;
    opponent_archetype: string;
    num_games?: number;
  }) => api.post('/simulation/vs-archetype', request),

  getAvailableArchetypes: (format: string = 'standard') =>
    api.get<string[]>('/simulation/archetypes', { params: { format } }),

  // Persistent simulation runs (background execution)
  listRuns: (params?: { limit?: number; offset?: number; status?: string }) =>
    api.get('/simulation/runs', { params }),

  createRun: (request: {
    deck_id: string;
    opponent_archetype: string;
    num_games?: number;
  }) => api.post('/simulation/runs', request),

  getRun: (id: string) => api.get(`/simulation/runs/${id}`),

  deleteRun: (id: string) => api.delete(`/simulation/runs/${id}`),

  retryRun: (id: string) => api.post(`/simulation/runs/${id}/retry`),
};
