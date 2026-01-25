// User types
export interface User {
  id: string;
  email: string;
  username: string;
  display_name?: string;
  avatar_url?: string;
  is_active: boolean;
  is_verified: boolean;
  is_superuser: boolean;
  created_at: string;
}

export interface UserPreferences {
  default_format: string;
  card_display_size: 'small' | 'medium' | 'large';
  show_card_prices: boolean;
  auto_save_decks: boolean;
}

// Card types
export interface Card {
  id: string;
  scryfall_id: string;
  oracle_id?: string;
  name: string;
  mana_cost?: string;
  cmc?: number;
  type_line?: string;
  oracle_text?: string;
  power?: string;
  toughness?: string;
  colors?: string[];
  color_identity?: string[];
  keywords?: string[];
  set_code?: string;
  set_name?: string;
  collector_number?: string;
  rarity?: string;
  image_uri?: string;
  image_uri_small?: string;
  image_uri_art_crop?: string;
  price_usd?: number;
  price_usd_foil?: number;
  is_standard_legal?: boolean;
}

export interface CardSearchParams {
  q?: string;
  colors?: string[];
  card_type?: string;
  cmc_min?: number;
  cmc_max?: number;
  standard_only?: boolean;
  limit?: number;
  offset?: number;
}

// Deck types
export interface DeckEntry {
  card_id?: string;
  card_name: string;
  quantity: number;
  set_code?: string;
  collector_number?: string;
  card?: Card;
}

export interface Deck {
  id: string;
  owner_id: string;
  name: string;
  description?: string;
  format: string;
  archetype?: string;
  commander?: DeckEntry; // For Commander/cEDH formats
  main_deck: DeckEntry[];
  sideboard: DeckEntry[];
  strategy_summary?: string;
  card_explanations?: Record<string, string>;
  matchup_notes?: Record<string, string>;
  visibility: 'private' | 'unlisted' | 'public';
  share_token?: string;
  is_validated: boolean;
  validation_errors?: ValidationError[];
  created_at: string;
  updated_at: string;
}

export interface ValidationError {
  error_type: string;
  message: string;
  card_name?: string;
}

export interface SlotRecommendation {
  slot_type: string;
  role_description: string;
  card_name: string;
  quantity: number;
  reasoning: string;
}

export interface SideboardEntry {
  card_name: string;
  quantity: number;
  matchups: string[];
  reasoning: string;
}

export interface ChangeLogEntry {
  action: 'added' | 'removed' | 'changed';
  card_name: string;
  old_quantity?: number;
  new_quantity?: number;
  reasoning: string;
}

// Sideboard Matrix types
export interface SideboardCardChange {
  card_name: string;
  quantity: number;
  reasoning: string;
}

export interface MatchupSideboardPlan {
  matchup: string;
  matchup_description: string;
  cards_in: SideboardCardChange[];
  cards_out: SideboardCardChange[];
  strategy_notes: string;
  key_cards_to_find: string[];
  cards_to_play_around: string[];
}

export interface SideboardMatrixResponse {
  deck_name: string;
  deck_archetype?: string;
  generated_at: string;
  matchups: MatchupSideboardPlan[];
  general_sideboard_notes: string;
}

// Conversation types
export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
}

export interface Conversation {
  id: string;
  user_id?: string;
  summary?: string;
  messages: Message[];
  current_deck?: Partial<Deck>;
  created_at: string;
  updated_at: string;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  deck?: Partial<Deck>;
  suggestions?: string[];
}

export interface CardExplanationResponse {
  card_name: string;
  role: string;
  explanation: string;
  synergies: string[];
  alternatives: string[];
}

// Meta types
export interface MetaArchetype {
  name: string;
  meta_percentage: number;
  sample_size: number;
  avg_finish: number;
  key_cards: string[];
}

export interface CooccurrenceData {
  card1_name: string;
  card2_name: string;
  cooccurrence_count: number;
}

// Auth types
export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
  password_confirm: string;
  display_name?: string;
}

// API response types
export interface ApiResponse<T> {
  data: T;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}
