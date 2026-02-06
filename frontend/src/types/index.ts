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

export interface ArchetypeTrend {
  name: string;
  current_percentage: number;
  previous_percentage: number;
  change: number;
  change_percent: number;
  sample_size: number;
  key_cards: string[];
}

export interface MetaTrendsResponse {
  format: string;
  current_date: string;
  comparison_date: string;
  rising: ArchetypeTrend[];
  falling: ArchetypeTrend[];
  new_archetypes: MetaArchetype[];
  disappeared: string[];
}

export interface MetaHealthResponse {
  format: string;
  snapshot_date: string;
  diversity_score: number;
  top_deck_share: number;
  top_3_share: number;
  total_archetypes: number;
  health_rating: 'Healthy' | 'Moderate' | 'Concentrated' | 'Unhealthy' | 'Unknown';
  assessment: string;
}

export interface CardArchetypeBreakdown {
  name: string;
  count: number;
  percentage: number;
}

export interface CardMetaStatsEntry {
  card_name: string;
  deck_count: number;
  total_decks: number;
  meta_percentage: number;
  main_deck_count: number;
  sideboard_count: number;
  avg_copies: number;
  archetypes: CardArchetypeBreakdown[];
}

export interface CardMetaStatsResponse {
  format: string;
  snapshot_date: string;
  total_cards: number;
  cards: CardMetaStatsEntry[];
}

export interface CardTrend {
  card_name: string;
  current_percentage: number;
  previous_percentage: number;
  change: number;
  change_percent: number;
  current_deck_count: number;
  avg_copies: number;
}

export interface CardTrendsResponse {
  format: string;
  current_date: string;
  comparison_date: string;
  rising: CardTrend[];
  falling: CardTrend[];
  new_cards: CardMetaStatsEntry[];
  disappeared: string[];
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

// Simulation types
export interface TurnAction {
  turn_number: number;
  active_player: string; // "you" or "opponent" (or "opponent_1", etc. for multiplayer)
  life_totals: Record<string, number>;
  actions: string[];
  board_state?: Record<string, any>;
}

export interface GameResult {
  game_number: number;
  winner: string; // 'you', 'opponent', or 'opponent_1', 'opponent_2', etc. for multiplayer
  turns: number;
  your_life: number;
  opponent_life: number;
  // Multiplayer support
  life_totals?: Record<string, number>; // {"you": 40, "opponent_1": 35, ...}
  elimination_order?: string[]; // Order players were eliminated
  win_condition: string;
  key_moments: string[];
  your_key_cards: string[];
  opponent_key_cards: string[];
  opponent_key_cards_by_player?: Record<string, string[]>; // Per-opponent in multiplayer
  sideboard_in?: string[];
  sideboard_out?: string[];
  transcript?: TurnAction[]; // Full turn-by-turn game data
}

export interface KeyCardAnalysis {
  card: string;
  importance: number;
  reason: string;
}

export interface DeckRecommendation {
  category: 'add_cards' | 'remove_cards' | 'adjust_quantities' | 'sideboard' | 'strategy';
  priority: 'high' | 'medium' | 'low';
  suggestion: string;
  cards_mentioned: string[];
  reasoning: string;
}

export interface MatchupAnalysisResult {
  your_deck_name: string;
  opponent_deck_name: string;
  games_played: number;
  your_wins: number;
  opponent_wins: number;
  win_rate: number;
  average_game_length: number;
  matchup_assessment: 'favored' | 'even' | 'unfavored';
  key_cards_for_you: KeyCardAnalysis[];
  key_cards_against_you: KeyCardAnalysis[];
  sideboard_guide: { in: string[]; out: string[] };
  strategic_advice: string[];
  mulligan_advice: string;
  games: GameResult[];
}

export interface SimulationRun {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  your_deck_id?: string;
  your_deck_name: string;
  opponent_deck_name: string;
  opponent_archetype?: string;
  // Multiplayer support
  num_players?: number;
  opponent_deck_names?: string[];
  opponent_archetypes?: string[];
  format: string;
  num_games: number;
  include_sideboard_games: boolean;
  games_completed: number;
  current_game_turn?: number;
  current_game_turns?: TurnAction[];  // Live turns from in-progress game

  // Results (when completed)
  your_wins?: number;
  opponent_wins?: number;
  // Multiplayer results
  first_place_count?: number;
  your_placement_avg?: number;
  win_rate?: number;
  average_game_length?: number;
  matchup_assessment?: 'favored' | 'even' | 'unfavored';
  games?: GameResult[];
  key_cards_for_you?: KeyCardAnalysis[];
  key_cards_against_you?: KeyCardAnalysis[];
  sideboard_guide?: { in: string[]; out: string[] };
  strategic_advice?: string[];
  mulligan_advice?: string;
  deck_recommendations?: DeckRecommendation[];

  error_message?: string;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
}

export interface SimulationRunListResponse {
  items: SimulationRun[];
  total: number;
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
