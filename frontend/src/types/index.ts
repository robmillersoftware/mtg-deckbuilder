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
  fit_flagged?: string[];
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

export interface ConversationContext {
  strategy?: string;
  colors?: string[];
  phase?: string;
  build_around_cards?: string[];
  archetype?: string;
  roles_suggested?: string[];
  user_preferences?: string;
  summary?: string;
}

export interface Conversation {
  id: string;
  user_id?: string;
  summary?: string;
  messages: Message[];
  current_deck?: Partial<Deck>;
  context?: ConversationContext;
  created_at: string;
  updated_at: string;
}

export interface CardFit {
  plan_fit: number;
  synergy: number | null;
}

export interface FitScore extends CardFit {
  anti_synergy: number;
}

export interface CardSuggestionItem {
  card_name: string;
  quantity: number;
  mana_cost?: string;
  type_line?: string;
  image_uri?: string;
  reasoning?: string;
  fit?: CardFit | null;
}

export interface CardSuggestionGroup {
  group_name: string;
  role: string;
  cards: CardSuggestionItem[];
  is_batch: boolean;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  deck?: Partial<Deck>;
  suggestions?: string[];
  card_suggestions?: CardSuggestionGroup[];
  simulation_id?: string;
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

// Forge simulation
export interface SimMatchup {
  opponent: string;
  share: number;
  wins: number;
  losses: number;
  draws: number;
  games?: number;
  win_rate?: number;
  lo?: number;
  hi?: number;
  label?: 'favored' | 'even' | 'unfavored';
  avg_turns?: number;
}

export interface SimEvent {
  at: string;
  text: string;
  kind: 'info' | 'tried' | 'kept';
}

export interface SimProgress {
  stage: string;
  games_done: number;
  games_planned: number;
  started_at: string;
  matchups: SimMatchup[];
  events: SimEvent[];
  deck?: DeckEntry[] | null;
}

export interface SimRate {
  win_rate: number;
  lo: number;
  hi: number;
  games: number;
}

export interface SimCardStat {
  name: string;
  copies: number;
  games_cast: number;
  cast_share: number;
  win_rate_when_cast: number | null;
  median_turn: number | null;
}

export interface SimChange {
  cut: string;
  add: string;
  copies: number;
  before: number;
  after: number;
  best_matchup: { opponent: string; before: number; after: number } | null;
}

export interface SimReport {
  overall: SimRate;
  baseline: SimRate | null;
  matchups: SimMatchup[];
  changes: SimChange[];
  cards: { strongest: SimCardStat[]; weakest: SimCardStat[]; too_few: string[] };
  mana: { mulligan_rate: number; screw_rate: number; flood_rate: number; advice: string[] };
  games: { win: string[] | null; loss: string[] | null };
  not_simulated: { cards: string[]; sideboard: boolean };
  limits: string;
  stopped: string | null;
}

export interface SimulationRun {
  id: string;
  kind: 'test' | 'build';
  status: 'queued' | 'running' | 'completed' | 'failed' | 'stopped';
  format: string;
  deck: { name?: string; main_deck: DeckEntry[]; sideboard?: DeckEntry[] };
  opponents?: string[] | null;
  games_per_matchup: number;
  progress?: SimProgress | null;
  report?: SimReport | null;
  final_deck?: { name?: string; main_deck: DeckEntry[]; sideboard?: DeckEntry[] } | null;
  error?: string | null;
  queue_position?: number | null;
  stop_requested?: boolean;
  created_at: string;
  updated_at: string;
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

export interface IdentityOverrides {
  tags_on: string[];
  tags_off: string[];
  pinned: string[];
  unpinned: string[];
}

export interface DeckIdentity {
  tags: string[];
  key_cards: string[];
  request_text?: string | null;
  overrides: IdentityOverrides;
}

export interface DeckFitResponse {
  identity: DeckIdentity | null;
  cards: Record<string, FitScore>;
  flagged: string[];
  available_tags: string[];
}
