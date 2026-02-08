import { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import clsx from 'clsx';
import { guidedBuildApi } from '@/services/api';
import {
  useGuidedBuildStore,
  type GuidedBuildStep,
  type ArchetypeOption,
  type ColorOption,
  type CardSlotGroup,
  type CardRecommendation,
  type LandRecommendation,
  type SideboardRecommendation,
} from '@/store/guidedBuild';

const STEP_ICONS: Record<GuidedBuildStep, string> = {
  strategy: '\u2694',   // crossed swords
  colors: '\u{1F3A8}', // palette
  core: '\u2B50',      // star
  support: '\u{1F6E1}', // shield
  mana_base: '\u{1F48E}', // gem
  sideboard: '\u{1F4CB}', // clipboard
  review: '\u2705',    // check mark
};

const COLOR_SYMBOLS: Record<string, { label: string; bg: string; text: string }> = {
  W: { label: 'White', bg: 'bg-amber-50', text: 'text-amber-900' },
  U: { label: 'Blue', bg: 'bg-blue-600', text: 'text-white' },
  B: { label: 'Black', bg: 'bg-gray-900', text: 'text-gray-100' },
  R: { label: 'Red', bg: 'bg-red-600', text: 'text-white' },
  G: { label: 'Green', bg: 'bg-green-600', text: 'text-white' },
};

export function GuidedBuilderPage() {
  const navigate = useNavigate();
  const store = useGuidedBuildStore();
  const [selectedFormat, setSelectedFormat] = useState('standard');

  const startBuild = useCallback(async () => {
    store.setLoading(true);
    store.setError(null);
    try {
      const response = await guidedBuildApi.start(selectedFormat);
      store.setFormat(selectedFormat);
      store.setStepResponse(response.data);
    } catch (err: any) {
      store.setError(err.response?.data?.detail || 'Failed to start guided build');
      toast.error('Failed to start guided build');
    } finally {
      store.setLoading(false);
    }
  }, [selectedFormat, store]);

  const advanceStep = useCallback(async (selections: Record<string, any>) => {
    if (!store.sessionId) return;
    store.setLoading(true);
    store.setError(null);
    try {
      const response = await guidedBuildApi.advance(store.sessionId, selections);
      store.setStepResponse(response.data);
    } catch (err: any) {
      store.setError(err.response?.data?.detail || 'Failed to advance step');
      toast.error('Failed to advance');
    } finally {
      store.setLoading(false);
    }
  }, [store]);

  const goBack = useCallback(async () => {
    if (!store.sessionId) return;
    store.setLoading(true);
    try {
      const response = await guidedBuildApi.goBack(store.sessionId);
      store.setStepResponse(response.data);
    } catch (err: any) {
      toast.error('Failed to go back');
    } finally {
      store.setLoading(false);
    }
  }, [store]);

  const completeBuild = useCallback(async (deckName: string, save: boolean) => {
    if (!store.sessionId) return;
    store.setLoading(true);
    try {
      const response = await guidedBuildApi.complete(store.sessionId, deckName, save);
      store.setCompletedDeck(response.data);
      if (save && response.data.deck_id) {
        toast.success('Deck saved!');
        navigate(`/deck/${response.data.deck_id}`);
      }
    } catch (err: any) {
      toast.error('Failed to complete build');
    } finally {
      store.setLoading(false);
    }
  }, [store, navigate]);

  // Reset on unmount
  useEffect(() => {
    return () => {
      store.reset();
    };
  }, []);

  // Not started yet - show intro
  if (!store.currentStep) {
    return (
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-white mb-4">
            Guided Deck Builder
          </h1>
          <p className="text-lg text-gray-400 max-w-2xl mx-auto">
            Build a competitive deck step by step with AI guidance. Choose your
            strategy, pick your colors, and let the AI help you select the perfect
            cards for each slot.
          </p>
        </div>

        {/* Format selection */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-8 mb-6">
          <h2 className="text-xl font-semibold text-white mb-4">Choose Format</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {['standard', 'historic', 'modern', 'legacy'].map((fmt) => (
              <button
                key={fmt}
                onClick={() => setSelectedFormat(fmt)}
                className={clsx(
                  'px-4 py-3 rounded-lg border text-sm font-medium capitalize transition-all',
                  selectedFormat === fmt
                    ? 'border-primary-500 bg-primary-500/10 text-primary-400'
                    : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-600'
                )}
              >
                {fmt}
              </button>
            ))}
          </div>
        </div>

        {/* Steps preview */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-8 mb-6">
          <h2 className="text-xl font-semibold text-white mb-4">How It Works</h2>
          <div className="space-y-3">
            {(['strategy', 'colors', 'core', 'support', 'mana_base', 'sideboard', 'review'] as GuidedBuildStep[]).map(
              (step, i) => (
                <div key={step} className="flex items-center gap-3 text-gray-300">
                  <span className="flex items-center justify-center w-8 h-8 rounded-full bg-gray-800 text-sm font-medium text-gray-400">
                    {i + 1}
                  </span>
                  <span className="text-lg mr-2">{STEP_ICONS[step]}</span>
                  <span className="capitalize">{step.replace('_', ' ')}</span>
                </div>
              )
            )}
          </div>
        </div>

        <button
          onClick={startBuild}
          disabled={store.isLoading}
          className="w-full py-4 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-700 text-white text-lg font-semibold rounded-xl transition-colors"
        >
          {store.isLoading ? 'Starting...' : "Let's Build a Deck"}
        </button>

        {store.error && (
          <p className="mt-4 text-center text-red-400">{store.error}</p>
        )}
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto">
      {/* Progress bar */}
      <ProgressBar
        currentIndex={store.stepIndex}
        totalSteps={store.totalSteps}
        currentStep={store.currentStep}
      />

      {/* AI Message */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 mb-6">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-full bg-primary-600 flex items-center justify-center flex-shrink-0 mt-0.5">
            <span className="text-sm text-white font-bold">AI</span>
          </div>
          <div>
            <p className="text-gray-300 leading-relaxed">{store.aiMessage}</p>
          </div>
        </div>
      </div>

      {/* Step content */}
      <div className="mb-6">
        {store.currentStep === 'strategy' && (
          <StrategyStep
            data={store.stepData}
            selected={store.selectedArchetype}
            onSelect={(arch) => store.setSelectedArchetype(arch)}
            onNext={() => advanceStep({ archetype: store.selectedArchetype })}
            isLoading={store.isLoading}
          />
        )}
        {store.currentStep === 'colors' && (
          <ColorsStep
            data={store.stepData}
            selected={store.selectedColors}
            onSelect={(colors) => store.setSelectedColors(colors)}
            onNext={() => advanceStep({ colors: store.selectedColors })}
            onBack={goBack}
            isLoading={store.isLoading}
          />
        )}
        {store.currentStep === 'core' && (
          <CardSelectionStep
            data={store.stepData}
            title="Core Cards"
            selectedCards={store.selectedCards}
            onUpdateCards={(cards) => store.setSelectedCards(cards)}
            onNext={() => advanceStep({ cards: store.selectedCards })}
            onBack={goBack}
            isLoading={store.isLoading}
          />
        )}
        {store.currentStep === 'support' && (
          <CardSelectionStep
            data={store.stepData}
            title="Support Cards"
            selectedCards={store.selectedCards}
            onUpdateCards={(cards) => store.setSelectedCards(cards)}
            onNext={() => advanceStep({ cards: store.selectedCards })}
            onBack={goBack}
            isLoading={store.isLoading}
          />
        )}
        {store.currentStep === 'mana_base' && (
          <ManaBaseStep
            data={store.stepData}
            selectedLands={store.selectedLands}
            onUpdateLands={(lands) => store.setSelectedLands(lands)}
            onNext={() => advanceStep({ lands: store.selectedLands })}
            onBack={goBack}
            isLoading={store.isLoading}
          />
        )}
        {store.currentStep === 'sideboard' && (
          <SideboardStep
            data={store.stepData}
            selectedCards={store.selectedSideboard}
            onUpdateCards={(cards) => store.setSelectedSideboard(cards)}
            onNext={() => advanceStep({ cards: store.selectedSideboard })}
            onBack={goBack}
            isLoading={store.isLoading}
          />
        )}
        {store.currentStep === 'review' && (
          <ReviewStep
            data={store.stepData}
            onBack={goBack}
            onComplete={completeBuild}
            isLoading={store.isLoading}
          />
        )}
      </div>
    </div>
  );
}

// --- Sub-components ---

function ProgressBar({
  currentIndex,
  totalSteps,
  currentStep,
}: {
  currentIndex: number;
  totalSteps: number;
  currentStep: GuidedBuildStep;
}) {
  const steps: GuidedBuildStep[] = ['strategy', 'colors', 'core', 'support', 'mana_base', 'sideboard', 'review'];
  const pct = ((currentIndex) / (totalSteps - 1)) * 100;

  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-xl font-bold text-white">
          {STEP_ICONS[currentStep]} {steps[currentIndex]?.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
        </h2>
        <span className="text-sm text-gray-400">
          Step {currentIndex + 1} of {totalSteps}
        </span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-primary-500 rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between mt-1">
        {steps.map((step, i) => (
          <div
            key={step}
            className={clsx(
              'w-2 h-2 rounded-full transition-colors',
              i <= currentIndex ? 'bg-primary-500' : 'bg-gray-700'
            )}
          />
        ))}
      </div>
    </div>
  );
}

function StrategyStep({
  data,
  selected,
  onSelect,
  onNext,
  isLoading,
}: {
  data: Record<string, any>;
  selected: string | null;
  onSelect: (archetype: string) => void;
  onNext: () => void;
  isLoading: boolean;
}) {
  const archetypes: ArchetypeOption[] = data.archetypes || [];

  return (
    <div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {archetypes.map((arch) => (
          <button
            key={arch.name}
            onClick={() => onSelect(arch.name)}
            className={clsx(
              'text-left p-5 rounded-xl border transition-all',
              selected === arch.name
                ? 'border-primary-500 bg-primary-500/10 ring-1 ring-primary-500/50'
                : 'border-gray-700 bg-gray-900 hover:border-gray-600'
            )}
          >
            <div className="flex items-start justify-between mb-2">
              <h3 className="text-lg font-semibold text-white">{arch.name}</h3>
              {arch.meta_percentage != null && arch.meta_percentage > 0 && (
                <span className="text-xs bg-gray-800 text-gray-400 px-2 py-1 rounded-full">
                  {arch.meta_percentage.toFixed(1)}% meta
                </span>
              )}
            </div>
            <p className="text-sm text-gray-400 mb-2">{arch.description}</p>
            <p className="text-xs text-gray-500 italic">{arch.playstyle}</p>
            {arch.example_cards.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {arch.example_cards.map((card) => (
                  <span
                    key={card}
                    className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded"
                  >
                    {card}
                  </span>
                ))}
              </div>
            )}
          </button>
        ))}
      </div>

      {data.meta_summary && (
        <p className="text-sm text-gray-500 mb-4">{data.meta_summary}</p>
      )}

      <div className="flex justify-end">
        <button
          onClick={onNext}
          disabled={!selected || isLoading}
          className="px-6 py-3 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-700 disabled:text-gray-500 text-white font-medium rounded-lg transition-colors"
        >
          {isLoading ? 'Loading...' : 'Next: Pick Colors'}
        </button>
      </div>
    </div>
  );
}

function ColorsStep({
  data,
  selected,
  onSelect,
  onNext,
  onBack,
  isLoading,
}: {
  data: Record<string, any>;
  selected: string[];
  onSelect: (colors: string[]) => void;
  onNext: () => void;
  onBack: () => void;
  isLoading: boolean;
}) {
  const options: ColorOption[] = data.options || [];

  return (
    <div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
        {options.map((opt) => {
          const isSelected =
            selected.length === opt.colors.length &&
            selected.every((c) => opt.colors.includes(c));
          return (
            <button
              key={opt.name}
              onClick={() => onSelect(opt.colors)}
              className={clsx(
                'text-left p-4 rounded-xl border transition-all',
                isSelected
                  ? 'border-primary-500 bg-primary-500/10 ring-1 ring-primary-500/50'
                  : 'border-gray-700 bg-gray-900 hover:border-gray-600'
              )}
            >
              <div className="flex items-center gap-2 mb-2">
                {opt.colors.map((c) => (
                  <span
                    key={c}
                    className={clsx(
                      'w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold',
                      COLOR_SYMBOLS[c]?.bg || 'bg-gray-600',
                      COLOR_SYMBOLS[c]?.text || 'text-white'
                    )}
                  >
                    {c}
                  </span>
                ))}
                <span className="text-white font-medium">{opt.name}</span>
              </div>
              <p className="text-xs text-gray-400 mb-2">{opt.description}</p>
              <div className="flex flex-wrap gap-1">
                {opt.strengths.map((s) => (
                  <span key={s} className="text-xs bg-green-900/30 text-green-400 px-1.5 py-0.5 rounded">
                    {s}
                  </span>
                ))}
              </div>
            </button>
          );
        })}
      </div>

      <StepNavigation
        onBack={onBack}
        onNext={onNext}
        nextDisabled={selected.length === 0 || isLoading}
        nextLabel={isLoading ? 'Loading...' : 'Next: Core Cards'}
      />
    </div>
  );
}

function CardSelectionStep({
  data,
  title,
  selectedCards,
  onUpdateCards,
  onNext,
  onBack,
  isLoading,
}: {
  data: Record<string, any>;
  title: string;
  selectedCards: Record<string, any>[];
  onUpdateCards: (cards: Record<string, any>[]) => void;
  onNext: () => void;
  onBack: () => void;
  isLoading: boolean;
}) {
  const slots: CardSlotGroup[] = data.slots || [];

  const toggleCard = (card: CardRecommendation) => {
    const existing = selectedCards.find((c) => c.card_name === card.card_name);
    if (existing) {
      onUpdateCards(selectedCards.filter((c) => c.card_name !== card.card_name));
    } else {
      onUpdateCards([
        ...selectedCards,
        {
          card_name: card.card_name,
          card_id: card.card_id,
          quantity: card.quantity,
          card: {
            type_line: card.type_line,
            mana_cost: card.mana_cost,
            image_uri: card.image_uri,
          },
        },
      ]);
    }
  };

  const updateQuantity = (cardName: string, qty: number) => {
    if (qty <= 0) {
      onUpdateCards(selectedCards.filter((c) => c.card_name !== cardName));
    } else {
      onUpdateCards(
        selectedCards.map((c) =>
          c.card_name === cardName ? { ...c, quantity: qty } : c
        )
      );
    }
  };

  const totalSelected = selectedCards.reduce((sum, c) => sum + (c.quantity || 1), 0);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
        <span className="text-sm text-gray-400">
          {totalSelected} cards selected
        </span>
      </div>

      {slots.map((slot) => (
        <div key={slot.slot_name} className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-md font-medium text-gray-300">{slot.slot_name}</h4>
            <span className="text-xs text-gray-500">
              Target: ~{slot.target_count} cards | {slot.description}
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {slot.recommendations.map((card) => {
              const isSelected = selectedCards.some(
                (c) => c.card_name === card.card_name
              );
              const selectedEntry = selectedCards.find(
                (c) => c.card_name === card.card_name
              );
              return (
                <div
                  key={card.card_name}
                  className={clsx(
                    'p-3 rounded-lg border transition-all cursor-pointer',
                    isSelected
                      ? 'border-primary-500 bg-primary-500/10'
                      : 'border-gray-700 bg-gray-900 hover:border-gray-600'
                  )}
                >
                  <div className="flex items-start justify-between" onClick={() => toggleCard(card)}>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-white truncate">
                        {card.card_name}
                      </p>
                      {card.mana_cost && (
                        <p className="text-xs text-gray-500">{card.mana_cost}</p>
                      )}
                      {card.type_line && (
                        <p className="text-xs text-gray-500 truncate">{card.type_line}</p>
                      )}
                    </div>
                    <div className={clsx(
                      'w-5 h-5 rounded border flex items-center justify-center flex-shrink-0 ml-2',
                      isSelected
                        ? 'bg-primary-500 border-primary-500 text-white'
                        : 'border-gray-600'
                    )}>
                      {isSelected && (
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1 line-clamp-2">{card.reasoning}</p>
                  {isSelected && (
                    <div className="flex items-center gap-2 mt-2">
                      <span className="text-xs text-gray-400">Qty:</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          updateQuantity(card.card_name, (selectedEntry?.quantity || 1) - 1);
                        }}
                        className="w-6 h-6 rounded bg-gray-800 text-gray-300 text-sm hover:bg-gray-700"
                      >
                        -
                      </button>
                      <span className="text-sm text-white w-4 text-center">
                        {selectedEntry?.quantity || 1}
                      </span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          updateQuantity(card.card_name, (selectedEntry?.quantity || 1) + 1);
                        }}
                        className="w-6 h-6 rounded bg-gray-800 text-gray-300 text-sm hover:bg-gray-700"
                      >
                        +
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}

      <StepNavigation
        onBack={onBack}
        onNext={onNext}
        nextDisabled={selectedCards.length === 0 || isLoading}
        nextLabel={isLoading ? 'Loading...' : 'Next'}
      />
    </div>
  );
}

function ManaBaseStep({
  data,
  selectedLands,
  onUpdateLands,
  onNext,
  onBack,
  isLoading,
}: {
  data: Record<string, any>;
  selectedLands: Record<string, any>[];
  onUpdateLands: (lands: Record<string, any>[]) => void;
  onNext: () => void;
  onBack: () => void;
  isLoading: boolean;
}) {
  const lands: LandRecommendation[] = data.lands || [];
  const totalLands = data.total_lands || 24;

  // Initialize with recommended lands on first render
  useEffect(() => {
    if (selectedLands.length === 0 && lands.length > 0) {
      onUpdateLands(
        lands.map((l) => ({
          card_name: l.card_name,
          card_id: l.card_id,
          quantity: l.quantity,
          card: { type_line: 'Land', image_uri: l.image_uri },
        }))
      );
    }
  }, [lands]);

  const updateLandQty = (cardName: string, qty: number) => {
    if (qty <= 0) {
      onUpdateLands(selectedLands.filter((l) => l.card_name !== cardName));
    } else {
      onUpdateLands(
        selectedLands.map((l) =>
          l.card_name === cardName ? { ...l, quantity: qty } : l
        )
      );
    }
  };

  const totalSelected = selectedLands.reduce((sum, l) => sum + (l.quantity || 0), 0);

  const categoryOrder = ['dual', 'fetch', 'utility', 'basic', 'other'];
  const grouped: Record<string, LandRecommendation[]> = {};
  for (const land of lands) {
    const cat = land.category || 'other';
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(land);
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Mana Base</h3>
        <span className={clsx(
          'text-sm',
          totalSelected === totalLands ? 'text-green-400' : 'text-yellow-400'
        )}>
          {totalSelected} / {totalLands} lands
        </span>
      </div>

      {data.mana_curve_note && (
        <p className="text-sm text-gray-400 mb-4">{data.mana_curve_note}</p>
      )}

      {categoryOrder.map((cat) => {
        const catLands = grouped[cat];
        if (!catLands || catLands.length === 0) return null;
        return (
          <div key={cat} className="mb-4">
            <h4 className="text-sm font-medium text-gray-400 uppercase tracking-wide mb-2">
              {cat === 'dual' ? 'Dual Lands' : cat === 'fetch' ? 'Fetch Lands' : cat === 'utility' ? 'Utility Lands' : cat === 'basic' ? 'Basic Lands' : 'Other'}
            </h4>
            <div className="space-y-2">
              {catLands.map((land) => {
                const selected = selectedLands.find((l) => l.card_name === land.card_name);
                const qty = selected?.quantity || 0;
                return (
                  <div
                    key={land.card_name}
                    className="flex items-center justify-between p-3 rounded-lg border border-gray-700 bg-gray-900"
                  >
                    <div>
                      <p className="text-sm font-medium text-white">{land.card_name}</p>
                      <p className="text-xs text-gray-500">{land.reasoning}</p>
                    </div>
                    <div className="flex items-center gap-2 ml-4">
                      <button
                        onClick={() => updateLandQty(land.card_name, qty - 1)}
                        className="w-7 h-7 rounded bg-gray-800 text-gray-300 hover:bg-gray-700 text-sm"
                      >
                        -
                      </button>
                      <span className="text-sm text-white w-6 text-center">{qty}</span>
                      <button
                        onClick={() => updateLandQty(land.card_name, qty + 1)}
                        className="w-7 h-7 rounded bg-gray-800 text-gray-300 hover:bg-gray-700 text-sm"
                      >
                        +
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      <StepNavigation
        onBack={onBack}
        onNext={onNext}
        nextDisabled={totalSelected === 0 || isLoading}
        nextLabel={isLoading ? 'Loading...' : 'Next: Sideboard'}
      />
    </div>
  );
}

function SideboardStep({
  data,
  selectedCards,
  onUpdateCards,
  onNext,
  onBack,
  isLoading,
}: {
  data: Record<string, any>;
  selectedCards: Record<string, any>[];
  onUpdateCards: (cards: Record<string, any>[]) => void;
  onNext: () => void;
  onBack: () => void;
  isLoading: boolean;
}) {
  const recommendations: SideboardRecommendation[] = data.recommendations || [];

  // Initialize from recommendations
  useEffect(() => {
    if (selectedCards.length === 0 && recommendations.length > 0) {
      onUpdateCards(
        recommendations.map((r) => ({
          card_name: r.card_name,
          card_id: r.card_id,
          quantity: r.quantity,
          card: { image_uri: r.image_uri },
        }))
      );
    }
  }, [recommendations]);

  const toggleCard = (rec: SideboardRecommendation) => {
    const existing = selectedCards.find((c) => c.card_name === rec.card_name);
    if (existing) {
      onUpdateCards(selectedCards.filter((c) => c.card_name !== rec.card_name));
    } else {
      onUpdateCards([
        ...selectedCards,
        {
          card_name: rec.card_name,
          card_id: rec.card_id,
          quantity: rec.quantity,
          card: { image_uri: rec.image_uri },
        },
      ]);
    }
  };

  const updateQty = (cardName: string, qty: number) => {
    if (qty <= 0) {
      onUpdateCards(selectedCards.filter((c) => c.card_name !== cardName));
    } else {
      onUpdateCards(
        selectedCards.map((c) =>
          c.card_name === cardName ? { ...c, quantity: qty } : c
        )
      );
    }
  };

  const totalSelected = selectedCards.reduce((sum, c) => sum + (c.quantity || 0), 0);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Sideboard</h3>
        <span className={clsx(
          'text-sm',
          totalSelected === 15 ? 'text-green-400' : totalSelected > 15 ? 'text-red-400' : 'text-yellow-400'
        )}>
          {totalSelected} / 15 cards
        </span>
      </div>

      {data.sideboard_strategy && (
        <p className="text-sm text-gray-400 mb-4">{data.sideboard_strategy}</p>
      )}

      <div className="space-y-2 mb-6">
        {recommendations.map((rec) => {
          const isSelected = selectedCards.some((c) => c.card_name === rec.card_name);
          const selected = selectedCards.find((c) => c.card_name === rec.card_name);
          return (
            <div
              key={rec.card_name}
              className={clsx(
                'p-3 rounded-lg border transition-all',
                isSelected
                  ? 'border-primary-500 bg-primary-500/10'
                  : 'border-gray-700 bg-gray-900'
              )}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 cursor-pointer" onClick={() => toggleCard(rec)}>
                  <div className="flex items-center gap-2">
                    <div className={clsx(
                      'w-5 h-5 rounded border flex items-center justify-center',
                      isSelected
                        ? 'bg-primary-500 border-primary-500 text-white'
                        : 'border-gray-600'
                    )}>
                      {isSelected && (
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                    </div>
                    <span className="text-sm font-medium text-white">{rec.card_name}</span>
                  </div>
                  <div className="flex flex-wrap gap-1 mt-1 ml-7">
                    {rec.target_matchups.map((m) => (
                      <span key={m} className="text-xs bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded">
                        vs {m}
                      </span>
                    ))}
                  </div>
                  <p className="text-xs text-gray-500 mt-1 ml-7">{rec.reasoning}</p>
                </div>
                {isSelected && (
                  <div className="flex items-center gap-2 ml-4">
                    <button
                      onClick={() => updateQty(rec.card_name, (selected?.quantity || 1) - 1)}
                      className="w-6 h-6 rounded bg-gray-800 text-gray-300 text-sm hover:bg-gray-700"
                    >
                      -
                    </button>
                    <span className="text-sm text-white w-4 text-center">
                      {selected?.quantity || 1}
                    </span>
                    <button
                      onClick={() => updateQty(rec.card_name, (selected?.quantity || 1) + 1)}
                      className="w-6 h-6 rounded bg-gray-800 text-gray-300 text-sm hover:bg-gray-700"
                    >
                      +
                    </button>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <StepNavigation
        onBack={onBack}
        onNext={onNext}
        nextDisabled={isLoading}
        nextLabel={isLoading ? 'Loading...' : 'Next: Review'}
      />
    </div>
  );
}

function ReviewStep({
  data,
  onBack,
  onComplete,
  isLoading,
}: {
  data: Record<string, any>;
  onBack: () => void;
  onComplete: (name: string, save: boolean) => void;
  isLoading: boolean;
}) {
  const [deckName, setDeckName] = useState(data.deck_name || 'My Deck');
  const mainDeck: Record<string, any>[] = data.main_deck || [];
  const sideboard: Record<string, any>[] = data.sideboard || [];
  const mainCount = data.main_deck_count || 0;
  const sbCount = data.sideboard_count || 0;
  const errors: string[] = data.validation_errors || [];
  const strengths: string[] = data.strengths || [];
  const weaknesses: string[] = data.weaknesses || [];

  // Group main deck by type
  const lands = mainDeck.filter((e) => {
    const tl = e.card?.type_line || '';
    return tl.toLowerCase().includes('land') ||
      ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest'].includes(e.card_name);
  });
  const nonlands = mainDeck.filter((e) => !lands.includes(e));

  return (
    <div>
      {/* Deck name */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-400 mb-1">Deck Name</label>
        <input
          type="text"
          value={deckName}
          onChange={(e) => setDeckName(e.target.value)}
          className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-primary-500"
        />
      </div>

      {/* Validation */}
      {errors.length > 0 && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 mb-4">
          <h4 className="text-sm font-medium text-red-400 mb-2">Validation Issues</h4>
          <ul className="text-xs text-red-300 space-y-1">
            {errors.map((e, i) => (
              <li key={i}>- {e}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Strengths / Weaknesses */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        {strengths.length > 0 && (
          <div className="bg-green-900/10 border border-green-800/30 rounded-lg p-4">
            <h4 className="text-sm font-medium text-green-400 mb-2">Strengths</h4>
            <ul className="text-xs text-green-300 space-y-1">
              {strengths.map((s, i) => (
                <li key={i}>+ {s}</li>
              ))}
            </ul>
          </div>
        )}
        {weaknesses.length > 0 && (
          <div className="bg-yellow-900/10 border border-yellow-800/30 rounded-lg p-4">
            <h4 className="text-sm font-medium text-yellow-400 mb-2">Weaknesses</h4>
            <ul className="text-xs text-yellow-300 space-y-1">
              {weaknesses.map((w, i) => (
                <li key={i}>- {w}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Deck lists */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-300 mb-3">
            Main Deck ({mainCount})
          </h4>

          {nonlands.length > 0 && (
            <>
              <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Spells</p>
              <ul className="space-y-0.5 mb-3">
                {nonlands.map((e) => (
                  <li key={e.card_name} className="text-sm text-gray-300 flex justify-between">
                    <span>{e.card_name}</span>
                    <span className="text-gray-500">x{e.quantity}</span>
                  </li>
                ))}
              </ul>
            </>
          )}

          {lands.length > 0 && (
            <>
              <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Lands</p>
              <ul className="space-y-0.5">
                {lands.map((e) => (
                  <li key={e.card_name} className="text-sm text-gray-300 flex justify-between">
                    <span>{e.card_name}</span>
                    <span className="text-gray-500">x{e.quantity}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-300 mb-3">
            Sideboard ({sbCount})
          </h4>
          <ul className="space-y-0.5">
            {sideboard.map((e) => (
              <li key={e.card_name} className="text-sm text-gray-300 flex justify-between">
                <span>{e.card_name}</span>
                <span className="text-gray-500">x{e.quantity}</span>
              </li>
            ))}
          </ul>
          {sideboard.length === 0 && (
            <p className="text-sm text-gray-600 italic">No sideboard cards selected.</p>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="px-5 py-2.5 text-gray-400 hover:text-white transition-colors"
        >
          Back
        </button>
        <div className="flex gap-3">
          <button
            onClick={() => onComplete(deckName, false)}
            disabled={isLoading}
            className="px-5 py-2.5 border border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 rounded-lg transition-colors"
          >
            Export Without Saving
          </button>
          <button
            onClick={() => onComplete(deckName, true)}
            disabled={isLoading}
            className="px-6 py-2.5 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-700 text-white font-medium rounded-lg transition-colors"
          >
            {isLoading ? 'Saving...' : 'Save Deck'}
          </button>
        </div>
      </div>
    </div>
  );
}

function StepNavigation({
  onBack,
  onNext,
  nextDisabled,
  nextLabel,
}: {
  onBack: () => void;
  onNext: () => void;
  nextDisabled: boolean;
  nextLabel: string;
}) {
  return (
    <div className="flex items-center justify-between pt-4 border-t border-gray-800">
      <button
        onClick={onBack}
        className="px-5 py-2.5 text-gray-400 hover:text-white transition-colors"
      >
        Back
      </button>
      <button
        onClick={onNext}
        disabled={nextDisabled}
        className="px-6 py-3 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-700 disabled:text-gray-500 text-white font-medium rounded-lg transition-colors"
      >
        {nextLabel}
      </button>
    </div>
  );
}
