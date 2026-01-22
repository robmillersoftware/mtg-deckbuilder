import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { decksApi } from '@/services/api';
import { ValidationError, DeckEntry } from '@/types';
import toast from 'react-hot-toast';

type ImportFormat = 'arena' | 'mtgo' | 'text';

interface ImportResult {
  success: boolean;
  deck?: {
    id: string;
    name: string;
    main_deck: DeckEntry[];
    sideboard: DeckEntry[];
  };
  errors?: {
    line_number?: number;
    card_name?: string;
    message: string;
    suggestions?: string[];
  }[];
  warnings?: string[];
  validation_errors?: ValidationError[];
}

export function DeckImportPage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [deckText, setDeckText] = useState('');
  const [format, setFormat] = useState<ImportFormat>('arena');
  const [deckName, setDeckName] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  const [importResult, setImportResult] = useState<ImportResult | null>(null);

  const handleImport = async () => {
    if (!deckText.trim()) {
      toast.error('Please enter a decklist');
      return;
    }

    setIsImporting(true);
    setImportResult(null);

    try {
      const response = await decksApi.import(deckText, format);
      const result = response.data as ImportResult;
      setImportResult(result);

      if (result.success && result.deck) {
        toast.success('Deck imported successfully');
      } else if (result.errors && result.errors.length > 0) {
        toast.error(`Import had ${result.errors.length} errors`);
      }
    } catch (error: unknown) {
      console.error('Import failed:', error);
      const message = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Import failed';
      toast.error(message);
    } finally {
      setIsImporting(false);
    }
  };

  const handleSaveDeck = async () => {
    if (!importResult?.deck) return;

    if (!deckName.trim()) {
      toast.error('Please enter a deck name');
      return;
    }

    try {
      const response = await decksApi.create({
        name: deckName,
        main_deck: importResult.deck.main_deck.map((e) => ({
          card_name: e.card_name,
          quantity: e.quantity,
        })),
        sideboard: importResult.deck.sideboard?.map((e) => ({
          card_name: e.card_name,
          quantity: e.quantity,
        })),
        format: 'standard',
      });
      toast.success('Deck saved successfully');
      navigate(`/deck/${response.data.id}`);
    } catch (error: unknown) {
      console.error('Save failed:', error);
      const message = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Failed to save deck';
      toast.error(message);
    }
  };

  const applySuggestion = (errorIndex: number, suggestion: string) => {
    if (!importResult?.errors) return;

    const error = importResult.errors[errorIndex];
    if (!error.card_name) return;

    // Replace the card name in the deck text
    const updatedText = deckText.replace(
      new RegExp(`(\\d+\\s+)${escapeRegExp(error.card_name)}`, 'gi'),
      `$1${suggestion}`
    );
    setDeckText(updatedText);
    toast.success(`Replaced "${error.card_name}" with "${suggestion}"`);
  };

  const escapeRegExp = (string: string) => {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  };

  const formatExamples: Record<ImportFormat, string> = {
    arena: `4 Lightning Strike (DMU) 137
4 Play with Fire (MID) 154
4 Monastery Swiftspear (BRO) 144
20 Mountain (SNC) 275

Sideboard
3 Rending Flame (VOW) 151
2 Abrade (VOW) 139`,
    mtgo: `4 Lightning Strike
4 Play with Fire
4 Monastery Swiftspear
20 Mountain

SB: 3 Rending Flame
SB: 2 Abrade`,
    text: `4 Lightning Strike
4 Play with Fire
4 Monastery Swiftspear
20 Mountain

3 Rending Flame
2 Abrade`,
  };

  const loadExample = () => {
    setDeckText(formatExamples[format]);
  };

  if (!user) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Please log in to import decks</div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-6">Import Deck</h1>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Import Form */}
        <div className="space-y-4">
          {/* Format Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Import Format
            </label>
            <div className="flex gap-2">
              {(['arena', 'mtgo', 'text'] as ImportFormat[]).map((f) => (
                <button
                  key={f}
                  onClick={() => setFormat(f)}
                  className={`px-4 py-2 text-sm rounded-lg transition-colors ${
                    format === f
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  {f === 'arena' ? 'Arena' : f === 'mtgo' ? 'MTGO' : 'Plain Text'}
                </button>
              ))}
            </div>
          </div>

          {/* Deck Text Area */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-300">
                Decklist
              </label>
              <button
                onClick={loadExample}
                className="text-xs text-primary-400 hover:text-primary-300"
              >
                Load Example
              </button>
            </div>
            <textarea
              value={deckText}
              onChange={(e) => setDeckText(e.target.value)}
              placeholder={`Paste your ${format === 'arena' ? 'Arena' : format === 'mtgo' ? 'MTGO' : ''} decklist here...`}
              className="w-full h-80 bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white font-mono text-sm focus:border-primary-500 focus:ring-1 focus:ring-primary-500 resize-none"
            />
          </div>

          {/* Import Button */}
          <button
            onClick={handleImport}
            disabled={isImporting || !deckText.trim()}
            className="w-full px-4 py-3 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
          >
            {isImporting ? 'Importing...' : 'Import Deck'}
          </button>

          {/* Format Help */}
          <div className="text-sm text-gray-400 bg-gray-800/50 rounded-lg p-4">
            <h4 className="font-medium text-gray-300 mb-2">Format Tips:</h4>
            <ul className="space-y-1 text-xs">
              {format === 'arena' && (
                <>
                  <li>Copy directly from Arena collection</li>
                  <li>Format: "4 Lightning Strike (DMU) 137"</li>
                  <li>Sideboard after blank line with "Sideboard" header</li>
                </>
              )}
              {format === 'mtgo' && (
                <>
                  <li>Standard MTGO export format</li>
                  <li>Format: "4 Lightning Strike"</li>
                  <li>Sideboard lines start with "SB:"</li>
                </>
              )}
              {format === 'text' && (
                <>
                  <li>Simple quantity + card name</li>
                  <li>Format: "4 Lightning Strike"</li>
                  <li>Sideboard after blank line</li>
                </>
              )}
            </ul>
          </div>
        </div>

        {/* Import Results */}
        <div className="space-y-4">
          {importResult ? (
            <>
              {/* Success - Show Deck */}
              {importResult.success && importResult.deck && (
                <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
                  <div className="flex items-center gap-2 mb-4">
                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <h3 className="text-lg font-semibold text-white">Import Successful</h3>
                  </div>

                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Deck Name
                    </label>
                    <input
                      type="text"
                      value={deckName}
                      onChange={(e) => setDeckName(e.target.value)}
                      placeholder="Enter deck name"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
                    />
                  </div>

                  <div className="mb-4 space-y-2 text-sm">
                    <div className="flex justify-between text-gray-400">
                      <span>Main Deck:</span>
                      <span className="text-white">
                        {importResult.deck.main_deck.reduce((sum, e) => sum + e.quantity, 0)} cards
                      </span>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>Sideboard:</span>
                      <span className="text-white">
                        {(importResult.deck.sideboard || []).reduce((sum, e) => sum + e.quantity, 0)} cards
                      </span>
                    </div>
                  </div>

                  {/* Validation Errors */}
                  {importResult.validation_errors && importResult.validation_errors.length > 0 && (
                    <div className="mb-4 bg-yellow-900/20 border border-yellow-800 rounded-lg p-3">
                      <h4 className="text-sm font-medium text-yellow-400 mb-2">Validation Warnings</h4>
                      <ul className="text-xs text-yellow-300 space-y-1">
                        {importResult.validation_errors.map((error, i) => (
                          <li key={i}>{error.message}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <button
                    onClick={handleSaveDeck}
                    className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors"
                  >
                    Save Deck
                  </button>
                </div>
              )}

              {/* Errors */}
              {importResult.errors && importResult.errors.length > 0 && (
                <div className="bg-gray-900 rounded-lg p-6 border border-red-800">
                  <div className="flex items-center gap-2 mb-4">
                    <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <h3 className="text-lg font-semibold text-white">Import Errors</h3>
                  </div>

                  <div className="space-y-3 max-h-80 overflow-y-auto">
                    {importResult.errors.map((error, i) => (
                      <div key={i} className="bg-gray-800 rounded-lg p-3">
                        <div className="flex items-start justify-between">
                          <div>
                            {error.line_number && (
                              <span className="text-xs text-gray-500">Line {error.line_number}: </span>
                            )}
                            <span className="text-sm text-red-400">{error.message}</span>
                            {error.card_name && (
                              <span className="block text-xs text-gray-400 mt-1">
                                Card: {error.card_name}
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Suggestions */}
                        {error.suggestions && error.suggestions.length > 0 && (
                          <div className="mt-2">
                            <span className="text-xs text-gray-400">Did you mean:</span>
                            <div className="flex flex-wrap gap-2 mt-1">
                              {error.suggestions.map((suggestion, si) => (
                                <button
                                  key={si}
                                  onClick={() => applySuggestion(i, suggestion)}
                                  className="px-2 py-1 text-xs bg-primary-900/50 text-primary-300 hover:bg-primary-900 rounded transition-colors"
                                >
                                  {suggestion}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Warnings */}
              {importResult.warnings && importResult.warnings.length > 0 && (
                <div className="bg-yellow-900/20 border border-yellow-800 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-yellow-400 mb-2">Warnings</h4>
                  <ul className="text-xs text-yellow-300 space-y-1">
                    {importResult.warnings.map((warning, i) => (
                      <li key={i}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800 flex items-center justify-center h-full min-h-[300px]">
              <p className="text-gray-500">Import results will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
