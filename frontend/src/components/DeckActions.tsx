import { useState } from 'react';
import { Deck } from '@/types';
import { decksApi } from '@/services/api';
import { SideboardMatrix } from './SideboardMatrix';
import toast from 'react-hot-toast';
import clsx from 'clsx';

interface DeckActionsProps {
  deck: Partial<Deck>;
  onSave?: () => void;
  onExport?: (text: string, format: string) => void;
  className?: string;
}

export function DeckActions({ deck, onSave, onExport, className }: DeckActionsProps) {
  const [isSaving, setIsSaving] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportFormat, setExportFormat] = useState<'arena' | 'mtgo' | 'plain'>('arena');
  const [showExportModal, setShowExportModal] = useState(false);
  const [showSideboardMatrix, setShowSideboardMatrix] = useState(false);

  const handleSave = async () => {
    if (!deck.main_deck?.length) {
      toast.error('Deck is empty');
      return;
    }

    setIsSaving(true);
    try {
      if (deck.id) {
        await decksApi.update(deck.id, {
          name: deck.name,
          description: deck.description,
          main_deck: deck.main_deck,
          sideboard: deck.sideboard,
        });
        toast.success('Deck saved');
      } else {
        // Try to create with original name, retry with timestamp if conflict
        let deckName = deck.name || 'Untitled Deck';
        try {
          await decksApi.create({
            name: deckName,
            description: deck.description,
            main_deck: deck.main_deck,
            sideboard: deck.sideboard,
            format: deck.format || 'standard',
            archetype: deck.archetype,
          });
          toast.success('Deck created');
        } catch (createError: unknown) {
          const axiosError = createError as { response?: { status?: number } };
          if (axiosError.response?.status === 409) {
            // Name conflict - retry with timestamp
            const timestamp = new Date().toLocaleString('en-US', {
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
            });
            deckName = `${deck.name || 'Untitled Deck'} (${timestamp})`;
            await decksApi.create({
              name: deckName,
              description: deck.description,
              main_deck: deck.main_deck,
              sideboard: deck.sideboard,
              format: deck.format || 'standard',
              archetype: deck.archetype,
            });
            toast.success(`Deck saved as "${deckName}"`);
          } else {
            throw createError;
          }
        }
      }
      onSave?.();
    } catch (error) {
      console.error('Save error:', error);
      toast.error('Failed to save deck');
    } finally {
      setIsSaving(false);
    }
  };

  const formatArenaEntry = (entry: { card_name: string; quantity: number; set_code?: string; collector_number?: string }) => {
    // Arena format: "4 Lightning Strike (DMU) 137"
    if (entry.set_code && entry.collector_number) {
      return `${entry.quantity} ${entry.card_name} (${entry.set_code.toUpperCase()}) ${entry.collector_number}`;
    }
    return `${entry.quantity} ${entry.card_name}`;
  };

  const handleExport = async () => {
    if (!deck.main_deck?.length) {
      toast.error('Deck is empty');
      return;
    }

    setIsExporting(true);
    try {
      // Build export text locally
      const lines: string[] = [];

      if (deck.name && exportFormat === 'plain') {
        lines.push(`// ${deck.name}`);
        lines.push('');
      }

      // Main deck
      for (const entry of deck.main_deck) {
        if (exportFormat === 'arena') {
          lines.push(formatArenaEntry(entry));
        } else if (exportFormat === 'mtgo') {
          lines.push(`${entry.quantity} ${entry.card_name}`);
        } else {
          lines.push(`${entry.quantity} ${entry.card_name}`);
        }
      }

      // Sideboard
      if (deck.sideboard?.length) {
        lines.push('');
        if (exportFormat === 'arena') {
          lines.push('Sideboard');
        }
        for (const entry of deck.sideboard) {
          if (exportFormat === 'arena') {
            lines.push(formatArenaEntry(entry));
          } else if (exportFormat === 'mtgo') {
            lines.push(`SB: ${entry.quantity} ${entry.card_name}`);
          } else {
            lines.push(`${entry.quantity} ${entry.card_name}`);
          }
        }
      }

      const exportText = lines.join('\n');

      // For MTGO, trigger file download
      if (exportFormat === 'mtgo') {
        const blob = new Blob([exportText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${deck.name || 'deck'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toast.success('Deck downloaded as .txt file');
      } else {
        // Copy to clipboard for Arena and plain text
        await navigator.clipboard.writeText(exportText);
        toast.success(`Deck copied to clipboard (${exportFormat.toUpperCase()} format)`);
      }

      onExport?.(exportText, exportFormat);
      setShowExportModal(false);
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Failed to export deck');
    } finally {
      setIsExporting(false);
    }
  };

  const handleShare = async () => {
    if (!deck.id || !deck.share_token) {
      toast.error('Save the deck first to share');
      return;
    }

    const shareUrl = `${window.location.origin}/deck/shared/${deck.share_token}`;

    try {
      await navigator.clipboard.writeText(shareUrl);
      toast.success('Share link copied to clipboard');
    } catch {
      toast.error('Failed to copy share link');
    }
  };

  return (
    <div className={clsx('flex flex-wrap gap-2', className)}>
      <button
        onClick={handleSave}
        disabled={isSaving || !deck.main_deck?.length}
        className={clsx(
          'px-4 py-2 rounded-lg font-medium transition-colors',
          deck.main_deck?.length && !isSaving
            ? 'bg-green-600 hover:bg-green-700 text-white'
            : 'bg-gray-700 text-gray-400 cursor-not-allowed'
        )}
      >
        {isSaving ? 'Saving...' : deck.id ? 'Save Changes' : 'Save Deck'}
      </button>

      <button
        onClick={() => setShowExportModal(true)}
        disabled={!deck.main_deck?.length}
        className={clsx(
          'px-4 py-2 rounded-lg font-medium transition-colors',
          deck.main_deck?.length
            ? 'bg-blue-600 hover:bg-blue-700 text-white'
            : 'bg-gray-700 text-gray-400 cursor-not-allowed'
        )}
      >
        Export
      </button>

      {deck.id && deck.visibility !== 'private' && (
        <button
          onClick={handleShare}
          className="px-4 py-2 rounded-lg font-medium bg-purple-600 hover:bg-purple-700 text-white transition-colors"
        >
          Share
        </button>
      )}

      {deck.id && deck.sideboard?.length ? (
        <button
          onClick={() => setShowSideboardMatrix(true)}
          className="px-4 py-2 rounded-lg font-medium bg-amber-600 hover:bg-amber-700 text-white transition-colors"
          title="Generate sideboard guide for all matchups"
        >
          Sideboard Guide
        </button>
      ) : null}

      {/* Sideboard Matrix Modal */}
      {showSideboardMatrix && (
        <SideboardMatrix
          deck={deck}
          onClose={() => setShowSideboardMatrix(false)}
        />
      )}

      {/* Export Modal */}
      {showExportModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold text-white mb-4">Export Deck</h3>

            <div className="space-y-3 mb-6">
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="radio"
                  name="exportFormat"
                  checked={exportFormat === 'arena'}
                  onChange={() => setExportFormat('arena')}
                  className="text-primary-600"
                />
                <span className="text-white">MTG Arena</span>
              </label>

              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="radio"
                  name="exportFormat"
                  checked={exportFormat === 'mtgo'}
                  onChange={() => setExportFormat('mtgo')}
                  className="text-primary-600"
                />
                <span className="text-white">MTGO</span>
              </label>

              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="radio"
                  name="exportFormat"
                  checked={exportFormat === 'plain'}
                  onChange={() => setExportFormat('plain')}
                  className="text-primary-600"
                />
                <span className="text-white">Plain Text</span>
              </label>
            </div>

            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowExportModal(false)}
                className="px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleExport}
                disabled={isExporting}
                className="px-4 py-2 rounded-lg bg-primary-600 hover:bg-primary-700 text-white transition-colors"
              >
                {isExporting ? 'Exporting...' : exportFormat === 'mtgo' ? 'Download .txt' : 'Copy to Clipboard'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
