/**
 * Canonical theme definitions — mirrors livetranslate-common theme.py.
 * Keep in sync with modules/shared/src/livetranslate_common/theme.py
 */

export type DisplayMode = 'split' | 'subtitle' | 'transcript' | 'interpreter';

export const SPEAKER_COLORS = [
  '#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336',
  '#00BCD4', '#E91E63', '#FFEB3B', '#795548', '#607D8B',
] as const;
