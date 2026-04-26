/**
 * Canonical theme definitions — mirrors livetranslate-common theme.py.
 * Keep in sync with modules/shared/src/livetranslate_common/theme.py
 *
 * Editorial Riso palette (D1.3) — warm earth tones that pair with the
 * Earwyrm peach/purple brand. Index alignment matters: the i-th speaker
 * in any session gets SPEAKER_COLORS[i % length] in BOTH the dashboard
 * and the bot's PIL renderer, so the two surfaces look like the same
 * publication.
 */

export type DisplayMode = 'split' | 'subtitle' | 'transcript' | 'interpreter';

export const SPEAKER_COLORS = [
  '#C26F49',  // terracotta
  '#C8893E',  // ochre
  '#7A8C5C',  // sage
  '#6B4A6B',  // plum
  '#5F6B85',  // slate-blue
  '#A0392E',  // oxblood
  '#B89B45',  // mustard
  '#5B7048',  // fern
  '#A37388',  // dusk
  '#4D5A8C',  // indigo
] as const;

export const SPEAKER_COLOR_NAMES = [
  'terracotta', 'ochre', 'sage', 'plum', 'slate-blue',
  'oxblood', 'mustard', 'fern', 'dusk', 'indigo',
] as const;
