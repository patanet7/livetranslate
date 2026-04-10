/**
 * Command parser — stateless pure function.
 *
 * Parses /slash commands from Google Meet chat messages.
 * Returns typed CommandResult or null for non-commands.
 */

export type CommandResult =
  | { type: 'set_language'; changes: { source_lang: string; target_lang: string } }
  | { type: 'set_config'; changes: Record<string, string | number | boolean> }
  | { type: 'adjust_font'; delta: number }
  | { type: 'query'; query: 'status' | 'help' }
  | { type: 'demo'; mode: string }
  | { type: 'unknown'; raw: string };

const VALID_MODES = new Set(['subtitle', 'split', 'interpreter']);
const VALID_THEMES: Record<string, string> = {
  dark: 'dark',
  light: 'light',
  contrast: 'high_contrast',
  high_contrast: 'high_contrast',
  minimal: 'minimal',
  corporate: 'corporate',
};
const VALID_SOURCES: Record<string, string> = {
  bot: 'bot_audio',
  fireflies: 'fireflies',
};

function parseToggle(value: string): boolean | null {
  if (value === 'on' || value === 'true' || value === 'yes') return true;
  if (value === 'off' || value === 'false' || value === 'no') return false;
  return null;
}

export function parseCommand(text: string): CommandResult | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith('/')) return null;

  const parts = trimmed.split(/\s+/);
  const cmd = parts[0].toLowerCase();
  const arg = parts[1]?.toLowerCase() ?? '';

  switch (cmd) {
    case '/lang': {
      if (!arg) return { type: 'unknown', raw: trimmed };
      if (arg.includes('-')) {
        const [source, target] = arg.split('-', 2);
        return { type: 'set_language', changes: { source_lang: source, target_lang: target } };
      }
      return { type: 'set_language', changes: { source_lang: 'auto', target_lang: arg } };
    }

    case '/font': {
      if (arg === 'up') return { type: 'adjust_font', delta: 4 };
      if (arg === 'down') return { type: 'adjust_font', delta: -4 };
      const size = parseInt(arg, 10);
      if (!isNaN(size) && size > 0) return { type: 'set_config', changes: { font_size: size } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/mode': {
      if (VALID_MODES.has(arg)) return { type: 'set_config', changes: { display_mode: arg } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/theme': {
      const theme = VALID_THEMES[arg];
      if (theme) return { type: 'set_config', changes: { theme } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/speakers': {
      const val = parseToggle(arg);
      if (val !== null) return { type: 'set_config', changes: { show_speakers: val } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/original': {
      const val = parseToggle(arg);
      if (val !== null) return { type: 'set_config', changes: { show_original: val } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/translate': {
      const val = parseToggle(arg);
      if (val !== null) return { type: 'set_config', changes: { translation_enabled: val } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/source': {
      const source = VALID_SOURCES[arg];
      if (source) return { type: 'set_config', changes: { caption_source: source } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/status':
      return { type: 'query', query: 'status' };

    case '/help':
      return { type: 'query', query: 'help' };

    case '/demo': {
      const mode = arg || 'replay';
      if (['replay', 'fireflies', 'passthrough', 'pretranslated', 'stop'].includes(mode)) {
        return { type: 'demo', mode: mode === 'fireflies' ? 'replay' : mode };
      }
      return { type: 'unknown', raw: trimmed };
    }

    default:
      return { type: 'unknown', raw: trimmed };
  }
}
