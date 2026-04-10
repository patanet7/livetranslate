import { describe, it, expect } from 'vitest';
import { parseCommand, CommandResult } from '../../src/chat/command_parser';

describe('parseCommand', () => {
  // --- Language commands ---
  it('parses /lang with single target', () => {
    const result = parseCommand('/lang zh');
    expect(result).toEqual({
      type: 'set_language',
      changes: { source_lang: 'auto', target_lang: 'zh' },
    });
  });

  it('parses /lang with explicit pair', () => {
    const result = parseCommand('/lang zh-en');
    expect(result).toEqual({
      type: 'set_language',
      changes: { source_lang: 'zh', target_lang: 'en' },
    });
  });

  // --- Font commands ---
  it('parses /font up', () => {
    const result = parseCommand('/font up');
    expect(result).toEqual({ type: 'adjust_font', delta: 4 });
  });

  it('parses /font down', () => {
    const result = parseCommand('/font down');
    expect(result).toEqual({ type: 'adjust_font', delta: -4 });
  });

  it('parses /font with exact size', () => {
    const result = parseCommand('/font 32');
    expect(result).toEqual({ type: 'set_config', changes: { font_size: 32 } });
  });

  // --- Display mode ---
  it('parses /mode subtitle', () => {
    const result = parseCommand('/mode subtitle');
    expect(result).toEqual({ type: 'set_config', changes: { display_mode: 'subtitle' } });
  });

  it('parses /mode split', () => {
    const result = parseCommand('/mode split');
    expect(result).toEqual({ type: 'set_config', changes: { display_mode: 'split' } });
  });

  it('parses /mode interpreter', () => {
    const result = parseCommand('/mode interpreter');
    expect(result).toEqual({ type: 'set_config', changes: { display_mode: 'interpreter' } });
  });

  // --- Theme ---
  it('parses /theme dark', () => {
    const result = parseCommand('/theme dark');
    expect(result).toEqual({ type: 'set_config', changes: { theme: 'dark' } });
  });

  it('parses /theme contrast as high_contrast', () => {
    const result = parseCommand('/theme contrast');
    expect(result).toEqual({ type: 'set_config', changes: { theme: 'high_contrast' } });
  });

  // --- Toggle commands ---
  it('parses /speakers on', () => {
    const result = parseCommand('/speakers on');
    expect(result).toEqual({ type: 'set_config', changes: { show_speakers: true } });
  });

  it('parses /speakers off', () => {
    const result = parseCommand('/speakers off');
    expect(result).toEqual({ type: 'set_config', changes: { show_speakers: false } });
  });

  it('parses /original on', () => {
    const result = parseCommand('/original on');
    expect(result).toEqual({ type: 'set_config', changes: { show_original: true } });
  });

  it('parses /translate off', () => {
    const result = parseCommand('/translate off');
    expect(result).toEqual({ type: 'set_config', changes: { translation_enabled: false } });
  });

  // --- Source ---
  it('parses /source bot', () => {
    const result = parseCommand('/source bot');
    expect(result).toEqual({ type: 'set_config', changes: { caption_source: 'bot_audio' } });
  });

  it('parses /source fireflies', () => {
    const result = parseCommand('/source fireflies');
    expect(result).toEqual({ type: 'set_config', changes: { caption_source: 'fireflies' } });
  });

  // --- Info commands ---
  it('parses /status', () => {
    const result = parseCommand('/status');
    expect(result).toEqual({ type: 'query', query: 'status' });
  });

  it('parses /help', () => {
    const result = parseCommand('/help');
    expect(result).toEqual({ type: 'query', query: 'help' });
  });

  // --- Non-commands ---
  it('returns null for non-command text', () => {
    expect(parseCommand('hello everyone')).toBeNull();
  });

  it('returns null for empty string', () => {
    expect(parseCommand('')).toBeNull();
  });

  it('returns unknown for unrecognized command', () => {
    const result = parseCommand('/unknown arg');
    expect(result).toEqual({ type: 'unknown', raw: '/unknown arg' });
  });

  // --- Edge cases ---
  it('handles extra whitespace', () => {
    const result = parseCommand('  /lang  zh  ');
    expect(result).toEqual({
      type: 'set_language',
      changes: { source_lang: 'auto', target_lang: 'zh' },
    });
  });

  it('is case insensitive for commands', () => {
    const result = parseCommand('/LANG zh');
    expect(result).toEqual({
      type: 'set_language',
      changes: { source_lang: 'auto', target_lang: 'zh' },
    });
  });

  it('rejects /font with invalid number', () => {
    const result = parseCommand('/font abc');
    expect(result).toEqual({ type: 'unknown', raw: '/font abc' });
  });

  it('rejects /mode with invalid mode', () => {
    const result = parseCommand('/mode invalid');
    expect(result).toEqual({ type: 'unknown', raw: '/mode invalid' });
  });
});
