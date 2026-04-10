import { describe, it, expect } from 'vitest';
import { parseCaptionText } from '../../src/chat/meet_captions_adapter';

describe('parseCaptionText', () => {
  it('parses speaker and text from "Speaker: text" format', () => {
    const result = parseCaptionText('Alice: Hello everyone');
    expect(result).toEqual({ speaker: 'Alice', text: 'Hello everyone' });
  });

  it('handles text without speaker prefix', () => {
    const result = parseCaptionText('Hello everyone');
    expect(result).toEqual({ speaker: 'Unknown', text: 'Hello everyone' });
  });

  it('handles speaker name with colon in text', () => {
    const result = parseCaptionText('Bob: Time is: 3pm');
    expect(result).toEqual({ speaker: 'Bob', text: 'Time is: 3pm' });
  });

  it('trims whitespace', () => {
    const result = parseCaptionText('  Alice :  Hello  ');
    expect(result).toEqual({ speaker: 'Alice', text: 'Hello' });
  });

  it('returns null for empty string', () => {
    expect(parseCaptionText('')).toBeNull();
  });

  it('returns null for whitespace-only', () => {
    expect(parseCaptionText('   ')).toBeNull();
  });
});
