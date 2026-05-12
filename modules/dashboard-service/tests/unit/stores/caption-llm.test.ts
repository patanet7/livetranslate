/**
 * LLM overrides slice of the caption store.
 *
 * Verifies the per-session sampling tunables that flow from the Toolbar
 * through the WebSocket as ConfigMessage.llm: defaults are null, updates
 * are patches (no full replacement), reset returns to null.
 *
 * Persistence to localStorage is tested via Playwright E2E rather than
 * Vitest because the default vitest environment doesn't provide a DOM /
 * localStorage and adding jsdom is out of scope for this PR.
 */
import { describe, it, expect, beforeEach } from 'vitest';

import { captionStore } from '$lib/stores/caption.svelte';

beforeEach(() => {
  captionStore.resetLLMOverrides();
});

describe('captionStore.llm', () => {
  it('starts with all overrides null (no client-side defaults)', () => {
    expect(captionStore.llm).toEqual({
      connectionId: null,
      model: null,
      temperature: null,
      maxTokens: null,
      topP: null,
      topK: null,
      repetitionPenalty: null,
      presencePenalty: null,
    });
  });

  it('updateLLMOverrides patches only the specified fields', () => {
    captionStore.updateLLMOverrides({ temperature: 0.3 });
    expect(captionStore.llm.temperature).toBe(0.3);
    expect(captionStore.llm.maxTokens).toBeNull();

    captionStore.updateLLMOverrides({ maxTokens: 512 });
    expect(captionStore.llm.temperature).toBe(0.3); // preserved
    expect(captionStore.llm.maxTokens).toBe(512);
  });

  it('updateLLMOverrides supports connection_id swap mid-session', () => {
    captionStore.updateLLMOverrides({ connectionId: 'conn-a' });
    expect(captionStore.llm.connectionId).toBe('conn-a');
    captionStore.updateLLMOverrides({ connectionId: 'conn-b' });
    expect(captionStore.llm.connectionId).toBe('conn-b');
  });

  it('updateLLMOverrides supports per-session model override', () => {
    captionStore.updateLLMOverrides({ model: 'qwen3:32b' });
    expect(captionStore.llm.model).toBe('qwen3:32b');
  });

  it('updateLLMOverrides accepts the full sampling surface', () => {
    captionStore.updateLLMOverrides({
      temperature: 0.42,
      maxTokens: 256,
      topP: 0.9,
      topK: 40,
      repetitionPenalty: 1.1,
      presencePenalty: 0.5,
    });
    expect(captionStore.llm).toMatchObject({
      temperature: 0.42,
      maxTokens: 256,
      topP: 0.9,
      topK: 40,
      repetitionPenalty: 1.1,
      presencePenalty: 0.5,
    });
  });

  it('resetLLMOverrides clears all fields back to null', () => {
    captionStore.updateLLMOverrides({ temperature: 0.5, model: 'm' });
    captionStore.resetLLMOverrides();
    expect(captionStore.llm.temperature).toBeNull();
    expect(captionStore.llm.model).toBeNull();
  });
});
