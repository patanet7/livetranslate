import { describe, it, expect } from 'vitest';
import { createSourceAdapter, LoopbackAdapter, FirefliesAdapter } from '$lib/audio/source-adapter';

describe('createSourceAdapter', () => {
  it('should create LoopbackAdapter for local source', () => {
    const adapter = createSourceAdapter('local');
    expect(adapter).toBeInstanceOf(LoopbackAdapter);
  });

  it('should create LoopbackAdapter for screencapture source', () => {
    const adapter = createSourceAdapter('screencapture');
    expect(adapter).toBeInstanceOf(LoopbackAdapter);
  });

  it('should create FirefliesAdapter for fireflies source', () => {
    const adapter = createSourceAdapter('fireflies');
    expect(adapter).toBeInstanceOf(FirefliesAdapter);
  });
});
