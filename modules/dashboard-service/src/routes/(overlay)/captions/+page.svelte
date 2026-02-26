<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { WS_BASE } from '$lib/config';
  import { WebSocketStore } from '$lib/stores/websocket.svelte';
  import { CaptionStore } from '$lib/stores/captions.svelte';
  import type { CaptionEvent, DisplayMode } from '$lib/types';

  // URL params
  const sessionId = $derived($page.url.searchParams.get('session') ?? '');
  const mode: DisplayMode = $derived(
    ($page.url.searchParams.get('mode') as DisplayMode) ?? 'both'
  );
  const fontSize = $derived(parseInt($page.url.searchParams.get('fontSize') ?? '18'));
  const maxCaptions = $derived(parseInt($page.url.searchParams.get('maxCaptions') ?? '5'));
  const bgColor = $derived($page.url.searchParams.get('bg') ?? 'transparent');

  // Dedicated instances (not singletons) for overlay
  const ws = new WebSocketStore();
  const captions = new CaptionStore(10_000);

  $effect(() => {
    captions.maxCaptions = maxCaptions;
  });

  onMount(() => {
    if (!sessionId) return;

    ws.connect(`${WS_BASE}/api/captions/stream/${sessionId}`);
    ws.onMessage = (event) => {
      const msg: CaptionEvent = JSON.parse(event.data);
      switch (msg.event) {
        case 'connected':
          msg.current_captions.forEach((c) => captions.addCaption(c));
          break;
        case 'caption_added':
          captions.addCaption(msg.caption);
          break;
        case 'caption_updated':
          captions.updateCaption(msg.caption);
          break;
        case 'caption_expired':
          captions.removeCaption(msg.caption_id);
          break;
        case 'interim_caption':
          captions.updateInterim(msg.caption.text);
          break;
        case 'session_cleared':
          captions.clear();
          break;
      }
    };

    captions.start();

    return () => {
      ws.disconnect();
      captions.stop();
    };
  });
</script>

<div class="captions-overlay" style="background: {bgColor}; font-size: {fontSize}px;">
  {#if !sessionId}
    <div class="no-session">
      <p>Missing ?session= parameter</p>
    </div>
  {:else}
    <div class="caption-list">
      {#each captions.captions as caption (caption.id)}
        <div class="caption-entry" data-caption-id={caption.id}>
          <span class="speaker" style="color: {caption.speaker_color}">
            {caption.speaker_name}
          </span>
          {#if mode === 'both' || mode === 'english'}
            <p class="original">{caption.original_text}</p>
          {/if}
          {#if mode === 'both' || mode === 'translated'}
            <p class="translated">{caption.text}</p>
          {/if}
        </div>
      {/each}

      {#if captions.interim && (mode === 'both' || mode === 'english')}
        <div class="caption-entry interim">
          <p class="original">{captions.interim}</p>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .captions-overlay {
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 20px;
    font-family: 'Segoe UI', system-ui, sans-serif;
    overflow: hidden;
  }

  .no-session {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #999;
  }

  .caption-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .caption-entry {
    background: rgba(0, 0, 0, 0.75);
    border-radius: 8px;
    padding: 8px 12px;
    animation: fadeIn 0.3s ease;
  }

  .caption-entry.interim {
    opacity: 0.6;
    border: 1px dashed rgba(255, 255, 255, 0.3);
  }

  .speaker {
    font-size: 0.75em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .original {
    color: rgba(255, 255, 255, 0.7);
    margin: 2px 0;
  }

  .translated {
    color: #fff;
    font-weight: 500;
    margin: 2px 0;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
</style>
