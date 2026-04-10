<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import { page } from '$app/stores';
  import { browser } from '$app/environment';
  import { WS_BASE } from '$lib/config';
  import { WebSocketStore } from '$lib/stores/websocket.svelte';
  import { CaptionStore } from '$lib/stores/captions.svelte';
  import type { CaptionEvent, DisplayMode } from '$lib/types';

  // --- URL parameters (static / not live-switchable) ---
  const sessionId = $derived($page.url.searchParams.get('session') ?? '');
  const lang = $derived($page.url.searchParams.get('lang') ?? '');
  const position = $derived(
    ($page.url.searchParams.get('position') ?? 'bottom') as 'top' | 'center' | 'bottom'
  );
  const maxCaptions = $derived.by(() => {
    const raw = parseInt($page.url.searchParams.get('maxCaptions') ?? '5', 10);
    return isNaN(raw) ? 5 : raw;
  });
  const fadeTime = $derived.by(() => {
    const raw = parseInt($page.url.searchParams.get('fadeTime') ?? '500', 10);
    return isNaN(raw) ? 500 : raw;
  });
  const bgParam = $derived($page.url.searchParams.get('bg') ?? '');
  const bgColor = $derived(
    bgParam
      ? /^[0-9a-fA-F]{3,8}$/.test(bgParam)
        ? `#${bgParam}`
        : bgParam
      : 'transparent'
  );
  const showStatus = $derived($page.url.searchParams.get('showStatus') !== 'false');

  // --- Live-switchable config (initialized from URL params, overridable via config_changed WS events) ---
  // Read URL params once at parse time (not in $effect, which would re-run and clobber WS values)
  const initFontSize = parseInt($page.url.searchParams.get('fontSize') ?? '18', 10) || 18;
  const initShowSpeaker = $page.url.searchParams.get('showSpeaker') !== 'false';
  const initShowOriginal = $page.url.searchParams.get('showOriginal') !== 'false';
  const initMode = $page.url.searchParams.get('mode') ?? 'both';

  let liveShowSpeaker = $state(initShowSpeaker);
  let liveFontSize = $state(initFontSize);
  let liveShowOriginal = $state(initShowOriginal);
  let liveMode = $state<string>(initMode);
  let liveTheme = $state<string>('dark');

  // Dedicated instances (not singletons) for overlay
  const ws = new WebSocketStore();
  const captions = new CaptionStore(10_000);

  // Sync maxCaptions param into the store
  $effect(() => {
    captions.maxCaptions = maxCaptions;
  });

  // Enable caption aggregation: same speaker within 3s window appends text
  $effect(() => {
    captions.aggregateWindowMs = 3_000;
  });

  // Track captions that are fading out
  let fadingIds = $state<Set<string>>(new Set());

  // Schedule fade-out before expiry.
  // Uses untrack for fadingIds to prevent re-running when fade state changes.
  $effect(() => {
    // Track captions list and fadeTime as dependencies
    const currentCaptions = captions.captions;
    const fadeDuration = fadeTime;
    const timers: ReturnType<typeof setTimeout>[] = [];

    // Read fadingIds without creating a dependency
    const currentFading = untrack(() => fadingIds);

    for (const caption of currentCaptions) {
      if (currentFading.has(caption.id)) continue;

      const expiresAt = new Date(caption.expires_at).getTime();
      const now = Date.now();
      // Start fade fadeDuration ms before expiry; if already past that point, start immediately
      const fadeStart = Math.max(0, expiresAt - fadeDuration - now);

      const timer = setTimeout(() => {
        fadingIds = new Set([...fadingIds, caption.id]);
      }, fadeStart);
      timers.push(timer);
    }

    return () => {
      for (const t of timers) clearTimeout(t);
    };
  });

  // Connection status indicator color
  const statusColor = $derived(
    ws.status === 'connected'
      ? '#22c55e'
      : ws.status === 'connecting'
        ? '#eab308'
        : '#ef4444'
  );

  const statusLabel = $derived(
    ws.status === 'connected'
      ? 'Connected'
      : ws.status === 'connecting'
        ? `Reconnecting (${ws.reconnectAttempt})`
        : 'Disconnected'
  );

  // Compute position CSS
  const positionAlign = $derived(
    position === 'top'
      ? 'flex-start'
      : position === 'center'
        ? 'center'
        : 'flex-end'
  );

  // Filter captions by language if lang param is set
  const filteredCaptions = $derived(
    lang
      ? captions.captions.filter((c) => c.target_language === lang)
      : captions.captions
  );

  // Determine whether to show original text based on both liveShowOriginal and liveMode.
  // Handles both legacy modes (both/translated/english) and spec canonical modes (subtitle/split/interpreter).
  const shouldShowOriginal = $derived(
    liveShowOriginal && (
      liveMode === 'both' || liveMode === 'english' ||
      liveMode === 'split' || liveMode === 'interpreter'
    )
  );
  const shouldShowTranslated = $derived(
    liveMode === 'both' || liveMode === 'translated' ||
    liveMode === 'subtitle' || liveMode === 'split' || liveMode === 'interpreter'
  );

  // Copy URL helper for the help screen
  let copySuccess = $state(false);

  function copyOverlayUrl() {
    if (!browser) return;
    const url = `${window.location.origin}${window.location.pathname}`;
    navigator.clipboard.writeText(url).then(() => {
      copySuccess = true;
      setTimeout(() => {
        copySuccess = false;
      }, 2000);
    }).catch(() => {
      // Fallback: select the text if clipboard API fails
      const el = document.querySelector<HTMLElement>('.example-url');
      if (el) {
        const range = document.createRange();
        range.selectNode(el);
        window.getSelection()?.removeAllRanges();
        window.getSelection()?.addRange(range);
      }
    });
  }

  onMount(() => {
    if (!sessionId) return;

    ws.connect(`${WS_BASE}/api/captions/stream/${sessionId}`);
    ws.onMessage = (event: MessageEvent) => {
      const msg: CaptionEvent = JSON.parse(event.data) as CaptionEvent;
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
          fadingIds = new Set([...fadingIds].filter((id) => id !== msg.caption_id));
          break;
        case 'interim_caption':
          captions.updateInterim(msg.caption.text);
          break;
        case 'session_cleared':
          captions.clear();
          fadingIds = new Set();
          break;
        case 'config_changed':
          if (msg.changes) {
            if ('font_size' in msg.changes) liveFontSize = msg.changes.font_size;
            if ('show_speakers' in msg.changes) liveShowSpeaker = msg.changes.show_speakers;
            if ('show_original' in msg.changes) liveShowOriginal = msg.changes.show_original;
            if ('display_mode' in msg.changes) liveMode = msg.changes.display_mode;
            if ('theme' in msg.changes) liveTheme = msg.changes.theme;
          }
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

{#if !sessionId}
  <!-- Setup help screen when no session param is provided -->
  <div class="help-screen">
    <div class="help-card">
      <h1>LiveTranslate Captions Overlay</h1>
      <p class="help-intro">
        Add this page as a Browser Source in OBS or any streaming tool.
        Pass URL parameters to configure the overlay.
      </p>

      <h2>Required</h2>
      <table class="param-table">
        <thead><tr><th>Parameter</th><th>Description</th><th>Example</th></tr></thead>
        <tbody>
          <tr>
            <td><code>session</code></td>
            <td>Session ID to connect to</td>
            <td><code>?session=abc-123</code></td>
          </tr>
        </tbody>
      </table>

      <h2>Optional</h2>
      <table class="param-table">
        <thead><tr><th>Parameter</th><th>Default</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td><code>lang</code></td><td><em>all</em></td><td>Target language filter (e.g. <code>es</code>, <code>fr</code>)</td></tr>
          <tr><td><code>mode</code></td><td><code>both</code></td><td>Display mode: <code>both</code>, <code>translated</code>, or <code>english</code></td></tr>
          <tr><td><code>showSpeaker</code></td><td><code>true</code></td><td>Show speaker names above captions</td></tr>
          <tr><td><code>showOriginal</code></td><td><code>true</code></td><td>Show original text alongside translation</td></tr>
          <tr><td><code>fontSize</code></td><td><code>18</code></td><td>Caption font size in pixels</td></tr>
          <tr><td><code>position</code></td><td><code>bottom</code></td><td>Vertical position: <code>top</code>, <code>center</code>, or <code>bottom</code></td></tr>
          <tr><td><code>maxCaptions</code></td><td><code>5</code></td><td>Maximum visible captions at once</td></tr>
          <tr><td><code>fadeTime</code></td><td><code>500</code></td><td>Fade out duration in milliseconds</td></tr>
          <tr><td><code>bg</code></td><td><em>transparent</em></td><td>Background color hex without # (e.g. <code>000000</code>)</td></tr>
          <tr><td><code>showStatus</code></td><td><code>true</code></td><td>Show connection status indicator</td></tr>
        </tbody>
      </table>

      <h2>Example URL</h2>
      <code class="example-url">
        /captions?session=my-session&lang=es&mode=translated&fontSize=24&position=bottom&maxCaptions=3&fadeTime=800&bg=000000&showStatus=true
      </code>

      <div class="copy-url-row">
        <button class="copy-btn" onclick={copyOverlayUrl}>
          {copySuccess ? 'Copied!' : 'Copy base URL'}
        </button>
        <span class="copy-hint">Paste into OBS Browser Source URL field</span>
      </div>

      <p class="help-tip">
        Tip: For OBS, set the Browser Source width/height to match your canvas
        and leave the background transparent for a clean overlay.
      </p>
    </div>
  </div>
{:else}
  <div
    class="captions-overlay"
    style="background: {bgColor}; font-size: {liveFontSize}px; justify-content: {positionAlign};"
  >
    <!-- Connection status indicator -->
    {#if showStatus}
      <div class="status-indicator">
        <span class="status-dot" style="background: {statusColor};"></span>
        <span class="status-text">{statusLabel}</span>
      </div>
    {/if}

    <div class="caption-list">
      {#each filteredCaptions as caption (caption.id)}
        <div
          class="caption-entry"
          class:fading={fadingIds.has(caption.id)}
          style="--fade-duration: {fadeTime}ms;"
          data-caption-id={caption.id}
        >
          {#if liveShowSpeaker}
            <span class="speaker" style="color: {caption.speaker_color};">
              {caption.speaker_name}
            </span>
          {/if}
          {#if shouldShowOriginal}
            <p class="original">{caption.original_text}</p>
          {/if}
          {#if shouldShowTranslated}
            <p class="translated">{caption.translated_text || caption.text}</p>
          {/if}
        </div>
      {/each}

      {#if captions.interim && shouldShowOriginal}
        <div class="caption-entry interim">
          <p class="original">{captions.interim}</p>
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  /* --- Help screen styles --- */
  .help-screen {
    width: 100vw;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0f0f0f;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 24px;
    box-sizing: border-box;
  }

  .help-card {
    max-width: 720px;
    width: 100%;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 32px;
  }

  .help-card h1 {
    margin: 0 0 8px;
    font-size: 24px;
    color: #fff;
  }

  .help-intro {
    color: #999;
    margin: 0 0 24px;
    font-size: 14px;
    line-height: 1.5;
  }

  .help-card h2 {
    font-size: 16px;
    color: #ccc;
    margin: 20px 0 8px;
    border-bottom: 1px solid #333;
    padding-bottom: 4px;
  }

  .param-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-bottom: 12px;
  }

  .param-table th {
    text-align: left;
    color: #888;
    font-weight: 600;
    padding: 6px 8px;
    border-bottom: 1px solid #333;
  }

  .param-table td {
    padding: 6px 8px;
    border-bottom: 1px solid #222;
    vertical-align: top;
  }

  .param-table code {
    background: #2a2a2a;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 12px;
    color: #7dd3fc;
  }

  .example-url {
    display: block;
    background: #2a2a2a;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 12px;
    color: #7dd3fc;
    word-break: break-all;
    line-height: 1.6;
    margin: 8px 0;
  }

  .copy-url-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 12px 0 4px;
  }

  .copy-btn {
    background: #2563eb;
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .copy-btn:hover {
    background: #1d4ed8;
  }

  .copy-btn:active {
    background: #1e40af;
  }

  .copy-hint {
    color: #666;
    font-size: 12px;
  }

  .help-tip {
    color: #888;
    font-size: 13px;
    margin: 16px 0 0;
    font-style: italic;
  }

  /* --- Overlay styles --- */
  .captions-overlay {
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 20px;
    font-family: 'Segoe UI', system-ui, sans-serif;
    overflow: hidden;
    position: relative;
  }

  /* --- Connection status indicator --- */
  .status-indicator {
    position: fixed;
    top: 8px;
    right: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 0, 0, 0.6);
    padding: 4px 10px;
    border-radius: 12px;
    z-index: 100;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
  }

  .status-text {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.7);
    white-space: nowrap;
  }

  /* --- Caption list --- */
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
    opacity: 1;
    transition: opacity var(--fade-duration, 500ms) ease;
  }

  .caption-entry.fading {
    opacity: 0;
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
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>
