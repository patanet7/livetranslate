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
  const captions = new CaptionStore(8_000);

  // Sync maxCaptions param into the store
  $effect(() => {
    captions.maxCaptions = maxCaptions;
  });

  // Enable caption aggregation: same speaker within 5s window appends text (matches Python CaptionBuffer)
  $effect(() => {
    captions.aggregateWindowMs = 5_000;
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

  // Connection status indicator color (editorial palette: sage / ochre / oxblood)
  const statusColor = $derived(
    ws.status === 'connected'
      ? '#7A9573'
      : ws.status === 'connecting'
        ? '#C49A4F'
        : '#8B3A3A'
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

<svelte:head><title>Captions Overlay — LiveTranslate</title></svelte:head>

{#if !sessionId}
  <!-- Setup help screen when no session param is provided -->
  <div class="help-screen">
    <div class="help-card">
      <p class="help-eyebrow">the wire · overlay</p>
      <h1>Captions Overlay</h1>
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
          style="--fade-duration: {fadeTime}ms; --speaker-ply: {caption.speaker_color};"
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
            {@const translatedText = caption.translated_text || caption.text}
            {@const isTranslating = translatedText === caption.original_text && caption.original_text}
            {#if isTranslating}
              <p class="translated translating">{translatedText}</p>
            {:else}
              <p class="translated">{translatedText}</p>
            {/if}
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
  /* ================================================================
   * THE OVERLAY — paint on glass.
   * Inverted editorial: ink plates carry paper-cream type. Peach
   * pulses for "live", earth-tone plies for speaker identity.
   * Hardcoded hex values are intentional — overlay must render
   * legibly atop arbitrary backgrounds (OBS, Meet, Zoom).
   * ================================================================ */

  /* --- Help screen (theme-aware, lives inside the dashboard) --- */
  .help-screen {
    width: 100vw;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #181410;
    color: #FAF6EE;
    font-family: 'Newsreader', Georgia, serif;
    padding: 24px;
    box-sizing: border-box;
  }

  .help-card {
    max-width: 720px;
    width: 100%;
    background: rgba(250, 246, 238, 0.04);
    border: 1px solid rgba(250, 246, 238, 0.14);
    border-radius: 2px;
    padding: 40px 44px;
    box-shadow: inset 0 0 0 1px rgba(250, 246, 238, 0.04);
  }

  .help-eyebrow {
    font-family: 'Fraunces', Georgia, serif;
    font-variation-settings: 'opsz' 14;
    font-feature-settings: 'smcp', 'c2sc';
    text-transform: lowercase;
    letter-spacing: 0.18em;
    font-size: 11px;
    color: #E8B4A0;
    margin: 0 0 6px;
  }

  .help-card h1 {
    font-family: 'Fraunces', Georgia, serif;
    font-variation-settings: 'opsz' 96, 'SOFT' 50, 'WONK' 1;
    font-weight: 500;
    margin: 0 0 14px;
    font-size: 34px;
    line-height: 1.05;
    color: #FAF6EE;
  }

  .help-intro {
    font-family: 'Newsreader', Georgia, serif;
    font-style: italic;
    color: rgba(250, 246, 238, 0.72);
    margin: 0 0 28px;
    font-size: 15px;
    line-height: 1.55;
  }

  .help-card h2 {
    font-family: 'Fraunces', Georgia, serif;
    font-variation-settings: 'opsz' 14;
    font-feature-settings: 'smcp', 'c2sc';
    text-transform: lowercase;
    letter-spacing: 0.18em;
    font-size: 11px;
    color: rgba(250, 246, 238, 0.54);
    margin: 24px 0 10px;
    border-bottom: 1px solid rgba(250, 246, 238, 0.12);
    padding-bottom: 6px;
  }

  .param-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-bottom: 12px;
    font-family: 'Newsreader', Georgia, serif;
  }

  .param-table th {
    text-align: left;
    color: rgba(250, 246, 238, 0.46);
    font-family: 'Fraunces', Georgia, serif;
    font-variation-settings: 'opsz' 14;
    font-feature-settings: 'smcp', 'c2sc';
    text-transform: lowercase;
    letter-spacing: 0.12em;
    font-weight: 500;
    font-size: 11px;
    padding: 8px 10px 6px;
    border-bottom: 1px solid rgba(250, 246, 238, 0.18);
  }

  .param-table td {
    padding: 8px 10px;
    border-bottom: 1px solid rgba(250, 246, 238, 0.08);
    vertical-align: top;
    color: rgba(250, 246, 238, 0.86);
  }

  .param-table code,
  .example-url {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    background: rgba(232, 180, 160, 0.10);
    padding: 2px 7px;
    border-radius: 1px;
    font-size: 12px;
    color: #E8B4A0;
  }

  .example-url {
    display: block;
    padding: 14px 18px;
    word-break: break-all;
    line-height: 1.7;
    margin: 8px 0;
    border-left: 2px solid #E8B4A0;
    background: rgba(232, 180, 160, 0.06);
  }

  .copy-url-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 16px 0 4px;
  }

  .copy-btn {
    background: #FAF6EE;
    color: #181410;
    border: none;
    border-radius: 1px;
    padding: 9px 18px;
    font-family: 'Fraunces', Georgia, serif;
    font-variation-settings: 'opsz' 14;
    font-feature-settings: 'smcp', 'c2sc';
    text-transform: lowercase;
    letter-spacing: 0.14em;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.18s ease, transform 0.1s ease;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .copy-btn:hover {
    background: #E8B4A0;
  }

  .copy-btn:active {
    transform: translateY(1px);
  }

  .copy-hint {
    color: rgba(250, 246, 238, 0.46);
    font-size: 12px;
    font-style: italic;
  }

  .help-tip {
    color: rgba(250, 246, 238, 0.62);
    font-size: 13px;
    margin: 20px 0 0;
    font-style: italic;
    border-left: 1px solid rgba(232, 180, 160, 0.4);
    padding-left: 14px;
  }

  /* --- The overlay itself (paint on glass) --- */
  .captions-overlay {
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 28px;
    font-family: 'Newsreader', Georgia, serif;
    overflow: hidden;
    position: relative;
  }

  /* --- Connection status indicator --- */
  .status-indicator {
    position: fixed;
    top: 10px;
    right: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(24, 20, 16, 0.78);
    padding: 5px 12px;
    border-radius: 1px;
    border: 1px solid rgba(250, 246, 238, 0.10);
    z-index: 100;
    backdrop-filter: blur(8px);
  }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
    box-shadow: 0 0 0 3px color-mix(in srgb, currentColor 18%, transparent);
  }

  .status-text {
    font-family: 'Fraunces', Georgia, serif;
    font-variation-settings: 'opsz' 14;
    font-feature-settings: 'smcp', 'c2sc';
    text-transform: lowercase;
    letter-spacing: 0.16em;
    font-size: 10px;
    color: rgba(250, 246, 238, 0.78);
    white-space: nowrap;
  }

  /* --- Caption list (the broadsheet) --- */
  .caption-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .caption-entry {
    background: rgba(24, 20, 16, 0.86);
    border-left: 4px solid var(--speaker-ply, #E8B4A0);
    border-radius: 1px;
    padding: 10px 16px 11px;
    animation: captionRise 0.32s cubic-bezier(0.2, 0.8, 0.2, 1);
    opacity: 1;
    transition: opacity var(--fade-duration, 500ms) ease;
    backdrop-filter: blur(6px);
    box-shadow: 0 1px 0 rgba(250, 246, 238, 0.04) inset,
                0 8px 24px rgba(0, 0, 0, 0.32);
  }

  .caption-entry.fading {
    opacity: 0;
  }

  .caption-entry.interim {
    opacity: 0.55;
    border-left-style: dashed;
    border-left-color: rgba(232, 180, 160, 0.6);
  }

  .speaker {
    display: inline-block;
    font-family: 'Fraunces', Georgia, serif;
    font-variation-settings: 'opsz' 14;
    font-feature-settings: 'smcp', 'c2sc';
    font-size: 0.68em;
    font-weight: 500;
    text-transform: lowercase;
    letter-spacing: 0.18em;
    margin-bottom: 2px;
  }

  .original {
    font-family: 'Newsreader', Georgia, serif;
    color: rgba(250, 246, 238, 0.66);
    margin: 2px 0;
    line-height: 1.42;
    font-feature-settings: 'kern', 'liga', 'onum';
  }

  .translated {
    font-family: 'Newsreader', Georgia, serif;
    color: #FAF6EE;
    font-weight: 450;
    margin: 2px 0;
    line-height: 1.4;
    transition: opacity 0.3s ease;
    font-feature-settings: 'kern', 'liga', 'onum';
  }

  .translated.translating {
    opacity: 0.5;
    font-style: italic;
    color: rgba(232, 180, 160, 0.78);
  }

  @keyframes captionRise {
    from {
      opacity: 0;
      transform: translateY(6px);
      filter: blur(2px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
      filter: blur(0);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .caption-entry {
      animation: none;
    }
  }
</style>
