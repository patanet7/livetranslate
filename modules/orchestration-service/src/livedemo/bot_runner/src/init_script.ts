/**
 * Init script — runs in the page context BEFORE any Meet JS, on every navigation.
 *
 * Responsibilities:
 *  1. Create a 1280x720 canvas, **append it to document.body** (B6: detached
 *     canvases stall captureStream after GC). Hidden via inline CSS.
 *  2. Override navigator.mediaDevices.getUserMedia so a video request returns
 *     the canvas stream. Audio requests fall through to the original device.
 *  3. Override enumerateDevices so Meet sees a fake "LiveTranslate Canvas Camera".
 *  4. Expose `window.__livedemo.setFrameDataUrl(dataUrl)` for Playwright to
 *     update the canvas pixels. The frame source-of-truth lives in Python; this
 *     script only paints.
 *
 * NOTE: This file is read by `runner.ts` at startup and injected via
 * `page.addInitScript`. It is NOT bundled or imported elsewhere — keep it
 * self-contained, no top-level imports.
 */

export const INIT_SCRIPT = `
(() => {
  if (window.__livedemoInjected) return;
  window.__livedemoInjected = true;

  const W = 1280, H = 720;
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  // B6 — must be in DOM so captureStream stays live across GC.
  canvas.style.position = 'fixed';
  canvas.style.left = '-99999px';
  canvas.style.top = '-99999px';
  canvas.style.opacity = '0';
  canvas.style.pointerEvents = 'none';
  canvas.style.width = '1px';
  canvas.style.height = '1px';
  if (document.body) {
    document.body.appendChild(canvas);
  } else {
    document.addEventListener('DOMContentLoaded', () => document.body.appendChild(canvas));
  }

  const ctx = canvas.getContext('2d');

  // Placeholder while Python isn't pushing frames yet.
  function drawPlaceholder() {
    ctx.fillStyle = '#161616';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#3b9eff';
    ctx.font = '36px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('LiveTranslate Bot', W/2, H/2 - 12);
    ctx.fillStyle = '#a0a0a0';
    ctx.font = '22px Arial';
    ctx.fillText('Waiting for captions…', W/2, H/2 + 24);
  }
  drawPlaceholder();

  // Keep redrawing at low fps so captureStream emits frames even when Python is idle.
  // (Otherwise some Chromium versions stall after a few hundred ms of stasis.)
  setInterval(() => {
    if (!window.__livedemo || !window.__livedemo.lastFrameDataUrl) {
      drawPlaceholder();
    } else {
      // No-op redraw of last frame to keep the stream warm.
      const img = window.__livedemo.lastImg;
      if (img && img.complete && img.naturalWidth > 0) {
        ctx.drawImage(img, 0, 0, W, H);
      }
    }
  }, 200);

  const stream = canvas.captureStream(30);

  // Public surface for Playwright (and our protocol).
  window.__livedemo = {
    canvas,
    stream,
    lastFrameDataUrl: null,
    lastImg: null,
    setFrameDataUrl(dataUrl) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, W, H);
        window.__livedemo.lastImg = img;
      };
      img.src = dataUrl;
      window.__livedemo.lastFrameDataUrl = dataUrl;
    },
    isCanvasInDom() {
      return canvas.parentElement === document.body;
    }
  };

  // ── getUserMedia override ─────────────────────────────────────
  // Skip override if mediaDevices unavailable (insecure context like about:blank).
  // Real Meet is HTTPS so this is always present in production.
  if (!navigator.mediaDevices) {
    console.warn('[livedemo] navigator.mediaDevices unavailable — skipping override');
    return;
  }
  const origGUM = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async function (constraints) {
    if (constraints && constraints.video) {
      const merged = new MediaStream();
      stream.getVideoTracks().forEach(t => merged.addTrack(t));
      if (constraints.audio) {
        try {
          const audioOnly = await origGUM({ audio: constraints.audio });
          audioOnly.getAudioTracks().forEach(t => merged.addTrack(t));
        } catch (_) { /* mic may not be available; continue video-only */ }
      }
      return merged;
    }
    return origGUM(constraints);
  };

  // ── enumerateDevices override ─────────────────────────────────
  const origED = navigator.mediaDevices.enumerateDevices.bind(navigator.mediaDevices);
  navigator.mediaDevices.enumerateDevices = async function () {
    const devices = await origED();
    if (devices.some(d => d.kind === 'videoinput')) return devices;
    return [
      ...devices,
      { deviceId: 'livedemo-canvas', kind: 'videoinput', label: 'LiveTranslate Canvas Camera', groupId: 'livedemo' },
    ];
  };

  // ── Audio capture pipeline ──────────────────────────────────
  //
  // Captures audio from a MediaStream (typically the one returned by our
  // overridden getUserMedia, which carries the bot's mic + any preserved audio)
  // and exposes 20ms int16 PCM chunks at 16kHz to the runner via takeChunk().
  //
  // The runner.ts polls takeChunk() and forwards bytes to the Python harness
  // as binary WS frames. Python harness queues them; MeetAudioSource pumps
  // them to the orchestration /api/audio/stream.
  const audioState = {
    chunks: [],
    sourceCtx: null,
    workletNode: null,
    sampleBuffer: [],
    sourceSampleRate: 48000,
  };

  function _flushSampleBuffer() {
    while (audioState.sampleBuffer.length >= 320) {
      const samples = audioState.sampleBuffer.splice(0, 320);
      const out = new ArrayBuffer(640);
      const view = new DataView(out);
      for (let i = 0; i < 320; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      audioState.chunks.push(out);
      if (audioState.chunks.length > 200) audioState.chunks.shift();
    }
  }

  function _resampleAndAccumulate(float32Frame, srcRate) {
    const targetRate = 16000;
    const ratio = srcRate / targetRate;
    const outLen = Math.floor(float32Frame.length / ratio);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * ratio;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, float32Frame.length - 1);
      const frac = srcIdx - lo;
      audioState.sampleBuffer.push(
        float32Frame[lo] * (1 - frac) + float32Frame[hi] * frac
      );
    }
    _flushSampleBuffer();
  }

  window.__livedemoAudio = {
    async startCapture(stream) {
      if (audioState.sourceCtx) {
        try { audioState.sourceCtx.close(); } catch (e) {}
      }
      const ctx = new AudioContext();
      audioState.sourceSampleRate = ctx.sampleRate;
      audioState.sourceCtx = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(2048, 1, 1);
      proc.onaudioprocess = (e) => {
        const ch = e.inputBuffer.getChannelData(0);
        const frame = new Float32Array(ch.length);
        frame.set(ch);
        _resampleAndAccumulate(frame, ctx.sampleRate);
      };
      src.connect(proc);
      proc.connect(ctx.destination);
      audioState.workletNode = proc;
    },
    takeChunk() {
      return audioState.chunks.shift() || null;
    },
    chunkBacklog() {
      return audioState.chunks.length;
    },
    stopCapture() {
      try { audioState.workletNode && audioState.workletNode.disconnect(); } catch (e) {}
      try { audioState.sourceCtx && audioState.sourceCtx.close(); } catch (e) {}
      audioState.workletNode = null;
      audioState.sourceCtx = null;
      audioState.sampleBuffer = [];
    },
  };

  console.log('[livedemo] init script loaded');
})();
`;
