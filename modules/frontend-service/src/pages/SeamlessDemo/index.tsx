import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Box, Button, Card, CardContent, Stack, Typography } from '@mui/material';

function floatTo16BitPCM(float32Array: Float32Array): Int16Array {
  const buffer = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    buffer[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return buffer;
}

function base64FromPCM(int16: Int16Array): string {
  const bytes = new Uint8Array(int16.buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

const SAMPLE_RATE = 16000;
const CHUNK_MS = 200;

const SeamlessDemo: React.FC = () => {
  const [connected, setConnected] = useState(false);
  const [transcriptPartial, setTranscriptPartial] = useState('');
  const [transcriptFinal, setTranscriptFinal] = useState('');
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const ringRef = useRef<Float32Array>(new Float32Array(0));
  const lastSendRef = useRef<number>(0);

  const sessionIdRef = useRef<string>(`seamless-${Date.now()}`);
  const sessionId = sessionIdRef.current;

  const connectWS = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) return;
    const base = (import.meta as any).env.VITE_API_BASE || 'http://localhost:8000';
    const url = base.replace('http', 'ws') + `/api/seamless/realtime/${sessionId}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => {
      setConnected(true);
      ws.send(JSON.stringify({ type: 'config', source: 'cmn', target: 'eng', emitPartials: true }));
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'translation_partial') {
          setTranscriptPartial(msg.text || '');
        } else if (msg.type === 'translation_final') {
          setTranscriptFinal((prev) => (prev ? prev + '\n' : '') + (msg.text || ''));
          setTranscriptPartial('');
        }
      } catch {}
    };
    ws.onclose = () => { setConnected(false); };
    ws.onerror = () => { setConnected(false); };
  }, [sessionId]);

  const start = useCallback(async () => {
    connectWS();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: SAMPLE_RATE } });
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE });
    audioCtxRef.current = ctx;
    const source = ctx.createMediaStreamSource(stream);
    sourceRef.current = source;
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;
    ringRef.current = new Float32Array(0);
    lastSendRef.current = performance.now();

    processor.onaudioprocess = (event: AudioProcessingEvent) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      const data = event.inputBuffer.getChannelData(0);
      const merged = new Float32Array(ringRef.current.length + data.length);
      merged.set(ringRef.current, 0);
      merged.set(data, ringRef.current.length);
      ringRef.current = merged;

      const now = performance.now();
      if (now - lastSendRef.current >= CHUNK_MS) {
        const samplesPerChunk = Math.floor((SAMPLE_RATE * CHUNK_MS) / 1000);
        if (ringRef.current.length >= samplesPerChunk) {
          const chunk = ringRef.current.slice(0, samplesPerChunk);
          ringRef.current = ringRef.current.slice(samplesPerChunk);
          const pcm16 = floatTo16BitPCM(chunk);
          const b64 = base64FromPCM(pcm16);
          wsRef.current.send(JSON.stringify({ type: 'audio_chunk', data: b64, sampleRate: SAMPLE_RATE, channels: 1, chunkMs: CHUNK_MS }));
          lastSendRef.current = now;
        }
      }
    };

    source.connect(processor);
    processor.connect(ctx.destination);
  }, [connectWS]);

  const stop = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end' }));
    }
    if (processorRef.current) {
      try { processorRef.current.disconnect(); } catch {}
      processorRef.current = null;
    }
    if (sourceRef.current) {
      try { sourceRef.current.disconnect(); } catch {}
      sourceRef.current = null;
    }
    if (audioCtxRef.current) {
      try { audioCtxRef.current.close(); } catch {}
      audioCtxRef.current = null;
    }
  }, []);

  useEffect(() => () => { stop(); if (wsRef.current) try { wsRef.current.close(); } catch {} }, [stop]);

  return (
    <Box p={3}>
      <Stack direction="row" spacing={2} mb={2}>
        <Button variant="contained" onClick={start} disabled={connected}>Start</Button>
        <Button variant="outlined" onClick={stop}>Stop</Button>
      </Stack>
      <Card>
        <CardContent>
          <Typography variant="h6">Partial (EN)</Typography>
          <Typography sx={{ whiteSpace: 'pre-wrap' }}>{transcriptPartial || '...'}</Typography>
        </CardContent>
      </Card>
      <Box height={16} />
      <Card>
        <CardContent>
          <Typography variant="h6">Final (EN)</Typography>
          <Typography sx={{ whiteSpace: 'pre-wrap' }}>{transcriptFinal || '...'}</Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default SeamlessDemo;


