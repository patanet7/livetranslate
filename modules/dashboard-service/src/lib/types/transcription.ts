export interface Segment {
  text: string;
  start_ms: number;
  end_ms: number;
  confidence: number;
  speaker_id: string | null;
}

export interface TranscriptionResult {
  text: string;
  language: string;
  confidence: number;
  segments: Segment[];
  stable_text: string;
  unstable_text: string;
  is_final: boolean;
  is_draft: boolean;
  speaker_id: string | null;
  should_translate: boolean;
  context_text: string;
}

export interface ModelInfo {
  name: string;
  backend: string;
  languages: string[];
  vram_mb: number;
  compute_type: string;
}
