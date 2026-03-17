# Whisper Architecture, Capabilities, and Limitations for Real-Time Transcription Systems

**Research Document | Compiled 2026-03-17**

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Language Identification (LID)](#2-language-identification-lid)
3. [Language Hint Behavior](#3-language-hint-behavior)
4. [Code-Switching Limitations](#4-code-switching-limitations)
5. [Performance Characteristics](#5-performance-characteristics)
6. [Noise and Echo Robustness](#6-noise-and-echo-robustness)
7. [Advanced Capabilities](#7-advanced-capabilities)
8. [Practical Recommendations for Real-Time Mixed-Language Systems](#8-practical-recommendations-for-real-time-mixed-language-systems)
9. [References](#9-references)

---

## 1. Architecture

### 1.1 Overview

Whisper (Radford et al., 2022) is an encoder-decoder transformer trained on 680,000 hours of weakly supervised multilingual speech data. The architecture performs joint multitask learning: transcription, translation (to English), language identification, and voice activity detection — all encoded into the decoder token sequence rather than separate heads.

### 1.2 Input Representation

| Parameter | Value |
|---|---|
| Mel channels | 80 |
| Window size | 25ms (400 samples at 16kHz) |
| Hop size | 10ms (160 samples at 16kHz) |
| Context window | 30 seconds (3,000 mel frames) |
| Sample rate | 16kHz (resampling required for other rates) |

The 30-second window is a hard architectural constraint. Audio shorter than 30s is zero-padded; audio longer is chunked.

### 1.3 Decoder Sequence

```
<|startoftranscript|> <|lang_token|> <|transcribe|> <|notimestamps|> ... text tokens ...
```

### 1.4 Vocabulary

| Model | Vocabulary Size | Tokenizer |
|---|---|---|
| Whisper large-v3 | 51,866 tokens | Multilingual BPE |
| Whisper large-v3-turbo | 51,866 tokens | Same as large-v3 |

The BPE tokenizer is jointly trained across all 99 supported languages. CJK characters share vocabulary space with Latin — there is no hard filter preventing CJK tokens from appearing in English-hinted output, only a strong soft bias.

### 1.5 Model Variants

| Model | Encoder Layers | Decoder Layers | Parameters | Notes |
|---|---|---|---|---|
| large-v3 | 32 | 32 | 1.55B | Full model |
| large-v3-turbo | 32 | **4** | ~809M | Distilled decoder, encoder unchanged |

**large-v3-turbo**: Decoder is ~8x faster. Transcription WER/CER largely preserved on clean audio. LID accuracy degrades moderately at short durations. Timestamp quality degrades noticeably.

---

## 2. Language Identification (LID)

### 2.1 Mechanism

LID is a single decoder forward pass: given encoder output, produce logits over 99 language token positions, softmax, argmax. Requires full encoder forward pass (same cost as transcription), adds only ~10-20ms decoder overhead.

LID operates on the **full 30-second mel window** — the dominant language by acoustic mass wins.

### 2.2 LID Accuracy by Chunk Duration

| Effective Speech Duration | large-v3 Accuracy | large-v3-turbo Accuracy |
|---|---|---|
| ~1 second | ~78% | ~71-73% |
| ~3 seconds | ~91% | ~84-86% |
| ~6 seconds | ~96% | ~89-91% |
| ~10 seconds | ~98% | ~93-95% |
| ~30 seconds | ~99% | ~94-96% |

large-v3-turbo degrades ~5-7pp at short durations due to reduced decoder capacity.

### 2.3 Granularity

LID is **per-chunk only**. No per-segment, no per-word. Each `transcribe()` call produces exactly one language determination.

### 2.4 `language=None` vs `language='en'`

| Mode | LID Step | Effect |
|---|---|---|
| `language=None` | Runs `detect_language()` | Auto-detect, ~10-20ms overhead |
| `language='en'` | Skipped entirely | Forces English token, saves latency |

---

## 3. Language Hint Behavior

### 3.1 Correct Hint

WER/CER identical to auto-detect by construction. Benefit is purely latency (~10-20ms saved).

### 3.2 Incorrect Hint — Catastrophic

| Effect | Details |
|---|---|
| CJK word drop rate | 60-70% for short utterances (1-3 words) with wrong hint |
| Output quality | Phonetic gibberish ("nee how" for "你好") |
| Severity | Near-total information loss for wrong-script languages |

### 3.3 Mixed-Chunk Behavior

| Composition | auto-detect result |
|---|---|
| 5s English + 1s Chinese | "en" ~95% of the time |
| 3s English + 3s Chinese | "en" ~60%, "zh" ~40% |
| 1s English + 5s Chinese | "zh" ~92% |

### 3.4 Short Utterance Detection

| Scenario | Outcome |
|---|---|
| 1-3 word Chinese, `hint='en'` | 60-70% drop rate |
| 1-3 word Chinese, `hint='zh'` | Preserved with high fidelity |
| 1-3 word Chinese, `hint=None` | LID ~78% at 1s → unreliable |

---

## 4. Code-Switching Limitations

### 4.1 Architectural Hard Limit

ONE language token per `transcribe()` call. Intra-chunk code-switching is **architecturally unsolvable** with standard Whisper.

### 4.2 Intra-Chunk

"Let me check the 季度报告 for Q3" — neither hint nor auto-detect preserves both languages. The minority-language portion is lost or corrupted regardless of configuration.

### 4.3 Inter-Sentence

Solvable via session restart at VAD boundaries. Finish current session, start fresh with correct language token. Latency: 100-300ms depending on VAD dwell time.

### 4.4 Research-Stage: Parallel Decoders

- Two decoder instances, different language tokens
- Cross-attention masking by predicted script
- 60-80% intra-sentence accuracy (controlled experiments)
- 1.4-1.6x compute cost
- Not production-ready as of early 2026

---

## 5. Performance Characteristics

### 5.1 Compute (6s chunk, Apple Silicon)

| Operation | large-v3-turbo | large-v3 |
|---|---|---|
| Encoder | ~80ms | ~180ms |
| LID step | ~10-20ms | ~10-20ms |
| Transcription decode | ~40-80ms | ~100-200ms |
| **Total (auto-detect)** | **~130-180ms** | **~290-400ms** |
| **Total (with hint)** | **~120-160ms** | **~280-380ms** |

### 5.2 `no_speech_prob` Threshold

Default 0.6. Minority-language speech can be classified as "no speech" when dominant language creates silence prior. **Recommended: lower to 0.4-0.5 for multilingual systems.**

---

## 6. Noise and Echo Robustness

### 6.1 LID Under Noise

| Condition | LID Accuracy (Mandarin, large-v3) |
|---|---|
| Clean audio | ~95-99% |
| Office noise, 15dB SNR | ~89% |
| Office noise, 10dB SNR | ~82% |
| Echo-heavy room | ~84-88% (estimated) |

### 6.2 Browser Audio Processing

WebRTC processing (echoCancellation, noiseSuppression) can **degrade** Whisper performance. For loopback/system capture, all browser audio processing should be **disabled**.

---

## 7. Advanced Capabilities

### 7.1 Frame-Level LID Probing

Reuse encoder output for multiple LID probes at sub-chunk granularity (100ms hops):
- Zero memory overhead (read-only)
- ~10-20ms per probe
- Accuracy follows Section 2.2 curves for effective window size

### 7.2 SustainedLanguageDetector Hysteresis

| Parameter | Recommended Value |
|---|---|
| Confidence margin | > 0.20 (20pp above second-best) |
| Consecutive frames | 6 (~600ms at 100ms hop) |
| Minimum dwell time | 250ms |
| Cooldown after switch | 500ms |

### 7.3 Hallucination Patterns

| Pattern | Trigger | Mitigation |
|---|---|---|
| Repetitive text | Long silence / low energy | N-gram overlap detection |
| Wrong-language output | Incorrect hint / LID failure | Script detection on output |
| Silence classification | Minority-language speech | Lower `no_speech_prob` threshold |

---

## 8. Practical Recommendations

### 8.1 Stride vs Accuracy vs Switch Responsiveness

| Stride | WER (EN) | CER (ZH) | Switch Detection |
|---|---|---|---|
| 1.5s | ~32% | ~20% | Fast (~1.5s) |
| 3s | ~25% | ~16% | Moderate (~3s) |
| 4.5s | ~22% | ~16% | Balanced |
| 6s | ~19% | ~9.5% | Slow (may miss short insertions) |

### 8.2 Language Hint Strategy

1. Maintain `current_language` state
2. Use as hint (skip LID, save ~10-20ms)
3. Run LID in parallel for switch detection
4. Update on sustained change (SustainedLanguageDetector)
5. Fall back to `language=None` only when confidence is low

### 8.3 Configuration Decision Matrix

| Scenario | Configuration |
|---|---|
| Single language, known | hint + 6s stride + 1.5s overlap |
| Mixed, inter-sentence | SustainedLanguageDetector + session restart at VAD |
| Mixed, intra-sentence | No reliable Whisper solution; consider Paraformer/SenseVoice |
| High-noise | Lower `no_speech_prob` to 0.4; increase overlap; disable browser audio processing |
| Low-latency (<2s) | 1.5s stride; accept higher WER; use hint |

---

## 9. References

- Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv:2212.04356
- OpenAI Whisper GitHub: https://github.com/openai/whisper
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- vLLM-MLX Whisper: Apple Silicon inference via MLX
- Peng et al. (CMU SpeechLab): Noise-corrupted LID benchmarks
- Community LID accuracy curves: faster-whisper issues #614, #583
