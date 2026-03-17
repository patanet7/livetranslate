# Multilingual ASR Models for Real-Time en↔zh Code-Switching (2026)

Research compiled 2026-03-17. See full analysis in brainstorming session.

## Top Candidates for LiveTranslate

### Tier 1: Drop-in Replacements

| Model | Params | VRAM | zh CER | en WER | Code-Switch | Speed | Apple Silicon | License |
|---|---|---|---|---|---|---|---|---|
| **Qwen3-ASR-1.7B** | 1.7B | ~5 GB | **4.97%** | **1.63%** | Yes (intra-sentence) | RTF 0.06 | **Native MLX** | Apache 2.0 |
| **Qwen3-ASR-0.6B** | 0.6B | ~2.2 GB | ~6-7% | ~2.5% | Yes | RTF 0.064 | **Native MLX** | Apache 2.0 |
| **SenseVoice-Small** | ~240M | ~1 GB | ~5.76% | ~4-5% | Yes (**per-word LID**) | 70ms/10s | ONNX+CoreML | Apache 2.0 |
| **Paraformer-trilingual** | ~220M | ~1 GB | ~7.4% | ~3-4% | Yes (zh/Canton/en) | >20x | ONNX+sherpa | MIT |

### Tier 2: High Accuracy (Offline/Heavy)

| Model | Params | VRAM | zh CER | en WER | Code-Switch | Apple Silicon |
|---|---|---|---|---|---|---|
| **Fun-ASR-Nano** | ~0.8B | ~2 GB | ~1.22% | ~1.57% | Yes (1.59%) | PyTorch MPS |
| **FireRedASR-AED** | 1.1B | ~4.5 GB | ~3.18% | 1.93% | Asserted (v2) | PyTorch MPS |

### Current: Whisper large-v3-turbo

| Model | Params | VRAM | zh CER | en WER | Code-Switch | Apple Silicon |
|---|---|---|---|---|---|---|
| Whisper large-v3-turbo | 809M | ~6 GB | ~9.86% | 7.75% | **None** (per-segment only) | Native MLX |

## Key Findings

### Qwen3-ASR-1.7B is the strongest candidate
- **50-70% CER reduction** vs Whisper on real-world Mandarin (WenetSpeech meeting: 5.88% vs 19.11%)
- Nearly equivalent English WER (1.63% vs 1.51%)
- Native MLX on Apple Silicon (documented M2 Max benchmarks at RTF 0.06)
- Supports 30 languages + 22 Chinese dialects
- Apache 2.0, models on HuggingFace

### SenseVoice-Small is uniquely valuable
- Only model with **per-word language ID output** — tells you which words are Chinese vs English
- 228 MB int8 ONNX — lightest model by far
- Could run as a fast language tagger alongside primary ASR

### Intra-sentence code-switching IS solvable
Unlike Whisper (one language per call), these models handle mixed-language sentences natively:
- Fun-ASR: 1.59% code-switching WER (offline)
- Paraformer: 3.70% CER on CS-Dialogue dataset
- SenseVoice: 6.71% MER on CS-Dialogue with per-word tagging

## Sources
- Qwen3-ASR Technical Report: arXiv:2601.21337
- SenseVoice: github.com/FunAudioLLM/SenseVoice
- Fun-ASR: arXiv:2509.12508v3
- FireRedASR: arXiv:2501.14350
- Paraformer-v2: arXiv:2409.17746
- CS-Dialogue benchmark: arXiv:2502.18913
