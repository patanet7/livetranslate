ecommended Module Breakdown:

  src/simul_whisper/
  ├── simul_whisper.py          (Reduced to ~200 lines - main coordinator)
  ├── model_loader.py            (NEW - Model initialization & device detection)
  ├── tokenizer_manager.py       (NEW - Tokenizer & language detection)
  ├── audio_buffer.py            (NEW - Audio segment management)
  ├── context_manager.py         (NEW - Context and token management)
  ├── decoder_engine.py          (NEW - Main decoding loop)
  ├── attention_processor.py     (NEW - Attention matrix processing)
  ├── cache_manager.py           (NEW - KV cache operations)
  └── debug_logger.py            (NEW - Logging utilities)

  Specific Responsibilities:

  model_loader.py (lines 36-77):
  - Device detection (CUDA/MPS/CPU)
  - Model loading
  - Hook installation for attention/KV cache
  - Alignment heads setup

  tokenizer_manager.py (lines 157-221, 369-399):
  - create_tokenizer()
  - set_task() - Dynamic task switching
  - lang_id() - Language detection
  - Token suppression logic

  audio_buffer.py (lines 327-356):
  - segments_len()
  - insert_audio()
  - _apply_minseglen()
  - Audio buffer rotation

  context_manager.py (lines 222-267):
  - init_context()
  - trim_context()
  - Context token management
  - Static/dynamic prompt handling

  decoder_engine.py (lines 404-703):
  - Main infer() loop
  - Token generation
  - Beam search coordination
  - Hypothesis generation

  attention_processor.py (lines 567-598):
  - Attention matrix extraction
  - Alignment head processing
  - Most-attended frame calculation
  - Median filtering

  cache_manager.py (lines 88-103, 358-367):
  - KV cache hooks
  - Cache cleanup (_clean_cache())
  - Cache state management

  debug_logger.py (lines 705-747):
  - logdir_save() - Debug output
  - debug_print_tokens()
  - WAV file export
⚠️ Duplicate eow_detection.py files:

  1. /src/eow_detection.py (301 lines) - Well-documented, modern, used by tests
  2. /src/simul_whisper/eow_detection.py (68 lines) - Minimal, used by simul_whisper.py

  Recommendation: Keep both for now (different use cases), but consider consolidating to /src/eow_detection.py and updating imports in simul_whisper.py.
