# This code was originally in simul_whisper/transcriber/simul_whisper.py . It is adapted a lot for SimulStreaming.

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class SimulWhisperConfig:
    '''Options that are common for all simul policies that could be implemented in SimulWhisper.'''
    model_path: str
    language: str = field(default="zh")
    nonspeech_prob: float = 1.0
    audio_min_len: float = 1.0
    decoder_type: Literal["greedy","beam"] = "greedy"
    beam_size: int = 5
    task: Literal["transcribe","translate"] = "transcribe"
    init_prompt: str = field(default=None)
    static_init_prompt: str = field(default=None)
    max_context_tokens: int = field(default=None)

    logdir: str = field(default="logdir", metadata={"help": "Directory to save audio segments and tokens for debugging purposes."})

@dataclass
class AlignAttConfig(SimulWhisperConfig):
    '''Options specific to the AlignAtt policy.'''
    eval_data_path: str = "tmp"
    segment_length: float = field(default=1.0, metadata = {"help": "in second"})
    frame_threshold: int = 4
    rewind_threshold: int = 200 # in frames. Max value is 1500. Higher value turns rewinds off.
    audio_max_len: float = 30.0
    cif_ckpt_path: str = ""
    never_fire: bool = False

    # Code-switching configuration (Phase 1)
    sliding_lid_window: float = field(default=0.9, metadata={"help": "Sliding window size for language detection (seconds)"})
    sustained_lang_duration: float = field(default=3.0, metadata={"help": "Duration language must be sustained before SOT reset (seconds)"})
    sustained_lang_min_silence: float = field(default=0.25, metadata={"help": "Minimum VAD silence required for SOT reset (seconds)"})
    soft_bias_enabled: bool = field(default=False, metadata={"help": "Enable soft bias token injection (experimental)"})
    token_dedup_enabled: bool = field(default=True, metadata={"help": "Enable token deduplication at chunk boundaries"})
    confidence_threshold: float = field(default=0.6, metadata={"help": "Confidence threshold for n-best rescoring"})

    # VAD configuration
    vad_min_speech_ms: int = field(default=120, metadata={"help": "Minimum speech duration (ms)"})
    vad_min_silence_ms: int = field(default=250, metadata={"help": "Minimum silence duration (ms)"})