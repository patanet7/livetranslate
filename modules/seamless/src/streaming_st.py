import base64
import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor, SeamlessM4TForSpeechToText


@dataclass
class StreamingConfig:
    source_lang: str = "cmn"  # Mandarin Chinese
    target_lang: str = "eng"  # English
    sample_rate: int = 16000
    device: str = os.getenv("DEVICE", "cpu")
    model_name: str = os.getenv("SEAMLESS_MODEL", "facebook/seamless-m4t-v2-large")
    max_seconds_per_inference: float = 3.0  # window size for partial inference


class StreamingTranslator:
    """
    Windowed S2TT translator using Seamless M4T via HuggingFace transformers.
    Not true internal streaming, but performs rolling-window inference to emit partials.
    """

    def __init__(self, config: Optional[StreamingConfig] = None) -> None:
        self.config = config or StreamingConfig()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_partial_text: str = ""

        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model = SeamlessM4TForSpeechToText.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        self.model.eval()

        # Prepare decoder prompt ids for translation to target language
        self._decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.config.target_lang,
            task="translate",
        )

    def add_audio_pcm16(self, pcm16_bytes: bytes) -> None:
        # Convert PCM16 little-endian to float32 [-1, 1]
        int16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        if int16.size == 0:
            return
        float32 = (int16.astype(np.float32)) / 32768.0
        # Append to buffer
        self._buffer = np.concatenate([self._buffer, float32])

    def _window_for_inference(self) -> np.ndarray:
        # Take the last N seconds for partial inference
        max_samples = int(self.config.sample_rate * self.config.max_seconds_per_inference)
        if self._buffer.size <= max_samples:
            return self._buffer
        return self._buffer[-max_samples:]

    @torch.inference_mode()
    def infer_partial(self) -> Optional[str]:
        audio = self._window_for_inference()
        if audio.size == 0:
            return None
        inputs = self.processor(
            audio=audio, sampling_rate=self.config.sample_rate, return_tensors="pt"
        ).to(self.config.device)
        generated_tokens = self.model.generate(
            **inputs,
            forced_decoder_ids=self._decoder_ids,
            max_new_tokens=64,
        )
        text = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        text = text.strip()
        if text and text != self._last_partial_text:
            self._last_partial_text = text
            return text
        return None

    @torch.inference_mode()
    def infer_final(self) -> str:
        if self._buffer.size == 0:
            return ""
        inputs = self.processor(
            audio=self._buffer, sampling_rate=self.config.sample_rate, return_tensors="pt"
        ).to(self.config.device)
        generated_tokens = self.model.generate(
            **inputs,
            forced_decoder_ids=self._decoder_ids,
            max_new_tokens=256,
        )
        text = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return text.strip()


