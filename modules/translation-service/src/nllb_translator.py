#!/usr/bin/env python3
"""
Direct NLLB Translation Implementation

Uses transformers library directly for NLLB models since they are encoder-decoder
models that are not compatible with vLLM.
"""

import logging
import time
from datetime import datetime
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class NLLBTranslator:
    """Direct transformers-based translator for NLLB models"""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.is_ready = False

        # NLLB language code mapping
        self.lang_map = {
            "en": "eng_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "pt": "por_Latn",
            "it": "ita_Latn",
            "ru": "rus_Cyrl",
            "ar": "arb_Arab",
            "hi": "hin_Deva",
            "th": "tha_Thai",
            "vi": "vie_Latn",
            "nl": "nld_Latn",
        }

        logger.info(f"🔧 Initializing NLLB translator with model: {model_path}")
        logger.info(f"📱 Device: {self.device}")

    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def initialize(self) -> bool:
        """Initialize the model and tokenizer"""
        try:
            logger.info(f"📦 Loading tokenizer from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)

            logger.info(f"🤖 Loading model from {self.model_path}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            self.is_ready = True
            logger.info("✅ NLLB translator initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize NLLB translator: {e}")
            return False

    def translate(
        self, text: str, source_lang: str = "en", target_lang: str = "es"
    ) -> dict[str, Any]:
        """Translate text using NLLB model"""
        if not self.is_ready:
            return {
                "error": "Translator not initialized",
                "translated_text": text,
                "confidence": 0.0,
            }

        start_time = time.time()

        try:
            # Map language codes to NLLB format
            src_code = self.lang_map.get(source_lang, "eng_Latn")
            tgt_code = self.lang_map.get(target_lang, "spa_Latn")

            logger.debug(f"Translating: {text[:50]}... ({src_code} -> {tgt_code})")

            # Tokenize input
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)

            # Set source language
            self.tokenizer.src_lang = src_code

            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                    max_length=512,
                    num_beams=5,
                    length_penalty=1.0,
                    early_stopping=True,
                )

            # Decode the translation
            translated_text = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]

            processing_time = time.time() - start_time

            return {
                "translated_text": translated_text,
                "original": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "confidence_score": 0.9,  # NLLB is generally high quality
                "processing_time": processing_time,
                "model_used": self.model_path,
                "backend_used": "nllb_transformers",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {
                "error": str(e),
                "translated_text": text,
                "original": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "confidence_score": 0.0,
                "processing_time": time.time() - start_time,
                "model_used": self.model_path,
                "backend_used": "nllb_transformers",
            }

    def get_supported_languages(self) -> dict[str, str]:
        """Get supported language codes"""
        return {code: f"{code}_language" for code in self.lang_map}

    def health_check(self) -> dict[str, Any]:
        """Check translator health"""
        return {
            "status": "healthy" if self.is_ready else "not_ready",
            "model_path": self.model_path,
            "device": self.device,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "supported_languages": len(self.lang_map),
        }


# Global translator instance
_nllb_translator: NLLBTranslator | None = None


def get_nllb_translator(model_path: str, device: str = "auto") -> NLLBTranslator:
    """Get or create global NLLB translator instance"""
    global _nllb_translator

    if _nllb_translator is None:
        _nllb_translator = NLLBTranslator(model_path, device)
        _nllb_translator.initialize()

    return _nllb_translator


if __name__ == "__main__":
    # Test the translator (use NLLB model for testing)
    translator = NLLBTranslator("./models/nllb-200-distilled-1.3B-8bit")
    if translator.initialize():
        result = translator.translate("Hello world", "en", "es")
        print(f"Translation: {result}")
    else:
        print("Failed to initialize translator")
