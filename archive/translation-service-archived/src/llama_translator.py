#!/usr/bin/env python3
"""
Direct Llama 3.1 Translation Implementation

Uses transformers library directly for Llama models for more reliable translation
without vLLM compatibility issues.
"""

import time
from datetime import datetime
from typing import Any

import torch
import transformers
from livetranslate_common.logging import get_logger

logger = get_logger()


class LlamaTranslator:
    """Direct transformers-based translator for Llama models"""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.is_ready = False

        # Language mappings for better prompts
        self.language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "pt": "Portuguese",
            "it": "Italian",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
            "th": "Thai",
            "vi": "Vietnamese",
            "nl": "Dutch",
        }

        logger.info(f"🔧 Initializing Llama translator with model: {model_path}")
        logger.info(f"📱 Device: {device}")

    def initialize(self) -> bool:
        """Initialize the model and pipeline"""
        try:
            logger.info(f"📦 Loading Llama model from {self.model_path}...")

            # Create text generation pipeline
            if self.device == "auto":
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=self.model_path,
                    model_kwargs={
                        "torch_dtype": torch.bfloat16,
                        "local_files_only": True,  # Use local model files only
                    },
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=self.model_path,
                    model_kwargs={
                        "torch_dtype": torch.bfloat16 if self.device != "cpu" else torch.float32,
                        "local_files_only": True,  # Use local model files only
                    },
                    device=self.device,
                    trust_remote_code=True,
                )

            self.is_ready = True
            logger.info("✅ Llama translator initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Llama translator: {e}")
            return False

    def translate(
        self, text: str, source_lang: str = "en", target_lang: str = "es"
    ) -> dict[str, Any]:
        """Translate text using Llama model"""
        if not self.is_ready:
            return {
                "error": "Translator not initialized",
                "translated_text": text,
                "confidence": 0.0,
            }

        start_time = time.time()

        try:
            # Map language codes to full names
            source_name = self.language_names.get(source_lang, source_lang)
            target_name = self.language_names.get(target_lang, target_lang)

            logger.debug(f"Translating: {text[:50]}... ({source_name} -> {target_name})")

            # Create translation prompt
            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert translator. Translate text from {source_name} to {target_name}. "
                    f"Provide only the translation without any explanation or additional text. "
                    f"Maintain the original meaning and tone.",
                },
                {
                    "role": "user",
                    "content": f"Translate this {source_name} text to {target_name}: {text}",
                },
            ]

            # Generate translation
            outputs = self.pipeline(
                messages,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )

            # Extract the translation from the generated text
            generated_text = outputs[0]["generated_text"]
            # Get the last message (assistant's response)
            translation = generated_text[-1]["content"].strip()

            # Clean up the translation
            translation = self._clean_translation(translation, text)

            processing_time = time.time() - start_time

            return {
                "translated_text": translation,
                "original": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "confidence_score": 0.9,  # Llama is generally high quality
                "processing_time": processing_time,
                "model_used": self.model_path,
                "backend_used": "llama_transformers",
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
                "backend_used": "llama_transformers",
            }

    def _clean_translation(self, translation: str, original_text: str) -> str:
        """Clean up the translation output"""
        # Remove common prefixes and artifacts
        prefixes_to_remove = [
            "Translation:",
            "translation:",
            "TRANSLATION:",
            "Here is the translation:",
            "The translation is:",
            f'"{original_text}" in',
            "translates to:",
            "becomes:",
            "In Spanish:",
            "In French:",
            "In German:",
            "In Chinese:",
            "In Japanese:",
            "In Korean:",
            "In Portuguese:",
            "In Italian:",
            "In Russian:",
            "In Arabic:",
            "In Hindi:",
            "In Thai:",
            "In Vietnamese:",
            "In Dutch:",
        ]

        cleaned = translation.strip()

        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()
                break

        # Remove quotes if the entire response is quoted
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()

        # Remove trailing punctuation if it's redundant
        if cleaned.endswith(".") and not original_text.endswith("."):
            cleaned = cleaned[:-1].strip()

        return cleaned

    def get_supported_languages(self) -> dict[str, str]:
        """Get supported language codes"""
        return self.language_names.copy()

    def health_check(self) -> dict[str, Any]:
        """Check translator health"""
        return {
            "status": "healthy" if self.is_ready else "not_ready",
            "model_path": self.model_path,
            "device": self.device,
            "pipeline_loaded": self.pipeline is not None,
            "supported_languages": len(self.language_names),
        }


# Global translator instance
_llama_translator: LlamaTranslator | None = None


def get_llama_translator(model_path: str, device: str = "auto") -> LlamaTranslator:
    """Get or create global Llama translator instance"""
    global _llama_translator

    if _llama_translator is None:
        _llama_translator = LlamaTranslator(model_path, device)
        _llama_translator.initialize()

    return _llama_translator


if __name__ == "__main__":
    # Test the translator
    translator = LlamaTranslator("./models/Llama-3.1-8B-Instruct")
    if translator.initialize():
        result = translator.translate("Hello world", "en", "es")
        print(f"Translation: {result}")
    else:
        print("Failed to initialize translator")
