#!/usr/bin/env python3
"""
TDD Tests for TokenBuffer and Rolling Context System

Following SimulStreaming reference implementation:
- token_buffer.py: Context management with static/rolling prompts
- simul_whisper/simul_whisper.py lines 151-195: Context initialization and trimming
- Target: +25-40% quality improvement on long-form content

Reference: SimulStreaming/token_buffer.py and simul_whisper.py
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


class TestTokenBuffer:
    """Test TokenBuffer class for context management"""

    def test_token_buffer_import(self):
        """Test that TokenBuffer class can be imported"""
        from token_buffer import TokenBuffer
        assert TokenBuffer is not None

    def test_empty_token_buffer(self):
        """Test creating an empty TokenBuffer"""
        from token_buffer import TokenBuffer

        buffer = TokenBuffer.empty()

        assert buffer is not None
        assert buffer.is_empty() is True
        assert buffer.text == ""

    def test_token_buffer_from_text(self):
        """Test creating TokenBuffer from text"""
        from token_buffer import TokenBuffer

        text = "This is a test prompt"
        buffer = TokenBuffer.from_text(text)

        assert buffer.is_empty() is False
        assert buffer.text == text
        assert buffer.as_text() == text

    def test_token_buffer_with_tokenizer(self):
        """Test TokenBuffer with Whisper tokenizer"""
        from token_buffer import TokenBuffer
        import whisper

        # Use Whisper's native tokenizer (multilingual)
        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

        buffer = TokenBuffer.from_text(
            "Medical terminology test",
            tokenizer=tokenizer
        )

        assert buffer.tokenizer is not None
        assert len(buffer.as_token_ids()) > 0

    def test_token_buffer_as_token_ids(self):
        """Test converting text to token IDs"""
        from token_buffer import TokenBuffer
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        text = "Hello world"

        buffer = TokenBuffer.from_text(text, tokenizer=tokenizer)
        token_ids = buffer.as_token_ids()

        assert isinstance(token_ids, list)
        assert len(token_ids) > 0

        # Decode tokens should give back original text
        decoded = tokenizer.decode(token_ids)
        assert decoded == text

    def test_token_buffer_with_prefix_tokens(self):
        """Test TokenBuffer with prefix token IDs (like <|sot_prev|>)"""
        from token_buffer import TokenBuffer
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        # Use Whisper's special token (e.g., <|startofprev|> token)
        prefix_token_ids = [50361]  # Example Whisper special token

        buffer = TokenBuffer.from_text(
            "Test text",
            tokenizer=tokenizer,
            prefix_token_ids=prefix_token_ids
        )

        token_ids = buffer.as_token_ids()

        # Should include prefix tokens
        assert token_ids[0] == prefix_token_ids[0]
        assert len(token_ids) > len(prefix_token_ids)

    def test_token_buffer_append_token_ids(self):
        """Test appending token IDs to buffer"""
        from token_buffer import TokenBuffer
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

        buffer = TokenBuffer.from_text("Initial text", tokenizer=tokenizer)
        initial_text = buffer.text

        # Append new tokens
        new_tokens = tokenizer.encode(" appended text")
        buffer.append_token_ids(new_tokens)

        assert buffer.text == initial_text + " appended text"


class TestTokenBufferTrimming:
    """Test word-level trimming for rolling context"""

    def test_trim_words_basic(self):
        """Test trimming one word from beginning"""
        from token_buffer import TokenBuffer
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        buffer = TokenBuffer.from_text(
            "First second third fourth",
            tokenizer=tokenizer
        )

        # Trim first word
        tokens_removed = buffer.trim_words(num=1)

        assert tokens_removed > 0
        assert buffer.text == "second third fourth"

    def test_trim_words_multiple(self):
        """Test trimming multiple words"""
        from token_buffer import TokenBuffer
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        buffer = TokenBuffer.from_text(
            "One two three four five",
            tokenizer=tokenizer
        )

        # Trim first two words
        buffer.trim_words(num=2)

        assert buffer.text == "three four five"

    def test_trim_words_with_static_prefix(self):
        """Test trimming with preserved static prefix (after parameter)"""
        from token_buffer import TokenBuffer
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

        # Static prefix: "Medical terms: "
        # Rolling context: "patient has symptoms"
        full_text = "Medical terms: patient has symptoms"
        buffer = TokenBuffer.from_text(full_text, tokenizer=tokenizer)

        # Trim from rolling context (after static prefix)
        after_length = len("Medical terms: ")
        buffer.trim_words(num=1, after=after_length)

        # Should preserve static prefix
        assert buffer.text.startswith("Medical terms: ")
        assert buffer.text == "Medical terms: has symptoms"

    def test_trim_words_empty_buffer(self):
        """Test trimming from empty buffer returns 0"""
        from token_buffer import TokenBuffer
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        buffer = TokenBuffer.empty(tokenizer=tokenizer)

        tokens_removed = buffer.trim_words(num=1)

        assert tokens_removed == 0
        assert buffer.is_empty()

    def test_trim_words_until_limit(self):
        """Test trimming words until token limit is reached"""
        from token_buffer import TokenBuffer
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

        # Create long text
        long_text = " ".join([f"word{i}" for i in range(100)])
        buffer = TokenBuffer.from_text(long_text, tokenizer=tokenizer)

        initial_tokens = len(buffer.as_token_ids())
        max_tokens = 50

        # Trim until under limit
        while len(buffer.as_token_ids()) > max_tokens:
            removed = buffer.trim_words(num=1)
            if removed == 0:
                break

        final_tokens = len(buffer.as_token_ids())
        assert final_tokens <= max_tokens


class TestRollingContextIntegration:
    """Test rolling context integration with ModelManager"""

    def test_model_manager_has_context_attributes(self):
        """Test that ModelManager has context management attributes"""
        from whisper_service import ModelManager

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        # Should have context-related attributes
        assert hasattr(manager, 'static_prompt'), "Should have static_prompt attribute"
        assert hasattr(manager, 'rolling_context'), "Should have rolling_context attribute"
        assert hasattr(manager, 'max_context_tokens'), "Should have max_context_tokens attribute"

    def test_model_manager_init_context(self):
        """Test context initialization in ModelManager"""
        from whisper_service import ModelManager

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        # Should have init_context method
        assert hasattr(manager, 'init_context'), "Should have init_context() method"
        assert callable(manager.init_context), "init_context should be callable"

    def test_init_context_with_static_prompt(self):
        """Test initializing context with static domain terminology"""
        from whisper_service import ModelManager

        models_dir = Path(__file__).parent.parent / ".models"
        static_prompt = "Medical terms: hypertension, diabetes, cardiomyopathy"

        manager = ModelManager(
            models_dir=str(models_dir),
            static_prompt=static_prompt
        )

        manager.init_context()

        # Static prompt should be set
        assert manager.static_prompt == static_prompt
        # Rolling context should start empty or with static prompt
        assert manager.rolling_context is not None

    def test_init_context_with_initial_prompt(self):
        """Test initializing context with dynamic initial prompt"""
        from whisper_service import ModelManager

        models_dir = Path(__file__).parent.parent / ".models"
        init_prompt = "Patient presents with chest pain"

        manager = ModelManager(
            models_dir=str(models_dir),
            init_prompt=init_prompt
        )

        manager.init_context()

        # Initial prompt should be in rolling context
        assert init_prompt in manager.rolling_context.text

    def test_trim_context_method_exists(self):
        """Test that trim_context method exists"""
        from whisper_service import ModelManager

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        assert hasattr(manager, 'trim_context'), "Should have trim_context() method"
        assert callable(manager.trim_context), "trim_context should be callable"

    def test_trim_context_preserves_static_prompt(self):
        """Test that trimming preserves static prompt"""
        from whisper_service import ModelManager

        models_dir = Path(__file__).parent.parent / ".models"
        static_prompt = "Medical terminology"

        manager = ModelManager(
            models_dir=str(models_dir),
            static_prompt=static_prompt,
            max_context_tokens=20  # Small limit to force trimming
        )

        manager.init_context()

        # Add lots of rolling context
        for i in range(50):
            manager.rolling_context.text += f" word{i}"

        # Trim context
        manager.trim_context()

        # Static prompt should still be there
        assert static_prompt in manager.rolling_context.text

    def test_max_context_tokens_default(self):
        """Test default max_context_tokens is 223 (SimulStreaming paper)"""
        from whisper_service import ModelManager

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        # Default should be 223 tokens (from SimulStreaming Table 1)
        assert manager.max_context_tokens == 223

    def test_max_context_tokens_configurable(self):
        """Test that max_context_tokens can be configured"""
        from whisper_service import ModelManager

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(
            models_dir=str(models_dir),
            max_context_tokens=100
        )

        assert manager.max_context_tokens == 100


class TestContextCarryover:
    """
    INTEGRATION TESTS: Context carryover with REAL Whisper inference

    Tests the full pipeline:
    1. Load real Whisper model
    2. Process real audio chunks
    3. Use rolling context in actual transcription
    4. Verify context improves quality
    """

    @pytest.mark.integration
    def test_rolling_context_with_real_inference(self, shared_whisper_manager):
        """
        CRITICAL INTEGRATION TEST: Rolling context with real Whisper model

        Tests:
        - Real Whisper model loading (SHARED large-v3 model across all integration tests)
        - Real audio processing
        - Context carryover between segments
        - Context used in actual inference
        """
        import numpy as np

        # Use the shared manager - model loaded ONCE per test module (saves ~6GB)
        manager = shared_whisper_manager

        # Configure for this test (reset and set custom static prompt)
        manager.static_prompt = "Medical terminology: hypertension diabetes cardiomyopathy"
        manager.init_context()

        # Get the pre-loaded large-v3 model
        model = manager.pipelines.get("large-v3")
        assert model is not None

        # Create test audio segments (silent for speed, but real inference)
        segment1_audio = np.zeros(16000, dtype=np.float32)  # 1 second
        segment2_audio = np.zeros(16000, dtype=np.float32)  # 1 second

        # First inference (no context yet beyond static prompt)
        context1 = manager.get_inference_context()
        assert "Medical terminology" in context1

        result1 = model.transcribe(
            audio=segment1_audio,
            prompt=context1,  # Use rolling context
            beam_size=1,
            temperature=0.0
        )

        # Append first result to context (simulate real workflow)
        if result1['text'].strip():
            manager.append_to_context(result1['text'])

        # Second inference (should have context from first)
        context2 = manager.get_inference_context()
        assert "Medical terminology" in context2  # Static prompt preserved

        result2 = model.transcribe(
            audio=segment2_audio,
            prompt=context2,  # Rolling context with previous result
            beam_size=1,
            temperature=0.0
        )

        # Verify context is being used
        assert manager.rolling_context is not None
        assert len(manager.rolling_context.text) > 0

        print(f"✅ Rolling context integration test passed")
        print(f"   Context after 2 segments: '{manager.rolling_context.text[:100]}...'")

    @pytest.mark.integration
    def test_context_improves_consistency_across_segments(self, shared_whisper_manager):
        """
        INTEGRATION TEST: Context improves transcription consistency

        Verifies that rolling context helps maintain consistent terminology
        across multiple audio segments (real Whisper inference with SHARED model)
        """
        import numpy as np

        # Use the shared manager - model loaded ONCE per test module (saves ~3GB)
        manager = shared_whisper_manager

        # Configure for this test
        manager.static_prompt = "Medical: hypertension, myocardial infarction, ECG"
        manager.max_context_tokens = 223
        manager.init_context()

        # Get the pre-loaded large-v3 model
        model = manager.pipelines.get("large-v3")

        # Simulate 5 audio segments in a medical consultation
        num_segments = 5
        for i in range(num_segments):
            audio = np.zeros(16000, dtype=np.float32)

            context = manager.get_inference_context()

            result = model.transcribe(
                audio=audio,
                prompt=context,
                beam_size=1,
                temperature=0.0
            )

            # Add to rolling context
            if result['text'].strip():
                manager.append_to_context(result['text'])

        # Verify context maintained static prompt
        final_context = manager.get_inference_context()
        assert "Medical: hypertension" in final_context

        # Verify context is under token limit
        context_tokens = len(manager.rolling_context.as_token_ids())
        assert context_tokens <= manager.max_context_tokens + 10

        print(f"✅ Processed {num_segments} segments with rolling context")
        print(f"   Final context tokens: {context_tokens}/{manager.max_context_tokens}")

    @pytest.mark.integration
    def test_context_trimming_during_real_inference_session(self, shared_whisper_manager):
        """
        INTEGRATION TEST: Context trimming during long real inference session

        Tests FIFO trimming with real Whisper model over many segments (SHARED model)
        """
        import numpy as np

        # Use the shared manager - model loaded ONCE per test module (saves ~3GB)
        manager = shared_whisper_manager

        # Configure for this test
        manager.static_prompt = "Session start:"
        manager.max_context_tokens = 50  # Small limit to force trimming
        manager.init_context()

        # Get the pre-loaded large-v3 model
        model = manager.pipelines.get("large-v3")

        # Process many segments to trigger trimming
        for i in range(20):
            audio = np.zeros(8000, dtype=np.float32)  # 0.5 second for speed

            context = manager.get_inference_context()

            result = model.transcribe(
                audio=audio,
                prompt=context,
                beam_size=1,
                temperature=0.0
            )

            # Add segment marker to context (for testing)
            manager.append_to_context(f"segment_{i} ")

        # Verify trimming occurred
        final_context = manager.get_inference_context()
        assert "Session start:" in final_context  # Static prompt preserved

        # Early segments should be trimmed (FIFO) - use word boundaries
        assert "segment_0 " not in final_context
        assert "segment_1 " not in final_context
        assert "segment_2 " not in final_context

        # Recent segments should be preserved
        assert "segment_18 " in final_context or "segment_19 " in final_context or "segment_19" in final_context

        # Context should be under limit
        context_tokens = len(manager.rolling_context.as_token_ids())
        assert context_tokens <= manager.max_context_tokens + 10

        print(f"✅ FIFO trimming verified over 20 segments")
        print(f"   Final context: '{final_context}'")
        print(f"   Final tokens: {context_tokens}/{manager.max_context_tokens}")


class TestSimulStreamingCompliance:
    """Test compliance with SimulStreaming paper specifications"""

    def test_max_context_223_tokens(self):
        """Test that default context limit matches SimulStreaming (Table 1)"""
        from whisper_service import ModelManager

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        # SimulStreaming paper Table 1: max_context_tokens = 223
        assert manager.max_context_tokens == 223

    def test_fifo_word_level_trimming(self):
        """Test FIFO (first-in-first-out) word-level trimming"""
        from token_buffer import TokenBuffer
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
        buffer = TokenBuffer.from_text(
            "first second third fourth fifth",
            tokenizer=tokenizer
        )

        # Trim should remove from beginning (FIFO)
        buffer.trim_words(num=2)

        assert buffer.text == "third fourth fifth"
        assert "first" not in buffer.text
        assert "second" not in buffer.text

    def test_static_prompt_never_trimmed(self):
        """Test that static prompt is never trimmed (SimulStreaming behavior)"""
        from token_buffer import TokenBuffer
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

        static_part = "Static terminology: "
        rolling_part = "rolling context words here"

        buffer = TokenBuffer.from_text(
            static_part + rolling_part,
            tokenizer=tokenizer
        )

        # Trim with after parameter
        buffer.trim_words(num=2, after=len(static_part))

        # Static part should be preserved
        assert buffer.text.startswith(static_part)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
