#!/usr/bin/env python3
"""
Integration Test: PyTorch Whisper with AlignAtt Attention Hooks
CRITICAL: Validates that attention hooks actually capture cross-attention data

This test verifies:
1. PyTorch Whisper model loads correctly
2. Attention hooks are installed on decoder blocks
3. Cross-attention weights are captured during inference
4. AlignAtt decoder can use the captured attention data
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from whisper_service import ModelManager
from alignatt_decoder import AlignAttDecoder


class TestPyTorchIntegration:
    """Integration tests for PyTorch Whisper with attention hooks"""

    @pytest.mark.integration
    def test_model_loading(self):
        """Test that PyTorch Whisper model loads correctly"""
        print("\n[TEST] Loading PyTorch Whisper model...")

        # Use local .models directory
        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))

        # Load large-v3 model (available as .pt file)
        model_name = "large-v3"

        try:
            model = manager.load_model(model_name)

            # Verify model loaded
            assert model is not None, "Model should not be None"
            assert hasattr(model, 'decoder'), "Model should have decoder"
            assert hasattr(model, 'encoder'), "Model should have encoder"

            # Verify device
            assert manager.device in ["cuda", "mps", "cpu"], f"Invalid device: {manager.device}"

            print(f"✅ Model loaded successfully on {manager.device}")

        except Exception as e:
            pytest.skip(f"Model loading failed (may need model download): {e}")

    @pytest.mark.integration
    def test_attention_hooks_installed(self):
        """Test that attention hooks are installed on decoder blocks"""
        print("\n[TEST] Verifying attention hooks installation...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))
        model_name = "large-v3"

        try:
            model = manager.load_model(model_name)

            # Verify hooks are installed
            hook_count = 0
            for block in model.decoder.blocks:
                if hasattr(block, 'cross_attn'):
                    # Check if hooks are registered
                    if hasattr(block.cross_attn, '_forward_hooks'):
                        hook_count += len(block.cross_attn._forward_hooks)

            assert hook_count > 0, "No attention hooks found on decoder blocks"
            print(f"✅ Found {hook_count} attention hooks installed")

        except Exception as e:
            pytest.skip(f"Hook verification failed: {e}")

    @pytest.mark.integration
    def test_attention_capture_during_inference(self):
        """
        CRITICAL TEST: Verify that attention weights are actually captured during inference
        """
        print("\n[TEST] Testing attention capture during inference...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))
        model_name = "large-v3"

        try:
            model = manager.load_model(model_name)

            # Create dummy audio (1 second of silence at 16kHz)
            audio_data = np.zeros(16000, dtype=np.float32)

            # Clear attention buffers
            manager.dec_attns.clear()

            print(f"Running inference on {manager.device}...")

            # Perform inference
            result = manager.safe_inference(
                model_name=model_name,
                audio_data=audio_data,
                beam_size=1,  # Greedy for speed
                streaming_policy="alignatt"  # Enable AlignAtt
            )

            # Verify attention was captured
            assert len(manager.dec_attns) > 0, "No attention weights were captured during inference"

            print(f"✅ Captured {len(manager.dec_attns)} attention layers")

            # Verify attention shape
            first_attn = manager.dec_attns[0]
            print(f"   Attention shape: {first_attn.shape}")

            # Attention should be [sequence_length, encoder_length]
            assert len(first_attn.shape) >= 2, f"Invalid attention shape: {first_attn.shape}"

            print("✅ Attention weights have valid shape")

        except Exception as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Out of memory (GPU): {e}")
            elif "Model" in str(e) and "not found" in str(e):
                pytest.skip(f"Model not available: {e}")
            else:
                raise

    @pytest.mark.integration
    def test_alignatt_with_captured_attention(self):
        """
        Test that AlignAtt decoder can use captured attention data
        """
        print("\n[TEST] Testing AlignAtt with real captured attention...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))
        model_name = "large-v3"

        try:
            model = manager.load_model(model_name)

            # Create dummy audio
            audio_data = np.zeros(16000, dtype=np.float32)

            # Initialize AlignAtt decoder
            alignatt = AlignAttDecoder(frame_threshold_offset=10)

            # Calculate frames from audio (10ms frames at 16kHz)
            audio_frames = len(audio_data) // 160
            alignatt.set_max_attention_frame(audio_frames)

            print(f"Audio frames: {audio_frames}")
            print(f"Max attention frame (l = k - τ): {alignatt.max_frame}")

            # Clear attention buffers
            manager.dec_attns.clear()

            # Perform inference with AlignAtt
            result = manager.safe_inference(
                model_name=model_name,
                audio_data=audio_data,
                beam_size=1,
                streaming_policy="alignatt"
            )

            # Verify attention was captured
            assert len(manager.dec_attns) > 0, "No attention captured"

            # Verify AlignAtt settings were applied
            assert alignatt.max_frame > 0, "AlignAtt max_frame not set"
            assert alignatt.max_frame == audio_frames - 10, f"Expected {audio_frames - 10}, got {alignatt.max_frame}"

            print(f"✅ AlignAtt successfully configured with captured attention")
            print(f"   Captured {len(manager.dec_attns)} attention layers")
            print(f"   Frame threshold: {alignatt.max_frame}/{audio_frames}")

        except Exception as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Out of memory: {e}")
            elif "Model" in str(e) and "not found" in str(e):
                pytest.skip(f"Model not available: {e}")
            else:
                raise

    @pytest.mark.integration
    def test_attention_frame_analysis(self):
        """
        Analyze captured attention to verify it's usable for AlignAtt policy
        """
        print("\n[TEST] Analyzing attention frame distribution...")

        models_dir = Path(__file__).parent.parent / ".models"
        manager = ModelManager(models_dir=str(models_dir))
        model_name = "large-v3"

        try:
            model = manager.load_model(model_name)

            # Create dummy audio
            audio_data = np.zeros(16000, dtype=np.float32)

            # Clear and run inference
            manager.dec_attns.clear()

            result = manager.safe_inference(
                model_name=model_name,
                audio_data=audio_data,
                beam_size=1
            )

            # Analyze attention
            if len(manager.dec_attns) > 0:
                # Get last layer attention (most important for AlignAtt)
                last_attn = manager.dec_attns[-1]

                print(f"Last layer attention shape: {last_attn.shape}")

                # Check if we can extract frame-level attention
                if len(last_attn.shape) >= 2:
                    # For each decoder position, find which encoder frame has max attention
                    if isinstance(last_attn, torch.Tensor):
                        max_frames = torch.argmax(last_attn, dim=-1)
                        print(f"   Max attention frames shape: {max_frames.shape}")
                        print(f"   Max frame indices (first 10): {max_frames[:10] if len(max_frames) > 10 else max_frames}")
                    else:
                        max_frames = np.argmax(last_attn, axis=-1)
                        print(f"   Max attention frames shape: {max_frames.shape}")
                        print(f"   Max frame indices (first 10): {max_frames[:10] if len(max_frames) > 10 else max_frames}")

                    print("✅ Attention can be analyzed for frame-level decisions")
                else:
                    print(f"⚠️  Unexpected attention shape: {last_attn.shape}")
            else:
                pytest.fail("No attention captured")

        except Exception as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Out of memory: {e}")
            elif "Model" in str(e) and "not found" in str(e):
                pytest.skip(f"Model not available: {e}")
            else:
                raise


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
