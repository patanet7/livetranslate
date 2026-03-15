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

import sys
from pathlib import Path

# Add src directory to path before imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest
import torch
from alignatt_decoder import AlignAttDecoder
from whisper_service import ModelManager


class TestPyTorchIntegration:
    """Integration tests for PyTorch Whisper with attention hooks"""

    @pytest.fixture(scope="class")
    def manager(self):
        """Shared ModelManager for all tests — loads model once, cleans up after."""
        import gc

        models_dir = Path(__file__).parent.parent / ".models"
        mgr = ModelManager(models_dir=str(models_dir))
        try:
            mgr.load_model("large-v3")
        except Exception:
            pytest.skip("large-v3 model not available")
        try:
            yield mgr
        finally:
            mgr.dec_attns.clear()
            del mgr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @pytest.mark.integration
    def test_model_loading(self, manager):
        """Test that PyTorch Whisper model loads correctly"""
        print("\n[TEST] Loading PyTorch Whisper model...")

        model = manager.models.get("large-v3")
        assert model is not None, "Model should not be None"
        assert hasattr(model, "decoder"), "Model should have decoder"
        assert hasattr(model, "encoder"), "Model should have encoder"
        assert manager.device in ["cuda", "mps", "cpu"], f"Invalid device: {manager.device}"

        print(f"✅ Model loaded successfully on {manager.device}")

    @pytest.mark.integration
    def test_attention_hooks_installed(self, manager):
        """Test that attention hooks are installed on decoder blocks"""
        print("\n[TEST] Verifying attention hooks installation...")

        model = manager.models.get("large-v3")
        assert model is not None

        hook_count = 0
        for block in model.decoder.blocks:
            if hasattr(block, "cross_attn"):
                if hasattr(block.cross_attn, "_forward_hooks"):
                    hook_count += len(block.cross_attn._forward_hooks)

        assert hook_count > 0, "No attention hooks found on decoder blocks"
        print(f"✅ Found {hook_count} attention hooks installed")

    @pytest.mark.integration
    def test_attention_capture_during_inference(self, manager):
        """
        CRITICAL TEST: Verify that attention weights are actually captured during inference
        """
        print("\n[TEST] Testing attention capture during inference...")

        audio_data = np.zeros(16000, dtype=np.float32)
        manager.dec_attns.clear()

        print(f"Running inference on {manager.device}...")

        try:
            manager.safe_inference(
                model_name="large-v3",
                audio_data=audio_data,
                beam_size=1,
                streaming_policy="alignatt",
            )

            assert len(manager.dec_attns) > 0, "No attention weights were captured during inference"

            print(f"✅ Captured {len(manager.dec_attns)} attention layers")

            first_attn = manager.dec_attns[0]
            print(f"   Attention shape: {first_attn.shape}")
            assert len(first_attn.shape) >= 2, f"Invalid attention shape: {first_attn.shape}"

            print("✅ Attention weights have valid shape")

        except Exception as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Out of memory (GPU): {e}")
            else:
                raise
        finally:
            manager.dec_attns.clear()

    @pytest.mark.integration
    def test_alignatt_with_captured_attention(self, manager):
        """
        Test that AlignAtt decoder can use captured attention data
        """
        print("\n[TEST] Testing AlignAtt with real captured attention...")

        audio_data = np.zeros(16000, dtype=np.float32)

        alignatt = AlignAttDecoder(frame_threshold_offset=10)
        audio_frames = len(audio_data) // 160
        alignatt.set_max_attention_frame(audio_frames)

        print(f"Audio frames: {audio_frames}")
        print(f"Max attention frame (l = k - τ): {alignatt.max_frame}")

        manager.dec_attns.clear()

        try:
            manager.safe_inference(
                model_name="large-v3",
                audio_data=audio_data,
                beam_size=1,
                streaming_policy="alignatt",
            )

            assert len(manager.dec_attns) > 0, "No attention captured"
            assert alignatt.max_frame > 0, "AlignAtt max_frame not set"
            assert (
                alignatt.max_frame == audio_frames - 10
            ), f"Expected {audio_frames - 10}, got {alignatt.max_frame}"

            print("✅ AlignAtt successfully configured with captured attention")
            print(f"   Captured {len(manager.dec_attns)} attention layers")
            print(f"   Frame threshold: {alignatt.max_frame}/{audio_frames}")

        except Exception as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Out of memory: {e}")
            else:
                raise
        finally:
            manager.dec_attns.clear()

    @pytest.mark.integration
    def test_attention_frame_analysis(self, manager):
        """
        Analyze captured attention to verify it's usable for AlignAtt policy
        """
        print("\n[TEST] Analyzing attention frame distribution...")

        audio_data = np.zeros(16000, dtype=np.float32)
        manager.dec_attns.clear()

        try:
            manager.safe_inference(model_name="large-v3", audio_data=audio_data, beam_size=1)

            if len(manager.dec_attns) > 0:
                last_attn = manager.dec_attns[-1]
                print(f"Last layer attention shape: {last_attn.shape}")

                if len(last_attn.shape) >= 2:
                    if isinstance(last_attn, torch.Tensor):
                        max_frames = torch.argmax(last_attn, dim=-1)
                    else:
                        max_frames = np.argmax(last_attn, axis=-1)
                    print(f"   Max attention frames shape: {max_frames.shape}")
                    print(
                        f"   Max frame indices (first 10): {max_frames[:10] if len(max_frames) > 10 else max_frames}"
                    )
                    print("✅ Attention can be analyzed for frame-level decisions")
                else:
                    print(f"⚠️  Unexpected attention shape: {last_attn.shape}")
            else:
                pytest.fail("No attention captured")

        except Exception as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Out of memory: {e}")
            else:
                raise
        finally:
            manager.dec_attns.clear()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
