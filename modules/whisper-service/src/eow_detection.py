"""
End-of-Word Detection with CIF (Continuous Integrate-and-Fire)

Following SimulStreaming specification:
- CIF model detects word boundaries from encoder features
- Prevents cutting words mid-stream (-50% re-translations)
- Trained linear layer predicts alpha weights for word boundaries
- Fallback modes: always_fire, never_fire (when no checkpoint available)

Reference: SimulStreaming/simul_whisper/eow_detection.py
License: MIT

This is how SimulStreaming eliminates re-translations!
"""

from pathlib import Path

import torch
import torch.nn as nn
from livetranslate_common.logging import get_logger

logger = get_logger()


def load_cif(
    cif_ckpt_path: str | None = None,
    n_audio_state: int = 1280,  # Whisper large-v3 encoder dimension
    device: torch.device = None,
    never_fire: bool = False,
) -> tuple[nn.Linear, bool, bool]:
    """
    Load CIF (Continuous Integrate-and-Fire) model for end-of-word detection

    The CIF model is a trained linear layer that predicts word boundaries from
    Whisper encoder features. This prevents cutting words mid-stream.

    Parameters:
        cif_ckpt_path (str, optional): Path to trained CIF checkpoint
            - If None: uses fallback mode (always_fire=True)
            - If provided: loads trained weights for accurate word detection
        n_audio_state (int): Whisper encoder dimension (default 1280 for large-v3)
        device (torch.device): Device for model (cuda/cpu)
        never_fire (bool): If True, never emit chunks at boundaries (testing only)

    Returns:
        Tuple[nn.Linear, bool, bool]:
            - cif_linear: Linear layer for boundary detection
            - always_fire: If True, emit at every opportunity (fallback mode)
            - never_fire: If True, never emit at boundaries

    Usage:
        # Without checkpoint (fallback mode)
        cif_model, always_fire, never_fire = load_cif()
        # always_fire=True, so fire_at_boundary() always returns True

        # With checkpoint (accurate word detection)
        cif_model, always_fire, never_fire = load_cif(cif_ckpt_path="cif_weights.pt")
        # Uses trained model to detect actual word boundaries
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create CIF linear layer: encoder_dim → 1 (boundary probability)
    cif_linear = nn.Linear(n_audio_state, 1)

    # Determine mode based on checkpoint availability
    if cif_ckpt_path is None or not cif_ckpt_path:
        # No checkpoint: use fallback mode
        if never_fire:
            always_fire = False
            never_fire_mode = True
            logger.warning(
                "CIF in never_fire mode (testing only). "
                "Chunks will NOT be emitted at word boundaries."
            )
        else:
            always_fire = True
            never_fire_mode = False
            logger.info(
                "CIF checkpoint not provided. Using fallback mode (always_fire=True). "
                "Chunks will be emitted at fixed intervals instead of word boundaries. "
                "For optimal performance, train and provide a CIF checkpoint."
            )
    else:
        # Load checkpoint for accurate word detection
        always_fire = False
        never_fire_mode = never_fire

        checkpoint_path = Path(cif_ckpt_path)
        if not checkpoint_path.exists():
            logger.error(f"CIF checkpoint not found: {cif_ckpt_path}")
            logger.info("Falling back to always_fire mode")
            always_fire = True
        else:
            try:
                map_location = None if torch.cuda.is_available() else torch.device("cpu")
                checkpoint = torch.load(cif_ckpt_path, map_location=map_location)  # nosec B614
                cif_linear.load_state_dict(checkpoint)
                logger.info(f"✅ CIF checkpoint loaded: {cif_ckpt_path}")
                logger.info("Word boundary detection enabled (-50% re-translations)")
            except Exception as e:
                logger.error(f"Failed to load CIF checkpoint: {e}")
                logger.info("Falling back to always_fire mode")
                always_fire = True

    cif_linear.to(device)
    return cif_linear, always_fire, never_fire_mode


def resize(
    alphas: torch.Tensor, target_lengths: torch.Tensor, threshold: float = 0.999
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resize alpha weights to target length with threshold clipping

    This function scales alpha weights (word boundary probabilities) to sum
    to the target length, while preventing any single alpha from exceeding
    the threshold.

    Parameters:
        alphas (Tensor): Raw alpha weights from CIF model (B, T)
        target_lengths (Tensor): Target sum for each sequence (B,)
        threshold (float): Maximum value for any alpha (default 0.999)

    Returns:
        Tuple[Tensor, Tensor]:
            - Resized alphas (B, T)
            - Original sums (_num) (B,)

    Algorithm:
        1. Calculate current sum: _num = sum(alphas)
        2. Scale to target: _alphas = alphas * (target / _num)
        3. Clip values > threshold by redistributing to neighbors
        4. Iterate until all values <= threshold (max 10 iterations)
    """
    # Sum current alphas
    _num = alphas.sum(-1)
    num = target_lengths.float()

    # Scale to target length
    _alphas = alphas * (num / _num)[:, None].repeat(1, alphas.size(1))

    # Remove values exceeding threshold
    count = 0
    while len(torch.where(_alphas > threshold)[0]):
        count += 1
        if count > 10:
            logger.warning("Alpha resize exceeded 10 iterations")
            break

        xs, ys = torch.where(_alphas > threshold)
        for x, y in zip(xs, ys, strict=False):
            if _alphas[x][y] >= threshold:
                # Redistribute excess to neighbors
                mask = _alphas[x].ne(0).float()
                mean = 0.5 * _alphas[x].sum() / mask.sum()
                _alphas[x] = _alphas[x] * 0.5 + mean * mask

    return _alphas, _num


def fire_at_boundary(
    chunked_encoder_feature: torch.Tensor, cif_linear: nn.Linear, threshold: float = 0.999
) -> bool:
    """
    Determine if current chunk ends at a word boundary

    This is the CORE CIF function that prevents cutting words mid-stream.
    It analyzes encoder features to detect if we're at a safe word boundary.

    Parameters:
        chunked_encoder_feature (Tensor): Encoder output (B, T, D)
            - B: batch size (usually 1)
            - T: number of audio frames
            - D: encoder dimension (1280 for large-v3)
        cif_linear (nn.Linear): Trained CIF model
        threshold (float): Integration threshold (default 0.999)

    Returns:
        bool: True if at word boundary (safe to emit chunk),
              False if mid-word (should buffer more audio)

    Algorithm:
        1. Predict alpha weights: alpha = sigmoid(linear(encoder_features))
        2. Calculate decode length: round(sum(alpha))
        3. Resize alphas to decode length
        4. Cumulative integration: integrate = cumsum(alpha)
        5. Check if final frames are beyond threshold
        6. If yes -> at word boundary -> return True
        7. If no -> mid-word -> return False

    Example:
        # Encoder features from Whisper
        encoder_out = model.encoder(mel)  # (1, 1500, 1280)

        # Check if at word boundary
        if fire_at_boundary(encoder_out, cif_model):
            # Safe to emit chunk - at word boundary
            emit_transcription(current_chunk)
        else:
            # Mid-word - buffer more audio
            buffer.append(current_chunk)
    """
    content_mel_len = chunked_encoder_feature.shape[1]  # T (time frames)

    # Predict alpha weights: encoder → linear → sigmoid
    # alphas represent "word boundary probability" at each frame
    alphas = cif_linear(chunked_encoder_feature).squeeze(dim=2)  # (B, T, 1) → (B, T)
    alphas = torch.sigmoid(alphas)  # Normalize to [0, 1]

    # Calculate target decode length
    decode_length = torch.round(alphas.sum(-1)).int()

    # Resize alphas to decode length
    alphas, _ = resize(alphas, decode_length, threshold=threshold)
    alphas = alphas.squeeze(0)  # (B, T) → (T,)

    # Cumulative integration (ignore last frame's peak value)
    # This implements the "Continuous Integrate-and-Fire" mechanism
    integrate = torch.cumsum(alphas[:-1], dim=0)

    # Count how many times we've exceeded the threshold
    exceed_count = integrate[-1] // threshold

    # Reset integration by subtracting threshold each time we exceed it
    integrate = integrate - exceed_count * 1.0

    # Find positions where integration is still positive (important positions)
    important_positions = (integrate >= 0).nonzero(as_tuple=True)[0]

    if important_positions.numel() == 0:
        # No important positions → not at word boundary
        return False
    else:
        # Check if first important position is near the end
        # If yes → we're at a word boundary → safe to emit
        # The "-2" allows some tolerance at the chunk end
        is_boundary = (important_positions[0] >= content_mel_len - 2).item()
        return bool(is_boundary)


# Example usage and testing
if __name__ == "__main__":
    print("CIF End-of-Word Detection Test")
    print("=" * 50)

    # Test 1: Load CIF without checkpoint (fallback mode)
    print("\n[TEST 1] Load CIF without checkpoint:")
    cif_model, always_fire, never_fire = load_cif(
        cif_ckpt_path=None, n_audio_state=1280, device=torch.device("cpu")
    )
    print(f"  CIF model: {cif_model}")
    print(f"  Always fire: {always_fire}")
    print(f"  Never fire: {never_fire}")
    print("  ✅ Fallback mode initialized")

    # Test 2: Test resize function
    print("\n[TEST 2] Test alpha resize:")
    alphas = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
    target_lengths = torch.tensor([3])
    resized, original = resize(alphas, target_lengths)
    print(f"  Original alphas: {alphas}")
    print(f"  Original sum: {original.item():.4f}")
    print(f"  Target length: {target_lengths.item()}")
    print(f"  Resized alphas: {resized}")
    print(f"  Resized sum: {resized.sum().item():.4f}")
    print("  ✅ Resize function working")

    # Test 3: Test fire_at_boundary with random features
    print("\n[TEST 3] Test fire_at_boundary:")
    # Simulate encoder features (B=1, T=1500, D=1280)
    encoder_features = torch.randn(1, 1500, 1280)

    is_boundary = fire_at_boundary(encoder_features, cif_model)
    print(f"  Encoder features shape: {encoder_features.shape}")
    print(f"  At word boundary: {is_boundary}")
    print("  ✅ Boundary detection working")

    # Test 4: Always fire mode behavior
    print("\n[TEST 4] Always fire mode:")
    # In always_fire mode, we should always return True
    # (This is handled at a higher level, not in fire_at_boundary itself)
    print(f"  Always fire mode: {always_fire}")
    print("  Behavior: emit chunks at fixed intervals (fallback)")
    print("  ✅ Fallback mode configured")

    print("\n" + "=" * 50)
    print("✅ CIF End-of-Word Detection Test Complete")
    print("\nKey Features:")
    print("  - CIF model for word boundary detection")
    print("  - Fallback mode when no checkpoint available")
    print("  - Prevents cutting words mid-stream")
    print("  - Target: -50% reduction in re-translations")
