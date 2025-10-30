#!/usr/bin/env python3
"""
Decoder utility functions for read-only operations.

Provides context managers and helpers for running decoder inference
without side effects (no KV cache mutations, no state changes).
"""

import contextlib
import torch
from typing import Optional, List


@contextlib.contextmanager
def disable_kv_cache(decoder, prompt_length: Optional[int] = None):
    """
    Context manager to disable KV cache mutations during decoder forward pass.

    This ensures read-only operation - the decoder call will not modify
    any cached state or register any hooks.

    For SimulStreaming compatibility: If prompt_length is provided, temporarily
    replaces the decoder's mask with a clean triangular mask sized for the prompt.

    Usage:
        with disable_kv_cache(model.decoder, prompt_length=3):
            logits = model.decoder(tokens, encoder_output, kv_cache=None)
            # No KV cache was modified, mask is clean for 3-token prompt

    Args:
        decoder: Whisper TextDecoder module
        prompt_length: If provided, create a clean mask for this many tokens
                      (required for SimulStreaming compatibility)

    Yields:
        None
    """
    import numpy as np

    # Save original cache state
    old_use_cache = getattr(decoder, "use_cache", True)

    # Save and remove any KV cache hooks
    hooks = []
    if hasattr(decoder, "_kv_hooks"):
        hooks = decoder._kv_hooks
        # Remove hooks temporarily
        for h in hooks:
            h.remove()
        decoder._kv_hooks = []

    # Save original mask and create clean one if needed
    old_mask = None
    if prompt_length is not None and hasattr(decoder, "mask"):
        old_mask = decoder.mask
        # Create clean triangular mask for prompt_length tokens
        # This ensures SimulStreaming decoder doesn't see accumulated context
        clean_mask = torch.empty(prompt_length, prompt_length).fill_(-np.inf).triu_(1)
        decoder.mask = clean_mask.to(old_mask.device)

    # Disable cache
    decoder.use_cache = False

    try:
        yield
    finally:
        # Restore cache state
        decoder.use_cache = old_use_cache

        # Restore original mask
        if old_mask is not None:
            decoder.mask = old_mask

        # Re-register hooks if they existed
        if hooks:
            decoder._kv_hooks = hooks


def decoder_first_step_readonly(
    decoder,
    encoder_output: torch.Tensor,
    prompt: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Run a single read-only decoder step with no side effects.

    This is used for language detection probes where we need to extract
    language token logits without modifying decoder state.

    For SimulStreaming compatibility: Creates a clean mask sized for the prompt
    to prevent conflicts with the decoder's accumulated context.

    Args:
        decoder: Whisper TextDecoder module
        encoder_output: Encoder output tensor [B, T, D]
        prompt: Token IDs for prompt [B, L] (e.g., [SOT, TRANSCRIBE, NO_TIMESTAMPS])
        device: Device to run on

    Returns:
        logits: Decoder output logits [B, L, V] where V is vocab size
    """
    with torch.inference_mode():
        # Get prompt length for mask sizing (batch_size, seq_len)
        prompt_length = prompt.shape[1]

        with disable_kv_cache(decoder, prompt_length=prompt_length):
            # Run decoder with no cache and clean mask
            logits = decoder(
                x=prompt.to(device),
                xa=encoder_output.to(device),
                kv_cache=None  # Explicitly no cache
            )
            return logits
