# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LSQ (Lattice Syndrome Quantization) ECC Cache Operations

Provides functional helpers for:
1. Hadamard rotation for keys and queries
2. LSQ encode/decode wrapper functions
3. Integration helpers for FlashAttention backend

LSQ pairs adjacent head_size dimensions and uses SECDED(16,11) encoding.
Keys should be rotated before encoding; Queries should be rotated before attention.
Values are NOT rotated.
"""

from typing import Optional

import torch


def apply_hadamard_rotation(
    tensor: torch.Tensor,
    inplace: bool = False
) -> torch.Tensor:
    """Apply Hadamard rotation for LSQ quantization.

    The Hadamard transform redistributes outliers across dimensions,
    making adjacent dimensions more correlated. This enables the
    LSQ predictor relationship to work effectively.

    Args:
        tensor: Input tensor of shape [..., head_size]
                head_size must be a power of 2.
        inplace: If True, modify tensor in place.

    Returns:
        Rotated tensor of same shape.
    """
    try:
        # Use existing hadacore implementation if available
        from vllm._custom_ops import hadacore_transform
        return hadacore_transform(tensor, inplace=inplace)
    except (ImportError, AttributeError):
        # Fallback: simple Hadamard using PyTorch
        # This is slower but works without hadacore compilation
        return _hadamard_fallback(tensor)


def _hadamard_fallback(tensor: torch.Tensor) -> torch.Tensor:
    """Fallback Hadamard transform using PyTorch.

    Uses recursive definition: H_n = [[H_{n/2}, H_{n/2}],
                                       [H_{n/2}, -H_{n/2}]]

    Normalized by 1/sqrt(n) to be orthonormal.
    """
    *batch_dims, n = tensor.shape
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"

    # Reshape for in-place operation
    x = tensor.clone()

    # Apply fast Hadamard transform (in-place butterfly)
    h = 1
    while h < n:
        # Process pairs at distance h
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[..., j]
                b = x[..., j + h]
                x[..., j] = a + b
                x[..., j + h] = a - b
        h *= 2

    # Normalize
    x = x / (n ** 0.5)
    return x


def lsq_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    apply_rotation: bool = True
) -> None:
    """Rotate keys (if enabled) and encode K/V to LSQ format.

    Args:
        key: Key tensor [num_tokens, num_heads, head_size]
        value: Value tensor [num_tokens, num_heads, head_size]
        key_cache: Key cache [num_blocks, block_size, num_heads, head_size/2] as uint16
        value_cache: Value cache [num_blocks, block_size, num_heads, head_size/2] as uint16
        slot_mapping: Slot mapping [num_tokens]
        apply_rotation: Whether to apply Hadamard rotation to keys
    """
    # Apply rotation to keys if enabled
    if apply_rotation:
        # Flatten for rotation, then reshape back
        orig_shape = key.shape
        key_flat = key.reshape(-1, orig_shape[-1])
        key_rotated = apply_hadamard_rotation(key_flat)
        key = key_rotated.reshape(orig_shape)

    # Encode to LSQ format
    torch.ops._C_cache_ops.lsq_ecc_encode(
        key, value, key_cache, value_cache, slot_mapping
    )


def lsq_gather_and_decode(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    seq_start_locs: torch.Tensor,
    workspace_k: torch.Tensor,
    workspace_v: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_seqs: int
) -> None:
    """Decode LSQ cache to FP16 workspace.

    For erasures, uses N-LERP reconstruction from temporal neighbors.

    Args:
        key_cache: Key cache [num_blocks, block_size, num_heads, head_size/2] as uint16
        value_cache: Value cache [num_blocks, block_size, num_heads, head_size/2] as uint16
        slot_mapping: Slot mapping [num_tokens]
        seq_start_locs: Sequence start locations [num_seqs + 1]
        workspace_k: Output key workspace [num_tokens, num_heads, head_size]
        workspace_v: Output value workspace [num_tokens, num_heads, head_size]
        num_tokens: Number of tokens
        num_heads: Number of attention heads
        head_size: Size of each head
        block_size: Cache block size
        num_seqs: Number of sequences
    """
    torch.ops._C_cache_ops.lsq_ecc_gather_decode(
        key_cache, value_cache,
        slot_mapping, seq_start_locs,
        workspace_k, workspace_v,
        num_tokens, num_heads, head_size, block_size, num_seqs
    )


def rotate_queries_for_lsq(
    query: torch.Tensor,
    apply_rotation: bool = True
) -> torch.Tensor:
    """Apply Hadamard rotation to queries for LSQ attention.

    Must match the rotation applied to keys during cache encoding.

    Args:
        query: Query tensor [num_tokens, num_heads, head_size]
        apply_rotation: Whether to apply rotation

    Returns:
        Rotated query tensor
    """
    if not apply_rotation:
        return query

    orig_shape = query.shape
    query_flat = query.reshape(-1, orig_shape[-1])
    query_rotated = apply_hadamard_rotation(query_flat)
    return query_rotated.reshape(orig_shape)


def allocate_lsq_cache(
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.int16  # uint16 stored as int16
) -> torch.Tensor:
    """Allocate a KV cache tensor for LSQ encoding.

    LSQ stores pairs of values in uint16, so the cache shape is:
    [num_blocks, block_size, num_heads, head_size // 2]

    Args:
        num_blocks: Number of cache blocks
        block_size: Number of tokens per block
        num_heads: Number of attention heads
        head_size: Size of each head (must be even)
        device: Device to allocate on
        dtype: Data type (int16 for uint16 storage)

    Returns:
        Allocated cache tensor
    """
    assert head_size % 2 == 0, f"head_size must be even for LSQ, got {head_size}"

    return torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        head_size // 2,
        dtype=dtype,
        device=device
    )


def is_lsq_cache_dtype(cache_dtype: str) -> bool:
    """Check if the cache dtype indicates LSQ mode."""
    return cache_dtype == "int4_ecc_lsq"
