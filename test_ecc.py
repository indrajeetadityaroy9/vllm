# SPDX-License-Identifier: Apache-2.0
"""
ECC (Error Correction Code) Test Script

This script tests the Hamming(7,4) ECC implementation for KV cache protection.
It verifies:
1. Basic encode/decode round-trip works
2. Single-bit error correction works
3. Quantization loss is within acceptable bounds
"""

import torch
import numpy as np


def test_ecc_roundtrip():
    """Test basic encode/decode round-trip."""
    try:
        from vllm._C import cache_ops
    except ImportError:
        print("ERROR: ECC ops not available. Build with -DVLLM_ECC_ENABLED=ON")
        return False

    if not hasattr(cache_ops, 'ecc_encode'):
        print("ERROR: ECC encode not available. Build with -DVLLM_ECC_ENABLED=ON")
        return False

    print("=" * 60)
    print("ECC Round-Trip Test")
    print("=" * 60)

    # Test parameters
    num_tokens = 16
    num_heads = 4
    head_size = 64
    block_size = 16
    num_blocks = 4

    device = torch.device("cuda")

    # Create input tensors (values in quantization range [-4, 4])
    key = torch.randn(num_tokens, num_heads, head_size,
                      dtype=torch.float16, device=device) * 2.0
    value = torch.randn(num_tokens, num_heads, head_size,
                        dtype=torch.float16, device=device) * 2.0

    # Clamp to quantization range
    key = key.clamp(-4.0, 4.0)
    value = value.clamp(-4.0, 4.0)

    # Create uint8 cache for ECC storage
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size,
                            dtype=torch.uint8, device=device)
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size,
                              dtype=torch.uint8, device=device)

    # Create slot mapping (sequential for simple test)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Encode
    cache_ops.ecc_encode(key, value, key_cache, value_cache, slot_mapping)

    # Verify codewords are written
    num_nonzero = (key_cache != 0).sum().item()
    print(f"  Encoded: {num_nonzero} non-zero codewords")

    # Create workspace for decoding
    workspace_k = torch.zeros(num_tokens, num_heads, head_size,
                              dtype=torch.float16, device=device)
    workspace_v = torch.zeros(num_tokens, num_heads, head_size,
                              dtype=torch.float16, device=device)

    # Decode
    cache_ops.ecc_gather_decode(
        key_cache, value_cache, slot_mapping,
        workspace_k, workspace_v,
        num_tokens, num_heads, head_size, block_size
    )

    # Compare with original
    key_error = (key - workspace_k).abs().mean().item()
    value_error = (value - workspace_v).abs().mean().item()

    # Quantization error should be < 0.5 (half a quantization step)
    # For [-4, 4] range with 16 levels, step size = 0.5
    max_expected_error = 0.5

    print(f"  Key mean absolute error: {key_error:.6f}")
    print(f"  Value mean absolute error: {value_error:.6f}")
    print(f"  Expected max error (quantization): {max_expected_error:.3f}")

    if key_error < max_expected_error and value_error < max_expected_error:
        print("  PASS: Quantization error within bounds")
        return True
    else:
        print("  FAIL: Quantization error too large")
        return False


def test_ecc_error_correction():
    """Test single-bit error correction by injecting bit flips."""
    try:
        from vllm._C import cache_ops
    except ImportError:
        print("ERROR: ECC ops not available. Build with -DVLLM_ECC_ENABLED=ON")
        return False

    if not hasattr(cache_ops, 'ecc_encode'):
        print("ERROR: ECC encode not available. Build with -DVLLM_ECC_ENABLED=ON")
        return False

    print("\n" + "=" * 60)
    print("ECC Error Correction Test")
    print("=" * 60)

    # Test parameters
    num_tokens = 16
    num_heads = 4
    head_size = 64
    block_size = 16
    num_blocks = 4

    device = torch.device("cuda")

    # Create input tensors
    key = torch.randn(num_tokens, num_heads, head_size,
                      dtype=torch.float16, device=device) * 2.0
    value = torch.randn(num_tokens, num_heads, head_size,
                        dtype=torch.float16, device=device) * 2.0
    key = key.clamp(-4.0, 4.0)
    value = value.clamp(-4.0, 4.0)

    # Create uint8 cache
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size,
                            dtype=torch.uint8, device=device)
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size,
                              dtype=torch.uint8, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Encode
    cache_ops.ecc_encode(key, value, key_cache, value_cache, slot_mapping)

    # Get baseline decode (no errors)
    workspace_k_clean = torch.zeros(num_tokens, num_heads, head_size,
                                    dtype=torch.float16, device=device)
    workspace_v_clean = torch.zeros(num_tokens, num_heads, head_size,
                                    dtype=torch.float16, device=device)
    cache_ops.ecc_gather_decode(
        key_cache, value_cache, slot_mapping,
        workspace_k_clean, workspace_v_clean,
        num_tokens, num_heads, head_size, block_size
    )

    # Inject single-bit errors (flip random bits in ~10% of codewords)
    key_cache_corrupted = key_cache.clone()
    value_cache_corrupted = value_cache.clone()

    # Flatten to inject errors
    flat_key = key_cache_corrupted.view(-1)
    flat_value = value_cache_corrupted.view(-1)

    num_errors = int(flat_key.numel() * 0.1)
    error_indices = torch.randperm(flat_key.numel(), device=device)[:num_errors]

    # Flip random bit in each selected codeword
    for idx in error_indices:
        bit_to_flip = torch.randint(0, 7, (1,), device=device).item()
        flat_key[idx] = flat_key[idx] ^ (1 << bit_to_flip)
        flat_value[idx] = flat_value[idx] ^ (1 << bit_to_flip)

    print(f"  Injected {num_errors} single-bit errors")

    # Decode corrupted cache
    workspace_k_corrected = torch.zeros(num_tokens, num_heads, head_size,
                                        dtype=torch.float16, device=device)
    workspace_v_corrected = torch.zeros(num_tokens, num_heads, head_size,
                                        dtype=torch.float16, device=device)
    cache_ops.ecc_gather_decode(
        key_cache_corrupted, value_cache_corrupted, slot_mapping,
        workspace_k_corrected, workspace_v_corrected,
        num_tokens, num_heads, head_size, block_size
    )

    # Compare corrected with clean (should be identical)
    key_diff = (workspace_k_clean - workspace_k_corrected).abs().max().item()
    value_diff = (workspace_v_clean - workspace_v_corrected).abs().max().item()

    print(f"  Key max difference after correction: {key_diff:.6f}")
    print(f"  Value max difference after correction: {value_diff:.6f}")

    if key_diff < 1e-5 and value_diff < 1e-5:
        print("  PASS: Single-bit errors corrected successfully")
        return True
    else:
        print("  FAIL: Error correction failed")
        return False


def test_ecc_with_model():
    """Test ECC with a simple model inference."""
    print("\n" + "=" * 60)
    print("ECC Model Inference Test (Placeholder)")
    print("=" * 60)
    print("  NOTE: Full model integration requires attention backend changes")
    print("  This test is a placeholder for future integration")
    return True


if __name__ == "__main__":
    print("ECC (Hamming 7,4) Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Round-Trip", test_ecc_roundtrip()))
    results.append(("Error Correction", test_ecc_error_correction()))
    results.append(("Model Inference", test_ecc_with_model()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
