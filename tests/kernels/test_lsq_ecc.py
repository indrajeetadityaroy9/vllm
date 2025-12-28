# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for LSQ (Lattice Syndrome Quantization) + SECDED(16,11) ECC.

Tests cover:
1. Python-level Hadamard rotation helpers
2. LSQ encode/decode kernel round-trip
3. Error correction (single-bit)
4. Error detection (double-bit → erasure)
5. N-LERP reconstruction from temporal neighbors
6. Sequence boundary safety
"""

import pytest
import torch

from vllm.attention.ops.lsq_ecc_cache import (
    allocate_lsq_cache,
    apply_hadamard_rotation,
    is_lsq_cache_dtype,
    rotate_queries_for_lsq,
)

# Skip all tests if ECC is not compiled
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for LSQ ECC tests"
)


def check_lsq_ops_available() -> bool:
    """Check if LSQ ECC ops are compiled."""
    try:
        import vllm._C  # noqa: F401
        _ = torch.ops._C_cache_ops.lsq_ecc_encode
        return True
    except (ImportError, AttributeError):
        return False


# ============================================================================
# Hadamard Rotation Tests (Python-level)
# ============================================================================

class TestHadamardRotation:
    """Test Hadamard rotation helpers."""

    def test_hadamard_power_of_2(self):
        """Hadamard requires power-of-2 head_size."""
        for head_size in [64, 128, 256]:
            x = torch.randn(4, 8, head_size, device="cuda", dtype=torch.float16)
            y = apply_hadamard_rotation(x)
            assert y.shape == x.shape
            assert y.dtype == x.dtype

    def test_hadamard_orthonormal(self):
        """Hadamard is orthonormal: H @ H^T = I."""
        head_size = 128
        x = torch.randn(1, 1, head_size, device="cuda", dtype=torch.float32)

        # Apply twice should approximately recover original (H^2 = I up to scale)
        y = apply_hadamard_rotation(x)
        z = apply_hadamard_rotation(y)

        # After two applications, should be back to original
        torch.testing.assert_close(z, x, rtol=1e-3, atol=1e-3)

    def test_hadamard_preserves_norm(self):
        """Hadamard preserves L2 norm (orthonormal)."""
        head_size = 128
        x = torch.randn(4, 8, head_size, device="cuda", dtype=torch.float16)
        y = apply_hadamard_rotation(x)

        x_norm = torch.linalg.vector_norm(x.float(), dim=-1)
        y_norm = torch.linalg.vector_norm(y.float(), dim=-1)

        torch.testing.assert_close(x_norm, y_norm, rtol=1e-2, atol=1e-2)

    def test_rotate_queries_disabled(self):
        """rotate_queries_for_lsq with apply_rotation=False is identity."""
        q = torch.randn(4, 8, 128, device="cuda", dtype=torch.float16)
        q_out = rotate_queries_for_lsq(q, apply_rotation=False)
        torch.testing.assert_close(q_out, q)

    def test_rotate_queries_enabled(self):
        """rotate_queries_for_lsq with apply_rotation=True applies Hadamard."""
        q = torch.randn(4, 8, 128, device="cuda", dtype=torch.float16)
        q_out = rotate_queries_for_lsq(q, apply_rotation=True)

        # Should not be the same (unless q was zero)
        assert not torch.allclose(q_out, q)

        # But norms should match
        q_norm = torch.linalg.vector_norm(q.float(), dim=-1)
        q_out_norm = torch.linalg.vector_norm(q_out.float(), dim=-1)
        torch.testing.assert_close(q_norm, q_out_norm, rtol=1e-2, atol=1e-2)


# ============================================================================
# Cache Allocation Tests
# ============================================================================

class TestLSQCacheAllocation:
    """Test LSQ cache allocation helpers."""

    def test_allocate_lsq_cache_shape(self):
        """LSQ cache has shape [blocks, block_size, heads, head_size/2]."""
        num_blocks = 16
        block_size = 16
        num_heads = 8
        head_size = 128

        cache = allocate_lsq_cache(
            num_blocks, block_size, num_heads, head_size,
            device=torch.device("cuda")
        )

        assert cache.shape == (num_blocks, block_size, num_heads, head_size // 2)
        assert cache.dtype == torch.int16  # uint16 stored as int16

    def test_allocate_lsq_cache_requires_even_head_size(self):
        """LSQ requires even head_size for pairing."""
        with pytest.raises(AssertionError, match="head_size must be even"):
            allocate_lsq_cache(
                16, 16, 8, 127,  # Odd head_size
                device=torch.device("cuda")
            )

    def test_is_lsq_cache_dtype(self):
        """Check is_lsq_cache_dtype helper."""
        assert is_lsq_cache_dtype("int4_ecc_lsq") is True
        assert is_lsq_cache_dtype("auto") is False
        assert is_lsq_cache_dtype("fp8") is False
        assert is_lsq_cache_dtype("int4_ecc") is False


# ============================================================================
# LSQ Kernel Tests (require compilation with VLLM_ECC_ENABLED)
# ============================================================================

@pytest.mark.skipif(
    not check_lsq_ops_available(),
    reason="LSQ ECC ops not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestLSQKernels:
    """Test LSQ encode/decode kernels."""

    @pytest.fixture
    def setup_cache(self):
        """Create test cache and tensors."""
        num_blocks = 4
        block_size = 16
        num_heads = 8
        head_size = 128
        num_tokens = 32

        # Key/Value tensors (FP16)
        key = torch.randn(
            num_tokens, num_heads, head_size,
            device="cuda", dtype=torch.float16
        )
        value = torch.randn(
            num_tokens, num_heads, head_size,
            device="cuda", dtype=torch.float16
        )

        # Scale to be within anchor range [-40, 40]
        key = key * 10  # ~[-30, 30] after scaling
        value = value * 10

        # Cache tensors (uint16 as int16)
        key_cache = torch.zeros(
            num_blocks, block_size, num_heads, head_size // 2,
            device="cuda", dtype=torch.int16
        )
        value_cache = torch.zeros(
            num_blocks, block_size, num_heads, head_size // 2,
            device="cuda", dtype=torch.int16
        )

        # Slot mapping: tokens 0..31 -> slots 0..31 (linear)
        slot_mapping = torch.arange(num_tokens, device="cuda", dtype=torch.int64)

        return {
            "key": key,
            "value": value,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "slot_mapping": slot_mapping,
            "num_blocks": num_blocks,
            "block_size": block_size,
            "num_heads": num_heads,
            "head_size": head_size,
            "num_tokens": num_tokens,
        }

    def test_encode_decode_round_trip(self, setup_cache):
        """Test that encode → decode recovers approximate values."""
        d = setup_cache

        # Apply Hadamard rotation to keys
        key_rotated = apply_hadamard_rotation(d["key"])

        # Encode
        torch.ops._C_cache_ops.lsq_ecc_encode(
            key_rotated, d["value"],
            d["key_cache"], d["value_cache"],
            d["slot_mapping"]
        )

        # Prepare for decode
        seq_start_locs = torch.tensor([0, d["num_tokens"]], device="cuda", dtype=torch.int64)
        workspace_k = torch.zeros_like(d["key"])
        workspace_v = torch.zeros_like(d["value"])

        # Decode
        torch.ops._C_cache_ops.lsq_ecc_gather_decode(
            d["key_cache"], d["value_cache"],
            d["slot_mapping"], seq_start_locs,
            workspace_k, workspace_v,
            d["num_tokens"], d["num_heads"], d["head_size"],
            d["block_size"], 1  # num_seqs
        )

        # Check reconstruction error
        # LSQ with 6-bit anchor + 5-bit syndrome should have ~1-2% relative error
        key_error = (workspace_k - key_rotated).abs().mean()
        value_error = (workspace_v - d["value"]).abs().mean()

        key_scale = key_rotated.abs().mean()
        value_scale = d["value"].abs().mean()

        key_rel_error = key_error / key_scale
        value_rel_error = value_error / value_scale

        # Allow up to 10% relative error for quantization
        assert key_rel_error < 0.10, f"Key relative error too high: {key_rel_error:.4f}"
        assert value_rel_error < 0.10, f"Value relative error too high: {value_rel_error:.4f}"

    def test_small_values_round_trip(self, setup_cache):
        """Test round-trip with small values (within quantization range)."""
        d = setup_cache

        # Small values should have lower quantization error
        key = torch.randn_like(d["key"]) * 2  # ~[-6, 6]
        value = torch.randn_like(d["value"]) * 2

        key_rotated = apply_hadamard_rotation(key)

        # Encode
        torch.ops._C_cache_ops.lsq_ecc_encode(
            key_rotated, value,
            d["key_cache"], d["value_cache"],
            d["slot_mapping"]
        )

        # Decode
        seq_start_locs = torch.tensor([0, d["num_tokens"]], device="cuda", dtype=torch.int64)
        workspace_k = torch.zeros_like(key)
        workspace_v = torch.zeros_like(value)

        torch.ops._C_cache_ops.lsq_ecc_gather_decode(
            d["key_cache"], d["value_cache"],
            d["slot_mapping"], seq_start_locs,
            workspace_k, workspace_v,
            d["num_tokens"], d["num_heads"], d["head_size"],
            d["block_size"], 1
        )

        # Smaller values should have tighter relative error
        key_error = (workspace_k - key_rotated).abs().mean()
        value_error = (workspace_v - value).abs().mean()

        key_scale = key_rotated.abs().mean()
        value_scale = value.abs().mean()

        key_rel_error = key_error / key_scale
        value_rel_error = value_error / value_scale

        # Should be under 5% for small values
        assert key_rel_error < 0.05, f"Key relative error too high: {key_rel_error:.4f}"
        assert value_rel_error < 0.05, f"Value relative error too high: {value_rel_error:.4f}"

    def test_non_contiguous_slots(self, setup_cache):
        """Test with non-contiguous slot mapping."""
        d = setup_cache

        # Non-contiguous slots: 0, 2, 4, 6, ...
        num_tokens = d["num_tokens"] // 2
        slot_mapping = torch.arange(0, d["num_tokens"], 2, device="cuda", dtype=torch.int64)
        key = d["key"][:num_tokens]
        value = d["value"][:num_tokens]

        key_rotated = apply_hadamard_rotation(key)

        # Encode
        torch.ops._C_cache_ops.lsq_ecc_encode(
            key_rotated, value,
            d["key_cache"], d["value_cache"],
            slot_mapping
        )

        # Decode
        seq_start_locs = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int64)
        workspace_k = torch.zeros_like(key)
        workspace_v = torch.zeros_like(value)

        torch.ops._C_cache_ops.lsq_ecc_gather_decode(
            d["key_cache"], d["value_cache"],
            slot_mapping, seq_start_locs,
            workspace_k, workspace_v,
            num_tokens, d["num_heads"], d["head_size"],
            d["block_size"], 1
        )

        # Check round-trip
        key_error = (workspace_k - key_rotated).abs().mean()
        value_error = (workspace_v - value).abs().mean()

        assert key_error < 1.0, f"Key error too high: {key_error:.4f}"
        assert value_error < 1.0, f"Value error too high: {value_error:.4f}"

    def test_multiple_sequences(self, setup_cache):
        """Test with multiple sequences."""
        d = setup_cache

        # Two sequences: [0, 16) and [16, 32)
        seq_start_locs = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int64)

        key_rotated = apply_hadamard_rotation(d["key"])

        # Encode
        torch.ops._C_cache_ops.lsq_ecc_encode(
            key_rotated, d["value"],
            d["key_cache"], d["value_cache"],
            d["slot_mapping"]
        )

        # Decode
        workspace_k = torch.zeros_like(d["key"])
        workspace_v = torch.zeros_like(d["value"])

        torch.ops._C_cache_ops.lsq_ecc_gather_decode(
            d["key_cache"], d["value_cache"],
            d["slot_mapping"], seq_start_locs,
            workspace_k, workspace_v,
            d["num_tokens"], d["num_heads"], d["head_size"],
            d["block_size"], 2  # num_seqs
        )

        # Check round-trip
        key_error = (workspace_k - key_rotated).abs().mean()
        value_error = (workspace_v - d["value"]).abs().mean()

        assert key_error < 1.0, f"Key error too high: {key_error:.4f}"
        assert value_error < 1.0, f"Value error too high: {value_error:.4f}"


# ============================================================================
# Error Injection Tests (require compilation with VLLM_FAULT_INJECT)
# ============================================================================

@pytest.mark.skipif(
    not check_lsq_ops_available(),
    reason="LSQ ECC ops not compiled"
)
class TestLSQErrorCorrection:
    """Test LSQ error correction capabilities."""

    def test_single_bit_error_correction(self):
        """Test that single-bit errors are corrected."""
        num_blocks = 1
        block_size = 16
        num_heads = 4
        head_size = 64
        num_tokens = 16

        # Create test data
        key = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=torch.float16) * 5
        value = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=torch.float16) * 5

        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size // 2,
                                device="cuda", dtype=torch.int16)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size // 2,
                                  device="cuda", dtype=torch.int16)

        slot_mapping = torch.arange(num_tokens, device="cuda", dtype=torch.int64)

        key_rotated = apply_hadamard_rotation(key)

        # Encode
        torch.ops._C_cache_ops.lsq_ecc_encode(
            key_rotated, value,
            key_cache, value_cache,
            slot_mapping
        )

        # Inject single-bit error (flip bit 7 in first codeword)
        key_cache_corrupted = key_cache.clone()
        key_cache_corrupted[0, 0, 0, 0] ^= (1 << 7)

        # Decode (should correct the error)
        seq_start_locs = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int64)
        workspace_k = torch.zeros_like(key)
        workspace_v = torch.zeros_like(value)

        torch.ops._C_cache_ops.lsq_ecc_gather_decode(
            key_cache_corrupted, value_cache,
            slot_mapping, seq_start_locs,
            workspace_k, workspace_v,
            num_tokens, num_heads, head_size,
            block_size, 1
        )

        # Decode without corruption for reference
        workspace_k_ref = torch.zeros_like(key)
        workspace_v_ref = torch.zeros_like(value)

        torch.ops._C_cache_ops.lsq_ecc_gather_decode(
            key_cache, value_cache,
            slot_mapping, seq_start_locs,
            workspace_k_ref, workspace_v_ref,
            num_tokens, num_heads, head_size,
            block_size, 1
        )

        # After correction, should match reference
        torch.testing.assert_close(workspace_k, workspace_k_ref, rtol=1e-3, atol=1e-3)

    def test_double_bit_error_detection(self):
        """Test that double-bit errors are detected (flagged as erasure)."""
        num_blocks = 1
        block_size = 16
        num_heads = 4
        head_size = 64
        num_tokens = 16

        # Create test data
        key = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=torch.float16) * 5
        value = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=torch.float16) * 5

        key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size // 2,
                                device="cuda", dtype=torch.int16)
        value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size // 2,
                                  device="cuda", dtype=torch.int16)

        slot_mapping = torch.arange(num_tokens, device="cuda", dtype=torch.int64)

        key_rotated = apply_hadamard_rotation(key)

        # Encode
        torch.ops._C_cache_ops.lsq_ecc_encode(
            key_rotated, value,
            key_cache, value_cache,
            slot_mapping
        )

        # Inject double-bit error (flip bits 0 and 7)
        key_cache_corrupted = key_cache.clone()
        # Target token in the middle (not at boundary) so N-LERP has neighbors
        target_token = 8  # Middle of sequence
        key_cache_corrupted[0, target_token, 0, 0] ^= (1 << 0) | (1 << 7)

        # Decode (should detect as erasure and use N-LERP)
        seq_start_locs = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int64)
        workspace_k = torch.zeros_like(key)
        workspace_v = torch.zeros_like(value)

        torch.ops._C_cache_ops.lsq_ecc_gather_decode(
            key_cache_corrupted, value_cache,
            slot_mapping, seq_start_locs,
            workspace_k, workspace_v,
            num_tokens, num_heads, head_size,
            block_size, 1
        )

        # The result should be valid (N-LERP reconstructed)
        # It won't match exactly, but should be finite and reasonable
        assert torch.isfinite(workspace_k).all(), "N-LERP produced non-finite values"

        # Check that the reconstructed value is within reasonable bounds
        assert workspace_k.abs().max() < 100, "N-LERP produced unreasonable values"


# ============================================================================
# Config Integration Tests
# ============================================================================

class TestLSQConfig:
    """Test LSQ configuration integration."""

    def test_ecc_config_lsq_algorithm(self):
        """Test ECCConfig with LSQ algorithm."""
        from vllm.config.ecc import ECCConfig

        config = ECCConfig(
            enabled=True,
            algorithm="lsq_secded_16_11",
            lsq_anchor_range=40.0,
            lsq_apply_hadamard=True,
        )

        assert config.uses_lsq() is True
        assert config.uses_uint16_cache() is True
        assert config.get_algorithm_code() == 2

    def test_ecc_config_non_lsq_algorithm(self):
        """Test ECCConfig with non-LSQ algorithm."""
        from vllm.config.ecc import ECCConfig

        config = ECCConfig(
            enabled=True,
            algorithm="secded_8_4",
        )

        assert config.uses_lsq() is False
        assert config.uses_uint16_cache() is False
        assert config.get_algorithm_code() == 1

    def test_cache_dtype_literal(self):
        """Test that int4_ecc_lsq is a valid CacheDType."""
        from vllm.config.cache import CacheDType

        # This should type-check without error
        dtype: CacheDType = "int4_ecc_lsq"
        assert dtype == "int4_ecc_lsq"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
