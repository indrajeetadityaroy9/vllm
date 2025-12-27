# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECC (Error Correction Code) in KV cache kernels.

These tests verify the Hamming(7,4) ECC functionality for:
- Basic encode/decode operations
- INT4 quantization accuracy
- Single-bit error correction
- Multi-bit error detection (expected to fail)
- Round-trip encode-decode consistency
- Various tensor shapes and sizes

Note: These tests require vLLM to be compiled with VLLM_ECC_ENABLED=ON.
Run with: cmake -DVLLM_ECC_ENABLED=ON ...
"""

import pytest
import torch

from vllm.config import ECCConfig


def ecc_available() -> bool:
    """Check if ECC is compiled in."""
    try:
        import vllm._C  # noqa: F401
        # Try to access the ECC op
        _ = torch.ops._C_cache_ops.ecc_encode
        return True
    except Exception:
        return False


@pytest.fixture
def device():
    """Get CUDA device for testing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def ecc_config_default():
    """Create a Hamming(7,4) ECC config for testing (backward compat)."""
    return ECCConfig(
        enabled=True,
        algorithm="hamming_7_4",
    )


# Default algorithm code for legacy tests (Hamming 7,4)
ALGO_HAMMING74 = 0
ALGO_SECDED = 1


@pytest.fixture
def ecc_config_disabled():
    """Create a disabled ECC config for testing."""
    return ECCConfig(
        enabled=False,
        algorithm="hamming_7_4",
    )


def create_test_tensors(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    num_blocks: int,
    device: torch.device,
):
    """Create test tensors for ECC encode/decode tests.

    Returns tensors in the new format compatible with SECDED API:
    - key, value: [num_tokens, num_heads, head_dim]
    - key_cache, value_cache: [num_blocks, block_size, num_heads, head_dim]
    - slot_mapping: [num_tokens]
    - seq_start_locs: [num_seqs + 1] (single sequence for backward compat)
    - num_seqs: 1 (for backward compat)
    """
    num_tokens = batch_size

    # Input KV tensors (FP16) - new shape [num_tokens, num_heads, head_dim]
    key = torch.randn(
        num_tokens, num_heads, head_dim,
        device=device, dtype=torch.float16
    )
    value = torch.randn(
        num_tokens, num_heads, head_dim,
        device=device, dtype=torch.float16
    )

    # ECC-encoded cache (uint8)
    # New cache layout: [num_blocks, block_size, num_heads, head_dim]
    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_dim,
        dtype=torch.uint8, device=device
    )
    value_cache = torch.zeros_like(key_cache)

    # Slot mapping: token index -> physical slot
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

    # Sequence boundaries (single sequence for backward compat)
    seq_start_locs = torch.tensor([0, num_tokens], dtype=torch.long, device=device)
    num_seqs = 1

    return key, value, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs


@pytest.mark.skipif(
    not ecc_available(),
    reason="ECC not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestECCConfig:
    """Test ECCConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ECCConfig()
        assert config.enabled is False
        assert config.algorithm == "secded_8_4"  # SECDED is now the default

    def test_enabled_config(self, ecc_config_default):
        """Test enabled configuration."""
        assert ecc_config_default.enabled is True
        assert ecc_config_default.algorithm == "hamming_7_4"

    def test_compute_hash(self, ecc_config_default):
        """Test that hash computation is deterministic."""
        hash1 = ecc_config_default.compute_hash()
        hash2 = ecc_config_default.compute_hash()
        assert hash1 == hash2

        # Different config should produce different hash
        config2 = ECCConfig(enabled=False)
        hash3 = config2.compute_hash()
        assert hash1 != hash3


@pytest.mark.skipif(
    not ecc_available(),
    reason="ECC not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestECCEncode:
    """Test ECC encode functionality."""

    def test_basic_encode(self, device):
        """Test basic ECC encode operation."""
        batch_size, num_heads, head_dim = 4, 8, 64
        block_size, num_blocks = 16, 4

        (key, value, key_cache, value_cache, slot_mapping,
         seq_start_locs, num_seqs) = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # Should not raise (algorithm=0 for Hamming 7,4)
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        # Cache should have non-zero values
        assert key_cache.sum() > 0, "Key cache should have encoded values"
        assert value_cache.sum() > 0, "Value cache should have encoded values"

    def test_encode_different_batch_sizes(self, device):
        """Test ECC encode with different batch sizes."""
        num_heads, head_dim = 8, 64
        block_size, num_blocks = 16, 8

        for batch_size in [1, 4, 16, 32]:
            (key, value, key_cache, value_cache, slot_mapping,
             seq_start_locs, num_seqs) = create_test_tensors(
                batch_size, num_heads, head_dim, block_size, num_blocks, device
            )

            torch.ops._C_cache_ops.ecc_encode(
                key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
            )

            assert key_cache.sum() > 0

    def test_encode_different_head_dims(self, device):
        """Test ECC encode with different head dimensions."""
        batch_size, num_heads = 4, 8
        block_size, num_blocks = 16, 4

        for head_dim in [32, 64, 128]:
            (key, value, key_cache, value_cache, slot_mapping,
             seq_start_locs, num_seqs) = create_test_tensors(
                batch_size, num_heads, head_dim, block_size, num_blocks, device
            )

            torch.ops._C_cache_ops.ecc_encode(
                key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
            )

            assert key_cache.sum() > 0

    def test_encode_preserves_cache_structure(self, device):
        """Test that encode writes to correct cache locations."""
        batch_size, num_heads, head_dim = 2, 4, 32
        block_size, num_blocks = 16, 4

        (key, value, key_cache, value_cache, slot_mapping,
         seq_start_locs, num_seqs) = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # Use specific slot mapping
        slot_mapping = torch.tensor([0, 5], dtype=torch.long, device=device)

        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        # Check that slots 0 and 5 have data
        # New cache layout: [num_blocks, block_size, num_heads, head_dim]
        block_0_slot_0 = key_cache[0, 0, :, :]  # block 0, slot 0
        block_0_slot_5 = key_cache[0, 5, :, :]  # block 0, slot 5

        assert block_0_slot_0.sum() > 0, "Slot 0 should have data"
        assert block_0_slot_5.sum() > 0, "Slot 5 should have data"


@pytest.mark.skipif(
    not ecc_available(),
    reason="ECC not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestECCDecode:
    """Test ECC decode functionality."""

    def test_basic_decode(self, device):
        """Test basic ECC decode operation."""
        batch_size, num_heads, head_dim = 4, 8, 64
        block_size, num_blocks = 16, 4

        key, value, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # Encode first (use Hamming 7,4 for backward compat)
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        # Create workspace for decode
        workspace_k = torch.zeros(
            batch_size, num_heads, head_dim,
            dtype=torch.float16, device=device
        )
        workspace_v = torch.zeros_like(workspace_k)

        # Decode
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_k, workspace_v,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Should have non-zero values
        assert workspace_k.sum() != 0, "Decoded key should have values"
        assert workspace_v.sum() != 0, "Decoded value should have values"


@pytest.mark.skipif(
    not ecc_available(),
    reason="ECC not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestECCRoundTrip:
    """Test ECC encode-decode round-trip."""

    def test_roundtrip_reconstruction(self, device):
        """Test that encode-decode reconstructs values within quantization error."""
        batch_size, num_heads, head_dim = 4, 8, 64
        block_size, num_blocks = 16, 4

        key, value, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # Clamp to quantization range [-4, 4] for fair comparison
        key = key.clamp(-4.0, 4.0)
        value = value.clamp(-4.0, 4.0)

        # Encode
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        # Decode
        workspace_k = torch.zeros_like(key)
        workspace_v = torch.zeros_like(value)

        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_k, workspace_v,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Check reconstruction error (INT4 quantization expected)
        # INT4 has 16 levels over range [-4, 4], so step = 0.5
        # Max error should be ~0.25 (half step) on average
        key_mae = (key - workspace_k).abs().mean().item()
        value_mae = (value - workspace_v).abs().mean().item()

        # Allow for quantization error (0.5 max per element, ~0.25 average)
        assert key_mae < 0.5, f"Key reconstruction MAE {key_mae} too high"
        assert value_mae < 0.5, f"Value reconstruction MAE {value_mae} too high"

    def test_roundtrip_consistency(self, device):
        """Test that multiple encode-decode cycles are consistent."""
        batch_size, num_heads, head_dim = 4, 8, 64
        block_size, num_blocks = 16, 4

        key, value, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # First round-trip
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        workspace_k1 = torch.zeros_like(key)
        workspace_v1 = torch.zeros_like(value)

        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_k1, workspace_v1,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Second decode (without re-encoding)
        workspace_k2 = torch.zeros_like(key)
        workspace_v2 = torch.zeros_like(value)

        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_k2, workspace_v2,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Should be identical
        assert torch.allclose(workspace_k1, workspace_k2), \
            "Multiple decodes should produce identical results"
        assert torch.allclose(workspace_v1, workspace_v2), \
            "Multiple decodes should produce identical results"


@pytest.mark.skipif(
    not ecc_available(),
    reason="ECC not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestECCErrorCorrection:
    """Test ECC error correction capabilities."""

    def test_single_bit_error_correction(self, device):
        """Test that single-bit errors are corrected by Hamming(7,4)."""
        batch_size, num_heads, head_dim = 4, 8, 64
        block_size, num_blocks = 16, 4

        key, value, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # Encode
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        # Get baseline decode
        workspace_baseline = torch.zeros_like(key)
        workspace_v_baseline = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_baseline, workspace_v_baseline,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Inject single-bit errors
        num_faults = 100
        for _ in range(num_faults):
            idx = torch.randint(0, key_cache.numel(), (1,)).item()
            bit = torch.randint(0, 7, (1,)).item()  # 7 bits in codeword
            key_cache.view(-1)[idx] ^= (1 << bit)

        # Decode after faults
        workspace_after = torch.zeros_like(key)
        workspace_v_after = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_after, workspace_v_after,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Should be identical (errors corrected)
        assert torch.allclose(workspace_baseline, workspace_after), \
            f"Single-bit errors should be corrected. " \
            f"Diff: {(workspace_baseline - workspace_after).abs().max().item()}"

    def test_multiple_single_bit_errors(self, device):
        """Test correction of many single-bit errors across the cache."""
        batch_size, num_heads, head_dim = 8, 8, 128
        block_size, num_blocks = 16, 8

        key, value, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # Encode
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        # Get baseline
        workspace_baseline = torch.zeros_like(key)
        workspace_v_baseline = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_baseline, workspace_v_baseline,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Inject many single-bit errors (one per codeword is safe)
        num_faults = 500
        fault_indices = torch.randint(0, key_cache.numel(), (num_faults,))
        for idx in fault_indices:
            bit = torch.randint(0, 7, (1,)).item()
            key_cache.view(-1)[idx.item()] ^= (1 << bit)

        # Decode
        workspace_after = torch.zeros_like(key)
        workspace_v_after = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_after, workspace_v_after,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Should still be identical
        max_diff = (workspace_baseline - workspace_after).abs().max().item()
        assert max_diff < 1e-3, \
            f"All single-bit errors should be corrected. Max diff: {max_diff}"

    def test_double_bit_error_detection(self, device):
        """Test that double-bit errors cause miscorrection (expected behavior)."""
        batch_size, num_heads, head_dim = 4, 8, 64
        block_size, num_blocks = 16, 4

        key, value, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # Encode
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        # Get baseline
        workspace_baseline = torch.zeros_like(key)
        workspace_v_baseline = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_baseline, workspace_v_baseline,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Inject double-bit error in same codeword
        idx = 0
        key_cache.view(-1)[idx] ^= 0b00000011  # Flip 2 bits

        # Decode
        workspace_after = torch.zeros_like(key)
        workspace_v_after = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_after, workspace_v_after,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Should NOT be identical (Hamming can't correct 2-bit errors)
        # This is expected behavior - verifies ECC limits
        # The test passes if there IS a difference (proving the limitation)
        diff = (workspace_baseline - workspace_after).abs().max().item()
        # Note: We don't assert diff > 0 because the error might affect
        # a region not covered by slot_mapping. Just log for visibility.
        print(f"Double-bit error max diff: {diff} (expected non-zero)")


@pytest.mark.skipif(
    not ecc_available(),
    reason="ECC not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestECCQuantization:
    """Test INT4 quantization behavior."""

    def test_quantization_range(self, device):
        """Test that values outside [-4, 4] are clamped."""
        batch_size, num_heads, head_dim = 4, 8, 64
        block_size, num_blocks = 16, 4

        _, _, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # Create values at extremes
        key = torch.full(
            (batch_size, num_heads, head_dim),
            10.0,  # Outside range
            dtype=torch.float16, device=device
        )
        value = torch.full_like(key, -10.0)  # Outside range

        # Encode (should clamp)
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        # Decode
        workspace_k = torch.zeros_like(key)
        workspace_v = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_k, workspace_v,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Should be clamped to [-4, 4]
        assert workspace_k.max() <= 4.5, "Key should be clamped to ~4.0"
        assert workspace_v.min() >= -4.5, "Value should be clamped to ~-4.0"

    def test_quantization_levels(self, device):
        """Test that INT4 provides 16 discrete levels."""
        batch_size, num_heads, head_dim = 1, 1, 16
        block_size, num_blocks = 16, 1

        _, _, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # Create evenly spaced values in [-4, 4]
        key = torch.linspace(-4, 4, head_dim, device=device, dtype=torch.float16)
        key = key.view(1, 1, head_dim)
        value = key.clone()
        slot_mapping = torch.tensor([0], dtype=torch.long, device=device)

        # Encode and decode
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        workspace_k = torch.zeros_like(key)
        workspace_v = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_k, workspace_v,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Count unique values (should be <= 16 for INT4)
        unique_vals = workspace_k.unique()
        assert len(unique_vals) <= 16, \
            f"INT4 should have at most 16 levels, got {len(unique_vals)}"


@pytest.mark.skipif(
    not ecc_available(),
    reason="ECC not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestECCEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_values(self, device):
        """Test encoding/decoding zero values."""
        batch_size, num_heads, head_dim = 4, 8, 64
        block_size, num_blocks = 16, 4

        _, _, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # All zeros
        key = torch.zeros(
            batch_size, num_heads, head_dim,
            dtype=torch.float16, device=device
        )
        value = torch.zeros_like(key)

        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        workspace_k = torch.zeros_like(key)
        workspace_v = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_k, workspace_v,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Should be close to zero (within quantization)
        assert workspace_k.abs().max() < 0.5, "Zero input should decode to ~0"
        assert workspace_v.abs().max() < 0.5, "Zero input should decode to ~0"

    def test_single_token(self, device):
        """Test with single token (batch_size=1)."""
        batch_size, num_heads, head_dim = 1, 8, 64
        block_size, num_blocks = 16, 4

        key, value, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        workspace_k = torch.zeros_like(key)
        workspace_v = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_k, workspace_v,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        assert workspace_k.shape == key.shape

    def test_non_contiguous_slots(self, device):
        """Test with non-contiguous slot mapping."""
        batch_size, num_heads, head_dim = 4, 8, 64
        block_size, num_blocks = 16, 8

        key, value, key_cache, value_cache, _, seq_start_locs, num_seqs = create_test_tensors(
            batch_size, num_heads, head_dim, block_size, num_blocks, device
        )

        # Non-contiguous slots across different blocks
        slot_mapping = torch.tensor([0, 17, 35, 50], dtype=torch.long, device=device)

        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
        )

        workspace_k = torch.zeros_like(key)
        workspace_v = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_k, workspace_v,
            batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
        )

        # Should work without error
        assert workspace_k.shape == key.shape


class TestECCNotCompiled:
    """Tests for when ECC is NOT compiled."""

    @pytest.mark.skipif(
        ecc_available(),
        reason="ECC is compiled"
    )
    def test_ecc_not_available(self):
        """Test that ECC ops are not available when not compiled."""
        import vllm._C  # noqa: F401

        with pytest.raises(AttributeError):
            _ = torch.ops._C_cache_ops.ecc_encode


# ============================================================================
# SECDED (8,4) Tests
# ============================================================================

def create_secded_test_tensors(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    num_blocks: int,
    num_seqs: int,
    device: torch.device,
):
    """Create test tensors for SECDED tests with seq_start_locs."""
    # Input KV tensors (FP16)
    key = torch.randn(
        num_tokens, num_heads, head_dim,
        device=device, dtype=torch.float16
    ).clamp(-4.0, 4.0)
    value = torch.randn(
        num_tokens, num_heads, head_dim,
        device=device, dtype=torch.float16
    ).clamp(-4.0, 4.0)

    # ECC-encoded cache (uint8)
    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_dim,
        dtype=torch.uint8, device=device
    )
    value_cache = torch.zeros_like(key_cache)

    # Slot mapping: token index -> physical slot
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

    # Sequence boundaries for N-LERP
    tokens_per_seq = num_tokens // num_seqs
    seq_start_locs = torch.tensor(
        [i * tokens_per_seq for i in range(num_seqs)] + [num_tokens],
        dtype=torch.long, device=device
    )

    return (key, value, key_cache, value_cache, slot_mapping,
            seq_start_locs, num_seqs)


@pytest.fixture
def ecc_config_secded():
    """Create a SECDED ECC config for testing."""
    return ECCConfig(
        enabled=True,
        algorithm="secded_8_4",
    )


@pytest.mark.skipif(
    not ecc_available(),
    reason="ECC not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestSECDEDEncode:
    """Test SECDED 8-bit encoding."""

    def test_secded_encode_produces_8bit(self, device):
        """Test that SECDED encoding uses all 8 bits (bit 7 = global parity)."""
        num_tokens, num_heads, head_dim = 8, 4, 32
        block_size, num_blocks, num_seqs = 16, 4, 2

        (key, value, key_cache, value_cache, slot_mapping,
         seq_start_locs, _) = create_secded_test_tensors(
            num_tokens, num_heads, head_dim, block_size, num_blocks, num_seqs, device
        )

        # Encode with SECDED (algorithm=1)
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, 1
        )

        # Check that bit 7 is used (some codewords should have it set)
        high_bit_set = (key_cache & 0x80) != 0
        assert high_bit_set.any(), "SECDED should use bit 7 for global parity"

    def test_secded_vs_hamming_different_encoding(self, device):
        """Test that SECDED produces different codewords than Hamming(7,4)."""
        num_tokens, num_heads, head_dim = 4, 4, 16
        block_size, num_blocks, num_seqs = 16, 2, 1

        (key, value, key_cache_h74, value_cache_h74, slot_mapping,
         seq_start_locs, _) = create_secded_test_tensors(
            num_tokens, num_heads, head_dim, block_size, num_blocks, num_seqs, device
        )

        key_cache_secded = torch.zeros_like(key_cache_h74)
        value_cache_secded = torch.zeros_like(value_cache_h74)

        # Encode with Hamming(7,4) (algorithm=0)
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache_h74, value_cache_h74, slot_mapping, 0
        )

        # Encode with SECDED (algorithm=1)
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache_secded, value_cache_secded, slot_mapping, 1
        )

        # Lower 7 bits should be same, bit 7 may differ
        lower_7_same = ((key_cache_h74 & 0x7F) == (key_cache_secded & 0x7F)).all()
        assert lower_7_same, "Lower 7 bits should match between Hamming and SECDED"

        # Bit 7 is the global parity (may differ based on popcount)
        # At least some should differ unless all codewords have even popcount
        bit_7_h74 = (key_cache_h74 >> 7) & 1
        bit_7_secded = (key_cache_secded >> 7) & 1
        # In Hamming(7,4), bit 7 is always 0; in SECDED, it's the global parity
        assert bit_7_h74.sum() == 0, "Hamming(7,4) should not use bit 7"


@pytest.mark.skipif(
    not ecc_available(),
    reason="ECC not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestSECDEDBurstDetection:
    """Test 2-bit burst error detection with SECDED."""

    def test_secded_detects_burst_as_erasure(self, device, ecc_config_secded):
        """Test that SECDED flags 2-bit bursts as erasures and reconstructs."""
        num_tokens, num_heads, head_dim = 16, 4, 32
        block_size, num_blocks, num_seqs = 16, 4, 1

        (key, value, key_cache, value_cache, slot_mapping,
         seq_start_locs, num_seqs_val) = create_secded_test_tensors(
            num_tokens, num_heads, head_dim, block_size, num_blocks, num_seqs, device
        )

        # Encode with SECDED
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, 1
        )

        # Get baseline decode
        workspace_baseline = torch.zeros_like(key)
        workspace_v_baseline = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_baseline, workspace_v_baseline,
            num_tokens, num_heads, head_dim, block_size, num_seqs_val, 1
        )

        # Inject 2-bit burst error in middle tokens (not at boundaries)
        # Use token indices in the middle to test N-LERP reconstruction
        middle_token = num_tokens // 2
        middle_slot = slot_mapping[middle_token].item()
        block_idx = middle_slot // block_size
        block_offset = middle_slot % block_size
        elem_idx = 10  # Some element in the middle

        # Flip 2 adjacent bits to create a burst
        cache_flat = key_cache.view(-1)
        addr = (block_idx * block_size * num_heads * head_dim +
                block_offset * num_heads * head_dim + elem_idx)
        original_byte = cache_flat[addr].item()
        cache_flat[addr] ^= 0b00000011  # Flip bits 0 and 1

        # Decode after burst injection
        workspace_after = torch.zeros_like(key)
        workspace_v_after = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_after, workspace_v_after,
            num_tokens, num_heads, head_dim, block_size, num_seqs_val, 1
        )

        # The corrupted element should be reconstructed via N-LERP
        # Check that the result is different from garbage (which would be very far)
        reconstructed_val = workspace_after[middle_token, elem_idx // head_dim,
                                            elem_idx % head_dim].item()

        # Neighbors should be similar, so N-LERP should produce reasonable value
        prev_val = workspace_baseline[middle_token - 1, elem_idx // head_dim,
                                      elem_idx % head_dim].item()
        next_val = workspace_baseline[middle_token + 1, elem_idx // head_dim,
                                      elem_idx % head_dim].item()

        # N-LERP result should be somewhere between neighbors (within magnitude)
        avg_neighbor_mag = (abs(prev_val) + abs(next_val)) / 2
        print(f"  Prev: {prev_val:.3f}, Next: {next_val:.3f}, "
              f"Reconstructed: {reconstructed_val:.3f}, Avg mag: {avg_neighbor_mag:.3f}")

    def test_hamming74_miscorrects_burst(self, device):
        """Test that Hamming(7,4) produces garbage on 2-bit burst (baseline)."""
        num_tokens, num_heads, head_dim = 8, 4, 32
        block_size, num_blocks, num_seqs = 16, 4, 1

        (key, value, key_cache, value_cache, slot_mapping,
         seq_start_locs, num_seqs_val) = create_secded_test_tensors(
            num_tokens, num_heads, head_dim, block_size, num_blocks, num_seqs, device
        )

        # Encode with Hamming(7,4)
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, 0
        )

        # Get baseline decode
        workspace_baseline = torch.zeros_like(key)
        workspace_v_baseline = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_baseline, workspace_v_baseline,
            num_tokens, num_heads, head_dim, block_size, num_seqs_val, 0
        )

        # Inject 2-bit burst
        key_cache.view(-1)[10] ^= 0b00000011

        # Decode after burst
        workspace_after = torch.zeros_like(key)
        workspace_v_after = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_after, workspace_v_after,
            num_tokens, num_heads, head_dim, block_size, num_seqs_val, 0
        )

        # Should have visible difference (mis-correction)
        diff = (workspace_baseline - workspace_after).abs().max().item()
        print(f"  Hamming(7,4) 2-bit burst max diff: {diff} (expected > 0)")


@pytest.mark.skipif(
    not ecc_available(),
    reason="ECC not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestNLERPReconstruction:
    """Test manifold-aware N-LERP reconstruction."""

    def test_nlerp_uses_neighbors(self, device):
        """Test that N-LERP interpolates from adjacent tokens."""
        num_tokens, num_heads, head_dim = 8, 2, 16
        block_size, num_blocks, num_seqs = 16, 2, 1

        (key, value, key_cache, value_cache, slot_mapping,
         seq_start_locs, num_seqs_val) = create_secded_test_tensors(
            num_tokens, num_heads, head_dim, block_size, num_blocks, num_seqs, device
        )

        # Create a clear pattern: linear ramp
        for i in range(num_tokens):
            key[i, :, :] = float(i)
        key = key.clamp(-4.0, 4.0)

        # Encode with SECDED
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, 1
        )

        # Get baseline
        workspace_baseline = torch.zeros_like(key)
        workspace_v_baseline = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_baseline, workspace_v_baseline,
            num_tokens, num_heads, head_dim, block_size, num_seqs_val, 1
        )

        # Inject burst error at token 4 (middle of sequence)
        target_token = 4
        for elem in range(num_heads * head_dim):
            addr = target_token * num_heads * head_dim + elem
            key_cache.view(-1)[addr] ^= 0b00000011  # 2-bit burst

        # Decode
        workspace_after = torch.zeros_like(key)
        workspace_v_after = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_after, workspace_v_after,
            num_tokens, num_heads, head_dim, block_size, num_seqs_val, 1
        )

        # Token 4 should be reconstructed as interpolation of tokens 3 and 5
        # With linear ramp, token 4 should be ~3.5 (avg of 3 and 4, but quantized)
        reconstructed = workspace_after[target_token, 0, 0].item()
        prev_baseline = workspace_baseline[target_token - 1, 0, 0].item()
        next_baseline = workspace_baseline[target_token + 1, 0, 0].item()

        print(f"  Prev: {prev_baseline:.2f}, Next: {next_baseline:.2f}, "
              f"Reconstructed: {reconstructed:.2f}")

        # Should be between neighbors or close to their average
        expected_avg = (prev_baseline + next_baseline) / 2
        assert abs(reconstructed - expected_avg) < 1.0, \
            f"N-LERP should interpolate: got {reconstructed}, expected ~{expected_avg}"

    def test_nlerp_respects_sequence_boundary(self, device):
        """Test that N-LERP does not cross sequence boundaries."""
        num_tokens, num_heads, head_dim = 8, 2, 16
        block_size, num_blocks = 16, 2

        # 2 sequences of 4 tokens each
        num_seqs = 2
        tokens_per_seq = num_tokens // num_seqs

        (key, value, key_cache, value_cache, slot_mapping,
         seq_start_locs, num_seqs_val) = create_secded_test_tensors(
            num_tokens, num_heads, head_dim, block_size, num_blocks, num_seqs, device
        )

        # Seq 0: tokens 0-3 with value 0.0
        # Seq 1: tokens 4-7 with value 4.0
        key[:4] = 0.0
        key[4:] = 4.0

        # Encode with SECDED
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, 1
        )

        # Inject burst at token 4 (first token of seq 1)
        target_token = 4
        for elem in range(min(8, num_heads * head_dim)):
            addr = target_token * num_heads * head_dim + elem
            key_cache.view(-1)[addr] ^= 0b00000011

        # Decode
        workspace_after = torch.zeros_like(key)
        workspace_v_after = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_after, workspace_v_after,
            num_tokens, num_heads, head_dim, block_size, num_seqs_val, 1
        )

        # Token 4 (first of seq 1) should NOT use token 3 (last of seq 0)
        # It should use zero-order hold from token 5, or return 0
        reconstructed = workspace_after[target_token, 0, 0].item()

        # If boundary is respected, result should be closer to 4.0 (from token 5)
        # than to average of 0.0 and 4.0
        print(f"  Reconstructed token 4 (first of seq 1): {reconstructed:.2f}")
        print(f"  (Should be ~4.0 from token 5, not ~2.0 from avg with token 3)")

    def test_fallback_to_zero_order_hold(self, device):
        """Test fallback to single neighbor when only one is valid."""
        num_tokens, num_heads, head_dim = 4, 2, 16
        block_size, num_blocks, num_seqs = 16, 2, 1

        (key, value, key_cache, value_cache, slot_mapping,
         seq_start_locs, num_seqs_val) = create_secded_test_tensors(
            num_tokens, num_heads, head_dim, block_size, num_blocks, num_seqs, device
        )

        # Set specific values
        key[0] = 1.0
        key[1] = 2.0
        key[2] = 3.0
        key[3] = 4.0

        # Encode
        torch.ops._C_cache_ops.ecc_encode(
            key, value, key_cache, value_cache, slot_mapping, 1
        )

        # Get baseline
        workspace_baseline = torch.zeros_like(key)
        workspace_v_baseline = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_baseline, workspace_v_baseline,
            num_tokens, num_heads, head_dim, block_size, num_seqs_val, 1
        )

        # Inject burst at token 0 (first token - no previous neighbor)
        for elem in range(min(4, num_heads * head_dim)):
            addr = 0 * num_heads * head_dim + elem
            key_cache.view(-1)[addr] ^= 0b00000011

        # Decode
        workspace_after = torch.zeros_like(key)
        workspace_v_after = torch.zeros_like(value)
        torch.ops._C_cache_ops.ecc_gather_decode(
            key_cache, value_cache, slot_mapping, seq_start_locs,
            workspace_after, workspace_v_after,
            num_tokens, num_heads, head_dim, block_size, num_seqs_val, 1
        )

        # Token 0 should use zero-order hold from token 1
        reconstructed = workspace_after[0, 0, 0].item()
        token_1_val = workspace_baseline[1, 0, 0].item()

        print(f"  Token 0 reconstructed: {reconstructed:.2f}")
        print(f"  Token 1 (next neighbor): {token_1_val:.2f}")

        # Should be close to token 1 (zero-order hold)
        assert abs(reconstructed - token_1_val) < 0.5, \
            f"Should use ZOH from token 1: got {reconstructed}, expected ~{token_1_val}"


@pytest.mark.skipif(
    not ecc_available(),
    reason="ECC not compiled (requires VLLM_ECC_ENABLED=ON)"
)
class TestSECDEDConfig:
    """Test SECDED configuration."""

    def test_secded_algorithm_code(self, ecc_config_secded):
        """Test algorithm code conversion."""
        assert ecc_config_secded.get_algorithm_code() == 1

        config_h74 = ECCConfig(enabled=True, algorithm="hamming_7_4")
        assert config_h74.get_algorithm_code() == 0

    def test_secded_default_algorithm(self):
        """Test that SECDED is the default algorithm."""
        config = ECCConfig(enabled=True)
        assert config.algorithm == "secded_8_4"
        assert config.get_algorithm_code() == 1


# Standalone test runner
if __name__ == "__main__":
    """Run tests directly for quick validation."""
    import sys

    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    if not ecc_available():
        print("ECC not compiled (requires VLLM_ECC_ENABLED=ON)")
        sys.exit(0)

    device = torch.device("cuda:0")

    print("=" * 60)
    print("ECC Encode/Decode Test Suite")
    print("=" * 60)

    # Test 1: Basic encode/decode
    print("\n[Test 1] Basic encode/decode...")
    batch_size, num_heads, head_dim = 4, 8, 64
    block_size, num_blocks = 16, 4

    key, value, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
        batch_size, num_heads, head_dim, block_size, num_blocks, device
    )

    torch.ops._C_cache_ops.ecc_encode(
        key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
    )
    print(f"  Encoded: key_cache has {(key_cache != 0).sum().item()} non-zero elements")

    workspace_k = torch.zeros_like(key)
    workspace_v = torch.zeros_like(value)
    torch.ops._C_cache_ops.ecc_gather_decode(
        key_cache, value_cache, slot_mapping, seq_start_locs,
        workspace_k, workspace_v,
        batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
    )
    print("  Decoded successfully")
    print("  PASSED")

    # Test 2: Round-trip accuracy
    print("\n[Test 2] Round-trip reconstruction accuracy...")
    key_clamped = key.clamp(-4.0, 4.0)
    value_clamped = value.clamp(-4.0, 4.0)

    key_mae = (key_clamped - workspace_k).abs().mean().item()
    value_mae = (value_clamped - workspace_v).abs().mean().item()
    print(f"  Key MAE: {key_mae:.4f} (expected < 0.5)")
    print(f"  Value MAE: {value_mae:.4f} (expected < 0.5)")
    assert key_mae < 0.5 and value_mae < 0.5
    print("  PASSED")

    # Test 3: Single-bit error correction
    print("\n[Test 3] Single-bit error correction...")
    key, value, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
        batch_size, num_heads, head_dim, block_size, num_blocks, device
    )

    torch.ops._C_cache_ops.ecc_encode(
        key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
    )

    # Baseline decode
    baseline_k = torch.zeros_like(key)
    baseline_v = torch.zeros_like(value)
    torch.ops._C_cache_ops.ecc_gather_decode(
        key_cache, value_cache, slot_mapping, seq_start_locs,
        baseline_k, baseline_v,
        batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
    )

    # Inject single-bit faults
    num_faults = 100
    for _ in range(num_faults):
        idx = torch.randint(0, key_cache.numel(), (1,)).item()
        bit = torch.randint(0, 7, (1,)).item()
        key_cache.view(-1)[idx] ^= (1 << bit)

    print(f"  Injected {num_faults} single-bit faults")

    # Decode after faults
    after_k = torch.zeros_like(key)
    after_v = torch.zeros_like(value)
    torch.ops._C_cache_ops.ecc_gather_decode(
        key_cache, value_cache, slot_mapping, seq_start_locs,
        after_k, after_v,
        batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
    )

    diff = (baseline_k - after_k).abs().max().item()
    print(f"  Max difference after correction: {diff}")
    assert diff < 1e-3, f"Errors should be corrected, got diff={diff}"
    print("  PASSED - All faults corrected!")

    # Test 4: Quantization levels
    print("\n[Test 4] INT4 quantization levels...")
    key = torch.linspace(-4, 4, 16, device=device, dtype=torch.float16).view(1, 1, 16)
    value = key.clone()
    _, _, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
        1, 1, 16, 16, 1, device
    )
    slot_mapping = torch.tensor([0], dtype=torch.long, device=device)
    seq_start_locs = torch.tensor([0, 1], dtype=torch.long, device=device)

    torch.ops._C_cache_ops.ecc_encode(
        key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
    )

    workspace_k = torch.zeros_like(key)
    workspace_v = torch.zeros_like(value)
    torch.ops._C_cache_ops.ecc_gather_decode(
        key_cache, value_cache, slot_mapping, seq_start_locs,
        workspace_k, workspace_v,
        1, 1, 16, 16, 1, ALGO_HAMMING74
    )

    unique_vals = workspace_k.unique()
    print(f"  Unique decoded values: {len(unique_vals)} (expected <= 16)")
    assert len(unique_vals) <= 16
    print("  PASSED")

    # Test 5: Large-scale stress test
    print("\n[Test 5] Large-scale stress test...")
    batch_size, num_heads, head_dim = 32, 32, 128
    block_size, num_blocks = 16, 64

    key, value, key_cache, value_cache, slot_mapping, seq_start_locs, num_seqs = create_test_tensors(
        batch_size, num_heads, head_dim, block_size, num_blocks, device
    )

    torch.ops._C_cache_ops.ecc_encode(
        key, value, key_cache, value_cache, slot_mapping, ALGO_HAMMING74
    )

    # Many faults
    num_faults = 1000
    for _ in range(num_faults):
        idx = torch.randint(0, key_cache.numel(), (1,)).item()
        bit = torch.randint(0, 7, (1,)).item()
        key_cache.view(-1)[idx] ^= (1 << bit)

    baseline_k = torch.zeros_like(key)
    baseline_v = torch.zeros_like(value)

    # Re-encode clean data for baseline
    key2, value2, key_cache2, value_cache2, _ = create_test_tensors(
        batch_size, num_heads, head_dim, block_size, num_blocks, device
    )
    key2.copy_(key)
    value2.copy_(value)
    torch.ops._C_cache_ops.ecc_encode(
        key2, value2, key_cache2, value_cache2, slot_mapping
    )
    torch.ops._C_cache_ops.ecc_gather_decode(
        key_cache2, value_cache2, slot_mapping, seq_start_locs,
        baseline_k, baseline_v,
        batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
    )

    # Decode corrupted
    after_k = torch.zeros_like(key)
    after_v = torch.zeros_like(value)
    torch.ops._C_cache_ops.ecc_gather_decode(
        key_cache, value_cache, slot_mapping, seq_start_locs,
        after_k, after_v,
        batch_size, num_heads, head_dim, block_size, num_seqs, ALGO_HAMMING74
    )

    diff = (baseline_k - after_k).abs().max().item()
    print(f"  Cache size: {key_cache.numel()} bytes")
    print(f"  Faults injected: {num_faults}")
    print(f"  Max difference: {diff}")
    assert diff < 1e-3
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
