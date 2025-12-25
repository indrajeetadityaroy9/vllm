# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for fault injection in KV cache kernels.

These tests verify the fault injection functionality for:
- Random bit-flip model
- Burst bit-flip model
- MSB-biased bit-flip model
- Page-local injection

Note: These tests require vLLM to be compiled with VLLM_FAULT_INJECT=ON.
Run with: cmake -DVLLM_FAULT_INJECT=ON ...
"""

import pytest
import torch

from vllm.config import FaultInjectionConfig


def fault_injection_available() -> bool:
    """Check if fault injection is compiled in."""
    try:
        return hasattr(torch.ops._C, "set_fault_injection_config")
    except Exception:
        return False


@pytest.fixture
def fault_config_random():
    """Create a random fault injection config for testing."""
    return FaultInjectionConfig(
        enabled=True,
        site="BOTH",
        subsite="CODEWORD",
        model="random",
        rate=0.1,
        flip_count=1,
        burst_len=8,
        msb_policy="BYTE_TOPBITS",
        msb_mask=0xF0,
        page_scope=-1,
        seed=42,
    )


@pytest.fixture
def fault_config_burst():
    """Create a burst fault injection config for testing."""
    return FaultInjectionConfig(
        enabled=True,
        site="KV_WRITE",
        subsite="CODEWORD",
        model="burst",
        rate=0.5,
        flip_count=1,
        burst_len=8,
        msb_policy="BYTE_TOPBITS",
        msb_mask=0xF0,
        page_scope=-1,
        seed=42,
    )


@pytest.fixture
def fault_config_msb():
    """Create an MSB-biased fault injection config for testing."""
    return FaultInjectionConfig(
        enabled=True,
        site="BOTH",
        subsite="VALUE",
        model="msb_biased",
        rate=0.3,
        flip_count=2,
        burst_len=8,
        msb_policy="FP16_EXPONENT",
        msb_mask=0xF0,
        page_scope=-1,
        seed=42,
    )


@pytest.fixture
def fault_config_page_local():
    """Create a page-local fault injection config for testing."""
    return FaultInjectionConfig(
        enabled=True,
        site="KV_WRITE",
        subsite="CODEWORD",
        model="page_local",
        rate=1.0,
        flip_count=1,
        burst_len=8,
        msb_policy="BYTE_TOPBITS",
        msb_mask=0xF0,
        page_scope=0,  # Target only block 0
        seed=42,
    )


@pytest.mark.skipif(
    not fault_injection_available(),
    reason="Fault injection not compiled (requires VLLM_FAULT_INJECT=ON)"
)
class TestFaultInjectionConfig:
    """Test FaultInjectionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FaultInjectionConfig()
        assert config.enabled is False
        assert config.site == "BOTH"
        assert config.subsite == "CODEWORD"
        assert config.model == "random"
        assert config.rate == 0.0
        assert config.flip_count == 1
        assert config.burst_len == 8
        assert config.msb_policy == "BYTE_TOPBITS"
        assert config.msb_mask == 0xF0
        assert config.page_scope == -1
        assert config.seed == 42

    def test_compute_hash(self, fault_config_random):
        """Test that hash computation is deterministic."""
        hash1 = fault_config_random.compute_hash()
        hash2 = fault_config_random.compute_hash()
        assert hash1 == hash2

        # Different config should produce different hash
        config2 = FaultInjectionConfig(enabled=True, rate=0.5, seed=123)
        hash3 = config2.compute_hash()
        assert hash1 != hash3

    def test_rate_validation(self):
        """Test rate validation bounds."""
        # Valid rates
        FaultInjectionConfig(rate=0.0)
        FaultInjectionConfig(rate=0.5)
        FaultInjectionConfig(rate=1.0)

        # Invalid rates should raise
        with pytest.raises(ValueError):
            FaultInjectionConfig(rate=-0.1)
        with pytest.raises(ValueError):
            FaultInjectionConfig(rate=1.5)

    def test_msb_mask_validation(self):
        """Test MSB mask validation bounds."""
        # Valid masks
        FaultInjectionConfig(msb_mask=0x00)
        FaultInjectionConfig(msb_mask=0xFF)

        # Invalid masks should raise
        with pytest.raises(ValueError):
            FaultInjectionConfig(msb_mask=-1)
        with pytest.raises(ValueError):
            FaultInjectionConfig(msb_mask=0x100)


@pytest.mark.skipif(
    not fault_injection_available(),
    reason="Fault injection not compiled (requires VLLM_FAULT_INJECT=ON)"
)
class TestSetFaultInjectionConfig:
    """Test the set_fault_injection_config custom op."""

    def test_set_config_random(self, fault_config_random):
        """Test setting random fault injection config."""
        from vllm._custom_ops import set_fault_injection_config

        # Should not raise
        set_fault_injection_config(
            enabled=fault_config_random.enabled,
            site=fault_config_random.site,
            subsite=fault_config_random.subsite,
            model=fault_config_random.model,
            rate=fault_config_random.rate,
            flip_count=fault_config_random.flip_count,
            burst_len=fault_config_random.burst_len,
            msb_policy=fault_config_random.msb_policy,
            msb_mask=fault_config_random.msb_mask,
            page_scope=fault_config_random.page_scope,
            seed=fault_config_random.seed,
        )

    def test_set_config_burst(self, fault_config_burst):
        """Test setting burst fault injection config."""
        from vllm._custom_ops import set_fault_injection_config

        set_fault_injection_config(
            enabled=fault_config_burst.enabled,
            site=fault_config_burst.site,
            subsite=fault_config_burst.subsite,
            model=fault_config_burst.model,
            rate=fault_config_burst.rate,
            flip_count=fault_config_burst.flip_count,
            burst_len=fault_config_burst.burst_len,
            msb_policy=fault_config_burst.msb_policy,
            msb_mask=fault_config_burst.msb_mask,
            page_scope=fault_config_burst.page_scope,
            seed=fault_config_burst.seed,
        )

    def test_set_config_msb(self, fault_config_msb):
        """Test setting MSB-biased fault injection config."""
        from vllm._custom_ops import set_fault_injection_config

        set_fault_injection_config(
            enabled=fault_config_msb.enabled,
            site=fault_config_msb.site,
            subsite=fault_config_msb.subsite,
            model=fault_config_msb.model,
            rate=fault_config_msb.rate,
            flip_count=fault_config_msb.flip_count,
            burst_len=fault_config_msb.burst_len,
            msb_policy=fault_config_msb.msb_policy,
            msb_mask=fault_config_msb.msb_mask,
            page_scope=fault_config_msb.page_scope,
            seed=fault_config_msb.seed,
        )

    def test_set_config_page_local(self, fault_config_page_local):
        """Test setting page-local fault injection config."""
        from vllm._custom_ops import set_fault_injection_config

        set_fault_injection_config(
            enabled=fault_config_page_local.enabled,
            site=fault_config_page_local.site,
            subsite=fault_config_page_local.subsite,
            model=fault_config_page_local.model,
            rate=fault_config_page_local.rate,
            flip_count=fault_config_page_local.flip_count,
            burst_len=fault_config_page_local.burst_len,
            msb_policy=fault_config_page_local.msb_policy,
            msb_mask=fault_config_page_local.msb_mask,
            page_scope=fault_config_page_local.page_scope,
            seed=fault_config_page_local.seed,
        )

    def test_disable_after_enable(self, fault_config_random):
        """Test that fault injection can be disabled after being enabled."""
        from vllm._custom_ops import set_fault_injection_config

        # Enable
        set_fault_injection_config(
            enabled=True,
            site=fault_config_random.site,
            subsite=fault_config_random.subsite,
            model=fault_config_random.model,
            rate=fault_config_random.rate,
            flip_count=fault_config_random.flip_count,
            burst_len=fault_config_random.burst_len,
            msb_policy=fault_config_random.msb_policy,
            msb_mask=fault_config_random.msb_mask,
            page_scope=fault_config_random.page_scope,
            seed=fault_config_random.seed,
        )

        # Disable
        set_fault_injection_config(
            enabled=False,
            site=fault_config_random.site,
            subsite=fault_config_random.subsite,
            model=fault_config_random.model,
            rate=0.0,
            flip_count=fault_config_random.flip_count,
            burst_len=fault_config_random.burst_len,
            msb_policy=fault_config_random.msb_policy,
            msb_mask=fault_config_random.msb_mask,
            page_scope=fault_config_random.page_scope,
            seed=fault_config_random.seed,
        )


class TestFaultInjectionNotCompiled:
    """Tests for when fault injection is NOT compiled."""

    @pytest.mark.skipif(
        fault_injection_available(),
        reason="Fault injection is compiled"
    )
    def test_warning_when_not_compiled(self, caplog):
        """Test that a warning is logged when fault injection is not compiled."""
        from vllm._custom_ops import set_fault_injection_config

        set_fault_injection_config(
            enabled=True,
            site="BOTH",
            subsite="CODEWORD",
            model="random",
            rate=0.1,
            flip_count=1,
            burst_len=8,
            msb_policy="BYTE_TOPBITS",
            msb_mask=0xF0,
            page_scope=-1,
            seed=42,
        )
        # Should log a warning about fault injection not being available


@pytest.mark.skipif(
    not fault_injection_available(),
    reason="Fault injection not compiled (requires VLLM_FAULT_INJECT=ON)"
)
class TestFaultInjectionDeterminism:
    """Test deterministic behavior with same seed."""

    def test_same_seed_same_result(self, fault_config_random):
        """Test that same seed produces same behavior."""
        from vllm._custom_ops import set_fault_injection_config

        # Set config with seed 42
        set_fault_injection_config(
            enabled=fault_config_random.enabled,
            site=fault_config_random.site,
            subsite=fault_config_random.subsite,
            model=fault_config_random.model,
            rate=fault_config_random.rate,
            flip_count=fault_config_random.flip_count,
            burst_len=fault_config_random.burst_len,
            msb_policy=fault_config_random.msb_policy,
            msb_mask=fault_config_random.msb_mask,
            page_scope=fault_config_random.page_scope,
            seed=42,
        )

        # Set same config again
        set_fault_injection_config(
            enabled=fault_config_random.enabled,
            site=fault_config_random.site,
            subsite=fault_config_random.subsite,
            model=fault_config_random.model,
            rate=fault_config_random.rate,
            flip_count=fault_config_random.flip_count,
            burst_len=fault_config_random.burst_len,
            msb_policy=fault_config_random.msb_policy,
            msb_mask=fault_config_random.msb_mask,
            page_scope=fault_config_random.page_scope,
            seed=42,
        )
        # Same seed should produce deterministic results


class TestFaultInjectionValidation:
    """Test input validation for fault injection config."""

    def test_invalid_site_raises(self):
        """Test that invalid site value raises ValueError."""
        from vllm._custom_ops import set_fault_injection_config

        with pytest.raises(ValueError, match="Invalid fault injection site"):
            set_fault_injection_config(
                enabled=True,
                site="INVALID_SITE",
                subsite="CODEWORD",
                model="random",
                rate=0.1,
                flip_count=1,
                burst_len=8,
                msb_policy="BYTE_TOPBITS",
                msb_mask=0xF0,
                page_scope=-1,
                seed=42,
            )

    def test_invalid_subsite_raises(self):
        """Test that invalid subsite value raises ValueError."""
        from vllm._custom_ops import set_fault_injection_config

        with pytest.raises(ValueError, match="Invalid fault injection subsite"):
            set_fault_injection_config(
                enabled=True,
                site="BOTH",
                subsite="INVALID_SUBSITE",
                model="random",
                rate=0.1,
                flip_count=1,
                burst_len=8,
                msb_policy="BYTE_TOPBITS",
                msb_mask=0xF0,
                page_scope=-1,
                seed=42,
            )

    def test_invalid_model_raises(self):
        """Test that invalid model value raises ValueError."""
        from vllm._custom_ops import set_fault_injection_config

        with pytest.raises(ValueError, match="Invalid fault injection model"):
            set_fault_injection_config(
                enabled=True,
                site="BOTH",
                subsite="CODEWORD",
                model="invalid_model",
                rate=0.1,
                flip_count=1,
                burst_len=8,
                msb_policy="BYTE_TOPBITS",
                msb_mask=0xF0,
                page_scope=-1,
                seed=42,
            )

    def test_invalid_msb_policy_raises(self):
        """Test that invalid msb_policy value raises ValueError."""
        from vllm._custom_ops import set_fault_injection_config

        with pytest.raises(ValueError, match="Invalid fault injection msb_policy"):
            set_fault_injection_config(
                enabled=True,
                site="BOTH",
                subsite="CODEWORD",
                model="random",
                rate=0.1,
                flip_count=1,
                burst_len=8,
                msb_policy="INVALID_POLICY",
                msb_mask=0xF0,
                page_scope=-1,
                seed=42,
            )
