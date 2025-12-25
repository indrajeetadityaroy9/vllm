# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

FaultInjectionSite = Literal["KV_WRITE", "KV_READ", "BOTH"]
FaultInjectionSubsite = Literal["CODEWORD", "VALUE"]
FaultInjectionModel = Literal["random", "burst", "msb_biased", "page_local"]
MSBPolicy = Literal["BYTE_TOPBITS", "FP16_EXPONENT", "INT4_NIBBLE"]


@config
@dataclass
class FaultInjectionConfig:
    """Configuration for KV cache fault injection testing.

    This feature enables bit-flip noise injection at KV cache write/read sites
    for testing ECC effectiveness and compute sensitivity. Requires compilation
    with VLLM_FAULT_INJECT=ON.

    WARNING: KV_READ injection only works with legacy PagedAttention backend.
    Set VLLM_ATTENTION_BACKEND=XFORMERS to enable KV_READ injection.
    """

    enabled: bool = False
    """Enable fault injection. Requires VLLM_FAULT_INJECT compile flag."""

    site: FaultInjectionSite = "BOTH"
    """Injection site: KV_WRITE (during cache population), KV_READ (during
    attention), or BOTH. Note: KV_READ requires legacy backend."""

    subsite: FaultInjectionSubsite = "CODEWORD"
    """CODEWORD: inject at packed/encoded representation (ECC-relevant).
    VALUE: inject after dequantization (compute sensitivity only)."""

    model: FaultInjectionModel = "random"
    """Fault model: random (uniform), burst (consecutive bits),
    msb_biased (significant bits), page_local (specific block)."""

    rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Probability that a block is corrupted (0.0-1.0)."""

    flip_count: int = Field(default=1, ge=1, le=8)
    """Number of bit flips per corrupted block (max 8)."""

    burst_len: int = Field(default=8, ge=1, le=64)
    """Number of consecutive bits to flip in burst model."""

    msb_policy: MSBPolicy = "BYTE_TOPBITS"
    """MSB-biased policy: BYTE_TOPBITS (top bits of byte),
    FP16_EXPONENT (exponent bits), INT4_NIBBLE (nibble MSBs)."""

    msb_mask: int = Field(default=0xF0, ge=0, le=0xFF)
    """Bitmask for BYTE_TOPBITS policy (default 0xF0 = top 4 bits)."""

    page_scope: int = Field(default=-1, ge=-1)
    """Physical block number to target (-1 = all blocks)."""

    seed: int = Field(default=42, ge=0)
    """Random seed for deterministic replay."""

    @field_validator("rate")
    @classmethod
    def _validate_rate(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"rate must be between 0.0 and 1.0, got {value}")
        return value

    @field_validator("msb_mask")
    @classmethod
    def _validate_msb_mask(cls, value: int) -> int:
        if value < 0 or value > 0xFF:
            raise ValueError(f"msb_mask must be between 0x00 and 0xFF, got {value}")
        return value

    def compute_hash(self) -> str:
        """Compute hash for CUDA graph caching.

        WARNING: Fault injection affects computation results, so changing
        any fault injection parameter invalidates CUDA graphs.
        """
        factors: list[Any] = [
            self.enabled,
            self.site,
            self.subsite,
            self.model,
            self.rate,
            self.flip_count,
            self.burst_len,
            self.msb_policy,
            self.msb_mask,
            self.page_scope,
            self.seed,
        ]
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str
