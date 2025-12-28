# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ECC (Error Correction Code) configuration for KV cache protection.

This module provides configuration for the Hamming(7,4) ECC layer that
protects the KV cache against single-bit memory faults.
"""

from typing import Any, Literal

from pydantic import Field
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

ECCAlgorithm = Literal["hamming_7_4", "secded_8_4", "lsq_secded_16_11"]


# Attention backends compatible with ECC
ECC_COMPATIBLE_BACKENDS = frozenset({
    "FLASH_ATTN",
    "FLASHINFER",
})


@config
@dataclass
class ECCConfig:
    """Configuration for KV cache ECC (Error Correction Code) protection.

    This feature enables error correction for the KV cache, protecting against
    memory faults during inference. Requires compilation with VLLM_ECC_ENABLED=ON.

    Three algorithms are supported:
    - **hamming_7_4**: Hamming(7,4) with single-bit error correction
    - **secded_8_4**: Extended Hamming(8,4) SECDED with:
      - Single-bit error correction
      - Double-bit error detection (flagged as erasures)
      - N-LERP manifold-aware reconstruction from temporal neighbors
    - **lsq_secded_16_11**: Lattice Syndrome Quantization + SECDED(16,11) with:
      - Pairs adjacent head_size dimensions (elem_2i, elem_2i+1)
      - 6-bit anchor (absolute quantization) + 5-bit syndrome (predictor-based)
      - Uses Hadamard rotation to enable predictor correlation
      - Requires rotation of Keys (and Queries) before attention

    The ECC layer:
    1. Quantizes FP16 values to INT4 (range [-4.0, 4.0] -> [0, 15])
    2. Encodes 4 data bits to 7-bit (Hamming) or 8-bit (SECDED) codeword
    3. Stores as uint8 in the cache
    4. On read: decodes, corrects/detects errors, dequantizes back to FP16
    5. (SECDED only) Reconstructs erased values via N-LERP interpolation

    For LSQ mode:
    1. Applies Hadamard rotation to Keys (and Queries)
    2. Pairs adjacent dimensions into 16-bit codewords
    3. Uses predictor-based syndrome for higher precision
    4. Stores as uint16 in the cache (same memory footprint)

    IMPORTANT LIMITATIONS (Proof-of-Concept):
    -----------------------------------------
    1. **FlashAttention-only**: ECC uses a linear workspace layout that is
       incompatible with PagedAttention V1/V2 kernels. You MUST use a
       FlashAttention-based backend (FLASH_ATTN or FLASHINFER).

    2. **No mixed mode**: All KV cache blocks use ECC encoding. You cannot
       have some blocks with ECC and some without.

    3. **INT4 quantization loss**: Values outside [-4.0, 4.0] are clamped.
       Models with large activation values may see quality degradation.
       LSQ mode uses [-40.0, 40.0] range for anchor values.

    4. **Memory overhead**: The ECC decode path requires a temporary
       workspace buffer (2x memory for active batch KV).

    5. **Error correction limits**:
       - Hamming(7,4): Corrects 1-bit errors, mis-corrects 2-bit bursts
       - SECDED(8,4): Corrects 1-bit, detects 2-bit, reconstructs via N-LERP
       - LSQ SECDED(16,11): Same as SECDED but with paired dimensions

    6. **LSQ requires even head_size**: Head dimensions must be divisible by 2.
    """

    enabled: bool = False
    """Enable ECC protection. Requires VLLM_ECC_ENABLED compile flag."""

    algorithm: ECCAlgorithm = "secded_8_4"
    """ECC algorithm to use:
    - hamming_7_4: Single-bit error correction (baseline, may mis-correct bursts)
    - secded_8_4: SECDED with N-LERP reconstruction (detects 2-bit bursts,
                  reconstructs via manifold-aware interpolation)
    - lsq_secded_16_11: LSQ with SECDED(16,11) and Hadamard rotation
    """

    # LSQ-specific options
    lsq_anchor_range: float = 40.0
    """Anchor quantization range for LSQ mode. Values in [-range, range]."""

    lsq_apply_hadamard: bool = True
    """Apply Hadamard rotation to keys (and queries) in LSQ mode."""

    # Quantization range is hardcoded in ecc_math.cuh:
    # Range: [-4.0, 4.0] -> [0, 15]
    # QUANT_SCALE = 1.875 (15.0 / 8.0)
    # QUANT_ZERO = 7.5
    #
    # If baseline quality drops (clipping), the range can be increased
    # to [-8.0, 8.0] by modifying QUANT_SCALE in ecc_math.cuh.

    def validate_backend(self, attention_backend: str) -> None:
        """Validate that the attention backend is compatible with ECC.

        Args:
            attention_backend: The attention backend name (e.g., "FLASH_ATTN")

        Raises:
            ValueError: If ECC is enabled but backend is incompatible.
        """
        if not self.enabled:
            return

        backend_upper = attention_backend.upper()
        if backend_upper not in ECC_COMPATIBLE_BACKENDS:
            raise ValueError(
                f"ECC protection requires a FlashAttention-based backend. "
                f"Got '{attention_backend}', but ECC only supports: "
                f"{sorted(ECC_COMPATIBLE_BACKENDS)}. "
                f"ECC uses a linear workspace layout incompatible with "
                f"PagedAttention V1/V2 kernels."
            )

    def check_compiled(self) -> bool:
        """Check if ECC support was compiled into vLLM.

        Returns:
            True if ECC ops are available, False otherwise.
        """
        if not self.enabled:
            return True  # Not needed if disabled

        try:
            import torch
            import vllm._C  # noqa: F401
            _ = torch.ops._C_cache_ops.ecc_encode
            return True
        except (ImportError, AttributeError):
            return False

    def get_algorithm_code(self) -> int:
        """Get the integer code for the selected algorithm.

        Returns:
            0 for hamming_7_4, 1 for secded_8_4, 2 for lsq_secded_16_11
        """
        if self.algorithm == "hamming_7_4":
            return 0
        elif self.algorithm == "secded_8_4":
            return 1
        else:  # lsq_secded_16_11
            return 2

    def uses_lsq(self) -> bool:
        """Check if LSQ mode is being used.

        LSQ uses uint16 cache storage instead of uint8.
        """
        return self.algorithm == "lsq_secded_16_11"

    def uses_uint16_cache(self) -> bool:
        """Check if this algorithm stores pairs as uint16.

        LSQ stores pairs of values in 16-bit codewords.
        """
        return self.uses_lsq()

    def compute_hash(self) -> str:
        """Compute hash for CUDA graph caching.

        ECC affects computation results, so changing any ECC parameter
        invalidates CUDA graphs.
        """
        factors: list[Any] = [
            self.enabled,
            self.algorithm,
            self.lsq_anchor_range,
            self.lsq_apply_hadamard,
        ]
        hash_str = safe_hash(str(factors).encode(),
                             usedforsecurity=False).hexdigest()
        return hash_str
