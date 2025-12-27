// SPDX-License-Identifier: Apache-2.0
/**
 * ECC Math Primitives for KV Cache Protection
 *
 * Implements both Hamming(7,4) and Extended Hamming(8,4) SECDED encoding.
 *
 * Hamming(7,4): Single-bit error correction
 * SECDED(8,4): Single-bit correction + Double-bit detection (via 8th parity bit)
 *
 * Uses systematic form where data bits are in positions 3,5,6,7
 * and parity bits are in positions 1,2,4.
 *
 * Generator Matrix (systematic form):
 *   G = [1 0 0 0 | 1 1 0]   d0 at pos 3
 *       [0 1 0 0 | 1 0 1]   d1 at pos 5
 *       [0 0 1 0 | 0 1 1]   d2 at pos 6
 *       [0 0 0 1 | 1 1 1]   d3 at pos 7
 *
 * Parity equations:
 *   p0 (pos 1) = d0 ^ d1 ^ d3
 *   p1 (pos 2) = d0 ^ d2 ^ d3
 *   p2 (pos 4) = d1 ^ d2 ^ d3
 *   p3 (pos 8, SECDED only) = global parity over bits 0-6
 *
 * Syndrome indicates error position (0 = no error, 1-7 = bit position)
 */

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace vllm {
namespace ecc {

// ============================================================================
// ECC Status Enum (for SECDED decode)
// ============================================================================
enum EccStatus {
    ECC_OK = 0,        // No error detected
    ECC_CORRECTED = 1, // Single-bit error corrected
    ECC_ERASURE = 2    // Double-bit error detected (cannot correct)
};

// ============================================================================
// Algorithm Selection Constants
// ============================================================================
#define ECC_ALGO_HAMMING74 0
#define ECC_ALGO_SECDED    1

// ============================================================================
// Quantization Constants
// ============================================================================
// Range: [-4.0, 4.0] → [0, 15]
// Safe for LLaMA-3 (SwiGLU outputs are usually small)
// If baseline quality drops (clipping), increase to [-8.0, 8.0]
constexpr float QUANT_SCALE = 1.875f;  // 15.0 / 8.0
constexpr float QUANT_ZERO = 7.5f;
constexpr float QUANT_MIN = -4.0f;
constexpr float QUANT_MAX = 4.0f;

// ============================================================================
// Quantization Functions
// ============================================================================

/**
 * Quantize FP16 to INT4 (0-15 range)
 * Uses symmetric quantization with clipping
 */
__device__ __forceinline__ uint8_t fp16_to_int4(half val) {
    float f = __half2float(val);
    // Clamp to quantization range
    f = fmaxf(QUANT_MIN, fminf(QUANT_MAX, f));
    // Scale and offset to [0, 15]
    float q = f * QUANT_SCALE + QUANT_ZERO;
    // Round and clamp to valid INT4 range
    int iq = __float2int_rn(q);
    iq = max(0, min(15, iq));
    return static_cast<uint8_t>(iq);
}

/**
 * Dequantize INT4 (0-15 range) back to FP16
 */
__device__ __forceinline__ half int4_to_fp16(uint8_t q) {
    float f = (static_cast<float>(q) - QUANT_ZERO) / QUANT_SCALE;
    return __float2half(f);
}

// ============================================================================
// Hamming(7,4) Encoding
// ============================================================================

/**
 * Encode 4 data bits to 7-bit Hamming codeword
 *
 * Input:  data4 = 0b0000dddd (4 data bits in lower nibble)
 * Output: 7-bit codeword packed in lower 7 bits
 *
 * Bit layout of codeword:
 *   bit 0 (pos 1): p0 = d0 ^ d1 ^ d3
 *   bit 1 (pos 2): p1 = d0 ^ d2 ^ d3
 *   bit 2 (pos 3): d0
 *   bit 3 (pos 4): p2 = d1 ^ d2 ^ d3
 *   bit 4 (pos 5): d1
 *   bit 5 (pos 6): d2
 *   bit 6 (pos 7): d3
 */
__device__ __forceinline__ uint8_t hamming_encode_4to7(uint8_t data4) {
    // Extract individual data bits
    uint8_t d0 = (data4 >> 0) & 1;
    uint8_t d1 = (data4 >> 1) & 1;
    uint8_t d2 = (data4 >> 2) & 1;
    uint8_t d3 = (data4 >> 3) & 1;

    // Calculate parity bits
    uint8_t p0 = d0 ^ d1 ^ d3;  // covers positions 1,3,5,7
    uint8_t p1 = d0 ^ d2 ^ d3;  // covers positions 2,3,6,7
    uint8_t p2 = d1 ^ d2 ^ d3;  // covers positions 4,5,6,7

    // Assemble codeword: p0 p1 d0 p2 d1 d2 d3
    uint8_t codeword = (p0 << 0) |  // bit 0: p0
                       (p1 << 1) |  // bit 1: p1
                       (d0 << 2) |  // bit 2: d0
                       (p2 << 3) |  // bit 3: p2
                       (d1 << 4) |  // bit 4: d1
                       (d2 << 5) |  // bit 5: d2
                       (d3 << 6);   // bit 6: d3

    return codeword;
}

// ============================================================================
// Hamming(7,4) Syndrome Calculation
// ============================================================================

/**
 * Calculate syndrome from 7-bit codeword
 *
 * Syndrome = 0: No error
 * Syndrome = 1-7: Bit position with error (1-indexed)
 *
 * Syndrome calculation:
 *   s0 = p0 ^ d0 ^ d1 ^ d3  (bit positions 1,3,5,7)
 *   s1 = p1 ^ d0 ^ d2 ^ d3  (bit positions 2,3,6,7)
 *   s2 = p2 ^ d1 ^ d2 ^ d3  (bit positions 4,5,6,7)
 */
__device__ __forceinline__ uint8_t hamming_syndrome(uint8_t codeword7) {
    // Extract all bits
    uint8_t b0 = (codeword7 >> 0) & 1;  // p0
    uint8_t b1 = (codeword7 >> 1) & 1;  // p1
    uint8_t b2 = (codeword7 >> 2) & 1;  // d0
    uint8_t b3 = (codeword7 >> 3) & 1;  // p2
    uint8_t b4 = (codeword7 >> 4) & 1;  // d1
    uint8_t b5 = (codeword7 >> 5) & 1;  // d2
    uint8_t b6 = (codeword7 >> 6) & 1;  // d3

    // Calculate syndrome bits
    uint8_t s0 = b0 ^ b2 ^ b4 ^ b6;  // positions 1,3,5,7
    uint8_t s1 = b1 ^ b2 ^ b5 ^ b6;  // positions 2,3,6,7
    uint8_t s2 = b3 ^ b4 ^ b5 ^ b6;  // positions 4,5,6,7

    // Syndrome value indicates error position (0 = no error)
    uint8_t syndrome = (s2 << 2) | (s1 << 1) | s0;
    return syndrome;
}

// ============================================================================
// Hamming(7,4) Error Correction
// ============================================================================

/**
 * Correct single-bit error based on syndrome
 *
 * Syndrome gives 1-indexed bit position of error.
 * We flip that bit to correct. Syndrome 0 means no correction needed.
 */
__device__ __forceinline__ uint8_t hamming_correct(uint8_t codeword7,
                                                    uint8_t syndrome) {
    if (syndrome == 0) {
        return codeword7;  // No error
    }
    // Syndrome is 1-indexed position, convert to 0-indexed for flip
    uint8_t error_pos = syndrome - 1;
    return codeword7 ^ (1 << error_pos);
}

// ============================================================================
// Hamming(7,4) Full Decode
// ============================================================================

/**
 * Extract 4 data bits from 7-bit codeword (no correction applied)
 * Used internally by decode functions
 */
__device__ __forceinline__ uint8_t hamming_extract_data(uint8_t cw7) {
    uint8_t d0 = (cw7 >> 2) & 1;
    uint8_t d1 = (cw7 >> 4) & 1;
    uint8_t d2 = (cw7 >> 5) & 1;
    uint8_t d3 = (cw7 >> 6) & 1;
    return (d3 << 3) | (d2 << 2) | (d1 << 1) | d0;
}

/**
 * Decode 7-bit codeword to 4 data bits with error correction
 *
 * Combines syndrome check, correction, and data extraction.
 */
__device__ __forceinline__ uint8_t hamming_decode_7to4(uint8_t codeword7) {
    // Calculate syndrome and correct if needed
    uint8_t syndrome = hamming_syndrome(codeword7);
    uint8_t corrected = hamming_correct(codeword7, syndrome);
    return hamming_extract_data(corrected);
}

// ============================================================================
// Extended Hamming(8,4) SECDED Encoding
// ============================================================================

/**
 * Encode 4 data bits to 8-bit SECDED codeword
 *
 * Extends Hamming(7,4) with an 8th global parity bit for double-error detection.
 * Bit 7 = XOR of all bits 0-6
 */
__device__ __forceinline__ uint8_t hamming_encode_4to8(uint8_t data4) {
    uint8_t cw7 = hamming_encode_4to7(data4);  // Existing 7-bit encoding
    uint8_t p = __popc(cw7) & 1;               // Global parity (popcount mod 2)
    return cw7 | (p << 7);
}

// ============================================================================
// Extended Hamming(8,4) SECDED Decoding
// ============================================================================

/**
 * Decode 8-bit SECDED codeword with error detection/correction
 *
 * Uses syndrome + global parity to distinguish error types:
 *   - syndrome=0, parity_ok: No error (ECC_OK)
 *   - syndrome!=0, parity_err: Odd errors (1-bit) -> correct (ECC_CORRECTED)
 *   - syndrome!=0, parity_ok: Even errors (2-bit) -> erasure (ECC_ERASURE)
 *   - syndrome=0, parity_err: Parity bit only flipped -> ignore (ECC_OK)
 *
 * @param codeword8 The 8-bit SECDED codeword
 * @param out_nibble Pointer to store decoded 4-bit data
 * @return EccStatus indicating decode result
 */
__device__ __forceinline__ EccStatus hamming_decode_8to4(
    uint8_t codeword8, uint8_t* out_nibble
) {
    uint8_t cw7 = codeword8 & 0x7F;            // Lower 7 bits
    uint8_t s = hamming_syndrome(cw7);         // Syndrome over bits 0-6
    uint8_t p_calc = __popc(cw7) & 1;          // Calculated parity
    uint8_t p_recv = (codeword8 >> 7) & 1;     // Received parity bit
    bool p_err = (p_calc != p_recv);           // Parity mismatch?

    if (s == 0 && !p_err) {
        // No error detected
        *out_nibble = hamming_extract_data(cw7);
        return ECC_OK;
    }
    if (s != 0 && p_err) {
        // Odd number of errors (1 bit) -> Correct it
        uint8_t corrected = hamming_correct(cw7, s);
        *out_nibble = hamming_extract_data(corrected);
        return ECC_CORRECTED;
    }
    if (s != 0 && !p_err) {
        // Even number of errors (2 bits) -> Uncorrectable erasure
        *out_nibble = 0;
        return ECC_ERASURE;
    }
    // s == 0 && p_err: Parity bit only was flipped, data is fine
    *out_nibble = hamming_extract_data(cw7);
    return ECC_OK;
}

/**
 * Combined quantize + encode for write path (Hamming 7,4)
 */
__device__ __forceinline__ uint8_t fp16_to_ecc(half val) {
    uint8_t q4 = fp16_to_int4(val);
    return hamming_encode_4to7(q4);
}

/**
 * Combined decode + dequantize for read path (Hamming 7,4)
 */
__device__ __forceinline__ half ecc_to_fp16(uint8_t codeword) {
    uint8_t q4 = hamming_decode_7to4(codeword);
    return int4_to_fp16(q4);
}

// ============================================================================
// SECDED Convenience Functions
// ============================================================================

/**
 * Combined quantize + encode for write path (SECDED 8,4)
 */
__device__ __forceinline__ uint8_t fp16_to_ecc_secded(half val) {
    uint8_t q4 = fp16_to_int4(val);
    return hamming_encode_4to8(q4);
}

/**
 * Combined decode + dequantize for read path (SECDED 8,4)
 * Returns EccStatus and writes decoded value to out_val
 */
__device__ __forceinline__ EccStatus ecc_secded_to_fp16(
    uint8_t codeword, half* out_val
) {
    uint8_t q4;
    EccStatus status = hamming_decode_8to4(codeword, &q4);
    *out_val = int4_to_fp16(q4);
    return status;
}

// ============================================================================
// N-LERP (Normalized Linear Interpolation) for Manifold Reconstruction
// ============================================================================

/**
 * Normalized Linear Interpolation (N-LERP) for scalar values
 *
 * Preserves magnitude while interpolating direction on the hypersphere.
 * Used to reconstruct erased values from temporal neighbors.
 *
 * Algorithm:
 *   1. Compute target norm as average of neighbor magnitudes
 *   2. Compute linear average of values
 *   3. Rescale result to target norm (preserving sign)
 *
 * @param prev Previous neighbor value
 * @param next Next neighbor value
 * @return Interpolated value with preserved magnitude characteristics
 */
__device__ __forceinline__ half nlerp_scalar(half prev, half next) {
    float fp = __half2float(prev);
    float fn = __half2float(next);

    // Target norm = average of neighbor magnitudes
    float target_norm = (fabsf(fp) + fabsf(fn)) * 0.5f;

    // Linear average
    float avg = (fp + fn) * 0.5f;

    // Rescale to target norm (preserve sign)
    // Handle near-zero average to avoid division issues
    if (fabsf(avg) < 1e-7f) {
        return __float2half(0.0f);
    }
    return __float2half(copysignf(target_norm, avg));
}

/**
 * Zero-order hold: use single valid neighbor
 */
__device__ __forceinline__ half zoh_scalar(half val) {
    return val;
}

}  // namespace ecc
}  // namespace vllm
