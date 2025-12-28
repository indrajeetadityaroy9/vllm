// SPDX-License-Identifier: Apache-2.0
/**
 * LSQ (Lattice Syndrome Quantization) Math Primitives for KV Cache Protection
 *
 * Implements LSQ + SECDED(16,11) encoding for paired adjacent dimensions.
 *
 * LSQ pairs adjacent head_size dimensions (elem_2i, elem_2i+1):
 * - Token A (Anchor): 6-bit absolute quantization [-40, 40]
 * - Token B (Syndrome): 5-bit modular quantization relative to reconstructed A
 *
 * Total: 11 data bits -> SECDED(16,11) = Hamming(15,11) + global parity
 *
 * CRITICAL: Encoder must use RECONSTRUCTED anchor as predictor, not raw value!
 * This ensures decoder can correctly unwrap syndrome using quantized anchor.
 */

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace vllm {
namespace lsq {

// ============================================================================
// ECC Status Enum (shared with ecc_math.cuh)
// ============================================================================
enum EccStatus {
    ECC_OK = 0,        // No error detected
    ECC_CORRECTED = 1, // Single-bit error corrected
    ECC_ERASURE = 2    // Double-bit error detected (cannot correct)
};

// ============================================================================
// LSQ Quantization Constants
// ============================================================================

// Anchor (A): 6-bit absolute quantization
// Range: [-40.0, 40.0] -> [0, 62] (63 levels, code 63 reserved)
constexpr float ANCHOR_RANGE_MIN = -40.0f;
constexpr float ANCHOR_RANGE_MAX = 40.0f;
constexpr int ANCHOR_LEVELS = 63;  // 6 bits -> 0..62
constexpr float ANCHOR_STEP = (ANCHOR_RANGE_MAX - ANCHOR_RANGE_MIN) / (ANCHOR_LEVELS - 1);
// ANCHOR_STEP = 80.0 / 62 = 1.2903...

// Syndrome (B): 5-bit modular quantization
// Lattice Interval = 2× Anchor Step (ensures correct unwrapping)
constexpr float LATTICE_INTERVAL = 2.0f * ANCHOR_STEP;  // ~2.58
constexpr int SYNDROME_LEVELS = 31;  // 5 bits -> 0..30
constexpr float SYNDROME_STEP = LATTICE_INTERVAL / SYNDROME_LEVELS;

// ============================================================================
// Anchor Encode/Decode (6-bit absolute quantization)
// ============================================================================

/**
 * Quantize FP16 to 6-bit anchor (absolute quantization)
 * Range: [-40, 40] -> [0, 62]
 */
__device__ __forceinline__ uint8_t anchor_encode(half val) {
    float f = __half2float(val);
    // Clamp to quantization range
    f = fmaxf(ANCHOR_RANGE_MIN, fminf(ANCHOR_RANGE_MAX, f));
    // Scale to [0, 62]
    float q = (f - ANCHOR_RANGE_MIN) / ANCHOR_STEP;
    int iq = __float2int_rn(q);
    return static_cast<uint8_t>(max(0, min(ANCHOR_LEVELS - 1, iq)));
}

/**
 * Dequantize 6-bit anchor back to FP16
 */
__device__ __forceinline__ half anchor_decode(uint8_t a) {
    float f = ANCHOR_RANGE_MIN + static_cast<float>(a) * ANCHOR_STEP;
    return __float2half(f);
}

// ============================================================================
// Syndrome Encode/Decode (5-bit modular quantization)
// ============================================================================

/**
 * Encode syndrome using predictor value
 *
 * Computes residual = actual - predictor, then wraps to [-L/2, L/2]
 * and quantizes to 5 bits (0..30)
 *
 * @param actual   The actual value to encode
 * @param predictor The reconstructed anchor value (MUST be quantized!)
 * @return 5-bit syndrome code
 */
__device__ __forceinline__ uint8_t syndrome_encode(half actual, half predictor) {
    float a = __half2float(actual);
    float p = __half2float(predictor);

    // Compute residual
    float residual = a - p;

    // Wrap to [-LATTICE_INTERVAL/2, LATTICE_INTERVAL/2]
    // Use fmodf with offset to handle negative values correctly
    float half_interval = LATTICE_INTERVAL / 2.0f;
    float wrapped = fmodf(residual + half_interval + LATTICE_INTERVAL * 1000.0f, LATTICE_INTERVAL);
    wrapped -= half_interval;

    // Quantize wrapped residual to [0, 30]
    // Map [-L/2, L/2] to [0, 30]
    float q = (wrapped + half_interval) / SYNDROME_STEP;
    int iq = __float2int_rn(q);
    return static_cast<uint8_t>(max(0, min(SYNDROME_LEVELS - 1, iq)));
}

/**
 * Decode syndrome using predictor to unwrap
 *
 * @param syn      5-bit syndrome code
 * @param predictor The reconstructed anchor value
 * @return Decoded FP16 value
 */
__device__ __forceinline__ half syndrome_decode(uint8_t syn, half predictor) {
    float p = __half2float(predictor);

    // Dequantize syndrome to residual in [-L/2, L/2]
    float half_interval = LATTICE_INTERVAL / 2.0f;
    float residual = (static_cast<float>(syn) * SYNDROME_STEP) - half_interval;

    // Unwrap: actual = predictor + residual
    float actual = p + residual;

    return __float2half(actual);
}

// ============================================================================
// SECDED(16,11) Encoding
// Hamming(15,11) + 1 global parity bit
// ============================================================================

/**
 * Calculate Hamming(15,11) parity bits
 *
 * Data bits d0-d10 are placed at positions:
 * pos 3:  d0    pos 9:  d5
 * pos 5:  d1    pos 10: d6
 * pos 6:  d2    pos 11: d7
 * pos 7:  d3    pos 12: d8
 *               pos 13: d9
 *               pos 14: d10
 *
 * Parity bits at positions 1, 2, 4, 8
 */
__device__ __forceinline__ uint16_t secded_16_11_encode(uint16_t data11) {
    // Extract data bits
    uint16_t d0  = (data11 >> 0)  & 1;
    uint16_t d1  = (data11 >> 1)  & 1;
    uint16_t d2  = (data11 >> 2)  & 1;
    uint16_t d3  = (data11 >> 3)  & 1;
    uint16_t d4  = (data11 >> 4)  & 1;
    uint16_t d5  = (data11 >> 5)  & 1;
    uint16_t d6  = (data11 >> 6)  & 1;
    uint16_t d7  = (data11 >> 7)  & 1;
    uint16_t d8  = (data11 >> 8)  & 1;
    uint16_t d9  = (data11 >> 9)  & 1;
    uint16_t d10 = (data11 >> 10) & 1;

    // Calculate parity bits
    // p0 (pos 1): covers positions 1,3,5,7,9,11,13,15
    uint16_t p0 = d0 ^ d1 ^ d3 ^ d4 ^ d6 ^ d8 ^ d10;
    // p1 (pos 2): covers positions 2,3,6,7,10,11,14,15
    uint16_t p1 = d0 ^ d2 ^ d3 ^ d5 ^ d6 ^ d9 ^ d10;
    // p2 (pos 4): covers positions 4,5,6,7,12,13,14,15
    uint16_t p2 = d1 ^ d2 ^ d3 ^ d7 ^ d8 ^ d9 ^ d10;
    // p3 (pos 8): covers positions 8,9,10,11,12,13,14,15
    uint16_t p3 = d4 ^ d5 ^ d6 ^ d7 ^ d8 ^ d9 ^ d10;

    // Assemble 15-bit Hamming codeword
    // Bit positions: 0(unused), 1=p0, 2=p1, 3=d0, 4=p2, 5=d1, 6=d2, 7=d3,
    //                8=p3, 9=d4, 10=d5, 11=d6, 12=d7, 13=d8, 14=d9
    // We shift by 1 to use bits 1-15 (bit 0 is not used in standard Hamming)
    // But for storage efficiency, we pack into bits 0-14 and use bit 15 for global parity

    uint16_t cw15 = (p0 << 0)  |  // position 1 -> bit 0
                    (p1 << 1)  |  // position 2 -> bit 1
                    (d0 << 2)  |  // position 3 -> bit 2
                    (p2 << 3)  |  // position 4 -> bit 3
                    (d1 << 4)  |  // position 5 -> bit 4
                    (d2 << 5)  |  // position 6 -> bit 5
                    (d3 << 6)  |  // position 7 -> bit 6
                    (p3 << 7)  |  // position 8 -> bit 7
                    (d4 << 8)  |  // position 9 -> bit 8
                    (d5 << 9)  |  // position 10 -> bit 9
                    (d6 << 10) |  // position 11 -> bit 10
                    (d7 << 11) |  // position 12 -> bit 11
                    (d8 << 12) |  // position 13 -> bit 12
                    (d9 << 13) |  // position 14 -> bit 13
                    (d10 << 14);  // position 15 -> bit 14

    // Add global parity (bit 15) for SECDED
    uint16_t global_p = __popc(cw15) & 1;
    return cw15 | (global_p << 15);
}

// ============================================================================
// SECDED(16,11) Decoding
// ============================================================================

/**
 * Calculate Hamming(15,11) syndrome from codeword
 */
__device__ __forceinline__ uint16_t hamming_15_11_syndrome(uint16_t cw15) {
    // Extract bits
    uint16_t b0  = (cw15 >> 0)  & 1;  // p0
    uint16_t b1  = (cw15 >> 1)  & 1;  // p1
    uint16_t b2  = (cw15 >> 2)  & 1;  // d0
    uint16_t b3  = (cw15 >> 3)  & 1;  // p2
    uint16_t b4  = (cw15 >> 4)  & 1;  // d1
    uint16_t b5  = (cw15 >> 5)  & 1;  // d2
    uint16_t b6  = (cw15 >> 6)  & 1;  // d3
    uint16_t b7  = (cw15 >> 7)  & 1;  // p3
    uint16_t b8  = (cw15 >> 8)  & 1;  // d4
    uint16_t b9  = (cw15 >> 9)  & 1;  // d5
    uint16_t b10 = (cw15 >> 10) & 1;  // d6
    uint16_t b11 = (cw15 >> 11) & 1;  // d7
    uint16_t b12 = (cw15 >> 12) & 1;  // d8
    uint16_t b13 = (cw15 >> 13) & 1;  // d9
    uint16_t b14 = (cw15 >> 14) & 1;  // d10

    // Calculate syndrome bits
    // s0: XOR of positions 1,3,5,7,9,11,13,15 (bits 0,2,4,6,8,10,12,14)
    uint16_t s0 = b0 ^ b2 ^ b4 ^ b6 ^ b8 ^ b10 ^ b12 ^ b14;
    // s1: XOR of positions 2,3,6,7,10,11,14,15 (bits 1,2,5,6,9,10,13,14)
    uint16_t s1 = b1 ^ b2 ^ b5 ^ b6 ^ b9 ^ b10 ^ b13 ^ b14;
    // s2: XOR of positions 4,5,6,7,12,13,14,15 (bits 3,4,5,6,11,12,13,14)
    uint16_t s2 = b3 ^ b4 ^ b5 ^ b6 ^ b11 ^ b12 ^ b13 ^ b14;
    // s3: XOR of positions 8,9,10,11,12,13,14,15 (bits 7,8,9,10,11,12,13,14)
    uint16_t s3 = b7 ^ b8 ^ b9 ^ b10 ^ b11 ^ b12 ^ b13 ^ b14;

    return (s3 << 3) | (s2 << 2) | (s1 << 1) | s0;
}

/**
 * Extract 11 data bits from 15-bit Hamming codeword
 */
__device__ __forceinline__ uint16_t extract_data_15_11(uint16_t cw15) {
    uint16_t d0  = (cw15 >> 2)  & 1;
    uint16_t d1  = (cw15 >> 4)  & 1;
    uint16_t d2  = (cw15 >> 5)  & 1;
    uint16_t d3  = (cw15 >> 6)  & 1;
    uint16_t d4  = (cw15 >> 8)  & 1;
    uint16_t d5  = (cw15 >> 9)  & 1;
    uint16_t d6  = (cw15 >> 10) & 1;
    uint16_t d7  = (cw15 >> 11) & 1;
    uint16_t d8  = (cw15 >> 12) & 1;
    uint16_t d9  = (cw15 >> 13) & 1;
    uint16_t d10 = (cw15 >> 14) & 1;

    return (d10 << 10) | (d9 << 9) | (d8 << 8) | (d7 << 7) |
           (d6 << 6) | (d5 << 5) | (d4 << 4) | (d3 << 3) |
           (d2 << 2) | (d1 << 1) | d0;
}

/**
 * Decode 16-bit SECDED codeword
 *
 * Uses syndrome + global parity to distinguish error types:
 * - syndrome=0, parity_ok: No error (ECC_OK)
 * - syndrome!=0, parity_err: Odd errors (1-bit) -> correct (ECC_CORRECTED)
 * - syndrome!=0, parity_ok: Even errors (2-bit) -> erasure (ECC_ERASURE)
 * - syndrome=0, parity_err: Global parity bit only flipped -> ignore (ECC_OK)
 *
 * @param codeword16 The 16-bit SECDED codeword
 * @param out_data Pointer to store decoded 11-bit data
 * @return EccStatus indicating decode result
 */
__device__ __forceinline__ EccStatus secded_16_11_decode(
    uint16_t codeword16, uint16_t* out_data
) {
    uint16_t cw15 = codeword16 & 0x7FFF;  // Lower 15 bits
    uint16_t syndrome = hamming_15_11_syndrome(cw15);

    // Check global parity
    uint16_t p_calc = __popc(cw15) & 1;
    uint16_t p_recv = (codeword16 >> 15) & 1;
    bool p_err = (p_calc != p_recv);

    if (syndrome == 0 && !p_err) {
        // No error detected
        *out_data = extract_data_15_11(cw15);
        return ECC_OK;
    }
    if (syndrome != 0 && p_err) {
        // Odd number of errors (1 bit) -> Correct it
        // Syndrome gives 1-indexed position, convert to 0-indexed
        uint16_t error_pos = syndrome - 1;
        if (error_pos < 15) {
            uint16_t corrected = cw15 ^ (1 << error_pos);
            *out_data = extract_data_15_11(corrected);
            return ECC_CORRECTED;
        }
        // Error in position 16 would be global parity only
        *out_data = extract_data_15_11(cw15);
        return ECC_CORRECTED;
    }
    if (syndrome != 0 && !p_err) {
        // Even number of errors (2 bits) -> Uncorrectable erasure
        *out_data = 0;
        return ECC_ERASURE;
    }
    // syndrome == 0 && p_err: Global parity bit only was flipped, data is fine
    *out_data = extract_data_15_11(cw15);
    return ECC_OK;
}

// ============================================================================
// LSQ Pair Encode/Decode
// ============================================================================

/**
 * Encode a pair of FP16 values to LSQ SECDED(16,11) codeword
 *
 * CRITICAL: Uses Reconstruction Loop!
 * 1. Quantize anchor (A) to 6 bits
 * 2. RECONSTRUCT anchor (dequantize)
 * 3. Encode syndrome (B) relative to RECONSTRUCTED anchor
 *
 * This ensures the decoder can correctly unwrap using the quantized anchor.
 *
 * @param val_a First value (becomes 6-bit anchor)
 * @param val_b Second value (becomes 5-bit syndrome)
 * @return 16-bit SECDED codeword
 */
__device__ __forceinline__ uint16_t lsq_pair_encode(half val_a, half val_b) {
    // Step 1: Quantize anchor
    uint8_t q_a = anchor_encode(val_a);

    // Step 2: RECONSTRUCT anchor (CRITICAL!)
    half rec_a = anchor_decode(q_a);

    // Step 3: Encode syndrome using RECONSTRUCTED anchor as predictor
    uint8_t q_b = syndrome_encode(val_b, rec_a);

    // Pack: [syndrome(5 bits) | anchor(6 bits)] = 11 data bits
    uint16_t data11 = (static_cast<uint16_t>(q_b) << 6) | q_a;

    return secded_16_11_encode(data11);
}

/**
 * Decode LSQ SECDED(16,11) codeword to pair of FP16 values
 *
 * @param codeword The 16-bit SECDED codeword
 * @param out_a Pointer to store first value (anchor)
 * @param out_b Pointer to store second value (syndrome-decoded)
 * @return EccStatus indicating decode result
 */
__device__ __forceinline__ EccStatus lsq_pair_decode(
    uint16_t codeword, half* out_a, half* out_b
) {
    uint16_t data11;
    EccStatus status = secded_16_11_decode(codeword, &data11);

    if (status == ECC_ERASURE) {
        *out_a = __float2half(0.0f);
        *out_b = __float2half(0.0f);
        return ECC_ERASURE;
    }

    // Unpack: [syndrome(5 bits) | anchor(6 bits)]
    uint8_t anchor = data11 & 0x3F;           // Lower 6 bits
    uint8_t syndrome = (data11 >> 6) & 0x1F;  // Upper 5 bits

    // Decode anchor
    *out_a = anchor_decode(anchor);

    // Use anchor as predictor to decode syndrome
    *out_b = syndrome_decode(syndrome, *out_a);

    return status;
}

// ============================================================================
// N-LERP (Normalized Linear Interpolation) for Manifold Reconstruction
// (Reused from ecc_math.cuh for consistency)
// ============================================================================

/**
 * Normalized Linear Interpolation (N-LERP) for scalar values
 *
 * Preserves magnitude while interpolating direction on the hypersphere.
 * Used to reconstruct erased values from temporal neighbors.
 */
__device__ __forceinline__ half nlerp_scalar(half prev, half next) {
    float fp = __half2float(prev);
    float fn = __half2float(next);

    // Target norm = average of neighbor magnitudes
    float target_norm = (fabsf(fp) + fabsf(fn)) * 0.5f;

    // Linear average
    float avg = (fp + fn) * 0.5f;

    // Rescale to target norm (preserve sign)
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

}  // namespace lsq
}  // namespace vllm
