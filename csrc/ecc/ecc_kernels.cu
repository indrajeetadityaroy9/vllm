// SPDX-License-Identifier: Apache-2.0
/**
 * ECC Kernels for KV Cache Protection
 *
 * This file contains CUDA kernels for:
 * 1. ECC Encode: FP16 -> INT4 quantize -> Hamming encode -> uint8 write
 * 2. ECC Decode: uint8 read -> decode (with correction/reconstruction) -> FP16
 *
 * Supports two algorithms via template dispatch:
 * - Hamming(7,4): Single-bit error correction
 * - SECDED(8,4): Single-bit correction + Double-bit detection with N-LERP reconstruction
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include "ecc_math.cuh"

namespace vllm {
namespace ecc {

// ============================================================================
// Helper: Binary Search for Sequence Index
// ============================================================================
/**
 * Find which sequence a token belongs to using binary search on seq_start_locs.
 *
 * @param token_idx The global token index
 * @param seq_start_locs Cumulative sequence start locations [num_seqs + 1]
 * @param num_seqs Number of sequences
 * @return Sequence index (0 to num_seqs-1)
 */
__device__ __forceinline__ int binary_search_seq(
    int token_idx,
    const int64_t* __restrict__ seq_start_locs,
    int num_seqs
) {
    int lo = 0;
    int hi = num_seqs;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (seq_start_locs[mid] <= token_idx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// ============================================================================
// Helper: Fetch Valid Neighbor for N-LERP Reconstruction
// ============================================================================
/**
 * Fetch a neighbor token's value with boundary and validity checks.
 *
 * Returns false if:
 * - token_idx is outside sequence boundaries
 * - token is a padding token (slot < 0)
 * - neighbor's codeword is also erased (for SECDED)
 *
 * @tparam ALGO Algorithm selector (ECC_ALGO_HAMMING74 or ECC_ALGO_SECDED)
 */
template<int ALGO>
__device__ bool fetch_valid_neighbor(
    int token_idx,
    int elem_idx,
    const uint8_t* __restrict__ cache,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ seq_start_locs,
    int seq_idx,
    int num_seqs,
    int n_elems,
    int block_size,
    half* out_val
) {
    // Boundary check within sequence
    int64_t start = seq_start_locs[seq_idx];
    int64_t end = seq_start_locs[seq_idx + 1];
    if (token_idx < start || token_idx >= end) {
        return false;
    }

    // Slot check
    int64_t slot = slot_mapping[token_idx];
    if (slot < 0) {
        return false;  // Padding token
    }

    // Compute cache address
    int64_t block_idx = slot / block_size;
    int64_t block_offset = slot % block_size;
    int64_t cache_stride = static_cast<int64_t>(block_size) * n_elems;
    int64_t addr = block_idx * cache_stride + block_offset * n_elems + elem_idx;

    // Load and decode
    uint8_t cw = cache[addr];

    if constexpr (ALGO == ECC_ALGO_HAMMING74) {
        // Hamming(7,4): Always returns a value (may be mis-corrected)
        uint8_t nibble = hamming_decode_7to4(cw);
        *out_val = int4_to_fp16(nibble);
        return true;
    } else {
        // SECDED: Check if neighbor is also erased
        uint8_t nibble;
        EccStatus status = hamming_decode_8to4(cw, &nibble);
        if (status == ECC_ERASURE) {
            return false;  // Neighbor is corrupted, can't use it
        }
        *out_val = int4_to_fp16(nibble);
        return true;
    }
}

// ============================================================================
// Templated ECC Encode Kernel (Write Path)
// ============================================================================
/**
 * Encode key/value tensors to ECC-protected format in paged cache.
 *
 * @tparam ALGO Algorithm: ECC_ALGO_HAMMING74 (7-bit) or ECC_ALGO_SECDED (8-bit)
 */
template<int ALGO>
__global__ void ecc_encode_kernel(
    const half* __restrict__ key,              // [num_tokens, num_heads, head_size]
    const half* __restrict__ value,            // [num_tokens, num_heads, head_size]
    uint8_t* __restrict__ key_cache,           // [num_blocks, block_size, num_heads, head_size]
    uint8_t* __restrict__ value_cache,         // [num_blocks, block_size, num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t key_stride,
    const int64_t value_stride,
    const int num_heads,
    const int head_size,
    const int block_size
) {
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];

    // Padding token - skip
    if (slot_idx < 0) {
        return;
    }

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;
    const int n_elems = num_heads * head_size;

    // Source pointers for this token
    const half* __restrict__ key_src = key + token_idx * key_stride;
    const half* __restrict__ value_src = value + token_idx * value_stride;

    // Destination pointers in paged cache
    // Cache layout: [num_blocks, block_size, num_heads, head_size]
    const int64_t cache_stride = static_cast<int64_t>(block_size) * n_elems;
    uint8_t* __restrict__ key_dst = key_cache + block_idx * cache_stride +
                                    block_offset * n_elems;
    uint8_t* __restrict__ value_dst = value_cache + block_idx * cache_stride +
                                      block_offset * n_elems;

    // Process elements in parallel across threads
    for (int i = threadIdx.x; i < n_elems; i += blockDim.x) {
        // Encode key
        half k_val = key_src[i];
        uint8_t k_ecc;
        if constexpr (ALGO == ECC_ALGO_HAMMING74) {
            k_ecc = fp16_to_ecc(k_val);           // 7-bit Hamming
        } else {
            k_ecc = fp16_to_ecc_secded(k_val);    // 8-bit SECDED
        }
        key_dst[i] = k_ecc;

        // Encode value
        half v_val = value_src[i];
        uint8_t v_ecc;
        if constexpr (ALGO == ECC_ALGO_HAMMING74) {
            v_ecc = fp16_to_ecc(v_val);
        } else {
            v_ecc = fp16_to_ecc_secded(v_val);
        }
        value_dst[i] = v_ecc;
    }
}

// ============================================================================
// Templated ECC Gather + Decode Kernel (Read Path with N-LERP)
// ============================================================================
/**
 * Gather and decode ECC-protected KV cache to linear FP16 workspace.
 *
 * For SECDED algorithm:
 * - Detects 2-bit errors as erasures
 * - Uses warp-level optimization to skip N-LERP when no erasures in warp
 * - Reconstructs erased values using N-LERP from temporal neighbors
 * - Respects sequence boundaries to prevent cross-sequence interpolation
 *
 * @tparam ALGO Algorithm: ECC_ALGO_HAMMING74 or ECC_ALGO_SECDED
 */
template<int ALGO>
__global__ void ecc_gather_decode_kernel(
    const uint8_t* __restrict__ key_cache,        // [num_blocks, block_size, num_heads, head_size]
    const uint8_t* __restrict__ value_cache,      // [num_blocks, block_size, num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,     // [num_tokens]
    const int64_t* __restrict__ seq_start_locs,   // [num_seqs + 1] for boundary safety
    half* __restrict__ workspace_k,               // [num_tokens, num_heads, head_size]
    half* __restrict__ workspace_v,               // [num_tokens, num_heads, head_size]
    const int num_tokens,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int num_seqs
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;  // Padding token

    const int64_t block_idx = slot / block_size;
    const int64_t block_offset = slot % block_size;
    const int n_elems = num_heads * head_size;
    const int64_t cache_stride = static_cast<int64_t>(block_size) * n_elems;

    // Source pointers in paged cache
    const uint8_t* __restrict__ key_src = key_cache + block_idx * cache_stride +
                                          block_offset * n_elems;
    const uint8_t* __restrict__ val_src = value_cache + block_idx * cache_stride +
                                          block_offset * n_elems;

    // Destination pointers in linear workspace
    half* __restrict__ key_dst = workspace_k + token_idx * n_elems;
    half* __restrict__ val_dst = workspace_v + token_idx * n_elems;

    // Find sequence index for this token (for boundary safety in N-LERP)
    int seq_idx = 0;
    if constexpr (ALGO == ECC_ALGO_SECDED) {
        seq_idx = binary_search_seq(token_idx, seq_start_locs, num_seqs);
    }

    // Process elements in parallel across threads
    for (int i = threadIdx.x; i < n_elems; i += blockDim.x) {
        // ===== Decode Key =====
        uint8_t k_cw = key_src[i];
        half k_val;
        bool k_erasure = false;

        if constexpr (ALGO == ECC_ALGO_HAMMING74) {
            // Hamming(7,4): Simple decode (may mis-correct bursts)
            k_val = ecc_to_fp16(k_cw);
        } else {
            // SECDED: Detect erasures
            EccStatus k_status = ecc_secded_to_fp16(k_cw, &k_val);
            if (k_status == ECC_ERASURE) {
                k_erasure = true;
            }
        }

        // ===== Warp Optimization: Skip N-LERP if no erasures in warp =====
        if constexpr (ALGO == ECC_ALGO_SECDED) {
            bool warp_has_k_erasure = __any_sync(0xFFFFFFFF, k_erasure);

            if (warp_has_k_erasure && k_erasure) {
                // N-LERP Reconstruction for key
                half prev_val, next_val;
                bool has_prev = fetch_valid_neighbor<ALGO>(
                    token_idx - 1, i, key_cache, slot_mapping,
                    seq_start_locs, seq_idx, num_seqs, n_elems, block_size, &prev_val
                );
                bool has_next = fetch_valid_neighbor<ALGO>(
                    token_idx + 1, i, key_cache, slot_mapping,
                    seq_start_locs, seq_idx, num_seqs, n_elems, block_size, &next_val
                );

                if (has_prev && has_next) {
                    k_val = nlerp_scalar(prev_val, next_val);
                } else if (has_prev) {
                    k_val = zoh_scalar(prev_val);  // Zero-order hold
                } else if (has_next) {
                    k_val = zoh_scalar(next_val);
                } else {
                    k_val = __float2half(0.0f);    // Total signal loss
                }
            }
        }

        key_dst[i] = k_val;

        // ===== Decode Value =====
        uint8_t v_cw = val_src[i];
        half v_val;
        bool v_erasure = false;

        if constexpr (ALGO == ECC_ALGO_HAMMING74) {
            v_val = ecc_to_fp16(v_cw);
        } else {
            EccStatus v_status = ecc_secded_to_fp16(v_cw, &v_val);
            if (v_status == ECC_ERASURE) {
                v_erasure = true;
            }
        }

        // ===== Warp Optimization for Value =====
        if constexpr (ALGO == ECC_ALGO_SECDED) {
            bool warp_has_v_erasure = __any_sync(0xFFFFFFFF, v_erasure);

            if (warp_has_v_erasure && v_erasure) {
                // N-LERP Reconstruction for value
                half prev_val, next_val;
                bool has_prev = fetch_valid_neighbor<ALGO>(
                    token_idx - 1, i, value_cache, slot_mapping,
                    seq_start_locs, seq_idx, num_seqs, n_elems, block_size, &prev_val
                );
                bool has_next = fetch_valid_neighbor<ALGO>(
                    token_idx + 1, i, value_cache, slot_mapping,
                    seq_start_locs, seq_idx, num_seqs, n_elems, block_size, &next_val
                );

                if (has_prev && has_next) {
                    v_val = nlerp_scalar(prev_val, next_val);
                } else if (has_prev) {
                    v_val = zoh_scalar(prev_val);
                } else if (has_next) {
                    v_val = zoh_scalar(next_val);
                } else {
                    v_val = __float2half(0.0f);
                }
            }
        }

        val_dst[i] = v_val;
    }
}

// Explicit template instantiations
template __global__ void ecc_encode_kernel<ECC_ALGO_HAMMING74>(
    const half*, const half*, uint8_t*, uint8_t*,
    const int64_t*, int64_t, int64_t, int, int, int);
template __global__ void ecc_encode_kernel<ECC_ALGO_SECDED>(
    const half*, const half*, uint8_t*, uint8_t*,
    const int64_t*, int64_t, int64_t, int, int, int);

template __global__ void ecc_gather_decode_kernel<ECC_ALGO_HAMMING74>(
    const uint8_t*, const uint8_t*, const int64_t*, const int64_t*,
    half*, half*, int, int, int, int, int);
template __global__ void ecc_gather_decode_kernel<ECC_ALGO_SECDED>(
    const uint8_t*, const uint8_t*, const int64_t*, const int64_t*,
    half*, half*, int, int, int, int, int);

}  // namespace ecc
}  // namespace vllm

// ============================================================================
// Python Bindings (Host Functions)
// ============================================================================

void ecc_encode(
    torch::Tensor& key,           // [num_tokens, num_heads, head_size]
    torch::Tensor& value,         // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,     // [num_blocks, block_size, num_heads, head_size] as uint8
    torch::Tensor& value_cache,   // [num_blocks, block_size, num_heads, head_size] as uint8
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t algorithm             // 0 = Hamming(7,4), 1 = SECDED(8,4)
) {
    TORCH_CHECK(key.dtype() == torch::kFloat16 || key.dtype() == torch::kBFloat16,
                "key must be float16 or bfloat16");
    TORCH_CHECK(key_cache.dtype() == torch::kUInt8,
                "key_cache must be uint8 for ECC encoding");
    TORCH_CHECK(value_cache.dtype() == torch::kUInt8,
                "value_cache must be uint8 for ECC encoding");
    TORCH_CHECK(algorithm == 0 || algorithm == 1,
                "algorithm must be 0 (Hamming74) or 1 (SECDED)");

    int num_tokens = slot_mapping.size(0);
    int num_heads = key.size(1);
    int head_size = key.size(2);
    int block_size = key_cache.size(1);

    int64_t key_stride = key.stride(0);
    int64_t value_stride = value.stride(0);

    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_size, 512));

    const at::cuda::OptionalCUDAGuard device_guard(key.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (algorithm == 0) {
        vllm::ecc::ecc_encode_kernel<ECC_ALGO_HAMMING74><<<grid, block, 0, stream>>>(
            reinterpret_cast<half*>(key.data_ptr()),
            reinterpret_cast<half*>(value.data_ptr()),
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            slot_mapping.data_ptr<int64_t>(),
            key_stride,
            value_stride,
            num_heads,
            head_size,
            block_size
        );
    } else {
        vllm::ecc::ecc_encode_kernel<ECC_ALGO_SECDED><<<grid, block, 0, stream>>>(
            reinterpret_cast<half*>(key.data_ptr()),
            reinterpret_cast<half*>(value.data_ptr()),
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            slot_mapping.data_ptr<int64_t>(),
            key_stride,
            value_stride,
            num_heads,
            head_size,
            block_size
        );
    }
}

void ecc_gather_decode(
    torch::Tensor& key_cache,       // [num_blocks, block_size, num_heads, head_size] as uint8
    torch::Tensor& value_cache,     // [num_blocks, block_size, num_heads, head_size] as uint8
    torch::Tensor& slot_mapping,    // [num_tokens]
    torch::Tensor& seq_start_locs,  // [num_seqs + 1] for boundary safety
    torch::Tensor& workspace_k,     // [num_tokens, num_heads, head_size]
    torch::Tensor& workspace_v,     // [num_tokens, num_heads, head_size]
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_size,
    int64_t block_size,
    int64_t num_seqs,
    int64_t algorithm               // 0 = Hamming(7,4), 1 = SECDED(8,4)
) {
    TORCH_CHECK(key_cache.dtype() == torch::kUInt8,
                "key_cache must be uint8 for ECC decoding");
    TORCH_CHECK(value_cache.dtype() == torch::kUInt8,
                "value_cache must be uint8 for ECC decoding");
    TORCH_CHECK(workspace_k.dtype() == torch::kFloat16,
                "workspace_k must be float16");
    TORCH_CHECK(workspace_v.dtype() == torch::kFloat16,
                "workspace_v must be float16");
    TORCH_CHECK(algorithm == 0 || algorithm == 1,
                "algorithm must be 0 (Hamming74) or 1 (SECDED)");

    dim3 grid(num_tokens);
    dim3 block(std::min(static_cast<int64_t>(num_heads * head_size), static_cast<int64_t>(512)));

    const at::cuda::OptionalCUDAGuard device_guard(key_cache.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (algorithm == 0) {
        vllm::ecc::ecc_gather_decode_kernel<ECC_ALGO_HAMMING74><<<grid, block, 0, stream>>>(
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            slot_mapping.data_ptr<int64_t>(),
            seq_start_locs.data_ptr<int64_t>(),
            reinterpret_cast<half*>(workspace_k.data_ptr()),
            reinterpret_cast<half*>(workspace_v.data_ptr()),
            static_cast<int>(num_tokens),
            static_cast<int>(num_heads),
            static_cast<int>(head_size),
            static_cast<int>(block_size),
            static_cast<int>(num_seqs)
        );
    } else {
        vllm::ecc::ecc_gather_decode_kernel<ECC_ALGO_SECDED><<<grid, block, 0, stream>>>(
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            slot_mapping.data_ptr<int64_t>(),
            seq_start_locs.data_ptr<int64_t>(),
            reinterpret_cast<half*>(workspace_k.data_ptr()),
            reinterpret_cast<half*>(workspace_v.data_ptr()),
            static_cast<int>(num_tokens),
            static_cast<int>(num_heads),
            static_cast<int>(head_size),
            static_cast<int>(block_size),
            static_cast<int>(num_seqs)
        );
    }
}
