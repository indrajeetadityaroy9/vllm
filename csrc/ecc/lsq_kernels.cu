// SPDX-License-Identifier: Apache-2.0
/**
 * LSQ (Lattice Syndrome Quantization) Kernels for KV Cache Protection
 *
 * This file contains CUDA kernels for:
 * 1. LSQ Encode: FP16 pairs -> LSQ quantize -> SECDED(16,11) encode -> uint16 write
 * 2. LSQ Decode: uint16 read -> decode (with N-LERP reconstruction) -> FP16 pairs
 *
 * Pairs adjacent head_size dimensions (elem_2i, elem_2i+1) within same token.
 * Keys should be Hadamard-rotated before encoding (done in Python layer).
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include "lsq_math.cuh"

namespace vllm {
namespace lsq {

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
// Helper: Fetch Valid Neighbor Pair for N-LERP Reconstruction
// ============================================================================
/**
 * Fetch a neighbor token's pair values with boundary and validity checks.
 *
 * Returns false if:
 * - token_idx is outside sequence boundaries
 * - token is a padding token (slot < 0)
 * - neighbor's codeword is erased
 *
 * @param token_idx Token index to fetch
 * @param pair_idx Pair index within the token
 * @param cache The uint16 cache (key or value)
 * @param slot_mapping Slot mapping array
 * @param seq_start_locs Sequence boundaries
 * @param seq_idx Current sequence index
 * @param num_seqs Total number of sequences
 * @param n_pairs Number of pairs per token
 * @param block_size Cache block size
 * @param out_a Output for first value of pair
 * @param out_b Output for second value of pair
 * @return true if neighbor is valid, false otherwise
 */
__device__ bool fetch_lsq_neighbor_pair(
    int token_idx,
    int pair_idx,
    const uint16_t* __restrict__ cache,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ seq_start_locs,
    int seq_idx,
    int num_seqs,
    int n_pairs,
    int block_size,
    half* out_a,
    half* out_b
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
    int64_t cache_stride = static_cast<int64_t>(block_size) * n_pairs;
    int64_t addr = block_idx * cache_stride + block_offset * n_pairs + pair_idx;

    // Load and decode
    uint16_t cw = cache[addr];
    EccStatus status = lsq_pair_decode(cw, out_a, out_b);

    // If neighbor is also erased, we can't use it
    if (status == ECC_ERASURE) {
        return false;
    }

    return true;
}

// ============================================================================
// LSQ Encode Kernel (Write Path)
// ============================================================================
/**
 * Encode key/value tensors to LSQ SECDED(16,11) format in paged cache.
 *
 * Processes pairs of adjacent dimensions:
 * - Thread i handles elements 2i and 2i+1
 * - Input: ROTATED FP16 values (Hadamard rotation applied in Python for keys)
 * - Output: uint16 codewords in KV cache
 *
 * Cache layout: [num_blocks, block_size, num_heads, head_size/2] as uint16
 *
 * Thread Mapping (for LLaMA head_size=128):
 * - Block size: 64 threads
 * - Each thread handles 1 pair (2 elements)
 * - 64 pairs = 128 elements = 1 full head
 * - 64 uint16 = 128 bytes = perfectly coalesced access
 */
__global__ void lsq_ecc_encode_kernel(
    const half* __restrict__ key,              // [num_tokens, num_heads, head_size]
    const half* __restrict__ value,            // [num_tokens, num_heads, head_size]
    uint16_t* __restrict__ key_cache,          // [num_blocks, block_size, num_heads, head_size/2]
    uint16_t* __restrict__ value_cache,        // [num_blocks, block_size, num_heads, head_size/2]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t key_stride,
    const int64_t value_stride,
    const int num_heads,
    const int head_size,                       // Must be even
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
    const int n_pairs = num_heads * (head_size / 2);  // Number of element pairs

    // Source pointers for this token
    const half* __restrict__ key_src = key + token_idx * key_stride;
    const half* __restrict__ value_src = value + token_idx * value_stride;

    // Destination pointers in paged cache
    // Cache layout: [num_blocks, block_size, num_heads, head_size/2] as uint16
    const int64_t cache_stride = static_cast<int64_t>(block_size) * n_pairs;
    uint16_t* __restrict__ key_dst = key_cache + block_idx * cache_stride +
                                     block_offset * n_pairs;
    uint16_t* __restrict__ value_dst = value_cache + block_idx * cache_stride +
                                       block_offset * n_pairs;

    // Process pairs in parallel across threads
    for (int pair_idx = threadIdx.x; pair_idx < n_pairs; pair_idx += blockDim.x) {
        int elem_2i = pair_idx * 2;
        int elem_2i_plus_1 = elem_2i + 1;

        // Encode key pair (keys should be Hadamard-rotated)
        half k_a = key_src[elem_2i];
        half k_b = key_src[elem_2i_plus_1];
        key_dst[pair_idx] = lsq_pair_encode(k_a, k_b);

        // Encode value pair (values are NOT rotated per design)
        half v_a = value_src[elem_2i];
        half v_b = value_src[elem_2i_plus_1];
        value_dst[pair_idx] = lsq_pair_encode(v_a, v_b);
    }
}

// ============================================================================
// LSQ Gather + Decode Kernel (Read Path with N-LERP)
// ============================================================================
/**
 * Gather and decode LSQ-protected KV cache to linear FP16 workspace.
 *
 * For erasures:
 * - Uses warp-level optimization to skip N-LERP when no erasures in warp
 * - Reconstructs erased values using N-LERP from temporal neighbors (t-1, t+1)
 * - Respects sequence boundaries to prevent cross-sequence interpolation
 *
 * Output values need inverse Hadamard applied for keys (done in Python layer).
 */
__global__ void lsq_ecc_gather_decode_kernel(
    const uint16_t* __restrict__ key_cache,        // [num_blocks, block_size, num_heads, head_size/2]
    const uint16_t* __restrict__ value_cache,      // [num_blocks, block_size, num_heads, head_size/2]
    const int64_t* __restrict__ slot_mapping,      // [num_tokens]
    const int64_t* __restrict__ seq_start_locs,    // [num_seqs + 1] for boundary safety
    half* __restrict__ workspace_k,                // [num_tokens, num_heads, head_size]
    half* __restrict__ workspace_v,                // [num_tokens, num_heads, head_size]
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

    const int n_pairs = num_heads * (head_size / 2);
    const int64_t block_idx = slot / block_size;
    const int64_t block_offset = slot % block_size;
    const int64_t cache_stride = static_cast<int64_t>(block_size) * n_pairs;

    // Source pointers in paged cache
    const uint16_t* __restrict__ key_src = key_cache + block_idx * cache_stride +
                                           block_offset * n_pairs;
    const uint16_t* __restrict__ val_src = value_cache + block_idx * cache_stride +
                                           block_offset * n_pairs;

    // Destination pointers in linear workspace
    const int n_elems = num_heads * head_size;
    half* __restrict__ key_dst = workspace_k + token_idx * n_elems;
    half* __restrict__ val_dst = workspace_v + token_idx * n_elems;

    // Find sequence index for this token (for boundary safety in N-LERP)
    int seq_idx = binary_search_seq(token_idx, seq_start_locs, num_seqs);

    // Process pairs in parallel across threads
    for (int pair_idx = threadIdx.x; pair_idx < n_pairs; pair_idx += blockDim.x) {
        int elem_2i = pair_idx * 2;
        int elem_2i_plus_1 = elem_2i + 1;

        // ===== Decode Key Pair =====
        uint16_t k_cw = key_src[pair_idx];
        half k_a, k_b;
        EccStatus k_status = lsq_pair_decode(k_cw, &k_a, &k_b);

        // Warp-level optimization: skip N-LERP if no erasures in warp
        bool k_erasure = (k_status == ECC_ERASURE);
        bool warp_has_k_erasure = __any_sync(0xFFFFFFFF, k_erasure);

        if (warp_has_k_erasure && k_erasure) {
            // N-LERP reconstruction for key pair
            half prev_a, prev_b, next_a, next_b;
            bool has_prev = fetch_lsq_neighbor_pair(
                token_idx - 1, pair_idx, key_cache, slot_mapping,
                seq_start_locs, seq_idx, num_seqs, n_pairs, block_size,
                &prev_a, &prev_b
            );
            bool has_next = fetch_lsq_neighbor_pair(
                token_idx + 1, pair_idx, key_cache, slot_mapping,
                seq_start_locs, seq_idx, num_seqs, n_pairs, block_size,
                &next_a, &next_b
            );

            if (has_prev && has_next) {
                k_a = nlerp_scalar(prev_a, next_a);
                k_b = nlerp_scalar(prev_b, next_b);
            } else if (has_prev) {
                k_a = zoh_scalar(prev_a);
                k_b = zoh_scalar(prev_b);
            } else if (has_next) {
                k_a = zoh_scalar(next_a);
                k_b = zoh_scalar(next_b);
            } else {
                k_a = __float2half(0.0f);  // Total signal loss
                k_b = __float2half(0.0f);
            }
        }

        key_dst[elem_2i] = k_a;
        key_dst[elem_2i_plus_1] = k_b;

        // ===== Decode Value Pair =====
        uint16_t v_cw = val_src[pair_idx];
        half v_a, v_b;
        EccStatus v_status = lsq_pair_decode(v_cw, &v_a, &v_b);

        // Warp-level optimization for value
        bool v_erasure = (v_status == ECC_ERASURE);
        bool warp_has_v_erasure = __any_sync(0xFFFFFFFF, v_erasure);

        if (warp_has_v_erasure && v_erasure) {
            // N-LERP reconstruction for value pair
            half prev_a, prev_b, next_a, next_b;
            bool has_prev = fetch_lsq_neighbor_pair(
                token_idx - 1, pair_idx, value_cache, slot_mapping,
                seq_start_locs, seq_idx, num_seqs, n_pairs, block_size,
                &prev_a, &prev_b
            );
            bool has_next = fetch_lsq_neighbor_pair(
                token_idx + 1, pair_idx, value_cache, slot_mapping,
                seq_start_locs, seq_idx, num_seqs, n_pairs, block_size,
                &next_a, &next_b
            );

            if (has_prev && has_next) {
                v_a = nlerp_scalar(prev_a, next_a);
                v_b = nlerp_scalar(prev_b, next_b);
            } else if (has_prev) {
                v_a = zoh_scalar(prev_a);
                v_b = zoh_scalar(prev_b);
            } else if (has_next) {
                v_a = zoh_scalar(next_a);
                v_b = zoh_scalar(next_b);
            } else {
                v_a = __float2half(0.0f);
                v_b = __float2half(0.0f);
            }
        }

        val_dst[elem_2i] = v_a;
        val_dst[elem_2i_plus_1] = v_b;
    }
}

}  // namespace lsq
}  // namespace vllm

// ============================================================================
// Python Bindings (Host Functions)
// ============================================================================

void lsq_ecc_encode(
    torch::Tensor& key,           // [num_tokens, num_heads, head_size]
    torch::Tensor& value,         // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,     // [num_blocks, block_size, num_heads, head_size/2] as uint16
    torch::Tensor& value_cache,   // [num_blocks, block_size, num_heads, head_size/2] as uint16
    torch::Tensor& slot_mapping   // [num_tokens]
) {
    TORCH_CHECK(key.dtype() == torch::kFloat16 || key.dtype() == torch::kBFloat16,
                "key must be float16 or bfloat16");
    TORCH_CHECK(key_cache.dtype() == torch::kInt16 || key_cache.dtype() == torch::kUInt8,
                "key_cache must be int16 (for uint16) or uint8");
    TORCH_CHECK(value_cache.dtype() == torch::kInt16 || value_cache.dtype() == torch::kUInt8,
                "value_cache must be int16 (for uint16) or uint8");

    int num_tokens = slot_mapping.size(0);
    int num_heads = key.size(1);
    int head_size = key.size(2);
    int block_size = key_cache.size(1);

    TORCH_CHECK(head_size % 2 == 0, "head_size must be even for LSQ pair encoding");

    int64_t key_stride = key.stride(0);
    int64_t value_stride = value.stride(0);

    // Use 64 threads for LLaMA (128 elements / 2 = 64 pairs)
    // Adjust for other head sizes
    int n_pairs = num_heads * (head_size / 2);
    dim3 grid(num_tokens);
    dim3 block(std::min(n_pairs, 256));

    const at::cuda::OptionalCUDAGuard device_guard(key.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    vllm::lsq::lsq_ecc_encode_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(key.data_ptr()),
        reinterpret_cast<half*>(value.data_ptr()),
        reinterpret_cast<uint16_t*>(key_cache.data_ptr()),
        reinterpret_cast<uint16_t*>(value_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size
    );
}

void lsq_ecc_gather_decode(
    torch::Tensor& key_cache,       // [num_blocks, block_size, num_heads, head_size/2] as uint16
    torch::Tensor& value_cache,     // [num_blocks, block_size, num_heads, head_size/2] as uint16
    torch::Tensor& slot_mapping,    // [num_tokens]
    torch::Tensor& seq_start_locs,  // [num_seqs + 1] for boundary safety
    torch::Tensor& workspace_k,     // [num_tokens, num_heads, head_size]
    torch::Tensor& workspace_v,     // [num_tokens, num_heads, head_size]
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_size,
    int64_t block_size,
    int64_t num_seqs
) {
    TORCH_CHECK(key_cache.dtype() == torch::kInt16 || key_cache.dtype() == torch::kUInt8,
                "key_cache must be int16 (for uint16) or uint8");
    TORCH_CHECK(value_cache.dtype() == torch::kInt16 || value_cache.dtype() == torch::kUInt8,
                "value_cache must be int16 (for uint16) or uint8");
    TORCH_CHECK(workspace_k.dtype() == torch::kFloat16,
                "workspace_k must be float16");
    TORCH_CHECK(workspace_v.dtype() == torch::kFloat16,
                "workspace_v must be float16");
    TORCH_CHECK(head_size % 2 == 0, "head_size must be even for LSQ pair decoding");

    int n_pairs = num_heads * (head_size / 2);
    dim3 grid(num_tokens);
    dim3 block(std::min(static_cast<int64_t>(n_pairs), static_cast<int64_t>(256)));

    const at::cuda::OptionalCUDAGuard device_guard(key_cache.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    vllm::lsq::lsq_ecc_gather_decode_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<uint16_t*>(key_cache.data_ptr()),
        reinterpret_cast<uint16_t*>(value_cache.data_ptr()),
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
