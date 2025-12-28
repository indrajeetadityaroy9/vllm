// SPDX-License-Identifier: Apache-2.0
/**
 * Fused LSQ PagedAttention Kernel for Decode Phase
 *
 * This kernel performs paged attention directly on LSQ-encoded KV cache,
 * avoiding the O(T^2) memory overhead of decoding to workspace first.
 *
 * Key features:
 * - Reads uint16 LSQ codewords from cache
 * - Decodes to FP16 on-the-fly in registers
 * - Computes attention scores and aggregates values
 * - Uses zero-masking for erasures (cheaper than N-LERP in fused kernel)
 *
 * For Prefill phase, use the shim approach (decode to workspace + flash_attn).
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

#include "lsq_math.cuh"

namespace vllm {
namespace lsq {

// Configuration
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int BLOCK_SIZE_THREADS = WARP_SIZE * NUM_WARPS;  // 128 threads

// ============================================================================
// Warp Reduction Utilities
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared_mem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < NUM_WARPS) ? shared_mem[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared_mem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < NUM_WARPS) ? shared_mem[threadIdx.x] : -FLT_MAX;
    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}

// ============================================================================
// Fused LSQ PagedAttention Kernel (Single Query per Sequence)
// ============================================================================
/**
 * This kernel handles the decode phase where each sequence has exactly 1 query.
 *
 * Grid: (num_seqs, num_heads)
 * Block: 128 threads (4 warps)
 *
 * Each block processes one (sequence, head) pair.
 * Threads cooperate to compute attention over all cached KV tokens.
 */
template <int HEAD_SIZE, int BLOCK_SIZE>
__global__ void paged_attention_lsq_v1_kernel(
    half* __restrict__ out,                   // [num_seqs, num_heads, head_size]
    const half* __restrict__ q,               // [num_seqs, num_heads, head_size]
    const uint16_t* __restrict__ k_cache,     // [num_blocks, block_size, num_kv_heads, head_size/2]
    const uint16_t* __restrict__ v_cache,     // [num_blocks, block_size, num_kv_heads, head_size/2]
    const int32_t* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int32_t* __restrict__ context_lens, // [num_seqs]
    const float scale,
    const int num_kv_heads,
    const int max_num_blocks_per_seq,
    const int q_stride,                       // num_heads * head_size
    const int kv_head_stride                  // head_size / 2
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // Map query head to KV head (for GQA/MQA)
    const int num_queries_per_kv = gridDim.y / num_kv_heads;
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    // Shared memory for reductions and caching
    __shared__ float shared_mem[NUM_WARPS];
    __shared__ float shared_logits[BLOCK_SIZE];  // Scores for one cache block
    __shared__ float shared_max;
    __shared__ float shared_sum;

    // Load query vector (each thread loads HEAD_SIZE / BLOCK_SIZE_THREADS elements)
    // For HEAD_SIZE=128, BLOCK_SIZE_THREADS=128: 1 element per thread
    float q_vec[HEAD_SIZE / BLOCK_SIZE_THREADS];
    #pragma unroll
    for (int i = 0; i < HEAD_SIZE / BLOCK_SIZE_THREADS; ++i) {
        int idx = tid + i * BLOCK_SIZE_THREADS;
        if (idx < HEAD_SIZE) {
            q_vec[i] = __half2float(q[seq_idx * q_stride + head_idx * HEAD_SIZE + idx]);
        }
    }

    // Initialize accumulators
    float global_max = -FLT_MAX;
    float global_sum = 0.0f;
    float acc_vec[HEAD_SIZE / BLOCK_SIZE_THREADS] = {0.0f};

    // Number of cache blocks for this sequence
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Iterate over cache blocks
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int physical_block_idx = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
        const int tokens_in_block = min(BLOCK_SIZE, context_len - block_idx * BLOCK_SIZE);

        // Process each token in the block
        for (int token_offset = 0; token_offset < tokens_in_block; ++token_offset) {
            // === Compute dot product Q * K ===
            float score = 0.0f;

            #pragma unroll
            for (int i = 0; i < HEAD_SIZE / BLOCK_SIZE_THREADS; ++i) {
                int dim_idx = tid + i * BLOCK_SIZE_THREADS;
                if (dim_idx < HEAD_SIZE) {
                    // Calculate pair index for this dimension
                    int pair_idx = dim_idx / 2;
                    int lane = dim_idx % 2;  // 0=anchor, 1=syndrome

                    // Load codeword from K cache
                    int64_t k_offset = (int64_t)physical_block_idx * (BLOCK_SIZE * num_kv_heads * (HEAD_SIZE / 2))
                                     + (int64_t)token_offset * (num_kv_heads * (HEAD_SIZE / 2))
                                     + (int64_t)kv_head_idx * (HEAD_SIZE / 2)
                                     + pair_idx;
                    uint16_t cw = k_cache[k_offset];

                    // Decode LSQ pair
                    half k_a, k_b;
                    EccStatus status = lsq_pair_decode(cw, &k_a, &k_b);

                    float k_val;
                    if (status == ECC_ERASURE) {
                        // Zero-mask erasures (contributes nothing to score)
                        k_val = 0.0f;
                    } else {
                        k_val = (lane == 0) ? __half2float(k_a) : __half2float(k_b);
                    }

                    score += q_vec[i] * k_val;
                }
            }

            // Reduce score across threads
            score = block_reduce_sum(score, shared_mem);

            // Scale and store in shared memory
            if (tid == 0) {
                shared_logits[token_offset] = score * scale;
            }
        }
        __syncthreads();

        // === Softmax Update ===
        // Find max in this block
        float local_max = -FLT_MAX;
        for (int i = tid; i < tokens_in_block; i += BLOCK_SIZE_THREADS) {
            local_max = fmaxf(local_max, shared_logits[i]);
        }
        local_max = block_reduce_max(local_max, shared_mem);

        if (tid == 0) {
            // Update running max
            float prev_max = global_max;
            global_max = fmaxf(global_max, local_max);

            // Rescale previous accumulator
            float scale_factor = expf(prev_max - global_max);
            global_sum *= scale_factor;
        }
        __syncthreads();

        // Broadcast updated max
        float curr_max = __shfl_sync(0xFFFFFFFF, global_max, 0);

        // Compute softmax weights for this block
        float block_sum = 0.0f;
        for (int i = tid; i < tokens_in_block; i += BLOCK_SIZE_THREADS) {
            float w = expf(shared_logits[i] - curr_max);
            shared_logits[i] = w;
            block_sum += w;
        }
        block_sum = block_reduce_sum(block_sum, shared_mem);

        if (tid == 0) {
            global_sum += block_sum;
            shared_sum = global_sum;
        }
        __syncthreads();

        // Rescale previous value accumulator
        float rescale = (tid == 0) ? expf(__shfl_sync(0xFFFFFFFF, global_max, 0) - curr_max) : 1.0f;
        rescale = __shfl_sync(0xFFFFFFFF, rescale, 0);

        // === Aggregate Values ===
        for (int token_offset = 0; token_offset < tokens_in_block; ++token_offset) {
            float weight = shared_logits[token_offset];

            #pragma unroll
            for (int i = 0; i < HEAD_SIZE / BLOCK_SIZE_THREADS; ++i) {
                int dim_idx = tid + i * BLOCK_SIZE_THREADS;
                if (dim_idx < HEAD_SIZE) {
                    int pair_idx = dim_idx / 2;
                    int lane = dim_idx % 2;

                    // Load codeword from V cache
                    int64_t v_offset = (int64_t)physical_block_idx * (BLOCK_SIZE * num_kv_heads * (HEAD_SIZE / 2))
                                     + (int64_t)token_offset * (num_kv_heads * (HEAD_SIZE / 2))
                                     + (int64_t)kv_head_idx * (HEAD_SIZE / 2)
                                     + pair_idx;
                    uint16_t cw = v_cache[v_offset];

                    // Decode LSQ pair
                    half v_a, v_b;
                    EccStatus status = lsq_pair_decode(cw, &v_a, &v_b);

                    float v_val;
                    if (status == ECC_ERASURE) {
                        v_val = 0.0f;
                    } else {
                        v_val = (lane == 0) ? __half2float(v_a) : __half2float(v_b);
                    }

                    acc_vec[i] += weight * v_val;
                }
            }
        }
        __syncthreads();
    }

    // === Normalize and Write Output ===
    float inv_sum = (global_sum > 0.0f) ? 1.0f / global_sum : 0.0f;

    #pragma unroll
    for (int i = 0; i < HEAD_SIZE / BLOCK_SIZE_THREADS; ++i) {
        int dim_idx = tid + i * BLOCK_SIZE_THREADS;
        if (dim_idx < HEAD_SIZE) {
            out[seq_idx * q_stride + head_idx * HEAD_SIZE + dim_idx] =
                __float2half(acc_vec[i] * inv_sum);
        }
    }
}

}  // namespace lsq
}  // namespace vllm

// ============================================================================
// Host Function
// ============================================================================

void paged_attention_lsq_v1(
    torch::Tensor& out,          // [num_seqs, num_heads, head_size]
    torch::Tensor& query,        // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,    // [num_blocks, block_size, num_kv_heads, head_size/2]
    torch::Tensor& value_cache,  // [num_blocks, block_size, num_kv_heads, head_size/2]
    torch::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& context_lens, // [num_seqs]
    float scale,
    int block_size
) {
    int num_seqs = query.size(0);
    int num_heads = query.size(1);
    int head_size = query.size(2);
    int num_kv_heads = key_cache.size(2);
    int max_num_blocks_per_seq = block_tables.size(1);

    int q_stride = num_heads * head_size;
    int kv_head_stride = head_size / 2;

    dim3 grid(num_seqs, num_heads);
    dim3 block(vllm::lsq::BLOCK_SIZE_THREADS);

    const at::cuda::OptionalCUDAGuard device_guard(query.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch based on head_size and block_size
    if (head_size == 128 && block_size == 16) {
        vllm::lsq::paged_attention_lsq_v1_kernel<128, 16><<<grid, block, 0, stream>>>(
            reinterpret_cast<half*>(out.data_ptr()),
            reinterpret_cast<const half*>(query.data_ptr()),
            reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),
            reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),
            block_tables.data_ptr<int32_t>(),
            context_lens.data_ptr<int32_t>(),
            scale,
            num_kv_heads,
            max_num_blocks_per_seq,
            q_stride,
            kv_head_stride
        );
    } else if (head_size == 128 && block_size == 32) {
        vllm::lsq::paged_attention_lsq_v1_kernel<128, 32><<<grid, block, 0, stream>>>(
            reinterpret_cast<half*>(out.data_ptr()),
            reinterpret_cast<const half*>(query.data_ptr()),
            reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),
            reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),
            block_tables.data_ptr<int32_t>(),
            context_lens.data_ptr<int32_t>(),
            scale,
            num_kv_heads,
            max_num_blocks_per_seq,
            q_stride,
            kv_head_stride
        );
    } else if (head_size == 64 && block_size == 16) {
        vllm::lsq::paged_attention_lsq_v1_kernel<64, 16><<<grid, block, 0, stream>>>(
            reinterpret_cast<half*>(out.data_ptr()),
            reinterpret_cast<const half*>(query.data_ptr()),
            reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),
            reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),
            block_tables.data_ptr<int32_t>(),
            context_lens.data_ptr<int32_t>(),
            scale,
            num_kv_heads,
            max_num_blocks_per_seq,
            q_stride,
            kv_head_stride
        );
    } else {
        TORCH_CHECK(false, "Unsupported head_size/block_size combination for LSQ PagedAttention: ",
                    "head_size=", head_size, ", block_size=", block_size);
    }
}
