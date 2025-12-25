// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#ifdef VLLM_FAULT_INJECT

#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>
#include <algorithm>  // for std::min

namespace vllm {
namespace fault_injection {

// FaultSpec struct matching Python FaultInjectionConfig
// Stored in CUDA Constant Memory to avoid kernel signature changes
struct FaultSpec {
  bool enabled;
  uint8_t site;       // 0=KV_WRITE, 1=KV_READ, 2=BOTH
  uint8_t subsite;    // 0=CODEWORD, 1=VALUE
  uint8_t model;      // 0=random, 1=burst, 2=msb_biased, 3=page_local
  float rate;         // Probability block is corrupted (0.0-1.0)
  uint8_t flip_count; // Number of bit flips per corrupted block (max 8)
  uint8_t burst_len;  // Consecutive bits to flip in burst model
  uint8_t msb_policy; // 0=BYTE_TOPBITS, 1=FP16_EXPONENT, 2=INT4_NIBBLE
  uint8_t msb_mask;   // Bitmask for BYTE_TOPBITS policy
  int32_t page_scope; // Physical block number to target (-1 = all)
  uint64_t seed;      // Random seed for deterministic replay
};

// Site enumeration values
constexpr uint8_t SITE_KV_WRITE = 0;
constexpr uint8_t SITE_KV_READ = 1;
constexpr uint8_t SITE_BOTH = 2;

// Subsite enumeration values
constexpr uint8_t SUBSITE_CODEWORD = 0;
constexpr uint8_t SUBSITE_VALUE = 1;

// Fault model enumeration values
constexpr uint8_t MODEL_RANDOM = 0;
constexpr uint8_t MODEL_BURST = 1;
constexpr uint8_t MODEL_MSB_BIASED = 2;
constexpr uint8_t MODEL_PAGE_LOCAL = 3;

// MSB policy enumeration values
constexpr uint8_t MSB_BYTE_TOPBITS = 0;
constexpr uint8_t MSB_FP16_EXPONENT = 1;
constexpr uint8_t MSB_INT4_NIBBLE = 2;

// ============================================================
// Host-callable function to set fault injection config
// Defined in cache_kernels.cu (where KV cache kernels live)
// This ensures cudaMemcpyToSymbol updates the symbol used by kernels
// ============================================================
void set_cache_fault_spec(const FaultSpec& spec);

inline void set_fault_spec(const FaultSpec& spec) {
  set_cache_fault_spec(spec);
}

// ============================================================
// Device code (only compiled by nvcc)
// ============================================================
#ifdef __CUDACC__

// Constant memory declaration (no kernel signature changes needed)
// The actual definition is in cache_kernels.cu (where KV cache kernels live)
// Each .cu file that includes this header gets its own copy of c_fault_spec
// Only cache_kernels.cu's copy is updated by set_cache_fault_spec()
//
// NOTE: We define c_fault_spec here without 'extern' because CUDA constant
// memory is per-compilation-unit. The set_cache_fault_spec() function in
// cache_kernels.cu updates that file's c_fault_spec using cudaMemcpyToSymbol.
__constant__ FaultSpec c_fault_spec;

// Philox-based RNG: produces deterministic pseudorandom values
// One call per block for efficiency
__device__ __forceinline__ uint32_t philox_hash(uint64_t seed,
                                                 uint64_t counter) {
  // Philox 4x32 round function (simplified single-stream version)
  // Based on Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3"
  uint64_t key = seed;
  uint64_t state = counter;
  // Standard Philox-4x32 multiplier (proper 64-bit constant)
  constexpr uint64_t PHILOX_M = 0xD2E7470EE14C6C93ULL;
  constexpr uint64_t PHILOX_W = 0x9E3779B97F4A7C15ULL;
#pragma unroll
  for (int i = 0; i < 10; i++) {
    state = state * PHILOX_M + key;
    key += PHILOX_W;
  }
  return static_cast<uint32_t>(state ^ (state >> 32));
}

// Decide if this block should be corrupted (one RNG call per block)
// Returns true with probability spec.rate for eligible blocks
__device__ __forceinline__ bool
should_corrupt_block(const FaultSpec& spec, int physical_block_number,
                     int layer_idx) {
  if (!spec.enabled) return false;

  // page_local model: only corrupt specified block
  if (spec.page_scope >= 0 && spec.page_scope != physical_block_number)
    return false;

  // Combine layer and block into counter for unique RNG per location
  uint64_t counter =
      (static_cast<uint64_t>(layer_idx) << 32) |
      static_cast<uint64_t>(static_cast<uint32_t>(physical_block_number));
  uint32_t rng = philox_hash(spec.seed, counter);
  float threshold =
      static_cast<float>(rng) / static_cast<float>(UINT32_MAX);
  return threshold < spec.rate;
}

// Generate bit positions to flip (bounded loop, max flip_count)
// out_positions must have space for at least 8 elements
__device__ __forceinline__ void
compute_flip_positions(const FaultSpec& spec, int physical_block_number,
                       int layer_idx, size_t nbytes, int* out_positions,
                       int* out_count) {
  *out_count = (spec.flip_count < 8) ? static_cast<int>(spec.flip_count) : 8; // Cap at 8 flips
  // Note: Cannot use #pragma unroll with variable count
  for (int i = 0; i < *out_count; i++) {
    uint64_t counter =
        (static_cast<uint64_t>(layer_idx) << 32) |
        static_cast<uint64_t>(static_cast<uint32_t>(physical_block_number));
    counter = counter * 31 + i; // Different position per flip
    uint32_t rng = philox_hash(spec.seed + i, counter);
    out_positions[i] = rng % (nbytes * 8); // Bit position within buffer
  }
}

// ============================================================
// VERSION A: For GLOBAL MEMORY buffers only (staging buffers, shared memory)
// NOT for local variables (tgt_key, tgt_value, k_vec, etc.)
// Use inject_fault_register for register values instead
// ============================================================
template <typename T>
__device__ void inject_fault_memory(T* data, size_t nbytes,
                                    int physical_block_number, int layer_idx) {
  if (!should_corrupt_block(c_fault_spec, physical_block_number, layer_idx))
    return;

  int positions[8];
  int count;
  compute_flip_positions(c_fault_spec, physical_block_number, layer_idx, nbytes,
                         positions, &count);

  uint8_t* bytes = reinterpret_cast<uint8_t*>(data);
  for (int i = 0; i < count; i++) {
    int bit_pos = positions[i];
    int byte_idx = bit_pos / 8;
    int bit_idx = bit_pos % 8;

    if (byte_idx >= static_cast<int>(nbytes)) continue;

    if (c_fault_spec.model == MODEL_MSB_BIASED) {
      // For MSB-biased, always flip a bit within the mask region
      uint8_t mask = c_fault_spec.msb_mask;
      if (mask != 0) {
        // Count set bits and select one based on random position
        int set_bit_count = __popc(static_cast<unsigned int>(mask));
        int target_idx = (positions[i] >> 3) % set_bit_count;
        int selected_bit = 0;
        for (int b = 0; b < 8 && target_idx >= 0; b++) {
          if (mask & (1 << b)) {
            if (target_idx == 0) {
              selected_bit = b;
              break;
            }
            target_idx--;
          }
        }
        bytes[byte_idx] ^= (1 << selected_bit);
      }
    } else if (c_fault_spec.model == MODEL_BURST && i == 0) {
      // Burst: flip consecutive bits starting at bit_pos
      for (int b = 0;
           b < c_fault_spec.burst_len && (bit_pos + b) < nbytes * 8; b++) {
        int target_byte = (bit_pos + b) / 8;
        int target_bit = (bit_pos + b) % 8;
        bytes[target_byte] ^= (1 << target_bit);
      }
    } else if (c_fault_spec.model != MODEL_BURST) {
      // Random or page_local: simple bit flip
      bytes[byte_idx] ^= (1 << bit_idx);
    }
  }
}

// ============================================================
// VERSION B: For REGISTER values (KV_WRITE and KV_READ)
// Returns modified value, avoids address-of-register spill
// Uses bitwise intrinsics on int representation
//
// CRITICAL: Uses nested std::conditional for type-width safety
// to match 1/2/4/8 byte types exactly (avoids union padding issues)
// ============================================================

// Type-width-safe integer type selector
template <typename T>
struct IntTypeForSize {
  using type = typename std::conditional<
      sizeof(T) == 1, uint8_t,
      typename std::conditional<
          sizeof(T) == 2, uint16_t,
          typename std::conditional<sizeof(T) == 4, uint32_t,
                                    uint64_t>::type>::type>::type;
};

template <typename T>
__device__ __forceinline__ T inject_fault_register(T value,
                                                    int physical_block_number,
                                                    int layer_idx) {
  if (!should_corrupt_block(c_fault_spec, physical_block_number, layer_idx))
    return value;

  // Use type-width-safe integer type
  using IntT = typename IntTypeForSize<T>::type;

  static_assert(sizeof(T) <= 8, "Type too large for register injection");
  static_assert(sizeof(T) == sizeof(IntT),
                "Type size mismatch in fault injector - check IntTypeForSize");

  union {
    T as_type;
    IntT as_bits;
  } u;
  u.as_type = value;

  int positions[8];
  int count;
  compute_flip_positions(c_fault_spec, physical_block_number, layer_idx,
                         sizeof(T), positions, &count);

  for (int i = 0; i < count; i++) {
    int bit_pos = positions[i] % (sizeof(T) * 8); // Constrain to valid range
    if (c_fault_spec.model == MODEL_MSB_BIASED) {
      // For MSB-biased, always flip a bit within the mask region
      // Map the random bit_in_byte to one of the set bits in msb_mask
      int byte_in_type = bit_pos / 8;
      uint8_t mask = c_fault_spec.msb_mask;
      if (mask != 0) {
        // Count set bits and select one based on random position
        int set_bit_count = __popc(static_cast<unsigned int>(mask));
        int target_idx = (positions[i] >> 3) % set_bit_count;
        int selected_bit = 0;
        for (int b = 0; b < 8 && target_idx >= 0; b++) {
          if (mask & (1 << b)) {
            if (target_idx == 0) {
              selected_bit = b;
              break;
            }
            target_idx--;
          }
        }
        int final_bit_pos = byte_in_type * 8 + selected_bit;
        u.as_bits ^= (static_cast<IntT>(1) << final_bit_pos);
      }
    } else if (c_fault_spec.model == MODEL_BURST && i == 0) {
      // Burst: flip consecutive bits
      for (int b = 0;
           b < c_fault_spec.burst_len && (bit_pos + b) < sizeof(T) * 8; b++) {
        u.as_bits ^= (static_cast<IntT>(1) << (bit_pos + b));
      }
    } else if (c_fault_spec.model != MODEL_BURST) {
      // Random or page_local
      u.as_bits ^= (static_cast<IntT>(1) << bit_pos);
    }
  }

  return u.as_type;
}

// ============================================================
// VERSION C: For vector types (float4, half2, etc.)
// Injects into one component based on bit position
// ============================================================
template <typename VecT, typename ScalarT>
__device__ __forceinline__ VecT
inject_fault_vec_register(VecT vec, int physical_block_number, int layer_idx) {
  if (!should_corrupt_block(c_fault_spec, physical_block_number, layer_idx))
    return vec;

  constexpr int num_elements = sizeof(VecT) / sizeof(ScalarT);
  ScalarT* elements = reinterpret_cast<ScalarT*>(&vec);

  int positions[8];
  int count;
  compute_flip_positions(c_fault_spec, physical_block_number, layer_idx,
                         sizeof(VecT), positions, &count);

  // Apply flip to the appropriate element based on bit position
  int bit_pos = positions[0];
  int elem_idx = (bit_pos / 8) / sizeof(ScalarT);
  if (elem_idx < num_elements) {
    elements[elem_idx] = inject_fault_register(elements[elem_idx],
                                                physical_block_number,
                                                layer_idx);
  }

  return vec;
}

// ============================================================
// Helper macros for conditional injection at different sites
// ============================================================

// Use at KV_WRITE sites (cache population)
#define VLLM_MAYBE_INJECT_FAULT_KV_WRITE(value, block_idx, layer_idx)          \
  do {                                                                          \
    if (c_fault_spec.enabled && c_fault_spec.site != SITE_KV_READ) {           \
      value = inject_fault_register(value, block_idx, layer_idx);              \
    }                                                                           \
  } while (0)

// Use at KV_READ sites (attention)
#define VLLM_MAYBE_INJECT_FAULT_KV_READ(value, block_idx, layer_idx)           \
  do {                                                                          \
    if (c_fault_spec.enabled && c_fault_spec.site != SITE_KV_WRITE) {          \
      value = inject_fault_register(value, block_idx, layer_idx);              \
    }                                                                           \
  } while (0)

// Use for vector types at KV_WRITE
#define VLLM_MAYBE_INJECT_FAULT_VEC_KV_WRITE(vec, scalar_t, block_idx,         \
                                             layer_idx)                         \
  do {                                                                          \
    if (c_fault_spec.enabled && c_fault_spec.site != SITE_KV_READ) {           \
      vec = inject_fault_vec_register<decltype(vec), scalar_t>(                \
          vec, block_idx, layer_idx);                                          \
    }                                                                           \
  } while (0)

// Use for vector types at KV_READ
#define VLLM_MAYBE_INJECT_FAULT_VEC_KV_READ(vec, scalar_t, block_idx,          \
                                            layer_idx)                          \
  do {                                                                          \
    if (c_fault_spec.enabled && c_fault_spec.site != SITE_KV_WRITE) {          \
      vec = inject_fault_vec_register<decltype(vec), scalar_t>(                \
          vec, block_idx, layer_idx);                                          \
    }                                                                           \
  } while (0)

#endif // __CUDACC__

} // namespace fault_injection
} // namespace vllm

#endif // VLLM_FAULT_INJECT
