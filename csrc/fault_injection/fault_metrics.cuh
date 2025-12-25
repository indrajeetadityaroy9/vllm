// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#ifdef VLLM_FAULT_INJECT

#include <cstdint>
#include <cuda_runtime.h>

namespace vllm {
namespace fault_injection {

// Telemetry counters for fault injection
// WARNING: At high fault rates, atomic contention can serialize GPU execution.
// This implementation uses warp-level reduction to minimize contention.
struct FaultMetrics {
  uint64_t injections_performed; // Total injection events
  uint64_t blocks_affected;      // Unique blocks that were corrupted
  uint64_t bits_flipped;         // Total bits flipped across all injections
};

// Global metrics in device memory (set via set_fault_metrics_ptr)
__device__ FaultMetrics* g_fault_metrics = nullptr;

// Host function to set metrics pointer (call before kernel launch)
inline void set_fault_metrics_ptr(FaultMetrics* d_metrics) {
  cudaMemcpyToSymbol(g_fault_metrics, &d_metrics, sizeof(FaultMetrics*));
}

// Host function to allocate and initialize metrics
inline FaultMetrics* alloc_fault_metrics() {
  FaultMetrics* d_metrics;
  cudaMalloc(&d_metrics, sizeof(FaultMetrics));
  cudaMemset(d_metrics, 0, sizeof(FaultMetrics));
  set_fault_metrics_ptr(d_metrics);
  return d_metrics;
}

// Host function to retrieve metrics
inline void get_fault_metrics(FaultMetrics* d_metrics, FaultMetrics* h_metrics) {
  cudaMemcpy(h_metrics, d_metrics, sizeof(FaultMetrics),
             cudaMemcpyDeviceToHost);
}

// Host function to reset metrics
inline void reset_fault_metrics(FaultMetrics* d_metrics) {
  cudaMemset(d_metrics, 0, sizeof(FaultMetrics));
}

// Host function to free metrics
inline void free_fault_metrics(FaultMetrics* d_metrics) {
  if (d_metrics) {
    FaultMetrics* null_ptr = nullptr;
    cudaMemcpyToSymbol(g_fault_metrics, &null_ptr, sizeof(FaultMetrics*));
    cudaFree(d_metrics);
  }
}

// ============================================================
// Warp-level reduction for efficient atomic updates
// Reduces contention by aggregating within warp before global atomic
// ============================================================

// Record an injection event with warp-level reduction
// Call this from the thread that performed the injection
// bits_count: number of bits flipped by this thread
__device__ __forceinline__ void record_injection_warp_reduced(int bits_count) {
  if (g_fault_metrics == nullptr) return;

  // Step 1: Warp-level aggregation using ballot
  unsigned active_mask = __activemask();
  unsigned did_inject_mask = __ballot_sync(active_mask, bits_count > 0);
  int warp_injections = __popc(did_inject_mask);

  // Step 2: Reduce bits_count across warp
  // Use shuffle to sum up bits_count from all lanes
  int warp_bits = bits_count;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    warp_bits += __shfl_xor_sync(active_mask, warp_bits, offset);
  }

  // Step 3: Only lane 0 of each warp does the atomic update
  int lane_id = threadIdx.x % 32;
  if (lane_id == 0 && warp_injections > 0) {
    atomicAdd(&g_fault_metrics->injections_performed,
              static_cast<uint64_t>(warp_injections));
    atomicAdd(&g_fault_metrics->bits_flipped,
              static_cast<uint64_t>(warp_bits));
  }
}

// Record a block being affected (called once per corrupted block)
// Uses warp-level reduction for efficiency
__device__ __forceinline__ void record_block_affected() {
  if (g_fault_metrics == nullptr) return;

  unsigned active_mask = __activemask();
  unsigned affected_mask = __ballot_sync(active_mask, true);
  int warp_count = __popc(affected_mask);

  int lane_id = threadIdx.x % 32;
  if (lane_id == 0 && warp_count > 0) {
    atomicAdd(&g_fault_metrics->blocks_affected,
              static_cast<uint64_t>(warp_count));
  }
}

// Simple (non-reduced) version for low-frequency updates
// Use only when injection rate is very low
__device__ __forceinline__ void record_injection_simple(int bits_count) {
  if (g_fault_metrics == nullptr || bits_count == 0) return;
  atomicAdd(&g_fault_metrics->injections_performed, 1ULL);
  atomicAdd(&g_fault_metrics->bits_flipped, static_cast<uint64_t>(bits_count));
}

} // namespace fault_injection
} // namespace vllm

#endif // VLLM_FAULT_INJECT
