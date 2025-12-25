// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// NOTE: This file is now a stub. The actual constant memory and setter
// are defined in cache_kernels.cu to ensure cudaMemcpyToSymbol updates
// the same symbol used by the KV cache kernels.
//
// See: csrc/cache_kernels.cu - set_cache_fault_spec()
// See: csrc/fault_injection/fault_injector.cuh - set_fault_spec() calls set_cache_fault_spec()

#ifdef VLLM_FAULT_INJECT
// Empty - all functionality moved to cache_kernels.cu
#endif  // VLLM_FAULT_INJECT
