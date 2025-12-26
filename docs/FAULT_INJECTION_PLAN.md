# vLLM Fault Injection: Complete Implementation & Evaluation Plan

## 1. Overview & Architectural Constraints

**Objective:** Implement bit-flip noise injection into vLLM's KV cache to study hardware fault tolerance.

**Scope:** Two injection sites (KV_WRITE, KV_READ) with four fault models (random, burst, msb_biased, page_local).

### Critical Constraints

1. **Backend Compatibility:**
   - **KV_WRITE:** Works universally (PagedAttention, FlashAttention, FlashMLA). This is the primary injection site.
   - **KV_READ:** Only works when `VLLM_ATTENTION_BACKEND=XFORMERS` (Legacy). Ignored by default vLLM V1 (FlashAttention).

2. **Performance Safety:**
   - Must use Register Injection (pass-by-value). Taking the address (`&var`) of a local variable causes register spills to Local Memory, destroying kernel performance.

3. **Type Safety:**
   - Bitwise operations must strictly match the width of the target type (`__half` vs `float`) to avoid undefined behavior in unions.

### Implementation Status

| Component              | Status  | Notes                                   |
|------------------------|---------|-----------------------------------------|
| Configuration System   | ✅ DONE | `vllm/config/fault_injection.py`        |
| CLI Arguments          | ✅ DONE | Added to `arg_utils.py`                 |
| CUDA Device Functions  | ✅ DONE | `csrc/fault_injection/fault_injector.cuh` |
| KV_WRITE Integration   | ✅ DONE | `csrc/cache_kernels.cu`                 |
| KV_READ Integration    | ✅ DONE | `csrc/attention/attention_kernels.cuh`  |
| Torch Bindings         | ✅ DONE | `csrc/torch_bindings.cpp`               |
| Python Bridge          | ✅ DONE | `vllm/_custom_ops.py`                   |
| Unit Tests             | ✅ DONE | `tests/kernels/test_fault_injection.py` |
| Telemetry/Metrics      | ⚠️ STUB | Header exists but not integrated        |
| Reliability Benchmarks | ❌ TODO | Phase 8 (new)                           |

---

## Phase 1: Configuration System

**Files:** `vllm/config/fault_injection.py`, `vllm/engine/arg_utils.py`

### 1.1 Python Configuration (✅ Implemented)

```python
@dataclass
class FaultInjectionConfig:
    enabled: bool = False
    site: str = "BOTH"        # KV_WRITE, KV_READ, BOTH
    subsite: str = "CODEWORD" # CODEWORD (Packed/ECC), VALUE (Decoded/Compute)
    model: str = "random"     # random, burst, msb_biased, page_local
    rate: float = 0.0         # 0.0-1.0
    flip_count: int = 1       # 1-8
    burst_len: int = 8        # 1-64
    msb_policy: str = "BYTE_TOPBITS"
    msb_mask: int = 0xF0
    page_scope: int = -1      # Physical block ID filter (-1 = all)
    seed: int = 42
```

### 1.2 CLI Arguments (✅ Implemented)

Flags: `--fault-injection-enabled`, `--fault-injection-rate`, `--fault-injection-site`, etc.

---

## Phase 2: CUDA Device Layer (The Core)

**File:** `csrc/fault_injection/fault_injector.cuh`

### 2.1 FaultSpec & Constant Memory (✅ Implemented)

```cpp
struct FaultSpec {
    bool enabled;
    uint8_t site;       // 0=WRITE, 1=READ, 2=BOTH
    uint8_t subsite;    // 0=CODEWORD, 1=VALUE
    uint8_t model;      // 0=random, 1=burst, 2=msb, 3=page_local
    float rate;
    uint8_t flip_count;
    uint8_t burst_len;
    uint8_t msb_policy;
    uint8_t msb_mask;
    int32_t page_scope;
    uint64_t seed;
};

__constant__ FaultSpec c_fault_spec;
```

### 2.2 Philox RNG (✅ Fixed)

**Bug Fix Applied:** Changed from invalid 40-bit constant to proper 64-bit:
```cpp
constexpr uint64_t PHILOX_M = 0xD2E7470EE14C6C93ULL;  // Was: 0xD2511F53D5
```

### 2.3 Register-Safe Injector with Type Safety (✅ Implemented)

```cpp
template<typename T>
__device__ __forceinline__ T inject_fault_register(T value, int block, int layer) {
    // Type-width-safe integer selection
    using IntT = typename std::conditional<sizeof(T) == 1, uint8_t,
                 typename std::conditional<sizeof(T) == 2, uint16_t,
                 typename std::conditional<sizeof(T) == 4, uint32_t,
                 uint64_t>::type>::type>::type;

    static_assert(sizeof(T) == sizeof(IntT), "Type size mismatch");

    union { T as_type; IntT as_bits; } u;
    u.as_type = value;
    // ... bit flip logic ...
    return u.as_type;
}
```

### 2.4 MSB-Biased Fix (✅ Implemented)

**Bug Fix Applied:** Now uses `__popc()` to select random bit WITHIN the mask:
```cpp
if (model == MSB_BIASED) {
    int set_bit_count = __popc(mask);
    int target_idx = rng % set_bit_count;
    // Find the target_idx'th set bit in mask
    // ... guaranteed to flip within mask region
}
```

---

## Phase 3: Integration - KV_WRITE (Universal) ✅

**File:** `csrc/cache_kernels.cu`

**Targets:** `reshape_and_cache_kernel`, `reshape_and_cache_flash_kernel`, `concat_and_cache_mla_kernel`

```cpp
#ifdef VLLM_FAULT_INJECT
if (c_fault_spec.enabled && c_fault_spec.site != SITE_KV_READ) {
    tgt_key = inject_fault_register(tgt_key, block_idx, layer_idx);
    tgt_value = inject_fault_register(tgt_value, block_idx, layer_idx);
}
#endif
key_cache[key_idx] = tgt_key;
```

---

## Phase 4: Integration - KV_READ (Legacy/Debug) ✅

**File:** `csrc/attention/attention_kernels.cuh`

**Warning:** Requires `VLLM_ATTENTION_BACKEND=XFORMERS`

```cpp
Qk_vec_k k_vec = *reinterpret_cast<Qk_vec_k*>(k_ptr);
#ifdef VLLM_FAULT_INJECT
if (c_fault_spec.enabled && c_fault_spec.site != SITE_KV_WRITE) {
    k_vec = inject_fault_vec_register<Qk_vec_k, scalar_t>(k_vec, block_number, 0);
}
#endif
```

**Note:** `layer_idx` is hardcoded to 0 (design limitation - not available in kernel).

---

## Phase 5: Host-Device Bridge ✅

**Files:** `csrc/torch_bindings.cpp`, `vllm/_custom_ops.py`

### 5.1 C++ Validation (✅ Added)

```cpp
TORCH_CHECK(site >= 0 && site <= 2, "Invalid site");
TORCH_CHECK(rate >= 0.0 && rate <= 1.0, "Invalid rate");
// ... validation for all parameters
```

### 5.2 Python Validation (✅ Added)

```python
if site not in site_map:
    raise ValueError(f"Invalid fault injection site: {site!r}")
# ... validation for all enum parameters
```

---

## Phase 6: Build System ✅

**File:** `CMakeLists.txt`

```cmake
option(VLLM_FAULT_INJECT "Enable fault injection" OFF)
if(VLLM_FAULT_INJECT)
    target_compile_definitions(_C PRIVATE VLLM_FAULT_INJECT)
endif()
```

### Build Commands

```bash
# 1. Clean
rm -rf build/ vllm/*.so
pip uninstall vllm -y

# 2. Build with fault injection (use available cores)
export VLLM_FAULT_INJECT=1
export MAX_JOBS=24
pip install -e . --no-build-isolation -v

# 3. Verify
python -c "import torch; print(hasattr(torch.ops._C, 'set_fault_injection_config'))"
# Should print: True
```

---

## Phase 7: Unit Tests ✅

**File:** `tests/kernels/test_fault_injection.py`

| Test                   | Status     |
|------------------------|------------|
| Default config values  | ✅         |
| Rate/mask validation   | ✅         |
| Hash determinism       | ✅         |
| Config enable/disable  | ✅         |
| Invalid enum rejection | ✅ (Added) |

```bash
pytest tests/kernels/test_fault_injection.py -v
```

---

## Phase 8: Inference Evaluation (TODO)

**Objective:** Quantify robustness beyond "it didn't crash."

**File:** `benchmarks/benchmark_reliability.py` (to be created)

### 8.1 Metrics

1. **KL Divergence:** Measure output distribution shift
2. **Top-1 Flip Rate:** How often does the token change?
3. **Crash Rate:** NaN/Inf causing CUDA exceptions

### 8.2 NIAH Reliability Test

Use `page_scope` to target specific block holding a "needle":

```python
cfg = FaultInjectionConfig(
    site="KV_WRITE",
    model="page_local",
    page_scope=needle_block_id,
    rate=1.0
)
```

### 8.3 Experimental Matrix

| Experiment | Site  | Model      | Metric     | Hypothesis                     |
|------------|-------|------------|------------|--------------------------------|
| Baseline   | -     | -          | Perplexity | Gold Standard                  |
| Bit Rot    | Write | Random     | Perplexity | Linear degradation             |
| Rowhammer  | Write | Burst      | Accuracy   | ECC failure simulation         |
| SDC Check  | Read  | MSB        | KL Div     | High confidence hallucinations |
| Retrieval  | Write | Page-Local | NIAH Score | Targeted memory corruption     |

---

## Execution Checklist

### Immediate (Build & Test)

- [ ] Clean old build artifacts
- [ ] Build with `VLLM_FAULT_INJECT=1`
- [ ] Verify `set_fault_injection_config` is available
- [ ] Run `pytest tests/kernels/test_fault_injection.py`
- [ ] **CRITICAL:** Smoke Test (must produce garbage output!)

### Smoke Test Command

```bash
# Force legacy backend to test BOTH read and write paths
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -c "
from vllm import LLM, SamplingParams
from vllm.config import FaultInjectionConfig

# Configure massive corruption
fi_config = FaultInjectionConfig(
    enabled=True,
    site='KV_WRITE',   # Universal site
    model='random',
    rate=1.0,          # 100% corruption probability
    seed=42
)

# Load model (use a small one for speed)
llm = LLM(model='facebook/opt-125m', fault_injection_config=fi_config, enforce_eager=True)

# Generate
print('--- GENERATION START ---')
print(llm.generate('The capital of France is', SamplingParams(temperature=0.0))[0].outputs[0].text)
print('--- GENERATION END ---')
"
```

**Success Signal:** Model outputs random characters/tokens (garbage).

**Failure Signal:** Model outputs coherent text (injector not working).

### Future (Evaluation)

- [ ] Create `benchmarks/benchmark_reliability.py`
- [ ] Implement KL divergence measurement
- [ ] Run experimental matrix sweeping rates `[1e-7, 1e-5, 1e-3]`
- [ ] Integrate telemetry counters (currently stub)

---

## Hardware Available

- **GPU:** 2x NVIDIA H100 80GB HBM3 (81,559 MiB each)
- **RAM:** 442 GB total (432 GB available)
- **CPU:** Intel Xeon Platinum 8480+ (26 cores, 52 threads)
- **CUDA:** 12.8
- **Driver:** 570.195.03
