# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ECC Noise Injection Benchmark

Evaluates signal reconstruction quality across different encoding modes
and noise injection levels using compiled vLLM ECC kernels directly.

Encoding Modes:
1. FP16 (baseline - no quantization, no ECC)
2. INT4 (quantization only, no ECC) - simulated baseline
3. INT4 + Hamming(7,4) ECC - compiled vLLM kernel
4. INT4 + Hamming(8,4) SECDED with N-LERP - compiled vLLM kernel

Noise Models:
- Random single-bit errors (uniformly distributed)
- Burst errors (2-bit errors in same codeword)
- Mixed errors (combination of single and burst)
"""

import argparse
import time
from dataclasses import dataclass

import torch
import numpy as np


def ecc_available() -> bool:
    """Check if ECC kernels are compiled in vLLM."""
    try:
        import vllm._C  # noqa: F401
        _ = torch.ops._C_cache_ops.ecc_encode
        return True
    except Exception:
        return False


# Algorithm constants (must match kernel)
ALGO_HAMMING74 = 0
ALGO_SECDED = 1


@dataclass
class BenchmarkConfig:
    num_tokens: int = 1024
    num_heads: int = 32
    head_dim: int = 128
    block_size: int = 16
    device: str = "cuda:0"
    seed: int = 42
    error_rates: tuple = (0.0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2)
    num_trials: int = 3


@dataclass
class BenchmarkResult:
    mode: str
    noise_type: str
    error_rate: float
    mse: float
    psnr: float
    max_error: float
    corruption_rate: float


# =============================================================================
# Vectorized GPU Noise Injection
# =============================================================================

def inject_random_errors(tensor: torch.Tensor, error_rate: float) -> torch.Tensor:
    """Inject random single-bit errors (vectorized GPU)."""
    if error_rate <= 0:
        return tensor
    result = tensor.clone()
    flat = result.view(-1)
    n = flat.numel()

    # Probability of at least one bit flip per byte
    prob = 1.0 - (1.0 - error_rate) ** 8
    mask = torch.rand(n, device=tensor.device) < prob
    num_errs = mask.sum().item()

    if num_errs > 0:
        bits = torch.randint(0, 8, (num_errs,), device=tensor.device, dtype=torch.uint8)
        errs = (1 << bits).to(torch.uint8)
        indices = mask.nonzero(as_tuple=True)[0]
        flat[indices] ^= errs
    return result


def inject_burst_errors(tensor: torch.Tensor, error_rate: float) -> torch.Tensor:
    """Inject 2-bit burst errors (vectorized GPU)."""
    if error_rate <= 0:
        return tensor
    result = tensor.clone()
    flat = result.view(-1)
    n = flat.numel()

    mask = torch.rand(n, device=tensor.device) < error_rate
    num_errs = mask.sum().item()

    if num_errs > 0:
        bits = torch.randint(0, 7, (num_errs,), device=tensor.device, dtype=torch.uint8)
        errs = ((1 << bits) | (1 << (bits + 1))).to(torch.uint8)
        indices = mask.nonzero(as_tuple=True)[0]
        flat[indices] ^= errs
    return result


def inject_mixed_errors(tensor: torch.Tensor, error_rate: float) -> torch.Tensor:
    """Inject mix of single-bit (70%) and burst (30%) errors."""
    if error_rate <= 0:
        return tensor
    result = inject_random_errors(tensor, error_rate * 0.7)
    result = inject_burst_errors(result, error_rate * 0.3)
    return result


def apply_noise(tensor: torch.Tensor, noise_type: str, error_rate: float) -> torch.Tensor:
    """Apply noise based on type."""
    if noise_type == "random":
        return inject_random_errors(tensor, error_rate)
    elif noise_type == "burst":
        return inject_burst_errors(tensor, error_rate)
    elif noise_type == "mixed":
        return inject_mixed_errors(tensor, error_rate)
    return tensor


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Compute quality metrics."""
    diff = original.float() - reconstructed.float()
    mse = (diff ** 2).mean().item()
    max_error = diff.abs().max().item()
    psnr = 10 * np.log10(64.0 / mse) if mse > 0 else float('inf')
    corruption_rate = (diff.abs() > 0.5).float().mean().item() * 100
    return {'mse': mse, 'psnr': psnr, 'max_error': max_error, 'corruption_rate': corruption_rate}


# =============================================================================
# FP16 Baseline (no protection)
# =============================================================================

def benchmark_fp16(key: torch.Tensor, noise_type: str, error_rate: float) -> BenchmarkResult:
    """FP16 baseline - shows vulnerability of unprotected storage."""
    key_bytes = key.clone().view(torch.uint8)
    key_bytes = apply_noise(key_bytes, noise_type, error_rate)
    key_decoded = key_bytes.view(torch.float16).view_as(key)
    key_decoded = torch.nan_to_num(key_decoded, nan=0.0, posinf=4.0, neginf=-4.0)

    m = compute_metrics(key, key_decoded)
    return BenchmarkResult("FP16", noise_type, error_rate, m['mse'], m['psnr'], m['max_error'], m['corruption_rate'])


# =============================================================================
# INT4 Baseline (quantization only, no ECC) - Simulated
# =============================================================================

def benchmark_int4(key: torch.Tensor, noise_type: str, error_rate: float) -> BenchmarkResult:
    """INT4 baseline - shows what happens without ECC protection.

    Uses same quantization formula as kernel: q = round(f * 1.875 + 7.5)
    """
    # Quantize (matching kernel exactly)
    f = key.float().clamp(-4.0, 4.0)
    q = torch.round(f * 1.875 + 7.5).clamp(0, 15).to(torch.uint8)

    # Pack 2 values per byte
    flat = q.view(-1)
    packed = (flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)

    # Inject noise
    packed = apply_noise(packed, noise_type, error_rate)

    # Unpack and dequantize (matching kernel exactly)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.zeros(packed.numel() * 2, dtype=torch.uint8, device=key.device)
    unpacked[0::2] = low
    unpacked[1::2] = high
    decoded = ((unpacked.float() - 7.5) / 1.875).to(torch.float16).view_as(key)

    m = compute_metrics(key, decoded)
    return BenchmarkResult("INT4", noise_type, error_rate, m['mse'], m['psnr'], m['max_error'], m['corruption_rate'])


# =============================================================================
# ECC Modes - Using ONLY Compiled vLLM Kernels
# =============================================================================

def benchmark_ecc(key: torch.Tensor, config: BenchmarkConfig,
                  algo: int, algo_name: str, noise_type: str, error_rate: float) -> BenchmarkResult:
    """Benchmark using compiled vLLM ECC kernels directly."""
    device = key.device
    num_blocks = (config.num_tokens + config.block_size - 1) // config.block_size + 10

    # Allocate cache and metadata
    key_cache = torch.zeros(num_blocks, config.block_size, config.num_heads, config.head_dim,
                            dtype=torch.uint8, device=device)
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.arange(config.num_tokens, dtype=torch.long, device=device)
    seq_start_locs = torch.tensor([0, config.num_tokens], dtype=torch.long, device=device)
    value = key.clone()

    # ENCODE using compiled kernel
    torch.ops._C_cache_ops.ecc_encode(key, value, key_cache, value_cache, slot_mapping, algo)
    torch.cuda.synchronize()

    # Inject noise into ECC-protected cache
    key_cache_noisy = apply_noise(key_cache, noise_type, error_rate)
    value_cache_noisy = apply_noise(value_cache, noise_type, error_rate)

    # DECODE using compiled kernel (with error correction / N-LERP reconstruction)
    workspace_k = torch.zeros_like(key)
    workspace_v = torch.zeros_like(value)
    torch.ops._C_cache_ops.ecc_gather_decode(
        key_cache_noisy, value_cache_noisy, slot_mapping, seq_start_locs,
        workspace_k, workspace_v, config.num_tokens, config.num_heads,
        config.head_dim, config.block_size, 1, algo
    )
    torch.cuda.synchronize()

    m = compute_metrics(key, workspace_k)
    return BenchmarkResult(algo_name, noise_type, error_rate, m['mse'], m['psnr'], m['max_error'], m['corruption_rate'])


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_benchmark(config: BenchmarkConfig) -> list:
    """Run the complete benchmark suite."""
    print("=" * 80)
    print("ECC Noise Injection Benchmark")
    print("=" * 80)
    print(f"Tokens: {config.num_tokens}, Heads: {config.num_heads}, HeadDim: {config.head_dim}")
    print(f"Trials: {config.num_trials}, Device: {config.device}")

    has_ecc = ecc_available()
    print(f"ECC Kernels: {'Compiled (using torch.ops._C_cache_ops)' if has_ecc else 'NOT AVAILABLE'}")
    print("=" * 80)

    device = torch.device(config.device)
    noise_types = ["random", "burst", "mixed"]
    results = []

    # Create test data
    torch.manual_seed(config.seed)
    key = torch.randn(config.num_tokens, config.num_heads, config.head_dim,
                      device=device, dtype=torch.float16).clamp(-4.0, 4.0)

    total = len(noise_types) * len(config.error_rates) * config.num_trials
    count = 0
    start_time = time.time()

    for noise_type in noise_types:
        for error_rate in config.error_rates:
            trial_results = {'FP16': [], 'INT4': [], 'Hamming(7,4)': [], 'SECDED+N-LERP': []}

            for trial in range(config.num_trials):
                torch.manual_seed(config.seed + trial)

                # FP16 baseline
                trial_results['FP16'].append(benchmark_fp16(key, noise_type, error_rate))

                # INT4 baseline (simulated - no kernel available)
                trial_results['INT4'].append(benchmark_int4(key, noise_type, error_rate))

                # ECC modes using compiled kernels
                if has_ecc:
                    trial_results['Hamming(7,4)'].append(
                        benchmark_ecc(key, config, ALGO_HAMMING74, "Hamming(7,4)", noise_type, error_rate))
                    trial_results['SECDED+N-LERP'].append(
                        benchmark_ecc(key, config, ALGO_SECDED, "SECDED+N-LERP", noise_type, error_rate))

                count += 1

            # Aggregate trials
            for mode, mode_results in trial_results.items():
                if not mode_results:
                    continue
                psnr_vals = [r.psnr for r in mode_results if r.psnr != float('inf')]
                avg_psnr = float(np.mean(psnr_vals)) if psnr_vals else float('inf')
                results.append(BenchmarkResult(
                    mode=mode,
                    noise_type=noise_type,
                    error_rate=error_rate,
                    mse=float(np.mean([r.mse for r in mode_results])),
                    psnr=avg_psnr,
                    max_error=float(np.mean([r.max_error for r in mode_results])),
                    corruption_rate=float(np.mean([r.corruption_rate for r in mode_results])),
                ))

            # Progress
            pct = count / total * 100
            if pct % 10 < (count - config.num_trials) / total * 100 % 10:
                print(f"  Progress: {pct:.0f}%")

    print(f"\nCompleted in {time.time() - start_time:.1f}s")
    return results


def print_summary_table(results: list):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (PSNR dB / Corruption %)")
    print("=" * 100)

    modes = ["FP16", "INT4", "Hamming(7,4)", "SECDED+N-LERP"]

    for noise_type in ["random", "burst", "mixed"]:
        print(f"\n{noise_type.upper()} ERRORS:")
        print("-" * 100)
        print(f"{'Error Rate':>12}", end="")
        for mode in modes:
            print(f" | {mode:>18}", end="")
        print(" |")
        print("-" * 100)

        for er in sorted(set(r.error_rate for r in results)):
            print(f"{er:>12.2e}", end="")
            for mode in modes:
                r = next((r for r in results if r.noise_type == noise_type and r.error_rate == er and r.mode == mode), None)
                if r:
                    psnr_str = f"{r.psnr:.1f}" if r.psnr != float('inf') else "inf"
                    print(f" | {psnr_str:>6}dB/{r.corruption_rate:>5.1f}%", end="")
                else:
                    print(f" | {'N/A':>18}", end="")
            print(" |")
    print("-" * 100)


def print_analysis(results: list):
    """Print comparative analysis."""
    print("\n" + "=" * 80)
    print("SECDED vs HAMMING COMPARISON (Burst Errors)")
    print("=" * 80)

    for er in sorted(set(r.error_rate for r in results)):
        if er == 0:
            continue
        h74 = next((r for r in results if r.noise_type == "burst" and r.error_rate == er and r.mode == "Hamming(7,4)"), None)
        sec = next((r for r in results if r.noise_type == "burst" and r.error_rate == er and r.mode == "SECDED+N-LERP"), None)

        if h74 and sec and h74.mse > 0:
            improv = (h74.mse - sec.mse) / h74.mse * 100
            print(f"  Error rate {er:.2e}: SECDED reduces MSE by {improv:.1f}% (PSNR: {h74.psnr:.1f}→{sec.psnr:.1f}dB)")


def save_csv(results: list, filename: str):
    """Save results to CSV."""
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'noise_type', 'error_rate', 'mse', 'psnr', 'max_error', 'corruption_rate'])
        for r in results:
            writer.writerow([r.mode, r.noise_type, r.error_rate, r.mse, r.psnr, r.max_error, r.corruption_rate])
    print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="ECC Noise Injection Benchmark")
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    config = BenchmarkConfig(
        num_tokens=args.num_tokens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_trials=args.num_trials,
        device=args.device,
    )

    results = run_benchmark(config)
    print_summary_table(results)
    print_analysis(results)

    if args.output:
        save_csv(results, args.output)


if __name__ == "__main__":
    main()
