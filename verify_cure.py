import argparse
import torch
import gc
import random
import sys
from vllm import LLM, SamplingParams

def cleanup():
    """Force garbage collection to prevent OOM between runs."""
    gc.collect()
    torch.cuda.empty_cache()

def run_reasoning_needle_test(llm, context_len=2000):
    """
    Generates a long context with two hidden numbers (Alpha, Beta).
    The model must retrieve them AND perform multiplication.

    Returns: True if the correct product is found in the output.
    """
    print(f"\n>>> Generating Math Needle Scenario (Len={context_len})...")

    # 1. Generate Haystack
    # Use coherent filler to fill the KV cache blocks without confusing the model
    filler_sentence = "The rapid brown fox jumps over the lazy dog repeatedly to test memory stability. "
    tokens_per_sentence = 14 # Approx
    repeats = context_len // tokens_per_sentence

    haystack = (filler_sentence * repeats).split()

    # 2. Insert Needles (Critical Data)
    val_a = random.randint(10, 50)
    val_b = random.randint(10, 50)
    target_product = val_a * val_b

    # Insert at 10% (Early context) and 90% (Late context)
    # This forces the attention mechanism to span the entire cache
    pos_a = int(len(haystack) * 0.1)
    pos_b = int(len(haystack) * 0.9)

    haystack.insert(pos_a, f" The secret variable Alpha is {val_a}. ")
    haystack.insert(pos_b, f" The secret variable Beta is {val_b}. ")

    context = " ".join(haystack)

    # Qwen-specific prompt format can help, but standard instruction works generally
    prompt = f"{context}\n\nQuestion: Calculate the product of Alpha and Beta. Think step by step and provide the numeric answer."

    # 3. Generate
    print(f"Goal: {val_a} * {val_b} = {target_product}")
    print("Generating response...")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    outputs = llm.generate(prompt, sampling_params)
    text = outputs[0].outputs[0].text

    print(f"Model Output: {text.strip()}")

    # 4. Strict Grading
    if str(target_product) in text:
        print("Analysis: ✅ Logic Verified (Correct Answer Found).")
        return True
    else:
        print("Analysis: ❌ Logic FAILED (Answer missing or wrong).")
        return False

def run_verification(model_name, fault_model, rate, algorithm):
    cleanup()
    print(f"\n==================================================")
    print(f"VERIFYING CURE: {fault_model.upper()} @ {rate} | Algo: {algorithm}")
    print(f"==================================================")

    # 1. Map Algorithm to vLLM Cache Type
    dtype_map = {
        "lsq_secded_16_11": "int4_ecc_lsq",  # The New Hero
        "secded_8_4":       "int4_ecc",      # The Old Attempt (Low Precision)
        "hamming_7_4":      "int4_hamming",  # The Baseline (Mis-corrects bursts)
        "none":             "fp8"            # Control Group
    }

    if algorithm not in dtype_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. Valid: {list(dtype_map.keys())}")

    selected_dtype = dtype_map[algorithm]
    print(f"Config: kv_cache_dtype='{selected_dtype}'")

    # 2. Configure the Threat (Fault Injection)
    # Burst len 2 is critical to distinguish SECDED (Detects) from Hamming (Fails)
    burst_len = 2 if fault_model == "burst" else 1

    # Only enable fault injection if rate > 0
    fault_enabled = rate > 0

    print(f"Config: Faults={fault_model}, Rate={rate}, BurstLen={burst_len}, Enabled={fault_enabled}")

    # 3. Load Model
    # kv_cache_dtype='fp8' or specific int4 type ensures 1-byte storage slots
    # Use field-based fault injection API
    try:
        llm = LLM(
            model=model_name,
            kv_cache_dtype=selected_dtype,
            enforce_eager=True,            # Safer for fault injection PoC
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            # Fault injection parameters (field-based API)
            fault_injection_enabled=fault_enabled,
            fault_injection_site="KV_WRITE",
            fault_injection_model=fault_model,
            fault_injection_rate=float(rate),
            fault_injection_burst_len=burst_len,
            fault_injection_seed=42,
        )
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to load model.\n{e}")
        return

    # 4. Run the Stress Test (3 Attempts)
    wins = 0
    attempts = 3

    for i in range(attempts):
        print(f"\n--- Attempt {i+1}/{attempts} ---")
        if run_reasoning_needle_test(llm, context_len=2000):
            wins += 1
            print("Result: PASS")
        else:
            print("Result: FAIL")

    # 5. Final Verdict
    print(f"\n==================================================")
    print(f"FINAL SCORE: {wins}/{attempts}")

    if algorithm == "lsq_secded_16_11" and wins >= 2:
        print(f"VERDICT: ✅ CURE SUCCESSFUL (LSQ Restored Logic)")
    elif wins >= 2:
        print(f"VERDICT: ✅ PASS (Baseline/Control)")
    else:
        print(f"VERDICT: ❌ FAILURE (Model Broken)")
    print(f"==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--fault-model", type=str, default="burst", choices=["random", "burst"])
    parser.add_argument("--rate", type=float, default=1e-4)
    parser.add_argument("--algorithm", type=str, default="lsq_secded_16_11",
                        choices=["lsq_secded_16_11", "secded_8_4", "hamming_7_4", "none"])
    args = parser.parse_args()

    run_verification(args.model, args.fault_model, args.rate, args.algorithm)
