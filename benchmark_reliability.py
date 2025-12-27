# SPDX-License-Identifier: Apache-2.0
# benchmark_reliability.py - Fault Injection Reliability Benchmark
import torch
import time
import gc
import sys

# Configuration
MODEL_NAME = "gpt2"  # Start with GPT-2 for quick testing
PROMPTS = [
    "Explain quantum entanglement in one sentence.",
    "Write a Python function to reverse a list.",
    "Who was the first US president?",
    "What is 25 * 4?",
    "The capital of France is",
] * 2  # 10 samples for quick test

def get_params():
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=50, ignore_eos=True)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def run_suite():
    from vllm import LLM
    results = []

    # 1. GOLDEN RUN (Baseline - no fault injection)
    print(f"\n{'='*60}")
    print(f"ESTABLISHING BASELINE ({MODEL_NAME})")
    print('='*60)
    cleanup()

    llm = LLM(model=MODEL_NAME, enforce_eager=True)
    outputs = llm.generate(PROMPTS, get_params())
    gold_texts = [o.outputs[0].text for o in outputs]

    print("Baseline outputs (first 3):")
    for i, text in enumerate(gold_texts[:3]):
        print(f"  [{i}]: {text[:60]}...")

    del llm
    cleanup()

    # 2. EXPERIMENTAL MATRIX
    experiments = [
        # A. Bit Rot (Random Noise) - increasing rates
        {"name": "BitRot_1e-7",  "rate": 1e-7, "model": "random", "site": "KV_WRITE"},
        {"name": "BitRot_1e-5",  "rate": 1e-5, "model": "random", "site": "KV_WRITE"},
        {"name": "BitRot_1e-3",  "rate": 1e-3, "model": "random", "site": "KV_WRITE"},
        {"name": "BitRot_1e-1",  "rate": 1e-1, "model": "random", "site": "KV_WRITE"},

        # B. Burst Errors (ECC failure simulation)
        {"name": "Burst_1e-5",   "rate": 1e-5, "model": "burst",  "site": "KV_WRITE", "burst_len": 8},

        # C. MSB Biased (Critical bit corruption)
        {"name": "MSB_1e-5",     "rate": 1e-5, "model": "msb_biased", "site": "KV_WRITE", "msb_policy": "FP16_EXPONENT"},
    ]

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp['name']}")
        print(f"  Rate: {exp['rate']}, Model: {exp['model']}, Site: {exp['site']}")
        print('='*60)
        cleanup()

        try:
            llm = LLM(
                model=MODEL_NAME,
                enforce_eager=True,
                fault_injection_enabled=True,
                fault_injection_site=exp["site"],
                fault_injection_model=exp["model"],
                fault_injection_rate=exp["rate"],
                fault_injection_burst_len=exp.get("burst_len", 8),
                fault_injection_msb_policy=exp.get("msb_policy", "BYTE_TOPBITS"),
                fault_injection_seed=42
            )

            start_t = time.time()
            outputs = llm.generate(PROMPTS, get_params())
            duration = time.time() - start_t

            # Metrics
            matches = 0
            total_drift = 0

            for i, out in enumerate(outputs):
                text = out.outputs[0].text
                if text == gold_texts[i]:
                    matches += 1
                else:
                    # Simple character-level drift
                    drift = sum(1 for a, b in zip(text, gold_texts[i]) if a != b)
                    drift += abs(len(text) - len(gold_texts[i]))
                    total_drift += drift

                    # Show first mismatch
                    if matches == 0 and i == 0:
                        print(f"\n  Sample Mismatch:")
                        print(f"    Gold:  {gold_texts[i][:50]}...")
                        print(f"    Fault: {text[:50]}...")

            acc = matches / len(PROMPTS)
            avg_drift = total_drift / len(PROMPTS)

            print(f"\n  RESULT: Accuracy={acc:.1%} | Avg Drift={avg_drift:.1f} chars | Time={duration:.2f}s")

            results.append({
                "name": exp["name"],
                "model": exp["model"],
                "rate": exp["rate"],
                "accuracy": acc,
                "avg_drift": avg_drift,
                "status": "OK"
            })

            del llm

        except Exception as e:
            print(f"\n  CRITICAL FAILURE: {str(e)[:100]}")
            results.append({
                "name": exp["name"],
                "model": exp["model"],
                "rate": exp["rate"],
                "accuracy": 0.0,
                "avg_drift": float('inf'),
                "status": "CRASH"
            })

    # 3. Final Report
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print('='*60)
    print(f"{'Experiment':<15} {'Model':<12} {'Rate':<10} {'Accuracy':<10} {'Status':<8}")
    print('-'*60)
    for r in results:
        print(f"{r['name']:<15} {r['model']:<12} {r['rate']:<10.0e} {r['accuracy']:<10.1%} {r['status']:<8}")

    return results

if __name__ == "__main__":
    print("Fault Injection Reliability Benchmark")
    print("="*60)
    run_suite()
