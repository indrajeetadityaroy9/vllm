# SPDX-License-Identifier: Apache-2.0
"""
KV Cache Stress Test: Simple Repetition Benchmark

This benchmark tests KV cache integrity by having the model repeat
patterns from earlier in the context. If the KV cache is corrupted,
the model will fail to reproduce the pattern correctly.

Key insight: By asking the model to complete a pattern it has seen,
we directly test whether the cached key-value pairs are intact.

Test design:
1. Show model a distinctive pattern: "ALPHA=12345, BETA=67890"
2. Add filler text to push the pattern into KV cache
3. Ask model to complete: "The value of ALPHA is"
4. Check if output contains "12345"

This is more robust than complex math for small models like GPT-2.
"""

import torch
import random
import time
import gc
from typing import List, Dict, Tuple

# Distinctive markers that are easy to check
MARKERS = [
    ("ZETA", "98765"),
    ("OMEGA", "13579"),
    ("DELTA", "24680"),
    ("SIGMA", "11111"),
    ("GAMMA", "99999"),
]

FILLER_SENTENCES = [
    "The weather patterns in coastal regions are influenced by ocean currents.",
    "Scientific research continues to advance our understanding of the universe.",
    "Technology has transformed how people communicate and work together.",
    "Environmental conservation requires cooperation between nations.",
    "Education systems evolve to meet changing societal needs.",
    "Healthcare improvements have increased life expectancy globally.",
    "Economic policies affect employment and growth rates.",
    "Cultural exchange enriches societies and promotes understanding.",
    "Infrastructure development supports urban and rural communities.",
    "Innovation drives progress in manufacturing and services.",
]


def generate_test_prompt(
    context_tokens: int,
    marker_name: str,
    marker_value: str,
    marker_position: float,  # 0.0 to 1.0, where in context to place marker
    tokenizer,
) -> Tuple[str, str]:
    """
    Generate a test prompt with a marker hidden in filler text.

    Returns:
        prompt: The full prompt ending with "The value of {marker_name} is"
        expected: The expected value (e.g., "98765")
    """
    # Calculate how much filler before and after the marker
    total_filler = context_tokens - 50  # Reserve space for marker and question
    filler_before = int(total_filler * marker_position)
    filler_after = total_filler - filler_before

    # Generate filler text
    def generate_filler(target_tokens):
        text_parts = []
        current_tokens = 0
        while current_tokens < target_tokens:
            sentence = random.choice(FILLER_SENTENCES)
            text_parts.append(sentence)
            current_tokens = len(tokenizer.encode(" ".join(text_parts)))
        return " ".join(text_parts)

    before_text = generate_filler(filler_before)
    after_text = generate_filler(filler_after)

    # Create the marker statement
    marker_statement = f"\n\n[IMPORTANT] The secret code {marker_name} equals {marker_value}.\n\n"

    # Build the full prompt
    prompt = f"{before_text}{marker_statement}{after_text}\n\nQuestion: What is the value of {marker_name}?\nAnswer: The value of {marker_name} is"

    return prompt, marker_value


def run_kv_stress_test(
    model_name: str = "gpt2",
    context_lengths: List[int] = [300, 600, 900],
    fault_rates: List[float] = [0.0, 1e-4, 1e-2, 0.1, 0.5],
    trials_per_config: int = 10,
    seed: int = 42,
):
    """
    Run KV cache stress test across different context lengths and fault rates.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = []

    print("=" * 70)
    print("KV CACHE STRESS TEST: Pattern Repetition Benchmark")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Context lengths: {context_lengths}")
    print(f"Fault rates: {fault_rates}")
    print(f"Trials per config: {trials_per_config}")
    print()

    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()

    for ctx_len in context_lengths:
        for fault_rate in fault_rates:
            print(f"\n{'='*70}")
            print(f"Context: {ctx_len} tokens | Fault Rate: {fault_rate:.0e}")
            print("=" * 70)

            cleanup()

            try:
                if fault_rate == 0.0:
                    llm = LLM(model=model_name, enforce_eager=True)
                else:
                    llm = LLM(
                        model=model_name,
                        enforce_eager=True,
                        fault_injection_enabled=True,
                        fault_injection_site="KV_WRITE",
                        fault_injection_model="random",
                        fault_injection_rate=fault_rate,
                        fault_injection_seed=seed,
                    )

                params = SamplingParams(temperature=0.0, max_tokens=20)

                correct = 0
                partial = 0

                for trial in range(trials_per_config):
                    # Select a random marker
                    marker_name, marker_value = random.choice(MARKERS)

                    # Place marker at different positions
                    position = 0.1 + (trial / trials_per_config) * 0.8  # 10% to 90%

                    prompt, expected = generate_test_prompt(
                        context_tokens=ctx_len,
                        marker_name=marker_name,
                        marker_value=marker_value,
                        marker_position=position,
                        tokenizer=tokenizer,
                    )

                    outputs = llm.generate([prompt], params)
                    output_text = outputs[0].outputs[0].text.strip()

                    # Check if the expected value appears in output
                    if expected in output_text:
                        correct += 1
                    elif any(c in output_text for c in expected[:3]):
                        # Partial match - some digits correct
                        partial += 1

                    # Debug first trial
                    if trial == 0:
                        print(f"\n  Trial 0:")
                        print(f"    Marker: {marker_name}={marker_value}")
                        print(f"    Position: {position:.0%} of context")
                        print(f"    Output: '{output_text[:50]}'")
                        print(f"    Match: {'YES' if expected in output_text else 'NO'}")

                accuracy = correct / trials_per_config
                partial_rate = partial / trials_per_config

                print(f"\n  RESULTS:")
                print(f"    Exact Match:   {accuracy:.0%} ({correct}/{trials_per_config})")
                print(f"    Partial Match: {partial_rate:.0%} ({partial}/{trials_per_config})")
                print(f"    Total Recall:  {(correct + partial) / trials_per_config:.0%}")

                results.append({
                    "context_length": ctx_len,
                    "fault_rate": fault_rate,
                    "accuracy": accuracy,
                    "partial": partial_rate,
                    "trials": trials_per_config,
                })

                del llm

            except Exception as e:
                print(f"  ERROR: {str(e)[:100]}")
                results.append({
                    "context_length": ctx_len,
                    "fault_rate": fault_rate,
                    "accuracy": 0.0,
                    "partial": 0.0,
                    "error": str(e)[:100],
                })

            cleanup()

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT: KV Cache Stress Test")
    print("=" * 70)
    print(f"{'Context':<10} {'Fault Rate':<12} {'Accuracy':<12} {'Partial':<12} {'Total':<12}")
    print("-" * 70)

    for r in results:
        ctx = r["context_length"]
        rate = r["fault_rate"]
        acc = r["accuracy"]
        part = r.get("partial", 0)
        total = acc + part
        print(f"{ctx:<10} {rate:<12.0e} {acc:<12.0%} {part:<12.0%} {total:<12.0%}")

    # Calculate degradation
    print("\n" + "-" * 70)
    print("DEGRADATION ANALYSIS:")

    # Group by context length
    for ctx_len in context_lengths:
        ctx_results = [r for r in results if r["context_length"] == ctx_len]
        baseline = next((r for r in ctx_results if r["fault_rate"] == 0.0), None)
        if baseline:
            print(f"\n  Context {ctx_len} tokens:")
            for r in ctx_results:
                if r["fault_rate"] > 0:
                    degradation = baseline["accuracy"] - r["accuracy"]
                    print(f"    Rate {r['fault_rate']:.0e}: {degradation:+.0%} degradation")

    return results


def run_quick_demo():
    """Quick demonstration of KV cache corruption."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("=" * 70)
    print("QUICK DEMO: KV Cache Corruption")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Simple test case
    marker = "OMEGA=13579"
    filler = " ".join([random.choice(FILLER_SENTENCES) for _ in range(15)])
    prompt = f"\n\n{marker}\n\n{filler}\n\nWhat is OMEGA? OMEGA="

    print(f"\nPrompt has {len(tokenizer.encode(prompt))} tokens")
    print(f"Hidden marker: {marker}")

    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()

    params = SamplingParams(temperature=0.0, max_tokens=10)

    # Baseline
    cleanup()
    llm = LLM(model="gpt2", enforce_eager=True)
    outputs = llm.generate([prompt], params)
    baseline = outputs[0].outputs[0].text.strip()
    print(f"\nBaseline output: '{baseline}'")
    del llm
    cleanup()

    # With 50% fault injection
    llm = LLM(
        model="gpt2",
        enforce_eager=True,
        fault_injection_enabled=True,
        fault_injection_site="KV_WRITE",
        fault_injection_model="random",
        fault_injection_rate=0.5,
        fault_injection_seed=42,
    )
    outputs = llm.generate([prompt], params)
    faulted = outputs[0].outputs[0].text.strip()
    print(f"Faulted output:  '{faulted}'")
    del llm
    cleanup()

    if baseline != faulted:
        print("\n✓ CORRUPTION DETECTED - outputs differ!")
    else:
        print("\n○ No corruption detected in this trial")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_quick_demo()
    else:
        run_kv_stress_test(
            model_name="gpt2",
            context_lengths=[300, 600],
            fault_rates=[0.0, 1e-3, 0.1, 0.5],
            trials_per_config=5,
        )
