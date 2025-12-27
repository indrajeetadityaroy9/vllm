# SPDX-License-Identifier: Apache-2.0
"""
Math-Needle Benchmark for LLaMA-3.1-8B: KV Cache Reasoning Stress Test

This benchmark tests KV cache integrity by requiring the model to:
1. Retrieve multiple values hidden in long context
2. Perform arithmetic operations on those values

Key insight: Retrieval tests only detect corruption in the retrieved block,
but reasoning tests detect corruption in ANY block needed for computation.
"""

import torch
import random
import time
import gc
from typing import List, Tuple, Optional

# Filler text corpus
FILLER_PARAGRAPHS = [
    "The process of photosynthesis converts light energy into chemical energy stored in glucose molecules. Plants use chlorophyll in their leaves to absorb sunlight, which drives the reaction between carbon dioxide and water.",
    "Ancient civilizations developed sophisticated astronomical observations to track celestial bodies. The Mayans created accurate calendars based on Venus cycles, while Babylonians recorded planetary movements on clay tablets.",
    "Neural networks consist of interconnected nodes organized in layers that process information. Each connection has an associated weight that is adjusted during training through backpropagation.",
    "The water cycle describes the continuous movement of water within the Earth and atmosphere. Evaporation from oceans and lakes creates water vapor that rises and condenses into clouds.",
    "Quantum mechanics describes the behavior of matter and energy at atomic scales. Particles exhibit wave-particle duality, existing in superposition states until measured.",
    "Economic markets operate through the interaction of supply and demand forces. When demand exceeds supply, prices tend to rise, incentivizing increased production.",
    "The human immune system provides defense against pathogens through multiple mechanisms. Innate immunity offers immediate but non-specific protection.",
    "Continental drift theory explains how landmasses have moved over geological time. Tectonic plates float on the semi-fluid asthenosphere and interact at boundaries.",
    "Machine learning algorithms identify patterns in data to make predictions or decisions. Supervised learning uses labeled examples to train models.",
    "The Renaissance marked a cultural rebirth in Europe spanning the 14th to 17th centuries. Artists like Leonardo da Vinci revolutionized painting and sculpture.",
]


def generate_filler_text(target_tokens: int, tokenizer) -> str:
    """Generate filler text of approximately target_tokens length."""
    paragraphs = []
    current_tokens = 0
    while current_tokens < target_tokens:
        para = random.choice(FILLER_PARAGRAPHS)
        paragraphs.append(para)
        current_tokens = len(tokenizer.encode(" ".join(paragraphs)))
    text = " ".join(paragraphs)
    tokens = tokenizer.encode(text)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text


def create_math_needle_prompt(
    context_length: int,
    var_positions: List[float],
    operation: str,
    tokenizer,
    seed: int = 42
) -> Tuple[str, int, dict]:
    """Create a prompt with hidden variables and a math question."""
    random.seed(seed)

    var_values = {
        "var_a": random.randint(10, 99),
        "var_b": random.randint(10, 99),
        "var_c": random.randint(10, 99),
    }

    if operation == "add":
        expected = var_values["var_a"] + var_values["var_b"]
        question = "Calculate var_a + var_b. Respond with only the number."
    elif operation == "multiply":
        expected = var_values["var_a"] * var_values["var_b"]
        question = "Calculate var_a * var_b. Respond with only the number."
    elif operation == "chain":
        expected = (var_values["var_a"] + var_values["var_b"]) * var_values["var_c"]
        question = "Calculate (var_a + var_b) * var_c. Respond with only the number."
    else:
        raise ValueError(f"Unknown operation: {operation}")

    total_filler = context_length - 300
    var_statements = [
        f"\n\n[CRITICAL DATA] var_a = {var_values['var_a']}\n\n",
        f"\n\n[CRITICAL DATA] var_b = {var_values['var_b']}\n\n",
        f"\n\n[CRITICAL DATA] var_c = {var_values['var_c']}\n\n",
    ]

    positions_tokens = [int(p * total_filler) for p in var_positions]
    full_filler = generate_filler_text(total_filler, tokenizer)
    filler_tokens = tokenizer.encode(full_filler)

    result_tokens = []
    last_pos = 0
    for i, pos in enumerate(positions_tokens):
        if i < len(var_statements):
            result_tokens.extend(filler_tokens[last_pos:pos])
            var_tokens = tokenizer.encode(var_statements[i])
            result_tokens.extend(var_tokens)
            last_pos = pos
    result_tokens.extend(filler_tokens[last_pos:])

    context = tokenizer.decode(result_tokens, skip_special_tokens=True)
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    metadata = {
        "var_values": var_values,
        "positions": var_positions,
        "operation": operation,
        "context_tokens": len(tokenizer.encode(context)),
    }

    return prompt, expected, metadata


def extract_number(text: str) -> Optional[int]:
    """Extract the first number from model output."""
    import re
    matches = re.findall(r'-?\d+', text.strip())
    if matches:
        try:
            return int(matches[0])
        except ValueError:
            return None
    return None


def run_benchmark(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    context_lengths: List[int] = [2000, 4000, 8000],
    fault_rates: List[float] = [0.0, 1e-5, 1e-4, 1e-3],
    num_trials: int = 5,
    seed: int = 42,
):
    """Run the Math-Needle benchmark with LLaMA-3.1-8B."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = []

    print("=" * 70)
    print("MATH-NEEDLE BENCHMARK: LLaMA-3.1-8B KV Cache Stress Test")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Context lengths: {context_lengths}")
    print(f"Fault rates: {fault_rates}")
    print(f"Trials per config: {num_trials}")
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
                    llm = LLM(model=model_name, enforce_eager=True, max_model_len=16384)
                else:
                    llm = LLM(
                        model=model_name,
                        enforce_eager=True,
                        max_model_len=16384,
                        fault_injection_enabled=True,
                        fault_injection_site="KV_WRITE",
                        fault_injection_model="random",
                        fault_injection_rate=fault_rate,
                        fault_injection_seed=seed,
                    )

                params = SamplingParams(temperature=0.0, max_tokens=20)

                correct_retrieval = 0
                correct_reasoning = 0

                for trial in range(num_trials):
                    trial_seed = seed + trial

                    # Test 1: Retrieval (single variable)
                    prompt_ret, _, meta_ret = create_math_needle_prompt(
                        context_length=ctx_len,
                        var_positions=[0.5],
                        operation="add",
                        tokenizer=tokenizer,
                        seed=trial_seed,
                    )
                    prompt_ret = prompt_ret.replace(
                        "Calculate var_a + var_b. Respond with only the number.",
                        "What is the value of var_a? Respond with only the number."
                    )
                    expected_ret = meta_ret["var_values"]["var_a"]

                    outputs = llm.generate([prompt_ret], params)
                    answer = extract_number(outputs[0].outputs[0].text)
                    if answer == expected_ret:
                        correct_retrieval += 1

                    # Test 2: Reasoning (both variables)
                    prompt_reason, expected_reason, meta_reason = create_math_needle_prompt(
                        context_length=ctx_len,
                        var_positions=[0.2, 0.8],
                        operation="multiply",
                        tokenizer=tokenizer,
                        seed=trial_seed + 1000,
                    )

                    outputs = llm.generate([prompt_reason], params)
                    answer = extract_number(outputs[0].outputs[0].text)
                    if answer == expected_reason:
                        correct_reasoning += 1

                    if trial == 0:
                        a = meta_reason["var_values"]["var_a"]
                        b = meta_reason["var_values"]["var_b"]
                        print(f"\n  Trial 0 - Retrieval: Expected={expected_ret}, Got={extract_number(outputs[0].outputs[0].text) if trial == 0 else 'N/A'}")
                        print(f"  Trial 0 - Reasoning: {a} * {b} = {expected_reason}")
                        print(f"    Output: {outputs[0].outputs[0].text[:60]}")

                retrieval_acc = correct_retrieval / num_trials
                reasoning_acc = correct_reasoning / num_trials

                print(f"\n  RESULTS:")
                print(f"    Retrieval Accuracy:  {retrieval_acc:.0%} ({correct_retrieval}/{num_trials})")
                print(f"    Reasoning Accuracy:  {reasoning_acc:.0%} ({correct_reasoning}/{num_trials})")

                if retrieval_acc < 1.0:
                    sensitivity = (1 - reasoning_acc) / (1 - retrieval_acc)
                    print(f"    Sensitivity Ratio:   {sensitivity:.2f}x")

                results.append({
                    "context_length": ctx_len,
                    "fault_rate": fault_rate,
                    "retrieval_accuracy": retrieval_acc,
                    "reasoning_accuracy": reasoning_acc,
                    "trials": num_trials,
                })

                del llm

            except Exception as e:
                print(f"  ERROR: {str(e)[:100]}")
                results.append({
                    "context_length": ctx_len,
                    "fault_rate": fault_rate,
                    "retrieval_accuracy": 0.0,
                    "reasoning_accuracy": 0.0,
                    "error": str(e)[:100],
                })

            cleanup()

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT: Math-Needle Benchmark (LLaMA-3.1-8B)")
    print("=" * 70)
    print(f"{'Context':<10} {'Fault Rate':<12} {'Retrieval':<12} {'Reasoning':<12}")
    print("-" * 70)

    for r in results:
        ctx = r["context_length"]
        rate = r["fault_rate"]
        ret = r["retrieval_accuracy"]
        reas = r["reasoning_accuracy"]
        print(f"{ctx:<10} {rate:<12.0e} {ret:<12.0%} {reas:<12.0%}")

    print("\nKEY INSIGHT: Reasoning accuracy should degrade faster than retrieval")
    print("because it requires multiple KV cache blocks to be correct.")

    return results


if __name__ == "__main__":
    run_benchmark(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        context_lengths=[2000, 4000],
        fault_rates=[0.0, 1e-5, 1e-4, 1e-3],
        num_trials=5,
    )
