# SPDX-License-Identifier: Apache-2.0
"""
Math-Needle Benchmark: KV Cache Stress Test via Reasoning

This benchmark tests KV cache integrity by requiring the model to:
1. Retrieve multiple values hidden in long context
2. Perform arithmetic operations on those values

Key insight: Retrieval tests only detect corruption in the retrieved block,
but reasoning tests detect corruption in ANY block needed for computation.
This doubles (or more) sensitivity to KV cache faults.

Example:
- Context: 10,000 tokens of filler with "var_a = 50" at token 1000
           and "var_b = 25" at token 9000
- Retrieval: "What is var_a?" -> Passes if var_a block is clean
- Reasoning: "Compute var_a * var_b" -> Fails if EITHER block is corrupted
"""

import torch
import random
import time
import gc
import json
from typing import List, Dict, Tuple, Optional

# Filler text corpus (Wikipedia-style)
FILLER_PARAGRAPHS = [
    "The process of photosynthesis converts light energy into chemical energy stored in glucose molecules. Plants use chlorophyll in their leaves to absorb sunlight, which drives the reaction between carbon dioxide and water. This fundamental process sustains nearly all life on Earth by producing oxygen and organic compounds.",
    "Ancient civilizations developed sophisticated astronomical observations to track celestial bodies. The Mayans created accurate calendars based on Venus cycles, while Babylonians recorded planetary movements on clay tablets. These early astronomers laid the groundwork for modern understanding of our solar system.",
    "Neural networks consist of interconnected nodes organized in layers that process information. Each connection has an associated weight that is adjusted during training. Deep learning architectures with many hidden layers can learn hierarchical representations of complex data patterns.",
    "The water cycle describes the continuous movement of water within the Earth and atmosphere. Evaporation from oceans and lakes creates water vapor that rises and condenses into clouds. Precipitation returns water to the surface, where it flows through rivers back to the seas.",
    "Quantum mechanics describes the behavior of matter and energy at atomic scales. Particles exhibit wave-particle duality, existing in superposition states until measured. The uncertainty principle limits simultaneous knowledge of position and momentum.",
    "Economic markets operate through the interaction of supply and demand forces. When demand exceeds supply, prices tend to rise, incentivizing increased production. Market equilibrium occurs when the quantity supplied equals quantity demanded at a given price.",
    "The human immune system provides defense against pathogens through multiple mechanisms. Innate immunity offers immediate but non-specific protection, while adaptive immunity develops targeted responses. Memory cells enable faster responses to previously encountered threats.",
    "Continental drift theory explains how landmasses have moved over geological time. Tectonic plates float on the semi-fluid asthenosphere and interact at boundaries. Mountain ranges form where plates collide, while new crust emerges at spreading ridges.",
    "Machine learning algorithms identify patterns in data to make predictions or decisions. Supervised learning uses labeled examples to train models for classification or regression. Unsupervised methods discover hidden structures without predefined categories.",
    "The Renaissance marked a cultural rebirth in Europe spanning the 14th to 17th centuries. Artists like Leonardo da Vinci and Michelangelo revolutionized painting and sculpture. Scientific inquiry flourished as scholars questioned traditional assumptions about nature.",
]


def generate_filler_text(target_tokens: int, tokenizer) -> str:
    """Generate filler text of approximately target_tokens length."""
    paragraphs = []
    current_tokens = 0

    while current_tokens < target_tokens:
        para = random.choice(FILLER_PARAGRAPHS)
        paragraphs.append(para)
        current_tokens = len(tokenizer.encode(" ".join(paragraphs)))

    # Trim to approximate target
    text = " ".join(paragraphs)
    tokens = tokenizer.encode(text)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        text = tokenizer.decode(tokens)

    return text


def create_math_needle_prompt(
    context_length: int,
    var_positions: List[float],  # Positions as fractions (0.0 to 1.0)
    operation: str,  # "add", "multiply", "subtract"
    tokenizer,
    seed: int = 42
) -> Tuple[str, int, Dict]:
    """
    Create a prompt with hidden variables and a math question.

    Returns:
        prompt: The full prompt string
        expected_answer: The correct numerical answer
        metadata: Dict with variable values and positions
    """
    random.seed(seed)

    # Generate random variable values (keep them simple for reliable parsing)
    var_values = {
        "var_a": random.randint(10, 99),
        "var_b": random.randint(10, 99),
        "var_c": random.randint(10, 99),
    }

    # Calculate expected answer
    if operation == "add":
        expected = var_values["var_a"] + var_values["var_b"]
        question = "Calculate var_a + var_b. Give only the number."
    elif operation == "multiply":
        expected = var_values["var_a"] * var_values["var_b"]
        question = "Calculate var_a * var_b. Give only the number."
    elif operation == "subtract":
        expected = var_values["var_a"] - var_values["var_b"]
        question = "Calculate var_a - var_b. Give only the number."
    elif operation == "chain":
        expected = (var_values["var_a"] + var_values["var_b"]) * var_values["var_c"]
        question = "Calculate (var_a + var_b) * var_c. Give only the number."
    else:
        raise ValueError(f"Unknown operation: {operation}")

    # Generate filler and insert variables
    total_filler = context_length - 200  # Reserve space for vars and question

    # Split filler into sections based on var_positions
    sections = []
    var_statements = [
        f"\n\n[IMPORTANT DATA] var_a = {var_values['var_a']}\n\n",
        f"\n\n[IMPORTANT DATA] var_b = {var_values['var_b']}\n\n",
        f"\n\n[IMPORTANT DATA] var_c = {var_values['var_c']}\n\n",
    ]

    num_vars = len(var_positions)
    positions_tokens = [int(p * total_filler) for p in var_positions]

    # Generate filler text
    full_filler = generate_filler_text(total_filler, tokenizer)
    filler_tokens = tokenizer.encode(full_filler)

    # Insert variables at specified positions
    result_tokens = []
    last_pos = 0

    for i, pos in enumerate(positions_tokens):
        if i < len(var_statements):
            # Add filler up to this position
            result_tokens.extend(filler_tokens[last_pos:pos])
            # Add variable statement
            var_tokens = tokenizer.encode(var_statements[i])
            result_tokens.extend(var_tokens)
            last_pos = pos

    # Add remaining filler
    result_tokens.extend(filler_tokens[last_pos:])

    # Convert back to text and add question
    context = tokenizer.decode(result_tokens)
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
    # Look for integers in the output
    matches = re.findall(r'-?\d+', text.strip())
    if matches:
        try:
            return int(matches[0])
        except ValueError:
            return None
    return None


def run_math_needle_benchmark(
    model_name: str = "gpt2",
    context_lengths: List[int] = [500, 800],  # GPT-2 max is 1024
    fault_rates: List[float] = [0.0, 1e-5, 1e-3, 0.1],
    num_trials: int = 5,
    seed: int = 42,
):
    """
    Run the Math-Needle benchmark across different context lengths and fault rates.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load tokenizer for context generation
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = []

    print("=" * 70)
    print("MATH-NEEDLE BENCHMARK: KV Cache Reasoning Stress Test")
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

            # Create LLM with or without fault injection
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

                # Run trials
                correct_retrieval = 0
                correct_reasoning = 0

                for trial in range(num_trials):
                    trial_seed = seed + trial

                    # Test 1: Simple Retrieval (single variable)
                    prompt_ret, expected_ret, meta_ret = create_math_needle_prompt(
                        context_length=ctx_len,
                        var_positions=[0.5],  # var_a at middle
                        operation="add",  # We'll just ask for var_a
                        tokenizer=tokenizer,
                        seed=trial_seed,
                    )
                    # Modify question for retrieval
                    prompt_ret = prompt_ret.replace(
                        "Calculate var_a + var_b. Give only the number.",
                        "What is the value of var_a? Give only the number."
                    )
                    expected_ret = meta_ret["var_values"]["var_a"]

                    outputs = llm.generate([prompt_ret], params)
                    answer = extract_number(outputs[0].outputs[0].text)
                    if answer == expected_ret:
                        correct_retrieval += 1

                    # Test 2: Reasoning (requires both variables)
                    prompt_reason, expected_reason, meta_reason = create_math_needle_prompt(
                        context_length=ctx_len,
                        var_positions=[0.2, 0.8],  # var_a early, var_b late
                        operation="multiply",
                        tokenizer=tokenizer,
                        seed=trial_seed + 1000,
                    )

                    outputs = llm.generate([prompt_reason], params)
                    answer = extract_number(outputs[0].outputs[0].text)
                    if answer == expected_reason:
                        correct_reasoning += 1

                    # Debug output for first trial
                    if trial == 0:
                        print(f"\n  Trial 0 - Retrieval:")
                        print(f"    Expected: {expected_ret}, Got: {answer}")
                        print(f"  Trial 0 - Reasoning:")
                        a, b = meta_reason["var_values"]["var_a"], meta_reason["var_values"]["var_b"]
                        print(f"    var_a={a}, var_b={b}, Expected: {expected_reason}")
                        print(f"    Model output: {outputs[0].outputs[0].text[:50]}")

                retrieval_acc = correct_retrieval / num_trials
                reasoning_acc = correct_reasoning / num_trials

                print(f"\n  RESULTS:")
                print(f"    Retrieval Accuracy:  {retrieval_acc:.0%} ({correct_retrieval}/{num_trials})")
                print(f"    Reasoning Accuracy:  {reasoning_acc:.0%} ({correct_reasoning}/{num_trials})")
                print(f"    Sensitivity Ratio:   {(1-reasoning_acc)/(1-retrieval_acc+0.001):.2f}x")

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
                    "trials": num_trials,
                    "error": str(e)[:100],
                })

            cleanup()

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT: Math-Needle Benchmark")
    print("=" * 70)
    print(f"{'Context':<10} {'Fault Rate':<12} {'Retrieval':<12} {'Reasoning':<12} {'Sensitivity':<12}")
    print("-" * 70)

    for r in results:
        ctx = r["context_length"]
        rate = r["fault_rate"]
        ret = r["retrieval_accuracy"]
        reas = r["reasoning_accuracy"]
        # Sensitivity: how much more reasoning fails compared to retrieval
        sens = (1 - reas) / (1 - ret + 0.001) if ret < 1.0 else 1.0
        print(f"{ctx:<10} {rate:<12.0e} {ret:<12.0%} {reas:<12.0%} {sens:<12.2f}x")

    print("\nKey Insight: Reasoning tasks should show HIGHER failure rates than")
    print("retrieval tasks at the same fault rate, because they require")
    print("multiple KV cache blocks to be correct.")

    return results


def run_long_context_benchmark(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    context_length: int = 8000,
    fault_rates: List[float] = [0.0, 1e-6, 1e-5, 1e-4],
    num_trials: int = 10,
):
    """
    Run extended Math-Needle benchmark with long context models.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("=" * 70)
    print("LONG-CONTEXT MATH-NEEDLE BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Context: {context_length} tokens")
    print(f"Testing chain reasoning: (var_a + var_b) * var_c")
    print(f"Variables placed at 10%, 50%, 90% of context")
    print()

    results = []

    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()

    for fault_rate in fault_rates:
        print(f"\n{'='*70}")
        print(f"Fault Rate: {fault_rate:.0e}")
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
                    fault_injection_seed=42,
                )

            params = SamplingParams(temperature=0.0, max_tokens=30)

            correct = 0
            for trial in range(num_trials):
                prompt, expected, meta = create_math_needle_prompt(
                    context_length=context_length,
                    var_positions=[0.1, 0.5, 0.9],  # 3 variables spread across context
                    operation="chain",  # (a + b) * c
                    tokenizer=tokenizer,
                    seed=42 + trial,
                )

                outputs = llm.generate([prompt], params)
                answer = extract_number(outputs[0].outputs[0].text)

                if answer == expected:
                    correct += 1
                elif trial == 0:
                    print(f"  Trial 0 mismatch:")
                    print(f"    var_a={meta['var_values']['var_a']}, var_b={meta['var_values']['var_b']}, var_c={meta['var_values']['var_c']}")
                    print(f"    Expected: {expected}, Got: {answer}")
                    print(f"    Output: {outputs[0].outputs[0].text[:80]}")

            accuracy = correct / num_trials
            print(f"  Accuracy: {accuracy:.0%} ({correct}/{num_trials})")

            results.append({
                "fault_rate": fault_rate,
                "accuracy": accuracy,
                "context_length": context_length,
            })

            del llm

        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")
            results.append({
                "fault_rate": fault_rate,
                "accuracy": 0.0,
                "error": str(e)[:100],
            })

        cleanup()

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "long":
        # Long context benchmark with LLaMA
        run_long_context_benchmark()
    else:
        # Quick GPT-2 benchmark
        run_math_needle_benchmark(
            model_name="gpt2",
            context_lengths=[400, 700],
            fault_rates=[0.0, 1e-4, 1e-2, 0.1],
            num_trials=5,
        )
