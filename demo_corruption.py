# SPDX-License-Identifier: Apache-2.0
# demo_corruption.py - Demonstrate fault injection corruption
import torch
import gc

MODEL_NAME = 'gpt2'
PROMPT = 'The capital of France is'

from vllm import LLM, SamplingParams
params = SamplingParams(temperature=0.0, max_tokens=30, ignore_eos=True)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

print('='*60)
print('CORRUPTION DEMONSTRATION - High Fault Rate (50%)')
print('='*60)

# Baseline
cleanup()
llm = LLM(model=MODEL_NAME, enforce_eager=True)
outputs = llm.generate([PROMPT], params)
baseline = outputs[0].outputs[0].text
print(f'Baseline:  {baseline}')
del llm
cleanup()

# High fault rate (50%)
llm = LLM(
    model=MODEL_NAME,
    enforce_eager=True,
    fault_injection_enabled=True,
    fault_injection_site='KV_WRITE',
    fault_injection_model='random',
    fault_injection_rate=0.5,
    fault_injection_seed=42
)
outputs = llm.generate([PROMPT], params)
faulted = outputs[0].outputs[0].text
print(f'Faulted:   {faulted}')
del llm
cleanup()

# Verify difference
if baseline == faulted:
    print('\nRESULT: Outputs are identical')
else:
    diff_count = sum(1 for a, b in zip(baseline, faulted) if a != b)
    print(f'\nRESULT: CORRUPTION DETECTED!')
    print(f'  {diff_count} characters differ out of {min(len(baseline), len(faulted))}')
