# MLX + Apple Silicon (M-Chips) — Practical Notes & Quirks

## Core Architecture

Apple Silicon uses:

- Unified Memory Architecture (UMA)
- Shared CPU/GPU memory pool
- Metal instead of CUDA
- High memory bandwidth
- Low discrete VRAM overhead

This fundamentally changes LLM inference behavior compared to NVIDIA systems.

---

# Unified Memory (UMA)

## What it means

CPU and GPU share the same RAM.

Not:

```text
CPU RAM + GPU VRAM
```

But:

```text
One shared memory pool
```

---

# Implications

## GOOD

- No PCIe transfer bottleneck
- Huge models possible
- Faster tensor sharing
- Simpler memory ownership

## BAD

- Browser tabs compete with LLMs
- GPU memory pressure affects entire system
- Swap destroys inference latency
- KV cache can consume all system RAM

---

# Main Bottleneck on M-Chips

Usually NOT compute.

Usually:

```text
Memory bandwidth
```

LLM inference on Apple Silicon is heavily memory-bound.

---

# KV Cache

## What it is

Transformer stores:

- K (keys)
- V (values)

for previous tokens.

This prevents recomputing full history every generation step.

---

# Why KV cache matters

KV cache grows linearly with:

- Context length
- Batch size
- Number of sequences
- Number of layers
- Model size

---

# Why long context kills Macs

Usually NOT model weights.

Usually:

```text
KV cache explosion
```

Example:

```text
7B Q4 weights:
~5GB

32k KV cache:
10GB+
```

---

# Symptoms of KV pressure

- Huge TTFT
- Random latency spikes
- macOS swap
- Memory compression
- Metal stalls
- System-wide slowdown

---

# Best Practices for Context Length

## Recommended

```json
"context_length": 2048
```

or:

```json
"context_length": 4096
```

---

# Avoid

```json
"context_length": 32768
```

unless absolutely necessary.

---

# Prefill

## Definition

Prefill = processing input prompt before generation.

This stage:

- tokenizes input
- runs attention
- builds KV cache

---

# Prefill Characteristics

## Prefill is:

- compute-heavy
- highly parallel
- bandwidth-intensive

---

# Long prompts

Long RAG chunks dramatically increase:

- TTFT
- temporary activations
- memory pressure

---

# Decode

## Definition

Decode = token-by-token generation.

---

# Decode Characteristics

## Decode is:

- sequential
- latency-sensitive
- memory-bound

---

# TTFT (Time To First Token)

## Formula intuition

```text
TTFT ≈
tokenization
+ prefill
+ KV allocation
+ Metal scheduling
+ first decode step
```

---

# Biggest TTFT killers on Apple Silicon

## 1. Cold model loading

- tokenizer init
- Metal kernel compilation
- graph initialization

---

## 2. Huge context length

- large KV reservation
- memory fragmentation

---

## 3. Long prompts

- prefill explosion

---

## 4. Large max_tokens

- larger decode planning
- larger KV growth

---

# Warmup Strategy

After startup:

```python
"hello"
```

or:

```python
"test"
```

This warms:

- Metal kernels
- tokenizer
- graph execution
- memory pools

---

# Typical improvement

Cold:

```text
3-10 seconds TTFT
```

Warm:

```text
150-700 ms TTFT
```

---

# Quantization on Apple Silicon

Quantization helps MORE on Apple than many NVIDIA systems.

Why:

```text
Reduced memory bandwidth pressure
```

---

# Recommended Quantization

## Best practical range

```text
Q4 / Q5
```

---

# Avoid

## Q8

Usually too memory-heavy for marginal gains.

## Q2

Often quality collapse.

---

# Batching

## Definition

Processing multiple requests simultaneously.

---

# Apple-specific batching reality

Apple systems generally prefer:

- smaller batches
- lower concurrency
- lower KV pressure

compared to datacenter GPUs.

---

# disable_batching

## Recommended for low latency

```json
"disable_batching": true
```

Good for:

- extraction
- RAG
- local assistant
- single-user inference

---

# batch_prefill_size

## Recommended

```json
"batch_prefill_size": 1
```

Higher values increase:

- RAM spikes
- Metal scheduling overhead
- memory contention

---

# batch_completion_size

## Recommended

```json
"batch_completion_size": 1
```

Higher values improve throughput but worsen:

- latency
- memory pressure

---

# batch_prefill_step_size

## Recommended

```json
"batch_prefill_step_size": 512
```

or:

```json
"batch_prefill_step_size": 1024
```

Smaller step sizes:

- reduce peak memory spikes
- improve responsiveness
- reduce Metal stalls

---

# max_tokens

## Recommendation

For extraction:

```json
"default_max_tokens": 64
```

or:

```json
"default_max_tokens": 128
```

---

# Why large max_tokens are bad

Large outputs:

- increase decode time
- increase KV cache
- increase hallucinations

---

# Prompt Cache

## Useful for repeated prompts

Good for:

- RAG
- extraction pipelines
- stable system prompts

---

# Recommended prompt cache

```json
"prompt_cache_size": 8,
"prompt_cache_max_bytes": 1073741824
```

---

# Avoid oversized prompt cache

Large cache:

- increases unified memory pressure
- increases eviction overhead

---

# on_demand

## Low-latency recommendation

```json
"on_demand": false
```

---

# Why

Cold loading causes:

- model load delay
- Metal recompilation
- tokenizer startup

---

# Resident Models

Small models (~3B Q4) should stay permanently loaded.

Apple Silicon handles this well.

---

# Recommended Low-Latency Config (3B Q4)

```json
{
  "context_length": 2048,
  "batch_completion_size": 1,
  "batch_prefill_size": 1,
  "batch_prefill_step_size": 512,
  "default_max_tokens": 128,
  "default_temperature": 0.0,
  "prompt_cache_size": 8,
  "prompt_cache_max_bytes": 1073741824,
  "disable_batching": true,
  "on_demand": false
}
```

---

# SwiftLM Gemma 4 E4B Benchmark Notes

These measurements were taken on the local Apple Silicon SwiftLM runtime using:

- `mlx-community/gemma-4-e4b-it-4bit`
- `parallel=1`
- `max_tokens=96`
- the benchmark harness at `scripts/bench_swiftlm_gemma4_e4b.py`

The useful knobs for this dense Gemma profile were:

- `ctx-size`
- `prefill-size`
- `turbo-kv`

The matrix result was:

| Profile | Prompt tokens | Prefill s | Decode tok/s | Peak GPU in-use GB | Note |
|---|---:|---:|---:|---:|---|
| `compact-4k-p256` | 2048 | 2.66 | 23.62 | 6.00 | Best performance per GB |
| `balanced-8k-p512` | 2048 | 2.56 | 24.28 | 7.25 | Best short-prompt throughput among 8k profiles |
| `balanced-8k-p512` | 6144 | 4.65 | 15.04 | 6.71 | Best 6k prompt profile |
| `fast-8k-p1024` | 2048 | 3.14 | 21.14 | 6.74 | Worse than prefill 512 on latency and throughput |
| `long-16k-p512` | 12000 | 8.00 | 9.57 | 7.34 | Best long-context option tested |
| `long-16k-p512-turbo` | 12000 | 14.56 | 4.26 | 6.10 | Lower memory, but much worse latency and throughput |

Practical takeaways:

- Use `SWIFT_LM_CONTEXT_SIZE=8192` and `SWIFT_LM_PREFILL_SIZE=512` as the default interactive profile.
- Use `SWIFT_LM_CONTEXT_SIZE=4096` and `SWIFT_LM_PREFILL_SIZE=256` only when memory pressure matters more than long-context headroom.
- Use `SWIFT_LM_CONTEXT_SIZE=16384` and `SWIFT_LM_PREFILL_SIZE=512` for long prompts around 12k tokens.
- Leave `SWIFT_LM_TURBO_KV=false` for this model. It regressed both TTFT and decode speed in every measured long-context run.
- Keep `SWIFT_LM_PARALLEL=1` when optimizing single-request latency. Raising concurrency is a different tradeoff and should be measured separately.

Recommended env profiles:

```dotenv
# Default interactive profile
SWIFT_LM_CONTEXT_SIZE=8192
SWIFT_LM_PREFILL_SIZE=512
SWIFT_LM_PARALLEL=1
SWIFT_LM_TURBO_KV=false
```

```dotenv
# Memory-saving profile
SWIFT_LM_CONTEXT_SIZE=4096
SWIFT_LM_PREFILL_SIZE=256
SWIFT_LM_PARALLEL=1
SWIFT_LM_TURBO_KV=false
```

```dotenv
# Long-context profile
SWIFT_LM_CONTEXT_SIZE=16384
SWIFT_LM_PREFILL_SIZE=512
SWIFT_LM_PARALLEL=1
SWIFT_LM_TURBO_KV=false
```

---

# Extraction-Specific Recommendations

## Use tiny prompts

GOOD:

```text
Extract:
Person
Organization
Concept

Output:
name|type
```

BAD:

- verbose instructions
- chain-of-thought
- giant schemas
- reasoning prompts

---

# Why small prompts matter

Smaller prompts:

- reduce prefill
- reduce TTFT
- reduce KV cache growth

---

# Apple Silicon Failure Modes

## Common issues

### Swap storms

Symptoms:

- system freeze
- 10-100x slowdown
- random latency spikes

---

### Memory compression thrashing

Symptoms:

- inconsistent performance
- jittery decode speed

---

### Metal kernel stalls

Symptoms:

- random pauses
- TTFT spikes

---

### Unified memory starvation

Symptoms:

- browser becomes slow
- inference collapses
- system UI lag

---

# Practical Advice

## DO

- use Q4/Q5
- keep context small
- keep prompts short
- keep max_tokens low
- preload models
- use warmup requests

---

## DON'T

- run huge contexts
- run giant batches
- keep Chrome open with 200 tabs
- use reasoning models for extraction
- use huge max_tokens for KG extraction

---

# Good Workloads for MLX

- RAG
- extraction
- embeddings
- local coding assistant
- chat inference
- lightweight agents

---

# Bad Workloads for MLX

- massive multi-user serving
- giant continuous batching
- ultra-long context reasoning
- heavy speculative decoding systems

---

# Golden Rules

## Long prompts kill prefill

## Long outputs kill decode

## Large batches kill memory

## KV cache kills everything
