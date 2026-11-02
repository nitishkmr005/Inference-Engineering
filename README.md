# LLM Inference Engineering

*A three-part course on what happens after `model.generate()`: what inference is, why it is slow, what it costs, and how it runs on a GPU. One model runs through every slide: **Qwen3.8-27B**.*

The course is three self-contained interactive HTML decks. Open any of them in Chrome; nothing to install.

| Part | Deck | Status |
|---|---|---|
| **1. Inference Foundations** | [`Part1_Inference_Foundations.html`](Part1_Inference_Foundations.html) | Complete: 39 slides, 6 chapters |
| **2. Inference Optimization** | [`Part2_Inference_Optimization.html`](Part2_Inference_Optimization.html) | Outline: title, contents and 9 chapter dividers |
| **3. Advanced Inference** | [`Part3_Advanced_Inference.html`](Part3_Advanced_Inference.html) | Outline: title, contents and 8 chapter dividers |

## Using the decks

| Key | Action |
|---|---|
| ← → | previous / next slide |
| Home / End | first / last slide |
| N | speaker notes |
| O | overview of all slides (Esc closes) |
| R | replay the current slide's animation |
| P | pause / resume animations (every animated slide also has a pause button) |

Link to a slide with `#sN`, e.g. `Part1_Inference_Foundations.html#s26`. On the contents slide, click a chapter to jump to it. Light theme only.

## Part 1. Inference Foundations

| Chapter | Slides | What it covers |
|---|---|---|
| **1. Inference at a glance** | 3–5 | What you will be able to answer; what inference is and the software that runs it |
| **2. The model: architecture and memory** | 6–14 | Qwen3.8-27B's hybrid architecture, how linear attention works, reading `config.json`, where the 27.8B parameters live, training vs inference memory, a memory calculator |
| **3. Prefill vs decode** | 15–20 | The KV cache, the two phases of every request, why they behave differently, four caches in a serving stack |
| **4. Metrics and cost** | 21–26 | TTFT, ITL vs TPOT, five ways to count throughput, hosting cost, and a worked example that derives TTFT, ITL, max users, throughput and $ per 1M tokens from the L40S spec sheet |
| **5. How inference runs on a GPU** | 27–32 | One AWS g6e.xlarge (L40S, 48 GB) from CPU to tensor cores; grids, blocks and warps; prefill and decode animated on the L40S; the memory hierarchy |
| **6. GPUs: hardware, NVIDIA, AWS** | 33–39 | The software stack, NVIDIA generations, reading a spec sheet, which specs move TTFT and ITL, product lines, AWS GPU instances |

**Every number is derived.** Each figure on a slide has its arithmetic on the slide or a source in the footer. Assumptions are labelled as such:

- 70% of peak memory bandwidth
- 50% of peak compute
- 3 ms of non-math work per step (scheduling, kernel launches, sampling)
- 90% of GPU memory usable, 1.5 GB for the runtime

They form a first-order model for estimating, not a benchmark. Check real numbers with vLLM's benchmark.

## Part 2 and Part 3 (outlines)

**Part 2, Inference Optimization:**
1. Where the time goes
2. Quantization in practice
3. KV cache management
4. Batching and scheduling
5. Inference frameworks compared
6. Speculative decoding basics
7. Multi-GPU on one machine
8. Benchmarking and running it
9. Savings around the model: caching, prompts and routing

**Part 3, Advanced Inference:**
1. Attention and GPU kernels
2. Advanced quantization and compression
3. Advanced speculative decoding
4. Long context and efficient architectures
5. Distributed inference and MoE
6. Disaggregated serving and KV tiers
7. Reasoning, agentic and multimodal workloads
8. Fleet scale and hardware beyond NVIDIA

## References

The reading list is in [`references.txt`](references.txt). The main sources cited in slide footers:

- Philip Kiely, *Inference Engineering*
- Lamhot Siagian, *Efficient LLM Inference*
- the Modular *LLM Inference Handbook*
- Sebastian Raschka's articles (architecture figure, KV cache intuition)
- NVIDIA datasheets, AWS instance and pricing pages
- the Qwen3.8-27B `config.json` on Hugging Face
