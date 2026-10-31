# LLM Inference Engineering

*A three-session course for an AI team: what happens after `model.generate()`, why it is slow, and how vLLM and friends make it fast, from one laptop to a rack of GPUs.*

Every technique is shown twice: first as a small piece of code you can read in a minute (a scheduler, a quantizer, a speculative decoder, a tensor-parallel split), then in a real engine with a measurement. The slides tell the story; the notebooks let you check it.

| Session | The question it answers | Slides | Notebook |
|---|---|---|---|
| **1. Basics** | What happens between a prompt and its answer, and why is it slow? | [`slides/Session1_Basics.pptx`](slides/Session1_Basics.pptx) (21) | [`notebooks/Session1_Basics.ipynb`](notebooks/Session1_Basics.ipynb) (16 sections) |
| **2. Intermediate** | Which tricks make serving fast, and what does each one actually do? | [`slides/Session2_Intermediate.pptx`](slides/Session2_Intermediate.pptx) (23) | [`notebooks/Session2_Intermediate.ipynb`](notebooks/Session2_Intermediate.ipynb) (28 sections) |
| **3. Advanced** | How do you serve a model bigger than one GPU, reliably, for many users? | [`slides/Session3_Advanced.pptx`](slides/Session3_Advanced.pptx) (22) | [`notebooks/Session3_Advanced.ipynb`](notebooks/Session3_Advanced.ipynb) (27 sections) |

The first slide of each deck shows this split, with the current session highlighted. Every slide has speaker notes, and a tag in the top-right corner points to the notebook section that reproduces it (`Notebook 2: §9`).

**Also here:** the original single-talk version of the course. [`01_LLM_Inference_Presentation.html`](01_LLM_Inference_Presentation.html) is a 42-slide interactive deck with a memory calculator, roofline, batching and paging animations, and a "will it fit on my machine?" planner. Open it in Chrome; arrow keys move between slides, and `?all` shows every step. It comes with [`02_LLM_Inference_Engineering.pptx`](02_LLM_Inference_Engineering.pptx) and the 40-section [`03_vLLM_Hands_On.ipynb`](03_vLLM_Hands_On.ipynb). The three sessions cover everything in those files and add the from-scratch code.

---

## Running the notebooks

```bash
pip install vllm openai matplotlib pandas transformers huggingface_hub pyarrow
jupyter lab notebooks/
```

- Keep `infer_utils.py` next to the notebooks. It holds the shared helpers (`vllm_serve`, `stop`, `measure_stream`, `bench_openai`, `gpu_mem`).
- Run the setup cell first. After that, each section starts and stops its own engine, so you can jump straight to the section you need.
- Labels on section headings:
  - *(no GPU needed)* or *(runs on CPU)*: works on a laptop;
  - *(reading)*: commands for hardware you may not have;
  - unlabelled: needs one NVIDIA GPU with 16 GB or more.
- The default model is `Qwen/Qwen2.5-1.5B-Instruct`. To use another, set `NB_MODEL` before starting Jupyter.
- SGLang and TensorRT-LLM pin different PyTorch and CUDA versions, so give each its own virtualenv or container. Point the notebook at them with `SGLANG_PY`, `TRTLLM_PY` and `TRTLLM_SERVE`.

> **What has been verified.** Every *(no GPU needed)* and *(runs on CPU)* section was executed end to end on an Apple M4 Pro, and the numbers quoted below come from those runs. The GPU sections were checked against vLLM's source and recipes (October 2026), but they have not been executed here because there is no NVIDIA GPU on this machine. Expect to fix a flag or two when you run them on a GPU.

---

## Session 1. Basics: how LLM inference works, and what limits it

The mental model every later session builds on. By the end, you can predict a model's decode speed and how many users fit on a GPU, before you run anything.

### Part 1. How a model generates text
- **The generation loop, by hand** (slide 6, notebook §3). Prefill is one forward pass over the whole prompt. Decode is one pass per new token, reusing the cached keys and values. On a laptop CPU with Qwen2.5-0.5B, decoding took **17.8 ms per token with the KV cache and 52.2 ms without it**.
- **Tokens** (§2): what the model actually sees, and why token counts rather than words drive cost.
- **Prefill vs decode** (slide 7). Prefill is compute-bound and sets time to first token. Decode is bandwidth-bound and sets time per output token.
- **The KV cache** (slides 6, 8). It trades memory for compute. This 0.5B model stores 12 KiB per token: 24 layers × 2 KV heads × 64 dims × K and V × 2 bytes.
- **Four numbers describe a server** (slide 9): TTFT, TPOT, throughput and goodput (throughput that meets the SLO).
- **Where the memory goes** (slide 10, §4). Weights are fixed and the KV cache grows. Llama-3.1-8B at 64 concurrent 8K-token chats needs four times as much memory for KV cache as for weights.

### Part 2. The GPU underneath
- **Memory hierarchy** (slide 12, §5): SRAM, L2, HBM, NVLink, CPU DRAM, PCIe and SSD, each roughly 3 to 10× slower than the one before.
- **The roofline** (slide 13). An H100 needs about 295 FLOPs per byte to stay compute-bound, and decode at batch 1 does about 1, which is why batching is nearly free.
- **The decode speed limit and MBU** (slide 14, §12): tokens/s ≤ bandwidth ÷ bytes read per token. Model bandwidth utilisation (MBU) measures how close a real engine gets.
- **Compute, bandwidth, overhead** (slide 15): Horace He's three regimes, and the fixes that belong to each.
- **Hardware cheat sheet** (slide 16): H100, H200, B200, MI300X, RTX 4090/5090, Mac M3 Ultra and DGX Spark, with the decode ceiling for each.

### Part 3. First steps with vLLM
- **Inside vLLM's V1 engine** (slide 18): API server, EngineCore (scheduler plus KV cache manager), and workers.
- **The offline `LLM` API** (§6–11): generation, the three configuration layers, sampling parameters, `llm.chat`, prompt formats (including pre-filling the answer), and batched vs sequential throughput.
- **The server** (§13–16): `vllm serve`, streaming with TTFT and TPOT measured by hand, a concurrency sweep showing the latency–throughput trade-off, and reading `/metrics`.
- **The eight flags that cover most tuning** (slide 19) and **cost per million tokens** (slide 20).

**Exercises:** predict and then measure the decode ceiling; find your goodput; size an 8K-context deployment; break the cache.

---

## Session 2. Intermediate: the optimization toolkit every engine implements

Each technique answers one of Session 1's two facts: decode leaves compute idle, and the KV cache fills memory.

### Part 1. Scheduling and memory
- **Continuous batching from scratch** (slides 5–6, notebook §1). A 30-line scheduler simulation; the only difference between the two policies is when requests are admitted. With 64 requests and 16 slots:

  | | Static | Continuous |
  |---|---|---|
  | Throughput | 183 tok/s | **324 tok/s** |
  | Latency p50 | 16.7 s | **6.2 s** |
  | Latency p99 | 31.6 s | **17.3 s** |

- **PagedAttention from scratch** (slide 7, §2). A block allocator with hash-based prefix sharing. Reserving the maximum length per request used 52% of the memory it held, against 99.6% with paging.
- **Prefix caching** (slide 8, §3–4): cold vs warm TTFT, resetting the cache, isolating tenants, and reading the hit rate.
- **KV cache capacity and engine knobs** (§5), and **chunked prefill measured live** (slide 9, §6).

### Part 2. Making each step cheaper
- **Kernels** (slide 11, §7–8): FlashAttention, kernel fusion, CUDA graphs and `torch.compile`, plus a measured roofline and FlashAttention against naive attention.
- **Quantization by hand** (slide 12, §9). Every method is written out in a few lines: absmax INT8 per tensor and per channel, zero-point UINT8, group-wise INT4, NF4, FP8 E4M3, AWQ (activation-aware scaling with an α search) and GPTQ (Hessian-based error compensation). Relative error on one real layer of Qwen2.5-0.5B:

  | Method | Bits | Error |
  |---|---|---|
  | INT8 per-tensor / per-channel | 8 | 0.059 / 0.014 |
  | FP8 E4M3 | 8 | 0.027 |
  | INT4 per-tensor / group 128 | 4 | 0.849 / 0.134 |
  | NF4 group 64 | 4 | 0.095 |

- **Choosing a format** (slide 13). W4A16 helps decode; W8A8 FP8 also helps prefill; NVFP4/MXFP4 need Blackwell; an FP8 KV cache doubles capacity.
- **Every method, measured on the whole model** (slide 14). WikiText-2 perplexity:
  - original 14.25;
  - INT8 per-channel 14.21;
  - FP8 14.28;
  - NF4 16.28;
  - INT4 group 128 19.50;
  - INT4 per-tensor 3.9 × 10⁷.

  For one layer's output error, GPTQ (0.110) beats AWQ (0.121) and round-to-nearest (0.122).
- **In vLLM** (§10, §17): BF16 against FP8, measured, then quantizing your own checkpoint with llm-compressor (`FP8_DYNAMIC`) and checking what it cost.

### Part 3. More than one token per step
- **Speculative decoding from scratch** (slides 16–17, §11).
  - Greedy, with Qwen2.5-0.5B drafting for 1.5B: **identical output, 48 target passes cut to 13, 67% acceptance**.
  - With sampling, the min(1, p/q) acceptance rule plus residual resampling reproduces the target distribution exactly; this was checked on 50,000 draws.
  - When it pays: speed-up = E / (k × c + 1), where E = (1 − aᵏ⁺¹)/(1 − a). Acceptance matters more than draft length.
- **In vLLM** (§12): n-gram speculation, EAGLE-3 and MTP.

### Part 4. Engines and features
- **vLLM vs SGLang vs TensorRT-LLM** (slides 19–20, §23–27). The commands for each; SGLang's RadixAttention and `fork`; one benchmark harness for any engine.
- **More than chat** (slide 21, §13–21):
  - structured outputs (JSON schema, regex, choice, grammar);
  - prompt-logprob scoring and perplexity;
  - beam search;
  - embeddings, reranking and classification;
  - zero-shot labelling at scale;
  - multi-LoRA serving with runtime hot-loading.
- **`vllm bench serve`** (§22) and **the tuning playbook** (slide 22, §28): from symptom to the lever to pull.

**Exercises:** tune the scheduler; break prefix caching; quantize harder; make speculative decoding lose; rank your own data by perplexity.

---

## Session 3. Advanced: scaling out and running in production

### Part 1. More than one GPU
- **How GPUs connect** (slide 5, §2): NVLink and NVSwitch (900 GB/s), PCIe Gen5 (64 GB/s), InfiniBand NDR (50 GB/s); `nvidia-smi topo -m`; a `torchrun` NCCL all-reduce benchmark; the NCCL variables you will actually set. Estimated TP=8 all-reduce time per decode step for a 70B model at batch 64: **0.65 ms on NVLink, 5.9 ms on InfiniBand, 9.2 ms on PCIe**, against a 10 to 15 ms step.
- **Parallelism from scratch** (slides 6–7, §1). Tensor parallel is a column split, a row split and one all-reduce, and it matches a single matmul. Pipeline parallel shown with its bubble: 75%, 43% and 16% idle at 1, 4 and 16 micro-batches. Expert parallel shown with its routing imbalance.
- **Which parallelism** (slides 8–9, §4): DP for throughput, TP inside a node, PP across nodes or over PCIe, DP attention plus EP for big MoEs, decode context parallel for long contexts.
- **Big models, real commands** (slide 10, §5–6): Llama 3.3 70B, Qwen3-235B, DeepSeek-V3.2 (TP and DP+EP layouts, MTP), Kimi K2, and two nodes with `--nnodes/--headless` or Ray.
- **Architecture choices are inference choices** (slide 11): MHA, GQA, MQA and MLA, and KV bytes per token.

### Part 2. Splitting the work further
- **Disaggregated prefill and decode** (slide 13, §7): when it pays and when chunked prefill is enough; KV offload (native and LMCache) and NIXL.
- **TensorRT-LLM** (§8–10): the offline API, `trtllm-serve`, and a head-to-head comparison on your GPU.

### Part 3. Hosting open-weight models
- **Read the recipe first** (slide 15, §11–15). Tool-call and reasoning parsers per model family; tool calling, reasoning output, vision-language models and gpt-oss, all runnable.
- **Will it fit on my machine** (slides 16–19, §23–27): a planner that reads any Hugging Face config; llama.cpp, MLX, Ollama, exo and KTransformers; MoE expert offload; the errors you will hit; what practitioners report.
- **Production** (slide 20, §16–22):
  - Docker, config files and Kubernetes probes;
  - priority scheduling;
  - the alerts to page on;
  - a launch checklist;
  - sleep mode;
  - `AsyncLLM`;
  - `apply_model` and `collective_rpc`;
  - batch jobs with `vllm run-batch`;
  - profiling.

**Exercises:** an uneven TP split; measuring your links; TP or DP; planning a model you want; writing the runbook.

---

## Reading list

**Start here**
- Philip Kiely, *Inference Engineering* (Baseten). The map of the whole field, free online.
- Databricks, *LLM Inference Performance Engineering: Best Practices*. Metrics, MBU and honest benchmarking.
- kipply, *Transformer Inference Arithmetic*. Latency and memory worked out from first principles.
- Horace He, *Making Deep Learning Go Brrrr From First Principles*. The compute, bandwidth and overhead lens.

**Go deeper**
- Pierre Lienhart, *LLM Inference Series* (Medium). The KV cache, step by step.
- Aleksa Gordić, *Inside vLLM*. Engine internals with diagrams.
- Hugging Face, *Continuous batching from first principles*.
- Lilian Weng, *Large Transformer Model Inference Optimization*.
- Modal, *LLM Engine Almanac*; SemiAnalysis, *InferenceX*.
- The vLLM recipes (`recipes/` in the vLLM repo) and the vLLM blog. These are where launch-day flags for new open-weight models appear first.

**Papers:** Orca (OSDI '22), PagedAttention (SOSP '23), FlashAttention 1–3, Sarathi-Serve and DistServe (OSDI '24), Speculative Decoding (Leviathan et al., 2023), EAGLE-3 (2025), AWQ (2023), GPTQ (2022), Megatron-LM (2019), DeepSeek-V3 (2024).
