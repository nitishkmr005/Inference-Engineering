# LLM Inference Engineering: Talk Materials

| File | What it is | How to use |
|---|---|---|
| `01_LLM_Inference_Presentation.html` | Light-theme deck, 41 slides, with interactive memory calculator, roofline, batching/paging animations, memory-hierarchy ladder and a "will it fit on my machine?" planner | Open in Chrome. `→`/Space next, `←` back, `F` fullscreen. Add `?all` to the URL to show every step at once. |
| `02_LLM_Inference_Engineering.pptx` | Same 41 slides as an editable PowerPoint, speaker notes on every slide | Share with the team / present from PowerPoint |
| `03_vLLM_Hands_On.ipynb` | Hands-on notebook, 35 sections: vLLM → SGLang → TensorRT-LLM → hardware → local big-model hosting | Run on a GPU. Use one virtualenv/container per engine; the notebook launches each engine as a subprocess. Set `SGLANG_PY`, `TRTLLM_PY`, `TRTLLM_SERVE` to point at those envs. |

The HTML deck loads fonts from Google Fonts; offline it falls back to system fonts.

## Deck structure (slide → notebook)

| Part | Topics | Notebook |
|---|---|---|
| Intro | Why / what / how · topic map · where this shows up for data scientists | — |
| 1 · Mental model | Prefill vs decode · TTFT, TPOT, goodput · memory math · roofline · decode speed limit & MBU · compute/bandwidth/overhead · GPU memory hierarchy · hardware cheat sheet (H100 → DGX Spark) | §2–3, §8, §30 |
| 2 · Toolkit | KV cache · continuous batching · PagedAttention · prefix caching · chunked prefill · kernels · quantization | §4–7, §12, §16–17 |
| 3 · Scaling up | Speculative decoding · parallelism & MoE · GQA/MLA · disaggregation | §18, §20 |
| 4 · Engines | vLLM V1 anatomy · picking an engine · flags · vLLM vs SGLang vs TensorRT-LLM commands | §9–15, §22–29 |
| 5 · Local hosting | Fit/speed planner · llama.cpp, MLX, Ollama, exo, KTransformers · MoE expert offload · edge cases · practitioner reports | §31–35 |
| 6 · Running it | Cost per 1M tokens · monitoring · tuning playbook · reading list · exercises | §13, §21 |

## Main references
- *Inference Engineering*, Philip Kiely (Baseten). Free at baseten.co/inference-engineering
- Databricks, *LLM Inference Performance Engineering: Best Practices* · kipply, *Transformer Inference Arithmetic* · Horace He, *Making Deep Learning Go Brrrr*
- P. Lienhart, *LLM Inference Series* (Medium) · Hugging Face, *Continuous batching from first principles* · Lilian Weng, *Large Transformer Model Inference Optimization*
- Modal, *LLM Engine Almanac* · SemiAnalysis InferenceX · Chip Huyen, P99 CONF 2026 keynote
- vLLM, SGLang, TensorRT-LLM, llama.cpp and mlx-lm documentation
- Papers: Orca, PagedAttention, FlashAttention, Sarathi-Serve, DistServe, EAGLE-3, DeepSeek-V3
