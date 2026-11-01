# References for the visual slide revision

The three decks use original, editable diagrams. The explanations combine the two PDFs in `docs/`, the supplied reading list, and primary sources checked on **31 October 2026**. Article illustrations were used as explanatory references; their artwork was not copied into the decks.

## Local foundations

| Source | Used for |
|---|---|
| [Philip Kiely, *Inference Engineering*](../docs/Inference%20Engineering.pdf), Baseten Books, 2026 | Runtime / infrastructure / tooling; token generation; arithmetic intensity; GPU architecture; kernels; quantization; parallelism; deployment and operations. Relevant sections appear in slide footers. |
| [Lamhot Siagian, *Efficient LLM Inference*, second edition](../docs/llm_Inference.pdf), 2026 | Worked KV-cache explanations, quantization mechanics, speculative decoding, attention variants and memory management. Technical claims were checked rather than adopted wholesale. |
| [Course notebooks](../notebooks/) | Runnable code, existing CPU simulation and quality results, GPU experiments and original section navigation. This revision did not rerun the experiments. |

## Supplied links

| Reference | Contribution and access scope |
|---|---|
| [Into AI: NVIDIA GPU ecosystem](https://www.intoai.pub/p/what-every-ai-engineer-must-know-about-nvidia-gpus) | Public preview: GPU, SM, CUDA/Tensor Core and memory framing. The remainder requires subscription access. Informed the hardware explanation; technical details use local PDFs, Modular and NVIDIA. |
| [Into AI: How to read GPU specs](https://www.intoai.pub/p/how-to-read-gpu-specs) | Accessible preview: form factor, memory capacity and bandwidth. Informed the spec → deployment-question table. Normalize GB/GiB and use NVIDIA's precision/sparsity footnotes. |
| [Daily Dose: How a GPU actually works](https://blog.dailydoseofds.com/p/how-a-gpu-actually-works) | Accessible newsletter introduction explains the arithmetic / data-movement mismatch and links to a separate article. The linked article could not be retrieved in this session; its full illustrations were not reviewed. Hardware mechanisms are supported by the local PDFs and primary sources below. |
| [The Neural Maze: Hands-on guide to LLM inference with vLLM](https://www.theneuralmaze.com/p/the-hands-on-guide-to-llm-inference) | Paid article's public introduction: a synchronous model handler is not the same as a concurrent serving system. Used as motivation; the deployment tutorial beyond the preview was not available. |
| [Hari Krishna: LLM GPU & training terminology](https://nandigamharikrishna.substack.com/p/llm-gpu-and-training-terminology) | Vocabulary coverage and a useful reminder to separate compute, traffic and capacity. Numerical claims require checking: its introductory 80 GB KV claim for a Llama 70B request at 32,768 tokens is inconsistent with the GQA configuration. The worked BF16 calculation is **10 GiB** before sharding and allocation overhead, assuming 80 layers, 8 KV heads and head dimension 128. |
| [Modular: LLM Inference Handbook](https://handbook.modular.com/) | Interactive / visual teaching references: hardware containment, memory reuse scope, batching comparisons, paging, speculation and parallelism. Specific chapters are linked below and in speaker notes. |
| [Inside vLLM: Anatomy of a high-throughput LLM inference system](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) | API → EngineCore → workers; schedule → execute → update. The article explicitly analyzes commit `42172ad` from 9 August 2025. The slide uses its architectural model, not a claim that this snapshot describes every current flag or internal API. |
| [elizabetht/100-days-of-inference](https://github.com/elizabetht/100-days-of-inference) | Learning progression: understand, implement, then measure; runtime → infrastructure → tooling. Its published plan includes unfinished projects, so listing an experiment is not evidence that it has been executed or validated. The course notebooks remain the runnable companion to these slides. |

## Primary checks and focused chapters

- [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/): H100 SXM capacity, HBM bandwidth, precision/sparsity footnotes and interconnect conventions. Approximately 989 TFLOP/s dense BF16 underlies the roofline example; the 1,979 headline is marked with sparsity. The 900 GB/s NVLink figure is aggregate bidirectional bandwidth per GPU.
- [Modular: threads, warps, blocks and grids](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/threads-warps-blocks/): logical execution hierarchy and 32-thread NVIDIA warps.
- [Modular: GPU memory hierarchy](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/gpu-memory/): registers, shared memory, L1/L2 and HBM; reuse scope matters as much as storage size.
- [Modular: static, dynamic and continuous batching](https://handbook.modular.com/inference-optimization/static-dynamic-continuous-batching/): batch-admission differences. The slide's slot grid is illustrative and makes no utilization benchmark claim.
- [Modular: PagedAttention](https://handbook.modular.com/inference-optimization/pagedattention/): logical-to-physical KV mapping and allocation efficiency. Paging does not eliminate the attention reads required by full attention.
- [Modular: parallelism](https://handbook.modular.com/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism/): what each dimension splits and how strategies compose.
- [Dao et al., FlashAttention](https://arxiv.org/abs/2205.14135): IO-aware tiling and exact attention without materializing the full attention-score matrix in HBM.
- [Leviathan et al., speculative decoding](https://arxiv.org/abs/2211.17192): acceptance and residual resampling preserve the target distribution. Separately sampled runs need not produce identical token sequences.

## Topic and slide map

The original three-column overview is preserved in all three decks. Original code, results, commands and tables remain in the light-theme decks; the original mixed-bandwidth ladder is improved as a GPU architecture diagram. Notebook section numbers do not change.

| Deck | Final count | Added visual walkthroughs |
|---|---|---|
| Session 1 | 30 | 7: Trace one generation iteration; 10: Cached rows and the new query; 12: Latency on a token timeline; 16: Grid, blocks, warps and SMs; 17: Matrix tiles and data reuse; 18: Memory scope and capacity; 23: What each spec number means; 24: Compare actual H100 variants; 27: The engine iteration in three stages |
| Session 2 | 27 | 7: Token budget and KV admission; 9: Logical-to-physical KV mapping; 14: FlashAttention before and after; 21: Acceptance plus correction probability |
| Session 3 | 26 | 6: Fabric boundaries and direction conventions; 9: Connect the two TP matrix shards; 14: Which query heads share KV; 17: KV transfer and its time floor |

### References by concept

| Concept | Shared reference used with the local PDFs |
|---|---|
| Generation, prefill/decode, KV | Modular foundations; Inside vLLM; the accessible introduction to the Daily Dose GPU article. |
| Latency, throughput and goodput | Modular key metrics; the existing notebook measurements. |
| GPU internals, execution and memory | Into AI GPU ecosystem preview; Modular GPU architecture chapters; local hardware chapters. |
| Reading GPU specs | Into AI spec-reading preview plus NVIDIA specifications: capacity, bandwidth, dense/sparse precision, form factor, power and interconnect direction. |
| Continuous batching, paging, prefix caching and chunked prefill | Inside vLLM and the corresponding Modular chapters. |
| FlashAttention, fusion and launch overhead | FlashAttention paper; Modular kernel chapters; local PDFs. |
| Quantization and format choice | Modular quantization; local PDFs; terminology article for vocabulary, with numerical claims checked. |
| Speculative decoding | Leviathan et al.; Inside vLLM; Modular speculation. The new probability-mass example explains the acceptance/resampling correction. |
| Parallelism, topology and disaggregation | Into AI ecosystem preview; Modular parallelism and P/D chapters; Inside vLLM. |
| Launch recipes and deployment | Neural Maze public introduction; Inside vLLM; the existing course notebook recipes. API flags remain version-dependent examples. |
| Local planning, benchmarking and operations | Modular deployment/observability chapters; 100 Days of Inference for experimental progression; local PDFs and existing notebooks. |

### Preservation and corrections

- `old/` is read-only input. Its file hashes are checked before delivery. The opening slide XML in each output deck is an exact copy of its original, preserving the supplied overview and its typography.
- Small corrections clarify BF16 versus FP32 KV size, per-step versus total attention complexity, arithmetic intensity, whole-service memory fit, the inverse bandwidth/bytes speed formula, and benchmark scope.
- **37 original tables had zero-height rows.** Their output copies now have explicit row heights and matching table frames, preserving the text and data. Two dense tables use a 2% font adjustment; the small pipeline table is given more room without shrinking its type. No overview content is changed.
- The original ring example already uses **450 GB/s** per direction for H100 NVLink, so its **0.65 ms** estimate remains. The PCIe example used 32 GB/s despite labeling Gen5 x16 at 64 GB/s per direction; the slide now uses **4.59 ms**. These remain bandwidth-only estimates. The notebook itself is unchanged.
- Added walkthroughs have native click-controlled fade transitions with no automatic advance. The diagrams remain editable; static previews do not show the transitions.

## Reading the figures

- **Blue:** main execution / existing state. **Orange:** new work, cost or a constraint. **Green:** reused / accepted state or an explicitly distinguished second shard. Text labels also carry meaning, so color is not the only cue.
- Hardware floorplans, schedules, token sequences and block pools are explanatory schematics, not scale drawings or new measurements. The memory-scope ladder is qualitative, not a bandwidth chart.
- The decode ceiling and disaggregation-transfer example are arithmetic bounds with stated omissions. The cost examples are scenarios, not current cloud prices.
- Quantization-quality values are retained course results, not new benchmarks. The reference-model label is “Original”; the CPU notebook setup uses FP32, so calling that row “BF16” was misleading.
- Code and complete engine launch recipes remain in the linked notebook sections. Engine versions, parsers, formats and CLI flags must match the deployment being tested.

All 83 slides have speaker notes. The original slide content is preserved through package-level copying; new native diagrams are authored with the Presentations skill and artifact-tool. The final decks are re-imported for rendering and checked for slide order, internal relationships, notes, transitions, source-overview preservation and unchanged old/ files. No GPU workload or live serving deployment is executed as part of this presentation revision.
