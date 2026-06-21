---
license: mit
base_model:
- zai-org/GLM-5.2
library_name: transformers
pipeline_tag: text-generation
tags:
- reap
- mixture-of-experts
- moe-pruning
- nvfp4
- glm
- legal
- vllm
---

# GLM-5.2-NVFP4-REAP-Recall (N=172)

A **recall-recovered** re-REAP of GLM-5.2-NVFP4.

Standard REAP (one-shot MoE expert pruning) compresses GLM-5.2 well for code, agentic, and reasoning workloads — but **closed-book factual recall, especially legal, collapses.** The cause is not the model and not the serving stack: it is the **calibration corpus** used to score experts, which excludes knowledge. This checkpoint fixes that by **re-running the REAP saliency pass on a knowledge- and legal-inclusive, axis-balanced calibration corpus**, recovering closed-book recall while keeping reasoning and termination intact.

- **Base model:** GLM-5.2 (z.ai)
- **NVFP4 quantization:** by **Luke Alonso** — the full 256-routed-expert NVFP4 checkpoint this work is built on (78 layers, top-8, modelopt NVFP4)
- **Pruning method:** **REAP** — Router-weighted Expert Activation Pruning (Cerebras Research, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999), ICLR 2026)
- **This work:** a recall-recovering re-REAP to **N=172 experts/layer** (from 256) — self-consistent, BF16 weights preserved (no re-quant), NVFP4 experts sliced byte-for-byte
- **License:** MIT

---

## The problem: REAP prunes away closed-book recall

REAP scores each routed expert by

```
S_j = mean over active tokens ( router_gate_j · || expert_output_j ||_2 )
```

then keeps the top-N experts per layer. The published GLM-5.2 REAP checkpoints calibrate this score on a **narrow** corpus — code/agentic plus termination traces, *notably excluding general and domain knowledge*. Knowledge and legal experts therefore score low, get pruned, and the model **denies famous closed-book facts**, even though in-context reasoning and tool-use survive.

A narrow-calibrated REAP of GLM-5.2 fails exactly here:

| Prompt | Narrow-REAP answer |
|---|---|
| "What is the capital of Kentucky?" | **Lexington** (wrong) |
| "In one sentence, what did Marbury v. Madison establish?" | empty / repetition loop |

This was confirmed to be a **calibration problem, not a serving problem** — the serving stack (kernels, fp4 KV cache, MTP, DCP, image) was fully exonerated via token-for-token A/Bs. The fix is therefore a **re-REAP with a knowledge- and legal-inclusive, balanced calibration.**

---

## What I did: the recall-recovering re-REAP

### 1. Start from the full, knowledge-intact NVFP4 model

The source is Luke Alonso's **full GLM-5.2-NVFP4** — all 256 routed experts intact (knowledge experts still present). BF16 components (attention, shared experts, dense layers 0–2, `lm_head`) are **kept exactly — no re-quant of anything.** The NVFP4 experts are sliced **byte-for-byte** (REAP paper §E), so the kept experts are bit-identical to the source.

### 2. Build a 4-axis *balanced* calibration corpus

The core insight: balance by **axis**, not by source, so no single direction dominates the saliency. Per the REAP paper's full calibration size:

- **12,228 samples total**, balanced **~3,057 per axis**
- **max sequence length 16,384**, with **no truncation and no packing** (every document at its natural length)

| Axis | Purpose | Sources |
|---|---|---|
| **1 — General knowledge** | recover closed-book recall | C4, Wikipedia, MMLU-aux, TriviaQA / Natural Questions, Luke-style diverse/deep calibration |
| **2 — Legal** (the "one arrow") | recover *domain* recall | **real Kentucky case law** (CAP markdown, 1,528 cases) + **Neo4j legal knowledge graph** — Case summaries (373), Headnotes (300), **Statutes (390)**, FactAtoms (113), WorkedExamples (353) |
| **3 — Code / agentic** | keep pruned model's strengths | evol-codealpaca, Magicoder, xLAM function-calling, SWE-smith, agentic-coding calibration |
| **4 — Reasoning / termination** | keep clean stop behavior | terminating `<think>…</think>` traces, `</think>`-region weighted ×6 |

The legal axis uses the **real corpus** (`fallback_used: false`) — actual CAP case text and live KG statute/case/headnote text, not snippets. Statutes are explicitly included.

### 3. Run a *real* block-wise NVFP4 saliency pass

REAP saliency requires a **real GPU forward pass** over the calibration corpus — no static router/bias proxy, no reuse of a fixed "core." The challenge: the intact 435 GB NVFP4 model cannot be loaded whole for a forward (REAP/Transformers cannot load `glm_moe_dsa` + modelopt NVFP4; a whole-model vLLM load OOMs before any forward).

The solution is a **block-wise NVFP4 saliency runner** (`tools/blockwise_nvfp4_saliency.py`):

1. Dequantize a **chunk of consecutive decoder layers** NVFP4 → BF16 **directly into VRAM** (chunked, never the full 435 GB at once — much faster than CPU offload for a multi-sample pass).
2. Forward the calibration corpus through that chunk using the `GlmMoeDsaNaiveMoe` modules (explicit per-expert outputs — the ideal saliency hook), capturing `S_j = router_gate_j · ||expert_output_j||_2` accumulated over active routed tokens.
3. Free the chunk, load the next; ~5–6 chunks cover all 78 layers.
4. **Anti-degeneracy guards** reject any layer whose scores are degenerate (all-zero/NaN or ≤4 distinct values).

Saliency actually run (recorded in the keep-map):
- Method: `real_nvfp4_gpu_forward_saliency_blockwise_natural_length`
- **12,228 rows, 7,368,253 active tokens scored, 75 MoE layers** (3–77)
- Non-degenerate: min nonzero experts/layer 102, score variance 0.0004–53.3

### 4. Prune to N=172, self-consistent

Keep the **top-172 experts per layer** by real saliency. The prune is **hard and correct at prune time** (no masking, no 256-vs-172 mismatch):

- Dropped experts **removed**; kept experts **renumbered contiguous 0…171**
- Router shrunk: `gate.weight[kept] → [172, 6144]`, `e_score_correction_bias[kept] → [172]`, renormalized
- Config consistent: `n_routed_experts = num_experts = 172`; MTP/nextn layer sliced the same way
- **Validated:** every MoE layer 3–77 has exactly 172 contiguous experts; **loads clean on stock vLLM with no repair step.**

---

## Results

### Closed-book recall — recovered

Sampling: `temperature 1.0, top_p 0.95, repetition_penalty 1.05, chat_template_kwargs {enable_thinking: true, reasoning_effort: high}, stop_token_ids [154820, 154827, 154829]`.

| Prompt | N=172 answer | Finish |
|---|---|---|
| What is the capital of Kentucky? | **Frankfort** | stop |
| In one sentence, what did Marbury v. Madison (1803) establish? | **judicial review** — the Supreme Court's authority to declare laws unconstitutional | stop |
| What is the capital of Texas? | **Austin** | stop |

(The narrow-REAP baseline returns *Lexington* / empty-loop on the first two.)

### Reasoning & termination — preserved

All terminate cleanly (`finish=stop`, no loops):

- Two-stage discount → **$144, 28%**
- Pipes A 4 h / B 6 h → **2 h 24 m**
- Anticipatory repudiation → **B wins**
- Syllogism (finches) → correct
- 8/8 trap prompts (car-wash-bring-the-car, oranges-no-knife, marble-on-table, R's in "strawberry", 10 sentences ending "apple", named-animal untangle, pen-just-type)

### Serving & KV cache

- 4× RTX PRO 6000 (96 GB, sm120, PCIe Gen5, no NVLink), TP4
- vLLM b12x fork with the **`nvfp4_ds_mla` 4-bit MLA KV cache** (≈ +1.47× context vs fp8)
- DCP4 + MTP (3 speculative tokens), `B12X_MOE_FORCE_A16=1` (w4a16 MoE decode for long-context correctness)
- Measured fp4 KV pool: **~397K tokens @ util 0.95 / DCP4** (up to ~599K at higher util)
- MTP-3 mean acceptance length **~3.1** at idle, ~2.6 under load

> Decode-throughput matrix (concurrency × context) is benchmarked with [local-inference-lab/llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench).

---

## Files

- 87 safetensors shards (294 GB), `config.json`, `model.safetensors.index.json`, tokenizer, `generation_config.json`, `chat_template.jinja`
- **`reap_recall_keep_map_with_scores.json`** — the per-expert real saliency scores and the kept-expert map (the replication artifact)

---

## Reproduce

- **Recipe + saliency runner + serving config + fp4 KV kernels:** https://github.com/brandonmmusic-max/GLM-5.2-Reap
- **Serving fixes for consumer Blackwell** (checkpoint repair, long-context A16 + DSA, thinking-budget): https://github.com/brandonmmusic-max/GLM-5.2-REAP-fixes

---

## Attribution

- **GLM-5.2** — z.ai (base model)
- **NVFP4 quantization of GLM-5.2** — **Luke Alonso** (the 256-expert NVFP4 checkpoint this work builds on)
- **REAP** — Cerebras Research, *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*

```bibtex
@inproceedings{lasby2026reap,
  title={{REAP} the Experts: Why Pruning Prevails for One-Shot MoE compression},
  author={Mike Lasby and Ivan Lazarevich and Nish Sinnadurai and Sean Lie and Yani Ioannou and Vithursan Thangarasa},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=ukGxWd2aDG}
}
```

## License

MIT — free to use, modify, and redistribute, with attribution to the sources above.
