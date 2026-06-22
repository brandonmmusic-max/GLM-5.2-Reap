# GLM-5.2-Reap — recall-recovery recipe + serving stack

A **recall-recovering re-REAP** of GLM-5.2-NVFP4 to **172 experts/layer**, plus the verified Docker image, launch script, and saliency methodology to reproduce it.

- **Model (HF):** [brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172](https://huggingface.co/brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172)
- **Docker image:** [`verdictai/gloriousluminousmonotheism:latest`](https://hub.docker.com/r/verdictai/gloriousluminousmonotheism)
- **Verdict log:** [`REAP_RECALL_VERDICT.md`](./REAP_RECALL_VERDICT.md)
- **License:** MIT

> **Attribution:** **GLM-5.2** by z.ai · **NVFP4 quantization** by **Luke Alonso** (the 256-expert base this work re-REAPs) · **REAP** by **Cerebras Research** ([arXiv:2510.13999](https://arxiv.org/abs/2510.13999), ICLR 2026).

---

## What this fixes — and what's specifically new here

Standard REAP compresses GLM-5.2 well for code, agentic, and reasoning workloads — but **closed-book factual recall collapses** (the published narrow-calibrated REAPs answer "Lexington" for the capital of Kentucky, and loop on *Marbury v. Madison*). This is **not a model defect, not a serving defect** — it's a calibration corpus that excludes knowledge.

**What I did specifically:**

1. **Diagnosed it as calibration, not the model** — A/B'd kernels, MTP, DCP, image, sampling. Same prompts, same parser, same image: behavior tracked the *calibration corpus*, not anything else.
2. **Built a 4-axis balanced calibration** — 12,228 samples (3,057 per axis), 16,384 max seq length, **no truncation, no packing**:
   - **Axis 1 — General knowledge:** C4, Wikipedia, MMLU-aux, TriviaQA, Natural Questions
   - **Axis 2 — Legal (real Kentucky case law):** **1,528 CAP markdown cases** + a live Neo4j knowledge graph (300 headnotes, 390 statutes, 373 case summaries, 113 fact-atoms, 353 worked-examples). `fallback_used: false`.
   - **Axis 3 — Code / agentic:** evol-codealpaca, Magicoder, xLAM function-calling, SWE-smith
   - **Axis 4 — Reasoning / termination:** terminating `<think>…</think>` traces, `</think>`-region weighted ×6
3. **Built a real block-wise NVFP4 saliency runner** ([`tools/blockwise_nvfp4_saliency.py`](./tools/blockwise_nvfp4_saliency.py)) — because the published REAP loader cannot consume `glm_moe_dsa + modelopt NVFP4`, and a whole-model vLLM load OOMs the intact 435 GB before any forward. The runner chunks decoder layers into VRAM, dequants NVFP4 → BF16 in place, runs the GlmMoeDsaNaiveMoe modules (which expose per-expert outputs — the ideal saliency hook), captures `S_j = mean over active tokens(router_gate_j · ||expert_output_j||₂)`, then frees the chunk. **Real GPU saliency over 7,368,253 active tokens across 75 MoE layers** — not a static proxy.
4. **Self-consistent prune at prune time, no repair step** — dropped experts removed (not masked); kept experts renumbered contiguous 0…171; router shrunk to `[172, 6144]`; bias to `[172]`; config `n_routed_experts = num_experts = 172`. Loads clean on stock vLLM with **no `repair_reap.py`**.
5. **Verified the result with the exact prompts standard REAP fails on:** Kentucky → Frankfort, Marbury → judicial review (with reasoning), Texas → Austin, plus 8/8 trap reasoning prompts (car-wash-bring-the-car, oranges-no-knife, marble-on-table, R's in "strawberry", 10 sentences ending "apple", named-animal untangle, pen-just-type, etc.) — all `finish=stop`, no loops.

The full ledger is in [`REAP_RECALL_VERDICT.md`](./REAP_RECALL_VERDICT.md).

---

## Quick start — serve the model

```bash
# 1. Pull the image
docker pull verdictai/gloriousluminousmonotheism:latest

# 2. Download the model (294 GB)
huggingface-cli download brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172 \
  --local-dir $HOME/models/GLM-5.2-NVFP4-REAP-Recall-N172

# 3. Serve (4x RTX PRO 6000 96GB, sm120)
bash serve/serve_glm52_reap_recall.sh

# 4. Sanity-check (closed-book recall — the prompt the narrow REAPs fail on)
curl -s http://127.0.0.1:9402/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2-nvfp4","messages":[{"role":"user","content":"what is the capital of kentucky?"}]}'
# expected: "Frankfort"
```

See [`serve/serve_glm52_reap_recall.sh`](./serve/serve_glm52_reap_recall.sh) for the full launcher (all knobs documented inline).

---

## The verified serving config

Measured on **4× RTX PRO 6000 96GB (sm120, PCIe Gen5, no NVLink)**, TP4. This is the exact combination that passes the 15/15 benchmark grid:

| Knob | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | 4 | 4× 96 GB |
| `--decode-context-parallel-size` | 4 | only >300K KV path on TP4 |
| `--gpu-memory-utilization` | 0.95 | with batched=2048, GPU1 has ~1.1 GB headroom for the remap |
| `--max-model-len` | 250000 | fits the 542K KV pool with conc 2.17× |
| `--max-num-seqs` | 16 | |
| `--max-num-batched-tokens` | **2048** | halves the DCP global-topk remap (576 → 144 MiB) — the lever that ends GPU1 OOMs |
| `--kv-cache-dtype` | **nvfp4_ds_mla** | 4-bit MLA KV cache (~+1.47× context vs fp8) |
| MTP | **num_speculative_tokens=3** | mean accept length 3.87 under sustained load |
| `B12X_MOE_FORCE_A16=1` | required | w4a4 accumulates error past ~1–2K gen tokens |
| `VLLM_DCP_GLOBAL_TOPK=1` | required | DCP>1 without it corrupts attention (and the DSA decode path requires the schedule metadata) |
| `VLLM_DCP_SHARD_DRAFT=1` | recommended | shards MTP/draft KV across DCP ranks |
| DSA `index_topk_pattern` | `FFFSSSF…SSS` | derived from `config.json` (line 54 of the launcher) |
| `-cc.cudagraph_mode=PIECEWISE` | **required for long context** | Under `FULL_AND_PIECEWISE` (the default), the CuTe-DSL JIT for `_dcp_pack_topk_candidates_kernel` deadlocks at first decode after a long prefill (>100K tokens) — `sample_tokens` RPC times out. PIECEWISE breaks the graph at `vllm::sparse_attn_indexer` (already in `splitting_ops`) so the indexer runs eagerly between captured pieces. **JSON form `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` silently drops to None — must use the CLI shortcut.** |

---

## Benchmarks (decode + prefill, 15/15 cells)

Benchmarked with [local-inference-lab/llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench), `--concurrency 1,2,4 --contexts 0,16384,32768,65536,131072`. Server held; **0 cells skipped**. VRAM peak: 97.93% (no waste, no OOM).

### Decode throughput (tok/s)

| ctx | C=1 (per-req) | C=2 (per-req) | C=4 (per-req) | C=4 (aggregate) |
|---|---|---|---|---|
| 0 | **80.7** | 45.4 | 40.4 | 161.8 |
| 16K | 65.6 | 46.3 | 38.2 | 152.7 |
| 32K | 52.1 | 43.2 | 34.2 | 136.7 |
| 64K | 50.9 | 36.9 | 34.1 | 136.2 |
| **128K** | **82.6** | 63.0 | 54.7 | **218.6** |

> The 128K outlier is real but reflects MTP3's near-perfect (0.99 / 0.95 / 0.93) acceptance on the benchmark's padded calibration text + prefix-cache lift. For natural English production traffic, expect the 0–64K shape (~50–80 t/s single-stream).

### Real-prompt single-user (Marbury essay × 5, thinking OFF, 1500-token cap)

| run | tokens | TTFT | decode tok/s |
|---|---|---|---|
| 1 | 1500 | 0.13 s | 58.94 |
| 2 | 1500 | 1.06 s | 59.93 |
| 3 | 1500 | 0.14 s | 57.52 |
| 4 | 1500 | 0.15 s | 57.17 |
| 5 | 1500 | 0.15 s | 58.70 |
| **avg** | | **0.33 s** | **58.45** |

**~58 t/s single-user on real English generation**, ±1.5 t/s.

### Prefill (tok/s, full ingest)

| ctx | tokens | TTFT | tok/s |
|---|---|---|---|
| 8K | 8,199 | 4.66 s | 1,761 |
| 16K | 16,228 | 10.87 s | 1,493 |
| 32K | 32,321 | 20.43 s | 1,582 |
| 64K | 64,513 | 42.27 s | 1,526 |
| 128K | 128,887 | 87.26 s | 1,477 |

MTP3 sustained acceptance length (under benchmark load): **3.87** (per-position 0.992 / 0.948 / 0.931).

---

## Reproducing the re-REAP

The full saliency runner is at [`tools/blockwise_nvfp4_saliency.py`](./tools/blockwise_nvfp4_saliency.py). The verdict log [`REAP_RECALL_VERDICT.md`](./REAP_RECALL_VERDICT.md) has the exact corpus assembly counts (`fallback_used: false`), the saliency method tag (`real_nvfp4_gpu_forward_saliency_blockwise_natural_length`), per-layer score statistics, and the structural validation of the N=172 checkpoint.

Calibration corpus (matched to REAP paper, axis-balanced):
- 12,228 samples total (3,057 per axis), max 16,384 tokens/sample, no truncation, no packing
- Legal axis sourced from CAP markdown + live Neo4j; if you don't have a legal KG, fold equivalent text from your domain into Axis 2

Saliency:
- `S_j = mean_{active routed tokens}( router_gate_j · ||expert_output_j||_2 )`
- Anti-degeneracy guard rejects any layer whose scores have ≤4 distinct values (catches dead-load layers fast)

Pruning (no `repair_reap.py` needed):
- Keep top-172 by real S_j per layer (3–77)
- Renumber kept experts 0…171 contiguous
- Shrink `gate.weight[kept] → [172, 6144]` and `e_score_correction_bias[kept] → [172]`
- Set `n_routed_experts = num_experts = 172`; rebuild `model.safetensors.index.json`

---

## Hardware notes

- Built and verified on **4× RTX PRO 6000 96GB** (sm120, PCIe Gen5, no NVLink) + 512 GB DDR5
- GPU1 carries ~1.5 GB of display overhead (COSMIC + VSCode) — this is the limiting shard and the reason `MAX_BATCHED=2048` (vs the launcher's old default of 8192) was necessary to fit the global-topk remap

For broader serving fixes on consumer Blackwell (checkpoint repair for legacy REAP releases, long-context A16 + DSA pattern, thinking-budget on the V2 runner), see the companion repo: [brandonmmusic-max/GLM-5.2-REAP-fixes](https://github.com/brandonmmusic-max/GLM-5.2-REAP-fixes).

---

## License

MIT for everything in this repo. Free to use, modify, redistribute — please credit:

- z.ai (GLM-5.2 base)
- Luke Alonso (NVFP4 base quantization of GLM-5.2)
- Cerebras Research (REAP method, arXiv:2510.13999)

```bibtex
@inproceedings{lasby2026reap,
  title={{REAP} the Experts: Why Pruning Prevails for One-Shot MoE compression},
  author={Mike Lasby and Ivan Lazarevich and Nish Sinnadurai and Sean Lie and Yani Ioannou and Vithursan Thangarasa},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=ukGxWd2aDG}
}
```
