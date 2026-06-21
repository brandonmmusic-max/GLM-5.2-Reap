# REAP Recall Verdict

Verdict: PASS. The final checkpoint is the N=172 real-saliency REAP recall build at:

`/media/brandonmusic/nvme0n1p3/models/GLM-5.2-NVFP4-REAP-recall`

N=172 was accepted per Brandon's later override. No further expert cuts were made.

## Milestones

- Corpus assembled: done, reused cached `work/calibration/reap_recall_calib.jsonl`.
- Real saliency: done, reused cached `work/saliency/reap_recall_real_saliency.json`.
- N=190: not remeasured this turn because the override said reuse N=172 and do not cut further.
- Final cut: N=172, structurally validated and load tested.

## Calibration Corpus

Total: 12,228 samples, balanced 3,057 per axis, max natural sample length 16,384 tokens, no packing.

Per-axis counts:

- Axis 1 general knowledge: 3,057
- Axis 2 legal: 3,057
- Axis 3 code/agentic: 3,057
- Axis 4 reasoning/termination: 3,057

Legal source counts:

- CAP markdown cases: 1,528
- Neo4j Case summaries: 373
- Neo4j Headnotes: 300
- Neo4j Statutes: 390
- Neo4j FactAtoms: 113
- Neo4j WorkedExamples: 353

Legal gate:

- `fallback_used:false`
- CAP primary path exists: `/home/brandonmusic/kentucky_legal_counsel_local/M:/cap_markdown_ky_complete`
- CAP primary path count observed by the cached assembler: 2,965 `.md`, 74,170,225 bytes
- Neo4j label counts recorded: Case 204,939; Headnote 322,587; Statute 58,760; WorkedExample 451,319; FactAtom 85,735

## Saliency

Real saliency was used, not the old proxy/static term-core map.

- Method: `real_nvfp4_gpu_forward_saliency_blockwise_natural_length`
- Formula: mean over routed active tokens of `topk_router_weight * l2_norm(unweighted_expert_output)`
- Rows: 12,228
- Active tokens scored: 7,368,253
- Max token length: 16,384
- Natural-length tokenization: true
- Layers scored: 75 MoE layers, 3 through 77
- Non-degenerate score sanity: min nonzero experts per layer 102; max 256; min variance 0.0003975; max variance 53.3344
- N=172 keep map: `work/reap_recall_keep_map_real_N172.json`, derived from the same real scores as N=190

## Checkpoint

- Final kept experts: 172
- Config: `n_routed_experts=172`, `num_experts=172`, `num_hidden_layers=78`, `first_k_dense_replace=3`
- Disk size: 294G
- Safetensors: 87 total index shards referenced; 84 numbered model shards plus inputscale/MTP support files
- Structural validation: passed with `tools/validate_reap_recall_checkpoint.py --n-experts 172`
- Validator result: layers 3-78 have exactly 172 contiguous experts `0..171`, gate `[172,6144]`, bias `[172]`
- Load test: passed on vLLM with no `repair_reap.py`

## Serve Measurement

Served as `glm52-nvfp4-recall-9402` on port 9402 using `klc/glm52-nvfp4-dcpmtp:v2-trim`.

Launcher settings used:

- `MAXLEN=200000`
- `GPU_UTIL=0.96`
- `DCP=4`
- `MTP=1`
- `NUM_SPEC=2`
- `MOE_A16=1`
- `LINEAR_BACKEND=auto`
- `KV_DTYPE=nvfp4_ds_mla`

vLLM boot evidence:

- Main checkpoint load: 87/87 safetensors, checkpoint size 293.64 GiB
- MTP load: 2/2 safetensors, checkpoint size 3.94 GiB
- `GPU KV cache size: 533,673 tokens`
- `Maximum concurrency for 200,000 tokens per request: 2.67x`
- Application startup complete

Live per-GPU memory after boot and tests:

- GPU0: 95,222 / 97,887 MiB
- GPU1: 95,792 / 97,887 MiB
- GPU2: 94,644 / 97,887 MiB
- GPU3: 94,642 / 97,887 MiB

## Recall Tests

Sampling used: temperature 1.0, top_p 0.95, repetition_penalty 1.05, `chat_template_kwargs={"enable_thinking":true,"reasoning_effort":"high"}`, stop token IDs `[154820,154827,154829]`.

Verbatim recall prompts:

| Prompt | Final N=172 answer | Finish |
| --- | --- | --- |
| `what is the capital of kentucky?` | `The capital of Kentucky is Frankfort.` | stop |
| `In one sentence, what did Marbury v. Madison (1803) establish?` | `Marbury v. Madison (1803) established the principle of judicial review, empowering the U.S. Supreme Court with the authority to declare laws and government actions unconstitutional.` | stop |
| `what is the capital of Texas?` | `The capital of Texas is Austin.` | stop |

The Marbury prompt initially hit a too-small `max_tokens=220` cap while still in hidden thinking; the successful rerun kept the prompt and sampling unchanged and only raised `max_tokens` to 2048.

Baseline/degraded reference: the run request identifies `-term` as failing these recall probes. A saved prior degraded baseline log at `/home/brandonmusic/klc-linux/codex_fp4_final.log` also shows Kentucky returning `Lexington` and Marbury thinking-high reaching `finish=length` with an empty answer. I did not boot a second baseline server in this turn because the verified recall server occupies the four GPUs and the instruction was one server at a time.

## Termination And Reasoning

Final N=172 tests all stopped cleanly after increasing caps where hidden thinking needed room:

| Test | Expected | Final N=172 answer | Finish |
| --- | --- | --- | --- |
| Two-stage discount | `$144`, `28%` | `The final price is $144, and the total percent discount is 28%.` | stop |
| Pipes A 4h / B 6h | `2h24m` | `2.4 hours (or 2 hours and 24 minutes)` | stop |
| Anticipatory repudiation | `B wins` | `B wins under the doctrine of anticipatory repudiation` | stop |
| Syllogism | finch conclusion | `Yes, every finch has feathers` | stop |

No termination loop was observed in the accepted final tests.
