#!/usr/bin/env bash
# =============================================================================
# GLM-5.2-NVFP4-REAP-504B (madeby561, Luke-Alonso-calibrated, 168-expert) — serve :9300 on 4x RTX PRO 6000 (sm120)
# =============================================================================
# WORKING config. Bundles the fixes from:
#   https://github.com/brandonmmusic-max/GLM-5.2-REAP-fixes
#   1) Checkpoint:  madeby561 504B is a clean 168-expert REAP — NO repair_reap.py;
#      ~309 GB NVFP4 fits 4x96 ALL-VRAM (no offload) and routes correctly.
#   2) Long-ctx:    MOE_A16=1 (B12X_MOE_FORCE_A16) + HF_OVERRIDES DSA index_topk_pattern.
#                   Without these, output degrades to token-salad past ~1-2K gen tokens.
#   3) thinking_token_budget on the V2 runner: set V2_THINK_PATCH_DIR to a dir with the
#      4 patched vLLM files (or use the baked image), and REASONING_CONFIG (set by default).
#   4) NCCL: unset NCCL_GRAPH_FILE (image bakes it empty -> "unhandled system error").
# arch = GlmMoeDsaForCausalLM (inherits the DSV4 b12x path: MLA, b12x MoE, DSA, MTP).
# Perf (PRIOR 156-expert REAP, single-user decode — RE-MEASURING on the 504B):
#   DCP=2 ~67 t/s / 256K KV; DCP=1 + MTP=1/NUM_SPEC=2 ~95 t/s / 154K; DCP=4 ~47 t/s / 512K.
#   MTP+DCP work together on the a86f74e image (see MTP= note below).
# CANNOT run alongside dsv4-9200-prod (needs all 4 GPUs); preflight refuses if prod is up.
# =============================================================================
set -euo pipefail
IMAGE="${IMAGE:-voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618}"
NAME="${NAME:-glm52-9300-test}"
MODEL_HOST_DIR="${MODEL_HOST_DIR:-/home/brandonmusic/models}"
MODEL_DIRNAME="${MODEL_DIRNAME:-GLM-5.2-NVFP4-REAP-504B}"   # Luke-Alonso-calibrated 504B REAP (madeby561); ready NVFP4, NO repair_reap.py needed
MODEL_PATH="${MODEL_PATH:-/models-archive/$MODEL_DIRNAME}"
SERVED_NAME="${SERVED_NAME:-glm-5.2-nvfp4}"
PORT="${PORT:-9300}"
TP="${TP:-4}"
GPU_UTIL="${GPU_UTIL:-0.88}"            # 0.88 for DCP4+MTP (GPU1 carries ~6.7GB display; 0.90+ OOMs it on all_gather). Raise to 0.90 if MTP=0.
MAXLEN="${MAXLEN:-350000}"             # >300K context; fits the ~364K DCP=4+MTP KV pool (use no-MTP for up to ~600K)
MAX_SEQS="${MAX_SEQS:-16}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"   # repaired REAP fits 4x96 all-VRAM; no offload
CPU_OFFLOAD_PARAMS="${CPU_OFFLOAD_PARAMS:-}"
KV_DTYPE="${KV_DTYPE:-fp8}"             # fine on the b12x sparse path
MTP="${MTP:-1}"                         # MTP+DCP FIXED on a86f74e image (upstream #39542 + a86f74e DCP-LSE): boots + errs=0 cc1-16, +13-33% speed. Costs KV (617K->364K). Set 0 for max KV.
MOE_BACKEND="${MOE_BACKEND:-b12x}"
LINEAR_BACKEND="${LINEAR_BACKEND:-auto}"  # auto REQUIRED for MTP (b12x has no NVFP4 nextn eh_proj kernel); fine for no-MTP too
NUM_SPEC="${NUM_SPEC:-2}"               # spec2 sweet spot (1 trained nextn head); only used if MTP=1
ATTN_BACKEND="${ATTN_BACKEND:-B12X_MLA_SPARSE}"
KV_OFFLOAD="${KV_OFFLOAD:-0}"
KV_CPU_BYTES="${KV_CPU_BYTES:-206158430208}"
SPARSE_INDEXER="${SPARSE_INDEXER:-1}"
DSV2_PATCH="${DSV2_PATCH:-}"
FUSED_TOPK="${FUSED_TOPK:-1}"
MOE_A16="${MOE_A16:-1}"                 # LONG-CTX FIX: force A16 MoE decode (w4a4 accumulates error over long gen)
DCP="${DCP:-4}"                         # decode-context-parallel: 4 = ~364K KV(MTP)/617K(no-MTP) = the only >300K path on TP4; 2=256K; 1=154K+fastest single-user
DCP_GLOBAL_TOPK="${DCP_GLOBAL_TOPK:-1}" # DCP TOP-K FIX (PR#30, needs the dcpmtp image): remap each shard's local top-k -> true GLOBAL selection. REQUIRED with DCP>1 or per-shard top-k corrupts attention (~11K KV cost). Default ON (v12 sets it; helps MTP+DCP).
SHARD_DRAFT="${SHARD_DRAFT:-1}"          # DCP SHARD DRAFT (PR#28, v12): shard the MTP/draft KV across DCP ranks instead of replicating -> helps MTP work correctly under DCP. Default ON. Harmless no-op on images lacking the code.
REASONING_PARSER="${REASONING_PARSER:-glm45}"
TOOL_PARSER="${TOOL_PARSER:-glm47}"
# DSA index_topk_pattern. Derive from config.json:
#   python -c "import json;c=json.load(open('config.json'));print(''.join('F' if t=='full' else 'S' for t in c['indexer_types']))"
INDEX_PATTERN="${INDEX_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
# LONG-CTX FIX (single-quoted JSON, no escaping). Override HF_OVERRIDES to disable.
if [ -z "${HF_OVERRIDES:-}" ]; then
  HF_OVERRIDES='{"use_index_cache":true,"index_topk_pattern":"'"$INDEX_PATTERN"'"}'
fi
# REASONING_CONFIG (--reasoning-config) is OPT-IN — ONLY for thinking_token_budget (caps the GLM-5
# thinking-loop) and REQUIRES the V2 patches (V2_THINK_PATCH_DIR or a thinkbudget-baked image).
# *** 2026-06-18 BUG FIX (was default-ON): on a plain image, --reasoning-config CLOBBERS
# --reasoning-parser glm45's GLM-aware <think> handling, so /v1/chat/completions stops priming the
# assistant turn with <think> (chat_template.jinja line 118) and the model answers in NO-THINK mode.
# This silently gimped the aider polyglot run (no reasoning, ~58% no-think vs the real thinking score).
# Verified: a RAW completion with a manual <think> prime thinks fully (1330-tok reasoning + </think>),
# but the chat path with reasoning-config does not. glm45 ALONE gives correct max-effort thinking out
# of the box (matches the HF card lukealonso/GLM-5.2-NVFP4 + local-inference-lab glm5.2_v11.md, which
# pass NO reasoning-config). Default OFF now. To use thinkbudget: set BOTH V2_THINK_PATCH_DIR (or a
# thinkbudget-baked image) AND REASONING_CONFIG explicitly.
REASONING_CONFIG="${REASONING_CONFIG:-}"
# Dir with the 4 patched vLLM files for V2 thinking_token_budget (from the GLM-5.2-REAP-fixes repo,
# applied to a copy of this image's vllm). If empty, thinking_token_budget needs a baked image.
V2_THINK_PATCH_DIR="${V2_THINK_PATCH_DIR:-}"
CACHE_DIR="${CACHE_DIR:-/home/brandonmusic/.cache/glm52-b12x}"
mkdir -p "$CACHE_DIR"
GRAPH_CAP="${GRAPH_CAP:-64}"

echo "== preflight =="
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FATAL: image not present: $IMAGE"; exit 1; }
if docker ps --format '{{.Names}}' | grep -qx dsv4-9200-prod; then
  echo "FATAL: dsv4-9200-prod is RUNNING and holds all 4 GPUs."
  echo "       Stop prod first:  docker stop dsv4-9200-prod   (restart: bash zz_serve_dsv4_v4_chthonic_9200.sh)"
  exit 1
fi
[ -d "$MODEL_HOST_DIR/$MODEL_DIRNAME" ] || { echo "FATAL: model dir missing: $MODEL_HOST_DIR/$MODEL_DIRNAME"; exit 1; }
ST=$(ls "$MODEL_HOST_DIR/$MODEL_DIRNAME"/*.safetensors 2>/dev/null | wc -l)
[ "$ST" -ge 1 ] || { echo "FATAL: no safetensors in $MODEL_HOST_DIR/$MODEL_DIRNAME"; exit 1; }
echo "image=$IMAGE model=$MODEL_DIRNAME util=$GPU_UTIL maxlen=$MAXLEN dcp=$DCP mtp=$MTP moe_a16=$MOE_A16 gtopk=$DCP_GLOBAL_TOPK sharddraft=$SHARD_DRAFT kv=$KV_DTYPE"

PATCH_MOUNT=""
if [ -n "$DSV2_PATCH" ]; then
  [ -f "$DSV2_PATCH" ] || { echo "FATAL: DSV2_PATCH not found: $DSV2_PATCH"; exit 1; }
  PATCH_MOUNT="-v $DSV2_PATCH:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py:ro"
  echo "MOUNTING patched deepseek_v2.py: $DSV2_PATCH"
fi
# V2 thinking_token_budget patches (github.com/brandonmmusic-max/GLM-5.2-REAP-fixes)
if [ -n "$V2_THINK_PATCH_DIR" ]; then
  VG=/opt/venv/lib/python3.12/site-packages/vllm
  for rel in v1/worker/gpu/sample/thinking_budget.py v1/worker/gpu/sample/sampler.py v1/worker/gpu/model_runner.py v1/engine/input_processor.py; do
    [ -f "$V2_THINK_PATCH_DIR/$rel" ] || { echo "FATAL: V2_THINK_PATCH_DIR missing $rel"; exit 1; }
    PATCH_MOUNT="$PATCH_MOUNT -v $V2_THINK_PATCH_DIR/$rel:$VG/$rel:ro"
  done
  echo "MOUNTING V2 thinking_token_budget patches from $V2_THINK_PATCH_DIR"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --gpus all --runtime nvidia --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_HOST_DIR":/models-archive:ro -v "$CACHE_DIR":/cache $PATCH_MOUNT \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUTE_DSL_ARCH=sm_120a \
  -e HF_HUB_OFFLINE=1 -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 -e B12X_MHC_MAX_TOKENS=16384 -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_USE_B12X_MHC=1 -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=$([ "$MOE_BACKEND" = "b12x" ] && echo 1 || echo 0) \
  -e VLLM_USE_B12X_SPARSE_INDEXER=$SPARSE_INDEXER -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=$FUSED_TOPK \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e B12X_MLA_SM120_UNIFIED=1 -e USES_B12X=True -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_MOE_FORCE_A16=$MOE_A16 \
  -e VLLM_DCP_GLOBAL_TOPK=$DCP_GLOBAL_TOPK -e VLLM_DCP_SHARD_DRAFT=$SHARD_DRAFT \
  "$IMAGE" \
  /bin/bash -lc "
    set -euo pipefail
    unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
    if [ '$KV_OFFLOAD' = '1' ]; then unset PYTORCH_CUDA_ALLOC_CONF; fi
    SPEC_ARGS=()
    if [ '$MTP' = '1' ]; then
      SPEC_ARGS=(--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC,\"draft_sample_method\":\"probabilistic\",\"moe_backend\":\"$MOE_BACKEND\",\"use_local_argmax_reduction\":true}')
    fi
    OFFLOAD_ARGS=()
    if [ -n '$CPU_OFFLOAD_PARAMS' ]; then OFFLOAD_ARGS=(--cpu-offload-params '$CPU_OFFLOAD_PARAMS'); fi
    KV_ARGS=()
    if [ '$KV_OFFLOAD' = '1' ]; then KV_ARGS=(--kv-transfer-config '{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$KV_CPU_BYTES}}'); fi
    HF_ARGS=()
    if [ -n '$HF_OVERRIDES' ]; then HF_ARGS=(--hf-overrides '$HF_OVERRIDES'); fi
    PARSER_ARGS=()
    if [ -n '$REASONING_PARSER' ]; then PARSER_ARGS+=(--reasoning-parser '$REASONING_PARSER'); fi
    if [ -n '$TOOL_PARSER' ]; then PARSER_ARGS+=(--tool-call-parser '$TOOL_PARSER' --enable-auto-tool-choice); fi
    if [ -n '$REASONING_CONFIG' ]; then PARSER_ARGS+=(--reasoning-config '$REASONING_CONFIG'); fi
    cd /
    exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve '$MODEL_PATH' \
      --served-model-name '$SERVED_NAME' --host 0.0.0.0 --port '$PORT' \
      --cpu-offload-gb '$CPU_OFFLOAD_GB' \
      --kv-cache-dtype '$KV_DTYPE' --block-size 256 --load-format safetensors \
      --tensor-parallel-size '$TP' --decode-context-parallel-size '$DCP' --moe-backend '$MOE_BACKEND' --linear-backend '$LINEAR_BACKEND' \
      --gpu-memory-utilization '$GPU_UTIL' --max-model-len '$MAXLEN' --max-num-seqs '$MAX_SEQS' \
      --enable-chunked-prefill --max-num-batched-tokens '$MAX_BATCHED' \
      --max-cudagraph-capture-size '$GRAPH_CAP' --attention-backend '$ATTN_BACKEND' \
      --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}' \
      --enable-flashinfer-autotune \
      \"\${OFFLOAD_ARGS[@]}\" \"\${KV_ARGS[@]}\" \"\${HF_ARGS[@]}\" \"\${PARSER_ARGS[@]}\" \"\${SPEC_ARGS[@]}\"
  "
echo "Launched $NAME (GLM-5.2-REAP: tp=$TP util=$GPU_UTIL maxlen=$MAXLEN dcp=$DCP mtp=$MTP moe_a16=$MOE_A16)"
echo "watch boot:  docker logs -f $NAME"
echo "smoke test:  curl -s http://127.0.0.1:$PORT/v1/models"
echo "budget test: curl -s http://127.0.0.1:$PORT/v1/chat/completions -d '{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"thinking_token_budget\":2000}'"
