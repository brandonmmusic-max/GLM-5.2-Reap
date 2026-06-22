#!/usr/bin/env bash
# =============================================================================
# GLM-5.2-NVFP4-REAP-Recall (N=172) — serve on 4x RTX PRO 6000 (sm120) / TP4
#   Model:    https://huggingface.co/brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172
#   Image:    docker.io/verdictai/gloriousluminousmonotheism:latest
#   Recipe:   https://github.com/brandonmmusic-max/GLM-5.2-Reap
#
# This is the EXACT verified config (Brandon Music, 2026-06-21):
#   util 0.95, DCP=4, MTP3, max-num-batched-tokens=2048, gtopk=1, nvfp4_ds_mla KV
# Measured on 4x RTX PRO 6000 96GB:
#   KV pool          : 542,857 tokens (max_conc@250K = 2.17x)
#   Single-user      : 80.7 t/s decode @ 0 ctx (synth padded text, MTP3 acc=3.87)
#                      ~58 t/s decode on natural English (Marbury essay, x5 runs)
#   Aggregate @ C=4  : 218.6 t/s @ 128K ctx
#   Prefill 128K     : 1,477 t/s (87s TTFT for full 128K)
#   VRAM peak        : 97.93% (no headroom waste, no OOM in benchmark)
#
# Hardware fixes carried over (from GLM-5.2-REAP-fixes):
#   1) MOE_A16=1 (B12X_MOE_FORCE_A16) for long-ctx correctness (w4a4 accumulates error)
#   2) HF_OVERRIDES.index_topk_pattern: DSA sparse-indexer pattern (derived from config.json)
#   3) NCCL: unset NCCL_GRAPH_FILE inside the container
#   4) -cc.cudagraph_mode=PIECEWISE (NOT JSON form): required for long-context decode.
#      Under FULL cudagraph capture, the CuTe-DSL JIT machinery's cooperative-grid
#      spin-barrier deadlocks because it's invoked from a captured stream — the
#      _dcp_pack_topk_candidates_kernel JIT-compiles at inference time on the first
#      long-prompt decode and hangs (sample_tokens RPC times out). PIECEWISE breaks
#      the graph at vllm::sparse_attn_indexer (already in splitting_ops) so the
#      indexer runs eagerly between captured pieces. The JSON form
#      `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` silently drops to
#      None — the CLI shortcut `-cc.cudagraph_mode=PIECEWISE` is the only working form.
# =============================================================================
set -euo pipefail

# --- Required: path to the downloaded HF model directory on the host ----------
# Download with:
#   huggingface-cli download brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172 \
#     --local-dir /path/to/models/GLM-5.2-NVFP4-REAP-Recall-N172
MODEL_HOST_DIR="${MODEL_HOST_DIR:-$HOME/models}"
MODEL_DIRNAME="${MODEL_DIRNAME:-GLM-5.2-NVFP4-REAP-Recall-N172}"
MODEL_PATH="${MODEL_PATH:-/models-archive/$MODEL_DIRNAME}"

# --- Serving identity ---------------------------------------------------------
IMAGE="${IMAGE:-verdictai/gloriousluminousmonotheism:latest}"
NAME="${NAME:-glm52-recall}"
SERVED_NAME="${SERVED_NAME:-glm-5.2-nvfp4}"
PORT="${PORT:-9402}"

# --- Verified parallelism / batch / KV config (the WORKING combo) --------------
TP="${TP:-4}"                            # 4x RTX PRO 6000 96GB
DCP="${DCP:-4}"                          # decode-context-parallel = TP for max KV
GPU_UTIL="${GPU_UTIL:-0.95}"             # 0.95 with batched=2048 leaves GPU1 ~1.1GB remap headroom
MAXLEN="${MAXLEN:-250000}"               # well above 128K; fits the 542K KV pool
MAX_SEQS="${MAX_SEQS:-16}"
MAX_BATCHED="${MAX_BATCHED:-2048}"       # smaller chunks halve DCP-global-topk remap (576 -> 144 MiB)
KV_DTYPE="${KV_DTYPE:-nvfp4_ds_mla}"     # 4-bit MLA KV cache (+1.47x context vs fp8)

# --- MTP / speculative decode --------------------------------------------------
MTP="${MTP:-1}"                          # MTP on/off
NUM_SPEC="${NUM_SPEC:-3}"                # 3 speculative tokens; mean accept ~3.87 under sustained load

# --- MoE / DSA / quantization knobs -------------------------------------------
MOE_BACKEND="${MOE_BACKEND:-b12x}"
LINEAR_BACKEND="${LINEAR_BACKEND:-auto}" # required for MTP (b12x has no NVFP4 nextn eh_proj kernel)
ATTN_BACKEND="${ATTN_BACKEND:-B12X_MLA_SPARSE}"
SPARSE_INDEXER="${SPARSE_INDEXER:-1}"
FUSED_TOPK="${FUSED_TOPK:-1}"
MOE_A16="${MOE_A16:-1}"                  # LONG-CTX FIX (w4a16 MoE decode)
DCP_GLOBAL_TOPK="${DCP_GLOBAL_TOPK:-1}"  # DCP top-k remap (required with DCP>1)
SHARD_DRAFT="${SHARD_DRAFT:-1}"          # shard MTP/draft KV across DCP ranks
GRAPH_CAP="${GRAPH_CAP:-64}"

# --- Reasoning / parsing -------------------------------------------------------
REASONING_PARSER="${REASONING_PARSER:-glm45}"
TOOL_PARSER="${TOOL_PARSER:-glm47}"
# REASONING_CONFIG is opt-in: ONLY set for thinking_token_budget; otherwise it CLOBBERS
# the glm45 parser's <think> priming and gimps thinking quality. Default OFF.
REASONING_CONFIG="${REASONING_CONFIG:-}"

# --- DSA sparse-indexer pattern. Derived from the model's config.json ----------
# python -c "import json;c=json.load(open('config.json'));print(''.join('F' if t=='full' else 'S' for t in c['indexer_types']))"
INDEX_PATTERN="${INDEX_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
if [ -z "${HF_OVERRIDES:-}" ]; then
  HF_OVERRIDES='{"use_index_cache":true,"index_topk_pattern":"'"$INDEX_PATTERN"'"}'
fi

# --- Caches --------------------------------------------------------------------
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/glm52-recall}"
mkdir -p "$CACHE_DIR"

# --- Preflight -----------------------------------------------------------------
echo "== preflight =="
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "image not local; pulling $IMAGE ..."; docker pull "$IMAGE"
}
[ -d "$MODEL_HOST_DIR/$MODEL_DIRNAME" ] || {
  echo "FATAL: model dir missing: $MODEL_HOST_DIR/$MODEL_DIRNAME"
  echo "  Download with:"
  echo "    huggingface-cli download brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172 \\"
  echo "      --local-dir $MODEL_HOST_DIR/$MODEL_DIRNAME"
  exit 1
}
ST=$(ls "$MODEL_HOST_DIR/$MODEL_DIRNAME"/*.safetensors 2>/dev/null | wc -l)
[ "$ST" -ge 1 ] || { echo "FATAL: no safetensors in $MODEL_HOST_DIR/$MODEL_DIRNAME"; exit 1; }

echo "image=$IMAGE model=$MODEL_DIRNAME util=$GPU_UTIL maxlen=$MAXLEN dcp=$DCP mtp=$MTP num_spec=$NUM_SPEC max_batched=$MAX_BATCHED moe_a16=$MOE_A16 gtopk=$DCP_GLOBAL_TOPK kv=$KV_DTYPE"

# --- Launch --------------------------------------------------------------------
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --gpus all --runtime nvidia --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_HOST_DIR":/models-archive:ro -v "$CACHE_DIR":/cache \
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
    SPEC_ARGS=()
    if [ '$MTP' = '1' ]; then
      SPEC_ARGS=(--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC,\"draft_sample_method\":\"probabilistic\",\"moe_backend\":\"$MOE_BACKEND\",\"use_local_argmax_reduction\":true}')
    fi
    HF_ARGS=()
    if [ -n '$HF_OVERRIDES' ]; then HF_ARGS=(--hf-overrides '$HF_OVERRIDES'); fi
    PARSER_ARGS=()
    if [ -n '$REASONING_PARSER' ]; then PARSER_ARGS+=(--reasoning-parser '$REASONING_PARSER'); fi
    if [ -n '$TOOL_PARSER' ]; then PARSER_ARGS+=(--tool-call-parser '$TOOL_PARSER' --enable-auto-tool-choice); fi
    if [ -n '$REASONING_CONFIG' ]; then PARSER_ARGS+=(--reasoning-config '$REASONING_CONFIG'); fi
    cd /
    exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve '$MODEL_PATH' \
      --served-model-name '$SERVED_NAME' --host 0.0.0.0 --port '$PORT' \
      --cpu-offload-gb 0 \
      --kv-cache-dtype '$KV_DTYPE' --block-size 256 --load-format safetensors \
      --tensor-parallel-size '$TP' --decode-context-parallel-size '$DCP' --moe-backend '$MOE_BACKEND' --linear-backend '$LINEAR_BACKEND' \
      --gpu-memory-utilization '$GPU_UTIL' --max-model-len '$MAXLEN' --max-num-seqs '$MAX_SEQS' \
      --enable-chunked-prefill --max-num-batched-tokens '$MAX_BATCHED' \
      --max-cudagraph-capture-size '$GRAPH_CAP' --attention-backend '$ATTN_BACKEND' \
      -cc.cudagraph_mode=PIECEWISE \
      --compilation-config '{\"custom_ops\":[\"all\"]}' \
      --enable-flashinfer-autotune \
      \"\${HF_ARGS[@]}\" \"\${PARSER_ARGS[@]}\" \"\${SPEC_ARGS[@]}\"
  "
echo "Launched $NAME (tp=$TP util=$GPU_UTIL maxlen=$MAXLEN dcp=$DCP mtp=$MTP num_spec=$NUM_SPEC batched=$MAX_BATCHED)"
echo "watch boot:  docker logs -f $NAME"
echo "smoke test:  curl -s http://127.0.0.1:$PORT/v1/models"
echo "completion:  curl -s http://127.0.0.1:$PORT/v1/chat/completions -d '{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"what is the capital of kentucky?\"}]}'"
