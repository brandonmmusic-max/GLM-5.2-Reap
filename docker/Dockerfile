# GLM-5.2-NVFP4-REAP-Recall (N=172) — serving image
#
# This image carries vLLM b12x-fork with:
#   - nvfp4_ds_mla 4-bit MLA KV cache kernels (kv-cache-dtype=nvfp4_ds_mla)
#   - DCP global-topk remap (PR#30) for correct DCP>1 attention
#   - DCP shard-draft (PR#28) for MTP under DCP
#   - DSV4 dangling-</think> recovery patch
#   - B12X_MOE_FORCE_A16 long-context correctness path
#
# To rebuild from source, see https://github.com/brandonmmusic-max/GLM-5.2-Reap

# Published as: docker.io/verdictai/gloriousluminousmonotheism:latest
#
# Base: voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618
# + nvfp4_ds_mla KV overlay + PR#30 DCP global-topk + PR#28 shard-draft + dangling-think patch.

FROM voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618

# Overlays are baked into the published image. To reproduce locally:
#   1) install vllm-b12x at commit a86f74e
#   2) apply patches from this repo's vllm-patches/ directory
#   3) add the nvfp4_ds_mla KV cache kernel registration
# (See README for the full reproduction recipe.)

LABEL org.opencontainers.image.title="GLM-5.2-NVFP4-REAP-Recall serving image"
LABEL org.opencontainers.image.source="https://github.com/brandonmmusic-max/GLM-5.2-Reap"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.description="Serving stack for GLM-5.2-NVFP4-REAP-Recall (N=172) on consumer Blackwell. Carries the 4-bit MLA KV cache + DCP global-topk + shard-draft patches."
