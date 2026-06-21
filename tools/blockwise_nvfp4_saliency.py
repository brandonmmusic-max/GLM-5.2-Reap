#!/usr/bin/env python3
"""Block-wise real saliency for GLM-5.2 ModelOpt NVFP4 experts.

This avoids whole-model construction.  The runner keeps the official
Transformers GLM-MoE-DSA decoder-layer math for dense/attention/shared paths,
but replaces the routed expert container with a saliency-aware module that
dequantizes the source NVFP4 expert tensors from safetensors and executes the
actual expert forward on GPU.  Consecutive decoder layers are kept resident in
VRAM by block, while only block-boundary activations are cached in CPU RAM.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.masking_utils import create_causal_mask
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaRotaryEmbedding,
    GlmMoeDsaTopkRouter,
)


BASE = Path("/home/brandonmusic/models/GLM-5.2-NVFP4")
CALIB = Path("work/calibration/reap_recall_calib.jsonl")
SUMMARY = Path("work/calibration/reap_recall_calib_summary.json")
OUT = Path("work/saliency/reap_recall_real_saliency.json")

MOE_LAYERS = list(range(3, 78))
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def load_index(model_dir: Path) -> dict[str, str]:
    return json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]


class TensorLoader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.index = load_index(model_dir)

    def tensor(self, name: str) -> torch.Tensor:
        shard = self.index[name]
        with safe_open(self.model_dir / shard, framework="pt") as f:
            return f.get_tensor(name)


def nvfp4_dequant(weight_packed: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Dequantize ModelOpt/Compressed-Tensors NVFP4 packed weights to BF16.

    Packed shape is [out, in/2].  Scale shape is [out, in/16] and scale2 is a
    scalar.  The stored local scales are FP8; casting to float32 gives the
    numeric scale used by ModelOpt's NVFP4QTensor.dequantize path.
    """
    packed = weight_packed.to(device=device, non_blocking=True)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    n_rows, half_cols = packed.shape
    vals = FP4_VALUES.to(device=device)[torch.stack((low, high), dim=-1).reshape(n_rows, half_cols * 2).long()]
    block = vals.view(n_rows, -1, 16)
    local_scale = scale.to(device=device, dtype=torch.float32, non_blocking=True)
    global_scale = scale2.to(device=device, dtype=torch.float32, non_blocking=True)
    return (block * (local_scale * global_scale).unsqueeze(-1)).reshape(n_rows, half_cols * 2).to(torch.bfloat16)


class DistributedNVFP4Experts(torch.nn.Module):
    def __init__(
        self,
        loader: TensorLoader,
        layer_idx: int,
        devices: list[torch.device],
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 256,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.devices = devices
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.active_mask_flat: torch.Tensor | None = None

        self.expert_ids_by_device: list[list[int]] = [[] for _ in devices]
        for expert_id in range(num_experts):
            self.expert_ids_by_device[expert_id % len(devices)].append(expert_id)

        self.gate_up: list[torch.Tensor] = []
        self.down: list[torch.Tensor] = []
        self.score_sum: list[torch.Tensor] = []
        self.score_count: list[torch.Tensor] = []

        t0 = time.time()
        for dev_idx, device in enumerate(devices):
            ids = self.expert_ids_by_device[dev_idx]
            gate_up = torch.empty((len(ids), 2 * intermediate_size, hidden_size), dtype=torch.bfloat16, device=device)
            down = torch.empty((len(ids), hidden_size, intermediate_size), dtype=torch.bfloat16, device=device)
            for local_idx, expert_id in enumerate(ids):
                base = f"model.layers.{layer_idx}.mlp.experts.{expert_id}"
                gate = self._load_proj(loader, f"{base}.gate_proj", device)
                up = self._load_proj(loader, f"{base}.up_proj", device)
                down_w = self._load_proj(loader, f"{base}.down_proj", device)
                gate_up[local_idx, :intermediate_size].copy_(gate)
                gate_up[local_idx, intermediate_size:].copy_(up)
                down[local_idx].copy_(down_w)
                del gate, up, down_w
            self.gate_up.append(gate_up)
            self.down.append(down)
            self.score_sum.append(torch.zeros(len(ids), dtype=torch.float64, device=device))
            self.score_count.append(torch.zeros(len(ids), dtype=torch.float64, device=device))
            torch.cuda.synchronize(device)
        print(f"  loaded NVFP4 experts for layer {layer_idx} across {len(devices)} GPUs in {time.time() - t0:.1f}s", flush=True)

    @staticmethod
    def _load_proj(loader: TensorLoader, prefix: str, device: torch.device) -> torch.Tensor:
        return nvfp4_dequant(
            loader.tensor(f"{prefix}.weight"),
            loader.tensor(f"{prefix}.weight_scale"),
            loader.tensor(f"{prefix}.weight_scale_2"),
            device,
        )

    def _device_forward(
        self,
        dev_idx: int,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        active_mask_flat: torch.Tensor | None,
    ) -> torch.Tensor:
        device = self.devices[dev_idx]
        local_final = torch.zeros((hidden_states.shape[0], self.hidden_size), dtype=hidden_states.dtype, device=device)
        hidden_d = hidden_states.to(device, non_blocking=True)
        top_idx_d = top_k_index.to(device, non_blocking=True)
        top_w_d = top_k_weights.to(device, non_blocking=True)
        active_d = active_mask_flat.to(device, non_blocking=True) if active_mask_flat is not None else None

        for local_idx, expert_id in enumerate(self.expert_ids_by_device[dev_idx]):
            token_idx, top_k_pos = torch.where(top_idx_d == expert_id)
            if token_idx.numel() == 0:
                continue
            current_state = hidden_d[token_idx]
            gate, up = F.linear(current_state, self.gate_up[dev_idx][local_idx]).chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.down[dev_idx][local_idx])
            weights = top_w_d[token_idx, top_k_pos]

            if active_d is not None:
                active = active_d[token_idx]
                if active.any():
                    contrib = weights[active].to(torch.float32) * expert_output[active].to(torch.float32).norm(dim=-1)
                    self.score_sum[dev_idx][local_idx] += contrib.to(torch.float64).sum()
                    self.score_count[dev_idx][local_idx] += active.to(torch.float64).sum()
            else:
                contrib = weights.to(torch.float32) * expert_output.to(torch.float32).norm(dim=-1)
                self.score_sum[dev_idx][local_idx] += contrib.to(torch.float64).sum()
                self.score_count[dev_idx][local_idx] += float(token_idx.numel())

            local_final.index_add_(0, token_idx, (expert_output * weights[:, None]).to(local_final.dtype))

        return local_final.to(hidden_states.device, non_blocking=True)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        active = self.active_mask_flat
        with ThreadPoolExecutor(max_workers=len(self.devices)) as ex:
            futs = [
                ex.submit(self._device_forward, dev_idx, hidden_states, top_k_index, top_k_weights, active)
                for dev_idx in range(len(self.devices))
            ]
            for fut in futs:
                final_hidden_states.add_(fut.result())
        return final_hidden_states

    def scores(self) -> tuple[list[float], list[float], list[float]]:
        sums = torch.zeros(self.num_experts, dtype=torch.float64)
        counts = torch.zeros(self.num_experts, dtype=torch.float64)
        for dev_idx, ids in enumerate(self.expert_ids_by_device):
            ss = self.score_sum[dev_idx].detach().cpu()
            cc = self.score_count[dev_idx].detach().cpu()
            for local_idx, expert_id in enumerate(ids):
                sums[expert_id] = ss[local_idx]
                counts[expert_id] = cc[local_idx]
        mean = torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))
        return mean.tolist(), sums.tolist(), counts.tolist()


def load_layer_weights(layer: torch.nn.Module, loader: TensorLoader, layer_idx: int, device: torch.device, sparse: bool) -> None:
    prefix = f"model.layers.{layer_idx}."
    state: dict[str, torch.Tensor] = {}
    for name in loader.index:
        if not name.startswith(prefix):
            continue
        if sparse and ".mlp.experts." in name:
            continue
        short = name[len(prefix) :]
        try:
            state[short] = loader.tensor(name).to(device=device, non_blocking=True)
        except KeyError:
            continue
    missing, unexpected = layer.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if not (sparse and m.startswith("mlp.experts."))]
    real_unexpected = [u for u in unexpected if not (".input_scale" in u or ".weight_scale" in u)]
    if real_missing:
        raise RuntimeError(f"Layer {layer_idx} missing weights: {real_missing[:10]}")
    if real_unexpected:
        # Shared DSA layers have indexer weights in the checkpoint but no module.
        bad = [u for u in real_unexpected if not u.startswith("self_attn.indexer.")]
        if bad:
            raise RuntimeError(f"Layer {layer_idx} unexpected weights: {bad[:10]}")


def validate_corpus(summary_path: Path, calib_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    if summary.get("fallback_used"):
        raise SystemExit("Calibration fallback_used is true; refusing saliency.")
    gate = summary.get("legal_gate", {})
    sampled = gate.get("neo4j_sampled", {})
    if gate.get("axis2_count_after_real_sources", 0) <= 0:
        raise SystemExit("Axis 2 legal gate empty; refusing saliency.")
    if gate.get("cap_sampled", 0) <= 0 or sampled.get("Statute.doc_text", 0) <= 0 or sampled.get("Case.summary", 0) <= 0:
        raise SystemExit("Axis 2 legal gate lacks CAP cases or Neo4j statutes/cases; refusing saliency.")
    print("MILESTONE corpus assembled", flush=True)
    print(json.dumps({"axis_counts": summary.get("axis_counts"), "source_counts": summary.get("source_counts"), "legal_gate": gate}, indent=2), flush=True)
    if not calib_path.exists():
        raise SystemExit(f"Missing calibration JSONL: {calib_path}")
    return summary


def load_calibration(path: Path, max_samples: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open() as f:
        for line in f:
            if max_samples is not None and len(rows) >= max_samples:
                break
            row = json.loads(line)
            rows.append({"axis": row.get("axis", ""), "source": row.get("source", ""), "text": row["text"]})
    return rows


def build_keep_map(scores: dict[int, list[float]], n_experts: int) -> dict[int, list[int]]:
    keep: dict[int, list[int]] = {}
    for layer in MOE_LAYERS:
        vals = scores[layer]
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        keep[layer] = sorted(order[:n_experts])
    # MTP layer 78 is not executed in the main forward.  Use the final main
    # layer's real saliency order so the draft layer remains structurally valid
    # without inventing router/static scores.
    keep[78] = list(keep[77])
    return keep


def write_keep_map(path: Path, keep: dict[int, list[int]], scores: dict[int, list[float]], counts: dict[int, list[float]], n_experts: int) -> None:
    meta = {
        "n_experts": n_experts,
        "method": "real_nvfp4_gpu_forward_saliency_blockwise",
        "score_formula": "mean_over_routed_active_tokens(topk_router_weight * l2_norm(unweighted_expert_output))",
        "layers_scored": list(MOE_LAYERS),
        "mtp_layer_78_policy": "kept same old expert IDs as layer 77 because MTP is not part of the main model forward",
        "keep_map": {str(k): v for k, v in keep.items()},
        "scores": {str(k): v for k, v in scores.items()},
        "counts": {str(k): v for k, v in counts.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def consecutive_blocks(layers: list[int], block_size: int) -> list[list[int]]:
    if block_size <= 0:
        raise ValueError("--block-size must be positive")
    blocks: list[list[int]] = []
    cur: list[int] = []
    prev: int | None = None
    for layer in layers:
        if cur and (len(cur) >= block_size or (prev is not None and layer != prev + 1)):
            blocks.append(cur)
            cur = []
        cur.append(layer)
        prev = layer
    if cur:
        blocks.append(cur)
    return blocks


def dsa_group_units(layers: list[int], indexer_types: list[str]) -> list[list[int]]:
    """Group layers so shared DSA layers never cross a block boundary."""
    selected = sorted(dict.fromkeys(layers))
    selected_set = set(selected)
    units: list[list[int]] = []
    i = 0
    while i < len(selected):
        layer = selected[i]
        if indexer_types[layer] == "shared":
            raise ValueError(
                f"Selected layer {layer} is a shared DSA layer without its full-indexer predecessor "
                "in the same block. Include the previous full indexer layer or use the full 0-77 run."
            )
        unit = [layer]
        j = i + 1
        while j < len(selected):
            nxt = selected[j]
            if nxt != unit[-1] + 1:
                break
            if indexer_types[nxt] != "shared":
                break
            unit.append(nxt)
            j += 1
        if any(x not in selected_set for x in unit):
            raise ValueError(f"Internal layer-selection error for DSA unit {unit}")
        units.append(unit)
        i = j
    return units


def pack_dsa_blocks(layers: list[int], indexer_types: list[str], block_size: int) -> list[list[int]]:
    if block_size <= 0:
        raise ValueError("--block-size must be positive")
    units = dsa_group_units(layers, indexer_types)
    blocks: list[list[int]] = []
    cur: list[int] = []
    for unit in units:
        if cur and len(cur) + len(unit) > block_size:
            blocks.append(cur)
            cur = []
        cur.extend(unit)
    if cur:
        blocks.append(cur)
    return blocks


def iter_jsonl_rows(path: Path, max_samples: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open() as f:
        for line in f:
            if max_samples is not None and len(rows) >= max_samples:
                break
            row = json.loads(line)
            rows.append({"axis": row.get("axis", ""), "source": row.get("source", ""), "text": row["text"]})
    return rows


def encode_rows(tokenizer: Any, rows: list[dict[str, str]], max_len: int, workers: int) -> list[torch.Tensor]:
    def one(text: str) -> list[int]:
        ids = tokenizer(text, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
        if len(ids) > max_len:
            raise ValueError(
                f"Calibration sample tokenized to {len(ids)} tokens, exceeding --max-len {max_len}. "
                "Refusing to truncate because this run requires no truncation."
            )
        if not ids:
            ids = [tokenizer.eos_token_id]
        return ids

    if workers <= 1:
        encoded = [one(row["text"]) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            encoded = list(ex.map(one, (row["text"] for row in rows)))
    return [torch.tensor(ids, dtype=torch.long) for ids in encoded]


def make_sample_groups(lengths: list[int], hidden_size: int, budget_gb: float, forced_group_size: int | None) -> list[list[int]]:
    indices = list(range(len(lengths)))
    if forced_group_size and forced_group_size > 0:
        return [indices[i : i + forced_group_size] for i in range(0, len(indices), forced_group_size)]

    token_budget = int((budget_gb * (1024**3)) // (hidden_size * 2 * 2))
    # Two CPU BF16 hidden buffers are alive at a block boundary.
    if token_budget <= 0:
        return [[i] for i in indices]
    groups: list[list[int]] = []
    cur: list[int] = []
    cur_tokens = 0
    for idx in indices:
        n = lengths[idx]
        if cur and cur_tokens + n > token_budget:
            groups.append(cur)
            cur = []
            cur_tokens = 0
        cur.append(idx)
        cur_tokens += n
    if cur:
        groups.append(cur)
    return groups


def make_batches(indices: list[int], lengths: list[int], batch_size: int, max_batch_tokens: int | None) -> list[list[int]]:
    ordered = sorted(indices, key=lambda i: lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for idx in ordered:
        next_max = max(cur_max, lengths[idx])
        token_count = next_max * (len(cur) + 1)
        if cur and (len(cur) >= batch_size or (max_batch_tokens and token_count > max_batch_tokens)):
            batches.append(cur)
            cur = []
            cur_max = 0
            next_max = lengths[idx]
        cur.append(idx)
        cur_max = next_max
    if cur:
        batches.append(cur)
    return batches


def pad_ids(batch: list[int], encoded: list[torch.Tensor], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(encoded[i].numel()) for i in batch)
    ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for row, idx in enumerate(batch):
        vals = encoded[idx]
        n = int(vals.numel())
        ids[row, :n] = vals
        mask[row, :n] = True
    return ids, mask


def pad_hidden(batch: list[int], hidden_cache: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(hidden_cache[i].shape[0]) for i in batch)
    hidden_size = int(next(iter(hidden_cache.values())).shape[-1])
    hs = torch.zeros((len(batch), max_len, hidden_size), dtype=torch.bfloat16)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for row, idx in enumerate(batch):
        vals = hidden_cache[idx]
        n = int(vals.shape[0])
        hs[row, :n].copy_(vals)
        mask[row, :n] = True
    return hs, mask


def add_scores(
    global_sums: dict[int, list[float]],
    global_counts: dict[int, list[float]],
    layer_idx: int,
    sums: list[float],
    counts: list[float],
) -> None:
    if layer_idx not in global_sums:
        global_sums[layer_idx] = [0.0] * len(sums)
        global_counts[layer_idx] = [0.0] * len(counts)
    for i, val in enumerate(sums):
        global_sums[layer_idx][i] += float(val)
    for i, val in enumerate(counts):
        global_counts[layer_idx][i] += float(val)


def means_from_sums(global_sums: dict[int, list[float]], global_counts: dict[int, list[float]]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for layer, sums in global_sums.items():
        counts = global_counts[layer]
        out[layer] = [float(s / c) if c > 0 else 0.0 for s, c in zip(sums, counts)]
    return out


def write_partial(
    out_path: Path,
    method: str,
    args: argparse.Namespace,
    summary: dict[str, Any],
    rows: list[dict[str, str]],
    lengths: list[int],
    active_tokens: int,
    axis_counts: Counter,
    source_counts: Counter,
    global_sums: dict[int, list[float]],
    global_counts: dict[int, list[float]],
) -> None:
    scores = means_from_sums(global_sums, global_counts)
    partial = {
        "method": method,
        "base": str(args.base),
        "calib": str(args.calib),
        "summary": summary,
        "rows": len(rows),
        "axis_counts_used": dict(axis_counts),
        "source_counts_used": dict(source_counts),
        "max_len": args.max_len,
        "natural_length_tokenization": True,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "active_tokens": active_tokens,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "block_size": args.block_size,
        "attn_impl": args.attn_impl,
        "scores": {str(k): v for k, v in scores.items()},
        "sums": {str(k): v for k, v in global_sums.items()},
        "counts": {str(k): v for k, v in global_counts.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(partial, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--calib", type=Path, default=CALIB)
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n-experts", type=int, default=190)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-batch-tokens", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=12)
    ap.add_argument("--layers", default="0-77")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--tokenizer-workers", type=int, default=min(48, os.cpu_count() or 1))
    ap.add_argument("--hidden-cache-gb", type=float, default=340.0)
    ap.add_argument("--sample-group-size", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    summary = validate_corpus(args.summary, args.calib)
    rows = iter_jsonl_rows(args.calib, args.max_samples)
    axis_counts = Counter(row["axis"] for row in rows)
    source_counts = Counter(f"{row['axis']}|{row['source'].split(':', 1)[0]}" for row in rows)
    print(f"loaded calibration rows={len(rows)} axis_counts={dict(axis_counts)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = encode_rows(tokenizer, rows, args.max_len, args.tokenizer_workers)
    lengths = [int(x.numel()) for x in encoded]
    active_tokens = int(sum(lengths))
    print(
        f"tokenized samples={len(encoded)} natural_len_min={min(lengths)} "
        f"natural_len_p50={sorted(lengths)[len(lengths)//2]} max={max(lengths)} "
        f"active_tokens={active_tokens}",
        flush=True,
    )

    cfg = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
    cfg._attn_implementation = args.attn_impl
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if len(devices) < 4:
        raise SystemExit(f"Expected 4 CUDA devices, got {len(devices)}")
    main_device = devices[0]
    loader = TensorLoader(args.base)

    rotary = GlmMoeDsaRotaryEmbedding(cfg).to(main_device)
    selected_layers = parse_layers(args.layers)
    blocks = pack_dsa_blocks(selected_layers, list(cfg.indexer_types), args.block_size)
    print(f"DSA-aligned blocks: {['%s-%s' % (b[0], b[-1]) for b in blocks]}", flush=True)

    saliency_sums: dict[int, list[float]] = {}
    saliency_counts: dict[int, list[float]] = {}
    if args.resume and args.out.exists():
        old = json.loads(args.out.read_text())
        saliency_sums = {int(k): v for k, v in old.get("sums", {}).items()}
        saliency_counts = {int(k): v for k, v in old.get("counts", {}).items()}
    resume_complete_layers = set(saliency_sums)

    sample_groups = make_sample_groups(lengths, cfg.hidden_size, args.hidden_cache_gb, args.sample_group_size)
    if len(sample_groups) > 1:
        print(
            f"sample grouping enabled groups={len(sample_groups)} hidden_cache_budget_gb={args.hidden_cache_gb}; "
            "weights will be reloaded per sample group",
            flush=True,
        )
    else:
        print("single sample group fits CPU hidden-cache budget; each weight block loads once", flush=True)

    embed = loader.tensor("model.embed_tokens.weight").to(device=main_device, dtype=torch.bfloat16)
    method = "real_nvfp4_gpu_forward_saliency_blockwise_natural_length"

    for group_no, group_indices in enumerate(sample_groups, 1):
        group_tokens = sum(lengths[i] for i in group_indices)
        print(
            f"sample group {group_no}/{len(sample_groups)} samples={len(group_indices)} tokens={group_tokens}",
            flush=True,
        )
        hidden_cache: dict[int, torch.Tensor] | None = None
        for block_no, block in enumerate(blocks, 1):
            print(
                f"loading block {block_no}/{len(blocks)} layers={block[0]}-{block[-1]} count={len(block)}",
                flush=True,
            )
            block_t0 = time.time()
            entries: list[tuple[int, torch.nn.Module, DistributedNVFP4Experts | None, float]] = []
            for layer_idx in block:
                sparse = cfg.mlp_layer_types[layer_idx] == "sparse"
                t_layer = time.time()
                cfg_layer = copy.deepcopy(cfg)
                if sparse:
                    # Keep a tiny placeholder expert table during construction, then
                    # restore the real 256-way router before loading checkpoint rows.
                    cfg_layer.num_local_experts = 1
                layer = GlmMoeDsaDecoderLayer(cfg_layer, layer_idx).to(main_device, dtype=torch.bfloat16).eval()
                if sparse:
                    layer.mlp.gate = GlmMoeDsaTopkRouter(cfg).to(main_device, dtype=torch.bfloat16).eval()
                    layer.mlp.n_routed_experts = cfg.n_routed_experts
                load_layer_weights(layer, loader, layer_idx, main_device, sparse=sparse)
                expert_mod: DistributedNVFP4Experts | None = None
                if sparse:
                    expert_mod = DistributedNVFP4Experts(
                        loader=loader,
                        layer_idx=layer_idx,
                        devices=devices,
                        hidden_size=cfg.hidden_size,
                        intermediate_size=cfg.moe_intermediate_size,
                        num_experts=cfg.n_routed_experts,
                    )
                    layer.mlp.experts = expert_mod
                entries.append((layer_idx, layer, expert_mod, t_layer))
            print(f"block {block[0]}-{block[-1]} resident in {time.time() - block_t0:.1f}s", flush=True)

            next_hidden: dict[int, torch.Tensor] = {}
            batches = make_batches(group_indices, lengths, args.batch_size, args.max_batch_tokens)
            for batch_no, batch in enumerate(batches, 1):
                if hidden_cache is None:
                    ids, am_cpu = pad_ids(batch, encoded, tokenizer.pad_token_id)
                    hs = F.embedding(ids.to(main_device, non_blocking=True), embed)
                else:
                    hs_cpu, am_cpu = pad_hidden(batch, hidden_cache)
                    hs = hs_cpu.to(main_device, non_blocking=True)
                    del hs_cpu
                am = am_cpu.to(main_device, non_blocking=True)
                bsz, seq_len = am.shape
                pos = torch.arange(seq_len, dtype=torch.long, device=main_device).unsqueeze(0).expand(bsz, -1)
                if args.attn_impl.startswith("flash") and bool(am.all()):
                    causal_mask = None
                else:
                    causal_mask = create_causal_mask(
                        config=cfg,
                        inputs_embeds=hs,
                        attention_mask=am.to(torch.long),
                        past_key_values=None,
                        position_ids=pos,
                    )
                pos_emb = rotary(hs, position_ids=pos)
                prev = None
                for layer_idx, layer, expert_mod, _ in entries:
                    if expert_mod is not None:
                        expert_mod.active_mask_flat = am.reshape(-1)
                    hs, prev = layer(
                        hs,
                        attention_mask=causal_mask,
                        position_ids=pos,
                        position_embeddings=pos_emb,
                        prev_topk_indices=prev,
                        use_cache=False,
                    )
                hs_cpu_out = hs.detach().cpu()
                for row, idx in enumerate(batch):
                    n = lengths[idx]
                    next_hidden[idx] = hs_cpu_out[row, :n].contiguous()
                del hs, hs_cpu_out, am, am_cpu, pos, causal_mask, pos_emb, prev
                if batch_no % 10 == 0 or batch_no == len(batches):
                    print(
                        f"  group {group_no}/{len(sample_groups)} block {block[0]}-{block[-1]} "
                        f"batch {batch_no}/{len(batches)}",
                        flush=True,
                    )

            hidden_cache = next_hidden

            for layer_idx, _layer, expert_mod, t_layer in entries:
                if expert_mod is None:
                    continue
                if layer_idx in resume_complete_layers:
                    print(f"layer {layer_idx}: existing resumed saliency preserved without double-counting", flush=True)
                    continue
                means, sums, counts = expert_mod.scores()
                if not all(math.isfinite(x) for x in means):
                    raise RuntimeError(f"Layer {layer_idx} produced non-finite saliency")
                add_scores(saliency_sums, saliency_counts, layer_idx, sums, counts)
                merged_scores = means_from_sums(saliency_sums, saliency_counts)[layer_idx]
                if max(merged_scores) <= 0 or len({round(x, 8) for x in merged_scores}) <= 4:
                    raise RuntimeError(
                        f"Layer {layer_idx} degenerate saliency: min={min(merged_scores)} max={max(merged_scores)}"
                    )
                sorted_scores = sorted(merged_scores)
                merged_counts = saliency_counts[layer_idx]
                print(
                    f"layer {layer_idx} real saliency: min={sorted_scores[0]:.6g} "
                    f"p50={sorted_scores[len(sorted_scores)//2]:.6g} max={sorted_scores[-1]:.6g} "
                    f"nonzero_counts={sum(1 for c in merged_counts if c > 0)} time={time.time() - t_layer:.1f}s",
                    flush=True,
                )

            write_partial(
                args.out,
                method,
                args,
                summary,
                rows,
                lengths,
                active_tokens,
                axis_counts,
                source_counts,
                saliency_sums,
                saliency_counts,
            )
            print(f"block {block[0]}-{block[-1]} done in {time.time() - block_t0:.1f}s", flush=True)

            del entries
            gc.collect()
            for device in devices:
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)

        del hidden_cache
        gc.collect()

    del embed
    saliency_scores = means_from_sums(saliency_sums, saliency_counts)
    missing = [layer for layer in MOE_LAYERS if layer not in saliency_scores]
    if missing:
        raise SystemExit(f"Completed replay but missing sparse saliency layers: {missing[:20]}")

    keep = build_keep_map(saliency_scores, args.n_experts)
    keep_path = Path("work") / f"reap_recall_keep_map_real_N{args.n_experts}.json"
    write_keep_map(keep_path, keep, saliency_scores, saliency_counts, args.n_experts)
    print("MILESTONE saliency done", flush=True)
    print(f"real saliency written: {args.out}", flush=True)
    print(f"real keep map written: {keep_path}", flush=True)


if __name__ == "__main__":
    main()
