from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from prometheus.distributed import get_tp_info
from prometheus.models.loader import drop_page_cache, iter_weight_files
from prometheus.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from prometheus.utils import cached_load_hf_config
from tqdm import tqdm

from ..qwen3_5_moe.weight import (
    _BF16_EXPERT_PART_RE,
    _Bf16ExpertPacker,
    _PACKED_EXPERT_PATTERN,
    _SCALE_SUFFIXES,
    _SHARED_GATE,
    _SHARED_UP,
    _try_fuse,
)
from .config import parse_config

# The ~102 GB hashed n-gram table never flows through the state dict: the shards are
# streamed into one pinned host tensor by Qwen4PleEmbedding.prepare().
_NGRAM_SHARD_RE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$")

# Hash constants recomputed in __init__ (pure functions of the config). The checkpoint
# ships layer_multipliers only; head sizes/offsets are never saved. All are dropped:
# the model derives them itself and they are not part of the BaseOP state dict.
_PLE_CONST_RE = re.compile(
    r"\.ple\.ple_embedding\.(layer_multipliers|ngram_heads_vocab_sizes|ngram_heads_offsets)$"
)

# QSA indexer weights (full-attention layers): NOT loaded in the dense-attention
# milestone -- attention over the full KV is identical to QSA's top-512-block
# selection for kv_len <= 2052, so short/medium contexts are exact without them.
_INDEXER_RE = re.compile(r"\.self_attn\.indexer\.")

# Flash-Next ships its routed experts PRE-STACKED whole-layer under a `.weight`-suffixed
# key (unlike original Qwen3-Next's per-expert parts and unlike the bank-side emitted
# names, which carry no suffix). Layout already matches the bank convention
# (gate_up [E, 2I, H] gate-first, down [E, H, I]) -- pass through minus the suffix.
_STACKED_EXPERT_RE = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)\.weight$"
)

# NVFP4 routed experts (modelopt checkpoint, e.g. RadixArk Flash-Next-NVFP4): per-expert,
# un-fused, under the raw ``model[.language_model].layers.N.mlp.experts.E.{proj}`` key --
# matched against the RAW weight_map key by the bank loader. The dense pass drops these
# tensors (and their modelopt scales); the pre-stacked bf16 layout above has no
# ``.mlp.experts.<int>.`` segment, so it is unaffected.
_NVFP4_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_EXPERT_KEY_RE = re.compile(
    r"^model\.(?:language_model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every Flash-Next layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)

# Gemma-style (1+weight) norms of the REUSED qwen3_5 modules (the loader bakes +1;
# GemmaRMSNorm scales by the raw weight). The qwen4_exp-native group norms
# (hc_norm / norm_key / norm_query / norm_conv) keep their weight raw and apply the
# +1 in float32 at runtime (Qwen4GroupRMSNorm).
_BAKED_GEMMA_SUFFIXES = (
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)


def _rename(raw_name: str) -> str | None:
    """HF key -> Prometheus state-dict key, or None to skip."""
    if raw_name.startswith(("mtp.", "model.visual.", "visual.")):
        return None
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    name = raw_name
    if name.startswith("model.language_model."):
        name = name[len("model.language_model.") :]
    if name == "lm_head.weight":
        return name
    if not name.startswith("model."):
        name = "model." + name  # text tower keys are model.model.* / model.layers.*
    return name


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pure-bf16 dense pass for Qwen3.8-Flash-Next.

    Reuses the qwen3_5_moe machinery for everything the two architectures share:
    q/k/v -> qkv_proj (q half 2x for the output gate), GDN in_proj_{qkv,z,b,a} ->
    in_proj, shared-expert gate|up merge, and the pre-fused stacked routed experts
    (``experts.{gate_up,down}_proj`` pass through to the offload banks verbatim)."""
    hf_config = cached_load_hf_config(model_path)
    config = parse_config(hf_config, model_path)
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading currently supports TP=1 only")

    shared_buf: dict[str, dict[str, torch.Tensor]] = {}
    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    bf16_packer = _Bf16ExpertPacker(config.num_experts)

    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            keyset = set(f.keys())
            for raw_name in f.keys():
                # Per-expert NVFP4 tensors (packed weights + modelopt scales) go to the
                # offload banks (load_nvfp4_expert_source_banks), never this dense pass.
                # A plain bf16 per-expert ``.weight`` (no scale sibling) falls through.
                if _NVFP4_EXPERT_RE.search(raw_name) and (
                    not raw_name.endswith(".weight")
                    or raw_name.removesuffix(".weight") + ".weight_scale" in keyset
                ):
                    continue
                if raw_name.endswith(_SCALE_SUFFIXES):
                    continue  # standalone modelopt scales are consumed with their .weight
                if (
                    _NGRAM_SHARD_RE.search(raw_name)
                    or _INDEXER_RE.search(raw_name)
                    or _PLE_CONST_RE.search(raw_name)
                ):
                    continue

                name = _rename(raw_name)
                if name is None:
                    continue

                if _BF16_EXPERT_PART_RE.match(name) is not None:
                    if not include_moe_experts:
                        continue  # routed experts live in the offload banks, not here
                    yield from bf16_packer.feed(name, f.get_tensor(raw_name))
                    continue

                if _STACKED_EXPERT_RE.match(name) is not None:
                    if not include_moe_experts:
                        continue  # routed experts live in the offload banks, not here
                    yield name[: -len(".weight")], f.get_tensor(raw_name)
                    continue

                is_expert = _PACKED_EXPERT_PATTERN.match(name) is not None
                if is_expert and not include_moe_experts:
                    continue
                if not is_expert and not include_non_moe:
                    continue

                tensor = f.get_tensor(raw_name)

                # merge shared-expert gate/up -> gate_up_proj
                if name.endswith(_SHARED_GATE) or name.endswith(_SHARED_UP):
                    prefix = name.rsplit(".mlp.shared_expert.", 1)[0]
                    slots = shared_buf.setdefault(prefix, {})
                    slots["gate" if name.endswith(_SHARED_GATE) else "up"] = tensor
                    if "gate" in slots and "up" in slots:
                        merged = torch.cat([slots["gate"], slots["up"]], dim=0)
                        del shared_buf[prefix]
                        yield f"{prefix}.mlp.shared_expert.gate_up_proj.weight", merged
                    continue

                # fuse q/k/v -> qkv_proj and GDN in_proj_{qkv,z,b,a} -> in_proj
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield fused
                    continue

                if name.endswith(_BAKED_GEMMA_SUFFIXES):
                    tensor = tensor + 1.0  # (1 + weight) baked for GemmaRMSNorm

                yield name, tensor

    assert not shared_buf, f"Incomplete shared-expert merges: {list(shared_buf.keys())}"
    assert not fuse_buf, f"Incomplete projection fusions: {list(fuse_buf.keys())}"
    parts, layers = bf16_packer.leftovers()
    assert not parts, f"Incomplete per-expert parts: {parts[:5]}..."
    assert not layers, f"Incomplete per-expert layers: {layers}"


def load_nvfp4_expert_sources(
    model_path: str, config, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """CPU NVFP4 expert source banks for the offload cache; see
    ``prometheus.models.nvfp4_banks.load_nvfp4_expert_source_banks``."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
):
    """parallel: same NVFP4 source banks via the common chunked multi-threaded O_DIRECT reader."""
    from prometheus.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = ["iter_weights", "load_nvfp4_expert_sources", "load_nvfp4_expert_sources_parallel"]
