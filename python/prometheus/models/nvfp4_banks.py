from __future__ import annotations

import collections
import json
import os
import re
from dataclasses import dataclass
from typing import Callable

import safetensors
import torch
from prometheus.utils import download_hf_weight
from tqdm import tqdm

LayerToBank = Callable[[int, object], int | None]
DropPageCache = Callable[[str], None]


@dataclass(frozen=True)
class Nvfp4ExpertSourceSpec:
    key_pattern: re.Pattern[str]
    proj_to_role: dict[str, str]
    layer_to_bank: LayerToBank
    desc: str


def _num_moe_layers(config) -> int:
    value = getattr(config, "num_moe_layers", None)
    if value is not None:
        return int(value)
    return int(config.num_layers) - int(getattr(config, "first_k_dense_replace", 0))


def _bank_layer(spec: Nvfp4ExpertSourceSpec, layer: int, config) -> int | None:
    bank_layer = spec.layer_to_bank(layer, config)
    if bank_layer is None:
        return None
    num_layers = _num_moe_layers(config)
    if bank_layer < 0 or bank_layer >= num_layers:
        raise ValueError(
            f"{spec.desc}: bank layer {bank_layer} for checkpoint layer {layer} "
            f"is outside [0, {num_layers})"
        )
    return bank_layer


def _alloc_nvfp4_host_banks(num_layers: int, E: int, H: int, I: int):
    """6 NVFP4 source banks, one ``[E, ...]`` tensor per layer (independent allocations),
    unpinned (pin-after-fill): register only after fill to skip cudaHostAlloc's slow
    commit. Caller fills each layer's ``.tensor`` then pins it (per-layer, via
    ``PinPipeline``, as its writes complete)."""
    from prometheus.moe.host_banks import alloc_layer_banks

    fp8 = torch.float8_e4m3fn
    return alloc_layer_banks({
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 16), fp8),
        "gate_up_global": ((E, 2 * I), torch.float16),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 16), fp8),
        "down_global": ((E, H), torch.float16),
    }, num_layers)


def _weight_map(folder: str) -> dict[str, str]:
    """The index.json weight_map, or a synthesized one for index-less checkpoints
    (single-file ct exports, e.g. Ttimms REAP NVFP4A16: one model.safetensors)."""
    index_path = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            return json.load(f)["weight_map"]
    import glob as _glob

    weight_map: dict[str, str] = {}
    for shard in sorted(
        os.path.basename(p) for p in _glob.glob(os.path.join(folder, "*.safetensors"))
    ):
        with safetensors.safe_open(os.path.join(folder, shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                weight_map[name] = shard
    if not weight_map:
        raise FileNotFoundError(f"no safetensors shards found in {folder}")
    return weight_map


def load_nvfp4_expert_source_banks(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    drop_page_cache: DropPageCache,
    primary: bool,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """Build the 6 native NVFP4 source banks by streaming checkpoint shards (serial per-shard read).

    ModelOpt row layout: gate/up fused on the output-row axis, down separate; the per-tensor
    global scale (weight_scale_2) is kept as a separate per-output-row FP16 bank (``*_global``),
    so dequant is ``fp4 * block_scale * global``. Each bank is one ``[E, ...]`` tensor per
    layer, indexed by ``[bank_layer][expert]``. (The marlin/b12x backends repack these and
    fold the global into per-expert alphas; see moe/nvfp4_backends.py.)

    ``layer_sink=None`` (serving): pin each bank layer as its writes complete, via an
    internally-owned :class:`PinPipeline`. ``layer_sink`` given (converter; for
    marlin/b12x the provider wraps it in a per-layer repacking sink first): the
    completion tracker fires into it instead -- nothing here is pinned, and the sink
    may release banks it has written out, so the returned tensors are only valid
    until then (the caller owns that tradeoff).
    """
    folder = download_hf_weight(model_path)
    weight_map = _weight_map(folder)

    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    num_layers = _num_moe_layers(config)

    for shard in sorted(set(weight_map.values())):
        drop_page_cache(os.path.join(folder, shard))

    weight_shards: dict[str, list[tuple[str, re.Match[str], int]]] = collections.defaultdict(list)
    global_shards: dict[str, list[tuple[str, re.Match[str], int]]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        bank_layer = _bank_layer(spec, layer, config)
        if bank_layer is None:
            continue
        proj = match.group("proj")
        if proj not in spec.proj_to_role:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert projection {proj!r}")
        kind = match.group("kind")
        if kind in ("weight_scale_2", "weight_global_scale"):
            global_shards[shard].append((name, match, bank_layer))
        elif kind in {"weight", "weight_scale", "weight_packed"}:
            weight_shards[shard].append((name, match, bank_layer))
        else:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")

    globals_map: dict[tuple[int, int, str], torch.Tensor] = {}
    for shard in sorted(global_shards):
        path = os.path.join(folder, shard)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name, match, _bank_layer_id in global_shards[shard]:
                key = (
                    int(match.group("layer")),
                    int(match.group("expert")),
                    match.group("proj"),
                )
                t = f.get_tensor(name)
                if match.group("kind") == "weight_global_scale":
                    # ct global is the quant-side scalar; native per-row global = 1/g
                    t = (1.0 / t.reshape(1).to(torch.float32)).to(torch.float16)
                globals_map[key] = t
        drop_page_cache(path)

    _hb = _alloc_nvfp4_host_banks(num_layers, E, H, I)  # unpinned; pinned after fill
    gate_up_packed = [b.tensor for b in _hb["gate_up_packed"]]
    gate_up_scale = [b.tensor for b in _hb["gate_up_scale"]]
    gate_up_global = [b.tensor for b in _hb["gate_up_global"]]
    down_packed = [b.tensor for b in _hb["down_packed"]]
    down_scale = [b.tensor for b in _hb["down_scale"]]
    down_global = [b.tensor for b in _hb["down_global"]]

    from prometheus.moe.host_banks import LayerCompletionTracker, PinPipeline

    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, _hb, sink)
        placed = 0
        for shard in tqdm(sorted(weight_shards), desc=f"Loading {spec.desc}", disable=not primary):
            path = os.path.join(folder, shard)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name, match, bank_layer_id in weight_shards[shard]:
                    layer = int(match.group("layer"))
                    expert = int(match.group("expert"))
                    proj = match.group("proj")
                    role = spec.proj_to_role[proj]
                    kind = match.group("kind")
                    tensor = f.get_tensor(name)
                    if kind in ("weight", "weight_packed"):
                        if role == "gate":
                            gate_up_packed[bank_layer_id][expert, :I] = tensor
                        elif role == "up":
                            gate_up_packed[bank_layer_id][expert, I:] = tensor
                        elif role == "down":
                            down_packed[bank_layer_id][expert] = tensor
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    else:
                        global_scale = globals_map[(layer, expert, proj)]
                        if role == "gate":
                            gate_up_scale[bank_layer_id][expert, :I] = tensor
                            gate_up_global[bank_layer_id][expert, :I] = global_scale
                        elif role == "up":
                            gate_up_scale[bank_layer_id][expert, I:] = tensor
                            gate_up_global[bank_layer_id][expert, I:] = global_scale
                        elif role == "down":
                            down_scale[bank_layer_id][expert] = tensor
                            down_global[bank_layer_id][expert] = global_scale
                        else:
                            raise ValueError(f"{spec.desc}: unknown projection role {role!r}")
                    tracker.note(bank_layer_id)
                    placed += 1
            drop_page_cache(path)
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = num_layers * E * 6
    assert placed == expected, f"{spec.desc}: loaded {placed} expert tensors, expected {expected}"
    return {
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global": gate_up_global,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global": down_global,
    }


def load_nvfp4_expert_source_banks_parallel(
    model_path: str,
    config,
    spec: Nvfp4ExpertSourceSpec,
    *,
    drop_page_cache: DropPageCache,
    primary: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """parallel counterpart of :func:`load_nvfp4_expert_source_banks`, byte-for-byte same
    placement. bulk weight/weight_scale read via chunked multi-threaded O_DIRECT reader
    (iter_expert_tensors_parallel); tiny globals (``weight_scale_2``) stay serial (negligible
    bytes). ``layer_sink``: see :func:`load_nvfp4_expert_source_banks`."""
    from prometheus.models.weight import iter_expert_tensors_parallel

    folder = download_hf_weight(model_path)
    weight_map = _weight_map(folder)

    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    num_layers = _num_moe_layers(config)

    weight_info: dict[str, tuple[re.Match[str], int]] = {}  # name -> (match, bank_layer)
    global_names_by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for name, shard in weight_map.items():
        match = spec.key_pattern.match(name)
        if match is None:
            continue
        bank_layer = _bank_layer(spec, int(match.group("layer")), config)
        if bank_layer is None:
            continue
        kind = match.group("kind")
        if kind in ("weight_scale_2", "weight_global_scale"):
            global_names_by_shard[shard].append(name)
        elif kind in {"weight", "weight_scale", "weight_packed"}:
            weight_info[name] = (match, bank_layer)
        else:
            raise ValueError(f"{spec.desc}: unknown NVFP4 expert tensor kind {kind!r}")

    # Pass 1: tiny per-tensor global scales (serial; data is scalar-per-expert).
    globals_map: dict[tuple[int, int, str], torch.Tensor] = {}
    for shard in sorted(global_names_by_shard):
        path = os.path.join(folder, shard)
        drop_page_cache(path)
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for name in global_names_by_shard[shard]:
                m = spec.key_pattern.match(name)
                t = f.get_tensor(name)
                if m.group("kind") == "weight_global_scale":
                    # ct quant-side scalar -> native per-row reciprocal global
                    t = (1.0 / t.reshape(1).to(torch.float32)).to(torch.float16)
                globals_map[(int(m.group("layer")), int(m.group("expert")), m.group("proj"))] = t
        drop_page_cache(path)

    _hb = _alloc_nvfp4_host_banks(num_layers, E, H, I)  # unpinned; pinned after fill
    gate_up_packed = [b.tensor for b in _hb["gate_up_packed"]]
    gate_up_scale = [b.tensor for b in _hb["gate_up_scale"]]
    gate_up_global = [b.tensor for b in _hb["gate_up_global"]]
    down_packed = [b.tensor for b in _hb["down_packed"]]
    down_scale = [b.tensor for b in _hb["down_scale"]]
    down_global = [b.tensor for b in _hb["down_global"]]

    from prometheus.moe.host_banks import LayerCompletionTracker, PinPipeline

    # Pass 2: bulk weight/weight_scale via the common parallel reader; place by name.
    def _load(sink) -> int:
        tracker = LayerCompletionTracker(E * 6, _hb, sink)
        placed = 0
        for name, tensor in iter_expert_tensors_parallel(
            folder, lambda n: n in weight_info, workers=workers, chunk=chunk
        ):
            match, bank_layer_id = weight_info[name]
            layer = int(match.group("layer"))
            expert = int(match.group("expert"))
            proj = match.group("proj")
            role = spec.proj_to_role[proj]
            kind = match.group("kind")
            if kind in ("weight", "weight_packed"):
                if role == "gate":
                    gate_up_packed[bank_layer_id][expert, :I] = tensor
                elif role == "up":
                    gate_up_packed[bank_layer_id][expert, I:] = tensor
                else:
                    down_packed[bank_layer_id][expert] = tensor
            else:
                g = globals_map[(layer, expert, proj)]
                if role == "gate":
                    gate_up_scale[bank_layer_id][expert, :I] = tensor
                    gate_up_global[bank_layer_id][expert, :I] = g
                elif role == "up":
                    gate_up_scale[bank_layer_id][expert, I:] = tensor
                    gate_up_global[bank_layer_id][expert, I:] = g
                else:
                    down_scale[bank_layer_id][expert] = tensor
                    down_global[bank_layer_id][expert] = g
            tracker.note(bank_layer_id)
            placed += 1
        return placed

    if layer_sink is not None:
        placed = _load(layer_sink)
    else:
        with PinPipeline() as pins:
            placed = _load(pins)

    expected = num_layers * E * 6
    assert placed == expected, f"{spec.desc}: loaded {placed} expert tensors, expected {expected}"
    return {
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global": gate_up_global,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global": down_global,
    }


__all__ = [
    "Nvfp4ExpertSourceSpec",
    "load_nvfp4_expert_source_banks",
    "load_nvfp4_expert_source_banks_parallel",
]
