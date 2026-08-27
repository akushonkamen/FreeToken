"""Qwen4-exp (Qwen3.8-Flash-Next) config parsing and PLE hash-constant parity.

Drives ``qwen4_exp.parse_config`` off a synthetic HF config shaped exactly like
Qwen/Qwen3.8-Flash-Next's shipped config.json (multimodal wrapper, text tower in
``text_config``, bf16, no quantization_config), and checks the derived geometry
against the values pinned in the deployment plan. The PLE hash constants
(splitmix64 multipliers, prime head vocab sizes, offsets, padded table size) are
compared against golden values computed with transformers-main's
``modeling_qwen4_exp`` algorithm (pure python, config seed default 1234).
"""

from __future__ import annotations

import pytest
import torch

from prometheus.models.qwen4_exp.config import parse_config
from prometheus.models.qwen4_exp.ple import Qwen4PleEmbedding


class _Cfg:
    """Attribute-access shim over a dict (what AutoConfig hands parse_config)."""

    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, _Cfg(v) if isinstance(v, dict) and k == "text_config" else v)


def _hf_config(num_layers: int = 48) -> _Cfg:
    interval = 4
    layer_types = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(num_layers)
    ]
    return _Cfg(
        {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "image_token_id": 248056,
            "quantization_config": None,
            "text_config": {
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 248044,
                "eos_token_id": 248044,
                "full_attention_interval": interval,
                "hc_count": 4,
                "hc_lowrank": 320,
                "head_dim": 256,
                "heads_per_ngram": 8,
                "hidden_act": "silu",
                "hidden_size": 2560,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "indexer_head_dim": 128,
                "indexer_kv_heads": 1,
                "indexer_n_heads": 4,
                "layer_types": layer_types,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "make_ngram_vocab_size_divisible_by": 128,
                "max_position_embeddings": 262144,
                "moe_intermediate_size": 640,
                "ngram_size": 3,
                "ngram_vocab_size_base": 20000000,
                "num_attention_heads": 24,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "num_hidden_layers": num_layers,
                "num_key_value_heads": 2,
                "output_gate_type": "sigmoid",
                "partial_rotary_factor": 0.25,
                "ple_conv_kernel_size": 4,
                "ple_embed_dim": 2560,
                "ple_layer_ids": [2],
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "mrope_interleaved": True,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default",
                },
                "shared_expert_intermediate_size": 640,
                "split_ngram_parts": 128,
                "tie_word_embeddings": False,
                "vocab_size": 248320,
            },
        }
    )


def test_parse_config_geometry():
    cfg = parse_config(_hf_config(), "/fake/path")
    assert cfg.num_layers == 48
    assert cfg.hidden_size == 2560
    assert cfg.num_experts == 512
    assert cfg.num_experts_per_tok == 10
    assert cfg.moe_intermediate_size == 640
    assert cfg.shared_expert_intermediate_size == 640
    assert cfg.norm_topk_prob is True
    assert cfg.vocab_size == 248320
    assert cfg.head_dim == 256
    assert cfg.num_qo_heads == 24
    assert cfg.num_kv_heads == 2
    assert cfg.rotary_config.rotary_dim == 64  # 256 * partial 0.25
    assert cfg.rotary_config.base == 10000000
    assert cfg.model_type == "qwen4_exp"

    full_ids = tuple(i for i in range(48) if (i + 1) % 4 == 0)  # 3,7,...,47
    linear_ids = tuple(i for i in range(48) if (i + 1) % 4 != 0)
    fg = [g for g in cfg.attention_groups if g.name == "full"][0]
    lg = cfg.linear_attention_group()
    assert fg.layer_ids == full_ids and len(full_ids) == 12
    assert lg.layer_ids == linear_ids and len(linear_ids) == 36
    assert lg.num_key_heads == 16 and lg.num_value_heads == 48
    assert lg.key_head_dim == 128 and lg.value_head_dim == 128
    assert lg.conv_kernel_dim == 4
    assert lg.in_proj_split is False  # loader fuses the unfused checkpoint parts

    q4 = cfg.q4_args
    assert q4.hc_count == 4 and q4.hc_lowrank == 320
    assert q4.ple_layer_idxs == (1,)  # HF ple_layer_ids=[2] -> decoder idx 1
    assert q4.eos_token_id == 248044
    assert q4.output_gate_type == "sigmoid"
    assert q4.ple_seed == 1234
    assert q4.indexer_budget == 2048 and q4.indexer_compress_ratio == 4


def test_parse_config_rejects_quantized():
    bad = _hf_config()
    bad.quantization_config = {"quant_method": "modelopt"}
    with pytest.raises(ValueError, match="bf16"):
        parse_config(bad, None)


# Golden values below were computed with the transformers-main
# modeling_qwen4_exp.py algorithms (_build_layer_multipliers / prime head sizes)
# for Flash-Next's config (vocab 248320, ngram 3, base 20e6, ple_layer_index 0,
# seed default 1234).
_GOLDEN_MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]
_GOLDEN_HEAD_SIZES = [
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
]
_GOLDEN_TOTAL = 320001446
_GOLDEN_PADDED = 320001536


def _embedding() -> Qwen4PleEmbedding:
    q4 = parse_config(_hf_config(), "/fake/path").q4_args
    head_dim = q4.ple_embed_dim // ((q4.ngram_size - 1) * q4.heads_per_ngram)
    emb = Qwen4PleEmbedding(q4, head_dim, ple_layer_index=0)
    emb._gpu = {
        "mult": emb._layer_multipliers,
        "vocab": emb._ngram_heads_vocab_sizes,
        "offsets": emb._ngram_heads_offsets,
    }
    return emb


def test_ple_hash_constants_match_hf():
    emb = _embedding()
    assert emb._layer_multipliers.tolist() == _GOLDEN_MULTIPLIERS
    assert emb._ngram_heads_vocab_sizes.tolist() == _GOLDEN_HEAD_SIZES
    offs = emb._ngram_heads_offsets.tolist()
    assert offs[0] == 0
    assert offs[-1] + _GOLDEN_HEAD_SIZES[-1] == _GOLDEN_TOTAL
    assert emb.padded_vocab() == _GOLDEN_PADDED
    assert emb._hpg == 8 and emb._head_dim == 160  # 2560 / 16 heads


def test_ple_head_ids_match_hf_math():
    """The device-side id math vs a literal HF replication (python ints)."""
    emb = _embedding()
    m = emb._layer_multipliers.tolist()
    sizes = emb._ngram_heads_vocab_sizes.tolist()
    offsets = emb._ngram_heads_offsets.tolist()

    eos = 248044
    tokens = [
        (5, eos, eos),   # segment start: both predecessors eos-filled
        (5, 4, eos),     # distance 1: s2 filled
        (5, 4, 3),       # distance 2: full trigram
        (0, 248319, 1),  # boundary values
    ]
    s0 = torch.tensor([t[0] for t in tokens], dtype=torch.int64)
    s1 = torch.tensor([t[1] for t in tokens], dtype=torch.int64)
    s2 = torch.tensor([t[2] for t in tokens], dtype=torch.int64)
    got = emb._head_ids(s0, s1, s2).tolist()

    mask64 = (1 << 64) - 1
    for n, (a, b, c) in enumerate(tokens):
        mixed2 = ((a * m[0]) & mask64) ^ ((b * m[1]) & mask64)
        mixed3 = mixed2 ^ ((c * m[2]) & mask64)
        want = (
            [mixed2 % sizes[h] + offsets[h] for h in range(8)]
            + [mixed3 % sizes[h] + offsets[h] for h in range(8, 16)]
        )
        assert got[n] == want, f"row {n}: {got[n]} != {want}"
