from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prometheus.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)


@dataclass(frozen=True)
class Qwen4ExpArgs:
    """Qwen4-exp (Qwen3.8-Flash-Next) specific geometry, opaque to engine code.

    Covers the three components this architecture adds on top of the qwen3_5_moe
    hybrid backbone: hyper-connections (per-layer gated residual mixers over an
    ``hc_count``-wide residual stream), the PLE hashed n-gram embedding layer, and
    the QSA indexer geometry of the full-attention layers.
    """

    hc_count: int
    hc_lowrank: int
    # PLE (Per-Layer Embedding): hashed n-gram table looked up per token.
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    ngram_divisor: int
    split_ngram_parts: int
    ple_layer_idxs: tuple[int, ...]  # decoder layer_idx values carrying a PLE
    eos_token_id: int
    # QSA indexer geometry (full-attention layers).
    indexer_budget: int
    indexer_compress_ratio: int
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    model_path: str | None = None
    # Unigram vocab (HF config vocab_size): bounds the splitmix hash multipliers.
    unigram_vocab: int = 0
    # GDN output-gate activation (HF ``output_gate_type``, None -> hidden_act): Flash-Next
    # ships "sigmoid" while qwen3_5 used "silu" -- the two are NOT interchangeable.
    output_gate_type: str = "silu"
    # seed of the PLE hash constants (HF config ``seed``, default 1234).
    ple_seed: int = 1234


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def parse_config(hf_config: Any, model_path: str | None = None) -> ModelConfig:
    """Qwen4ExpForConditionalGeneration (text tower) -> ModelConfig.

    bf16-only milestone: any quantization_config is rejected loudly. The multimodal
    wrapper's ``model.visual.*`` tower is dropped by the weight loader; the text
    tower is served text-only, exactly like the qwen3_5_moe conditional-generation
    checkpoints before it.
    """
    text = getattr(hf_config, "text_config", hf_config)

    if getattr(hf_config, "quantization_config", None):
        raise ValueError("qwen4_exp weight loading supports the bf16 checkpoint only")

    head_dim = (
        getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads
    )
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    rotary_dim = round(head_dim * partial)
    # Text-only with the default rope type: the mRoPE params reduce to standard
    # partial NeoX rope (same reduction qwen3_5_moe performs).
    assert rope_params.get("rope_type", "default") in (None, "default"), (
        f"unsupported rope type: {rope_params.get('rope_type')}"
    )

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=None,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        output_gate=True,
        in_proj_split=False,  # bf16 checkpoint ships unfused in_proj_{qkv,z,b,a}
        in_proj_nvfp4=False,
    )
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    eos = text.eos_token_id
    eos = eos[0] if isinstance(eos, (list, tuple)) else eos
    # ple_layer_ids names layer_idx+1 (HF convention); translate to decoder layer_idx.
    ple_layer_idxs = tuple(
        int(i) - 1 for i in (getattr(text, "ple_layer_ids", None) or []) if int(i) > 0
    )
    q4_args = Qwen4ExpArgs(
        hc_count=int(text.hc_count),
        hc_lowrank=int(text.hc_lowrank),
        ple_embed_dim=int(text.ple_embed_dim),
        ple_conv_kernel_size=int(text.ple_conv_kernel_size),
        ngram_size=int(text.ngram_size),
        heads_per_ngram=int(text.heads_per_ngram),
        ngram_vocab_size_base=int(text.ngram_vocab_size_base),
        ngram_divisor=int(text.make_ngram_vocab_size_divisible_by),
        split_ngram_parts=int(getattr(text, "split_ngram_parts", 1) or 1),
        ple_layer_idxs=ple_layer_idxs,
        eos_token_id=int(eos),
        output_gate_type=str(
            getattr(text, "output_gate_type", None) or text.hidden_act or "silu"
        ),
        unigram_vocab=int(text.vocab_size),
        ple_seed=int(getattr(text, "seed", 1234) or 1234),
        indexer_budget=int(getattr(text, "indexer_budget", 0) or 0),
        indexer_compress_ratio=int(getattr(text, "indexer_compress_ratio", 0) or 0),
        indexer_n_heads=int(getattr(text, "indexer_n_heads", 0) or 0),
        indexer_kv_heads=int(getattr(text, "indexer_kv_heads", 0) or 0),
        indexer_head_dim=int(getattr(text, "indexer_head_dim", 0) or 0),
        model_path=model_path,
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0),
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=text.num_experts,
        num_experts_per_tok=getattr(text, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0),
        shared_expert_intermediate_size=getattr(text, "shared_expert_intermediate_size", 0),
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        moe_scoring_func=str(getattr(text, "scoring_func", None) or "softmax"),
        moe_enabled=True,
        use_qk_norm=True,
        model_type="qwen4_exp",
        architectures=getattr(hf_config, "architectures", ["Qwen4ExpForConditionalGeneration"]),
        vision_config=None,
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        q4_args=q4_args,
    )


__all__ = ["parse_config", "Qwen4ExpArgs"]
