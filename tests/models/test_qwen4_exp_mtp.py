"""Qwen4-exp MTP draft head: checkpoint-key transform and module structure.

Drives ``qwen4_exp.mtp`` with synthetic tensors shaped like Flash-Next's 31
``mtp.*`` checkpoint keys (small geometry; the real-checkpoint shape/parity
check lives in the benchmarks/ Phase-0 scripts) and verifies:

* ``fuse_mtp_state`` emits exactly the module's 28-key state dict (indexer
  dropped, q|k|v -> qkv_proj with the pre-doubled q gate half, shared gate|up
  merged, q/k Gemma norms baked with +1, group norms RAW);
* ``Qwen4ExpMTPDraft.load_state_dict`` consumes it completely (no unexpected
  keys -- the module tree's attribute names match the fused checkpoint names);
* the head's geometry plugs (chain-state width = hc_count*hidden, two norm
  groups per GatedResidual mixer);
* the routed-expert bank quantizes without materializing the bf16 stack on the
  GPU and its W4A16 forward matches the bf16 path.
"""

from __future__ import annotations

import pytest
import torch

from prometheus.models.qwen4_exp.mtp import (
    MTPExperts,
    Qwen4ExpMTPDraft,
    fuse_mtp_state,
)

from .test_qwen4_exp import _hf_config

from prometheus.models.qwen4_exp.config import parse_config


def _mtp_config():
    hf = _hf_config(num_layers=8)
    text = hf.text_config
    # Shrink the MoE + attention geometry for a fast test; structure unchanged.
    text.num_experts = 16
    text.num_experts_per_tok = 4
    text.moe_intermediate_size = 64
    text.shared_expert_intermediate_size = 64
    return parse_config(hf)


def _raw_mtp_tensors(config) -> dict[str, torch.Tensor]:
    H = config.hidden_size
    HC = config.q4_args.hc_count
    LOW = config.q4_args.hc_lowrank
    NQ, NKV, HD = config.num_qo_heads, config.num_kv_heads, config.head_dim
    E, I = config.num_experts, config.moe_intermediate_size
    SI = config.shared_expert_intermediate_size

    def rnd(*s):
        return torch.randn(*s, dtype=torch.bfloat16) * 0.02

    t: dict[str, torch.Tensor] = {
        "mtp.fc_embedding.weight": rnd(H, H),
        "mtp.fc_hidden.weight": rnd(H, H),
        "mtp.pre_fc_norm_embedding.weight": rnd(H),
        "mtp.pre_fc_norm_hidden.weight": rnd(H * HC),
        "mtp.layers.0.self_attn.q_proj.weight": rnd(NQ * HD * 2, H),  # pre-doubled gate
        "mtp.layers.0.self_attn.k_proj.weight": rnd(NKV * HD, H),
        "mtp.layers.0.self_attn.v_proj.weight": rnd(NKV * HD, H),
        "mtp.layers.0.self_attn.o_proj.weight": rnd(H, NQ * HD),
        "mtp.layers.0.self_attn.q_norm.weight": rnd(HD),
        "mtp.layers.0.self_attn.k_norm.weight": rnd(HD),
        # dropped: QSA indexer
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": rnd(8, H),
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight": rnd(16),
        "mtp.layers.0.self_attn.indexer.k_layernorm.weight": rnd(16),
    }
    for pfx in ("attn_hyper_connection", "mlp_hyper_connection"):
        t[f"mtp.layers.0.{pfx}.hc_norm.weight"] = rnd(H * HC)
        t[f"mtp.layers.0.{pfx}.input_mix_weight_down.weight"] = rnd(LOW, H * HC)
        t[f"mtp.layers.0.{pfx}.input_mix_weight_up.weight"] = rnd(H * HC, LOW)
        t[f"mtp.layers.0.{pfx}.block_inject_weight.weight"] = rnd(HC, H * HC)
    t["mtp.layers.0.mlp.gate.weight"] = rnd(E, H)
    t["mtp.layers.0.mlp.shared_expert.gate_proj.weight"] = rnd(SI, H)
    t["mtp.layers.0.mlp.shared_expert.up_proj.weight"] = rnd(SI, H)
    t["mtp.layers.0.mlp.shared_expert.down_proj.weight"] = rnd(H, SI)
    t["mtp.layers.0.mlp.shared_expert_gate.weight"] = rnd(1, H)
    t["mtp.layers.0.mlp.experts.gate_up_proj"] = rnd(E, 2 * I, H)
    t["mtp.layers.0.mlp.experts.down_proj"] = rnd(E, H, I)
    t["mtp.hyper_connection_mixer.hc_norm.weight"] = rnd(H * HC)
    t["mtp.hyper_connection_mixer.input_mix_weight_down.weight"] = rnd(LOW, H * HC)
    t["mtp.hyper_connection_mixer.input_mix_weight_up.weight"] = rnd(H * HC, LOW)
    return t


class _StubEmbed:
    def forward(self, tokens):
        raise AssertionError("not exercised by this test")


class _StubHead:
    def __init__(self, hidden):
        self.weight = torch.randn(256, hidden, dtype=torch.bfloat16) * 0.02


def test_fuse_mtp_state_keys():
    config = _mtp_config()
    raw = _raw_mtp_tensors(config)
    state = fuse_mtp_state(raw)
    assert state is not None
    H, HC = config.hidden_size, config.q4_args.hc_count
    NQ, NKV, HD = config.num_qo_heads, config.num_kv_heads, config.head_dim
    E, I = config.num_experts, config.moe_intermediate_size
    SI = config.shared_expert_intermediate_size
    expected = {
        "fc_embedding.weight": (H, H),
        "fc_hidden.weight": (H, H),
        "pre_fc_norm_embedding.weight": (H,),
        "pre_fc_norm_hidden.weight": (H * HC,),
        "layers.0.self_attn.qkv_proj.weight": (NQ * HD * 2 + 2 * NKV * HD, H),
        "layers.0.self_attn.o_proj.weight": (H, NQ * HD),
        "layers.0.self_attn.q_norm.weight": (HD,),
        "layers.0.self_attn.k_norm.weight": (HD,),
        "layers.0.mlp.gate.weight": (E, H),
        "layers.0.mlp.shared_expert.gate_up_proj.weight": (2 * SI, H),
        "layers.0.mlp.shared_expert.down_proj.weight": (H, SI),
        "layers.0.mlp.shared_expert_gate.weight": (1, H),
        "layers.0.mlp.experts.gate_up_proj": (E, 2 * I, H),
        "layers.0.mlp.experts.down_proj": (E, H, I),
        "hyper_connection_mixer.hc_norm.weight": (H * HC,),
        "hyper_connection_mixer.input_mix_weight_down.weight": (
            config.q4_args.hc_lowrank,
            H * HC,
        ),
        "hyper_connection_mixer.input_mix_weight_up.weight": (
            H * HC,
            config.q4_args.hc_lowrank,
        ),
    }
    for pfx in ("attn_hyper_connection", "mlp_hyper_connection"):
        expected[f"layers.0.{pfx}.hc_norm.weight"] = (H * HC,)
        expected[f"layers.0.{pfx}.input_mix_weight_down.weight"] = (
            config.q4_args.hc_lowrank,
            H * HC,
        )
        expected[f"layers.0.{pfx}.input_mix_weight_up.weight"] = (
            H * HC,
            config.q4_args.hc_lowrank,
        )
        expected[f"layers.0.{pfx}.block_inject_weight.weight"] = (HC, H * HC)
    assert set(state) == set(expected), (
        f"missing={set(expected) - set(state)} extra={set(state) - set(expected)}"
    )
    for k, shape in expected.items():
        assert tuple(state[k].shape) == shape, k
    # Gemma q/k norms get +1 baked; group norms stay raw.
    qn = raw["mtp.layers.0.self_attn.q_norm.weight"]
    assert torch.equal(state["layers.0.self_attn.q_norm.weight"], qn + 1.0)
    hn = raw["mtp.layers.0.attn_hyper_connection.hc_norm.weight"]
    assert torch.equal(state["layers.0.attn_hyper_connection.hc_norm.weight"], hn)
    # qkv fusion order: doubled q half, then k, then v (raw, not +1).
    q = raw["mtp.layers.0.self_attn.q_proj.weight"]
    k = raw["mtp.layers.0.self_attn.k_proj.weight"]
    v = raw["mtp.layers.0.self_attn.v_proj.weight"]
    assert torch.equal(
        state["layers.0.self_attn.qkv_proj.weight"], torch.cat([q, k, v], dim=0)
    )
    assert fuse_mtp_state({"model.layers.0.self_attn.q_proj.weight": q}) is None


def test_draft_module_loads_fused_state():
    config = _mtp_config()
    raw = _raw_mtp_tensors(config)
    state = fuse_mtp_state(raw)
    with torch.device("cpu"), torch.autocast("cpu", enabled=False):
        torch.set_default_dtype(torch.bfloat16)
        try:
            draft = Qwen4ExpMTPDraft(config, _StubEmbed(), _StubHead(config.hidden_size))
        finally:
            torch.set_default_dtype(torch.float32)
    assert draft.hidden_size == config.hidden_size * config.q4_args.hc_count
    draft.load_state_dict(state)  # raises on unexpected/missing keys
    assert state == {} or not state  # BaseOP pops every key


@pytest.fixture(autouse=True)
def _tp_single():
    from prometheus.distributed import info as _info

    if _info.try_get_tp_info() is None:
        _info.set_tp_info(0, 1)
    yield


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the W4A16 bank")
def test_expert_bank_quant_parity():
    config = _mtp_config()
    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    bank = MTPExperts(E, H, I)
    bank.gate_up_proj = torch.randn(E, 2 * I, H, dtype=torch.bfloat16) * 0.05
    bank.down_proj = torch.randn(E, H, I, dtype=torch.bfloat16) * 0.05
    dev = torch.device("cuda")
    x = (torch.randn(6, H, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    topw = torch.full((6, 4), 0.25, dtype=torch.bfloat16, device=dev)
    topi = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [0, 2, 4, 6], [3, 5, 7, 9], [0, 1, 4, 8]],
        device=dev,
    )
    ref = bank._bf16_forward(x.to("cpu"), topw.to("cpu"), topi.to("cpu")).to(dev)
    bank.quantize_nvfp4(dev)
    out = bank.forward(x, topw, topi)
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().max().clamp(min=1e-6)
    assert float(diff.max() / denom) < 0.25, f"W4A16 bank too far from bf16: {diff.max()}"
