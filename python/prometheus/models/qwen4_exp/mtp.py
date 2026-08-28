"""Qwen4-exp (Qwen3.8-Flash-Next) MTP draft head for speculative decoding.

The checkpoint stores the draft as bf16 ``mtp.*`` tensors (31 keys; modelopt's
ignore list keeps the head bf16 on the NVFP4 checkpoint): two entry projections
(``fc_embedding`` H->H shared, ``fc_hidden`` H->H applied per hyper-stream), two
group-RMS entry norms, ONE full-attention decoder layer shaped exactly like a
target layer (hyper-connection mixers + a 512-expert MoE + gated shared
expert), and a final ``hyper_connection_mixer`` (use_combine=False) that folds
the stream down for the lm_head. Embedding and lm_head are SHARED with the
target model (``mtp_use_dedicated_embeddings=false``) -- bound by reference at
build time, never copied into the state dict.

Forward, per predicted position j (token x_j paired with the target's last-layer
PRE-MIXER hyper stream H_{j-1} [10240] -- captured by Qwen4ExpForCausalLM
before ``hyper_connection_mixer``):

    e = pre_fc_norm_embedding(embed(x_j))                  # [2560]  group-RMS
    h = pre_fc_norm_hidden(H_{j-1})                        # [10240] group-RMS(2560)
    hyper_s = fc_hidden(h_s) + fc_embedding(e)             # per-stream add (locked
                                                           # against a bf16 reference
                                                           # forward, see benchmarks/)
    z_hyper = decoder_layer(hyper)                         # [10240] EAGLE chain state
    token = argmax(lm_head(hyper_connection_mixer(z_hyper)))  # [2560] mix-then-head

The output is deliberately TWO-headed where the 27B head had one: ``z_hyper``
[10240] is the EAGLE-style chained state (fed back in place of a target hidden,
exactly like the qwen3_5 MTP's z), while only its stream-mixed projection
[2560] ever reaches the lm_head.

Attention geometry is identical to the 27B draft layer (gated full attention,
q_proj pre-doubled for the output gate, Gemma q/k norms, partial NeoX rope
rotary_dim=64 theta=1e7 -- one shared cos/sin cache with the target), so the
layer reuses :class:`qwen3_5_moe.mtp.MTPAttention` verbatim. The QSA indexer
tensors of the checkpoint are dropped, matching the target's decision.

The 512 routed experts (~5.03 GB bf16) are draft-RESIDENT (they never touch the
engine's offload slot cache): ``quantize_linears_nvfp4`` streams them into an
NVFP4 W4A16 bank (~1.26 GB), expert by expert, so the bf16 weight is never
fully materialized on the GPU. A worse draft argmax only lowers acceptance --
the scheduler's greedy longest-prefix verify still commits the target's own
argmaxes, so outputs are unchanged (same losslessness argument as the 27B head).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
import torch.nn.functional as F
from prometheus.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    silu_and_mul,
)
from prometheus.kernel.triton.nvfp4_linear import (
    nvfp4_dense_linear_t,
    nvfp4_transpose_resident,
)

from ..qwen3_5_moe.mtp import MTPAttention, _Fp4LinearShim, _quantize_nvfp4_rowmajor
from .model import Qwen4GatedResidual
from .ple import Qwen4GroupRMSNorm

if TYPE_CHECKING:
    from prometheus.core import Req
    from prometheus.layers import ParallelLMHead, VocabParallelEmbedding
    from prometheus.models.config import ModelConfig


class MTPExperts(BaseOP):
    """The draft layer's 512 routed experts, pre-stacked whole-layer exactly like
    the checkpoint keys (``gate_up_proj`` [E, 2I, H] gate-first, ``down_proj``
    [E, H, I], no ``.weight`` suffix). Kept bf16 until
    :meth:`quantize_nvfp4` streams them into a W4A16 bank.

    The placeholders live on CPU on purpose: the bf16 bank is ~5 GB and the GPU
    may not have that much headroom next to the offloaded target; the quantized
    path streams one expert at a time and the bf16 tensors never leave the host.
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = torch.empty(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16
        )
        self.down_proj = torch.empty(
            num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16
        )
        self._packed: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None

    def _bf16_forward(
        self, x: torch.Tensor, top_weights: torch.Tensor, top_ids: torch.Tensor
    ) -> torch.Tensor:
        T, H = x.shape
        routed = torch.zeros_like(x)
        flat_ids = top_ids.reshape(-1).tolist()
        flat_w = top_weights.reshape(-1).tolist()
        order = sorted(range(len(flat_ids)), key=lambda r: flat_ids[r])
        i = 0
        while i < len(order):
            j = i
            eidx = flat_ids[order[i]]
            while j < len(order) and flat_ids[order[j]] == eidx:
                j += 1
            rows = torch.tensor([order[r] for r in range(i, j)], device=x.device)
            xe = x[rows // top_ids.shape[1]]
            gu = xe @ self.gate_up_proj[eidx].to(x.device, dtype=x.dtype).t()
            g, u = gu[:, : self.intermediate_size], gu[:, self.intermediate_size :]
            ye = (F.silu(g) * u) @ self.down_proj[eidx].to(x.device, dtype=x.dtype).t()
            w = torch.tensor(
                [flat_w[order[r]] for r in range(i, j)], device=x.device, dtype=x.dtype
            ).unsqueeze(-1)
            routed.index_add_(0, rows // top_ids.shape[1], ye * w)
            i = j
        return routed

    def _quant_forward(
        self, x: torch.Tensor, top_weights: torch.Tensor, top_ids: torch.Tensor
    ) -> torch.Tensor:
        routed = torch.zeros_like(x)
        flat_ids = top_ids.reshape(-1).tolist()
        flat_w = top_weights.reshape(-1).tolist()
        order = sorted(range(len(flat_ids)), key=lambda r: flat_ids[r])
        i = 0
        while i < len(order):
            j = i
            eidx = flat_ids[order[i]]
            while j < len(order) and flat_ids[order[j]] == eidx:
                j += 1
            rows = torch.tensor([order[r] for r in range(i, j)], device=x.device)
            xe = x[rows // top_ids.shape[1]]
            gu_p, gu_s, gu_g = self._packed[0][eidx]
            dn_p, dn_s, dn_g = self._packed[1][eidx]
            gu = nvfp4_dense_linear_t(xe, gu_p, gu_s, gu_g)
            g, u = gu[:, : self.intermediate_size], gu[:, self.intermediate_size :]
            ye = nvfp4_dense_linear_t(F.silu(g) * u, dn_p, dn_s, dn_g)
            w = torch.tensor(
                [flat_w[order[r]] for r in range(i, j)], device=x.device, dtype=x.dtype
            ).unsqueeze(-1)
            routed.index_add_(0, rows // top_ids.shape[1], ye * w)
            i = j
        return routed

    def forward(
        self, x: torch.Tensor, top_weights: torch.Tensor, top_ids: torch.Tensor
    ) -> torch.Tensor:
        if self._packed is not None:
            return self._quant_forward(x, top_weights, top_ids)
        return self._bf16_forward(x, top_weights, top_ids)

    def quantize_nvfp4(self, device: torch.device) -> None:
        """Stream the bf16 experts (CPU) into a GPU NVFP4 W4A16 bank, one expert
        at a time (peak GPU cost: one expert's bf16 copy + its packed form)."""
        assert self._packed is None
        banks: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [[], []]
        for bank_idx, stacked in enumerate((self.gate_up_proj, self.down_proj)):
            for eidx in range(self.num_experts):
                w = stacked[eidx].to(device)
                p, s, g = _quantize_nvfp4_rowmajor(w)
                pt, st = nvfp4_transpose_resident(p, s)
                banks[bank_idx].append((pt, st, g))
        self._packed = (banks[0], banks[1])
        self.gate_up_proj = self.gate_up_proj[:0]  # free the host bf16 bank
        self.down_proj = self.down_proj[:0]
        torch.cuda.empty_cache()


class MTPMoE(BaseOP):
    """Routed MoE (512 experts, top-10) plus a sigmoid-gated shared expert --
    :class:`Qwen3_5MoE`'s math with a draft-resident expert bank instead of the
    engine's offload-slot MoE (the draft layer runs outside the forward ctx)."""

    def __init__(self, config: "ModelConfig"):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _MTPSharedExpert(
            config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)
        self.experts = MTPExperts(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        self._topk = config.num_experts_per_tok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate.forward(x)
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self._topk, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)  # norm_topk_prob (HF semantics)
        shared = self.shared_expert.forward(x) * torch.sigmoid(
            self.shared_expert_gate.forward(x)
        )
        routed = self.experts.forward(x, topv, topi)
        return routed + shared

    def quantize_linears_nvfp4(self, device: torch.device) -> None:
        self.gate = _Fp4LinearShim(self.gate)
        self.shared_expert_gate = _Fp4LinearShim(self.shared_expert_gate)
        se = self.shared_expert
        se.gate_up_proj = _Fp4LinearShim(se.gate_up_proj)
        se.down_proj = _Fp4LinearShim(se.down_proj)
        self.experts.quantize_nvfp4(device)


class _MTPSharedExpert(BaseOP):
    """SwiGLU shared expert with the checkpoint's fused-projection attribute
    names (``gate_up_proj`` / ``down_proj``), matching the loader merges."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False
        )
        self.down_proj = LinearRowParallel(intermediate_size, hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class MTPHyperLayer(BaseOP):
    """One decoder layer of the draft head: :class:`Qwen4ExpDecoderLayer`'s
    hyper-connection structure (no PLE, no input/post layernorms -- the residual
    normalization lives in each GatedResidual's hc_norm) with the standalone
    draft attention and the draft-resident MoE."""

    def __init__(self, config: "ModelConfig"):
        q4 = config.q4_args
        self.self_attn = MTPAttention(config, layer_id=0)
        self.attn_hyper_connection = Qwen4GatedResidual(config, q4, use_combine=True)
        self.mlp_hyper_connection = Qwen4GatedResidual(config, q4, use_combine=True)
        self.mlp = MTPMoE(config)

    def forward(
        self,
        hyper: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_start: int,
    ) -> torch.Tensor:
        mixed, _, inj = self.attn_hyper_connection.forward_with_inject(hyper)
        out = self.self_attn.forward(mixed, positions, k_cache, v_cache, kv_start)
        hyper = hyper + (out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        mixed, _, inj = self.mlp_hyper_connection.forward_with_inject(hyper)
        out = self.mlp.forward(mixed)
        hyper = hyper + (out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        return hyper


class Qwen4ExpMTPDraft(BaseOP):
    """The qwen4_exp MTP draft module. Shared target modules bind by reference
    under underscore attrs (skipped by state_dict), so ``load_state_dict`` sees
    exactly the 28 fused checkpoint keys of the head itself (31 raw mtp.* keys
    minus the 3 dropped indexer tensors).

    ``hidden_size`` is the CHAIN-STATE width (hc_count * hidden = 10240): the
    manager's z-substitution bridge feeds ``z_hyper`` back through the same
    entry path a target hidden takes."""

    def __init__(
        self,
        config: "ModelConfig",
        embed_tokens: "VocabParallelEmbedding",
        lm_head: "ParallelLMHead",
    ):
        q4 = config.q4_args
        hidden = config.hidden_size
        self.hidden_size = hidden * q4.hc_count
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.fc_embedding = LinearReplicated(hidden, hidden, has_bias=False)
        self.fc_hidden = LinearReplicated(hidden, hidden, has_bias=False)
        self.pre_fc_norm_embedding = Qwen4GroupRMSNorm(hidden, hidden, config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen4GroupRMSNorm(
            hidden * q4.hc_count, hidden, config.rms_norm_eps
        )
        self.layers = OPList([MTPHyperLayer(config)])
        self.hyper_connection_mixer = Qwen4GatedResidual(config, q4, use_combine=False)
        self._embed_tokens = embed_tokens
        self._hc = q4.hc_count
        self._hidden = hidden
        # Draft argmax runs OUTSIDE forward_batch (no ctx.batch), so it cannot go
        # through lm_head.forward. Keep an NVFP4 (W4A16) copy of the bf16 head
        # (~1.27 GB -> ~318 MB packed) for the DRAFT ONLY -- same mechanism and
        # losslessness argument as the 27B head.
        _p, _s, self._lmq_global = _quantize_nvfp4_rowmajor(lm_head.weight)
        self._lmq_packed, self._lmq_scale = nvfp4_transpose_resident(_p, _s)

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        hiddens: torch.Tensor,
        position_start: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_start: int,
    ) -> torch.Tensor:
        """Run the draft over L (token, target-hyper) pairs. Row i lands at KV row
        ``kv_start + i`` / position ``position_start + i``; returns the chained
        hyper state z_hyper [L, hc*hidden] (row i's mix for the lm_head predicts
        position position_start + i + 1)."""
        e = self._embed_tokens.forward(tokens)
        e = self.pre_fc_norm_embedding.forward(e)
        h = self.pre_fc_norm_hidden.forward(hiddens)
        fe = self.fc_embedding.forward(e)
        fh = self.fc_hidden.forward(h.view(-1, self._hidden)).view(
            -1, self._hc, self._hidden
        )
        hyper = (fh + fe.unsqueeze(1)).flatten(-2)
        positions = torch.arange(
            position_start,
            position_start + tokens.shape[0],
            dtype=torch.int32,
            device=tokens.device,
        )
        return self.layers.op_list[0].forward(
            hyper, positions, k_cache, v_cache, kv_start
        )

    def mix(self, z_hyper: torch.Tensor) -> torch.Tensor:
        """Fold the hyper stream down to the lm_head input [T, hidden]."""
        return self.hyper_connection_mixer.forward(z_hyper)

    def argmax_next(self, z_hyper: torch.Tensor) -> torch.Tensor:
        """Greedy next token of the draft head: [1, hc*hidden] -> [1] int64 on
        device, through the draft-side NVFP4 lm_head copy."""
        return torch.argmax(
            nvfp4_dense_linear_t(
                self.mix(z_hyper), self._lmq_packed, self._lmq_scale, self._lmq_global
            ),
            dim=-1,
        )

    def quantize_linears_nvfp4(self) -> None:
        """Swap the head's bf16 linears for NVFP4 (W4A16) shims and stream the
        5 GB routed-expert bank down to ~1.26 GB (draft-side only: a worse z just
        lowers acceptance, the verify still commits the target's own argmaxes)."""
        device = self._lmq_packed.device
        self.fc_embedding = _Fp4LinearShim(self.fc_embedding)
        self.fc_hidden = _Fp4LinearShim(self.fc_hidden)
        layer = self.layers.op_list[0]
        attn = layer.self_attn
        attn.qkv_proj = _Fp4LinearShim(attn.qkv_proj)
        attn.o_proj = _Fp4LinearShim(attn.o_proj)
        layer.mlp.quantize_linears_nvfp4(device)
        for gated in (
            layer.attn_hyper_connection,
            layer.mlp_hyper_connection,
            self.hyper_connection_mixer,
        ):
            gated.input_mix_weight_down = _Fp4LinearShim(gated.input_mix_weight_down)
            gated.input_mix_weight_up = _Fp4LinearShim(gated.input_mix_weight_up)
        torch.cuda.empty_cache()


# Gemma (1+weight) norms of the draft head: the loader bakes the +1 like the main
# model's (GemmaRMSNorm scales by the raw weight). The qwen4_exp-native group
# norms (pre_fc_*, hc_norm) keep their weight raw -- +1 in fp32 at runtime.
_MTP_GEMMA_SUFFIXES = (
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)
_MTP_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj": (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"),
}
_SHARED_GATE = ".mlp.shared_expert.gate_proj.weight"
_SHARED_UP = ".mlp.shared_expert.up_proj.weight"


def fuse_mtp_state(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor] | None:
    """Transform raw ``mtp.``-prefixed checkpoint tensors into the draft module's
    fused state dict (28 keys from 31 raw: the 3 QSA indexer tensors drop; q|k|v
    fuse into qkv_proj, shared gate|up merge into gate_up_proj; q/k Gemma norms
    get the +1 baked). ``None`` when nothing mtp.* was present."""
    from prometheus.models.loader import ct_bf16_fuse

    state: Dict[str, torch.Tensor] = {}
    fuse_buf: dict = {}
    shared_buf: dict[str, dict[str, torch.Tensor]] = {}
    found = False
    for name, tensor in raw.items():
        if not name.startswith("mtp."):
            continue
        if ".self_attn.indexer." in name:
            continue  # dropped: dense attention identical for kv_len <= 2052
        found = True
        short = name[len("mtp.") :]
        if any(name.endswith(s) for s in _MTP_GEMMA_SUFFIXES):
            tensor = tensor + 1.0
        if short.endswith(_SHARED_GATE) or short.endswith(_SHARED_UP):
            prefix = short.rsplit(".mlp.shared_expert.", 1)[0]
            slot = "gate" if short.endswith(_SHARED_GATE) else "up"
            shared_buf.setdefault(prefix, {})[slot] = tensor
            slots = shared_buf[prefix]
            if "gate" in slots and "up" in slots:
                merged = torch.cat([slots["gate"], slots["up"]], dim=0)
                del shared_buf[prefix]
                state[f"{prefix}.mlp.shared_expert.gate_up_proj.weight"] = merged
            continue
        emit = ct_bf16_fuse(short[: -len(".weight")], tensor, fuse_buf, _MTP_FUSIONS)
        if emit is None:  # not a fusion part: keep as-is (has its suffix)
            state[short] = tensor
        else:
            state.update(emit)  # [] while buffered, [(key, cat)] once complete
    assert not fuse_buf, f"Incomplete MTP projection fusions: {list(fuse_buf.keys())}"
    assert not shared_buf, f"Incomplete MTP shared-expert merges: {list(shared_buf.keys())}"
    return state if found else None


def load_mtp_state_dict(model_path: str, device: torch.device) -> Dict[str, torch.Tensor] | None:
    """Read the checkpoint's bf16 ``mtp.*`` tensors into the draft module's state
    dict (28 fused keys; the 3 QSA indexer tensors are dropped like the target's).
    The two ~5 GB routed-expert stacks are read to CPU -- they only meet the GPU
    expert-by-expert inside ``MTPExperts.quantize_nvfp4``."""
    from prometheus.models.loader import ShardReader

    reader = ShardReader(model_path, device)
    cpu_reader: ShardReader | None = None
    raw: Dict[str, torch.Tensor] = {}
    found = False
    try:
        for file in reader.files():
            for name in reader.names_in(file):
                if not name.startswith("mtp."):
                    continue
                found = True
                short = name[len("mtp.") :]
                if short in (
                    "layers.0.mlp.experts.gate_up_proj",
                    "layers.0.mlp.experts.down_proj",
                ):
                    if cpu_reader is None:
                        cpu_reader = ShardReader(model_path, torch.device("cpu"))
                    raw[name] = cpu_reader.get_tensor(name)
                else:
                    raw[name] = reader.get_tensor(name)
    finally:
        reader.close()
        if cpu_reader is not None:
            cpu_reader.close()
    if not found:
        return None
    return fuse_mtp_state(raw)


__all__ = ["Qwen4ExpMTPDraft", "MTPHyperLayer", "MTPMoE", "MTPExperts", "load_mtp_state_dict"]
