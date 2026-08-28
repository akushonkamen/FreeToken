"""Qwen3.5 MTP (multi-token prediction) draft head for speculative decoding.

The checkpoint stores the draft as bf16 ``mtp.*`` tensors (kept bf16 even on NVFP4
checkpoints -- quantization ignores the head): a ``fc`` combiner, two pre-fc norms, ONE
full-attention decoder layer, and a final norm. Embedding and lm_head are SHARED with the
target model (``mtp_use_dedicated_embeddings=false``) -- bound by reference at build time,
never copied into the state dict.

Forward, per predicted position j (token x_j paired with the target's post-final-norm
hidden H_{j-1} -- DeepSeek-V3 nextn layer):

    h = fc(cat[pre_fc_norm_embedding(embed(x_j)), pre_fc_norm_hidden(H_{j-1})])
    h', res = decoder_layer(h, residual=None)          # standard Gemma-norm block
    z  = norm(h', res)                                  # z_j predicts position j+1

Chaining is EAGLE-style: after a forward, ``z`` is argmaxed through the shared lm_head to
propose the next draft token, which is fed back with ``z`` in place of a target hidden.
All 7 norm tensors are Gemma-style (1+weight); the loader bakes the +1 exactly like the
main model's.

Runtime state lives in :class:`MTPDraftManager` (per-request KV rows in private
contiguous buffers -- the draft layer never touches the engine's paged KV pool). The
greedy longest-prefix verify (scheduler) keeps this lossless: committed tokens are always
the target's own argmaxes; drafts only propose candidates.

Losslessness here is SEMANTIC, not bitwise. The spec-verify forward feeds the whole
1+k-row span through the GDN chunk kernel / multi-row GEMMs while plain decode feeds one
row through the decode kernel -- different fp reduction orders, so an argmax can
occasionally flip and the committed stream may drift from (and later rejoin) the no-spec
stream. This is a property of the shared spec machinery, not of the MTP head (n-gram spec
drifts the same way); making it bitwise would need kernel-level determinism work, which
is out of scope by decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import torch
import torch.nn.functional as F
from prometheus.layers import (
    BaseOP,
    GemmaRMSNorm,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    silu_and_mul,
)
from prometheus.layers.rotary import get_rope
from prometheus.kernel.triton.nvfp4_linear import (
    nvfp4_dense_linear,
    nvfp4_dense_linear_t,
    nvfp4_transpose_resident,
)

if TYPE_CHECKING:
    from prometheus.core import Req
    from prometheus.layers import ParallelLMHead, VocabParallelEmbedding
    from prometheus.models.config import ModelConfig


def _quantize_nvfp4_rowmajor(w: torch.Tensor, chunk: int = 8192):
    """Quantize a bf16 [N, K] weight into the checkpoint-native NVFP4 (W4A16) row-major
    layout: packed uint8 [N, K//2] (low nibble = even k), e4m3 block scales [N, K//16]
    (16 consecutive k per block), fp16 per-row global scales. Two-level scaling
    (weight = e2m1 * block * global) keeps the e4m3 block scales in range. Rows are
    processed in chunks so the fp32 temporaries stay bounded."""
    N, K = w.shape
    assert K % 16 == 0
    packed = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    scale = torch.empty(N, K // 16, dtype=torch.float8_e4m3fn, device=w.device)
    glob = torch.empty(N, dtype=torch.float16, device=w.device)
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=w.device)
    for s0 in range(0, N, chunk):
        e = min(s0 + chunk, N)
        rows = e - s0
        wf = w[s0:e].float()
        amax_row = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        g = (amax_row / 2688.0).to(torch.float16)  # 448 (e4m3 max) * 6 (e2m1 max)
        gf = g.float()
        blk = (wf.abs().view(rows, K // 16, 16).amax(dim=-1) / (6.0 * gf)).clamp(1e-12, 448.0)
        sc = blk.to(torch.float8_e4m3fn)
        scale[s0:e] = sc
        glob[s0:e] = g.squeeze(1)
        denom = (sc.float() * gf).clamp(min=1e-12)
        q = (wf.view(rows, K // 16, 16) / denom.unsqueeze(-1)).clamp(-6.0, 6.0)
        code = torch.bucketize(q.abs(), mids).to(torch.uint8)
        code |= (q < 0).to(torch.uint8) << 3
        code = code.view(rows, K)
        packed[s0:e] = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, scale, glob


class _Fp4LinearShim:
    """Forward a loaded bf16 linear through an NVFP4 (W4A16) quantized copy of its
    weight (proposal-side only -- the verify path stays on the bf16 target)."""

    def __init__(self, linear):
        w = linear.weight
        packed, scale, glob = _quantize_nvfp4_rowmajor(w.detach())
        # K-major resident layout: coalesced N-wide gemv loads
        self.packed, self.scale = nvfp4_transpose_resident(packed, scale)
        self.glob = glob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nvfp4_dense_linear_t(x, self.packed, self.scale, self.glob)


class MTPAttention(BaseOP):
    """The draft layer's attention: identical projection/norm/rope math to
    :class:`Qwen3_5Attention`, but runs OUTSIDE the engine's forward context -- it
    takes explicit positions and attends over caller-owned contiguous KV rows instead
    of the paged KV pool, so it cannot reuse ``ctx.attn_backend``."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        head_dim = config.head_dim
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = self.num_q * head_dim
        self.kv_attn_dim = self.num_kv * head_dim
        self._qkv_split = [self.num_q * head_dim * 2, self.kv_attn_dim, self.kv_attn_dim]
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, self._qkv_split, has_bias=False
        )
        self.q_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        # Same rope params as the main full-attention layers -> get_rope's cache shares
        # one cos/sin table between target and draft.
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items())
                if config.rotary_config.scaling
                else None
            ),
        )
        self.o_proj = LinearReplicated(self.qo_attn_dim, config.hidden_size, has_bias=False)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_start: int,
    ) -> torch.Tensor:
        L = x.shape[0]
        qkv = self.qkv_proj.forward(x)
        qg, k, v = torch.split(qkv, self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.view(-1, self.num_kv, self.head_dim).contiguous()
        v = v.contiguous()
        q = self.q_norm.forward(q).reshape(-1, self.qo_attn_dim)
        k = self.k_norm.forward(k).reshape(-1, self.kv_attn_dim)
        q, k = self.rotary.forward(positions, q, k)
        T = kv_start + L
        k_cache[kv_start:T] = k
        v_cache[kv_start:T] = v
        # GQA via SDPA over the private contiguous rows. Query row l sits at position
        # kv_start+l+1, cache row t at position t+1 -> causal within the new block only;
        # the [0, kv_start) prefix is fully visible (no mask needed for it).
        q4 = q.view(L, self.num_q, self.head_dim).transpose(0, 1).unsqueeze(0)
        k4 = k_cache[:T].view(T, self.num_kv, self.head_dim).transpose(0, 1).unsqueeze(0)
        v4 = v_cache[:T].view(T, self.num_kv, self.head_dim).transpose(0, 1).unsqueeze(0)
        if L == 1:
            attn_out = F.scaled_dot_product_attention(q4, k4, v4, enable_gqa=True)
        elif kv_start == 0:
            attn_out = F.scaled_dot_product_attention(
                q4, k4, v4, is_causal=True, enable_gqa=True
            )
        else:
            mask = torch.arange(T, device=x.device) <= torch.arange(
                kv_start, T, device=x.device
            )[:, None]
            attn_out = F.scaled_dot_product_attention(
                q4, k4, v4, attn_mask=mask, enable_gqa=True
            )
        out = attn_out.squeeze(0).transpose(0, 1).reshape(L, self.qo_attn_dim)
        gated = out * torch.sigmoid(gate)
        return self.o_proj.forward(gated)


class MTPMLP(BaseOP):
    """Plain SwiGLU MLP with the checkpoint's fused-projection attribute names
    (``gate_up_proj`` / ``down_proj``), matching the main model's loader fusions."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False
        )
        self.down_proj = LinearRowParallel(intermediate_size, hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class MTPDecoderLayer(BaseOP):
    """Pre-norm block mirroring :class:`Qwen3_5DecoderLayer`'s full-attention branch
    (residual=None entry: the fc output is the first residual stream), with the
    draft attention and a dense SwiGLU MLP (the checkpoint's mtp layer is not MoE)."""

    def __init__(self, config: "ModelConfig"):
        self.self_attn = MTPAttention(config, layer_id=0)
        self.mlp = MTPMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden
        hidden = self.input_layernorm.forward(hidden)
        hidden = self.self_attn.forward(hidden, positions, k_cache, v_cache, kv_start)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class MTPDraft(BaseOP):
    """The MTP draft module. Shared target modules bind by reference under
    underscore attrs (skipped by state_dict), so ``load_state_dict`` sees exactly the
    13 fused checkpoint keys of the head itself."""

    def __init__(
        self,
        config: "ModelConfig",
        embed_tokens: "VocabParallelEmbedding",
        lm_head: "ParallelLMHead",
    ):
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.fc = LinearReplicated(2 * hidden, hidden, has_bias=False)
        self.pre_fc_norm_embedding = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.layers = OPList([MTPDecoderLayer(config)])
        self.norm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self._embed_tokens = embed_tokens
        # Draft argmax runs OUTSIDE forward_batch (no ctx.batch), so it cannot go through
        # lm_head.forward. The bf16 head is a ~2.5 GB read per draft token (the dominant
        # draft cost); keep an NVFP4 (W4A16) quantized copy for the DRAFT ONLY. A worse
        # argmax here just lowers acceptance -- the scheduler's greedy longest-prefix
        # verify still commits the target's own argmaxes, so outputs are unchanged.
        _p, _s, self._lmq_global = _quantize_nvfp4_rowmajor(lm_head.weight)
        # K-major resident layout: coalesced N-wide gemv loads
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
        """Run the draft over L (token, target-hidden) pairs. Row i lands at KV row
        ``kv_start + i`` / position ``position_start + i``; returns the post-norm
        hiddens z [L, hidden] (row i predicts position position_start + i + 1)."""
        e = self._embed_tokens.forward(tokens)
        h = torch.cat(
            (self.pre_fc_norm_embedding.forward(e), self.pre_fc_norm_hidden.forward(hiddens)),
            dim=-1,
        )
        h = self.fc.forward(h)
        positions = torch.arange(
            position_start,
            position_start + tokens.shape[0],
            dtype=torch.int32,
            device=tokens.device,
        )
        h, residual = self.layers.op_list[0].forward(h, positions, k_cache, v_cache, kv_start)
        z, _ = self.norm.forward_add_residual(h, residual)
        return z

    def quantize_linears_nvfp4(self) -> None:
        """Swap the head's bf16 linears for NVFP4 (W4A16) shims (draft-side only:
        a worse z just lowers acceptance, the verify still commits the target's own
        argmaxes). Drops the 849 MB bf16 head weights for a ~215 MB packed copy."""
        self.fc = _Fp4LinearShim(self.fc)
        layer = self.layers.op_list[0]
        attn, mlp = layer.self_attn, layer.mlp
        attn.qkv_proj = _Fp4LinearShim(attn.qkv_proj)
        attn.o_proj = _Fp4LinearShim(attn.o_proj)
        mlp.gate_up_proj = _Fp4LinearShim(mlp.gate_up_proj)
        mlp.down_proj = _Fp4LinearShim(mlp.down_proj)
        torch.cuda.empty_cache()

    def argmax_next(self, z: torch.Tensor) -> torch.Tensor:
        """Greedy next token of the draft head: [1, hidden] -> [1] int64 on device,
        through the draft-side NVFP4 lm_head copy (4x less weight traffic than bf16)."""
        return torch.argmax(
            nvfp4_dense_linear_t(z, self._lmq_packed, self._lmq_scale, self._lmq_global),
            dim=-1,
        )


# Gemma (1+weight) norms of the MTP head: the loader bakes the +1 like the main model's.
_MTP_GEMMA_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)
# q|k|v -> qkv_proj and gate|up -> gate_up_proj, matching the split order above.
_MTP_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj": (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"),
    ".mlp.gate_up_proj": (".mlp.gate_proj", ".mlp.up_proj"),
}


def load_mtp_state_dict(model_path: str, device: torch.device) -> Dict[str, torch.Tensor] | None:
    """Read the checkpoint's bf16 ``mtp.*`` tensors into the draft module's state dict
    (13 fused keys), or None when the checkpoint has no MTP head. The main weight
    loaders DROP mtp.* unconditionally (weight.py keeps that path untouched); this
    dedicated reader is the only consumer."""
    from prometheus.models.loader import ShardReader, ct_bf16_fuse

    reader = ShardReader(model_path, device)
    state: Dict[str, torch.Tensor] = {}
    fuse_buf: dict = {}
    found = False
    try:
        for file in reader.files():
            for name in reader.names_in(file):
                if not name.startswith("mtp."):
                    continue
                found = True
                tensor = reader.get_tensor(name)
                if any(name.endswith(s) for s in _MTP_GEMMA_SUFFIXES):
                    tensor = tensor + 1.0
                emit = ct_bf16_fuse(name[len("mtp.") : -len(".weight")], tensor, fuse_buf, _MTP_FUSIONS)
                if emit is None:  # not a fusion part: keep as-is (already has .weight)
                    state[name[len("mtp.") :]] = tensor
                else:
                    state.update(emit)  # [] while buffered, [(key, cat)] once complete
        assert not fuse_buf, f"Incomplete MTP projection fusions: {list(fuse_buf.keys())}"
    finally:
        reader.close()
    return state if found else None


class _MTPReqState:
    """Per-request draft state. KV rows are PRIVATE contiguous buffers (1 layer, so a
    paged pool is overkill): row i holds the token x_{i+1} of the committed stream at
    position i+1. Invariant at every quiescent point: rows [0, valid_len) hold exactly
    the committed tokens x_1..x_{valid_len}; the pending chain's rows
    [valid_len, valid_len + len(draft) - 1) hold draft[0..len-2]; ``z`` is the draft's
    post-norm output for x_{valid_len} (predicts position valid_len + 1); chain_z[i]
    is the output that produced draft[i + 1]."""

    __slots__ = ("k_cache", "v_cache", "cap", "valid_len", "z", "draft_t", "draft_pin", "draft_n", "chain_z", "hist")

    def __init__(self, k_cache: torch.Tensor, v_cache: torch.Tensor):
        self.hist: List[tuple] = []
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.cap = k_cache.shape[0]
        self.valid_len = 0
        self.z: torch.Tensor | None = None
        # Pending chain output, kept ON DEVICE (int64 [draft_n]) so the chain never
        # host-syncs: seating copies it device-side and the drain reads the pinned
        # mirror. ``draft_pin`` is allocated lazily (pin_memory at alloc cost).
        self.draft_t: torch.Tensor | None = None
        self.draft_pin: torch.Tensor | None = None
        self.draft_n: int = 0
        self.chain_z: List[torch.Tensor] = []


def _trace(msg: str) -> None:
    import os
    import sys
    if os.getenv("PROM_MTP_TRACE"):
        print(f"[MTP-TRACE] {msg}", file=sys.stderr, flush=True)


class MTPDraftManager:
    """Scheduler-side driver: seeds the draft from prefill hiddens, advances it over
    committed tokens, chains new drafts, and hands the seated draft list to the
    spec-verify scheduler. All GPU work is issued on the caller's current stream
    (the scheduler's drain runs after its wait on the engine stream), so it is
    program-ordered around the target's forwards without extra sync."""

    def __init__(self, draft: MTPDraft, num_draft: int, device: torch.device):
        self.draft = draft
        self.num_draft = num_draft
        self.device = device
        self.states: Dict[int, _MTPReqState] = {}
        # Requests that can never spec (non-greedy / multimodal: no lossless greedy
        # verification exists for them; a prefix-cache hit is NOT here -- it bridges).
        self._ineligible: Set[int] = set()

    def remove(self, uid: int) -> None:
        self.states.pop(uid, None)
        # Ineligibility is per-request, not per-uid: the offline LLM API renumbers
        # uids from 0 on every generate() call, so a stale entry here would poison
        # every later request that reuses the uid (spec stays off forever).
        self._ineligible.discard(uid)

    def seat_drafts(self, reqs: List["Req"], k: int) -> List[torch.Tensor] | None:
        """Trim each state's pending chain to the seatable draft (budget-capped) and
        return the per-req GPU tensors, or None if ANY request has no draft (the
        caller falls back to a plain decode batch, like n-gram v1 all-or-nothing).
        Also refreshes each state's pinned mirror (async D2H): the caller's drain
        materializes the CPU lists from it after the forward drains, so the chain
        never host-syncs between verify steps."""
        # Two-pass: validate EVERY req before mutating ANY state. A mid-loop bail
        # (e.g. a sibling req whose prefill is still in flight -> state is None)
        # used to leave earlier reqs' chains trimmed while the caller fell back to
        # a plain decode batch -- the trimmed state then desynced from the
        # committed stream at the next verify (valid_len == m - 1).
        budgets: List[int] = []
        for req in reqs:
            state = self.states.get(req.uid)
            if state is None or state.draft_t is None or state.draft_n == 0:
                _trace(
                    f"FALLBACK seat: uid={req.uid} "
                    f"state={'None' if state is None else 'dry'} "
                    f"running={[r.uid for r in reqs]}"
                )
                return None
            budget = min(k, state.draft_n, req.remain_len - 1)
            if budget <= 0:
                _trace(
                    f"FALLBACK seat: uid={req.uid} budget={budget} "
                    f"draft_n={state.draft_n} remain={req.remain_len}"
                )
                return None
            budgets.append(budget)
        drafts: List[torch.Tensor] = []
        for req, budget in zip(reqs, budgets):
            state = self.states[req.uid]
            # Keep the state 1:1 with the seated list (acceptance bookkeeping reads
            # chain rows by position against it). The batch holds the same tensor
            # object; re-chaining REPLACES it (never mutates in place), so the
            # in-flight batch's view stays intact.
            state.draft_t = state.draft_t[:budget]
            state.draft_n = budget
            state.chain_z = state.chain_z[: budget - 1]
            if state.draft_pin is None:
                state.draft_pin = torch.empty(
                    self.num_draft, dtype=torch.int64, pin_memory=True
                )
            state.draft_pin[:budget].copy_(state.draft_t, non_blocking=True)
            _trace(f"seat uid={req.uid} budget={budget} valid_len={state.valid_len} cached_len={req.cached_len} device_len={req.device_len}")
            drafts.append(state.draft_t)
        return drafts

    # ---------------- drain-side hooks (scheduler) ----------------

    def on_prefill_hidden(self, req: "Req", hidden_rows: torch.Tensor, final: bool) -> None:
        """Advance the draft over one prefill chunk: row r of the chunk pairs token
        x_{first_row + r + 1} with hidden row r (token t pairs the hidden of the token
        BEFORE it). Tokens come from req.input_ids, which spans the whole prompt plus
        (final chunk) the sampled token. ``final`` chains the first draft list;
        intermediate chunks only extend the KV so the final chunk pairs align.

        At the final-chunk hook the sampled token is already appended AND counted,
        so cached_len = processed + 1 there; intermediate hooks run pre-sample with
        cached_len = processed. Invariant at every quiescent point afterwards:
        valid_len == cached_len - 1 (row r holds token x_{r+1})."""
        state = self.states.get(req.uid)
        if state is None:
            if req.uid in self._ineligible:
                return
            if not req.sampling_params.is_greedy or req.mm_embeds is not None:
                self._ineligible.add(req.uid)
                return
            bridge_n = req.cached_len - hidden_rows.shape[0] - (1 if final else 0)
            if bridge_n > 0:
                # Prefix-cache hit: rows [0, C - rows - final) were never forwarded
                # this run, so their target hiddens do not exist. Bridge them with
                # the z-substitution the decode path uses (the draft's own hidden
                # stands in for the target's), rolled out over the ACTUAL committed
                # tokens, so the draft KV still covers the whole prompt. One-time
                # cost at admission; lossless (drafts are only candidates).
                state = self._alloc(req)
                self._bridge_prefix(state, req, bridge_n)
            else:
                # bridge_n == 0 (full forward this run) or < 0 (final hook where
                # cached_len does not yet count the sampled bonus -- batch-shape
                # dependent complete_one ordering): nothing to bridge either way.
                state = self._alloc(req)
        C = req.cached_len
        rows = hidden_rows.shape[0]
        expected = C - rows - (1 if final else 0)
        # The final-chunk hook's cached_len may or may not include the sampled
        # bonus yet (both orderings occur depending on batch shape), so accept
        # the +/-1 both conventions produce. A genuinely skipped chunk shifts
        # coverage by a whole chunk (rows >> 1) and still trips this.
        if state.valid_len not in (expected, expected + 1):
            # Hidden-row coverage broke (skipped chunk); drop the request from spec.
            _trace(
                f"DROP prefill uid={req.uid} final={final} "
                f"valid_len={state.valid_len} C={C} rows={rows} hist={state.hist}"
            )
            self.remove(req.uid)
            self._ineligible.add(req.uid)
            return
        state.hist.append(("prefill", state.valid_len, C, rows, 1 if final else 0))
        state.hist = state.hist[-12:]
        _trace(f"prefill uid={req.uid} final={final} valid_len={state.valid_len} C={C} rows={rows}")
        # Row r of the chunk pairs token x_{valid+r+1}: exactly `rows` tokens. The
        # end index derives from rows (authoritative), not C -- C's bonus-inclusion
        # at the final hook is ordering-dependent. Under the counted-bonus
        # convention this is identical to the old `C` / `C+1` endpoints.
        end = state.valid_len + 1 + rows
        if final:
            end = min(end, req.input_ids.numel())
        tokens = req.input_ids[state.valid_len + 1 : end]
        need = state.valid_len + rows + (self.num_draft - 1 if final else 0)
        self._ensure_cap(state, need, req.max_device_len)
        self._run_pairs(state, tokens, hidden_rows)
        if final:
            self._rechain(state, req.max_device_len)

    def on_decode_committed(self, req: "Req", token: int, finished: bool) -> None:
        """A plain decode step committed ``token`` (the CUDA-graph step cannot expose
        its target hidden, so pairs bridge with the draft's own z). Only runs when
        spec seating fell back to plain decode for the batch."""
        if finished:
            self.remove(req.uid)
            return
        state = self.states.get(req.uid)
        if state is None:
            return
        # draft_n == 1 (num_draft=1, e.g. --spec-mtp 1) leaves chain_z empty and the
        # committed token's KV row unwritten: there is no chain row to consume, so
        # take the substitution path (run the pair, reseed) instead of popping.
        if (
            state.draft_t is not None
            and state.draft_n > 1
            and int(state.draft_t[0]) == token
        ):
            # The first pending draft was exactly right: keep its chain row (the
            # token/position pair match by construction) and the rest of the chain.
            # (int() host-syncs, but this fallback-decode path runs with the chain
            # long drained, so the read is instant.)
            state.draft_t = state.draft_t[1:]
            state.draft_n -= 1
            state.z = state.chain_z.pop(0)
            state.valid_len += 1
            state.hist.append(("decode-fast", state.valid_len, req.cached_len, state.draft_n))
            state.hist = state.hist[-12:]
            _trace(f"decode-fast uid={req.uid} valid_len={state.valid_len} cached_len={req.cached_len}")
            return
        # Mismatch (or drained chain): z-substitution -- feed the committed token
        # against the draft's last hidden, then re-seed the chain. Drafts are only
        # candidates, so the substitution stays lossless.
        tokens = torch.tensor([token], dtype=torch.int32)
        self._ensure_cap(state, state.valid_len + 1 + self.num_draft - 1, req.max_device_len)
        self._run_pairs(state, tokens, state.z)
        self._rechain(state, req.max_device_len)
        state.hist.append(("decode-subst", state.valid_len, req.cached_len, state.draft_n))
        state.hist = state.hist[-12:]
        _trace(f"decode-subst uid={req.uid} valid_len={state.valid_len} cached_len={req.cached_len}")

    def on_spec_accept(
        self,
        req: "Req",
        m: int,
        hidden_rows: torch.Tensor | None,
        preds: List[int],
        draft: List[int],
        finished: bool,
    ) -> None:
        """Post-verify advance: keep the chain rows of the accepted prefix
        (== min(accepted, len(draft)-1) rows, built on exactly those draft tokens),
        re-run the rejected tail with the TARGET's verify hiddens (row i of the spec
        forward is H_{m+i}: pairs draft[i] and, at i == accepted, the bonus token),
        then chain a fresh draft from the new z."""
        state = self.states.get(req.uid)
        if state is None:
            return
        if finished:
            self.remove(req.uid)
            return
        # Quiescent invariant: valid_len == m (row r holds token x_{r+1}; rows
        # x_1..x_valid_len built, and m = cached_len with committed == m+1).
        # One legal exception: the FIRST verify after prefill can run with no
        # intervening plain decode (overlap scheduling admits the spec batch
        # straight off the prefill drain), and on_prefill_hidden deliberately
        # leaves the seated token x_m's row unbuilt (it needs the pre-forward
        # hidden, unavailable at prefill time). Bridge it with z-substitution --
        # exactly the substitution on_decode_committed's fallback and the prefix
        # bridge use. Lossless: drafts are only candidates.
        if state.valid_len == m - 1:
            seated = torch.tensor([int(req.input_ids[m])], dtype=torch.int32)
            self._ensure_cap(state, m + self.num_draft, req.max_device_len)
            self._run_pairs(state, seated, state.z)
        if state.valid_len != m:
            _trace(
                f"DESYNC uid={req.uid} valid_len={state.valid_len} m={m} "
                f"cached_len={req.cached_len} device_len={req.device_len} "
                f"draft_n={state.draft_n} ineligible={self._ineligible} "
                f"hist={state.hist}"
            )
        assert state.valid_len == m, "MTP chain desynced from the committed stream"
        state.hist.append(("accept", state.valid_len, m, req.cached_len, state.draft_n))
        state.hist = state.hist[-12:]
        _trace(f"accept uid={req.uid} m={m} cached_len={req.cached_len} draft={draft} preds={preds}")
        accepted = 0
        while accepted < len(draft) and preds[accepted] == draft[accepted]:
            accepted += 1
        keep = min(accepted, len(draft) - 1)
        state.valid_len = m + keep  # chain rows [m, m+keep) survive verbatim
        tokens = torch.tensor(draft[keep:accepted] + [preds[accepted]], dtype=torch.int32)
        self._ensure_cap(state, state.valid_len + tokens.shape[0] + self.num_draft - 1, req.max_device_len)
        if hidden_rows is not None:
            hiddens = hidden_rows[keep : accepted + 1]
            self._run_pairs(state, tokens, hiddens)
        else:
            # Degraded drain (no target hiddens on the batch): roll the rejected
            # tail + bonus out with the draft's own z -- the same substitution the
            # prefix bridge uses, just chained. Keeps valid_len exact; drafts stay
            # candidates, so the committed stream is untouched.
            self._rollout_z(state, tokens)
        self._rechain(state, req.max_device_len)

    # ---------------- GPU plumbing ----------------

    def _bridge_prefix(self, state: _MTPReqState, req: "Req", missed: int) -> None:
        """Build the draft's KV rows for a prefix-cache hit's missed span [0, missed)
        with the z-substitution bridge: row j pairs the ACTUAL token x_{j+1} with the
        draft's own previous hidden (zeros for j=0) instead of the target's -- those
        rows were never forwarded this run, so no target hiddens exist. Sequential by
        construction (row j's input z is row j-1's output); one-time cost at
        admission. Lossless: drafts are only candidates."""
        if missed <= 0:
            return
        z = torch.zeros(1, self.draft.hidden_size, dtype=torch.bfloat16, device=self.device)
        toks = req.input_ids[1 : missed + 1].to(self.device, dtype=torch.int32)
        self._ensure_cap(state, missed + self.num_draft - 1, req.max_device_len)
        for i in range(missed):
            z = self.draft.forward_tokens(
                toks[i : i + 1], z, state.valid_len + 1, state.k_cache, state.v_cache, state.valid_len
            )
            state.valid_len += 1
        state.z = z

    def _rollout_z(self, state: _MTPReqState, tokens: torch.Tensor) -> None:
        """Sequentially forward (token, previous z) pairs, appending KV rows at
        valid_len -- the z-substitution rollout used when a verify drain has no
        target hiddens for the rejected tail."""
        z = state.z
        toks = tokens.to(self.device, dtype=torch.int32)
        for i in range(toks.shape[0]):
            z = self.draft.forward_tokens(
                toks[i : i + 1], z, state.valid_len + 1, state.k_cache, state.v_cache, state.valid_len
            )
            state.valid_len += 1
        state.z = z[-1:]

    def _alloc(self, req: "Req") -> _MTPReqState:
        kv_dim = self.draft.num_kv * self.draft.head_dim
        cap = min(req.max_device_len, max(1024, req.cached_len + self.num_draft + 1))
        _trace(f"alloc uid={req.uid} cached_len={req.cached_len}")
        state = _MTPReqState(
            k_cache=torch.empty(cap, kv_dim, dtype=torch.bfloat16, device=self.device),
            v_cache=torch.empty(cap, kv_dim, dtype=torch.bfloat16, device=self.device),
        )
        self.states[req.uid] = state
        return state

    def _ensure_cap(self, state: _MTPReqState, need: int, hard_cap: int) -> None:
        if need <= state.cap:
            return
        new_cap = min(max(need, state.cap * 2), hard_cap)
        kv_dim = state.k_cache.shape[1]
        k2 = torch.empty(new_cap, kv_dim, dtype=torch.bfloat16, device=self.device)
        v2 = torch.empty(new_cap, kv_dim, dtype=torch.bfloat16, device=self.device)
        k2[: state.cap].copy_(state.k_cache)
        v2[: state.cap].copy_(state.v_cache)
        state.k_cache, state.v_cache, state.cap = k2, v2, new_cap

    def _run_pairs(
        self, state: _MTPReqState, tokens: torch.Tensor, hiddens: torch.Tensor
    ) -> None:
        """Forward the (token, hidden) pairs, appending KV rows at valid_len."""
        if hiddens.is_cuda:
            # The target hiddens were allocated on the engine stream; keep the block
            # from being recycled into an engine-stream reuse before this stream's
            # enqueued reads complete.
            hiddens.record_stream(torch.cuda.current_stream())
        toks = tokens.to(self.device, dtype=torch.int32)
        z = self.draft.forward_tokens(
            toks, hiddens, state.valid_len + 1, state.k_cache, state.v_cache, state.valid_len
        )
        state.valid_len += tokens.shape[0]
        state.z = z[-1:]

    def _rechain(self, state: _MTPReqState, hard_cap: int) -> None:
        """Seed from z and chain num_draft-1 more rows: a fully on-device loop (each
        argmax feeds the next embed), one host sync at the final tolist(). The draft
        shrinks near the output-budget end so the chain stays inside the KV rows the
        request's max length can hold (seat_drafts trims to the commit budget anyway)."""
        k = min(self.num_draft, max(1, hard_cap - state.valid_len))
        z = state.z
        seed = self.draft.argmax_next(z)
        toks = [seed]
        zs: List[torch.Tensor] = []
        self._ensure_cap(state, state.valid_len + k, hard_cap)
        for i in range(k - 1):
            z = self.draft.forward_tokens(
                toks[-1].to(torch.int32),
                z,
                state.valid_len + i + 1,
                state.k_cache,
                state.v_cache,
                state.valid_len + i,
            )
            zs.append(z)
            toks.append(self.draft.argmax_next(z))
        state.draft_t = torch.cat(toks)
        state.draft_n = state.draft_t.shape[0]
        state.chain_z = zs


__all__ = ["MTPDraft", "MTPDraftManager", "load_mtp_state_dict"]
