from __future__ import annotations

import torch
import torch.nn.functional as F
from prometheus.core import get_global_ctx
from prometheus.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from prometheus.layers import BaseOP, LinearColParallelMerged

from prometheus.kernel.triton.fp8_block_linear import Fp8BlockColMerged
from prometheus.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged
from prometheus.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged

from .gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla
from .quant_linear import make_replicated_quant

# Device-side const tensors for GDN spec readvance (see Qwen3_5GatedDeltaNet.
# spec_readvance): (rows, slot, device_index) -> (cu, idx, has_init).
_SPEC_READVANCE_CONSTS: dict = {}


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class _GatedRMSNorm(BaseOP):
    """RMSNorm of x followed by a silu(z) gate (HF Qwen3_5MoeRMSNormGated).

    Uses the fused fla ``rms_norm_gated`` triton kernel (norm(x) * silu(z) in one
    kernel) instead of the unfused pow/mean/rsqrt/mul/silu chain, matching sglang's
    ``RMSNormGated`` -- collapses ~8 elementwise kernels per GDN layer into one."""

    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from prometheus.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x, weight=self.weight, bias=None, z=z, eps=self.eps,
            is_rms_norm=True, norm_before_gate=True, activation="silu",
        )


class Qwen3_5GatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``prometheus.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, expert_quant: str = "none",
        attn_quant: str = "none", in_proj_split: bool = False,
        in_proj_nvfp4: bool = False,
    ):
        self.layer_id = layer_id
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        # qkv|z carry a weight scale (block-fp8 weight_scale_inv, or per-tensor FP8
        # weight_scale); b|a stay bf16. Both quant modes therefore split the four-way
        # fusion into an fp8 qkvz GEMM + a bf16 ba GEMM (matches sglang/vLLM).
        self._block_fp8 = expert_quant == "fp8_block"
        self._pertensor_fp8 = attn_quant == "fp8_pertensor"
        self._fp8 = self._block_fp8 or self._pertensor_fp8
        # compressed-tensors NVFP4 checkpoints that quantize the unfused in_proj parts keep
        # qkv|z packed: the W4A16 GEMM replaces the bf16/dequant one. Its dequant math is
        # identical to the load-time dequant (same fp4 * block * global product), so this is
        # lossless -- it only keeps the packed tensors resident instead of a bf16 copy that
        # alone would push a 27B model past a 24 GB card.
        self._nvfp4_qkvz = attn_quant == "nvfp4" and in_proj_nvfp4
        # Modelopt NVFP4 checkpoints with the pre-fused split layout (Qwen3-Next) keep
        # in_proj_qkvz/in_proj_ba bf16: same two-GEMM path as fp8, but a bf16 qkvz GEMM.
        self._split = self._fp8 or in_proj_split or self._nvfp4_qkvz

        self._in_proj_split = [self.conv_dim, self.value_dim, num_v_heads, num_v_heads]
        if self._split:
            if self._fp8:
                ColMerged = Fp8BlockColMerged if self._block_fp8 else Fp8PerTensorColMerged
                self.in_proj_qkvz = ColMerged(
                    hidden_size, [self.conv_dim, self.value_dim], has_bias=False
                )
            elif self._nvfp4_qkvz:
                # packed qkv|z (W4A16): same [conv_dim, value_dim] output split and the same
                # in_proj_qkvz.forward(hidden_states) call site below as the bf16/fp8 paths.
                self.in_proj_qkvz = Nvfp4DenseColMerged(
                    hidden_size, [self.conv_dim, self.value_dim], has_bias=False
                )
            else:
                self.in_proj_qkvz = LinearColParallelMerged(
                    hidden_size, [self.conv_dim, self.value_dim], has_bias=False
                )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a.
            self.in_proj = LinearColParallelMerged(hidden_size, self._in_proj_split, has_bias=False)
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast.
        self.dt_bias = torch.empty(num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(num_v_heads, dtype=torch.float32)
        self.norm = _GatedRMSNorm(head_v_dim, eps=rms_norm_eps)
        # out_proj follows the checkpoint quant: block-fp8 / per-tensor-fp8 / compressed-tensors
        # NVFP4 (W4A16) / bf16. in_proj_ba stays bf16 in every mode (above); a compressed-
        # tensors NVFP4 checkpoint with packed in_proj parts (_nvfp4_qkvz) also makes qkvz
        # and out_proj native FP4.
        self.out_proj = make_replicated_quant(
            expert_quant, attn_quant, self.value_dim, hidden_size, has_bias=False
        )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata."""
        li = pool.local_index(self.layer_id)
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
        out = causal_conv1d_varlen(x, self._conv_weight(), pool.conv_states[li],
                                   cu_seqlens, cache_indices, has_initial_state)
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a DIRECT copy (h is [V,K], the
        state pool is [K,V]; they coincide because GDN requires head_k_dim == head_v_dim).
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    def spec_readvance(self, conv_in: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                       slot: int, device: torch.device,
                       graph_consts: tuple | None = None) -> None:
        """Spec partial-accept rollback: re-advance ONLY this layer's conv + recurrent
        state over a row slice of the stashed spec-verify span. The slot must already
        hold the pre-forward snapshot (``SpecLinearSnapshots.restore``); this re-runs
        just the state-updating kernels (causal_conv1d_varlen + chunk scan) over the
        caller's slice of the stash -- the verify forward's own (conv_in, a, b) -- so
        the landed state is identical to the old full-model replay's, without the
        ~100ms eager forward it used to cost. Output is discarded.

        When ``graph_consts`` is given, the (cu_seqlens, cache_idx, has_init) device
        tensors are passed in from the caller (graph-owned persistent buffers) instead
        of being looked up from the ``_SPEC_READVANCE_CONSTS`` cache -- this avoids
        H2D copies inside a CUDA graph capture region."""
        pool = get_global_ctx().linear_state_pool
        li = pool.local_index(self.layer_id)
        rows = conv_in.shape[0]
        if graph_consts is not None:
            cu, idx, has_init = graph_consts
        else:
            # Const tensors (cu/idx/has_init) are cached device-side: three pageable H2D
            # copies per layer per rollback (~90 under a busy engine stream) cost more
            # than the kernels themselves. Keyed by (rows, slot, device) -- a handful of
            # entries (rows <= 1+k, slots are per-request stable).
            key = (rows, slot, device.index)
            consts = _SPEC_READVANCE_CONSTS.get(key)
            if consts is None:
                consts = (
                    torch.tensor([0, rows], dtype=torch.int64, device=device),  # ONE sequence
                    torch.tensor([slot], dtype=torch.int32, device=device),
                    torch.tensor([True], dtype=torch.bool, device=device),
                )
                _SPEC_READVANCE_CONSTS[key] = consts
            cu, idx, has_init = consts
        mixed = self._conv_prefill(conv_in, pool, cu, idx, has_init)
        dtype = conv_in.dtype
        qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = qf.reshape(1, rows, self.num_k_heads, self.head_k_dim).to(dtype)
        k = kf.reshape(1, rows, self.num_k_heads, self.head_k_dim).to(dtype)
        v = vf.reshape(1, rows, self.num_v_heads, self.head_v_dim).to(dtype)
        g, beta = self._gate_params(a, b)
        g = g.reshape(1, rows, self.num_v_heads)
        beta = beta.float().reshape(1, rows, self.num_v_heads)
        gdn_prefill_chunk_fla(
            q, k, v, g, beta,
            state_source=pool.recurrent_states[li], indices=idx,
            cu_seqlens=cu, scale=self.head_k_dim ** -0.5,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from prometheus.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        if self._split:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
        z = z.reshape(total, self.num_v_heads, self.head_v_dim)
        if batch.spec_verify:
            # Partial-accept rollback input stash (Scheduler._replay_spec_states): the
            # GDN state re-advance over the accepted span needs only these pre-conv
            # projections, so keep them per layer -- the rollback then runs a few tiny
            # state kernels instead of replaying a ~100ms full model forward.
            if batch.spec_gdn_stash is None:
                batch.spec_gdn_stash = {}
            batch.spec_gdn_stash[self.layer_id] = (conv_in, a, b)
        li = pool.local_index(self.layer_id)

        if batch.is_decode:
            # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
            # per-request state read/write-by-index, all in one kernel (no gather/scatter,
            # no clone, no external l2norm). q/k stay at num_k_heads (kernel handles GQA).
            mixed = self._conv_decode(conv_in, fla.cache_indices, pool)  # [B, conv_dim]
            B = mixed.shape[0]
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
            core_out = gdn_decode_fla(
                q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
            )
        else:
            mixed = self._conv_prefill(
                conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state)
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None
            result = gdn_prefill_chunk_fla(
                q, k, v, g, beta,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                return_h=track,
            )
            if track:
                core_out, h = result
                self._write_track_snapshot(pool, li, conv_in, h, fla)
            else:
                core_out = result

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(total, -1)
        return self.out_proj.forward(out)


__all__ = ["Qwen3_5GatedDeltaNet"]
