"""Pure-torch GDN decode/prefill fallback for non-CUDA devices (Apple Silicon / MPS).

The Triton GDN kernels are CUDA-only. On a machine without Triton (macOS), the
engine cannot run the vendored fla kernels. This module provides a minimal eager
implementation of the decode (single-token recurrent update) and prefill
(chunked recurrence) paths that ``gdn_kernels.py`` calls.

Numerics mirror the Triton kernel in
``fused_sigmoid_gating_recurrent.py`` exactly:

  g = -exp(A_log) * softplus(a + dt_bias)
  beta = sigmoid(b)
  q, k l2norm (optional)
  q *= scale
  h *= exp(g)
  v = v - sum(h * k, dim=0)
  v *= beta
  h = h + k[:, None] * v[None, :]
  o = sum(h * q, dim=0)

The state pool layout matches the Triton path: ``[num_slots, HV, K, V]`` fp32,
indexed per-request via ``initial_state_indices``.

Only the paths Prometheus exercises at inference time (decode + chunked prefill)
are implemented; training/backward is not needed (inference server).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize the last dim."""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,       # [HV] scalar per head
    a: torch.Tensor,           # [T, HV] (GDN) gate pre-activation
    dt_bias: torch.Tensor,    # [HV]
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,          # [B, T, H, K]
    k: torch.Tensor,          # [B, T, H, K]
    v: torch.Tensor,          # [B, T, HV, V]
    b: torch.Tensor,          # [B, T, HV, 1] or [T, HV]
    initial_state_source: torch.Tensor,  # [num_slots, HV, K, V] fp32
    initial_state_indices: torch.Tensor, # [N] slot per sequence
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eager GDN decode (single-token or short verify).

    Mirrors ``fused_sigmoid_gating_delta_rule_update`` for the decode path.
    For decode, T=1 per request; for target_verify T>1 with intermediate cache.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    HV = v.shape[2]
    if scale is None:
        scale = K ** -0.5

    # Resolve per-sequence slices (varlen: cu_seqlens; batch: equal-length).
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        seq_ranges = [(int(cu_seqlens[i]), int(cu_seqlens[i + 1])) for i in range(N)]
    else:
        N = B
        seq_ranges = [(i * T, (i + 1) * T) for i in range(N)]

    o = q.new_empty(1, B * T if cu_seqlens is None else cu_seqlens[-1].item(), HV, V) \
        if False else q.new_empty(1, v.shape[1] if cu_seqlens is None else 1, HV, V)
    # Simpler: allocate output matching v's layout.
    o = torch.empty_like(v)

    for n, (bos, eos) in enumerate(seq_ranges):
        slot = int(initial_state_indices[n].item()) if initial_state_indices is not None else -1
        seq_len = eos - bos
        for t in range(seq_len):
            idx = bos + t
            for ihv in range(HV):
                # Load state row [K, V] from the pool.
                if slot >= 0 and initial_state_source is not None:
                    h = initial_state_source[slot, ihv].clone()  # [K, V] fp32
                else:
                    h = torch.zeros(K, V, dtype=torch.float32, device=q.device)

                # q/k for this head: GDN uses H heads, HV value heads (GQA).
                ih = ihv // (HV // H) if HV > H else 0
                qt = q[n, t, ih].to(torch.float32)  # [K]
                kt = k[n, t, ih].to(torch.float32)  # [K]
                vt = v[n, t, ihv].to(torch.float32)  # [V]
                bt = b[n, t, ihv].to(torch.float32) if b.dim() >= 3 else b[t, ihv].to(torch.float32)

                if use_qk_l2norm_in_kernel:
                    qt = _l2norm(qt)
                    kt = _l2norm(kt)
                qt = qt * scale

                A = A_log[ihv].to(torch.float32)
                at = a[t, ihv].to(torch.float32) if a.dim() == 2 else a[n, t, ihv].to(torch.float32)
                dtb = dt_bias[ihv].to(torch.float32)

                x = at + dtb
                beta_x = softplus_beta * x
                softplus_x = torch.where(
                    beta_x <= softplus_threshold,
                    (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x)),
                    x,
                )
                g = -torch.exp(A) * softplus_x
                beta = torch.sigmoid(bt)

                h = h * torch.exp(g)
                vt = vt - (h * kt.unsqueeze(1)).sum(0)
                vt = vt * beta
                h = h + kt.unsqueeze(1) * vt.unsqueeze(0)
                out = (h * qt.unsqueeze(1)).sum(0)
                o[n, t, ihv] = out.to(o.dtype)

                if not disable_state_update and slot >= 0 and initial_state_source is not None:
                    initial_state_source[slot, ihv] = h.to(initial_state_source.dtype)

    return o.squeeze(0) if B == 1 and cu_seqlens is None else o


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Eager chunked GDN prefill.

    For prefill the recurrence is run token-by-token (the chunked-block
    optimization of the Triton path is not reproduced; correctness over speed).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    HV = v.shape[2]
    if scale is None:
        scale = K ** -0.5

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)
    q = q * scale

    o = torch.empty_like(v)
    final_h = None

    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        seq_ranges = [(int(cu_seqlens[i]), int(cu_seqlens[i + 1])) for i in range(N)]
    else:
        N = B
        seq_ranges = [(i * T, (i + 1) * T) for i in range(N)]

    for n, (bos, eos) in enumerate(seq_ranges):
        slot = int(initial_state_indices[n].item()) if initial_state_indices is not None else -1
        for ihv in range(HV):
            ih = ihv // (HV // H) if HV > H else 0
            if slot >= 0 and initial_state is not None:
                h = initial_state[slot, ihv].clone().to(torch.float32)
            else:
                h = torch.zeros(K, V, dtype=torch.float32, device=q.device)
            for t in range(bos, eos):
                qt = q[n, t, ih].to(torch.float32)
                kt = k[n, t, ih].to(torch.float32)
                vt = v[n, t, ihv].to(torch.float32)
                gt = g[n, t, ihv].to(torch.float32) if g.dim() == 3 else g[n, t].to(torch.float32)
                bt = beta[n, t, ihv].to(torch.float32) if beta.dim() == 3 else beta[n, t].to(torch.float32)

                h = h * torch.exp(gt)
                vt = vt - (h * kt.unsqueeze(1)).sum(0)
                vt = vt * torch.sigmoid(bt)
                h = h + kt.unsqueeze(1) * vt.unsqueeze(0)
                o[n, t, ihv] = (h * qt.unsqueeze(1)).sum(0).to(o.dtype)

            if slot >= 0 and initial_state is not None:
                initial_state[slot, ihv] = h.to(initial_state.dtype)

    return o, None, final_h
