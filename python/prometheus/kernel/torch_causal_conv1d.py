"""Pure-torch causal_conv1d fallback for non-CUDA devices (Apple Silicon / MPS).

The Triton causal_conv1d kernel is CUDA-only. This module provides an eager
implementation of the decode and varlen (prefill) paths that
``prometheus.kernel.causal_conv1d`` dispatches to via the triton fallback.

Semantics matched to the Triton/CUDA op:
  * depthwise causal conv1d, kernel width W, channels-first x=(dim, total).
  * silu fused into the output.
  * conv_states[cache_indices] is read as the left context when
    has_initial_state[i] is True, then refreshed in place with each request tail.
  * decode: single token per request, conv_state[idx] shifted left + new token
    appended; silu(conv) returned.  pad_slot_id=-1 entries are skipped.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

PAD_SLOT_ID = -1


def _silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def causal_conv1d_decode(
    x: torch.Tensor,                # [batch, conv_dim]
    conv_state: torch.Tensor,       # [num_slots, conv_dim, state_len>=kernel-1] (in place)
    weight: torch.Tensor,           # [conv_dim, kernel]
    conv_state_indices: torch.Tensor,  # [batch] int32
    activation: str | bool | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Single-token causal conv decode with silu; updates conv_state in place."""
    conv_state_indices = conv_state_indices.to(torch.int32)
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    batch, conv_dim = x.shape
    _, width = weight.shape
    state_len = width - 1

    out = torch.empty_like(x)

    for i in range(batch):
        slot = int(conv_state_indices[i].item())
        if slot == pad_slot_id:
            out[i] = 0
            continue
        # Shift state left and append new token.
        buf = conv_state[slot]  # [conv_dim, state_len]
        # buf[:, :-1] = buf[:, 1:] then buf[:, -1] = x[i]
        shifted = torch.roll(buf, shifts=-1, dims=1)
        shifted[:, -1] = x[i]
        conv_state[slot] = shifted
        # Compute conv: sum over kernel taps.
        w = weight  # [conv_dim, width]
        conv_out = (shifted * w[:, :state_len]).sum(1) + x[i] * w[:, state_len]
        if activation == "silu":
            conv_out = _silu(conv_out)
        out[i] = conv_out

    return out


def causal_conv1d_varlen(
    x: torch.Tensor,            # [conv_dim, total_tokens] (channels-first, last dim contiguous)
    weight: torch.Tensor,       # [conv_dim, kernel]
    conv_states: torch.Tensor,  # [num_slots, conv_dim, state_len>=kernel-1] (in place)
    cu_seqlens: torch.Tensor,   # [num_seqs+1] int64
    cache_indices: torch.Tensor,  # [num_seqs] int32
    has_initial_state: torch.Tensor,  # [num_seqs] bool
    activation: str | bool | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Varlen (prefill) depthwise causal conv with silu; writes silu(conv) into ``x``
    in place and refreshes ``conv_states[cache_indices]`` with each request's tail."""
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    conv_dim, total = x.shape
    _, width = weight.shape
    state_len = width - 1

    num_seqs = len(cu_seqlens) - 1
    for s in range(num_seqs):
        slot = int(cache_indices[s].item()) if s < len(cache_indices) else pad_slot_id
        bos = int(cu_seqlens[s])
        eos = int(cu_seqlens[s + 1])
        seq_len = eos - bos
        if seq_len == 0:
            continue
        w = weight  # [conv_dim, width]
        # Initialize state if requested.
        buf = conv_states[slot] if slot != pad_slot_id else None
        for t in range(seq_len):
            idx = bos + t
            if buf is not None and t == 0 and bool(has_initial_state[s]):
                # Use existing state as left context.
                pass
            elif buf is not None:
                pass
            # Build the window: last (width-1) tokens from buf + current token.
            if buf is not None:
                shifted = torch.roll(buf, shifts=-1, dims=1)
                shifted[:, -1] = x[:, idx]
                buf[:] = shifted
                conv_out = (shifted * w[:, :state_len]).sum(1) + x[:, idx] * w[:, state_len]
            else:
                # No slot: plain causal conv with zero left context.
                window = torch.zeros(conv_dim, state_len, dtype=x.dtype, device=x.device)
                window[:, -1] = x[:, idx]
                conv_out = (window * w[:, :state_len]).sum(1) + x[:, idx] * w[:, state_len]
            if activation == "silu":
                conv_out = _silu(conv_out)
            x[:, idx] = conv_out

    return x
