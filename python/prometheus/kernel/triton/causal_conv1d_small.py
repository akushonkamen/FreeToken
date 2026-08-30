"""Tiny-window varlen causal conv1d + silu for spec-verify / short extends.

The sgl_kernel varlen op + surrounding transpose/contiguous/.to(int32) costs
10-100us per GDN layer at verify shapes (total <= 32 tokens); this Triton kernel
does the whole thing in ONE launch: takes x [total, dim] row-major directly (no
transpose), reads the per-seq left context from conv_states[idx], writes silu(conv)
to a fresh contiguous out [total, dim], and refreshes conv_states[idx] in place
(last W-1 raw inputs per sequence). Semantics matched to the vendored op
(conv_states oldest->newest, raw pre-activation state, silu on output).

The old state columns are snapshotted into registers BEFORE any state store and
all later uses (output taps + refresh shift) select from the snapshot: the
refresh writes the same locations the output reads, and predicated branches let
the compiler schedule those stores ahead of the loads (deterministically wrong
for ~half the channels at T=1), so no state re-load may happen after the first.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _conv1d_small_kernel(
    x_ptr, w_ptr, st_ptr, o_ptr,
    cu_ptr, idx_ptr,
    dim,
    stride_x_t, stride_x_d,
    stride_st_s, stride_st_d, stride_st_w,
    stride_o_t, stride_o_d,
    KERNEL_WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_T: tl.constexpr,
):
    pid = tl.program_id(0)
    i_n = tl.program_id(1)
    offs = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    m = offs < dim

    bos = tl.load(cu_ptr + i_n).to(tl.int32)
    eos = tl.load(cu_ptr + i_n + 1).to(tl.int32)
    slot = tl.load(idx_ptr + i_n).to(tl.int32)
    T = eos - bos
    STATE_LEN: tl.constexpr = KERNEL_WIDTH - 1

    st_base = st_ptr + slot * stride_st_s + offs * stride_st_d
    # snapshot old state cols (oldest -> newest) -- the ONLY state loads in the kernel.
    # (The state-refresh writes these same locations; predicated branches let the
    # compiler schedule stores ahead of loads, so no state re-load may happen after
    # the first store.)  W is small constexpr, so explicit unrolled loads compile fine.
    old0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    old1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    old2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    if STATE_LEN >= 1:
        old0 = tl.load(st_base + 0 * stride_st_w, mask=m, other=0.0).to(tl.float32)
    if STATE_LEN >= 2:
        old1 = tl.load(st_base + 1 * stride_st_w, mask=m, other=0.0).to(tl.float32)
    if STATE_LEN >= 3:
        old2 = tl.load(st_base + 2 * stride_st_w, mask=m, other=0.0).to(tl.float32)

    # outputs: y[t] = silu(sum_j w[j] * hist[t - W + 1 + j]); hist position < 0 -> old col
    for t in range(0, MAX_T):
        if t < T:
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            for j in tl.static_range(KERNEL_WIDTH):
                src = t - KERNEL_WIDTH + 1 + j
                wv = tl.load(w_ptr + offs * KERNEL_WIDTH + j, mask=m, other=0.0).to(tl.float32)
                if src >= 0:
                    xv = tl.load(x_ptr + (bos + src) * stride_x_t + offs * stride_x_d,
                                 mask=m, other=0.0).to(tl.float32)
                    acc += wv * xv
                else:
                    col = src + KERNEL_WIDTH - 1  # 0..STATE_LEN-1
                    sv = tl.where(col == 0, old0,
                        tl.where(col == 1, old1,
                        tl.where(col == 2, old2,
                        tl.zeros([BLOCK_D], dtype=tl.float32))))
                    acc += wv * sv
            y = acc * tl.sigmoid(acc)
            tl.store(o_ptr + (bos + t) * stride_o_t + offs * stride_o_d,
                     y.to(o_ptr.dtype.element_ty), mask=m)

    # state refresh: last STATE_LEN raw inputs ending at eos-1 (oldest -> newest);
    # short windows take the tail from the snapshot (never re-loaded from memory)
    for j in range(STATE_LEN):
        src = T - STATE_LEN + j
        if src >= 0:
            nv = tl.load(x_ptr + (bos + src) * stride_x_t + offs * stride_x_d,
                         mask=m, other=0.0)
        else:
            col = src + STATE_LEN  # old col that shifts into position j
            fv = tl.where(col == 0, old0,
                tl.where(col == 1, old1,
                tl.where(col == 2, old2,
                tl.zeros([BLOCK_D], dtype=tl.float32))))
            nv = fv.to(st_ptr.dtype.element_ty)
        tl.store(st_base + j * stride_st_w, nv, mask=m)


def causal_conv1d_varlen_small(
    x: torch.Tensor,             # [total, dim] row-major (raw pre-conv activations)
    weight: torch.Tensor,        # [dim, kernel] (row-major)
    conv_states: torch.Tensor,   # [num_slots, dim, kernel-1] (updated in place)
    cu_seqlens: torch.Tensor,    # [n+1] int (any int dtype)
    cache_indices: torch.Tensor, # [n] int slot per sequence
    max_t: int,                  # longest per-seq window (host int; graph-capture safe)
) -> torch.Tensor:
    """Returns silu(conv(x)) as a fresh contiguous [total, dim] tensor."""
    total, dim = x.shape
    n = cu_seqlens.shape[0] - 1
    W = weight.shape[1]
    # The kernel keeps the left context in registers old0/1/2 (STATE_LEN<=3):
    # a wider conv must take the sgl path, never silently zero-fill old3.
    assert W <= 4, f"tiny conv kernel supports KERNEL_WIDTH<=4, got W={W}"
    out = torch.empty_like(x)
    BLOCK_D = 256
    grid = (triton.cdiv(dim, BLOCK_D), n)
    _conv1d_small_kernel[grid](
        x, weight, conv_states, out,
        cu_seqlens, cache_indices,
        dim,
        x.stride(0), x.stride(1),
        conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
        out.stride(0), out.stride(1),
        KERNEL_WIDTH=W, BLOCK_D=BLOCK_D, MAX_T=max_t,
        num_warps=4,
    )
    return out
