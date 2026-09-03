"""Vendored flash-linear-attention (fla) GatedDeltaNet triton kernels.

Borrowed from sglang's fla fork (`sglang/srt/layers/attention/fla/`), which itself
adapts https://github.com/fla-org/flash-linear-attention. We vendor sglang's fork — not
upstream fla — because the fork carries features Prometheus needs that upstream lacks:
the indexed state pool (`initial_state_indices` threaded into the chunk recurrence) and
the fused sigmoid-gating decode kernel, plus an H100-safe single-config autotune that
won't corrupt the in-place state pool.

Kernel code is intentionally "dirty" (pure triton, sglang lineage) and lives here under
``kernel/fla/`` rather than under ``models/`` so the model code stays clean.

Public entry points:
- ``chunk_gated_delta_rule`` — chunked prefill; reads/writes the recurrent state pool
  in place by ``initial_state_indices``, with optional in-kernel q/k l2norm.
- ``fused_sigmoid_gating_delta_rule_update`` — single-token decode; gating + in-kernel
  l2norm + delta-rule update + per-slot state read/write, all in one kernel.

Provenance: https://github.com/sgl-project/sglang, ``python/sglang/srt/layers/attention/fla/``
(NVIDIA path only; the ``is_intel`` XPU branch and the ``torch_release`` sglang import were
stripped/inlined on vendoring). Keep ``chunk_delta_h.py``'s single fixed ``triton.Config`` — restoring
upstream's multi-config autotune corrupts the in-place state pool. Tune via the env knobs
``SGLANG_GDN_CHUNK_H_BV`` / ``SGLANG_GDN_CHUNK_H_NUM_WARPS`` / ``SGLANG_GDN_CHUNK_H_NUM_STAGES``.
"""
try:
    import triton  # noqa: F401 -- guards the submodules below

    from prometheus.kernel.fla.chunk import chunk_gated_delta_rule
    from prometheus.kernel.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from prometheus.kernel.fla.layernorm_gated import rms_norm_gated
except ImportError:
    # Non-CUDA host (e.g. Apple Silicon / MPS): Triton has no wheel there.
    # Fall back to the pure-torch eager implementations.
    from prometheus.kernel.fla.torch_fallback import (
        chunk_gated_delta_rule,
        fused_sigmoid_gating_delta_rule_update,
    )

    def rms_norm_gated(x, weight, eps, z=None):
        """Eager gated RMSNorm fallback (used by GDN prefill)."""
        t = x.to(torch.float32)
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)
        t = t * weight
        if z is not None:
            t = t * torch.sigmoid(z)
        return (x.dtype == torch.bfloat16 and t.to(torch.bfloat16)) or t

__all__ = [
    "chunk_gated_delta_rule",
    "fused_sigmoid_gating_delta_rule_update",
    "rms_norm_gated",
]
