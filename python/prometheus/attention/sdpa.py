"""SDPA attention backend for non-CUDA devices (Apple Silicon / MPS).

The Triton paged-attention backend is CUDA-only. On MPS, use PyTorch's
``scaled_dot_product_attention`` (SDPA) which has an MPS-optimized path.

Only FULL attention is served (SWA/GDN linear paths use the GDN kernel fallback).
Decode is per-request attention against the KV cache; prefill is varlen SDPA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from prometheus.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from prometheus.models import ModelConfig


@dataclass
class SDPAMetadata(BaseAttnMetadata):
    seq_lens: torch.Tensor = None
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    positions: torch.Tensor = None


class SDPAAttentionBackend(BaseAttnBackend):
    """Minimal SDPA-based FULL attention backend for MPS/CPU."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = None
        self.max_seq_len = 0

    def _ensure_device(self):
        if self.device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        self._ensure_device()
        metadata = batch.attn_metadata
        if not isinstance(metadata, SDPAMetadata):
            metadata = SDPAMetadata()

        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_raw = self.kvcache.k_cache(layer_id)
        v_raw = self.kvcache.v_cache(layer_id)
        kv_heads, head_dim = k_raw.shape[-2], k_raw.shape[-1]

        q_heads = q.shape[-2]
        is_prefill = batch.is_prefill

        if is_prefill:
            return self._prefill(q, k_raw, v_raw, batch, q_heads, kv_heads, head_dim)
        else:
            return self._decode(q, k_raw, v_raw, batch, q_heads, kv_heads, head_dim)

    def _decode(self, q, k_cache, v_cache, batch, q_heads, kv_heads, head_dim):
        """Single-token decode: q [B, H, D] against full KV cache."""
        num_reqs = len(batch.reqs)
        o = q.new_empty(num_reqs, q_heads, head_dim)

        for i, req in enumerate(batch.reqs):
            seq_len = req.cached_len + 1  # +1 for the new token
            kv_start = req.table_idx if hasattr(req, "table_idx") else 0
            locs = batch.out_loc[i * 1: (i + 1)]

            # Gather KV for this request from the page table.
            page_table = self.kvcache.page_table if hasattr(self.kvcache, "page_table") else None
            if page_table is not None:
                pages = page_table[req.table_idx][:seq_len]
                keys = k_cache[pages].reshape(-1, kv_heads, head_dim)[:seq_len]
                vals = v_cache[pages].reshape(-1, kv_heads, head_dim)[:seq_len]
            else:
                keys = k_cache[:seq_len]
                vals = v_cache[:seq_len]

            qi = q[i : i + 1].transpose(0, 1)  # [H, 1, D]
            ki = keys.transpose(0, 1)  # [kv_heads, S, D]
            vi = vals.transpose(0, 1)  # [kv_heads, S, D]

            # GQA: repeat KV heads to match Q heads.
            if q_heads != kv_heads:
                rep = q_heads // kv_heads
                ki = ki.repeat_interleave(rep, dim=0)
                vi = vi.repeat_interleave(rep, dim=0)

            scale = head_dim ** -0.5
            out = F.scaled_dot_product_attention(qi, ki, vi, scale=scale)  # [H, 1, D]
            o[i] = out.squeeze(1)

        return o

    def _prefill(self, q, k_cache, v_cache, batch, q_heads, kv_heads, head_dim):
        """Varlen prefill: per-request SDPA over its prefix."""
        num_reqs = len(batch.reqs)
        total = q.shape[0]
        o = q.new_empty(total, q_heads, head_dim)
        offset = 0

        for i, req in enumerate(batch.reqs):
            seq_len = req.input_len
            qi = q[offset : offset + seq_len].transpose(0, 1)  # [H, S, D]

            page_table = self.kvcache.page_table if hasattr(self.kvcache, "page_table") else None
            if page_table is not None:
                pages = page_table[req.table_idx][:seq_len]
                keys = k_cache[pages].reshape(-1, kv_heads, head_dim)[:seq_len]
                vals = v_cache[pages].reshape(-1, kv_heads, head_dim)[:seq_len]
            else:
                keys = k_cache[:seq_len]
                vals = v_cache[:seq_len]

            ki = keys.transpose(0, 1)
            vi = vals.transpose(0, 1)

            if q_heads != kv_heads:
                rep = q_heads // kv_heads
                ki = ki.repeat_interleave(rep, dim=0)
                vi = vi.repeat_interleave(rep, dim=0)

            scale = head_dim ** -0.5
            out = F.scaled_dot_product_attention(qi, ki, vi, scale=scale)  # [H, S, D]
            o[offset : offset + seq_len] = out.transpose(0, 1)
            offset += seq_len

        return o

    def prepare_metadata(self, batch: Batch) -> None:
        self._ensure_device()
        num_reqs = len(batch.reqs)
        if batch.is_prefill:
            seq_lens = torch.tensor([r.input_len for r in batch.reqs], dtype=torch.int32, device=self.device)
            cu_q = torch.zeros(num_reqs + 1, dtype=torch.int32, device=self.device)
            for i, r in enumerate(batch.reqs):
                cu_q[i + 1] = cu_q[i] + r.input_len
            batch.attn_metadata = SDPAMetadata(
                seq_lens=seq_lens,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_q,
                positions=torch.zeros(num_reqs, dtype=torch.int32, device=self.device),
            )
        else:
            batch.attn_metadata = SDPAMetadata(
                seq_lens=torch.tensor([r.cached_len + 1 for r in batch.reqs], dtype=torch.int32, device=self.device),
            )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.max_seq_len = max_seq_len

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.prepare_metadata(batch)

    def reset_capture(self) -> None:
        pass
