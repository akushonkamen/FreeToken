"""n-gram speculative decoding (v1).

A per-request 3-gram index over prompt + generated tokens proposes up to ``k`` draft
continuations. A spec-verify batch feeds [last committed token + drafts] through ONE
ragged extend forward (``phase="prefill"`` for attention, decode-style MoE routing)
and the scheduler commits the longest greedy-matching draft prefix plus the
correction/bonus token -- so one PCIe-bound MoE forward amortizes 1..k+1 output
tokens. Greedy longest-prefix verification is lossless: the committed sequence equals
what consecutive greedy decode steps would emit.

Hybrid linear-attention (GDN) models ride the same path with a state-snapshot
rollback: the verify forward advances the recurrent+conv state in place by all 1+k
rows, which KV-slot rollback alone cannot undo. ``SpecLinearSnapshots`` holds one
pre-forward state row per request; a partially-accepted draft restores it and
re-advances the state over the accepted span WITHOUT another model forward -- the
verify forward stashed each GDN layer's pre-conv projections (conv_in/a/b) on the
batch, so ``Scheduler._replay_spec_states`` re-runs only the state-updating kernels
over the accepted row slice. A fully-accepted draft needs neither (the
forward-advanced state lands exactly on the committed prefix).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Dict, Iterable, List

import torch
from prometheus.core import Batch, Req

if TYPE_CHECKING:
    from prometheus.kvcache.linear_state_pool import LinearStatePool
    from prometheus.models.qwen3_5_moe.mtp import MTPDraftManager

_NGRAM = 3  # draft context size


def _maybe_pinned(t: torch.Tensor) -> torch.Tensor:
    """Pinning only buys the async H2D copy below; without a device it just raises."""
    return t.pin_memory() if torch.cuda.is_available() else t


class NGramDraftModel:
    """Per-request 3-gram index over prompt + generated token ids.

    ``update`` is incremental (call it before ``propose``; it only walks tokens
    appended since the last call), so the schedule-time cost stays O(new tokens).
    Only the most recent occurrences are ever read, so each position deque is
    bounded to keep long-generation memory flat.
    """

    def __init__(self, n: int = _NGRAM, max_occurrences: int = 8):
        self.n = n
        self.max_occurrences = max_occurrences
        self._index: Dict[int, Dict[tuple, deque]] = {}
        self._next_start: Dict[int, int] = {}

    def update(self, uid: int, ids: List[int]) -> None:
        index = self._index.setdefault(uid, {})
        start = self._next_start.get(uid, 0)
        end = len(ids) - self.n + 1
        for i in range(start, max(start, end)):
            key = tuple(ids[i : i + self.n])
            index.setdefault(key, deque(maxlen=self.max_occurrences)).append(i)
        if end > start:
            self._next_start[uid] = end

    def propose(self, uid: int, ids: List[int], k: int) -> List[int]:
        """Continuation after the most recent NON-SELF occurrence of the trailing
        3-gram; ``[]`` when there is none (the caller falls back to a normal decode
        step). The result may be shorter than ``k`` (source occurrence near the end).
        """
        n = len(ids)
        if k <= 0 or n < self.n:
            return []
        key = tuple(ids[n - self.n : n])
        occ = self._index.get(uid, {}).get(key)
        if not occ:
            return []
        p = occ[-1]
        if p == n - self.n:  # the trailing 3-gram matches itself; no continuation yet
            if len(occ) < 2:
                return []
            p = occ[-2]
        return ids[p + self.n : p + self.n + k]

    def remove(self, uid: int) -> None:
        self._index.pop(uid, None)
        self._next_start.pop(uid, None)


class SpecLinearSnapshots:
    """Per-request GDN (recurrent + conv) state snapshots for speculative verification.

    The spec-verify forward feeds [last committed token + k drafts] through the GDN
    chunk kernel, which advances the live state slot IN PLACE by all 1+k steps. The
    kernel only materializes per-CHUNK (64-token) intermediate states, so a mid-span
    state cannot be sliced out: rollback is whole-slot. This buffer holds one
    pre-forward snapshot row per request (keyed by ``table_idx``, which is < the row
    count and stable across the forward/drain window), covering both the hybrid
    live slot (``linear_slot_idx``) and the naive ``table_idx`` slot keying.

    A single buffer (no ping-pong) suffices: the spec state machine drains batch N
    before batch N+1 is scheduled, so a snapshot is never overwritten while its
    restore is still pending. Snapshot/restore are plain device-to-device copies
    issued on the engine stream by the scheduler (program-ordered around the
    forward's in-place state updates).
    """

    def __init__(self, pool: "LinearStatePool", num_rows: int) -> None:
        self.pool = pool
        # Same per-slot geometry/dtype as the pool, independent storage: a runtime
        # pool rebuild (LinearStatePool.rebuild) reallocates the pool tensors but
        # never changes the per-slot geometry, so the buffer stays valid (contents
        # are per-forward anyway).
        self.conv_snap = pool.conv_states[:, :num_rows].clone()
        self.rec_snap = pool.recurrent_states[:, :num_rows].clone()

    def _slot(self, req: Req) -> int:
        # hybrid-radix live slot when allocated, else table_idx (naive GDN keying)
        return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx

    def slot(self, req: Req) -> int:
        """The request's live GDN state slot (public: the O(1) rollback re-advance
        targets it directly; see Scheduler._replay_spec_states)."""
        return self._slot(req)

    def snapshot(self, reqs: Iterable[Req]) -> None:
        """Copy each request's live GDN state into its buffer row (engine stream)."""
        pool = self.pool
        for r in reqs:
            row, slot = r.table_idx, self._slot(r)
            self.conv_snap[:, row].copy_(pool.conv_states[:, slot])
            self.rec_snap[:, row].copy_(pool.recurrent_states[:, slot])

    def restore(self, req: Req) -> None:
        """Copy a request's snapshot row back into its live GDN slot (engine stream)."""
        pool = self.pool
        row, slot = req.table_idx, self._slot(req)
        pool.conv_states[:, slot].copy_(self.conv_snap[:, row])
        pool.recurrent_states[:, slot].copy_(self.rec_snap[:, row])

    def bytes(self) -> int:
        return (
            self.conv_snap.numel() * self.conv_snap.element_size()
            + self.rec_snap.numel() * self.rec_snap.element_size()
        )


def _seat_drafts_and_build(
    reqs: List[Req], drafts: List[List[int]], token_pool: torch.Tensor
) -> Batch:
    """Seat per-req draft lists and build the spec-verify batch (shared by the n-gram
    and MTP draft providers)."""
    for req, draft in zip(reqs, drafts):
        base = req.device_len  # position m+1: first slot past the committed tokens
        host = _maybe_pinned(torch.tensor(draft, dtype=token_pool.dtype))
        token_pool[req.table_idx, base : base + len(draft)].copy_(
            host.to(token_pool.device, non_blocking=True)
        )
        req.device_len = base + len(draft)
    batch = Batch(reqs=list(reqs), phase="prefill", spec_verify=True)
    batch.spec_drafts = drafts
    return batch


def _spec_eligible(reqs: List[Req]) -> bool:
    """All-or-nothing eligibility (shared): greedy sampling, no multimodal prompt, and
    a commit budget for at least one draft token + the bonus."""
    for req in reqs:
        if not req.sampling_params.is_greedy or req.mm_embeds is not None:
            return False
        if req.remain_len - 1 <= 0:
            return False
    return True


def make_spec_batch(
    reqs: List[Req],
    k: int,
    token_pool: torch.Tensor,
    draft_model: NGramDraftModel,
) -> Batch | None:
    """Build a spec-verify batch, or None to fall back to a normal decode batch.

    v1 is all-or-nothing: ANY request that cannot spec (non-greedy sampling, a
    multimodal request, no commit budget, or no draft hit) sends the whole batch
    through the plain decode path, which stays CUDA-graph eligible.

    State at call time (steady decode): cached_len == m, device_len == m+1, and
    input_ids/token_pool already seat every committed token through position m.
    This seats drafts d1..dk at positions [m+1, m+1+k) and bumps device_len so
    ``allocate_paged`` reserves the 1+k slots the forward fills.
    """
    if not _spec_eligible(reqs):
        return None
    drafts: List[List[int]] = []
    for req in reqs:
        # This step commits at most len(draft)+1 tokens; the -1 keeps the final
        # append_host inside max_device_len (the equivalent decode steps would too).
        budget = req.remain_len - 1
        draft_model.update(req.uid, req.input_ids.tolist())
        drafts.append(draft_model.propose(req.uid, req.input_ids.tolist(), min(k, budget)))
    if any(not draft for draft in drafts):
        return None
    return _seat_drafts_and_build(reqs, drafts, token_pool)


def make_spec_batch_mtp(
    reqs: List[Req],
    k: int,
    token_pool: torch.Tensor,
    manager: "MTPDraftManager",
) -> Batch | None:
    """Spec-verify batch from the MTP draft head. The drafts were chained at the
    previous drain (prefill seeding / post-verify rechain); this only trims them to
    the commit budget and seats them. All-or-nothing like the n-gram path."""
    if not _spec_eligible(reqs):
        return None
    drafts = manager.seat_drafts(reqs, k)
    if drafts is None:
        return None
    # Device-side seating: the chain's GPU tensors stream straight into the token
    # pool (no CPU roundtrip), so the verify forward launches before the drafts
    # ever reach the host -- scheduling and the verify-graph launch overlap the
    # chain's still-queued kernels instead of idling the GPU between them.
    deferred = []
    for req, draft_t in zip(reqs, drafts):
        base = req.device_len  # position m+1: first slot past the committed tokens
        n = draft_t.shape[0]
        token_pool[req.table_idx, base : base + n].copy_(draft_t)
        req.device_len = base + n
        state = manager.states[req.uid]
        deferred.append((state.draft_pin, n))
    batch = Batch(reqs=list(reqs), phase="prefill", spec_verify=True)
    batch.spec_draft_deferred = deferred
    return batch
