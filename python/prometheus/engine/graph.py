from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

import torch
from prometheus.core import Batch, Req, get_global_ctx
from prometheus.distributed import get_tp_info
from prometheus.utils import init_logger, mem_GB
from prometheus.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from prometheus.attention import BaseAttnBackend
    from prometheus.models import BaseLLMModel
    from prometheus.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class RechainGraphEntry:
    """One fixed-shape rechain graph for a specific committed row count.

    The varlen cu_seqlens approach (one graph, cu[1]=committed) is broken: the GDN
    kernels reshape q/k/v to ``conv_in.shape[0]`` (=max_rows) but cu_seqlens says
    ``committed`` — when committed < max_rows the extra rows are stale buffer data
    that contaminates the recurrent state. So we capture one graph per committed
    value (1..k), each with buffer shapes exactly [committed, ...]."""
    graph: torch.cuda.CUDAGraph
    cu_seqlens: torch.Tensor      # [2] int64, [0, rows] (rows == this entry's committed)
    cache_idx: torch.Tensor       # [1] int32 slot id for GDN kernels
    has_init: torch.Tensor        # [1] bool, const True
    slot_buf: torch.Tensor        # [1] int64 restore destination slot
    row_buf: torch.Tensor         # [1] int64 restore source row
    # Per-layer persistent input buffers (graph-pool, contents refreshed per replay)
    conv_in_bufs: Dict[int, torch.Tensor]   # layer_id -> [rows, conv_dim]
    a_bufs: Dict[int, torch.Tensor]        # layer_id -> [rows, num_v_heads]
    b_bufs: Dict[int, torch.Tensor]        # layer_id -> [rows, num_v_heads]


@dataclass
class RechainGraph:
    """Captured GDN state re-advance for spec partial-accept rollback (max_rows=k).

    The eager re-advance path (``Scheduler._replay_spec_states`` →
    ``readvance_gdn_states`` → 48 layers × ``spec_readvance``) launches ~96 tiny
    kernels per step entirely from Python — ~37 ms of pure dispatch overhead for a
    few microseconds of GPU work. This graph captures the whole rechain as one
    ``graph.replay()`` launch.

    One entry per committed value (1..max_rows): the GDN kernels reshape q/k/v to
    ``conv_in.shape[0]``, so the buffer shape must equal the actual committed count
    (a single varlen graph with cu[1]=committed leaves stale rows in the reshape).
    ``replay_rechain`` picks the entry matching ``committed``.

    All step-varying state lives in persistent buffers filled by ``replay_rechain``
    before replay:

      cache_idx[0]           the request's live GDN state slot
      slot_buf / row_buf     the snapshot restore indices
      per-layer conv_in/a/b  the stashed verify-forward projections, sliced [0:committed]

    Everything else is capture-stable by construction: fixed row count per entry,
    the conv and chunk kernels are cu_seqlens-driven shape-launched, and the FLA
    chunk-index/offset tables (built from the constant cu_seqlens at warmup) are
    pinned against LRU(4) eviction exactly like SpecGraph.
    """

    max_rows: int
    entries: Dict[int, RechainGraphEntry]   # committed -> entry
    # Pin FLA chunk tables per committed value (same hazard as SpecGraph).
    fla_chunk_indices: Dict[int, torch.Tensor] = field(default_factory=dict)
    fla_chunk_offsets: Dict[int, torch.Tensor] = field(default_factory=dict)
    fla_chunk_indices_o: Dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class SpecGraph:
    """Captured spec-verify forward (bs=1, ext=1+k rows, page_size=1).

    The eager spec-verify forward costs ~98ms of pure CPU launch time (the GPU work
    is tiny); this graph replays the whole model forward + argmax as one launch. All
    step-varying state lives in persistent buffers filled by ``replay_spec`` before
    ``graph.replay()``:

      input_ids / positions / out_loc   the 1+k verify rows
      kv_indices / kv_indptr            the request's page-table row + current length
      prefix_lens                       cached prefix length m
      fla_cache_idx                     the request's live GDN state slot

    Everything else is capture-stable by construction: fixed row count (the graph is
    only used when every draft has exactly k tokens), triton extend attention reads
    lengths from device tensors with a shape-only grid, the GDN chunk kernels are
    shape-launched, and the one host-syncing helper (``prepare_chunk_indices``) is
    memoized per tensor identity (tensor_cache) by the pre-capture warmup run. The
    captured ``hidden_states`` and per-layer GDN ``spec_gdn_stash`` are graph-pool
    tensors whose CONTENTS refresh on every replay; replay_spec re-points the live
    batch at them so the scheduler drain and the partial-accept rollback
    (``Scheduler._replay_spec_states``) consume them as if the forward ran eagerly.
    """

    ext: int
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor        # [ext] int32
    positions: torch.Tensor        # [ext] int32
    out_loc: torch.Tensor          # [ext] int32
    next_tokens: torch.Tensor      # [ext] int32 (captured argmax of the logits)
    kv_indices: torch.Tensor       # [max_seq_len] int32 page-table slots
    kv_indptr: torch.Tensor        # [2] int32
    prefix_lens: torch.Tensor      # [1] int32
    fla_cache_idx: torch.Tensor    # [1] int32 live GDN state slot
    hidden_states: torch.Tensor | None = None   # graph-pool [ext, hidden]
    stash: dict | None = None                     # capture batch's spec_gdn_stash
    # Metadata objects built at capture: they own the small const device tensors the
    # recorded kernels read (FLA cu_seqlens / has_initial_state, triton cu_seqlens_q).
    # Keeping these references is load-bearing -- the capture batch itself goes out of
    # scope, and without an owner those tensors would be freed and their memory reused.
    fla_metadata: object | None = None
    attn_metadata: object | None = None
    # The chunk kernels' launch reads the tensor_cache'd chunk-index table (built from
    # the constant cu_seqlens at warmup). That LRU(4) cache would otherwise be the
    # table's only owner: prefill forwards allocate their own entries and can evict
    # ours, letting the allocator reuse the memory the recorded kernels still read.
    # This reference pins it for the graph's lifetime.
    fla_chunk_indices: torch.Tensor | None = None
    # Same pin for the chunk-offset table chunk_gated_delta_rule_fwd_h feeds its
    # kernel (prepare_chunk_offsets, a separate tensor_cache): the spec rollback's
    # re-advance (Scheduler._replay_spec_states) also runs the chunk path with its own
    # cu_seqlens, and its cache insertions evict -- and thereby free -- this table,
    # leaving the recorded kernel reading reallocated memory (replay #2 illegal
    # access). Load-bearing, exactly like fla_chunk_indices.
    fla_chunk_offsets: torch.Tensor | None = None
    # And a third entry, for chunk_fwd_o alone: it derives its own block size from T
    # (BT = min(CHUNK_SIZE, max(16, next_pow2(T)) ) -- 16 for ext < 16) and resolves a
    # SEPARATE tensor_cache'd index table for that BT. Same eviction hazard, same pin.
    fla_chunk_indices_o: torch.Tensor | None = None
    # qwen4_exp PLE conv-input stash (layer_id -> buffer): the captured conv
    # writeback advances the state over all 1+k rows, so the drain's partial-accept
    # rollback re-lands it from these rows (mirrors `stash` for the GDN).
    ple_stash: dict | None = None


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        from prometheus.attention.linear import FLAMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        spec_ext: int | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.model = model
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.stream = stream
        self.device = device
        self.max_seq_len = max_seq_len
        self._graph_pool = None
        self.spec: SpecGraph | None = None
        self.rechain: RechainGraph | None = None
        self._capture_graphs(max_seq_len, vocab_size, model)
        if spec_ext is not None and self.max_graph_bs > 0:
            self._capture_spec_graph(model, vocab_size, spec_ext)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            batch.cuda_graph_capture = True
            model.prepare_cuda_graph_capture(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        self._reset_moe_offload_cache()
        self._graph_pool = pool  # shared with the spec-verify graph (memory reuse)
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_spec_graph(self, model: BaseLLMModel, vocab_size: int, spec_ext: int) -> None:
        """Capture the fixed-shape spec-verify forward (see SpecGraph). Runs after the
        decode captures so it can share their memory pool; uses the triton spec backend
        (ctx.spec_attn_backend) that spec-verify batches are routed to."""
        from prometheus.attention.linear import FLAMetadata
        from prometheus.attention.triton import TritonMetadata

        ctx = get_global_ctx()
        assert ctx.spec_attn_backend is not None, "spec graph requires the spec attn backend"
        device, ext = self.device, spec_ext
        dummy = self.dummy_req
        cap = SpecGraph(
            ext=ext,
            graph=torch.cuda.CUDAGraph(),
            input_ids=torch.zeros(ext, dtype=torch.int32, device=device),
            positions=torch.arange(ext, dtype=torch.int32, device=device),
            out_loc=torch.zeros(ext, dtype=torch.int32, device=device),
            next_tokens=torch.zeros(ext, dtype=torch.int32, device=device),
            kv_indices=torch.zeros(self.max_seq_len, dtype=torch.int32, device=device),
            kv_indptr=torch.tensor([0, ext], dtype=torch.int32, device=device),
            prefix_lens=torch.zeros(1, dtype=torch.int32, device=device),
            fla_cache_idx=torch.zeros(1, dtype=torch.int32, device=device),
        )
        # Capture-time scratch pointers: KV/out_loc slot 0 and the GDN padding slot are
        # written/read during the warmup + capture forwards only (mirrors the decode
        # capture's zero out_loc); every real replay overwrites them via replay_spec.
        cap.kv_indices[:ext].zero_()

        # A fake continuing request: cached_len > 0 so the GDN span carries state
        # (has_initial_state=True) exactly like a real mid-generation verify step.
        # cached_len must stay < ext (the Req ctor asserts cached_len < len(input_ids)),
        # so small-k captures fall back to a 1-token history.
        fake_cached = 4 if ext > 4 else 1
        fake = Req(
            input_ids=torch.zeros(ext, dtype=torch.int32, device="cpu"),
            table_idx=dummy.table_idx,
            cached_len=fake_cached,
            output_len=ext + 8,
            uid=-1,
            sampling_params=None,  # type: ignore[arg-type]
            cache_handle=None,  # type: ignore[arg-type]
        )
        fake.linear_slot_idx = dummy.linear_slot_idx
        fake.device_len = fake_cached + ext
        batch = Batch(reqs=[fake], phase="prefill", spec_verify=True)
        batch.padded_reqs = batch.reqs
        batch.spec_drafts = [[0] * (ext - 1)]
        batch.input_ids = cap.input_ids
        batch.positions = cap.positions
        batch.out_loc = cap.out_loc
        batch.return_hidden = True
        batch.forward_row_lens = [ext]
        # FLA metadata over persistent buffers: cu is a constant, the slot is a buffer.
        cap.fla_cache_idx.fill_(dummy.linear_slot_idx
                                if dummy.linear_slot_idx is not None else dummy.table_idx)
        fla_cu = torch.tensor([0, ext], dtype=torch.int64, device=device)
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=fla_cu,
            cache_indices=cap.fla_cache_idx,
            has_initial_state=torch.tensor([True], dtype=torch.bool, device=device),
        )
        # Pin the tensor_cache'd chunk-index table (see SpecGraph.fla_chunk_indices):
        # resolve it here so the warmup/capture below hit the same cached tensor.
        import triton

        from prometheus.kernel.fla.chunk import CHUNK_SIZE
        from prometheus.kernel.fla.index import prepare_chunk_indices, prepare_chunk_offsets

        cap.fla_chunk_indices = prepare_chunk_indices(fla_cu, CHUNK_SIZE)
        cap.fla_chunk_offsets = prepare_chunk_offsets(fla_cu, CHUNK_SIZE)
        bt_o = min(CHUNK_SIZE, max(16, triton.next_power_of_2(ext)))
        if bt_o != CHUNK_SIZE:
            cap.fla_chunk_indices_o = prepare_chunk_indices(fla_cu, bt_o)
        batch.attn_metadata = TritonMetadata(
            cu_seqlens_q_gpu=torch.tensor([0, ext], dtype=torch.int32, device=device),
            indptr=cap.kv_indptr,
            indices=cap.kv_indices,
            q_to_req=torch.zeros(ext, dtype=torch.int32, device=device),    # extend path: unused
            q_positions=torch.zeros(ext, dtype=torch.int64, device=device),  # extend path: unused
            is_decode=False,
            prefix_lens=cap.prefix_lens,
            max_q_len=ext,
        )
        cap.prefix_lens.fill_(fake_cached)

        # qwen4_exp PLE: the spec-verify rows must ride the graph-safe branches --
        # the eager prefill branches host-sync on cu_seqlens (cu.cpu()) and abort
        # the capture. Flag the batch and size the model's persistent buffers
        # BEFORE the warmup run, so _ngram_embeddings/_conv_forward read only the
        # staged buffers inside the captured region.
        #
        # Size ONLY the spec-owned buffers (prepare_spec_graph_capture). Do NOT
        # route through model.prepare_cuda_graph_capture here: that call sizes the
        # DECODE graphs' staging (qwen4_exp PLE grows _graph_output [bs] -> [ext]),
        # and the decode graphs are already captured -- a realloc orphans their
        # captured reads of the old storage and every later decode replay computes
        # on freed memory (observed: first decode step after prefill writes NaN
        # into the GDN/PLE state pools). The decode capture pass already ensured
        # the shared pools exist before this runs.
        batch.cuda_graph_capture = True
        spec_capture_prep = getattr(model, "prepare_spec_graph_capture", None)
        if spec_capture_prep is not None:
            spec_capture_prep(ext)

        torch.cuda.synchronize(self.device)
        logger.info_rank0(f"Capturing spec-verify CUDA graph (ext={ext})")
        with ctx.forward_batch(batch):
            # Warmup run: settles triton autotune and populates the tensor_cache that
            # memoizes prepare_chunk_indices' host read of the constant cu_seqlens --
            # without it the capture itself would host-sync inside the graph region.
            model.forward()
            self._reset_moe_offload_cache()
            with torch.cuda.graph(cap.graph, pool=self._graph_pool, stream=self.stream):
                logits = model.forward()
                cap.next_tokens.copy_(torch.argmax(logits, dim=-1).to(torch.int32))
        cap.hidden_states = batch.hidden_states
        cap.stash = batch.spec_gdn_stash
        ple_stash_fn = getattr(model, "ple_spec_stash", None)
        cap.ple_stash = ple_stash_fn() if ple_stash_fn is not None else None
        cap.fla_metadata = batch.fla_metadata
        cap.attn_metadata = batch.attn_metadata
        self.spec = cap

    def capture_rechain_graph(self, model: BaseLLMModel, max_rows: int,
                             snapshots: object) -> None:
        """Capture the GDN state re-advance graph (see RechainGraph). Called by the
        scheduler after SpecLinearSnapshots is built (it owns the restore source
        tensors). Shares the spec graph's memory pool.

        One graph per committed value (1..max_rows): the GDN kernels reshape q/k/v
        to ``conv_in.shape[0]``, so the buffer shape must exactly match the committed
        count (a single varlen graph leaves stale rows in the reshape when
        committed < max_rows)."""
        from prometheus.kernel.fla.chunk import CHUNK_SIZE
        from prometheus.kernel.fla.index import prepare_chunk_indices, prepare_chunk_offsets

        ctx = get_global_ctx()
        pool = ctx.linear_state_pool
        assert pool is not None, "rechain graph requires a linear state pool"
        device = self.device

        linear_layers = [
            l for l in model.model.layers.op_list if l._is_linear
        ]
        assert linear_layers, "rechain graph requires GDN (linear) layers"
        first = linear_layers[0].linear_attn
        conv_dim = first.conv_dim
        num_v_heads = first.num_v_heads

        cap = RechainGraph(max_rows=max_rows, entries={})
        import triton

        for rows in range(1, max_rows + 1):
            cu = torch.tensor([0, rows], dtype=torch.int64, device=device)
            idx = torch.tensor([0], dtype=torch.int32, device=device)
            has_init = torch.tensor([True], dtype=torch.bool, device=device)
            slot_buf = torch.tensor([0], dtype=torch.int64, device=device)
            row_buf = torch.tensor([0], dtype=torch.int64, device=device)

            conv_in_bufs: Dict[int, torch.Tensor] = {}
            a_bufs: Dict[int, torch.Tensor] = {}
            b_bufs: Dict[int, torch.Tensor] = {}
            for layer in linear_layers:
                la = layer.linear_attn
                lid = la.layer_id
                conv_in_bufs[lid] = torch.zeros(rows, conv_dim, dtype=torch.bfloat16, device=device)
                a_bufs[lid] = torch.zeros(rows, num_v_heads, dtype=torch.bfloat16, device=device)
                b_bufs[lid] = torch.zeros(rows, num_v_heads, dtype=torch.bfloat16, device=device)

            entry = RechainGraphEntry(
                graph=torch.cuda.CUDAGraph(),
                cu_seqlens=cu, cache_idx=idx, has_init=has_init,
                slot_buf=slot_buf, row_buf=row_buf,
                conv_in_bufs=conv_in_bufs, a_bufs=a_bufs, b_bufs=b_bufs,
            )

            # Pin FLA chunk tables for this row count.
            cap.fla_chunk_indices[rows] = prepare_chunk_indices(cu, CHUNK_SIZE)
            cap.fla_chunk_offsets[rows] = prepare_chunk_offsets(cu, CHUNK_SIZE)
            bt_o = min(CHUNK_SIZE, max(16, triton.next_power_of_2(rows)))
            if bt_o != CHUNK_SIZE:
                cap.fla_chunk_indices_o[rows] = prepare_chunk_indices(cu, bt_o)

            def _restore_graph(e=entry):
                pool.conv_states.index_copy_(1, e.slot_buf,
                    snapshots.conv_snap.index_select(1, e.row_buf))
                pool.recurrent_states.index_copy_(1, e.slot_buf,
                    snapshots.rec_snap.index_select(1, e.row_buf))

            def _rechain_graph(e=entry):
                for layer in linear_layers:
                    la = layer.linear_attn
                    la.spec_readvance(
                        e.conv_in_bufs[la.layer_id], e.a_bufs[la.layer_id], e.b_bufs[la.layer_id],
                        0, device, graph_consts=(e.cu_seqlens, e.cache_idx, e.has_init),
                    )

            torch.cuda.synchronize(device)
            logger.info_rank0(f"Capturing rechain CUDA graph (rows={rows})")
            _restore_graph()
            _rechain_graph()
            with torch.cuda.graph(entry.graph, pool=self._graph_pool, stream=self.stream):
                _restore_graph()
                _rechain_graph()

            cap.entries[rows] = entry

        self.rechain = cap

    def replay_rechain(self, stash: dict, start: int, committed: int,
                       slot: int, row: int) -> None:
        """Fill persistent rechain buffers and replay the captured graph.

        ``stash`` is the verify forward's per-layer GDN input stash (conv_in/a/b).
        ``start`` is the row offset into the stashed span; ``committed`` is the
        number of rows to re-advance; ``slot`` is the live GDN state slot;
        ``row`` is the snapshot buffer row (req.table_idx)."""
        rc = self.rechain
        assert rc is not None
        assert 1 <= committed <= rc.max_rows
        e = rc.entries[committed]
        # Fill indices (cu_seqlens is already [0, committed] at capture; no update needed)
        e.cache_idx[0].fill_(slot)
        e.slot_buf[0].fill_(slot)
        e.row_buf[0].fill_(row)
        # Copy stash slices into per-layer persistent input buffers (exact [committed] shape)
        for lid, conv_in_buf in e.conv_in_bufs.items():
            src_conv_in, src_a, src_b = stash[lid]
            conv_in_buf.copy_(src_conv_in[start : start + committed])
            e.a_bufs[lid].copy_(src_a[start : start + committed])
            e.b_bufs[lid].copy_(src_b[start : start + committed])
        e.graph.replay()

    def can_use_spec_graph(self, batch: Batch) -> bool:
        """bs=1 spec-verify batch with the full 1+k draft seated (the captured shape)."""
        s = self.spec
        return (
            s is not None
            and batch.spec_verify
            and batch.size == 1
            and batch.reqs[0].extend_len == s.ext
        )

    def replay_spec(self, batch: Batch) -> torch.Tensor:
        """Fill the persistent spec buffers from the batch, replay the captured verify
        forward + argmax, and re-point the batch at the graph-pool hidden states / GDN
        stash so the scheduler drain consumes them exactly like the eager forward's."""
        s = self.spec
        assert s is not None and self.can_use_spec_graph(batch)
        req = batch.reqs[0]
        s.input_ids.copy_(batch.input_ids)
        s.positions.copy_(batch.positions)
        s.out_loc.copy_(batch.out_loc)
        page_table = get_global_ctx().page_table
        n = req.device_len
        s.kv_indices[:n].copy_(page_table[req.table_idx, :n])
        s.kv_indptr[1].fill_(n)
        s.prefix_lens[0].fill_(req.cached_len)
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        s.fla_cache_idx[0].fill_(slot)
        # qwen4_exp PLE: stage the verify rows' ngram embeddings from the current
        # hist (host gather) before the replay reads the staged buffer.
        spec_replay_prep = getattr(self.model, "prepare_spec_graph_replay", None)
        if spec_replay_prep is not None:
            spec_replay_prep(batch)
        s.graph.replay()
        if s.hidden_states is not None:
            batch.hidden_states = s.hidden_states
        if s.stash is not None:
            batch.spec_gdn_stash = s.stash
        if s.ple_stash is not None:
            batch.spec_ple_stash = s.ple_stash
        return s.next_tokens

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.model.prepare_cuda_graph_replay(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        self.spec = None
        self.rechain = None
        gc.collect()
