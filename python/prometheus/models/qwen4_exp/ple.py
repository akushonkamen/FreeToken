from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F
from prometheus.core import get_global_ctx
from prometheus.layers import BaseOP, LinearReplicated

# splitmix64 constants of the HF hashed n-gram embedding (modeling_qwen4_exp).
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class Qwen4GroupRMSNorm(BaseOP):
    """Gemma-style (1+weight) RMSNorm whose mean is taken per ``group_size`` slice
    (HF ``Qwen4ExpTextRMSNorm(group_size=hidden_size)``). The weight is kept RAW --
    the +1 is applied in float32 here, NOT baked at load (bf16-baked +1 on this
    checkpoint's small initializer range is visibly lossy)."""

    def __init__(self, dim: int, group_size: int, eps: float):
        self.weight = torch.empty(dim)
        self._group = group_size
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float().unflatten(-1, (-1, self._group))
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self._eps)
        xf = xf.flatten(-2)
        return (xf * (1.0 + self.weight.float())).to(x.dtype)


class _PleConv1d(BaseOP):
    """Dilated depthwise conv weight ``[channels, 1, kernel]`` (key ``conv1d.weight``)."""

    def __init__(self, channels: int, kernel: int, dilation: int):
        self.weight = torch.empty(channels, 1, kernel)
        self._dilation = dilation


class Qwen4PleEmbedding(BaseOP):
    """Host-resident hashed n-gram embedding table (HF ``Qwen4ExpTextNGramEmbedding``).

    The table (~102 GB bf16 for Flash-Next) never flows through the state dict:
    ``iter_weights`` skips the ``ngram_embedding.shard_*`` keys and ``prepare()``
    streams the shards into one pinned host tensor. Hash inputs are the
    segment-shifted token views (eos-filled); ids are computed on GPU with the
    int64 splitmix multipliers, then gathered on host (16 rows x 160 dims per
    token -- a few KB per decode step).
    """

    def __init__(self, args, head_dim_per_ngram: int, ple_layer_index: int = 0):
        ngram_heads = (args.ngram_size - 1) * args.heads_per_ngram
        # Hash constants are pure functions of the config (HF nn.Buffer, same values
        # whether the checkpoint ships them or not): computed here so serving never
        # depends on their presence in the state dict.
        seed = getattr(args, "ple_seed", 1234)
        max_long = (1 << 63) - 1
        # NB: the multiplier bound uses the UNIGRAM vocab (HF passes config.vocab_size),
        # not ngram_vocab_size_base.
        multiplier_max = max_long // max(args.unigram_vocab, 1)
        half_bound = max(1, multiplier_max // 2)
        base_seed = seed + _PRIME_1 * ple_layer_index
        multipliers = []
        for index in range(args.ngram_size):
            value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
            multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
        # _-prefixed: kept out of the BaseOP state dict. The checkpoint ships only
        # layer_multipliers (persistent nn.Buffer in HF); the head sizes/offsets are
        # never saved -- both are pure functions of the config, computed here.
        self._layer_multipliers = torch.tensor(multipliers, dtype=torch.int64, device="cpu")
        head_vocab_sizes = []
        head_offsets = []
        total = 0
        for head_idx in range(ngram_heads):
            global_head_idx = ple_layer_index * ngram_heads + head_idx
            size = _find_nth_prime_after(args.ngram_vocab_size_base - 1, global_head_idx + 1)
            head_vocab_sizes.append(size)
            head_offsets.append(total)
            total += size
        self._ngram_heads_vocab_sizes = torch.tensor(head_vocab_sizes, dtype=torch.int64, device="cpu")
        self._ngram_heads_offsets = torch.tensor(head_offsets, dtype=torch.int64, device="cpu")
        self._ngram_heads = ngram_heads
        self._hpg = args.heads_per_ngram  # heads per ngram order (bigram, trigram)
        self._head_dim = head_dim_per_ngram
        self._model_path = args.model_path
        self._divisor = args.ngram_divisor
        self._table: torch.Tensor | None = None  # pinned host bf16 [padded, head_dim]
        self._gpu: dict | None = None  # {mult, vocab, offsets} device-side consts
        # CUDA-graph staging: the captured decode reads the host-gathered ngram rows
        # from this stable device buffer; it is filled eagerly before each replay.
        self._graph_output: torch.Tensor | None = None
        # Spec-verify graph staging: a SEPARATE buffer, because the decode graphs
        # capture reads of _graph_output's storage and a later realloc (the spec
        # graph is captured after them, with ext = 1+k rows) would orphan those
        # captured pointers.
        self._spec_graph_output: torch.Tensor | None = None

    def padded_vocab(self) -> int:
        # HF: offsets are cumulative head start rows, so total = last offset + last
        # head's vocab (verified: 320,001,446 -> padded 320,001,536 = 128 x 2,500,012).
        total = int(self._ngram_heads_offsets[-1]) + int(self._ngram_heads_vocab_sizes[-1])
        return math.ceil(total / self._divisor) * self._divisor

    def prepare(self, device: torch.device) -> None:
        """Stream the shard tensors into one pinned host table (post-load hook)."""
        if self._table is not None:
            return
        import json
        import os

        path = self._model_path
        assert path, "Qwen4PleEmbedding needs model_path (q4_args.model_path)"
        index = json.load(open(os.path.join(path, "model.safetensors.index.json")))
        shard_re = re.compile(
            r"^(.*)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
        )
        shards: dict[int, tuple[str, str]] = {}
        scale_file = scale_key = None
        scale_re = re.compile(
            r"^(.*)\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
        )
        for key, fname in index["weight_map"].items():
            m = shard_re.match(key)
            if m:
                shards[int(m.group(2))] = (os.path.join(path, fname), key)
                continue
            ms = scale_re.match(key)
            if ms:  # FP8-PLE table: per-table scalar scale alongside the shards
                scale_file, scale_key = os.path.join(path, fname), key
        assert shards, f"no ngram_embedding shards under {path}"
        import safetensors

        padded = self.padded_vocab()
        # Allocate the pinned destination first and stream each shard straight into
        # its slice -- holding all 128 parts at once would double the ~102 GB peak.
        table = torch.empty(padded, self._head_dim, dtype=torch.bfloat16, pin_memory=True)
        fp8_scale = None
        off = 0
        for idx in sorted(shards):
            file, key = shards[idx]
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                t = f.get_tensor(key)
            if t.dtype != torch.bfloat16:  # FP8-PLE shard (float8_e4m3fn + scalar scale)
                if scale_key is None:
                    raise ValueError(
                        f"FP8 ngram shard {idx} but no ngram_embedding.weight_scale in index"
                    )
                if fp8_scale is None:
                    with safetensors.safe_open(scale_file, framework="pt", device="cpu") as sf:
                        fp8_scale = sf.get_tensor(scale_key).to(torch.bfloat16)
                t = t.to(torch.bfloat16) * fp8_scale
            assert t.shape[1] == self._head_dim, f"shard shape {t.shape} != {-1, self._head_dim}"
            end = off + t.shape[0]
            assert end <= padded, f"shard {idx} overruns the padded table ({end} > {padded})"
            table[off:end].copy_(t)
            off = end
            del t
        assert off == padded, f"ngram shards hold {off} rows, buffers say {padded}"
        self._table = table
        self._gpu = {
            "mult": self._layer_multipliers.to(device),
            "vocab": self._ngram_heads_vocab_sizes.to(device),
            "offsets": self._ngram_heads_offsets.to(device),
        }

    def _head_ids(self, s0: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """Hashed n-gram row ids [N, ngram_heads] (int64, device-side)."""
        g = self._gpu
        mixed2 = s0 * g["mult"][0]
        mixed2 = mixed2 ^ (s1 * g["mult"][1])
        mixed3 = mixed2 ^ (s2 * g["mult"][2])
        hpg = self._hpg
        # heads [0, hpg): bigram hash; heads [hpg, 2*hpg): trigram hash (HF order).
        return torch.cat(
            [
                mixed2.unsqueeze(-1) % g["vocab"][:hpg] + g["offsets"][:hpg],
                mixed3.unsqueeze(-1) % g["vocab"][hpg:] + g["offsets"][hpg:],
            ],
            dim=-1,
        )

    def forward(self, s0: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """s*: the ngram_size shifted token views [N] (int64, eos-filled, GPU).
        Returns the concatenated head embeddings [N, ngram_heads * head_dim] on GPU."""
        g = self._gpu
        assert g is not None, "Qwen4PleEmbedding.prepare() not called"
        ids = self._head_ids(s0, s1, s2)  # [N, ngram_heads] int64
        cpu_ids = ids.reshape(-1).cpu()
        emb = self._table.index_select(0, cpu_ids).view(ids.shape[0], -1)
        return emb.to(s0.device, non_blocking=False)

    def prepare_cuda_graph_capture(self, token_count: int, device: torch.device) -> None:
        if self._graph_output is None or self._graph_output.shape[0] < token_count:
            self._graph_output = torch.zeros(
                token_count, self._ngram_heads * self._head_dim,
                device=device, dtype=torch.bfloat16,
            )

    def stage_graph_output(self, s0: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> None:
        """Eagerly run the host-side ngram gather into the stable graph buffer."""
        assert self._graph_output is not None, "stage before prepare_cuda_graph_capture"
        ids = self._head_ids(s0, s1, s2)  # [N, ngram_heads] int64
        cpu_ids = ids.reshape(-1).cpu()
        emb = self._table.index_select(0, cpu_ids).view(ids.shape[0], -1)
        self._graph_output[: emb.shape[0]].copy_(emb)

    def prepare_spec_graph_capture(self, token_count: int, device: torch.device) -> None:
        """Size the spec-verify graph's ngram staging buffer before capture."""
        if self._spec_graph_output is None or self._spec_graph_output.shape[0] < token_count:
            self._spec_graph_output = torch.zeros(
                token_count, self._ngram_heads * self._head_dim,
                device=device, dtype=torch.bfloat16,
            )

    def stage_spec_graph_output(self, s0: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> None:
        """Eagerly run the host-side ngram gather into the spec graph's buffer."""
        assert self._spec_graph_output is not None, "stage before prepare_spec_graph_capture"
        ids = self._head_ids(s0, s1, s2)  # [N, ngram_heads] int64
        cpu_ids = ids.reshape(-1).cpu()
        emb = self._table.index_select(0, cpu_ids).view(ids.shape[0], -1)
        self._spec_graph_output[: emb.shape[0]].copy_(emb)


class Qwen4PleLayer(BaseOP):
    """Per-Layer Embedding (HF ``Qwen4ExpTextPLELayer``) for the serving engine.

    Stateful like the GDN: per-request token-history (last 2 raw ids + the absolute
    position of the last eos) and the 9-step dilated-conv input window, keyed by the
    SAME fla cache_indices the GatedDeltaNet layers use. The state math replicates
    HF's ``_shift_right_ignore_eos``: a shifted view is valid iff the query token's
    distance from its segment (last-eos) start is >= shift, which for ngram_size=3
    needs only the raw last-2 ids plus that distance (a same-segment predecessor
    always exists when the distance condition holds).
    """

    def __init__(self, args, config, layer_id: int, ple_layer_index: int = 0):
        hc = config.hidden_size * args.hc_count
        self.ple_embedding = Qwen4PleEmbedding(
            args,
            args.ple_embed_dim // ((args.ngram_size - 1) * args.heads_per_ngram),
            ple_layer_index,
        )
        self.key_proj = LinearReplicated(args.ple_embed_dim, hc, has_bias=False)
        self.value_proj = LinearReplicated(args.ple_embed_dim, config.hidden_size, has_bias=False)
        self.norm_key = Qwen4GroupRMSNorm(hc, config.hidden_size, config.rms_norm_eps)
        self.norm_query = Qwen4GroupRMSNorm(hc, config.hidden_size, config.rms_norm_eps)
        self.norm_conv = Qwen4GroupRMSNorm(hc, config.hidden_size, config.rms_norm_eps)
        self.conv1d = _PleConv1d(hc, args.ple_conv_kernel_size, args.ngram_size)
        self._layer_id = layer_id
        self._hc = args.hc_count
        self._hidden = config.hidden_size
        self._hc_dim = hc
        self._eos = args.eos_token_id
        self._conv_state_len = (args.ple_conv_kernel_size - 1) * args.ngram_size
        # Runtime state (underscore: invisible to state_dict). Slots are indexed by
        # the fla cache_indices, grown geometrically, re-initialized per fresh request.
        self._hist = None        # [S, 2] int64: (id_{p-2}, id_{p-1})
        self._last_eos = None    # [S] int64: absolute position of last eos, -1 = none
        self._conv_state = None  # [S, hc, conv_state_len] bf16, conv INPUT window
        # Spec-verify support: the verify forward's conv-input stash for the graph
        # path (the eager path stashes into batch.spec_ple_stash instead), and the
        # pre-forward state snapshot the drain-side partial-accept rollback uses.
        self._spec_conv_buf: torch.Tensor | None = None
        self._spec_snap: dict = {}

    def prepare(self, device: torch.device) -> None:
        self.ple_embedding.prepare(device)

    def prepare_cuda_graph_capture(self, token_count: int) -> None:
        """Size the graph-owned state before decode capture: the embedding staging
        buffer plus the hist/eos/conv pools. Replay writes real slot ids into the
        graph's static index buffer, so the pools must cover every live slot."""
        device = self.conv1d.weight.device
        self.ple_embedding.prepare_cuda_graph_capture(token_count, device)
        pool = get_global_ctx().linear_state_pool
        self._ensure_slots(pool.num_slots if pool is not None else token_count, device)

    def prepare_cuda_graph_replay(self, batch) -> None:
        """Host half of the PLE decode step, run eagerly before graph replay: the
        ngram lookup (device hash -> cpu gather -> staging buffer) and the hist /
        last-eos window slide. The captured graph reads only the staged buffer and
        the conv state pool."""
        slots = batch.linear_table_idx
        assert slots is not None
        slots = slots.to(torch.int64)
        ids = batch.input_ids.reshape(-1).to(torch.int64)
        pos = batch.positions.reshape(-1).to(torch.int64)
        hist = self._hist[slots]
        le = self._last_eos[slots]
        sp = pos - le - 1
        s0 = ids
        s1 = torch.where(sp >= 1, hist[:, 1], ids.new_full((), self._eos))
        s2 = torch.where(sp >= 2, hist[:, 0], ids.new_full((), self._eos))
        self.ple_embedding.stage_graph_output(s0, s1, s2)
        self._hist[slots, 0] = hist[:, 1]
        self._hist[slots, 1] = s0
        self._last_eos[slots] = torch.where(s0 == self._eos, pos, le)

    def prepare_spec_graph_capture(self, token_count: int) -> None:
        """Size the spec-verify graph's PLE buffers before capture: the ngram
        staging buffer and the conv-input stash (allocated here so the warmup run
        and the captured region never allocate)."""
        device = self.conv1d.weight.device
        self.ple_embedding.prepare_spec_graph_capture(token_count, device)
        if self._spec_conv_buf is None or self._spec_conv_buf.shape[0] < token_count:
            self._spec_conv_buf = torch.zeros(
                token_count, self._hc_dim, dtype=torch.bfloat16, device=device
            )

    def prepare_spec_graph_replay(self, batch) -> None:
        """Host half of the PLE spec-verify step, run eagerly before the spec graph
        replays: stage the 1+k verify rows' ngram embeddings from the CURRENT hist
        (exact -- the previous drain corrected it after the last commit). Unlike
        decode replay prep this does NOT slide hist/last-eos: a partial accept
        would leave them past the committed prefix, so the scheduler drain owns
        the post-commit correction (spec_commit, from the committed ids)."""
        ids = batch.input_ids.reshape(-1).to(torch.int64)
        L = ids.shape[0]
        # fla_metadata is still None here: the FLA op builds it lazily inside the
        # (captured) forward, so derive the slot the same way replay_spec fills
        # s.fla_cache_idx -- the spec graph is bs=1 (can_use_spec_graph).
        slots = torch.tensor(
            [
                r.linear_slot_idx if r.linear_slot_idx is not None else r.table_idx
                for r in batch.reqs
            ],
            dtype=torch.int64,
            device=ids.device,
        )
        hist = self._hist[slots]   # [1, 2]
        win = torch.cat([hist.reshape(-1), ids])  # [2 + L]
        # Shifted views with HF's exact windowed segment logic (see the eager
        # prefill branch): s_k[i] = win[i + 2 - k], valid iff sp[i] >= k where sp
        # comes from window-coordinate eos positions.
        wlocal = torch.arange(L + 2, device=ids.device, dtype=torch.int64)
        weos = torch.where(win == self._eos, wlocal, torch.full_like(wlocal, -1))
        prev_excl = torch.cat(
            [weos.new_full((1,), -1), torch.cummax(weos, dim=0).values[:-1]]
        )
        sp = (wlocal - prev_excl - 1)[2:]
        eos = ids.new_full((), self._eos)
        s1 = torch.where(sp >= 1, win[1 : 1 + L], eos)
        s2 = torch.where(sp >= 2, win[0:L], eos)
        self.ple_embedding.stage_spec_graph_output(ids, s1, s2)

    def spec_snapshot(self, slots) -> None:
        """Pre-forward snapshot of the spec batch's PLE state rows. The verify
        forward advances hist/last-eos/conv window by the full 1+k draft rows
        (eager prefill branch, or the captured writeback); a partial accept must
        re-land them at m+committed, which needs the pre-forward values. Device-side
        clones only -- no host sync on the scheduling path."""
        self._spec_snap = {
            int(s): (
                self._hist[s].clone(),        # [2]
                self._last_eos[s].clone(),    # []
                self._conv_state[s].clone(),  # [hc, conv_state_len]
            )
            for s in slots
        }

    def spec_commit(self, slot: int, m: int, committed_ids) -> None:
        """Exact post-commit hist/last-eos correction: leave the state exactly the
        equivalent ``committed`` greedy decode steps would have. Called by the
        scheduler drain for every non-finished spec commit (both graph and eager
        paths -- the graph path never slides, the eager path over-slides)."""
        snap = self._spec_snap.get(slot)
        if snap is None:
            return
        h, le, _conv = snap
        ids_t = torch.tensor(committed_ids, dtype=torch.int64, device=h.device)
        # hist = last 2 of (snapshot hist ++ committed ids)
        w = torch.cat([h, ids_t])
        self._hist[slot, 0], self._hist[slot, 1] = w[-2], w[-1]
        # last eos = max(snapshot's, last eos among the committed ids at abs pos)
        idx = torch.arange(ids_t.shape[0], device=h.device, dtype=torch.int64)
        epos = torch.where(ids_t == self._eos, m + idx, torch.full_like(idx, -1))
        self._last_eos[slot] = torch.maximum(le, epos.max())

    def spec_readvance(self, stash_conv_in: torch.Tensor, start: int, committed: int,
                       slot: int) -> None:
        """Re-land the conv-input window at m+committed after a partial accept:
        (snapshot window ++ the committed rows' conv inputs)[-conv_state_len:]."""
        snap = self._spec_snap.get(slot)
        assert snap is not None, "PLE spec readvance without a pre-forward snapshot"
        rows = stash_conv_in[start : start + committed]  # [committed, hc]
        new = torch.cat(
            [snap[2], rows.transpose(0, 1)], dim=-1
        )[..., -self._conv_state_len :]
        self._conv_state[slot] = new

    def _ensure_slots(self, n: int, device) -> None:
        if self._hist is not None and self._hist.shape[0] >= n:
            return
        new_n = max(n, 64)
        if self._hist is not None:
            new_n = max(new_n, self._hist.shape[0] * 2)
            old_h, old_le, old_c = self._hist, self._last_eos, self._conv_state
        else:
            old_h = old_le = old_c = None
        hist = torch.full((new_n, 2), self._eos, dtype=torch.int64, device=device)
        last_eos = torch.full((new_n,), -1, dtype=torch.int64, device=device)
        conv = torch.zeros(
            (new_n, self._hc_dim, self._conv_state_len), dtype=torch.bfloat16, device=device
        )
        if old_h is not None:
            hist[: old_h.shape[0]] = old_h
            last_eos[: old_le.shape[0]] = old_le
            conv[: old_c.shape[0]] = old_c
        self._hist, self._last_eos, self._conv_state = hist, last_eos, conv

    def _init_slots(self, slots: torch.Tensor) -> None:
        if slots.numel() == 0:
            return
        slots = slots.to(torch.int64)
        self._hist[slots] = self._eos
        self._last_eos[slots] = -1
        self._conv_state[slots] = 0

    def _fresh_slots(self, fla, slots: torch.Tensor) -> torch.Tensor:
        if fla.fresh_state_indices is not None:
            return fla.fresh_state_indices.to(torch.int64)
        hi = fla.has_initial_state
        if hi is None:
            return slots
        hi = hi.cpu() if hi.device.type != "cpu" else hi
        return slots[~hi.to(torch.bool)]

    def _ngram_embeddings(self, ids: torch.Tensor, fla, batch) -> torch.Tensor:
        """ids: flat [T] int64 for this forward. Returns [T, ple_embed_dim]."""
        emb = self.ple_embedding
        if batch.cuda_graph_capture and batch.spec_verify:
            # Spec-verify graph: the 1+k rows' ngram gather was staged eagerly by
            # prepare_spec_graph_replay (which leaves hist alone -- the drain owns
            # the post-commit correction). The captured region reads only the
            # staged buffer; everything below host-syncs on cu_seqlens.
            out = emb._spec_graph_output
            assert out is not None, "spec graph capture before PLE spec staging buffer"
            return out[: ids.shape[0]]
        if batch.is_decode:
            if batch.cuda_graph_capture:
                # Graph decode: the ngram rows were staged eagerly (replay prep
                # also slid the hist window); capture only reads the buffer.
                out = emb._graph_output
                assert out is not None
                return out[: ids.shape[0]]
            slots = fla.cache_indices.to(torch.int64)
            self._ensure_slots(int(slots.max()) + 1, ids.device)
            hist = self._hist[slots]  # [B, 2]
            le = self._last_eos[slots]
            pos = batch.positions.reshape(-1).to(torch.int64)
            sp = pos - le - 1  # distance from segment start
            s0 = ids
            s1 = torch.where(sp >= 1, hist[:, 1], ids.new_full((), self._eos))
            s2 = torch.where(sp >= 2, hist[:, 0], ids.new_full((), self._eos))
            out = emb.forward(s0, s1, s2)
            # The window slides over (old hist, new ids).
            self._hist[slots, 0] = hist[:, 1]
            self._hist[slots, 1] = s0
            self._last_eos[slots] = torch.where(s0 == self._eos, pos, le)
            return out
        # ---- prefill: per-request slices (chunked prefill keeps R small) ----
        cu = fla.cu_seqlens
        if cu.device.type != "cpu":
            cu = cu.cpu()
        cu_list = cu.tolist()
        slots = fla.cache_indices
        slots = slots.cpu() if slots.device.type != "cpu" else slots
        slots = slots.to(torch.int64)
        self._ensure_slots(int(slots.max()) + 1, ids.device)
        self._init_slots(self._fresh_slots(fla, slots).to(ids.device))
        embs = []
        positions = batch.positions.reshape(-1).to(torch.int64)
        for r in range(len(cu_list) - 1):
            beg, end = int(cu_list[r]), int(cu_list[r + 1])
            if beg == end:
                continue
            slot = int(slots[r])
            L = end - beg
            ids_r = ids[beg:end]
            pos0 = int(positions[beg])
            # History window [id_{p-2}, id_{p-1}] ++ chunk ids; the shift-k view of
            # row i is win[i + 2 - k] (HF: shifted_k over the full history, last L
            # rows), i.e. s1 = win[1:L+1] and s2 = win[0:L]. (s2 used to read
            # win[2:2+L] -- each row's OWN token, breaking both HF parity and the
            # decode branch's semantics; verified against the real HF method.)
            win = torch.cat([self._hist[slot], ids_r])  # [L + 2]
            t2, t1 = self._hist[slot, 0], self._hist[slot, 1]
            # Segment distance, exactly HF's windowed _shift_right_ignore_eos: eos
            # positions live in WINDOW coordinates (0..L+1) so the -1 "no eos"
            # sentinel cannot collide with a real position, and the initial
            # previous_eos is -1 (a fresh request's hist is eos-initialized, so an
            # eos inside the window is seen at its own index). The old chunk-local
            # cummax lost both properties and eos-filled row 1's trigram whenever
            # no eos sat inside the chunk.
            wlocal = torch.arange(L + 2, device=ids.device, dtype=torch.int64)
            weos = torch.where(win == self._eos, wlocal, torch.full_like(wlocal, -1))
            prev_excl = torch.cat(
                [weos.new_full((1,), -1), torch.cummax(weos, dim=0).values[:-1]]
            )
            sp = wlocal - prev_excl - 1  # row i's distance = sp[i + 2]
            sp = sp[2:]
            s0 = ids_r
            s1 = torch.where(sp >= 1, win[1 : 1 + L], ids_r.new_full((), self._eos))
            s2 = torch.where(sp >= 2, win[0:L], ids_r.new_full((), self._eos))
            embs.append(emb.forward(s0, s1, s2))
            # Slide the history window: last 2 of (old hist ++ chunk ids).
            self._hist[slot, 0] = ids_r[-2] if L >= 2 else t1
            self._hist[slot, 1] = ids_r[-1]
            new_le = weos[2:]
            new_le = new_le[new_le >= 0]
            if new_le.numel() > 0:
                self._last_eos[slot] = pos0 + int(new_le.max().item())
        return torch.cat(embs, dim=0)

    def _conv_forward(self, conv_in: torch.Tensor, fla, batch) -> torch.Tensor:
        """Dilated depthwise conv + silu over the per-request state-padded stream.
        ``conv_in`` is [T, hc_dim]; returns [T, hc_dim]. Updates the conv state."""
        w = self.conv1d.weight  # [C, 1, K]
        dilation = self.conv1d._dilation
        if batch.is_decode:
            slots = fla.cache_indices.to(torch.int64)
            state = self._conv_state[slots]  # [B, C, 9]
            x = torch.cat([state, conv_in.unsqueeze(-1)], dim=-1)  # [B, C, 10]
            out = F.silu(F.conv1d(x, w, groups=x.shape[1], dilation=dilation))
            self._conv_state[slots] = x[..., 1:]
            return out.squeeze(-1)
        if batch.cuda_graph_capture and batch.spec_verify:
            # Spec-verify graph: static single-chunk shape (bs=1, L=1+k), so the
            # host cu_seqlens loop below is both illegal under capture (D2H sync)
            # and unnecessary -- everything here is a device op on the one chunk.
            # The captured state writeback advances over ALL 1+k rows; the conv
            # inputs are stashed so the drain can re-land the window on a partial
            # accept (spec_readvance).
            slots = fla.cache_indices.to(torch.int64)
            state = self._conv_state[slots]  # [1, C, W]
            x = torch.cat([state, conv_in.unsqueeze(0).transpose(1, 2)], dim=-1)
            out = F.silu(F.conv1d(x, w, groups=x.shape[1], dilation=dilation))
            self._conv_state[slots] = x[..., -self._conv_state_len :]
            buf = self._spec_conv_buf
            assert (
                buf is not None and buf.shape[0] >= conv_in.shape[0]
            ), "spec graph conv stash missing (prepare_spec_graph_capture)"
            buf[: conv_in.shape[0]].copy_(conv_in)
            return out.squeeze(0).transpose(0, 1)
        if batch.spec_verify:
            # Eager spec verify: stash this layer's full-batch conv inputs for the
            # drain-side partial-accept re-advance (mirror of the GDN spec stash).
            if batch.spec_ple_stash is None:
                batch.spec_ple_stash = {}
            batch.spec_ple_stash[self._layer_id] = conv_in
        cu = fla.cu_seqlens
        if cu.device.type != "cpu":
            cu = cu.cpu()
        cu_list = cu.tolist()
        slots = fla.cache_indices
        slots = slots.cpu() if slots.device.type != "cpu" else slots
        slots = slots.to(torch.int64)
        outs = []
        for r in range(len(cu_list) - 1):
            beg, end = int(cu_list[r]), int(cu_list[r + 1])
            if beg == end:
                continue
            slot = int(slots[r])
            state = self._conv_state[slot]  # [C, 9]
            x = torch.cat([state, conv_in[beg:end].transpose(0, 1)], dim=-1)  # [C, 9+L]
            out = F.silu(
                F.conv1d(x.unsqueeze(0), w, groups=x.shape[0], dilation=dilation)
            )
            outs.append(out.squeeze(0).transpose(0, 1))  # [L, C]
            self._conv_state[slot] = x[..., -self._conv_state_len :]
        return torch.cat(outs, dim=0)

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """hidden: [T, hc_count*hidden] hyper-connection stream; input_ids: flat [T].
        Returns the PLE addition [T, hc_count*hidden]."""
        ctx = get_global_ctx()
        batch = ctx.batch
        fla = batch.fla_metadata
        if fla is None:
            from prometheus.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden.device)
            batch.fla_metadata = fla
        ids = input_ids.reshape(-1).to(torch.int64)
        emb = self._ngram_embeddings(ids, fla, batch)  # [T, ple_embed_dim]
        key = self.norm_key.forward(self.key_proj.forward(emb))
        key = key.unflatten(-1, (self._hc, self._hidden))
        value = self.value_proj.forward(emb)
        query = self.norm_query.forward(hidden).unflatten(-1, (self._hc, self._hidden))
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self._hidden)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)  # [T, hc, hidden]
        gated_flat = gated.flatten(-2)
        conv_in = self.norm_conv.forward(gated_flat)
        conv_out = self._conv_forward(conv_in, fla, batch)
        return gated_flat + conv_out


__all__ = ["Qwen4PleLayer", "Qwen4PleEmbedding", "Qwen4GroupRMSNorm"]
