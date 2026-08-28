from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from prometheus.core import get_global_ctx
from prometheus.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from prometheus.models.blocks import BaseLLMModel
from prometheus.utils import nvtx_annotate

from .config import Qwen4ExpArgs
from .ple import Qwen4GroupRMSNorm, Qwen4PleLayer
from ..qwen3_5_moe.attention import Qwen3_5Attention
from ..qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from ..qwen3_5_moe.moe import Qwen3_5MoE

if TYPE_CHECKING:
    from prometheus.core import Batch
    from prometheus.models.config import ModelConfig


class Qwen4GatedResidual(BaseOP):
    """Hyper-connection mixer (HF ``Qwen4ExpTextGatedResidual``).

    Norms the ``hc_count``-stream residual, computes a low-rank sigmoid input-mix
    weight per stream, averages the weighted streams into the block input, and (with
    ``use_combine``) a per-stream injection weight for adding the block output back:

        mixed = mean_s(sigmoid(up(silu(down(normed)/hc))) * normed)_s
        inj   = 2 * sigmoid(block_inject(normed) / hc)
        out   = hyper + block(mixed).unsqueeze(stream) * inj

    The final model-level mixer (``use_combine=False``) only mixes the streams down
    to ``[hidden]`` for the lm_head.
    """

    def __init__(self, config: ModelConfig, q4: Qwen4ExpArgs, use_combine: bool = True):
        hc = config.hidden_size * q4.hc_count
        self.hc_norm = Qwen4GroupRMSNorm(hc, config.hidden_size, config.rms_norm_eps)
        self.input_mix_weight_down = LinearReplicated(hc, q4.hc_lowrank, has_bias=False)
        self.input_mix_weight_up = LinearReplicated(q4.hc_lowrank, hc, has_bias=False)
        self.block_inject_weight = (
            LinearReplicated(hc, q4.hc_count, has_bias=False) if use_combine else None
        )
        self._hc = q4.hc_count
        self._hidden = config.hidden_size

    def _mixed(self, hyper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (block input [T, hidden], normed hyper [T, hc*hidden])."""
        n = self.hc_norm.forward(hyper)
        w = torch.nn.functional.silu(self.input_mix_weight_down.forward(n) / self._hc)
        w = torch.sigmoid(self.input_mix_weight_up.forward(w))
        T = hyper.shape[0]
        mixed = (w.view(T, self._hc, self._hidden) * n.view(T, self._hc, self._hidden)).mean(
            dim=1
        )
        return mixed, n

    def forward(self, hyper: torch.Tensor) -> torch.Tensor:
        """Model-level combine mixer (use_combine=False): stream mix only."""
        mixed, _ = self._mixed(hyper)
        return mixed

    def forward_with_inject(self, hyper: torch.Tensor):
        """Per-block mixer: returns (block input, hyper passthrough, injection weights)."""
        mixed, n = self._mixed(hyper)
        inj = 2 * torch.sigmoid(self.block_inject_weight.forward(n) / self._hc)  # [T, hc]
        return mixed, hyper, inj


class Qwen4ExpDecoderLayer(BaseOP):
    """One qwen4_exp block over the ``hc_count``-wide hyper-connection stream.

    Optional PLE addition (hashed n-gram features) first, then two gated-residual
    sublayers (mixer: GatedDeltaNet or gated full attention; MLP: MoE + shared
    expert). No per-layer input/post layernorms -- the residual normalization lives
    inside each GatedResidual's hc_norm.
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        q4: Qwen4ExpArgs = config.q4_args
        if layer_id in q4.ple_layer_idxs:
            self.ple = Qwen4PleLayer(
                q4, config, layer_id,
                ple_layer_index=q4.ple_layer_idxs.index(layer_id),
            )
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                gate_activation=q4.output_gate_type,
            )
        else:
            # Same gated attention as qwen3_5 (q_proj 2x half for the output gate,
            # Gemma q/k norms, partial NeoX rope). The QSA indexer weights are NOT
            # loaded (milestone: dense attention; identical to QSA for kv_len <= 2052).
            self.self_attn = Qwen3_5Attention(config, layer_id)
        self.attn_hyper_connection = Qwen4GatedResidual(config, q4)
        self.mlp_hyper_connection = Qwen4GatedResidual(config, q4)
        self.mlp = Qwen3_5MoE(config, layer_id)
        self._hidden = config.hidden_size

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "ple"):
            hidden = hidden + self.ple.forward(hidden, input_ids)
        mixed, hyper, inj = self.attn_hyper_connection.forward_with_inject(hidden)
        out = (
            self.linear_attn.forward(mixed)
            if self._is_linear
            else self.self_attn.forward(mixed)
        )
        hidden = hyper + (out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        mixed, hyper, inj = self.mlp_hyper_connection.forward_with_inject(hidden)
        out = self.mlp.forward(mixed)
        hidden = hyper + (out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        return hidden


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = Qwen4GatedResidual(
            config, config.q4_args, use_combine=False
        )
        self._hc = config.q4_args.hc_count

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the decoder stack and return the PRE-MIXER hyper stream
        [T, hc_count*hidden] (the MTP drafter's input; the mixer folds it down)."""
        x = self.embed_tokens.forward(input_ids)
        x = x.repeat(1, self._hc)  # every stream starts from the same embedding
        for layer in self.layers.op_list:
            x = layer.forward(x, input_ids)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.hyper_connection_mixer.forward(self.forward_hidden(input_ids))

    def prepare_cuda_graph_capture(self, token_count: int) -> None:
        for layer in self.layers.op_list:
            if hasattr(layer, "ple"):
                layer.ple.prepare_cuda_graph_capture(token_count)

    def prepare_cuda_graph_replay(self, batch: "Batch") -> None:
        for layer in self.layers.op_list:
            if hasattr(layer, "ple"):
                layer.ple.prepare_cuda_graph_replay(batch)

    def readvance_gdn_states(self, stash: dict, start: int, rows: int, slot: int,
                             device: torch.device,
                             graph_consts: tuple | None = None) -> None:
        """Spec partial-accept rollback: re-advance every GDN layer's conv + recurrent
        state over rows [start, start+rows) of the stashed spec-verify span
        (Scheduler._replay_spec_states). Attention layers need no state rollback --
        their KV for the accepted rows was already written correctly by the verify
        forward. Same layer topology as qwen3_5 (``_is_linear`` GDN mixers keyed by
        ``linear_attn.layer_id``); ``graph_consts`` carries the rechain graph's
        persistent buffers on the replay path."""
        for layer in self.layers.op_list:
            if layer._is_linear:
                conv_in, a, b = stash[layer.linear_attn.layer_id]
                layer.linear_attn.spec_readvance(
                    conv_in[start : start + rows], a[start : start + rows],
                    b[start : start + rows], slot, device,
                    graph_consts=graph_consts,
                )


class Qwen4ExpForCausalLM(BaseLLMModel):
    """Text tower of Qwen4ExpForConditionalGeneration (Qwen3.8-Flash-Next).

    Served text-only: the checkpoint's ``model.visual.*`` tower and ``mtp.*`` head
    are dropped by the weight loader. ``prepare_for_runtime`` streams the ~102 GB
    PLE n-gram table into pinned host memory (it is not part of the state dict)."""

    def __init__(self, config: ModelConfig):
        self._config = config  # underscore attr: kept out of state_dict
        self.model = Qwen4ExpModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=(
                self.model.embed_tokens if config.tie_word_embeddings else None
            ),
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        x = self.model.forward_hidden(batch.input_ids)
        if batch.return_hidden:
            # MTP spec path: the scheduler-side drafter consumes the target's
            # last-layer PRE-MIXER hyper states (this [T, 10240] tensor) at drain
            # time, outside the forward ctx -- NOT the post-mixer mixed hidden.
            batch.hidden_states = x
        return self.lm_head.forward(self.model.hyper_connection_mixer.forward(x))

    def prepare_cuda_graph_capture(self, batch: "Batch") -> None:
        self.model.prepare_cuda_graph_capture(batch.input_ids.numel())

    def prepare_cuda_graph_replay(self, batch: "Batch") -> None:
        self.model.prepare_cuda_graph_replay(batch)

    def readvance_gdn_states(self, stash: dict, start: int, rows: int, slot: int,
                             device: torch.device,
                             graph_consts: tuple | None = None) -> None:
        """Delegate to the backbone (spec partial-accept rollback entry point for the
        scheduler; see Qwen4ExpModel.readvance_gdn_states)."""
        return self.model.readvance_gdn_states(
            stash, start, rows, slot, device, graph_consts=graph_consts,
        )

    def build_mtp_draft(self, model_path: str, device: torch.device):
        """Build the qwen4_exp MTP draft head from the checkpoint's bf16 mtp.*
        tensors, sharing this model's embed_tokens and lm_head by reference.
        Raises when the checkpoint carries no MTP head."""
        from prometheus.layers import ParallelLMHead

        from .mtp import Qwen4ExpMTPDraft, load_mtp_state_dict

        assert isinstance(self.lm_head, ParallelLMHead), (
            "--spec-mtp requires the plain bf16 lm_head (untied, unquantized)"
        )
        state = load_mtp_state_dict(model_path, device)
        if state is None:
            raise ValueError(f"--spec-mtp: no MTP draft head (mtp.*) in {model_path}")
        from prometheus.utils import torch_dtype

        with torch.device(device), torch_dtype(torch.bfloat16):
            draft = Qwen4ExpMTPDraft(self._config, self.model.embed_tokens, self.lm_head)
        draft.load_state_dict(state)
        draft.quantize_linears_nvfp4()
        return draft

    def prepare_for_runtime(self) -> None:
        q4: Qwen4ExpArgs = self._config.q4_args
        device = next(
            t.device
            for t in self.state_dict().values()
            if t.device.type != "meta"
        )
        for layer in self.model.layers.op_list:
            if hasattr(layer, "ple"):
                layer.ple.prepare(device)


__all__ = ["Qwen4ExpForCausalLM", "Qwen4ExpModel", "Qwen4ExpDecoderLayer", "Qwen4GatedResidual"]
