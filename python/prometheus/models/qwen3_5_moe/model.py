from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from prometheus.core import get_global_ctx
from prometheus.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from prometheus.models.blocks import BaseLLMModel
from prometheus.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from prometheus.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
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
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
                in_proj_split=g.in_proj_split,
                in_proj_nvfp4=g.in_proj_nvfp4,
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward_add_residual(x, residual)
        return x

    def readvance_gdn_states(self, stash: dict, start: int, rows: int, slot: int,
                             device: torch.device,
                             graph_consts: tuple | None = None) -> None:
        """Spec partial-accept rollback: re-advance every GDN layer's conv + recurrent
        state over rows [start, start+rows) of the stashed spec-verify span
        (Scheduler._replay_spec_states). Attention layers need no state rollback -- their
        KV for the accepted rows was already written correctly by the verify forward.

        ``graph_consts`` is forwarded to each layer's ``spec_readvance``: when not None
        it carries the (cu, idx, has_init) persistent buffers owned by the rechain CUDA
        graph, so the replay path avoids H2D copies inside the capture region."""
        for layer in self.layers.op_list:
            if layer._is_linear:
                conv_in, a, b = stash[layer.linear_attn.layer_id]
                layer.linear_attn.spec_readvance(
                    conv_in[start : start + rows], a[start : start + rows],
                    b[start : start + rows], slot, device,
                    graph_consts=graph_consts,
                )


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self._config = config  # underscore attr: kept out of state_dict
        self.model = Qwen3_5Model(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from prometheus.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        batch = get_global_ctx().batch
        if batch.return_hidden:
            # MTP spec path: the scheduler-side drafter consumes the target's final
            # hidden states (this tensor) at drain time, outside the forward ctx.
            batch.hidden_states = output
        return self.lm_head.forward(output)

    def readvance_gdn_states(self, stash: dict, start: int, rows: int, slot: int,
                             device: torch.device,
                             graph_consts: tuple | None = None) -> None:
        """Delegate to the backbone (spec partial-accept rollback entry point for the
        scheduler; see Qwen3_5Model.readvance_gdn_states)."""
        return self.model.readvance_gdn_states(
            stash, start, rows, slot, device, graph_consts=graph_consts,
        )

    def build_mtp_draft(self, model_path: str, device: torch.device):
        """Build the MTP draft head from the checkpoint's bf16 mtp.* tensors, sharing
        this model's embed_tokens and lm_head by reference. Raises when the checkpoint
        carries no MTP head or the lm_head is not a plain bf16 head."""
        from prometheus.layers import ParallelLMHead

        from .mtp import MTPDraft, load_mtp_state_dict

        from prometheus.kernel.triton.nvfp4_linear import Nvfp4LMHead

        assert isinstance(self.lm_head, (ParallelLMHead, Nvfp4LMHead)), (
            "--spec-mtp requires the plain bf16 or NVFP4 lm_head (untied)"
        )
        state = load_mtp_state_dict(model_path, device)
        if state is None:
            raise ValueError(f"--spec-mtp: no MTP draft head (mtp.*) in {model_path}")
        from prometheus.utils import torch_dtype

        import json
        import os

        moe = False
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                # .get: an index without a weight_map key (other index schemas)
                # simply has no MTP experts -> moe=False, not a boot-time KeyError.
                moe = any(
                    k.startswith("mtp.layers.0.mlp.experts.")
                    for k in json.load(f).get("weight_map", {})
                )
        with torch.device(device), torch_dtype(torch.bfloat16):
            draft = MTPDraft(self._config, self.model.embed_tokens, self.lm_head, moe=moe)
        draft.load_state_dict(state)
        draft.quantize_linears_nvfp4()
        return draft


__all__ = ["Qwen3_5MoEForCausalLM"]
