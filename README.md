# Prometheus

## Installation

### CLI

Install Prometheus with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "prometheus[accel]"
```

Or build from source:

```bash
git clone https://github.com/AgentsEngine/prometheus-infer.git && cd prometheus-infer
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Supported models

| Family | Architectures | Notes |
|---|---|---|
| **Qwen3-Next-80B-A3B** | `Qwen3NextForCausalLM` | **new**: bf16 / block-FP8 / NVFP4, offload & hybrid backends; 96.9 tok/s decode (NVFP4 offload, single RTX 4090) |
| Qwen3.6 / Qwen3.5-MoE | `Qwen3_5MoeForConditionalGeneration` | bf16 / NVFP4, GDN linear attention |
| Qwen3-MoE / Qwen3 / Qwen2 | `Qwen3MoeForCausalLM`, `Qwen3ForCausalLM`, `Qwen2ForCausalLM` | dense + MoE |
| DeepSeek-V4-Flash | `DeepseekV4ForCausalLM` | native MXFP4 experts |
| GLM-4-MoE / GLM-MoE-DSA | `Glm4MoeForCausalLM`, `GlmMoeDsaForCausalLM` | |
| MiniMax M2 / M3-Sparse | `MiniMaxM2ForCausalLM`, `MiniMaxM3SparseForCausalLM` | |
| GPT-OSS | `GptOssForCausalLM` | |
| Gemma 4 | `Gemma4ForCausalLM` / `Gemma4ForConditionalGeneration` / GGUF variant | |
| Mistral / Mistral-3 | `MistralForCausalLM`, `Mistral3ForConditionalGeneration` | |
| Muse-Glimmer-30B | `MuseGlimmerForConditionalGeneration` | text-only serving |
| Llama | `LlamaForCausalLM` | |

## UI Playground

A local web playground (`ui/`) for chatting with Prometheus models, streaming
thinking tokens, and watching live GPU/throughput stats plus decode benchmarks
side-by-side.

```bash
ssh -fN -L 8790:127.0.0.1:42424 -L 8791:127.0.0.1:31337 219  # tunnels to DSV4 + Qwen3.5
cd ui && python3 server.py --port 1420
# open http://localhost:1420
```

`server.py` routes `/v1/chat/completions` to the right backend by the model
field in the request body, merges `/v1/models` from every tunnel, and proxies
`/bench/<model>/<backend>` to the benchmark JSON on the remote host.
`index.html` switches the benchmark table when the model dropdown changes.
