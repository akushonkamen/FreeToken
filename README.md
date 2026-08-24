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
