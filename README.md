# Prometheus

## Installation

### CLI

Install Prometheus with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "prometheus[accel]"
```

Or build from source:

```bash
git clone https://github.com/akushonkamen/prometheus-infer.git && cd prometheus-infer
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```
