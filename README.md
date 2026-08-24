<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/prometheus-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/prometheus-logo-light.svg">
    <img alt="Prometheus" src="assets/prometheus-logo.svg" width=65%>
  </picture>
</div>

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
