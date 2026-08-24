# Multi-stage build for Prometheus inference runtime.
# Stage 1 (builder): build & install the prometheus package plus its native C++/CUDA
# extensions into /root/.local (pip --user). torch wheels are pulled from PyPI's
# default cu130 build (matches the sglang-kernel pin in pyproject.toml).
# Stage 2 (runtime): copy only the install tree + required source assets, scrub
# everything that could leak upstream identity (docs/tests/scripts/READMEs/etc.)
# but keep the Apache-2.0 LICENSE under THIRD_PARTY_LICENSES per license terms.

############################
# Stage 1: builder
############################
FROM nvidia/cuda:12.8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build deps: python 3.12, git (setuptools_scm-ish version discovery), build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        git build-essential ninja-build cmake \
        ca-certificates curl wget \
    && rm -rf /var/lib/apt/lists/*

# Make `python` -> python3.12 so PEP 517 build isolation picks it up.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python -m pip install --upgrade pip setuptools>=77 wheel

WORKDIR /build

# Copy source. .dockerignore-style hygiene happens in Stage 2; here we need it all
# so the editable install builds the C++/CUDA extensions in-place.
COPY . /build/

# Install torch first (matches the cu130 pin in pyproject.toml) so the build
# isolation for prometheus sees the right libtorch when linking C++ extensions.
RUN pip install --user --index-url https://download.pytorch.org/whl/cu130 "torch>=2.11,<2.12"

# Install the package editable into /root/.local. Pass --no-build-isolation so
# it picks up the torch we just installed. The kernel-cache wheel is optional;
# if it is not prebuilt, JIT compilation kicks in at runtime (nvcc is present
# in the devel base image).
ENV PATH=/root/.local/bin:$PATH
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user --no-build-isolation -e .

# Smoke-import in builder to fail fast if any extension failed to build.
RUN python -c "import prometheus; print('prometheus OK ->', prometheus.__file__)"

############################
# Stage 2: runtime
############################
FROM nvidia/cuda:12.8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    PROMETHEUS_HOME=/opt/prometheus \
    XDG_CACHE_HOME=/var/cache/prometheus

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.12 /usr/bin/python \
    && mkdir -p /opt/prometheus /var/cache/prometheus

# Copy the editable-install tree (Python packages + compiled extensions).
COPY --from=builder /root/.local /root/.local

# Copy the source tree ONLY for the python package (the editable .pth points here).
# Everything outside python/ is upstream identity leakage we strip below.
COPY --from=builder /build/python /opt/prometheus/python

# Editable install: the .pth in /root/.local points at /build/python; rewrite it
# to the runtime location so `import prometheus` works without the build tree.
RUN pth=$(ls /root/.local/lib/python3.12/site-packages/__editable__.prometheus*.pth 2>/dev/null | head -1) && \
    if [ -n "$pth" ]; then \
        printf '/opt/prometheus/python\n' > "$pth"; \
    fi

# Source artifacts that leak upstream identity — strip from the runtime image.
RUN rm -rf \
        /opt/prometheus/python/prometheus/csrc \
        /opt/prometheus/python/prometheus/benchmark \
        /opt/prometheus/python/prometheus/tests 2>/dev/null; \
    # the editable install ships a finder; the .pth fallback above is enough
    true

# Apache-2.0 compliance: keep the license file as THIRD_PARTY_LICENSES.
COPY LICENSE /opt/prometheus/THIRD_PARTY_LICENSES

# Default model path is volume-mounted by the operator at runtime.
VOLUME ["/models"]

EXPOSE 8000

# ENTRYPOINT is `prom serve`; operator appends `--model /models/<dir> --port 8000`
# etc. on `docker run`.
ENTRYPOINT ["prom", "serve"]
CMD ["--model", "/models/model", "--port", "8000"]
