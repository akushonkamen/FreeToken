"""Device abstraction for CPU-only (Apple Silicon / MPS) hosts.

Prometheus's engine is CUDA-first: :class:`~prometheus.engine.Engine` hard-codes
``torch.device("cuda:N")``, a default CUDA stream, CUDA events for copy-sync, and
``torch.cuda.mem_get_info`` for free-memory accounting. On a machine without CUDA
(apple Silicon, MPS) those calls raise at construction time.

This module provides the minimal shim layer so the engine can boot on MPS:

  * :func:`resolve_device` picks ``cuda``, ``mps``, or ``cpu`` in that order.
  * :func:`device_stream` returns a real CUDA stream on CUDA, a no-op context
    manager on MPS/CPU (MPS is single-stream; no explicit stream needed).
  * :func:`device_sync` calls ``torch.cuda.synchronize`` on CUDA and
    ``torch.mps.synchronize`` on MPS.
  * :func:`free_memory` returns the free VRAM (CUDA) or the free system unified
    memory (MPS/CPU, via ``torch.mps.*`` or ``psutil``/``os.sched``).
  * :func:`empty_cache` purges the caching allocator (no-op on CPU).

The engine's CUDA-specific fast paths (CUDA graph capture, FP8 kernels, the
offload expert cache) remain CUDA-only and raise at *use* time on MPS; this shim
only unblocks model construction + eager forward.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch


def has_cuda() -> bool:
    return torch.cuda.is_available()


def has_mps() -> bool:
    return torch.backends.mps.is_available()


def resolve_device(rank: int = 0) -> torch.device:
    """Pick the best available device for ``rank``."""
    if has_cuda():
        return torch.device(f"cuda:{rank}")
    if has_mps():
        return torch.device("mps")
    return torch.device("cpu")


class _NullStream:
    """Stand-in for torch.cuda.Stream on devices without streams (MPS/CPU)."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or resolve_device()

    def wait_event(self, _event: object) -> None:
        pass

    def wait_stream(self, _stream: object) -> None:
        pass

    def record_event(self) -> _NullEvent:
        return _NullEvent()

    def synchronize(self) -> None:
        device_sync(self.device)


class _NullEvent:
    """Stand-in for torch.cuda.Event on devices without streams."""

    def __init__(self, enable_timing: bool = False) -> None:
        self._enable_timing = enable_timing
        self._t: float | None = None

    def record(self) -> None:
        import time

        if self._enable_timing:
            self._t = time.monotonic()

    def wait(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, other: "_NullEvent") -> float:
        if self._t is not None and other._t is not None:
            return (other._t - self._t) * 1000.0
        return 0.0


def make_stream(device: torch.device) -> _NullStream | torch.cuda.Stream:
    """Create a stream for ``device`` (CUDA real, MPS/CPU null)."""
    if device.type == "cuda":
        return torch.cuda.Stream(device=device)
    return _NullStream(device)


def set_stream(stream: _NullStream | torch.cuda.Stream) -> None:
    """Set the current stream (no-op on MPS/CPU)."""
    if isinstance(stream, _NullStream):
        return
    torch.cuda.set_stream(stream)


def device_sync(device: torch.device) -> None:
    """Synchronize the device's compute queue."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    """Purge the device's caching allocator."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # MPS cache purge is a no-op (torch.mps.empty_cache exists in some builds but
    # is a hint, not a hard reclaim); leave it to the OS.


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def free_memory(device: torch.device) -> int:
    """Free memory in bytes on the device (CUDA) or unified memory (MPS/CPU)."""
    if device.type == "cuda":
        return torch.cuda.mem_get_info(device)[0]
    # Apple Silicon unified memory: report the system free.
    # torch.mps.current_allocated_memory / driver_allocated_memory give the *used*
    # side; free = total - used.
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        # Fallback: os.schedemm (Linux) or a conservative constant.
        return 8 * 1024**3  # 8 GiB placeholder


def get_device_name(device: torch.device) -> str | None:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Silicon (MPS)"
    return None


@contextmanager
def stream_context(stream: _NullStream | torch.cuda.Stream) -> Iterator[None]:
    """Context manager that activates ``stream`` (no-op for null streams)."""
    if isinstance(stream, _NullStream):
        yield
        return
    with torch.cuda.stream(stream):
        yield
