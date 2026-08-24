"""Re-export the package version. ``prometheus.version`` is torch-free (the package ``__init__``
imports nothing but this), so this stays within the daemon's §2 import budget."""

from __future__ import annotations

from prometheus.version import __version__ as DAEMON_VERSION

__all__ = ["DAEMON_VERSION"]
