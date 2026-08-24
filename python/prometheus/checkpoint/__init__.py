"""Prometheus Weight (FTW) checkpoint: a unified, O_DIRECT-friendly on-disk weight format.

See :mod:`prometheus.checkpoint.ftw` for the format and :mod:`prometheus.checkpoint.convert`
for the safetensors -> FTW converter (also exposed as ``prom checkpoint``).
"""

from .ftw import (
    FTWReader,
    FTWWriter,
    is_ftw_checkpoint,
    iter_ftw_weights,
    load_ftw_banks,
)
from .convert import convert_checkpoint

__all__ = [
    "FTWReader", "FTWWriter", "is_ftw_checkpoint",
    "iter_ftw_weights", "load_ftw_banks", "convert_checkpoint",
]
