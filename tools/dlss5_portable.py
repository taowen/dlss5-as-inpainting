"""Backward-compatible facade for :mod:`dlss5.portable`."""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from dlss5.portable import (  # noqa: E402,F401
    PORTABLE_FORMAT,
    DLSS5PortableModel,
    load_portable_checkpoint,
)

__all__ = ["PORTABLE_FORMAT", "DLSS5PortableModel", "load_portable_checkpoint"]
