"""Backward-compatible import/CLI facade for the DLSS5 model package.

Install the repository with ``pip install -e .`` for normal imports.  The
path bootstrap below also keeps the historical ``python tools/<script>.py``
commands working from a checkout without installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from dlss5 import (  # noqa: E402,F401
    CCT_CUBIC_SILU_ABS_COEFF,
    CCT_CUBIC_SILU_BIAS,
    CCT_CUBIC_SILU_LINEAR_COEFF,
    CCTCubicSiLU,
    CCTVitAttention,
    DLSS5Graph,
    DLSS5PortableModel,
    DLSS5WeightLoader,
    DLSS5WeightMap,
    DecInputUpsample,
    PatchExpand,
    PatchMerging,
    PreSwinDownsample,
    SplitSwinBlock,
    SwinBlock,
    SwinDownBlock,
    SwinTransitionDown,
    SwinTransitionUp,
    SwinUpBlock,
    ViTBlock,
    WindowCosineAttention,
    WeightRecord,
    assemble_pre_front_feature_lanes,
    build_pre_front_sass_candidate,
    cct_cubic_silu,
    decode_fp8_matrix,
    decode_hmma_16816_f16_tile,
    decode_post_output_tile_candidate,
    decode_post_output_tile_column_major,
    decode_s_e4m3,
    load_portable_checkpoint,
    quantize_s_e4m3_satfinite,
)
from dlss5.cli import _self_test, main  # noqa: E402,F401

__all__ = [
    "DLSS5Graph",
    "DLSS5PortableModel",
    "DLSS5WeightLoader",
    "DLSS5WeightMap",
    "WeightRecord",
    "decode_s_e4m3",
    "quantize_s_e4m3_satfinite",
    "decode_fp8_matrix",
    "decode_post_output_tile_candidate",
    "decode_post_output_tile_column_major",
    "decode_hmma_16816_f16_tile",
    "assemble_pre_front_feature_lanes",
    "build_pre_front_sass_candidate",
    "cct_cubic_silu",
    "CCTCubicSiLU",
    "load_portable_checkpoint",
]


if __name__ == "__main__":
    main()
