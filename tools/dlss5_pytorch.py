"""Compatibility facade for the split DLSS5 PyTorch implementation.

The implementation now lives under :mod:`tools.dlss5` (or the top-level
``dlss5`` package when a script is executed from ``tools``).  This module is
kept intentionally small so existing research scripts and user code can keep
importing ``dlss5_pytorch`` without knowing the internal layout.
"""

from __future__ import annotations

try:  # ``python tools/probe_dlss5_pytorch.py`` puts tools/ on sys.path.
    from dlss5.blocks import (  # noqa: F401
        CCTVitAttention,
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
    )
    from dlss5.cli import _self_test, main  # noqa: F401
    from dlss5.graph import DLSS5Graph
    from dlss5.loaders import DLSS5WeightLoader  # noqa: F401
    from dlss5.layouts import (  # noqa: F401
        DOWNSAMPLE_SWIN_BLOCKS,
        KNOWN_DEC_INPUT_LAYOUT,
        KNOWN_POST_SWIN_LAYOUT,
        KNOWN_PRE_SWIN_LAYOUT,
        KNOWN_SPLIT_BLOB_LAYOUT,
        KNOWN_STANDARD_SWIN_LAYOUT,
        KNOWN_SWIN_ATTENTION_BIAS,
        KNOWN_UPSAMPLE_SWIN_LAYOUT,
        KNOWN_VIT_BLOB_LAYOUT,
        STANDARD_SWIN_BLOCKS,
        STANDARD_SWIN_FFN_DIMS,
        SWIN_BODY_BLOCKS,
        UPSAMPLE_SWIN_BLOCKS,
    )
    from dlss5.ops import (  # noqa: F401
        CCT_CUBIC_SILU_ABS_COEFF,
        CCT_CUBIC_SILU_BIAS,
        CCT_CUBIC_SILU_LINEAR_COEFF,
        CCTCubicSiLU,
        assemble_pre_front_feature_lanes,
        build_pre_front_sass_candidate,
        cct_cubic_silu,
        decode_hmma_16816_f16_tile,
        decode_post_output_tile_candidate,
        decode_post_output_tile_column_major,
        decode_s_e4m3,
        quantize_s_e4m3_satfinite,
    )
    from dlss5.weights import DLSS5WeightMap, WeightRecord, decode_fp8_matrix  # noqa: F401
except ModuleNotFoundError:  # ``import tools.dlss5_pytorch`` from the repo root.
    from .dlss5.blocks import (  # noqa: F401
        CCTVitAttention,
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
    )
    from .dlss5.cli import _self_test, main  # noqa: F401
    from .dlss5.graph import DLSS5Graph
    from .dlss5.loaders import DLSS5WeightLoader  # noqa: F401
    from .dlss5.layouts import (  # noqa: F401
        DOWNSAMPLE_SWIN_BLOCKS,
        KNOWN_DEC_INPUT_LAYOUT,
        KNOWN_POST_SWIN_LAYOUT,
        KNOWN_PRE_SWIN_LAYOUT,
        KNOWN_SPLIT_BLOB_LAYOUT,
        KNOWN_STANDARD_SWIN_LAYOUT,
        KNOWN_SWIN_ATTENTION_BIAS,
        KNOWN_UPSAMPLE_SWIN_LAYOUT,
        KNOWN_VIT_BLOB_LAYOUT,
        STANDARD_SWIN_BLOCKS,
        STANDARD_SWIN_FFN_DIMS,
        SWIN_BODY_BLOCKS,
        UPSAMPLE_SWIN_BLOCKS,
    )
    from .dlss5.ops import (  # noqa: F401
        CCT_CUBIC_SILU_ABS_COEFF,
        CCT_CUBIC_SILU_BIAS,
        CCT_CUBIC_SILU_LINEAR_COEFF,
        CCTCubicSiLU,
        assemble_pre_front_feature_lanes,
        build_pre_front_sass_candidate,
        cct_cubic_silu,
        decode_hmma_16816_f16_tile,
        decode_post_output_tile_candidate,
        decode_post_output_tile_column_major,
        decode_s_e4m3,
        quantize_s_e4m3_satfinite,
    )
    from .dlss5.weights import DLSS5WeightMap, WeightRecord, decode_fp8_matrix  # noqa: F401


__all__ = [
    "DLSS5Graph",
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
]


if __name__ == "__main__":
    main()
