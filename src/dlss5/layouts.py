"""Recovered inner tensor layouts and block-to-record mappings."""

from __future__ import annotations

KNOWN_VIT_BLOB_LAYOUT = {
    "layer0": {
        "operator": "FFN expand",
        "fp8_offset": 0,
        "fp8_shape": (4096, 1024),
        "trailing_bytes": 16,
    },
    "layer1": {
        "operator": "FFN contract",
        "fp8_offset": 0,
        "fp8_shape": (1024, 4096),
        "cos_skip_offset": 4194304,
        "cos_skip_shape": (1024,),
    },
    "layer2": {
        "operator": "QKV",
        # This 64-half header is present before the 3072x1024 FP8 matrix in
        # every ViT blob.  Its producer semantics are not yet identified.
        "fp8_offset": 128,
        "fp8_shape": (3072, 1024),
        "header_f16": 64,
    },
    "layer3": {"operator": "attention scalar", "f16_shape": (1,)},
    "layer4": {
        "operator": "projection",
        "fp8_offset": 0,
        "fp8_shape": (1024, 1024),
        "cos_skip_offset": 1048576,
        "cos_skip_shape": (1024,),
    },
}


KNOWN_SPLIT_BLOB_LAYOUT = {
    "layer0": {
        "operator": "FFN gated pair",
        "fp8_offsets": (0, 262144),
        "fp8_shapes": ((512, 512), (512, 512)),
    },
    "layer1": {
        "operator": "FFN contract",
        "fp8_offset": 0,
        "fp8_shape": (512, 512),
        "cos_skip_offset": 262144,
        "cos_skip_shape": (512,),
    },
    "layer2": {
        "operator": "QKV + attention metadata",
        "fp8_offset": 0,
        "fp8_shape": (1536, 512),
        "attn_bias_offset": 786432,
        "attn_bias_shape": (16, 64, 64),
        "attn_scale_offset": 917504,
        "attn_scale_shape": (16,),
        "attn_scale_dtype": "float32",
    },
    "layer3": {
        "operator": "projection",
        "fp8_offset": 0,
        "fp8_shape": (512, 512),
        "cos_skip_offset": 262144,
        "cos_skip_shape": (512,),
    },
}


# Ordinary (non-downsample/non-upsample) Swin blobs use a custom MLP width
# rather than the usual 4*C.  These widths and byte offsets are recovered by
# matching the host-side names (weight1/weight2/ffn_cos_skip/qkv_weight/
# attn_bias/projection_weight/attn_cos_skip) against the record sizes.  The
# gap immediately before projection is a packed FP32 per-head attention
# scale, with alignment padding for the smaller channel families; the other
# gaps remain opaque.
STANDARD_SWIN_FFN_DIMS = {32: 128, 64: 224, 128: 384, 256: 704}

KNOWN_STANDARD_SWIN_LAYOUT = {
    32: {
        "expected_bytes": 20672,
        "weight1": (0, (128, 32)),
        "weight2": (4096, (32, 128)),
        "ffn_cos_skip": (8208, (32,)),
        "qkv": (8272, (96, 32)),
        "attn_bias": (11360, (1, 64, 64)),
        "attn_scale": (19552, (1,)),
        "projection": (19568, (32, 32)),
        "attn_cos_skip": (20592, (32,)),
    },
    64: {
        "expected_bytes": 61760,
        "weight1": (0, (224, 64)),
        "weight2": (14336, (64, 224)),
        "ffn_cos_skip": (28688, (64,)),
        "qkv": (28816, (192, 64)),
        "attn_bias": (41120, (2, 64, 64)),
        "attn_scale": (57504, (2,)),
        "projection": (57520, (64, 64)),
        "attn_cos_skip": (61616, (64,)),
    },
    128: {
        "expected_bytes": 197184,
        "weight1": (0, (384, 128)),
        "weight2": (49152, (128, 384)),
        "ffn_cos_skip": (98320, (128,)),
        "qkv": (98576, (384, 128)),
        "attn_bias": (147744, (4, 64, 64)),
        "attn_scale": (180512, (4,)),
        "projection": (180528, (128, 128)),
        "attn_cos_skip": (196912, (128,)),
    },
    256: {
        "expected_bytes": 689232,
        "weight1": (0, (704, 256)),
        "weight2": (180224, (256, 704)),
        "ffn_cos_skip": (360464, (256,)),
        "qkv": (360976, (768, 256)),
        "attn_bias": (557600, (8, 64, 64)),
        "attn_scale": (623136, (8,)),
        "projection": (623168, (256, 256)),
        "attn_cos_skip": (688704, (256,)),
    },
}

# Upsample blocks prepend a 2*C^2-byte transition region and insert an
# additional C-sized opaque ``sin`` region before QKV.  The Swin body
# matrices/scales/bias below are independently aligned and can therefore be
# loaded without guessing the transition operand.
KNOWN_UPSAMPLE_SWIN_LAYOUT = {
    32: {
        "weight1": (2048, (128, 32)),
        "weight2": (6144, (32, 128)),
        "ffn_cos_skip": (10256, (32,)),
        "qkv": (10400, (96, 32)),
        "attn_bias": (13472, (1, 64, 64)),
        "attn_scale": (21664, (1,)),
        "projection": (21680, (32, 32)),
        "attn_cos_skip": (22704, (32,)),
        "expected_bytes": 22784,
        "prefix_bytes": 2048,
        "opaque_before_qkv": (10320, 80),
    },
    64: {
        "weight1": (8192, (224, 64)),
        "weight2": (22528, (64, 224)),
        "ffn_cos_skip": (36864, (64,)),
        "qkv": (37120, (192, 64)),
        "attn_bias": (49408, (2, 64, 64)),
        "attn_scale": (65792, (2,)),
        "projection": (65808, (64, 64)),
        "attn_cos_skip": (69904, (64,)),
        "expected_bytes": 70048,
        "prefix_bytes": 8192,
        "opaque_before_qkv": (36992, 128),
    },
    128: {
        "weight1": (32768, (384, 128)),
        "weight2": (81920, (128, 384)),
        "ffn_cos_skip": (131072, (128,)),
        "qkv": (131584, (384, 128)),
        "attn_bias": (180736, (4, 64, 64)),
        "attn_scale": (213504, (4,)),
        "projection": (213520, (128, 128)),
        "attn_cos_skip": (229904, (128,)),
        "expected_bytes": 230176,
        "prefix_bytes": 32768,
        "opaque_before_qkv": (131328, 256),
    },
    256: {
        "weight1": (131072, (704, 256)),
        "weight2": (311296, (256, 704)),
        "ffn_cos_skip": (491520, (256,)),
        "qkv": (492544, (768, 256)),
        "attn_bias": (689152, (8, 64, 64)),
        "attn_scale": (754688, (8,)),
        "projection": (754720, (256, 256)),
        "attn_cos_skip": (820256, (256,)),
        "expected_bytes": 820784,
        "prefix_bytes": 131072,
        "opaque_before_qkv": (492032, 512),
    },
}

KNOWN_POST_SWIN_LAYOUT = {
    "expected_bytes": 21808,
    "weight1": (0, (128, 32)),
    "weight2": (4096, (32, 128)),
    "ffn_cos_skip": (8208, (32,)),
    "qkv": (8400, (96, 32)),
    "attn_bias": (11472, (1, 64, 64)),
    "projection": (19680, (32, 32)),
    "attn_cos_skip": (20704, (32,)),
    "attn_scale": (19664, (1,)),
    "opaque_before_qkv": (8272, 128),
    # The post factory's first four operations are convolution -> alias ->
    # mul -> add.  The 128-byte prefix is two 32-channel FP16 vectors:
    # depthwise input projection and the skip/input scale.  ``sin`` is a
    # dynamic interpolation operand and has no serialized payload here.
    "input_dw_weight_f16": (8272, (32,)),
    "input_scale_f16": (8336, (32,)),
    # Weight registration names put ``out_gain`` before ``out_conv_weight``.
    # The 1040-byte tail therefore splits exactly into an 8-half (16-byte)
    # gain slot followed by a 16x32-half output tile.  The latter is still a
    # tensor-core-swizzled tile; its physical rows are not yet proven to be
    # ordinary row-major output channels.
    "out_gain_f16": (20768, (8,)),
    "opaque_output_tail": (20768, 1040),
    "out_conv_weight_f16": (20784, (16, 32)),
}

KNOWN_PRE_SWIN_LAYOUT = {
    "expected_bytes": 21696,
    # The pre kernel's two body matrices start at the record base, like the
    # ordinary C32 body. Its extra texture/front-end payload is inserted
    # after those matrices, immediately before the residual/QKV metadata.
    "weight1": (0, (128, 32)),
    "weight2": (4096, (32, 128)),
    # Two 512-byte FP16 tiles are loaded by the HMMA front-end before the
    # body. Their producer feature assembly is still texture-dependent.
    "front_weight0_f16": (8208, (16, 16)),
    "front_weight1_f16": (8720, (16, 16)),
    "ffn_cos_skip": (9232, (32,)),
    "qkv": (9296, (96, 32)),
    "attn_bias": (12384, (1, 64, 64)),
    "projection": (20592, (32, 32)),
    "attn_cos_skip": (21616, (32,)),
    "attn_scale": (20576, (1,)),
}

KNOWN_DEC_INPUT_LAYOUT = {
    "expected_bytes": 525312,
    "conv_weight": (0, (512, 1024)),
    "dw_weight": (524288, (512,)),
    "opaque_sin": (525312, 0),
}

# Keep the bias-only view available to callers that want a cheap probe.  The
# offsets here are the starts of the actual FP16 bias arrays, not the starts
# of the preceding opaque alignment bytes.
KNOWN_SWIN_ATTENTION_BIAS = {
    32: {"heads": 1, "offset": 11360},
    64: {"heads": 2, "offset": 41120},
    128: {"heads": 4, "offset": 147744},
    256: {"heads": 8, "offset": 557600},
}

STANDARD_SWIN_BLOCKS = {
    **{index: (32, index - 1) for index in range(1, 4)},
    **{index: (64, index - 5) for index in range(5, 8)},
    **{index: (128, index - 9) for index in range(9, 14)},
    **{index: (256, index - 15) for index in range(15, 22)},
    **{index: (256, index - 49) for index in range(49, 56)},
    **{index: (128, index - 57) for index in range(57, 62)},
    **{index: (64, index - 63) for index in range(63, 66)},
    **{index: (32, index - 67) for index in range(67, 70)},
}

# The downsample variants reuse the ordinary Swin body at the same byte
# offsets and append one extra transition operand.  The operand's matrix shape
# is loaded below; its exact spatial packing remains kernel-specific.
DOWNSAMPLE_SWIN_BLOCKS = {
    4: (32, 0),
    8: (64, 0),
    14: (128, 0),
    22: (256, 0),
}
UPSAMPLE_SWIN_BLOCKS = {
    48: (256, 0),
    56: (128, 0),
    62: (64, 0),
    66: (32, 0),
}
SWIN_BODY_BLOCKS = {
    **STANDARD_SWIN_BLOCKS,
    **DOWNSAMPLE_SWIN_BLOCKS,
    **UPSAMPLE_SWIN_BLOCKS,
}
