"""WEIGHTS_HT deserialization methods for the semantic graph."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn

from .layouts import (
    DOWNSAMPLE_SWIN_BLOCKS, KNOWN_DEC_INPUT_LAYOUT, KNOWN_POST_SWIN_LAYOUT,
    KNOWN_PRE_SWIN_LAYOUT, KNOWN_SPLIT_BLOB_LAYOUT, KNOWN_STANDARD_SWIN_LAYOUT,
    KNOWN_UPSAMPLE_SWIN_LAYOUT, KNOWN_VIT_BLOB_LAYOUT, STANDARD_SWIN_BLOCKS,
    SWIN_BODY_BLOCKS, UPSAMPLE_SWIN_BLOCKS,
)
from .ops import (
    decode_post_output_tile_candidate, decode_post_output_tile_column_major,
    _fp8_boundary,
)
from .weights import (
    DLSS5WeightMap, _copy_parameter, _decode_blob_f16, _decode_blob_f32,
    _decode_blob_matrix, _expect_blob_size, decode_fp8_matrix,
)

class DLSS5WeightLoader:
    def load_swin_weights(
        self,
        weights: DLSS5WeightMap,
        *,
        blocks: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Load the proven ordinary-Swin matrices, scales, and bias.

        The ordinary block is one fused ``layer0.layer`` record.  Its inner
        layout is now fixed for all four channel families, including the
        packed FP32 per-head ``attn_scale`` immediately before projection.
        """

        targets: dict[int, SwinBlock] = {}
        for index, module in enumerate(self.enc32):
            targets[1 + index] = module
        targets[4] = self.down32_64.body
        for index, module in enumerate(self.enc64):
            targets[5 + index] = module
        targets[8] = self.down64_128.body
        for index, module in enumerate(self.enc128):
            targets[9 + index] = module
        targets[14] = self.down128_256.body
        for index, module in enumerate(self.enc256):
            targets[15 + index] = module
        targets[22] = self.down256_512.body
        for index, module in enumerate(self.dec256):
            targets[49 + index] = module
        targets[48] = self.up512_256.body
        for index, module in enumerate(self.dec128):
            targets[57 + index] = module
        targets[56] = self.up256_128.body
        for index, module in enumerate(self.dec64):
            targets[63 + index] = module
        targets[62] = self.up128_64.body
        for index, module in enumerate(self.dec32):
            targets[67 + index] = module
        targets[66] = self.up64_32.body
        transition_targets: dict[int, nn.Module] = {
            4: self.down32_64.downsample,
            8: self.down64_128.downsample,
            14: self.down128_256.downsample,
            22: self.down256_512.downsample,
            48: self.up512_256.upsample,
            56: self.up256_128.upsample,
            62: self.up128_64.upsample,
            66: self.up64_32.upsample,
        }

        if blocks is None:
            blocks = sorted(SWIN_BODY_BLOCKS)
        loaded: list[str] = []
        metadata: list[dict[str, Any]] = []
        skipped: list[str] = []
        for block_index in blocks:
            if block_index not in SWIN_BODY_BLOCKS:
                skipped.append(f"block{block_index} non-decoded Swin fused layout")
                continue
            channels, _ = SWIN_BODY_BLOCKS[block_index]
            if block_index in UPSAMPLE_SWIN_BLOCKS:
                spec = KNOWN_UPSAMPLE_SWIN_LAYOUT[channels]
            else:
                spec = KNOWN_STANDARD_SWIN_LAYOUT[channels]
            name = f"block{block_index}.layer0.layer"

            if block_index in STANDARD_SWIN_BLOCKS:
                _expect_blob_size(weights, name, int(spec["expected_bytes"]))
            else:
                required_end = int(spec["attn_cos_skip"][0]) + 2 * channels
                if weights.record(name).data_size < required_end:
                    raise ValueError(f"truncated Swin body in {name}")
            raw = weights.fp8(name)
            weight1_offset, weight1_shape = spec["weight1"]
            weight2_offset, weight2_shape = spec["weight2"]
            qkv_offset, qkv_shape = spec["qkv"]
            projection_offset, projection_shape = spec["projection"]
            weight1 = decode_fp8_matrix(raw, tuple(weight1_shape), byte_offset=int(weight1_offset))
            weight2 = decode_fp8_matrix(raw, tuple(weight2_shape), byte_offset=int(weight2_offset))
            qkv = decode_fp8_matrix(raw, tuple(qkv_shape), byte_offset=int(qkv_offset))
            projection = decode_fp8_matrix(
                raw, tuple(projection_shape), byte_offset=int(projection_offset)
            )
            _copy_parameter(targets[block_index].mlp[0].weight, weight1, f"{name}.weight1")
            _copy_parameter(targets[block_index].mlp[2].weight, weight2, f"{name}.weight2")
            _copy_parameter(targets[block_index].attn.qkv.weight, qkv, f"{name}.qkv_weight")
            _copy_parameter(targets[block_index].attn.proj.weight, projection, f"{name}.projection_weight")
            loaded.extend(
                [
                    f"{name}.weight1 -> mlp[0]",
                    f"{name}.weight2 -> mlp[2]",
                    f"{name}.qkv_weight -> attn.qkv",
                    f"{name}.projection_weight -> attn.proj",
                ]
            )

            ffn_cos_skip_offset, ffn_cos_skip_shape = spec["ffn_cos_skip"]
            ffn_cos_skip = _decode_blob_f16(
                weights, name, int(ffn_cos_skip_offset), tuple(ffn_cos_skip_shape)
            )
            _copy_parameter(
                targets[block_index].ffn_cos_skip,
                ffn_cos_skip,
                f"{name}.ffn_cos_skip",
            )
            loaded.append(f"{name}.ffn_cos_skip -> ffn_cos_skip")

            attn_cos_skip_offset, attn_cos_skip_shape = spec["attn_cos_skip"]
            attn_cos_skip = _decode_blob_f16(
                weights, name, int(attn_cos_skip_offset), tuple(attn_cos_skip_shape)
            )
            _copy_parameter(
                targets[block_index].attn_cos_skip,
                attn_cos_skip,
                f"{name}.attn_cos_skip",
            )
            loaded.append(f"{name}.attn_cos_skip -> attn_cos_skip")

            attn_bias_offset, attn_bias_shape = spec["attn_bias"]
            bias = _decode_blob_f16(
                weights,
                name,
                int(attn_bias_offset),
                tuple(attn_bias_shape),
            )
            if not torch.isfinite(bias).all():
                bad = int((~torch.isfinite(bias)).sum())
                skipped.append(f"{name}.attn_bias contains {bad} non-finite FP16 value(s)")
            else:
                _copy_parameter(targets[block_index].attn.attn_bias, bias, f"{name}.attn_bias")
                loaded.append(
                    f"{name}[{attn_bias_offset}:{int(attn_bias_offset) + bias.numel() * 2}] -> attn_bias"
                )

            scale_offset, scale_shape = spec["attn_scale"]
            scale = _decode_blob_f32(weights, name, int(scale_offset), tuple(scale_shape))
            _copy_parameter(
                targets[block_index].attn.attn_scale,
                scale,
                f"{name}.attn_scale",
            )
            loaded.append(
                f"{name}[{int(scale_offset)}:{int(scale_offset) + scale.numel() * 4}] -> attn_scale"
            )
            if block_index in DOWNSAMPLE_SWIN_BLOCKS:
                transition_bytes = 2 * channels * channels
                transition_offset = weights.record(name).data_size - transition_bytes
                transition = decode_fp8_matrix(
                    raw,
                    (2 * channels, channels),
                    byte_offset=transition_offset,
                )
                _copy_parameter(
                    transition_targets[block_index].conv_weight.weight,
                    transition,
                    f"{name}.downsample_conv",
                )
                loaded.append(f"{name}.downsample_conv -> transition")
                metadata.append(
                    {
                        "name": f"block{block_index}.downsample_spatial",
                        "applied": False,
                        "reason": "kernel spatial reduction is represented by average pooling",
                    }
                )
            if block_index in UPSAMPLE_SWIN_BLOCKS:
                transition = decode_fp8_matrix(
                    raw,
                    (channels, 2 * channels),
                    byte_offset=0,
                )
                _copy_parameter(
                    transition_targets[block_index].conv_weight.weight,
                    transition,
                    f"{name}.upsample_conv",
                )
                loaded.append(f"{name}.upsample_conv -> transition")
                metadata.append(
                    {
                        "name": f"block{block_index}.upsample_prefix",
                        "bytes": int(spec["prefix_bytes"]),
                        "applied": True,
                        "reason": "loaded as transition convolution",
                    }
                )
                opaque_offset, opaque_bytes = spec["opaque_before_qkv"]
                metadata.append(
                    {
                        "name": f"block{block_index}.sin_or_opaque",
                        "offset": int(opaque_offset),
                        "bytes": int(opaque_bytes),
                        "applied": False,
                        "reason": "inserted before QKV; operand meaning is not resolved",
                    }
                )
        return {"loaded": loaded, "metadata": metadata, "skipped": skipped}

    def load_swin_attention_biases(
        self,
        weights: DLSS5WeightMap,
        *,
        blocks: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Compatibility alias for :meth:`load_swin_weights`."""

        return self.load_swin_weights(weights, blocks=blocks)

    def load_pre_weights(self, weights: DLSS5WeightMap) -> dict[str, Any]:
        """Load block 0's C32 body and retain its unresolved front tiles."""

        name = "block0.layer0.layer"
        spec = KNOWN_PRE_SWIN_LAYOUT
        _expect_blob_size(weights, name, int(spec["expected_bytes"]))
        raw = weights.fp8(name)
        loaded: list[str] = []
        metadata: list[dict[str, Any]] = [
            {
                "name": "block0.pre_texture_front_weight0_f16",
                "offset": int(spec["front_weight0_f16"][0]),
                "bytes": math.prod(spec["front_weight0_f16"][1]) * 2,
                "applied": False,
                "shape": list(spec["front_weight0_f16"][1]),
                "reason": "sm_120 pre HMMA front tile; texture feature producer is unresolved",
            },
            {
                "name": "block0.pre_texture_front_weight1_f16",
                "offset": int(spec["front_weight1_f16"][0]),
                "bytes": math.prod(spec["front_weight1_f16"][1]) * 2,
                "applied": False,
                "shape": list(spec["front_weight1_f16"][1]),
                "reason": "sm_120 pre HMMA front tile; texture feature producer is unresolved",
            },
        ]
        skipped: list[str] = []

        for section, destination in (
            ("front_weight0_f16", self.pre_front_weight0_f16),
            ("front_weight1_f16", self.pre_front_weight1_f16),
        ):
            offset, shape = spec[section]
            tile = _decode_blob_f16(weights, name, int(offset), tuple(shape))
            _copy_parameter(destination, tile, f"{name}.{section}")
            loaded.append(f"{name}.{section} -> audit buffer")

        # Keep the old direct-RGB path runnable for smoke tests and callers
        # without a reconstruction of the texture front-end, but do not
        # report it as a serialized adapter. There is no proven RGB->32
        # matrix in this record, so use a zero fallback rather than slicing
        # body or front payload bytes into a false projection.
        if self.input_channels:
            metadata.append(
                {
                    "name": "block0.rgb_fallback_weight",
                    "offset": None,
                    "bytes": 0,
                    "applied": False,
                    "reason": "zero fallback; no serialized RGB->32 projection is proven",
                }
            )

        for section, destination in (
            ("weight1", self.pre_body.mlp[0].weight),
            ("qkv", self.pre_body.attn.qkv.weight),
            ("projection", self.pre_body.attn.proj.weight),
        ):
            offset, shape = spec[section]
            matrix = decode_fp8_matrix(raw, tuple(shape), byte_offset=int(offset))
            _copy_parameter(destination, matrix, f"{name}.{section}")
            loaded.append(f"{name}.{section} -> pre_body")

        weight2_offset, weight2_shape = spec["weight2"]
        weight2 = decode_fp8_matrix(raw, tuple(weight2_shape), byte_offset=int(weight2_offset))
        if not torch.isfinite(weight2).all():
            bad = int((~torch.isfinite(weight2)).sum())
            # Keep the valid 4092 entries.  The four 0x7f/0xff E4M3FN
            # markers are isolated invalid slots in an otherwise complete
            # matrix; zero is the least surprising runnable fallback, and
            # the report records that it is not a bit-exact interpretation.
            weight2 = torch.nan_to_num(weight2, nan=0.0, posinf=0.0, neginf=0.0)
            metadata.append(
                {
                    "name": f"{name}.weight2.nonfinite",
                    "count": bad,
                    "applied": True,
                    "value": "zero fallback",
                    "reason": "four isolated E4M3FN NaN markers were present in the serialized matrix",
                }
            )
        _copy_parameter(self.pre_body.mlp[2].weight, weight2, f"{name}.weight2")
        loaded.append(f"{name}.weight2 -> pre_body")

        for section, destination in (
            ("ffn_cos_skip", self.pre_body.ffn_cos_skip),
            ("attn_cos_skip", self.pre_body.attn_cos_skip),
        ):
            offset, shape = spec[section]
            scale = _decode_blob_f16(weights, name, int(offset), tuple(shape))
            _copy_parameter(destination, scale, f"{name}.{section}")
            loaded.append(f"{name}.{section} -> pre_body")

        attn_scale = _decode_blob_f32(
            weights, name, int(spec["attn_scale"][0]), tuple(spec["attn_scale"][1])
        )
        _copy_parameter(self.pre_body.attn.attn_scale, attn_scale, f"{name}.attn_scale")
        loaded.append(f"{name}.attn_scale -> pre_body.attn_scale")

        bias_offset, bias_shape = spec["attn_bias"]
        bias = _decode_blob_f16(weights, name, int(bias_offset), tuple(bias_shape))
        if not torch.isfinite(bias).all():
            bad = int((~torch.isfinite(bias)).sum())
            skipped.append(f"{name}.attn_bias contains {bad} non-finite FP16 value(s)")
        else:
            _copy_parameter(self.pre_body.attn.attn_bias, bias, f"{name}.attn_bias")
            loaded.append(f"{name}.attn_bias -> pre_body")
        return {"loaded": loaded, "metadata": metadata, "skipped": skipped}

    def load_post_weights(self, weights: DLSS5WeightMap) -> dict[str, Any]:
        """Load block 70's body and its padded FP16 output projection."""

        name = "block70.layer0.layer"
        spec = KNOWN_POST_SWIN_LAYOUT
        _expect_blob_size(weights, name, int(spec["expected_bytes"]))
        raw = weights.fp8(name)
        loaded: list[str] = []
        metadata: list[dict[str, Any]] = []

        for section, destination in (
            ("weight1", self.post_body.mlp[0].weight),
            ("weight2", self.post_body.mlp[2].weight),
            ("qkv", self.post_body.attn.qkv.weight),
            ("projection", self.post_body.attn.proj.weight),
        ):
            offset, shape = spec[section]
            matrix = decode_fp8_matrix(raw, tuple(shape), byte_offset=int(offset))
            _copy_parameter(destination, matrix, f"{name}.{section}")
            loaded.append(f"{name}.{section} -> post_body")

        for section, destination in (
            ("ffn_cos_skip", self.post_body.ffn_cos_skip),
            ("attn_cos_skip", self.post_body.attn_cos_skip),
        ):
            offset, shape = spec[section]
            scale = _decode_blob_f16(weights, name, int(offset), tuple(shape))
            _copy_parameter(destination, scale, f"{name}.{section}")
            loaded.append(f"{name}.{section} -> post_body")

        bias_offset, bias_shape = spec["attn_bias"]
        bias = _decode_blob_f16(weights, name, int(bias_offset), tuple(bias_shape))
        if not torch.isfinite(bias).all():
            bad = int((~torch.isfinite(bias)).sum())
            return {
                "loaded": loaded,
                "metadata": metadata,
                "skipped": [f"{name}.attn_bias contains {bad} non-finite FP16 value(s)"],
            }
        _copy_parameter(self.post_body.attn.attn_bias, bias, f"{name}.attn_bias")
        loaded.append(f"{name}.attn_bias -> post_body")

        input_dw = _decode_blob_f16(
            weights,
            name,
            int(spec["input_dw_weight_f16"][0]),
            tuple(spec["input_dw_weight_f16"][1]),
        )
        _copy_parameter(
            self.post_input_projection.weight,
            input_dw.reshape(32, 1, 1, 1),
            f"{name}.dw_weight",
        )
        loaded.append(f"{name}.dw_weight -> post_input_projection")
        input_scale = _decode_blob_f16(
            weights,
            name,
            int(spec["input_scale_f16"][0]),
            tuple(spec["input_scale_f16"][1]),
        )
        _copy_parameter(self.post_input_scale, input_scale, f"{name}.inp_upsample_input_scale")
        loaded.append(f"{name}.inp_upsample_input_scale -> post_input_scale")

        gain_offset, gain_shape = spec["out_gain_f16"]
        out_gain = _decode_blob_f16(weights, name, int(gain_offset), tuple(gain_shape))
        metadata.append(
            {
                "name": "block70.out_gain",
                "offset": int(gain_offset),
                "bytes": int(math.prod(gain_shape) * 2),
                "applied": False,
                "min": float(out_gain.min()),
                "max": float(out_gain.max()),
                "zeros": int((out_gain == 0).sum()),
                "reason": "the registered 8-half out_gain slot is decoded and is all zero; static tracing finds only constructor registration, not an independent post-execute consumer, but index-based binding prevents proving it is padding",
            }
        )

        output_offset, output_shape = spec["out_conv_weight_f16"]
        padded_output = _decode_blob_f16(
            weights, name, int(output_offset), tuple(output_shape)
        )
        if self.output_channels > padded_output.shape[0]:
            raise ValueError(
                f"post output has only {padded_output.shape[0]} padded channels, "
                f"requested {self.output_channels}"
            )
        if self.post_output_layout == "tensor_core_candidate":
            if self.output_channels > 4:
                raise ValueError(
                    "tensor_core_candidate post output layout supports at most 4 channels"
                )
            logical_output = decode_post_output_tile_candidate(
                padded_output,
                output_channels=self.output_channels,
            )
            output_weight = logical_output.reshape(logical_output.shape[0], 32, 1, 1)
        elif self.post_output_layout == "column_major_prefix":
            logical_output = decode_post_output_tile_column_major(
                padded_output,
                output_channels=self.output_channels,
            )
            output_weight = logical_output.reshape(logical_output.shape[0], 32, 1, 1)
        else:
            output_weight = padded_output[: self.output_channels].reshape(
                self.output_channels, 32, 1, 1
            )
        _copy_parameter(self.post_out.weight, output_weight, f"{name}.out_conv_weight")
        loaded.append(
            f"{name}.out_conv_weight -> post_out[1x1] ({self.post_output_layout})"
        )

        scale_offset, scale_shape = spec["attn_scale"]
        scale = _decode_blob_f32(weights, name, int(scale_offset), tuple(scale_shape))
        _copy_parameter(self.post_body.attn.attn_scale, scale, f"{name}.attn_scale")
        loaded.append(
            f"{name}[{int(scale_offset)}:{int(scale_offset) + scale.numel() * 4}] -> post_body.attn_scale"
        )
        metadata.extend(
            [
                {
                    "name": "block70.front_scale_or_activation",
                    "offset": int(spec["opaque_before_qkv"][0]),
                    "bytes": int(spec["opaque_before_qkv"][1]),
                    "applied": False,
                    "reason": "opaque pre-QKV section remains unresolved; post-body attention scale is loaded from the FP32 gap before projection",
                },
                {
                    "name": "block70.out_conv_weight.tile_swizzle",
                    "offset": int(output_offset),
                    "bytes": int(padded_output.numel() * 2),
                    "applied": False,
                    "mode": self.post_output_layout,
                    "reason": "out_conv_weight is loaded from the post-gain 16x32 FP16 tile; tensor-core row/lane swizzle is not resolved even when the evidence-based candidate is selected",
                },
            ]
        )
        return {"loaded": loaded, "metadata": metadata, "skipped": []}

    def load_dec_input_weights(self, weights: DLSS5WeightMap) -> dict[str, Any]:
        """Load block 39's 1024-to-512 projection and residual scale."""

        name = "block39.layer0.layer"
        spec = KNOWN_DEC_INPUT_LAYOUT
        _expect_blob_size(weights, name, int(spec["expected_bytes"]))
        raw = weights.fp8(name)
        conv_offset, conv_shape = spec["conv_weight"]
        conv_weight = decode_fp8_matrix(raw, tuple(conv_shape), byte_offset=int(conv_offset))
        _copy_parameter(self.dec_input.conv_weight.weight, conv_weight, f"{name}.conv_weight")
        dw_offset, dw_shape = spec["dw_weight"]
        dw_weight = _decode_blob_f16(weights, name, int(dw_offset), tuple(dw_shape))
        _copy_parameter(self.dec_input.dw_weight, dw_weight, f"{name}.dw_weight")
        return {
            "loaded": [
                f"{name}.conv_weight -> dec_input",
                f"{name}.dw_weight -> dec_input",
            ],
            "metadata": [
                {
                    "name": "block39.sin",
                    "offset": int(spec["opaque_sin"][0]),
                    "bytes": int(spec["opaque_sin"][1]),
                    "applied": False,
                    "reason": "sin/tile interpolation is represented by explicit bilinear reference code",
                }
            ],
            "skipped": [],
        }

    def load_vit_weights(
        self,
        weights: DLSS5WeightMap,
        *,
        block_start: int = 31,
        block_count: int = 8,
    ) -> dict[str, Any]:
        """Load the strictly identified ViT FP8 matrices and residual scales.

        The matrix byte order is the explicit row-major candidate documented
        by :func:`decode_fp8_matrix`.  The QKV blob has a verified 64-half
        prefix whose producer meaning is still unknown; it is reported but
        not injected into a PyTorch parameter.
        """

        if block_count != len(self.vit):
            raise ValueError("the embedded ViT loader expects eight blocks")
        loaded: list[str] = []
        metadata: list[dict[str, Any]] = []
        for module_index, block_index in enumerate(range(block_start, block_start + block_count)):
            module = self.vit[module_index]
            prefix = f"block{block_index}"

            def matrix(layer: int) -> Tensor:
                name = f"{prefix}.layer{layer}.layer"
                spec = KNOWN_VIT_BLOB_LAYOUT[f"layer{layer}"]
                if layer == 0:
                    expected = 4096 * 1024 + 16
                elif layer == 1:
                    expected = 1024 * 4096 + 1024 * 2
                elif layer == 2:
                    expected = 128 + 3072 * 1024
                else:
                    expected = 1024 * 1024 + 1024 * 2
                _expect_blob_size(weights, name, expected)
                return _decode_blob_matrix(weights, name, spec)

            ffn_expand = matrix(0)
            _copy_parameter(module.ffn_expand.weight, ffn_expand, f"{prefix}.layer0")
            loaded.append(f"{prefix}.layer0 -> vit[{module_index}].ffn_expand.weight")

            ffn_contract = matrix(1)
            _copy_parameter(module.ffn_contract.weight, ffn_contract, f"{prefix}.layer1")
            ffn_cos_skip = _decode_blob_f16(
                weights,
                f"{prefix}.layer1.layer",
                KNOWN_VIT_BLOB_LAYOUT["layer1"]["cos_skip_offset"],
                (1024,),
            )
            _copy_parameter(module.ffn_cos_skip, ffn_cos_skip, f"{prefix}.layer1.ffn_cos_skip")
            loaded.append(f"{prefix}.layer1 -> vit[{module_index}].ffn_contract/ffn_cos_skip")

            qkv_spec = KNOWN_VIT_BLOB_LAYOUT["layer2"]
            qkv = _decode_blob_matrix(weights, f"{prefix}.layer2.layer", qkv_spec)
            _copy_parameter(module.qkv.weight, qkv, f"{prefix}.layer2")
            header = _decode_blob_f16(weights, f"{prefix}.layer2.layer", 0, (64,))
            metadata.append(
                {
                    "name": f"{prefix}.layer2.header",
                    "elements": 64,
                    "min": float(header.min()),
                    "max": float(header.max()),
                }
            )
            loaded.append(f"{prefix}.layer2 -> vit[{module_index}].qkv.weight")

            scalar = _decode_blob_f16(weights, f"{prefix}.layer3.layer", 0, (1,))
            _expect_blob_size(weights, f"{prefix}.layer3.layer", 2)
            _copy_parameter(module.attention.attn_scale, scalar, f"{prefix}.layer3.attention_scalar")
            loaded.append(f"{prefix}.layer3 -> vit[{module_index}].attention.attn_scale")

            projection = matrix(4)
            _copy_parameter(module.projection.weight, projection, f"{prefix}.layer4")
            attn_cos_skip = _decode_blob_f16(
                weights,
                f"{prefix}.layer4.layer",
                KNOWN_VIT_BLOB_LAYOUT["layer4"]["cos_skip_offset"],
                (1024,),
            )
            _copy_parameter(module.attn_cos_skip, attn_cos_skip, f"{prefix}.layer4.attn_cos_skip")
            loaded.append(f"{prefix}.layer4 -> vit[{module_index}].projection/attn_cos_skip")
        return {"loaded": loaded, "metadata": metadata, "skipped": []}

    def load_split_weights(
        self,
        weights: DLSS5WeightMap,
        *,
        blocks: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Load the four proven matrix/attention sections of Split-Swin.

        ``layer4`` contains the proven 1024x512 pointwise FP8 operand.  The
        host graph has a second convolution slot, but the record has no
        independent depthwise bytes and the final-head cubin has no matching
        depthwise multiply; the reference therefore treats that slot as
        identity.  The 32-value attention-scale section is retained in the
        report rather than guessing how it maps onto the 16 heads.
        """

        if self.window_size != 8:
            raise ValueError("embedded Split-Swin blobs are for 8x8 windows")
        if blocks is None:
            blocks = list(range(23, 31)) + list(range(40, 48))
        modules: dict[int, SplitSwinBlock] = {}
        for index, module in enumerate(self.split_enc):
            modules[23 + index] = module
        for index, module in enumerate(self.split_dec):
            modules[40 + index] = module

        loaded: list[str] = []
        metadata: list[dict[str, Any]] = []
        skipped: list[str] = []
        for block_index in blocks:
            if block_index not in modules:
                raise ValueError(f"unsupported Split-Swin block: {block_index}")
            module = modules[block_index]
            prefix = f"block{block_index}"

            layer0_name = f"{prefix}.layer0.layer"
            layer0_spec = KNOWN_SPLIT_BLOB_LAYOUT["layer0"]
            _expect_blob_size(weights, layer0_name, 2 * 512 * 512)
            ffn_a = decode_fp8_matrix(
                weights.fp8(layer0_name), tuple(layer0_spec["fp8_shapes"][0]), byte_offset=0
            )
            ffn_b = decode_fp8_matrix(
                weights.fp8(layer0_name), tuple(layer0_spec["fp8_shapes"][1]), byte_offset=262144
            )
            _copy_parameter(module.ffwd.weight, ffn_a, f"{prefix}.layer0.weight0")
            _copy_parameter(module.ffwd_gate.weight, ffn_b, f"{prefix}.layer0.weight1")
            loaded.append(f"{prefix}.layer0 -> ffwd/ffwd_gate")

            ffn_contract = _decode_blob_matrix(
                weights, f"{prefix}.layer1.layer", KNOWN_SPLIT_BLOB_LAYOUT["layer1"]
            )
            _expect_blob_size(weights, f"{prefix}.layer1.layer", 512 * 512 + 512 * 2)
            _copy_parameter(module.ffwd_proj.weight, ffn_contract, f"{prefix}.layer1")
            ffn_cos_skip = _decode_blob_f16(
                weights,
                f"{prefix}.layer1.layer",
                KNOWN_SPLIT_BLOB_LAYOUT["layer1"]["cos_skip_offset"],
                (512,),
            )
            _copy_parameter(module.ffn_cos_skip, ffn_cos_skip, f"{prefix}.layer1.ffn_cos_skip")
            loaded.append(f"{prefix}.layer1 -> ffwd_proj/ffn_cos_skip")

            qkv = _decode_blob_matrix(
                weights, f"{prefix}.layer2.layer", KNOWN_SPLIT_BLOB_LAYOUT["layer2"]
            )
            _expect_blob_size(weights, f"{prefix}.layer2.layer", 1536 * 512 + 65536 * 2 + 32 * 2)
            _copy_parameter(module.qkv_attn.qkv.weight, qkv, f"{prefix}.layer2.qkv")
            attn_bias = _decode_blob_f16(
                weights, f"{prefix}.layer2.layer", 786432, (16, 64, 64)
            )
            _copy_parameter(module.qkv_attn.attn_bias, attn_bias, f"{prefix}.layer2.attn_bias")
            # The 64-byte tail is 16 FP32 per-head scales, not 32 FP16
            # values.  Reading it as half produces implausible alternating
            # magnitudes; FP32 gives the expected positive scale vector.
            attn_scale = _decode_blob_f32(
                weights, f"{prefix}.layer2.layer", 917504, (16,)
            )
            _copy_parameter(
                module.qkv_attn.attn_scale,
                attn_scale,
                f"{prefix}.layer2.attn_scale",
            )
            loaded.append(f"{prefix}.layer2 -> qkv/attn_bias/attn_scale")

            projection = _decode_blob_matrix(
                weights, f"{prefix}.layer3.layer", KNOWN_SPLIT_BLOB_LAYOUT["layer3"]
            )
            _expect_blob_size(weights, f"{prefix}.layer3.layer", 512 * 512 + 512 * 2)
            _copy_parameter(module.projection.weight, projection, f"{prefix}.layer3")
            attn_cos_skip = _decode_blob_f16(
                weights,
                f"{prefix}.layer3.layer",
                KNOWN_SPLIT_BLOB_LAYOUT["layer3"]["cos_skip_offset"],
                (512,),
            )
            _copy_parameter(module.attn_cos_skip, attn_cos_skip, f"{prefix}.layer3.attn_cos_skip")
            loaded.append(f"{prefix}.layer3 -> projection/attn_cos_skip")

            if f"{prefix}.layer4.layer" in weights:
                if block_index != 30:
                    skipped.append(f"{prefix}.layer4 unexpected FinalHead")
                    continue
                final_name = f"{prefix}.layer4.layer"
                _expect_blob_size(weights, final_name, 1024 * 512 + 16)
                final_weight = decode_fp8_matrix(
                    weights.fp8(final_name), (1024, 512), byte_offset=0
                ).reshape(1024, 512, 1, 1)
                _copy_parameter(
                    module.final_output.weight,
                    final_weight,
                    f"{prefix}.layer4.weight",
                )
                loaded.append(f"{prefix}.layer4.weight -> FinalHead.pointwise")
                metadata.append(
                    {
                        "name": f"{prefix}.layer4.dw_weight",
                        "applied": True,
                        "value": "identity (no independent bytes in layer4 record)",
                        "reason": "final-head cubin has one 1024x512 HMMA matrix; the host slot has no separate serialized operand",
                    }
                )
        return {"loaded": loaded, "metadata": metadata, "skipped": skipped}

    def load_known_weights(
        self,
        weights: DLSS5WeightMap,
        *,
        swin: bool = True,
        vit: bool = True,
        split: bool = True,
        post: bool = True,
        pre: bool = True,
        dec_input: bool = True,
    ) -> dict[str, Any]:
        """Load all currently proven sections and return an audit report."""

        report: dict[str, Any] = {"loaded": [], "metadata": [], "skipped": []}
        if "block70.layer0.blend_scale" in weights:
            value = weights.float16("block70.layer0.blend_scale")[0].float()
            with torch.no_grad():
                self.blend_scale.copy_(value.to(device=self.blend_scale.device))
            report["loaded"].append("block70.layer0.blend_scale")
        if vit:
            result = self.load_vit_weights(weights)
            for key in report:
                report[key].extend(result[key])
        if swin:
            result = self.load_swin_weights(weights)
            for key in report:
                report[key].extend(result[key])
        if split:
            result = self.load_split_weights(weights)
            for key in report:
                report[key].extend(result[key])
        if post:
            result = self.load_post_weights(weights)
            for key in report:
                report[key].extend(result[key])
        if pre:
            result = self.load_pre_weights(weights)
            for key in report:
                report[key].extend(result[key])
        if dec_input:
            result = self.load_dec_input_weights(weights)
            for key in report:
                report[key].extend(result[key])
        return report

    @classmethod
    def with_weight_map(
        cls,
        root: str | Path,
        *,
        load_known: bool = False,
        load_swin: bool = True,
        load_vit: bool = True,
        load_split: bool = True,
        load_post: bool = True,
        load_pre: bool = True,
        load_dec_input: bool = True,
        **kwargs: Any,
    ) -> tuple["DLSS5Graph", DLSS5WeightMap]:
        """Construct the graph and attach the parsed outer map.

        By default only ``blend_scale`` is loaded.  ``load_known=True`` also
        loads the proven pre/post, Swin, ViT, and Split-Swin sections whose
        byte counts and operands are established by the cubin.  Remaining
        fused sections are returned in the audit report and are never
        silently reshaped.
        """

        root = Path(root)
        weights = DLSS5WeightMap.from_file(root / "WEIGHTS_HT.bin")
        model = cls(**kwargs)
        if "block70.layer0.blend_scale" in weights:
            model.blend_scale.copy_(weights.float16("block70.layer0.blend_scale")[0].float())
        if load_known:
            model.weight_report = model.load_known_weights(
                weights,
                swin=load_swin,
                vit=load_vit,
                split=load_split,
                post=load_post,
                pre=load_pre,
                dec_input=load_dec_input,
            )
        return model, weights
